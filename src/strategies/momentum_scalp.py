from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, fields, replace
from datetime import date, datetime, timedelta
from math import ceil
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.analytics.math_signals import (
    ExpectedValueEstimate,
    LeaderSignal,
    RegimeProbabilities,
    build_entry_ev_table,
    build_leader_signals,
    compute_market_shock_signal,
    compute_regime_probabilities,
    load_recent_scorecards,
)
from src.analytics.forecast_outcomes import ForecastOutcomeLedger
from src.analytics.price_prediction import ShortHorizonPrediction, predict_short_horizon_return
from src.models import Order, OrderResult, OrderSide, OrderType, Position, Quote
from src.strategy import BaseStrategy
from src.strategies.momentum_scalp_types import (
    DEFAULT_INVERSE_ETFS,
    DEFAULT_STATIC_WATCHLIST,
    DailyPnL,
    INTRADAY_STRATEGY,
    INTRADAY_BASE_QUEUE_SOURCES,
    LIVE_LONG_QUEUE_SOURCES,
    MomentumScalpConfig,
    OPENING_STRATEGY,
    OPENING_LONG_QUEUE_SOURCES,
    PositionState,
)
from src.strategies.momentum_scalp_exit_plans import long_exit_hold_profile
from src.strategies.momentum_scalp_exit import (
    LongExitSnapshot,
    decide_long_exit,
)
from src.strategies.momentum_scalp_fills import handle_order_filled
from src.strategies.momentum_scalp_micro import symbol_micro_edge_metrics
from src.strategies.momentum_scalp_pnl import (
    calculate_trade_pnl_from_prices,
    estimate_trade_net_pnl_from_prices,
    estimate_trade_net_pnl_unrounded,
)
from src.strategies.momentum_scalp_state import (
    DAILY_PNL_SNAPSHOT_FIELDS,
    empty_daily_pnl_snapshot,
    rebuild_daily_pnl_snapshot_from_ledgers,
)
from src.strategies.regime_router import RegimeStrategyRouter


logger = logging.getLogger("kis_trader.strategy.momentum_scalp")
order_logger = logging.getLogger("kis_trader.orders")

LEGACY_LONG_STRATEGY = "bull_breakout_strategy"


@dataclass(frozen=True)
class ExpectedValueTradePlan:
    allowed: bool
    reject_reason: str = ""
    quantity: int = 0
    budget: int = 0
    expected_net: float = 0.0
    predicted_net: int = 0
    lower_net: int = 0
    upper_net: int = 0
    win_probability: float = 0.0
    break_even_probability: float = 1.0
    planned_target_net: int = 0
    planned_stop_net_loss_abs: int = 0
    planned_risk_net_loss_abs: int = 0
    planned_take_profit_pct: float = 0.0
    planned_stop_loss_pct: float = 0.0
    prediction: Optional[ShortHorizonPrediction] = None
    reject_detail: str = ""


@dataclass
class ExpectedValueCandidate:
    quote: Quote
    strategy_name: str
    metadata: Dict[str, Any]
    plan: ExpectedValueTradePlan


class MomentumScalpStrategy(BaseStrategy):
    """기댓값 기반 롱 진입 경로만 남긴 모멘텀 스캘프 전략.

    실전 경로:
    - opening_conviction_long_strategy
    - intraday_conviction_long_strategy

    인버스 ETF도 별도 라우트 없이 위 롱 경로에서 동일하게 예측/EV로 평가한다.
    """

    @staticmethod
    def _is_supported_long_symbol(symbol: str) -> bool:
        normalized = str(symbol or "").strip()
        return len(normalized) == 6 and normalized.isdigit()

    def __init__(
        self,
        market_data,
        config: MomentumScalpConfig,
        pool_override: Optional[List[str]] = None,
    ):
        self.market_data = market_data
        self.config = config
        self.cfg = config
        self.pool_override = list(pool_override or [])

        self.positions: Dict[str, PositionState] = {}
        self.daily_pnl = DailyPnL()
        self._breaker_excluded_realized_net_pnl = 0
        self._halted = False
        self._halt_reason = ""
        self._sell_fill_ledger: List[Dict[str, Any]] = []
        self._closed_trade_ledger: Dict[str, Dict[str, Any]] = {}
        self._ledger_seed_snapshot: Dict[str, int] = self._empty_daily_pnl_snapshot()

        self._simulated_now: Optional[datetime] = None
        self._session_start_at: Optional[datetime] = None
        self._active_day: Optional[str] = None

        self._pool: List[str] = list(self.pool_override or self.config.static_watchlist)
        self._quotes_cache: Dict[str, Quote] = {}
        self._recent_quotes: Dict[str, deque[Quote]] = {}
        self._avg_volumes: Dict[str, int] = {}
        self._pending_entry_meta: Dict[str, Dict[str, Any]] = {}
        self._forecast_outcomes = ForecastOutcomeLedger(
            Path(self.config.forecast_outcome_root)
        )

        self._inverse_symbols = set(self.config.inverse_etfs or [])
        self._latest_math_leader_signals: Dict[str, LeaderSignal] = {}
        self._latest_math_queue_symbols: List[str] = []
        self._latest_math_backfill_symbols: List[str] = []
        self._latest_math_queue_source: Dict[str, str] = {}
        self._latest_opening_fast_symbols: set[str] = set()
        self._latest_opening_hot_symbols: set[str] = set()

        self._latest_regime_probabilities = RegimeProbabilities(
            bull_prob=0.25,
            neutral_prob=0.25,
            soft_bear_prob=0.25,
            bear_prob=0.25,
        )
        self._adaptive_market_state: Dict[str, float] = {
            "quote_count": 0.0,
            "avg_change": 0.0,
            "decliner_ratio": 0.5,
            "advancer_ratio": 0.5,
            "bull_prob": 0.25,
            "bear_prob": 0.25,
            "leader_density": 0.0,
            "elite_leader_density": 0.0,
            "vs_open_p90": 0.0,
            "accel_p70": 0.0,
            "tape_heat": 0.0,
            "tape_caution": 0.5,
            "overheat": 0.0,
        }
        self._latest_market_shock_signal = compute_market_shock_signal(
            minutes_since_open=0,
            crash_window_minutes=self.config.market_shock_window_minutes_after_open,
            index_gap_open_pct=0.0,
            index_gap_ma5_pct=0.0,
            index_gap_ma20_pct=0.0,
            avg_change=0.0,
            decliner_ratio=0.5,
            falling_speed_pct=0.0,
            inverse_leader_count=0,
        )
        self._entry_ev_table: Dict[Tuple[str, str, str, str], ExpectedValueEstimate] = {}
        self._entry_ev_history_records: List[Dict[str, Any]] = []
        self._symbol_entry_cooldown_until: Dict[str, datetime] = {}
        self._symbol_order_unavailable: Dict[str, Dict[str, Any]] = {}
        self._restore_ignore_until: Dict[str, datetime] = {}
        self._last_long_shortlist_symbols: List[str] = []
        self._state_restored_today = False
        self._last_daily_state_save_at: Optional[datetime] = None

        self._bear_score = 0
        self._bull_market_context = "broad_bull"
        self._strong_bull_override_active = False
        self._bull_loss_count_today = 0

        self._regime_router = RegimeStrategyRouter()

    # ------------------------------------------------------------------
    # 기본 수명주기 / 운영 상태
    # ------------------------------------------------------------------
    def set_simulated_now(self, now: Optional[datetime]):
        self._simulated_now = now

    def _now(self) -> datetime:
        return self._simulated_now or datetime.now()

    def _today(self) -> str:
        return self._now().strftime("%Y%m%d")

    def _ensure_recent_quote_window(self, symbol: str) -> deque[Quote]:
        window = self._recent_quotes.get(symbol)
        if window is None:
            window = deque(maxlen=max(4, int(self.config.setup_recent_quote_window)))
            self._recent_quotes[symbol] = window
        return window

    def _reset_for_new_day(self):
        self.daily_pnl = DailyPnL()
        self._breaker_excluded_realized_net_pnl = 0
        self._halted = False
        self._halt_reason = ""
        self._sell_fill_ledger = []
        self._closed_trade_ledger = {}
        self._ledger_seed_snapshot = self._empty_daily_pnl_snapshot()
        self._bull_loss_count_today = 0
        self._symbol_entry_cooldown_until = {}
        self._symbol_order_unavailable = {}
        self._pending_entry_meta = {}
        self._last_daily_state_save_at = None
        self._active_day = self._today()
        now = self._now()
        self._session_start_at = now.replace(hour=9, minute=0, second=0, microsecond=0)

    def initialize(self):
        if self._active_day != self._today():
            self._reset_for_new_day()
        elif self._session_start_at is None:
            now = self._now()
            self._session_start_at = now.replace(hour=9, minute=0, second=0, microsecond=0)
        self._load_entry_ev_data()
        self._load_daily_state()
        self._update_daily_breakers()

    def get_watchlist(self) -> List[str]:
        symbols = list(self.pool_override or self._pool or self.config.static_watchlist or DEFAULT_STATIC_WATCHLIST)
        symbols.extend(self.config.inverse_etfs or DEFAULT_INVERSE_ETFS)
        seen = set()
        ordered: List[str] = []
        for symbol in symbols:
            normalized = str(symbol or "").strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(normalized)
        return ordered

    def update_runtime_pool(self, symbols: List[str]):
        if self.pool_override:
            return
        dynamic_limit = max(1, int(self.config.dynamic_pool_size))
        static_core_limit = min(10, max(4, dynamic_limit // 2))
        runtime_symbols = list(symbols or [])[:dynamic_limit]
        runtime_symbols.extend(list(self.config.static_watchlist or DEFAULT_STATIC_WATCHLIST)[:static_core_limit])
        seen = set()
        ordered: List[str] = []
        for symbol in runtime_symbols:
            normalized = str(symbol or "").strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(normalized)
        if ordered and ordered != self._pool:
            self._pool = ordered
            self._save_daily_state_if_due(min_interval_seconds=10)

    def load_avg_volumes(self, avg_volumes: Dict[str, int]):
        self._avg_volumes = {
            str(symbol): int(volume)
            for symbol, volume in (avg_volumes or {}).items()
            if int(volume or 0) > 0
        }

    def should_continue(self) -> bool:
        return not self._halted

    # ------------------------------------------------------------------
    # 상태 저장 / 복구
    # ------------------------------------------------------------------
    def _daily_state_path(self) -> Path:
        return Path(self.config.daily_state_path)

    @staticmethod
    def _serialize_datetime(value: Optional[datetime]) -> str:
        return value.isoformat() if isinstance(value, datetime) else ""

    @staticmethod
    def _deserialize_datetime(value: Any) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value))
        except (TypeError, ValueError):
            return None

    def _serialize_datetime_map(self, items: Dict[str, datetime]) -> Dict[str, str]:
        serialized: Dict[str, str] = {}
        now = self._now()
        for key, value in (items or {}).items():
            if not isinstance(value, datetime) or value <= now:
                continue
            normalized = str(key or "").strip()
            if not normalized:
                continue
            serialized[normalized] = value.isoformat()
        return serialized

    def _deserialize_datetime_map(self, items: Dict[str, Any]) -> Dict[str, datetime]:
        restored: Dict[str, datetime] = {}
        now = self._now()
        for key, raw_value in (items or {}).items():
            normalized = str(key or "").strip()
            parsed = self._deserialize_datetime(raw_value)
            if not normalized or parsed is None or parsed <= now:
                continue
            restored[normalized] = parsed
        return restored

    @staticmethod
    def _daily_pnl_snapshot_fields() -> Tuple[str, ...]:
        return DAILY_PNL_SNAPSHOT_FIELDS

    def _empty_daily_pnl_snapshot(self) -> Dict[str, int]:
        return empty_daily_pnl_snapshot()

    def _capture_daily_pnl_snapshot(self) -> Dict[str, int]:
        snapshot = {
            field: int(getattr(self.daily_pnl, field, 0) or 0)
            for field in self._daily_pnl_snapshot_fields()
        }
        snapshot["breaker_excluded_realized_net_pnl"] = int(self._breaker_excluded_realized_net_pnl or 0)
        return snapshot

    def _apply_daily_pnl_snapshot(self, snapshot: Dict[str, Any]) -> None:
        resolved = dict(snapshot or {})
        for field in self._daily_pnl_snapshot_fields():
            setattr(self.daily_pnl, field, int(resolved.get(field, 0) or 0))
        self._breaker_excluded_realized_net_pnl = int(
            resolved.get("breaker_excluded_realized_net_pnl", 0) or 0
        )

    def _rebuild_daily_pnl_from_ledgers(self) -> None:
        snapshot = rebuild_daily_pnl_snapshot_from_ledgers(
            seed_snapshot=self._ledger_seed_snapshot or self._empty_daily_pnl_snapshot(),
            sell_fill_ledger=self._sell_fill_ledger,
            closed_trade_ledger=self._closed_trade_ledger,
        )
        self._apply_daily_pnl_snapshot(snapshot)

    def _make_trade_key(self, symbol: str, timestamp: Optional[datetime] = None) -> str:
        moment = timestamp or self._now()
        return f"{str(symbol or '').strip()}:{moment.isoformat(timespec='seconds')}"

    def _make_sell_fill_id(self, result: OrderResult) -> str:
        order_no = str(getattr(result, "order_no", "") or "").strip()
        if order_no:
            return order_no
        symbol = str(getattr(result, "symbol", "") or "").strip() or "unknown"
        stamp = self._serialize_datetime(getattr(result, "timestamp", None) or self._now())
        return f"{symbol}:{stamp}:{len(self._sell_fill_ledger) + 1}"

    def _upsert_closed_trade_record(
        self,
        *,
        trade_key: str,
        symbol: str,
        strategy_name: str,
        setup_name: str,
    ) -> None:
        normalized_trade_key = str(trade_key or "").strip()
        if not normalized_trade_key:
            return
        fills = [
            entry
            for entry in self._sell_fill_ledger
            if str(entry.get("trade_key", "") or "").strip() == normalized_trade_key
        ]
        if not fills:
            self._closed_trade_ledger.pop(normalized_trade_key, None)
            return
        self._closed_trade_ledger[normalized_trade_key] = {
            "trade_key": normalized_trade_key,
            "symbol": str(symbol or "").strip(),
            "strategy_name": str(strategy_name or "").strip(),
            "setup_name": str(setup_name or "").strip(),
            "net_pnl": sum(int(entry.get("net_pnl", 0) or 0) for entry in fills),
        }

    def _append_sell_fill_record(
        self,
        *,
        result: OrderResult,
        position: PositionState,
        quantity: int,
        sell_price: int,
        gross_pnl: int,
        net_pnl: int,
        fees: int,
        taxes: int,
        count_as_closed_trade: bool,
        counts_for_daily_breaker: bool,
        price_estimated: bool,
    ) -> None:
        trade_key = str(getattr(position, "trade_key", "") or "").strip() or self._make_trade_key(position.symbol, position.buy_time)
        self._sell_fill_ledger.append(
            {
                "fill_id": self._make_sell_fill_id(result),
                "order_no": str(getattr(result, "order_no", "") or "").strip(),
                "symbol": str(position.symbol or "").strip(),
                "trade_key": trade_key,
                "quantity": int(quantity or 0),
                "buy_price": int(position.buy_price or 0),
                "sell_price": int(sell_price or 0),
                "gross_pnl": int(gross_pnl or 0),
                "net_pnl": int(net_pnl or 0),
                "fees": int(fees or 0),
                "taxes": int(taxes or 0),
                "counts_for_daily_breaker": bool(counts_for_daily_breaker),
                "count_as_closed_trade": bool(count_as_closed_trade),
                "price_estimated": bool(price_estimated),
                "fill_mode": str(getattr(result, "fill_mode", "") or ""),
                "timestamp": self._serialize_datetime(getattr(result, "timestamp", None) or self._now()),
                "requested_reason": str(getattr(result, "requested_reason", "") or ""),
                "entry_strategy_name": str(getattr(position, "entry_strategy_name", "") or ""),
                "entry_setup_name": str(getattr(position, "entry_setup_name", "") or ""),
                "entry_queue_source": str(getattr(position, "queue_source", "") or ""),
                "entry_is_restored": bool(getattr(position, "is_restored", False)),
                "entry_reason": str(getattr(position, "entry_reason", "") or ""),
                "entry_grade": str(getattr(position, "entry_grade", "") or ""),
                "entry_grade_math": str(getattr(position, "entry_grade_math", "") or ""),
                "entry_ev": float(getattr(position, "entry_ev", 0.0) or 0.0),
                "entry_ev_confidence": str(getattr(position, "entry_ev_confidence", "") or ""),
                "conviction_tier": str(getattr(position, "conviction_tier", "") or ""),
                "bull_risk_mode": str(getattr(position, "bull_risk_mode", "") or ""),
                "post_loss_admission_class": str(getattr(position, "post_loss_admission_class", "") or ""),
                "candidate_class": str(getattr(position, "candidate_class", "") or ""),
                "execution_mode": str(getattr(position, "execution_mode", "") or ""),
                "live_route": str(getattr(position, "live_route", "") or ""),
                "size_multiplier": float(getattr(position, "size_multiplier", 1.0) or 1.0),
                "conviction_score": float(getattr(position, "conviction_score", 0.0) or 0.0),
                "conviction_rank": int(getattr(position, "conviction_rank", 0) or 0),
                "planned_target_net_pnl": int(getattr(position, "planned_target_net_pnl", 0) or 0),
                "planned_stop_net_loss_abs": int(getattr(position, "planned_stop_net_loss_abs", 0) or 0),
                "planned_risk_net_loss_abs": int(getattr(position, "planned_risk_net_loss_abs", 0) or 0),
                "entry_expected_net_pnl": float(getattr(position, "entry_expected_net_pnl", 0.0) or 0.0),
                "entry_prediction_net_pnl": int(getattr(position, "entry_prediction_net_pnl", 0) or 0),
                "entry_prediction_lower_net_pnl": int(getattr(position, "entry_prediction_lower_net_pnl", 0) or 0),
                "entry_prediction_win_probability": float(
                    getattr(position, "entry_prediction_win_probability", 0.0) or 0.0
                ),
            }
        )
        if count_as_closed_trade:
            self._upsert_closed_trade_record(
                trade_key=trade_key,
                symbol=position.symbol,
                strategy_name=position.entry_strategy_name,
                setup_name=position.entry_setup_name,
            )
        self._rebuild_daily_pnl_from_ledgers()

    def confirm_reconciled_sell_fills(self, account_api, *, results: Optional[List[OrderResult]] = None) -> List[Dict[str, Any]]:
        pending_entries = [
            entry
            for entry in self._sell_fill_ledger
            if bool(entry.get("price_estimated"))
            and str(entry.get("order_no", "") or "").strip()
        ]
        if not pending_entries or account_api is None:
            return []

        history = account_api.get_order_history(
            self._today(),
            self._today(),
            side="01",
        )
        if getattr(history, "empty", True):
            return []

        fill_rows: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for row in history.to_dict("records"):
            if not isinstance(row, dict):
                continue
            order_no = str(row.get("odno", "") or "").strip()
            symbol = str(row.get("pdno", "") or "").strip()
            if not order_no or not symbol:
                continue
            qty = account_api._coerce_int(row.get("tot_ccld_qty")) or 0
            price = account_api._coerce_int(row.get("avg_prvs")) or 0
            if qty > 0 and price <= 0:
                total_amt = account_api._coerce_int(row.get("tot_ccld_amt")) or 0
                if total_amt > 0:
                    price = int(round(total_amt / qty))
            if qty <= 0 or price <= 0:
                continue
            key = (order_no, symbol)
            existing = fill_rows.get(key)
            existing_qty = account_api._coerce_int(existing.get("tot_ccld_qty")) if isinstance(existing, dict) else 0
            if existing is None or qty >= existing_qty:
                normalized = dict(row)
                normalized["_resolved_qty"] = qty
                normalized["_resolved_price"] = price
                fill_rows[key] = normalized

        corrections: List[Dict[str, Any]] = []
        result_by_key: Dict[Tuple[str, str], OrderResult] = {}
        for result in results or []:
            result_key = (
                str(getattr(result, "order_no", "") or "").strip(),
                str(getattr(result, "symbol", "") or "").strip(),
            )
            if result_key[0] and result_key[1]:
                result_by_key[result_key] = result

        for entry in pending_entries:
            key = (
                str(entry.get("order_no", "") or "").strip(),
                str(entry.get("symbol", "") or "").strip(),
            )
            row = fill_rows.get(key)
            if row is None:
                continue
            correction = self.correct_reconciled_sell_fill(
                fill_id=str(entry.get("fill_id", "") or ""),
                order_no=key[0],
                symbol=key[1],
                confirmed_quantity=int(row.get("_resolved_qty", 0) or 0),
                confirmed_price=int(row.get("_resolved_price", 0) or 0),
            )
            if not correction:
                continue
            matched_result = result_by_key.get(key)
            if matched_result is not None:
                matched_result.quantity = int(correction.get("quantity", matched_result.quantity) or matched_result.quantity)
                matched_result.price = int(correction.get("corrected_price", matched_result.price) or matched_result.price)
                matched_result.fill_mode = "account_reconciled_confirmed"
            corrections.append(correction)
        return corrections

    def correct_reconciled_sell_fill(
        self,
        *,
        fill_id: str,
        order_no: str,
        symbol: str,
        confirmed_quantity: int,
        confirmed_price: int,
    ) -> Optional[Dict[str, Any]]:
        resolved_fill_id = str(fill_id or "").strip()
        resolved_order_no = str(order_no or "").strip()
        resolved_symbol = str(symbol or "").strip()
        if confirmed_quantity <= 0 or confirmed_price <= 0:
            return None

        entry = None
        for candidate in self._sell_fill_ledger:
            candidate_fill_id = str(candidate.get("fill_id", "") or "").strip()
            candidate_order_no = str(candidate.get("order_no", "") or "").strip()
            candidate_symbol = str(candidate.get("symbol", "") or "").strip()
            if resolved_fill_id and candidate_fill_id == resolved_fill_id:
                entry = candidate
                break
            if resolved_order_no and candidate_order_no == resolved_order_no and candidate_symbol == resolved_symbol:
                entry = candidate
                break
        if entry is None or not bool(entry.get("price_estimated")):
            return None

        previous_qty = int(entry.get("quantity", 0) or 0)
        if previous_qty <= 0 or confirmed_quantity != previous_qty:
            return None

        previous_price = int(entry.get("sell_price", 0) or 0)
        if confirmed_price == previous_price:
            entry["price_estimated"] = False
            entry["fill_mode"] = "account_reconciled_confirmed"
            self._rebuild_daily_pnl_from_ledgers()
            self._update_daily_breakers()
            self._save_daily_state()
            return None

        quantity = previous_qty
        buy_price = int(entry.get("buy_price", 0) or 0)
        previous_net_pnl = int(entry.get("net_pnl", 0) or 0)
        previous_gross_pnl = int(entry.get("gross_pnl", 0) or 0)
        previous_fees = int(entry.get("fees", 0) or 0)
        previous_taxes = int(entry.get("taxes", 0) or 0)
        corrected_pnl = calculate_trade_pnl_from_prices(
            entry_price=buy_price,
            exit_price=confirmed_price,
            quantity=quantity,
            commission_rate=float(self.config.commission_rate),
            tax_slippage_rate=float(self.config.tax_slippage_rate),
        )
        corrected_gross_pnl = int(corrected_pnl.gross_pnl)
        corrected_net_pnl = int(corrected_pnl.net_pnl)
        corrected_fees = int(corrected_pnl.fees)
        corrected_taxes = int(corrected_pnl.taxes)

        entry["sell_price"] = confirmed_price
        entry["gross_pnl"] = corrected_gross_pnl
        entry["net_pnl"] = corrected_net_pnl
        entry["fees"] = corrected_fees
        entry["taxes"] = corrected_taxes
        entry["price_estimated"] = False
        entry["fill_mode"] = "account_reconciled_confirmed"

        trade_key = str(entry.get("trade_key", "") or "").strip()
        position = self.positions.get(resolved_symbol)
        if position is not None and str(getattr(position, "trade_key", "") or "").strip() == trade_key:
            position.realized_gross_pnl_so_far += corrected_gross_pnl - previous_gross_pnl
            position.realized_net_pnl_so_far += corrected_net_pnl - previous_net_pnl
            position.realized_fees_paid_so_far += corrected_fees - previous_fees
            position.realized_taxes_paid_so_far += corrected_taxes - previous_taxes
        if trade_key and trade_key in self._closed_trade_ledger:
            self._upsert_closed_trade_record(
                trade_key=trade_key,
                symbol=resolved_symbol,
                strategy_name=str(entry.get("entry_strategy_name", "") or ""),
                setup_name=str(entry.get("entry_setup_name", "") or ""),
            )

        self._rebuild_daily_pnl_from_ledgers()
        self._update_daily_breakers()
        self._save_daily_state()

        delta_net_pnl = corrected_net_pnl - previous_net_pnl
        logger.info(
            "매도 체결 정정: %s order_no=%s previous_price=%d corrected_price=%d previous_net_pnl=%d corrected_net_pnl=%d delta_net_pnl=%d",
            resolved_symbol,
            resolved_order_no or "-",
            previous_price,
            confirmed_price,
            previous_net_pnl,
            corrected_net_pnl,
            delta_net_pnl,
        )
        return {
            "fill_id": resolved_fill_id or resolved_order_no,
            "order_no": resolved_order_no,
            "symbol": resolved_symbol,
            "quantity": quantity,
            "previous_price": previous_price,
            "corrected_price": confirmed_price,
            "previous_net_pnl": previous_net_pnl,
            "corrected_net_pnl": corrected_net_pnl,
            "previous_gross_pnl": previous_gross_pnl,
            "corrected_gross_pnl": corrected_gross_pnl,
            "delta_net_pnl": delta_net_pnl,
        }

    def _buy_time_from_trade_key(self, trade_key: str, fallback: datetime) -> datetime:
        normalized = str(trade_key or "").strip()
        _symbol, separator, raw_timestamp = normalized.partition(":")
        if not separator:
            return fallback
        parsed = self._deserialize_datetime(raw_timestamp)
        return parsed or fallback

    def _pop_estimated_exit_for_account_restore(
        self,
        symbol: str,
        *,
        quantity: int,
        buy_price: int,
        now: datetime,
    ) -> Optional[PositionState]:
        normalized = str(symbol or "").strip()
        if not normalized or quantity <= 0:
            return None
        matched_index = -1
        matched: Optional[Dict[str, Any]] = None
        for index in range(len(self._sell_fill_ledger) - 1, -1, -1):
            entry = self._sell_fill_ledger[index]
            if str(entry.get("symbol", "") or "").strip() != normalized:
                continue
            if not bool(entry.get("price_estimated")):
                continue
            if str(entry.get("fill_mode", "") or "") != "account_reconciled_estimated":
                continue
            if str(entry.get("order_no", "") or "").strip():
                continue
            if bool(entry.get("entry_is_restored", False)):
                continue
            entry_qty = int(entry.get("quantity", 0) or 0)
            if entry_qty > 0 and entry_qty != int(quantity):
                continue
            matched_index = index
            matched = dict(entry)
            break
        if matched is None or matched_index < 0:
            return None

        self._sell_fill_ledger.pop(matched_index)
        trade_key = str(matched.get("trade_key", "") or "").strip()
        if trade_key:
            self._upsert_closed_trade_record(
                trade_key=trade_key,
                symbol=normalized,
                strategy_name=str(matched.get("entry_strategy_name", "") or ""),
                setup_name=str(matched.get("entry_setup_name", "") or ""),
            )
        self._rebuild_daily_pnl_from_ledgers()

        resolved_buy_price = int(buy_price or 0) or int(matched.get("buy_price", 0) or 0)
        resolved_quantity = int(quantity or 0) or int(matched.get("quantity", 0) or 0)
        buy_time = self._buy_time_from_trade_key(trade_key, now)
        entry_strategy_name = str(matched.get("entry_strategy_name", "") or "") or INTRADAY_STRATEGY
        entry_setup_name = str(matched.get("entry_setup_name", "") or "") or "expected_value"
        position = PositionState(
            symbol=normalized,
            buy_price=max(1, int(resolved_buy_price or 0)),
            quantity=max(1, int(resolved_quantity or 0)),
            invested_amount=max(0, int(resolved_buy_price or 0) * int(resolved_quantity or 0)),
            buy_time=buy_time,
            is_restored=False,
            entry_strategy_name=entry_strategy_name,
            entry_setup_name=entry_setup_name,
            entry_reason=str(matched.get("entry_reason", "") or entry_setup_name),
            regime_label=self._resolve_regime_profile_name(),
            planned_risk_stage=self._current_bull_risk_mode(),
            entry_grade=str(matched.get("entry_grade", "") or ""),
            entry_grade_math=str(matched.get("entry_grade_math", "") or ""),
            entry_ev=float(matched.get("entry_ev", 0.0) or 0.0),
            entry_ev_confidence=str(matched.get("entry_ev_confidence", "") or ""),
            conviction_tier=str(matched.get("conviction_tier", "") or ""),
            bull_risk_mode=str(matched.get("bull_risk_mode", "") or self._current_bull_risk_mode()),
            post_loss_admission_class=str(matched.get("post_loss_admission_class", "") or "general"),
            candidate_class=str(matched.get("candidate_class", "") or ""),
            execution_mode=str(matched.get("execution_mode", "") or "live"),
            live_route=str(matched.get("live_route", "") or entry_strategy_name),
            queue_source=str(matched.get("entry_queue_source", "") or ""),
            size_multiplier=float(matched.get("size_multiplier", 1.0) or 1.0),
            conviction_score=float(matched.get("conviction_score", 0.0) or 0.0),
            conviction_rank=int(matched.get("conviction_rank", 0) or 0),
            planned_target_net_pnl=int(matched.get("planned_target_net_pnl", 0) or 0),
            planned_stop_net_loss_abs=int(matched.get("planned_stop_net_loss_abs", 0) or 0),
            planned_risk_net_loss_abs=int(matched.get("planned_risk_net_loss_abs", 0) or 0),
            entry_expected_net_pnl=float(matched.get("entry_expected_net_pnl", 0.0) or 0.0),
            entry_prediction_net_pnl=int(matched.get("entry_prediction_net_pnl", 0) or 0),
            entry_prediction_lower_net_pnl=int(matched.get("entry_prediction_lower_net_pnl", 0) or 0),
            entry_prediction_win_probability=float(matched.get("entry_prediction_win_probability", 0.0) or 0.0),
            trade_key=trade_key or self._make_trade_key(normalized, buy_time),
        )
        logger.warning(
            "추정청산 후 계좌에 다시 나타난 포지션을 원래 거래로 복구합니다: %s %d주 @ %d원 "
            "(removed_estimated_net=%d)",
            normalized,
            int(position.quantity),
            int(position.buy_price),
            int(matched.get("net_pnl", 0) or 0),
        )
        return position

    def _serialize_position_state(self, pos: PositionState) -> Dict[str, Any]:
        payload = dict(vars(pos))
        payload["buy_time"] = self._serialize_datetime(getattr(pos, "buy_time", None))
        payload["restored_at"] = self._serialize_datetime(getattr(pos, "restored_at", None))
        payload["pending_exit_started_at"] = self._serialize_datetime(getattr(pos, "pending_exit_started_at", None))
        payload["pending_entry_started_at"] = self._serialize_datetime(getattr(pos, "pending_entry_started_at", None))
        return payload

    def _deserialize_position_state(self, payload: Dict[str, Any]) -> Optional[PositionState]:
        symbol = str((payload or {}).get("symbol") or "").strip()
        if not symbol:
            return None
        state = dict(payload or {})
        state["buy_time"] = self._deserialize_datetime(state.get("buy_time")) or self._now()
        state["restored_at"] = self._deserialize_datetime(state.get("restored_at"))
        state["pending_exit_started_at"] = self._deserialize_datetime(state.get("pending_exit_started_at"))
        state["pending_entry_started_at"] = self._deserialize_datetime(state.get("pending_entry_started_at"))
        allowed_fields = {item.name for item in fields(PositionState)}
        state = {key: value for key, value in state.items() if key in allowed_fields}
        try:
            return PositionState(**state)
        except TypeError:
            logger.warning("포지션 상태 복구 중 알 수 없는 필드가 있어 건너뜁니다: %s", symbol)
            return None

    def has_runtime_state_snapshot(self) -> bool:
        daily_pnl = getattr(self, "daily_pnl", None)
        has_daily_pnl = bool(
            daily_pnl is not None
            and (
                int(getattr(daily_pnl, "trade_count", 0) or 0) > 0
                or int(getattr(daily_pnl, "realized_net_pnl", 0) or 0) != 0
            )
        )
        return bool(
            self._state_restored_today
            or self.positions
            or self._pool
            or self._latest_math_queue_symbols
            or self._latest_math_backfill_symbols
            or self._latest_opening_fast_symbols
            or self._latest_opening_hot_symbols
            or self._symbol_entry_cooldown_until
            or self._symbol_order_unavailable
            or self._restore_ignore_until
            or self._halted
            or has_daily_pnl
        )

    def _save_daily_state_if_due(self, *, force: bool = False, min_interval_seconds: int = 5) -> None:
        now = self._now()
        if not force:
            last_saved_at = getattr(self, "_last_daily_state_save_at", None)
            if last_saved_at is not None and (now - last_saved_at).total_seconds() < max(0, int(min_interval_seconds)):
                return
        self._save_daily_state()

    def _save_daily_state(self):
        path = self._daily_state_path()
        payload = {
            "date": self._today(),
            "bull_loss_count_today": self._bull_loss_count_today,
            "halted": self._halted,
            "halt_reason": str(getattr(self, "_halt_reason", "") or ""),
            "ledger_seed_snapshot": dict(self._ledger_seed_snapshot or {}),
            "daily_pnl": {
                "realized_gross_pnl": self.daily_pnl.realized_gross_pnl,
                "realized_net_pnl": self.daily_pnl.realized_net_pnl,
                "fees_paid": self.daily_pnl.fees_paid,
                "taxes_paid": self.daily_pnl.taxes_paid,
                "trade_count": self.daily_pnl.trade_count,
                "win_count": self.daily_pnl.win_count,
                "loss_count": self.daily_pnl.loss_count,
                "breakeven_count": self.daily_pnl.breakeven_count,
                "winning_net_pnl_sum": self.daily_pnl.winning_net_pnl_sum,
                "losing_net_pnl_sum": self.daily_pnl.losing_net_pnl_sum,
                "largest_win_net": self.daily_pnl.largest_win_net,
                "largest_loss_net": self.daily_pnl.largest_loss_net,
            },
            "breaker_excluded_realized_net_pnl": self._breaker_excluded_realized_net_pnl,
            "session_start_at": self._serialize_datetime(self._session_start_at),
            "pool": list(self._pool or []),
            "latest_math_queue_symbols": list(self._latest_math_queue_symbols or []),
            "latest_math_backfill_symbols": list(self._latest_math_backfill_symbols or []),
            "latest_opening_fast_symbols": sorted(self._latest_opening_fast_symbols or set()),
            "latest_opening_hot_symbols": sorted(self._latest_opening_hot_symbols or set()),
            "latest_math_queue_source": dict(self._latest_math_queue_source or {}),
            "pending_entry_meta": {
                str(symbol): dict(meta)
                for symbol, meta in (self._pending_entry_meta or {}).items()
                if str(symbol or "").strip() and isinstance(meta, dict)
            },
            "symbol_entry_cooldown_until": self._serialize_datetime_map(self._symbol_entry_cooldown_until),
            "symbol_order_unavailable": {
                str(symbol): dict(meta)
                for symbol, meta in (self._symbol_order_unavailable or {}).items()
                if str(symbol or "").strip() and isinstance(meta, dict)
            },
            "restore_ignore_until": self._serialize_datetime_map(self._restore_ignore_until),
            "sell_fill_ledger": list(self._sell_fill_ledger or []),
            "closed_trade_ledger": dict(self._closed_trade_ledger or {}),
            "positions": [
                self._serialize_position_state(pos)
                for pos in self.positions.values()
            ],
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            self._last_daily_state_save_at = self._now()
        except OSError:
            logger.exception("일일 상태 저장 실패")

    def _load_daily_state(self):
        path = self._daily_state_path()
        self._state_restored_today = False
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.exception("일일 상태 로드 실패")
            return
        if str(payload.get("date") or "") != self._today():
            return

        self._bull_loss_count_today = int(payload.get("bull_loss_count_today", 0) or 0)
        restored_halted = bool(payload.get("halted", False))
        self._halted = restored_halted
        self._halt_reason = str(payload.get("halt_reason", "") or "")
        daily_pnl = payload.get("daily_pnl") or {}
        for field in self._daily_pnl_snapshot_fields():
            if hasattr(self.daily_pnl, field):
                setattr(self.daily_pnl, field, int(daily_pnl.get(field, getattr(self.daily_pnl, field)) or 0))
        self._breaker_excluded_realized_net_pnl = int(payload.get("breaker_excluded_realized_net_pnl", 0) or 0)
        self._ledger_seed_snapshot = payload.get("ledger_seed_snapshot") or {}
        if not self._ledger_seed_snapshot:
            self._ledger_seed_snapshot = self._capture_daily_pnl_snapshot()
        self._sell_fill_ledger = [
            dict(item)
            for item in (payload.get("sell_fill_ledger") or [])
            if isinstance(item, dict)
        ]
        self._normalize_sell_fill_breaker_flags()
        raw_closed_trade_ledger = payload.get("closed_trade_ledger") or {}
        if isinstance(raw_closed_trade_ledger, dict):
            self._closed_trade_ledger = {
                str(key or "").strip(): dict(value)
                for key, value in raw_closed_trade_ledger.items()
                if str(key or "").strip() and isinstance(value, dict)
            }
        else:
            self._closed_trade_ledger = {}
        if self._sell_fill_ledger or self._closed_trade_ledger:
            self._rebuild_daily_pnl_from_ledgers()
        session_start_at = self._deserialize_datetime(payload.get("session_start_at"))
        if session_start_at is not None:
            self._session_start_at = session_start_at
        pool = [str(symbol or "").strip() for symbol in (payload.get("pool") or []) if str(symbol or "").strip()]
        if pool:
            self._pool = pool
        self._latest_math_queue_symbols = [
            str(symbol or "").strip()
            for symbol in (payload.get("latest_math_queue_symbols") or [])
            if str(symbol or "").strip()
        ]
        self._latest_math_backfill_symbols = [
            str(symbol or "").strip()
            for symbol in (payload.get("latest_math_backfill_symbols") or [])
            if str(symbol or "").strip()
        ]
        self._latest_opening_fast_symbols = {
            str(symbol or "").strip()
            for symbol in (payload.get("latest_opening_fast_symbols") or [])
            if str(symbol or "").strip()
        }
        self._latest_opening_hot_symbols = {
            str(symbol or "").strip()
            for symbol in (payload.get("latest_opening_hot_symbols") or [])
            if str(symbol or "").strip()
        }
        self._latest_math_queue_source = {
            str(symbol or "").strip(): str(source or "").strip()
            for symbol, source in (payload.get("latest_math_queue_source") or {}).items()
            if str(symbol or "").strip()
        }
        self._pending_entry_meta = {
            str(symbol or "").strip(): dict(meta)
            for symbol, meta in (payload.get("pending_entry_meta") or {}).items()
            if str(symbol or "").strip() and isinstance(meta, dict)
        }
        self._symbol_entry_cooldown_until = self._deserialize_datetime_map(payload.get("symbol_entry_cooldown_until") or {})
        self._symbol_order_unavailable = {
            str(symbol or "").strip(): dict(meta)
            for symbol, meta in (payload.get("symbol_order_unavailable") or {}).items()
            if str(symbol or "").strip() and isinstance(meta, dict)
        }
        self._restore_ignore_until = self._deserialize_datetime_map(payload.get("restore_ignore_until") or {})
        restored_positions: Dict[str, PositionState] = {}
        for item in (payload.get("positions") or []):
            pos = self._deserialize_position_state(item)
            if pos is None:
                continue
            restored_positions[pos.symbol] = pos
        if restored_positions:
            self.positions = restored_positions
        if (
            restored_halted
            and bool(self.config.allow_hard_stop_bypass_for_day)
            and int(self.daily_pnl.realized_net_pnl) < int(self.config.daily_profit_target)
        ):
            restored_realized_net_pnl = int(self.daily_pnl.realized_net_pnl)
            self._halted = False
            self._halt_reason = ""
            if not bool(self.config.use_restored_pnl_for_daily_breaker):
                self._breaker_excluded_realized_net_pnl = restored_realized_net_pnl
                if isinstance(self._ledger_seed_snapshot, dict):
                    self._ledger_seed_snapshot["breaker_excluded_realized_net_pnl"] = restored_realized_net_pnl
            logger.warning(
                "당일 하드스탑 우회 설정으로 저장된 halted 상태를 해제합니다: "
                "restored_realized_net_pnl=%d breaker_realized_net_pnl=%d",
                restored_realized_net_pnl,
                self._realized_net_pnl_for_daily_breaker(),
            )
            self._save_daily_state_if_due(force=True)
        if self._restored_loss_halt_is_stale():
            restored_realized_net_pnl = int(self.daily_pnl.realized_net_pnl)
            restored_total_net_pnl = self._total_net_pnl_for_daily_breaker()
            previous_halt_reason = str(getattr(self, "_halt_reason", "") or "")
            self._halted = False
            self._halt_reason = ""
            logger.warning(
                "저장된 손실 하드스탑이 현재 손익과 맞지 않아 해제합니다: "
                "previous_reason=%s realized_net_pnl=%d breaker_net_pnl=%d",
                previous_halt_reason,
                restored_realized_net_pnl,
                restored_total_net_pnl,
            )
            self._save_daily_state_if_due(force=True)
        self._state_restored_today = bool(
            restored_positions
            or self._pool
            or self._latest_math_queue_symbols
            or self._latest_math_backfill_symbols
            or self._latest_opening_fast_symbols
            or self._latest_opening_hot_symbols
            or self._symbol_entry_cooldown_until
            or self._symbol_order_unavailable
            or self._restore_ignore_until
            or self._halted
            or int(getattr(self.daily_pnl, "trade_count", 0) or 0) > 0
            or int(getattr(self.daily_pnl, "realized_net_pnl", 0) or 0) != 0
        )

    def _position_from_entry_meta(
        self,
        *,
        symbol: str,
        buy_price: int,
        quantity: int,
        invested_amount: int,
        buy_time: datetime,
        meta: Dict[str, Any],
    ) -> PositionState:
        meta = dict(meta or {})
        return PositionState(
            symbol=symbol,
            buy_price=int(buy_price or 0),
            quantity=int(quantity or 0),
            invested_amount=max(0, int(invested_amount or 0)),
            buy_time=buy_time,
            entry_strategy_name=str(meta.get("strategy_name", "")),
            entry_setup_name=str(meta.get("setup_name", "")),
            entry_reason=str(meta.get("entry_reason", meta.get("setup_name", ""))),
            regime_label=str(meta.get("regime_label", self._resolve_regime_profile_name())),
            bear_score=int(meta.get("bear_score", self._bear_score) or 0),
            planned_risk_stage=str(meta.get("planned_risk_stage", self._current_bull_risk_mode())),
            entry_grade=str(meta.get("entry_grade", "")),
            leader_score=float(meta.get("leader_score", 0.0) or 0.0),
            leader_percentile=float(meta.get("leader_percentile", 0.0) or 0.0),
            entry_grade_math=str(meta.get("entry_grade_math", "")),
            entry_ev=float(meta.get("entry_ev", 0.0) or 0.0),
            entry_ev_confidence=str(meta.get("entry_ev_confidence", "")),
            conviction_tier=str(meta.get("conviction_tier", "")),
            bull_risk_mode=str(meta.get("bull_risk_mode", self._current_bull_risk_mode())),
            post_loss_admission_class=str(meta.get("post_loss_admission_class", "general")),
            candidate_class=str(meta.get("candidate_class", "")),
            execution_mode=str(meta.get("execution_mode", "live")),
            live_route=str(meta.get("live_route", "")),
            queue_source=str(meta.get("queue_source", "")),
            size_multiplier=float(meta.get("size_multiplier", 1.0) or 1.0),
            conviction_score=float(meta.get("conviction_score", 0.0) or 0.0),
            conviction_rank=int(meta.get("conviction_rank", 0) or 0),
            bull_prob=float(meta.get("bull_prob", 0.0) or 0.0),
            neutral_prob=float(meta.get("neutral_prob", 0.0) or 0.0),
            soft_bear_prob=float(meta.get("soft_bear_prob", 0.0) or 0.0),
            bear_prob=float(meta.get("bear_prob", 0.0) or 0.0),
            shock_score=float(meta.get("shock_score", 0.0) or 0.0),
            shock_confidence=float(meta.get("shock_confidence", 0.0) or 0.0),
            adaptive_take_profit_pct=float(meta.get("adaptive_take_profit_pct", meta.get("planned_take_profit_pct", 0.0)) or 0.0),
            adaptive_stop_loss_pct=float(meta.get("adaptive_stop_loss_pct", 0.0) or 0.0),
            adaptive_trailing_activation_pct=float(meta.get("adaptive_trailing_activation_pct", 0.0) or 0.0),
            adaptive_trailing_stop_pct=float(meta.get("adaptive_trailing_stop_pct", 0.0) or 0.0),
            adaptive_max_hold_minutes=int(meta.get("adaptive_max_hold_minutes", 0) or 0),
            planned_target_net_pnl=int(meta.get("planned_target_net_pnl", 0) or 0),
            planned_stop_net_loss_abs=int(meta.get("planned_stop_net_loss_abs", 0) or 0),
            planned_risk_net_loss_abs=int(
                meta.get("planned_risk_net_loss_abs", meta.get("planned_stop_net_loss_abs", 0)) or 0
            ),
            entry_expected_net_pnl=float(meta.get("entry_expected_net_pnl", 0.0) or 0.0),
            entry_prediction_net_pnl=int(meta.get("price_prediction_net_pnl", meta.get("entry_prediction_net_pnl", 0)) or 0),
            entry_prediction_lower_net_pnl=int(
                meta.get("price_prediction_lower_net_pnl", meta.get("entry_prediction_lower_net_pnl", 0)) or 0
            ),
            entry_prediction_upper_net_pnl=int(
                meta.get("price_prediction_upper_net_pnl", meta.get("entry_prediction_upper_net_pnl", 0)) or 0
            ),
            entry_prediction_win_probability=float(meta.get("entry_prediction_win_probability", 0.0) or 0.0),
            entry_signal_price=int(
                meta.get(
                    "entry_signal_price",
                    meta.get("pending_order_reference_price", meta.get("price_prediction_signal_price", 0)),
                )
                or 0
            ),
            entry_prediction_return_pct=float(meta.get("price_prediction_return_pct", 0.0) or 0.0),
            entry_prediction_lower_pct=float(meta.get("price_prediction_lower_pct", 0.0) or 0.0),
            entry_prediction_upper_pct=float(meta.get("price_prediction_upper_pct", 0.0) or 0.0),
            trade_key=self._make_trade_key(symbol, buy_time),
        )

    def _pending_entry_reconcile_grace_seconds(self) -> int:
        return max(
            30,
            int(
                getattr(
                    self.config,
                    "pending_entry_reconcile_grace_seconds",
                    getattr(self.config, "pending_order_block_seconds", 180),
                )
                or 180
            ),
        )

    def _pending_entry_meta_in_reconcile_grace(
        self,
        meta: Dict[str, Any],
        *,
        now: Optional[datetime] = None,
    ) -> bool:
        moment = now or self._now()
        created_at = self._deserialize_datetime(
            meta.get("pending_order_created_at") or meta.get("entry_signal_timestamp")
        )
        if created_at is None:
            return True
        return (moment - created_at).total_seconds() < self._pending_entry_reconcile_grace_seconds()

    def _clear_pending_entry_position_state(self, pos: PositionState) -> None:
        pos.pending_entry_started_at = None
        pos.pending_entry_reference_price = 0
        pos.pending_entry_fill_mode = ""

    def _net_for_signal_projection_pct(
        self,
        *,
        signal_price: int,
        entry_price: int,
        quantity: int,
        return_pct: float,
    ) -> int:
        qty = max(0, int(quantity or 0))
        base_price = max(0, int(signal_price or 0))
        buy_price = max(0, int(entry_price or 0))
        if qty <= 0 or base_price <= 0 or buy_price <= 0:
            return 0
        projected_exit_price = max(
            1,
            int(
                round(
                    float(base_price)
                    * (1.0 + (float(return_pct) / 100.0) - float(self.config.exit_market_slippage_rate))
                )
            ),
        )
        return self._estimate_trade_net_pnl_from_prices(
            entry_price=buy_price,
            exit_price=projected_exit_price,
            quantity=qty,
        )

    def _reprice_position_ev_after_confirmed_entry(
        self,
        pos: PositionState,
        *,
        signal_price: int = 0,
    ) -> bool:
        if pos is None or str(getattr(pos, "entry_setup_name", "") or "") != "expected_value":
            return False
        qty = max(0, int(getattr(pos, "quantity", 0) or 0))
        buy_price = max(0, int(getattr(pos, "buy_price", 0) or 0))
        base_price = max(
            0,
            int(signal_price or 0),
            int(getattr(pos, "entry_signal_price", 0) or 0),
            int(getattr(pos, "pending_entry_reference_price", 0) or 0),
        )
        predicted_pct = float(getattr(pos, "entry_prediction_return_pct", 0.0) or 0.0)
        lower_pct = float(getattr(pos, "entry_prediction_lower_pct", 0.0) or 0.0)
        upper_pct = float(getattr(pos, "entry_prediction_upper_pct", 0.0) or 0.0)
        if qty <= 0 or buy_price <= 0 or base_price <= 0 or predicted_pct == 0.0:
            return False

        predicted_net = self._net_for_signal_projection_pct(
            signal_price=base_price,
            entry_price=buy_price,
            quantity=qty,
            return_pct=predicted_pct,
        )
        lower_net = self._net_for_signal_projection_pct(
            signal_price=base_price,
            entry_price=buy_price,
            quantity=qty,
            return_pct=lower_pct,
        )
        upper_net = self._net_for_signal_projection_pct(
            signal_price=base_price,
            entry_price=buy_price,
            quantity=qty,
            return_pct=upper_pct,
        )
        flat_net = self._estimate_trade_net_pnl_from_prices(
            entry_price=buy_price,
            exit_price=buy_price,
            quantity=qty,
        )
        cost_floor = abs(min(0, int(flat_net))) + 1
        lower_risk = abs(min(0, int(lower_net)))
        prior_stop = max(0, int(getattr(pos, "planned_stop_net_loss_abs", 0) or 0))
        prior_risk = max(
            prior_stop,
            int(getattr(pos, "planned_risk_net_loss_abs", 0) or 0),
        )
        repriced_stop = max(1, cost_floor, lower_risk)
        if prior_stop > 0 and lower_risk <= cost_floor:
            planned_stop = max(cost_floor, prior_stop)
        else:
            planned_stop = min(prior_stop, repriced_stop) if prior_stop > 0 else repriced_stop
        planned_stop = max(1, int(planned_stop))
        execution_buffer = max(
            1,
            int(
                round(
                    float(cost_floor) * 0.10
                    + float(buy_price * qty)
                    * max(0.0, float(self.config.exit_market_slippage_rate))
                    * 0.25
                )
            ),
        )
        planned_risk = max(
            1,
            prior_risk,
            max(planned_stop, lower_risk) + execution_buffer,
        )
        win_probability = self._clip_float(
            float(getattr(pos, "entry_prediction_win_probability", 0.0) or 0.0),
            0.0,
            1.0,
        )
        break_even_probability = (
            float(planned_risk) / max(1.0, float(max(0, int(predicted_net)) + planned_risk))
            if predicted_net > 0
            else 1.0
        )
        expected_net = (
            (win_probability * float(predicted_net))
            - ((1.0 - win_probability) * float(planned_risk))
        )

        pos.entry_signal_price = int(base_price)
        pos.entry_prediction_net_pnl = int(predicted_net)
        pos.entry_prediction_lower_net_pnl = int(lower_net)
        pos.entry_prediction_upper_net_pnl = int(upper_net)
        pos.entry_expected_net_pnl = round(float(expected_net), 2)
        pos.entry_ev = round(float(expected_net), 2)
        pos.entry_ev_confidence = "live_plan_repriced"

        prior_target = max(0, int(getattr(pos, "planned_target_net_pnl", 0) or 0))
        if predicted_net > 0:
            pos.planned_target_net_pnl = max(1, min(prior_target or int(predicted_net), int(predicted_net)))
        else:
            pos.planned_target_net_pnl = 1

        if predicted_net <= 0 or expected_net <= 0.0 or win_probability <= break_even_probability:
            pos.planned_stop_net_loss_abs = 1
            pos.planned_risk_net_loss_abs = int(planned_risk)
            if int(getattr(pos, "adaptive_max_hold_minutes", 0) or 0) <= 0:
                pos.adaptive_max_hold_minutes = 1
            logger.warning(
                "실체결가 기준 EV 재가격정 결과 보유 우위가 사라져 방어 청산 계획으로 전환합니다: "
                "%s signal=%d entry=%d qty=%d pred_net=%d lower_net=%d exp=%.2f win=%.3f be=%.3f",
                pos.symbol,
                int(base_price),
                int(buy_price),
                int(qty),
                int(predicted_net),
                int(lower_net),
                float(expected_net),
                float(win_probability),
                float(break_even_probability),
            )
        else:
            pos.planned_stop_net_loss_abs = int(planned_stop)
            pos.planned_risk_net_loss_abs = int(planned_risk)
            logger.info(
                "실체결가 기준 EV 재가격정: %s signal=%d entry=%d qty=%d pred_net=%d "
                "lower_net=%d exp=%.2f stop=%d target=%d",
                pos.symbol,
                int(base_price),
                int(buy_price),
                int(qty),
                int(predicted_net),
                int(lower_net),
                float(expected_net),
                int(pos.planned_stop_net_loss_abs),
                int(pos.planned_target_net_pnl),
            )
        return True

    def _position_pending_entry_reconcile_pending(
        self,
        pos: PositionState,
        *,
        now: Optional[datetime] = None,
    ) -> bool:
        started_at = getattr(pos, "pending_entry_started_at", None)
        if started_at is None:
            return False
        fill_mode = str(getattr(pos, "pending_entry_fill_mode", "") or "")
        if fill_mode not in {"market_pending", "limit_then_market_pending", "partial_fill_pending"}:
            return False
        moment = now or self._now()
        return (moment - started_at).total_seconds() < self._pending_entry_reconcile_grace_seconds()

    def _position_pending_exit_reconcile_pending(
        self,
        pos: PositionState,
        *,
        now: Optional[datetime] = None,
    ) -> bool:
        started_at = getattr(pos, "pending_exit_started_at", None)
        if started_at is None:
            return False
        fill_mode = str(getattr(pos, "pending_exit_fill_mode", "") or "")
        if fill_mode not in {"market_pending", "limit_then_market_pending", "order_result_pending"}:
            return False
        if int(getattr(pos, "pending_exit_quantity", 0) or 0) <= 0:
            return False
        moment = now or self._now()
        return (moment - started_at).total_seconds() < self._pending_entry_reconcile_grace_seconds()

    def _clear_account_absent_stale_pending_entries(self, account_symbols: set[str]) -> None:
        if not self._pending_entry_meta:
            return
        now = self._now()
        grace_seconds = self._pending_entry_reconcile_grace_seconds()
        cleared: List[str] = []
        for raw_symbol, raw_meta in list(self._pending_entry_meta.items()):
            symbol = str(raw_symbol or "").strip()
            if not symbol or symbol in account_symbols or symbol in self.positions:
                continue
            if not isinstance(raw_meta, dict):
                continue
            created_at = self._deserialize_datetime(
                raw_meta.get("pending_order_created_at") or raw_meta.get("entry_signal_timestamp")
            )
            if created_at is None:
                continue
            if (now - created_at).total_seconds() < grace_seconds:
                continue
            self._pending_entry_meta.pop(symbol, None)
            cleared.append(symbol)
        if cleared:
            logger.warning(
                "계좌 보유에서 확인되지 않은 오래된 pending 진입 메타를 정리합니다: %s",
                ",".join(sorted(cleared)),
            )
            self._save_daily_state_if_due(force=True)

    def _reconcile_pending_exit_account_quantity(
        self,
        pos: PositionState,
        *,
        account_quantity: int,
    ) -> Optional[PositionState]:
        before_quantity = max(0, int(getattr(pos, "quantity", 0) or 0))
        after_quantity = max(0, int(account_quantity or 0))
        if after_quantity >= before_quantity or getattr(pos, "pending_exit_started_at", None) is None:
            return pos

        sell_quantity = before_quantity - after_quantity
        sell_price = max(0, int(getattr(pos, "pending_exit_reference_price", 0) or 0))
        if sell_price <= 0:
            quote = self._quotes_cache.get(pos.symbol)
            if quote is None:
                recent_quotes = list(self._recent_quotes.get(pos.symbol, []))
                quote = recent_quotes[-1] if recent_quotes else None
            if quote is not None:
                sell_price = max(0, int(getattr(quote, "current_price", 0) or 0))
        if sell_price <= 0:
            logger.warning(
                "계좌 수량 감소를 확인했지만 매도 추정가가 없어 포지션을 유지합니다: %s %d→%d주",
                pos.symbol,
                before_quantity,
                after_quantity,
            )
            return pos

        synthetic = OrderResult(
            success=True,
            order_no=str(getattr(pos, "pending_exit_order_no", "") or ""),
            message="account_quantity_decrease_reconciled",
            symbol=pos.symbol,
            side=OrderSide.SELL,
            quantity=sell_quantity,
            price=sell_price,
            reference_price=sell_price,
            fill_mode="account_reconciled_estimated",
            requested_reason=str(getattr(pos, "pending_exit_reason", "") or "account_reconcile"),
            timestamp=self._now(),
        )
        logger.warning(
            "pending 매도 후 계좌 수량 감소를 체결로 반영합니다: %s %d주 @ %d원 (계좌 %d→%d주)",
            pos.symbol,
            sell_quantity,
            sell_price,
            before_quantity,
            after_quantity,
        )
        self.on_order_filled(synthetic)
        return self.positions.get(pos.symbol)

    def sync_positions_from_account(self, account_positions: List[Position]):
        now = self._now()
        synced: Dict[str, PositionState] = {}
        account_symbols = {
            str(getattr(item, "symbol", "") or "").strip()
            for item in (account_positions or [])
            if str(getattr(item, "symbol", "") or "").strip()
        }
        self._restore_ignore_until = {
            symbol: until
            for symbol, until in self._restore_ignore_until.items()
            if until > now
        }
        for item in account_positions or []:
            symbol = str(getattr(item, "symbol", "") or "").strip()
            if not symbol:
                continue
            ignore_until = self._restore_ignore_until.get(symbol)
            if ignore_until is not None and ignore_until > now:
                logger.info("계좌 복구 무시 유지: %s (%.0fs 남음)", symbol, (ignore_until - now).total_seconds())
                continue
            existing = self.positions.get(symbol)
            if existing is not None:
                account_quantity = max(0, int(getattr(item, "quantity", 0) or 0))
                existing = self._reconcile_pending_exit_account_quantity(
                    existing,
                    account_quantity=account_quantity,
                )
                if existing is None:
                    continue
                previous_quantity = max(0, int(getattr(existing, "quantity", 0) or 0))
                pending_reference_price = max(
                    0,
                    int(getattr(existing, "pending_entry_reference_price", 0) or 0),
                    int(getattr(existing, "entry_signal_price", 0) or 0),
                )
                existing.quantity = account_quantity
                existing.buy_price = int(round(float(getattr(item, "avg_price", existing.buy_price) or existing.buy_price)))
                existing.invested_amount = max(0, existing.buy_price * existing.quantity)
                existing.high_since_buy = max(existing.high_since_buy, existing.buy_price)
                if getattr(existing, "pending_entry_started_at", None) is not None:
                    if pending_reference_price > 0:
                        existing.entry_signal_price = int(pending_reference_price)
                    self._reprice_position_ev_after_confirmed_entry(
                        existing,
                        signal_price=int(pending_reference_price),
                    )
                    self._clear_pending_entry_position_state(existing)
                    self._pending_entry_meta.pop(symbol, None)
                    logger.warning(
                        "계좌 동기화로 pending 매수 임시 포지션을 확정합니다: %s %d주 @ %d원",
                        symbol,
                        int(existing.quantity or 0),
                        int(existing.buy_price or 0),
                    )
                elif account_quantity > previous_quantity:
                    self._reprice_position_ev_after_confirmed_entry(
                        existing,
                        signal_price=int(pending_reference_price),
                    )
                    logger.warning(
                        "계좌 수량 증가를 실제 총수량 기준 EV에 반영합니다: %s %d→%d주 @ %d원",
                        symbol,
                        previous_quantity,
                        account_quantity,
                        int(existing.buy_price or 0),
                    )
                if not str(getattr(existing, "trade_key", "") or "").strip():
                    existing.trade_key = self._make_trade_key(symbol, existing.buy_time)
                synced[symbol] = existing
                continue
            pending_meta = self._pending_entry_meta.pop(symbol, None)
            if isinstance(pending_meta, dict):
                buy_price = int(round(float(getattr(item, "avg_price", 0) or 0)))
                quantity = int(getattr(item, "quantity", 0) or 0)
                position = self._position_from_entry_meta(
                    symbol=symbol,
                    buy_price=buy_price,
                    quantity=quantity,
                    invested_amount=max(0, buy_price * quantity),
                    buy_time=now,
                    meta=pending_meta,
                )
                self._reprice_position_ev_after_confirmed_entry(
                    position,
                    signal_price=int(
                        pending_meta.get(
                            "entry_signal_price",
                            pending_meta.get("pending_order_reference_price", 0),
                        )
                        or 0
                    ),
                )
                synced[symbol] = position
                logger.warning(
                    "계좌 동기화로 pending 매수 체결을 실전 포지션으로 복구합니다: %s %d주 @ %d원",
                    symbol,
                    quantity,
                    buy_price,
                )
                continue
            restored_quantity = int(getattr(item, "quantity", 0) or 0)
            restored_buy_price = int(round(float(getattr(item, "avg_price", 0) or 0)))
            recovered_position = self._pop_estimated_exit_for_account_restore(
                symbol,
                quantity=restored_quantity,
                buy_price=restored_buy_price,
                now=now,
            )
            if recovered_position is not None:
                synced[symbol] = recovered_position
                continue
            restored_strategy = OPENING_STRATEGY if self._opening_conviction_window_active() else INTRADAY_STRATEGY
            synced[symbol] = PositionState(
                symbol=symbol,
                buy_price=restored_buy_price,
                quantity=restored_quantity,
                invested_amount=int(getattr(item, "eval_amount", 0) or 0),
                buy_time=now,
                is_restored=True,
                restored_at=now,
                entry_strategy_name=restored_strategy,
                entry_setup_name="restored_position",
                entry_reason="restored_position",
                regime_label=self._resolve_regime_profile_name(),
                bear_score=self._bear_score,
                planned_risk_stage=self._current_bull_risk_mode(),
                queue_source="account_restore",
                live_route=restored_strategy,
                trade_key=self._make_trade_key(symbol, now),
            )
            order_logger.info(
                "기존보유 복원: %s %d주 @ 평균단가 %s원 "
                "(실시간 매수주문 아님, source=account_restore, setup_name=restored_position)",
                symbol,
                restored_quantity,
                f"{restored_buy_price:,}" if restored_buy_price > 0 else "0",
            )
        for symbol, existing in list(self.positions.items()):
            if symbol in synced or symbol in account_symbols:
                continue
            if self._position_pending_entry_reconcile_pending(existing, now=now):
                synced[symbol] = existing
                logger.warning(
                    "계좌 보유 미반영 pending 매수 포지션을 로컬에서 유지합니다: %s %d주 @ %d원 "
                    "(fill_mode=%s)",
                    symbol,
                    int(existing.quantity or 0),
                    int(existing.buy_price or 0),
                    str(getattr(existing, "pending_entry_fill_mode", "") or ""),
                )
                continue
            if getattr(existing, "pending_entry_started_at", None) is not None:
                logger.warning(
                    "유예시간 동안 계좌에서 확인되지 않은 pending 매수 임시 포지션을 손익 없이 제거합니다: "
                    "%s %d주 @ %d원",
                    symbol,
                    int(existing.quantity or 0),
                    int(existing.buy_price or 0),
                )
                continue
            if getattr(existing, "pending_exit_started_at", None) is not None:
                reconciled = self._reconcile_pending_exit_account_quantity(
                    existing,
                    account_quantity=0,
                )
                if reconciled is not None:
                    synced[symbol] = reconciled
        self.positions = synced
        self._clear_account_absent_stale_pending_entries(account_symbols)
        self._state_restored_today = self._state_restored_today or bool(synced)
        self._save_daily_state()

    # ------------------------------------------------------------------
    # 데이터 / 게이트 로딩
    # ------------------------------------------------------------------
    def _load_entry_ev_data(self):
        report_root = Path("reports")
        if not report_root.exists():
            self._entry_ev_table = {}
            self._entry_ev_history_records = []
            return
        raw_scorecards = load_recent_scorecards(
            report_root,
            max(int(self.config.ev_window_days), int(self.config.conviction_ev_window_days)),
        )
        max_age_days = max(
            int(getattr(self.config, "ev_scorecard_max_age_days", 8) or 0),
            int(getattr(self.config, "conviction_ev_scorecard_max_age_days", 14) or 0),
        )
        scorecards: List[Dict[str, Any]] = []
        today = date.today()
        for card in raw_scorecards:
            card_date_text = str(card.get("date") or "")
            try:
                card_date = date.fromisoformat(card_date_text)
            except ValueError:
                scorecards.append(card)
                continue
            if max_age_days <= 0 or (today - card_date).days <= max_age_days:
                scorecards.append(card)
        self._entry_ev_table = build_entry_ev_table(
            scorecards,
            window_days=int(self.config.ev_window_days),
            min_samples=int(self.config.ev_min_samples),
        )
        history_records: List[Dict[str, Any]] = []
        conviction_window = int(self.config.conviction_ev_window_days)
        window_cards = scorecards[-conviction_window:] if conviction_window > 0 else scorecards
        for card in window_cards:
            history_records.extend(((card.get("log_analysis") or {}).get("trade_records") or []))
        self._entry_ev_history_records = history_records

    # ------------------------------------------------------------------
    # 레짐 / 시간 / 공통 상태
    # ------------------------------------------------------------------
    def _market_elapsed_since_open_seconds(self) -> int:
        now = self._now()
        session_start = self._session_start_at or now.replace(hour=9, minute=0, second=0, microsecond=0)
        return max(0, int((now - session_start).total_seconds()))

    def _minutes_since_market_open(self) -> int:
        return self._market_elapsed_since_open_seconds() // 60


    def _market_close_minutes_since_open(self) -> int:
        now = self._now()
        session_start = self._session_start_at or now.replace(hour=9, minute=0, second=0, microsecond=0)
        market_close = session_start.replace(hour=15, minute=30, second=0, microsecond=0)
        return max(0, int((market_close - session_start).total_seconds() // 60))




    def _opening_conviction_window_active(self) -> bool:
        if not self.config.enable_opening_conviction_lane:
            return False
        return self._minutes_since_market_open() < int(self.config.opening_conviction_window_minutes)

    def _intraday_conviction_window_active(self) -> bool:
        if not self.config.enable_intraday_conviction_lane:
            return False
        minutes = self._minutes_since_market_open()
        if minutes < int(self.config.opening_conviction_window_minutes):
            return False
        end_minutes = min(
            int(getattr(self.config, "intraday_conviction_end_minutes_after_open", 381) or 381),
            self._market_close_minutes_since_open(),
        )
        return minutes < end_minutes

    def _opening_fast_window_active(self) -> bool:
        return self.config.enable_opening_fast_lane and self._minutes_since_market_open() < int(self.config.opening_fast_window_minutes)

    def _opening_candidate_window_active(self) -> bool:
        return self._minutes_since_market_open() < int(self.config.opening_candidate_window_minutes)

    def _resolve_regime_profile_name(self) -> str:
        probs = self._latest_regime_probabilities
        if probs is not None:
            dominant = probs.dominant_profile()
            if dominant == "bull":
                return "bull"
            if dominant == "soft_bear":
                return "soft_bear"
            if dominant == "bear":
                return "bear"
            return "neutral"
        if self._bear_score >= 3:
            return "bear"
        if self._bear_score >= 2:
            return "soft_bear"
        if self._bear_score == 1:
            return "neutral"
        return "bull"

    def _current_bull_risk_mode(self) -> str:
        first, second, third = self.config.bull_risk_mode_loss_thresholds
        count = int(self._bull_loss_count_today)
        if count >= int(third):
            if self._bull_risk_stop_softened_by_market_context():
                return "restricted"
            return "stop"
        if count >= int(second):
            return "restricted"
        if count >= int(first):
            return "guarded"
        return "normal"

    def _bull_risk_stop_softened_by_market_context(self) -> bool:
        if not bool(getattr(self.config, "bull_risk_mode_stop_soften_enabled", True)):
            return False
        if self._halted:
            return False
        realized = int(self._realized_net_pnl_for_daily_breaker())
        min_net = int(getattr(self.config, "bull_risk_mode_stop_soften_min_net_pnl", -2500) or -2500)
        if realized < min_net:
            return False
        if self._resolve_regime_profile_name() == "bear":
            return False
        if int(self._bear_score) >= 2 and not self._strong_bull_override_active:
            return False
        return self._strong_bull_override_active or self._bull_market_context in {"broad_bull", "bull"}

    def _recover_bull_loss_count_after_win(self, net_pnl: int) -> None:
        if not bool(getattr(self.config, "bull_risk_mode_profit_recovery_enabled", True)):
            return
        if int(self._bull_loss_count_today) <= 0:
            return
        threshold = max(1, int(getattr(self.config, "bull_risk_mode_profit_recovery_min_net", 1200) or 1200))
        if int(net_pnl) < threshold:
            return
        self._bull_loss_count_today = max(0, int(self._bull_loss_count_today) - 1)
        logger.info(
            "수익 회복으로 상승장 손절 카운트를 낮춥니다: net_pnl=%d bull_loss_count=%d",
            int(net_pnl),
            int(self._bull_loss_count_today),
        )

    def _cooldown_remaining(self, cooldowns: Dict[str, datetime], key: str) -> float:
        normalized = str(key or "").strip()
        if not normalized:
            return 0.0
        until = cooldowns.get(normalized)
        if until is None:
            return 0.0
        remaining = (until - self._now()).total_seconds()
        if remaining <= 0:
            cooldowns.pop(normalized, None)
            return 0.0
        return float(remaining)

    def _mark_cooldown(self, cooldowns: Dict[str, datetime], key: str, *, seconds: int) -> None:
        normalized = str(key or "").strip()
        cooldown_seconds = max(0, int(seconds))
        if not normalized or cooldown_seconds <= 0:
            return
        candidate_until = self._now() + timedelta(seconds=cooldown_seconds)
        current_until = cooldowns.get(normalized)
        if current_until is None or candidate_until > current_until:
            cooldowns[normalized] = candidate_until
            self._save_daily_state_if_due(force=True)

    def _symbol_entry_cooldown_remaining(self, symbol: str) -> float:
        return self._cooldown_remaining(self._symbol_entry_cooldown_until, symbol)

    def _mark_symbol_entry_cooldown(self, symbol: str, *, seconds: int) -> None:
        self._mark_cooldown(self._symbol_entry_cooldown_until, symbol, seconds=seconds)

    def _is_symbol_order_unavailable(self, symbol: str) -> bool:
        normalized = str(symbol or "").strip()
        if not normalized:
            return False
        return normalized in (self._symbol_order_unavailable or {})

    def _mark_symbol_order_unavailable(self, symbol: str, *, reason: str = "", error_code: str = "") -> None:
        normalized = str(symbol or "").strip()
        if not normalized:
            return
        self._symbol_order_unavailable[normalized] = {
            "reason": str(reason or "symbol_order_unavailable"),
            "error_code": str(error_code or ""),
            "marked_at": self._now().isoformat(timespec="seconds"),
        }
        logger.warning(
            "종목 주문불가 캐시 등록: %s reason=%s error_code=%s",
            normalized,
            self._symbol_order_unavailable[normalized]["reason"],
            self._symbol_order_unavailable[normalized]["error_code"] or "-",
        )
        self._save_daily_state_if_due(force=True)


    def _client_rate_limit_cooldown(self) -> float:
        client = getattr(self.market_data, "client", None)
        remaining_fn = getattr(client, "rate_limit_cooldown_remaining", None)
        if not callable(remaining_fn):
            return 0.0
        try:
            return max(0.0, float(remaining_fn()))
        except Exception:
            return 0.0


    def _long_position_symbols(self) -> List[str]:
        return list(self.positions.keys())


    def _max_long_positions_allowed(self) -> int:
        profile = self._resolve_regime_profile_name()
        candidates = [
            ("bull", self.config.bull_max_position_count),
            ("neutral", self.config.neutral_max_position_count),
            ("soft_bear", self.config.soft_bear_max_position_count),
            ("bear", self.config.bear_max_position_count),
        ]
        override = {name: value for name, value in candidates}.get(profile)
        base = int(override if override is not None else self.config.max_position_count)
        if base <= 0:
            return 999999
        if self._bull_market_context == "fragile_bull" and int(self.config.fragile_bull_max_long_positions) > 0:
            return min(base, int(self.config.fragile_bull_max_long_positions))
        return base

    # ------------------------------------------------------------------
    # 시그널 / 큐 / EV
    # ------------------------------------------------------------------
    def _leader_signal_for_quote(self, quote: Quote) -> LeaderSignal:
        cached = self._latest_math_leader_signals.get(quote.symbol)
        if cached is not None:
            return cached
        avg_volume = int(self._avg_volumes.get(quote.symbol, 0) or 0)
        vs_open_pct = (
            ((int(quote.current_price or 0) - int(quote.open_price or 0)) / max(1, int(quote.open_price or 0))) * 100.0
            if int(quote.open_price or 0) > 0
            else 0.0
        )
        high_proximity = (
            min(1.0, max(0.0, int(quote.current_price or 0) / max(1, int(quote.high_price or 0))))
            if int(quote.high_price or 0) > 0
            else 0.0
        )
        volume_vs_avg = (int(quote.volume or 0) / avg_volume) if avg_volume > 0 else 1.0
        recent_accel = 0.0
        recent_quotes = list(self._recent_quotes.get(quote.symbol, []))
        if len(recent_quotes) >= 2:
            start = recent_quotes[0].current_price
            end = recent_quotes[-1].current_price
            if int(start or 0) > 0:
                recent_accel = ((int(end or 0) - int(start or 0)) / int(start)) * 100.0
        return LeaderSignal(
            symbol=quote.symbol,
            leader_score=0.0,
            leader_percentile=0.0,
            entry_grade="C",
            change_rate=float(quote.change_rate or 0.0),
            trade_amount=int(quote.trade_amount or 0),
            vs_open_pct=round(vs_open_pct, 6),
            high_proximity=round(high_proximity, 6),
            volume_vs_avg=round(volume_vs_avg, 6),
            reclaim_speed_ticks=99,
            recent_acceleration_pct=round(recent_accel, 6),
            effective_leader_score=0.0,
        )

    def _queue_source_for_symbol(self, symbol: str) -> str:
        if symbol in self._latest_opening_fast_symbols and self._opening_fast_window_active():
            return "opening_fast_queue"
        if symbol in self._latest_opening_hot_symbols and self._opening_candidate_window_active():
            return "opening_hot_queue"
        if symbol in self._latest_math_queue_symbols:
            return "math_queue"
        if symbol in self._latest_math_backfill_symbols:
            return "math_backfill"
        source = self._latest_math_queue_source.get(symbol, "")
        if source in {"opening_fast_queue", "opening_hot_queue"} and not self._opening_candidate_window_active():
            return ""
        return source

    def _strategy_family_names(self, strategy_name: str) -> List[str]:
        if strategy_name in {OPENING_STRATEGY, INTRADAY_STRATEGY, LEGACY_LONG_STRATEGY}:
            return [strategy_name, OPENING_STRATEGY, INTRADAY_STRATEGY, LEGACY_LONG_STRATEGY]
        return [strategy_name]

    def _wrap_estimate(
        self,
        strategy_name: str,
        regime_label: str,
        hour_bucket: str,
        entry_grade_math: str,
        source: ExpectedValueEstimate,
    ) -> ExpectedValueEstimate:
        return ExpectedValueEstimate(
            strategy_name=strategy_name,
            regime_label=regime_label,
            hour_bucket=hour_bucket,
            entry_grade=entry_grade_math,
            entry_ev=float(source.entry_ev),
            p_win=float(source.p_win),
            confidence=str(source.confidence),
            closed_trades=int(source.closed_trades),
        )

    def _aggregate_family_history(
        self,
        family_names: Sequence[str],
        regime_label: str,
        entry_grade_math: str,
        strategy_name: str,
        hour_bucket: str,
    ) -> ExpectedValueEstimate:
        family = set(family_names)

        def _select_records(*, exact_regime: bool, exact_grade: bool) -> List[Dict[str, Any]]:
            selected: List[Dict[str, Any]] = []
            for item in self._entry_ev_history_records:
                if str(item.get("strategy_name") or "") not in family:
                    continue
                item_regime = str(item.get("regime_label") or "")
                item_grade = str(item.get("entry_grade_math") or "")
                if exact_regime and item_regime != regime_label:
                    continue
                if exact_grade and item_grade != entry_grade_math:
                    continue
                selected.append(item)
            return selected

        min_samples = max(1, int(self.config.ev_min_samples))
        records = _select_records(exact_regime=True, exact_grade=True)
        if len(records) < min_samples:
            broader = _select_records(exact_regime=True, exact_grade=False)
            if len(broader) > len(records):
                records = broader
        if len(records) < min_samples and strategy_name in {OPENING_STRATEGY, INTRADAY_STRATEGY}:
            broader = _select_records(exact_regime=False, exact_grade=True)
            if len(broader) > len(records):
                records = broader
        if len(records) < min_samples and strategy_name in {OPENING_STRATEGY, INTRADAY_STRATEGY}:
            broader = _select_records(exact_regime=False, exact_grade=False)
            if len(broader) > len(records):
                records = broader
        if not records:
            return ExpectedValueEstimate(
                strategy_name=strategy_name,
                regime_label=regime_label,
                hour_bucket=hour_bucket,
                entry_grade=entry_grade_math,
                entry_ev=0.0,
                p_win=0.5,
                confidence="none",
                closed_trades=0,
            )
        pnl_values = [float(item.get("net_pnl") or 0.0) for item in records]
        wins = sum(1 for value in pnl_values if value > 0)
        trades = len(pnl_values)
        p_win = (wins + 1) / (trades + 2)
        positive = [value for value in pnl_values if value > 0]
        negative = [value for value in pnl_values if value < 0]
        avg_win = mean(positive) if positive else 0.0
        avg_loss = mean(negative) if negative else 0.0
        entry_ev = (p_win * avg_win) + ((1 - p_win) * avg_loss)
        if trades < int(self.config.ev_min_samples):
            confidence = "low"
        elif trades < int(self.config.ev_min_samples) * 2:
            confidence = "medium"
        else:
            confidence = "high"
        return ExpectedValueEstimate(
            strategy_name=strategy_name,
            regime_label=regime_label,
            hour_bucket=hour_bucket,
            entry_grade=entry_grade_math,
            entry_ev=round(entry_ev, 2),
            p_win=round(p_win, 6),
            confidence=confidence,
            closed_trades=trades,
        )

    def _entry_ev_for_context(
        self,
        *,
        strategy_name: str,
        regime_label: str,
        entry_grade_math: str,
    ) -> ExpectedValueEstimate:
        hour_bucket = self._now().strftime("%H")
        family = self._strategy_family_names(strategy_name)
        candidates: List[Tuple[int, ExpectedValueEstimate]] = []
        for family_name in family:
            for (table_strategy, table_regime, table_hour, table_grade), estimate in self._entry_ev_table.items():
                if table_strategy != family_name or table_regime != regime_label:
                    continue
                priority = 100
                if table_grade == entry_grade_math and table_hour == hour_bucket:
                    priority = 0
                elif table_grade == entry_grade_math:
                    priority = 1
                elif table_hour == hour_bucket:
                    priority = 2
                elif table_grade == "unknown":
                    priority = 3
                else:
                    priority = 4
                candidates.append((priority, estimate))
        if candidates:
            candidates.sort(key=lambda item: (item[0], -int(item[1].closed_trades), -float(item[1].entry_ev)))
            best = candidates[0][1]
            wrapped = self._wrap_estimate(strategy_name, regime_label, hour_bucket, entry_grade_math, best)
            if (
                strategy_name in {OPENING_STRATEGY, INTRADAY_STRATEGY}
                and int(wrapped.closed_trades) < int(self.config.ev_min_samples)
            ):
                aggregated = self._aggregate_family_history(
                    family,
                    regime_label,
                    entry_grade_math,
                    strategy_name,
                    hour_bucket,
                )
                if int(aggregated.closed_trades) >= int(wrapped.closed_trades):
                    return aggregated
            return wrapped
        if strategy_name in {OPENING_STRATEGY, INTRADAY_STRATEGY}:
            aggregated = self._aggregate_family_history(
                family,
                regime_label,
                entry_grade_math,
                strategy_name,
                hour_bucket,
            )
            if aggregated.closed_trades > 0:
                return aggregated
        return ExpectedValueEstimate(
            strategy_name=strategy_name,
            regime_label=regime_label,
            hour_bucket=hour_bucket,
            entry_grade=entry_grade_math,
            entry_ev=0.0,
            p_win=0.5,
            confidence="none",
            closed_trades=0,
        )






    @staticmethod
    def _clip_float(value: float, low: float, high: float) -> float:
        return max(low, min(high, float(value)))

    @staticmethod
    def _percentile_value(values: Sequence[float], percentile: float) -> float:
        clean = sorted(float(value) for value in values if value is not None)
        if not clean:
            return 0.0
        if len(clean) == 1:
            return clean[0]
        rank = (len(clean) - 1) * MomentumScalpStrategy._clip_float(percentile, 0.0, 1.0)
        lower = int(rank)
        upper = min(len(clean) - 1, lower + 1)
        weight = rank - lower
        return clean[lower] * (1.0 - weight) + clean[upper] * weight

    def _refresh_adaptive_market_state(
        self,
        quotes: Sequence[Quote],
        leader_signals: Dict[str, LeaderSignal],
        *,
        avg_change: float,
        decliner_ratio: float,
        probs: RegimeProbabilities,
    ) -> None:
        quote_count = max(0, len(quotes))
        if quote_count <= 0:
            return
        advancer_ratio = 1.0 - self._clip_float(decliner_ratio, 0.0, 1.0)
        signals = list(leader_signals.values())
        signal_count = max(1, len(signals))
        leader_density = sum(1 for signal in signals if float(signal.leader_percentile or 0.0) >= 0.80) / signal_count
        elite_density = sum(1 for signal in signals if float(signal.leader_percentile or 0.0) >= 0.95) / signal_count
        vs_open_p90 = self._percentile_value([float(signal.vs_open_pct or 0.0) for signal in signals], 0.90)
        accel_p70 = self._percentile_value([float(signal.recent_acceleration_pct or 0.0) for signal in signals], 0.70)
        shock = float(getattr(self._latest_market_shock_signal, "shock_confidence", 0.0) or 0.0)

        bull_prob = float(probs.bull_prob)
        bear_prob = float(probs.bear_prob)
        avg_up = self._clip_float(float(avg_change) / 3.0, 0.0, 1.0)
        avg_down = self._clip_float(-float(avg_change) / 2.0, 0.0, 1.0)
        heat = self._clip_float(
            0.28 * bull_prob
            + 0.24 * advancer_ratio
            + 0.20 * avg_up
            + 0.16 * leader_density
            + 0.12 * elite_density
            - 0.22 * bear_prob
            - 0.16 * float(decliner_ratio),
            0.0,
            1.0,
        )
        caution = self._clip_float(
            0.34 * bear_prob
            + 0.28 * float(decliner_ratio)
            + 0.20 * shock
            + 0.18 * avg_down,
            0.0,
            1.0,
        )
        overheat = self._clip_float(
            0.42 * self._clip_float(vs_open_p90 / 18.0, 0.0, 1.0)
            + 0.22 * elite_density
            + 0.20 * avg_up
            + 0.16 * self._clip_float(accel_p70 / 0.60, 0.0, 1.0),
            0.0,
            1.0,
        )
        self._adaptive_market_state = {
            "quote_count": float(quote_count),
            "avg_change": round(float(avg_change), 6),
            "decliner_ratio": round(float(decliner_ratio), 6),
            "advancer_ratio": round(advancer_ratio, 6),
            "bull_prob": round(bull_prob, 6),
            "bear_prob": round(bear_prob, 6),
            "leader_density": round(leader_density, 6),
            "elite_leader_density": round(elite_density, 6),
            "vs_open_p90": round(vs_open_p90, 6),
            "accel_p70": round(accel_p70, 6),
            "shock_confidence": round(shock, 6),
            "tape_heat": round(heat, 6),
            "tape_caution": round(caution, 6),
            "overheat": round(overheat, 6),
        }

    def _adaptive_market_entry_thresholds(self) -> Dict[str, float]:
        state = dict(getattr(self, "_adaptive_market_state", {}) or {})
        quote_count = float(state.get("quote_count", 0.0) or 0.0)
        if (
            not bool(getattr(self.config, "enable_adaptive_market_thresholds", True))
            or quote_count < float(getattr(self.config, "adaptive_market_min_quote_count", 8) or 8)
        ):
            return {
                "adaptive_enabled": 0.0,
                "heat": 0.0,
                "caution": 0.0,
                "overheat": 0.0,
                "leader_percentile_delta": 0.0,
                "effective_score_delta": 0.0,
                "recent_accel_delta": 0.0,
                "volume_delta": 0.0,
                "vs_open_ceiling_delta": 0.0,
                "discount_scale": 1.0,
                "rebound_scale": 1.0,
                "negative_ev_floor_scale": 1.0,
                "math_queue_percentile_delta": 0.0,
                "vs_open_p90": 0.0,
            }
        heat = self._clip_float(float(state.get("tape_heat", 0.0) or 0.0), 0.0, 1.0)
        caution = self._clip_float(float(state.get("tape_caution", 0.0) or 0.0), 0.0, 1.0)
        overheat = self._clip_float(float(state.get("overheat", 0.0) or 0.0), 0.0, 1.0)
        hot_overextension = max(0.0, overheat - heat)
        return {
            "adaptive_enabled": 1.0,
            "heat": heat,
            "caution": caution,
            "overheat": overheat,
            "leader_percentile_delta": -0.055 * heat + 0.060 * caution + 0.030 * hot_overextension,
            "effective_score_delta": -0.180 * heat + 0.220 * caution + 0.100 * hot_overextension,
            "recent_accel_delta": -0.040 * heat + 0.075 * caution + 0.030 * hot_overextension,
            "volume_delta": -0.110 * heat + 0.130 * caution,
            "vs_open_ceiling_delta": 4.5 * heat - 3.4 * caution - 1.8 * hot_overextension,
            "discount_scale": self._clip_float(1.0 - 0.34 * heat + 0.48 * caution + 0.25 * hot_overextension, 0.55, 1.80),
            "rebound_scale": self._clip_float(1.0 - 0.26 * heat + 0.36 * caution + 0.16 * hot_overextension, 0.60, 1.65),
            "negative_ev_floor_scale": self._clip_float(1.0 + 0.34 * heat - 0.30 * caution - 0.16 * hot_overextension, 0.65, 1.45),
            "math_queue_percentile_delta": -0.090 * heat + 0.075 * caution + 0.030 * hot_overextension,
            "vs_open_p90": float(state.get("vs_open_p90", 0.0) or 0.0),
        }

    def _adaptive_math_queue_percentile_floor(self, base_floor: float) -> float:
        adaptive = self._adaptive_market_entry_thresholds()
        floor = float(base_floor) + float(adaptive.get("math_queue_percentile_delta", 0.0) or 0.0)
        return round(self._clip_float(floor, 0.62, 0.94), 6)






































    def _conviction_tier(self, estimate: ExpectedValueEstimate) -> str:
        if str(estimate.confidence) in {"none", "low"}:
            return "low_confidence"
        if int(estimate.closed_trades) < int(self.config.ev_min_samples):
            return "low_confidence"
        return "confirmed"












    def _build_entry_metadata(
        self,
        symbol: str,
        setup_name: str,
        payload: str,
        *,
        strategy_name: str,
        quote: Optional[Quote] = None,
    ) -> Dict[str, Any]:
        actual_quote = quote or self._quotes_cache.get(symbol)
        if actual_quote is None:
            raise KeyError(symbol)
        leader = self._leader_signal_for_quote(actual_quote)
        regime_label = self._resolve_regime_profile_name()
        estimate = self._entry_ev_for_context(
            strategy_name=strategy_name,
            regime_label=regime_label,
            entry_grade_math=leader.entry_grade,
        )
        queue_source = self._queue_source_for_symbol(symbol)
        meta = {
            "symbol": symbol,
            "strategy_name": strategy_name,
            "setup_name": setup_name,
            "entry_reason": setup_name,
            "payload": payload,
            "regime_label": regime_label,
            "bear_score": int(self._bear_score),
            "planned_risk_stage": self._current_bull_risk_mode(),
            "entry_grade": leader.entry_grade,
            "entry_grade_math": leader.entry_grade,
            "leader_score": float(leader.leader_score),
            "effective_leader_score": float(leader.effective_leader_score or leader.leader_score),
            "leader_percentile": float(leader.leader_percentile),
            "leader_pct": float(leader.leader_percentile),
            "recent_accel": float(leader.recent_acceleration_pct),
            "vs_open_pct": float(leader.vs_open_pct),
            "high_proximity": float(leader.high_proximity),
            "volume_vs_avg": float(leader.volume_vs_avg),
            "entry_ev": float(estimate.entry_ev),
            "entry_ev_confidence": str(estimate.confidence),
            "entry_ev_conf": str(estimate.confidence),
            "entry_ev_closed_trades": int(estimate.closed_trades),
            "entry_ev_trades": int(estimate.closed_trades),
            "bull_prob": float(self._latest_regime_probabilities.bull_prob),
            "neutral_prob": float(self._latest_regime_probabilities.neutral_prob),
            "soft_bear_prob": float(self._latest_regime_probabilities.soft_bear_prob),
            "bear_prob": float(self._latest_regime_probabilities.bear_prob),
            "candidate_class": "",
            "execution_mode": "rejected",
            "live_route": strategy_name,
            "queue_source": queue_source,
            "size_multiplier": 1.0,
            "conviction_score": 0.0,
            "conviction_rank": 0,
            "conviction_tier": self._conviction_tier(estimate),
            "bull_risk_mode": self._current_bull_risk_mode(),
            "post_loss_admission_class": "general",
            "shock_score": float(self._latest_market_shock_signal.shock_score),
            "shock_confidence": float(self._latest_market_shock_signal.shock_confidence),
            "market_heat": float(self._adaptive_market_state.get("tape_heat", 0.0) or 0.0),
            "market_caution": float(self._adaptive_market_state.get("tape_caution", 0.0) or 0.0),
            "market_overheat": float(self._adaptive_market_state.get("overheat", 0.0) or 0.0),
            "market_avg_change": float(self._adaptive_market_state.get("avg_change", 0.0) or 0.0),
            "market_decliner_ratio": float(self._adaptive_market_state.get("decliner_ratio", 0.5) or 0.5),
            "market_vs_open_p90": float(self._adaptive_market_state.get("vs_open_p90", 0.0) or 0.0),
        }
        meta.update(symbol_micro_edge_metrics(self, actual_quote, leader=leader))
        return meta


    def _long_entry_shortlist(self, incoming_quotes: Sequence[Quote]) -> List[Quote]:
        candidates: List[Quote] = []
        seen = set()
        available_budget = min(
            self._remaining_long_exposure_budget([]),
            self._remaining_long_seed_exposure_budget([]),
        )
        for quote in self._fresh_market_state_quotes(incoming_quotes):
            if (
                quote.symbol in seen
                or quote.symbol in self.positions
                or not self._is_supported_long_symbol(quote.symbol)
                or self._is_symbol_order_unavailable(quote.symbol)
                or not self._long_ev_strategy_name_for_quote(quote)
                or int(quote.current_price or 0) <= 0
                or int(quote.current_price or 0) > available_budget
            ):
                continue
            seen.add(quote.symbol)
            candidates.append(quote)
        return sorted(candidates, key=lambda quote: quote.symbol)


    def _refresh_runtime_math_candidate_queue(self, quotes: Sequence[Quote]):
        quotes = [
            quote
            for quote in quotes
            if self._is_supported_long_symbol(quote.symbol)
        ]
        previous_signature = (
            tuple(self._latest_math_queue_symbols),
            tuple(self._latest_math_backfill_symbols),
            tuple(sorted(self._latest_opening_fast_symbols)),
            tuple(sorted(self._latest_opening_hot_symbols)),
            tuple(sorted(self._latest_math_queue_source.items())),
        )
        if not quotes:
            self._latest_math_queue_symbols = []
            self._latest_math_backfill_symbols = []
            if previous_signature != (
                tuple(self._latest_math_queue_symbols),
                tuple(self._latest_math_backfill_symbols),
                tuple(sorted(self._latest_opening_fast_symbols)),
                tuple(sorted(self._latest_opening_hot_symbols)),
                tuple(sorted(self._latest_math_queue_source.items())),
            ):
                self._save_daily_state_if_due(min_interval_seconds=20)
            return

        new_signals = build_leader_signals(
            quotes,
            avg_volumes=self._avg_volumes,
            recent_quotes_by_symbol=self._recent_quotes,
            regime_score=int(self._bear_score),
        )
        self._latest_math_leader_signals.update(new_signals)

        math_queue_floor = self._adaptive_math_queue_percentile_floor(float(self.config.math_queue_percentile_floor))
        ranked = sorted(
            (
                (signal.effective_leader_score or signal.leader_score, signal.leader_percentile, symbol)
                for symbol, signal in new_signals.items()
                if signal.leader_percentile >= math_queue_floor
            ),
            reverse=True,
        )
        top_n = max(1, int(self.config.math_queue_top_n))
        backfill_n = max(0, int(self.config.math_queue_backfill_slots))
        new_queue = [symbol for _, _, symbol in ranked[:top_n]]
        new_backfill = [symbol for _, _, symbol in ranked[top_n : top_n + backfill_n]]

        opening_fast: set[str] = set()
        opening_hot: set[str] = set()
        if self._opening_fast_window_active():
            fast_ranked = sorted(
                (
                    (float(signal.effective_leader_score or signal.leader_score), symbol)
                    for symbol, signal in new_signals.items()
                ),
                reverse=True,
            )
            opening_fast = {symbol for _, symbol in fast_ranked[: max(1, int(self.config.opening_fast_live_top_n))]}
        if self._opening_candidate_window_active():
            opening_hot_floor = self._adaptive_math_queue_percentile_floor(float(self.config.opening_hot_percentile_floor))
            hot_ranked = sorted(
                (
                    (float(signal.leader_percentile), float(signal.effective_leader_score or signal.leader_score), symbol)
                    for symbol, signal in new_signals.items()
                    if float(signal.leader_percentile) >= opening_hot_floor
                ),
                reverse=True,
            )
            opening_hot = {symbol for _, _, symbol in hot_ranked[: max(1, int(self.config.opening_hot_top_n))]}

        preserve_opening_sources = self._opening_candidate_window_active()
        preserved = {
            symbol: source
            for symbol, source in self._latest_math_queue_source.items()
            if preserve_opening_sources
            and symbol in new_signals
            and source in {"opening_fast_queue", "opening_hot_queue"}
        }
        updated_source = dict(preserved)
        for symbol in new_queue:
            updated_source[symbol] = "math_queue"
        for symbol in new_backfill:
            updated_source[symbol] = "math_backfill"
        for symbol in opening_hot:
            updated_source[symbol] = "opening_hot_queue"
        for symbol in opening_fast:
            updated_source[symbol] = "opening_fast_queue"

        self._latest_math_queue_symbols = new_queue
        self._latest_math_backfill_symbols = new_backfill
        self._latest_opening_fast_symbols = opening_fast
        self._latest_opening_hot_symbols = opening_hot
        self._latest_math_queue_source = updated_source
        current_signature = (
            tuple(self._latest_math_queue_symbols),
            tuple(self._latest_math_backfill_symbols),
            tuple(sorted(self._latest_opening_fast_symbols)),
            tuple(sorted(self._latest_opening_hot_symbols)),
            tuple(sorted(self._latest_math_queue_source.items())),
        )
        if current_signature != previous_signature:
            self._save_daily_state_if_due(min_interval_seconds=20)


    def _long_ev_strategy_name_for_quote(self, quote: Quote) -> str:
        if not self._is_supported_long_symbol(quote.symbol):
            return ""
        source = self._queue_source_for_symbol(quote.symbol)
        if self._opening_conviction_window_active() and source in OPENING_LONG_QUEUE_SOURCES:
            return OPENING_STRATEGY
        if self._intraday_conviction_window_active() and (
            source in LIVE_LONG_QUEUE_SOURCES or source in INTRADAY_BASE_QUEUE_SOURCES
        ):
            return INTRADAY_STRATEGY
        return ""

    # ------------------------------------------------------------------
    # 진입 허용 / 수량
    # ------------------------------------------------------------------
    def _long_ev_precheck_reject_reason(
        self,
        quote: Quote,
        *,
        pending_orders: Sequence[Order],
        strategy_name_override: str = "",
        entry_meta: Optional[Dict[str, Any]] = None,
        skip_capacity: bool = False,
    ) -> str:
        _ = entry_meta
        if not self._is_supported_long_symbol(quote.symbol):
            return "unsupported_long_symbol"
        strategy_name = strategy_name_override or self._long_ev_strategy_name_for_quote(quote)
        if not strategy_name:
            return "ev_route_disabled"
        if self._halted:
            return "daily_halt"
        if self._is_symbol_order_unavailable(quote.symbol):
            return "symbol_order_unavailable"
        if self._symbol_entry_cooldown_remaining(quote.symbol) > 0:
            return "symbol_recent_loss_cooldown"
        if self._client_rate_limit_cooldown() > 0:
            return "api_rate_limit_cooldown"
        quote_symbol = str(quote.symbol or "").strip()
        if any(order.side == OrderSide.BUY and str(order.symbol or "").strip() == quote_symbol for order in pending_orders):
            return "duplicate_pending_order"
        if quote_symbol in self._pending_entry_meta and quote_symbol not in self.positions:
            return "duplicate_pending_order"
        if self._has_unresolved_pending_long_entry(pending_orders, exclude_symbol=quote_symbol):
            return "pending_long_entry_unresolved"
        open_longs = len(self._long_position_symbols())
        pending_longs = sum(
            1
            for order in pending_orders
            if order.side == OrderSide.BUY
        )
        if not skip_capacity and open_longs + pending_longs >= self._max_long_positions_allowed():
            return "long_precheck_capacity"
        if not skip_capacity and self._remaining_long_exposure_budget(pending_orders) <= 0:
            return "long_total_exposure_cap"
        return ""





    def _long_total_exposure_cap(self) -> int:
        profile = self._resolve_regime_profile_name()
        pct = {
            "bull": self.config.bull_capital_utilization_pct,
            "neutral": self.config.neutral_capital_utilization_pct,
            "soft_bear": self.config.soft_bear_capital_utilization_pct,
            "bear": self.config.bear_capital_utilization_pct,
        }.get(profile)
        if pct is None:
            pct = self.config.capital_utilization_pct
        return max(0, int(int(self.config.seed_money) * float(pct or 0.0)))

    def _current_long_exposure_amount(self) -> int:
        total = 0
        for symbol, pos in self.positions.items():
            invested = int(getattr(pos, "invested_amount", 0) or 0)
            if invested <= 0:
                invested = int(getattr(pos, "buy_price", 0) or 0) * int(getattr(pos, "quantity", 0) or 0)
            total += max(0, invested)
        return total

    @staticmethod
    def _coerce_pending_meta_int(meta: Dict[str, Any], *keys: str) -> int:
        for key in keys:
            value = meta.get(key)
            if value is None or value == "":
                continue
            try:
                coerced = int(round(float(value)))
            except (TypeError, ValueError):
                continue
            if coerced > 0:
                return coerced
        return 0

    def _pending_long_entry_meta_items(self, pending_orders: Sequence[Order]) -> List[Tuple[str, Dict[str, Any]]]:
        pending_order_symbols = {
            str(order.symbol or "").strip()
            for order in pending_orders
            if order.side == OrderSide.BUY and str(order.symbol or "").strip()
        }
        items: List[Tuple[str, Dict[str, Any]]] = []
        for raw_symbol, raw_meta in (self._pending_entry_meta or {}).items():
            symbol = str(raw_symbol or "").strip()
            if (
                not symbol
                or symbol in self.positions
                or symbol in pending_order_symbols
            ):
                continue
            if not isinstance(raw_meta, dict):
                continue
            items.append((symbol, raw_meta))
        return items

    def _has_unresolved_pending_long_entry(self, pending_orders: Sequence[Order], *, exclude_symbol: str = "") -> bool:
        excluded = str(exclude_symbol or "").strip()
        for order in pending_orders:
            symbol = str(order.symbol or "").strip()
            if (
                order.side == OrderSide.BUY
                and symbol
                and symbol != excluded
            ):
                return True
        for raw_symbol, position in (self.positions or {}).items():
            symbol = str(raw_symbol or getattr(position, "symbol", "") or "").strip()
            if symbol and symbol != excluded and getattr(position, "pending_entry_started_at", None) is not None:
                return True
        for symbol, _meta in self._pending_long_entry_meta_items(pending_orders):
            if symbol != excluded:
                return True
        return False

    def _pending_long_exposure_amount(self, pending_orders: Sequence[Order]) -> int:
        total = 0
        for order in pending_orders:
            if order.side != OrderSide.BUY:
                continue
            quote = self._quotes_cache.get(order.symbol)
            price = int(order.price or getattr(order, "reference_price", 0) or 0)
            if price <= 0 and quote is not None:
                price = int(quote.current_price or 0)
            total += max(0, price) * max(0, int(order.quantity or 0))
        for symbol, meta in self._pending_long_entry_meta_items(pending_orders):
            qty = self._coerce_pending_meta_int(
                meta,
                "pending_order_quantity",
                "entry_ev_live_model_quantity",
            )
            if qty <= 0:
                continue
            price = self._coerce_pending_meta_int(
                meta,
                "pending_order_reference_price",
                "entry_signal_price",
                "price_prediction_signal_price",
            )
            if price <= 0:
                quote = self._quotes_cache.get(symbol)
                if quote is not None:
                    price = int(getattr(quote, "current_price", 0) or 0)
            total += max(0, price) * qty
        return total

    def _remaining_long_exposure_budget(self, pending_orders: Sequence[Order]) -> int:
        cap = self._long_total_exposure_cap()
        used = self._current_long_exposure_amount() + self._pending_long_exposure_amount(pending_orders)
        return max(0, cap - used)

    def _remaining_long_seed_exposure_budget(self, pending_orders: Sequence[Order]) -> int:
        cap = max(0, int(getattr(self.config, "seed_money", 0) or 0))
        used = self._current_long_exposure_amount() + self._pending_long_exposure_amount(pending_orders)
        return max(0, cap - used)




    def _daily_target_remaining_net(self) -> int:
        target = max(0, int(getattr(self.config, "daily_profit_target", 0) or 0))
        if target <= 0:
            return 0
        realized = int(self._realized_net_pnl_for_daily_breaker())
        return max(0, target - realized)




    def _target_net_for_take_profit_pct(
        self,
        quote: Quote,
        *,
        quantity: int,
        take_profit_pct: float,
        entry_slippage_rate: Optional[float] = None,
    ) -> int:
        qty = max(0, int(quantity or 0))
        if qty <= 0:
            return 0
        current_price = max(1, int(quote.current_price or 0))
        entry_rate = (
            float(self.config.entry_market_slippage_rate)
            if entry_slippage_rate is None
            else max(0.0, float(entry_slippage_rate))
        )
        entry_price = max(
            1,
            int(round(current_price * (1.0 + entry_rate))),
        )
        projected_exit_price = max(
            1,
            int(round(entry_price * (1.0 + (float(take_profit_pct) / 100.0) - float(self.config.exit_market_slippage_rate)))),
        )
        return self._estimate_trade_net_pnl_from_prices(
            entry_price=entry_price,
            exit_price=projected_exit_price,
            quantity=qty,
        )


    def _take_profit_pct_for_net_target(
        self,
        quote: Quote,
        *,
        quantity: int,
        target_net: int,
        return_hint_pct: float = 0.0,
        entry_slippage_rate: Optional[float] = None,
    ) -> float:
        qty = max(0, int(quantity or 0))
        target = max(1, int(target_net or 0))
        if qty <= 0:
            return 0.0
        low = 0.01
        high = max(low, float(return_hint_pct or 0.0), float(self.config.take_profit_pct or 0.0))
        for _ in range(10):
            if (
                self._target_net_for_take_profit_pct(
                    quote,
                    quantity=qty,
                    take_profit_pct=high,
                    entry_slippage_rate=entry_slippage_rate,
                )
                >= target
            ):
                break
            high *= 1.5
            if high >= 8.0:
                high = 8.0
                break
        if (
            self._target_net_for_take_profit_pct(
                quote,
                quantity=qty,
                take_profit_pct=high,
                entry_slippage_rate=entry_slippage_rate,
            )
            < target
        ):
            return 0.0
        for _ in range(24):
            mid = (low + high) / 2.0
            if (
                self._target_net_for_take_profit_pct(
                    quote,
                    quantity=qty,
                    take_profit_pct=mid,
                    entry_slippage_rate=entry_slippage_rate,
                )
                >= target
            ):
                high = mid
            else:
                low = mid
        return round(high, 4)

    def _net_for_return_pct(
        self,
        quote: Quote,
        *,
        quantity: int,
        return_pct: float,
        entry_slippage_rate: Optional[float] = None,
    ) -> int:
        qty = max(0, int(quantity or 0))
        if qty <= 0:
            return 0
        current_price = max(1, int(quote.current_price or 0))
        entry_rate = (
            float(self.config.entry_market_slippage_rate)
            if entry_slippage_rate is None
            else max(0.0, float(entry_slippage_rate))
        )
        entry_price = max(
            1,
            int(round(current_price * (1.0 + entry_rate))),
        )
        projected_exit_price = max(
            1,
            int(round(current_price * (1.0 + (float(return_pct) / 100.0) - float(self.config.exit_market_slippage_rate)))),
        )
        return self._estimate_trade_net_pnl_from_prices(
            entry_price=entry_price,
            exit_price=projected_exit_price,
            quantity=qty,
        )

    def _entry_execution_slippage_rate(
        self,
        prediction: Optional[ShortHorizonPrediction],
        entry_meta: Optional[Dict[str, Any]] = None,
    ) -> float:
        _ = prediction, entry_meta
        return max(0.0, float(self.config.entry_market_slippage_rate))

    def _round_trip_execution_cost_pct(self) -> float:
        return 100.0 * (
            2.0 * float(self.config.commission_rate)
            + float(self.config.tax_slippage_rate)
            + float(self.config.entry_market_slippage_rate)
            + float(self.config.exit_market_slippage_rate)
        )

    def _pending_long_planned_loss_risk(self, pending_orders: Sequence[Order]) -> int:
        total = 0
        for order in pending_orders:
            if order.side != OrderSide.BUY:
                continue
            meta = self._pending_entry_meta.get(order.symbol, {})
            planned = int(meta.get("planned_risk_net_loss_abs", meta.get("planned_stop_net_loss_abs", 0)) or 0)
            if planned > 0:
                total += planned
                continue
            qty = max(0, int(order.quantity or 0))
            if qty <= 0:
                continue
            quote = self._quotes_cache.get(order.symbol)
            price = int(order.price or getattr(order, "reference_price", 0) or 0)
            if price <= 0 and quote is not None:
                price = int(getattr(quote, "current_price", 0) or 0)
            if price <= 0:
                continue
            entry_price = max(1, int(round(price * (1.0 + float(self.config.entry_market_slippage_rate)))))
            fallback = min(
                int(self.config.long_stop_loss_cap_amount),
                int(entry_price * qty * float(self.config.long_stop_loss_notional_pct)),
            )
            total += max(0, fallback)
        for symbol, meta in self._pending_long_entry_meta_items(pending_orders):
            planned = self._coerce_pending_meta_int(
                meta,
                "planned_risk_net_loss_abs",
                "planned_stop_net_loss_abs",
            )
            if planned > 0:
                total += planned
                continue
            qty = self._coerce_pending_meta_int(
                meta,
                "pending_order_quantity",
                "entry_ev_live_model_quantity",
            )
            price = self._coerce_pending_meta_int(
                meta,
                "pending_order_reference_price",
                "entry_signal_price",
                "price_prediction_signal_price",
            )
            if price <= 0:
                quote = self._quotes_cache.get(symbol)
                if quote is not None:
                    price = int(getattr(quote, "current_price", 0) or 0)
            if qty <= 0 or price <= 0:
                continue
            entry_price = max(1, int(round(price * (1.0 + float(self.config.entry_market_slippage_rate)))))
            fallback = min(
                int(self.config.long_stop_loss_cap_amount),
                int(entry_price * qty * float(self.config.long_stop_loss_notional_pct)),
            )
            total += max(0, fallback)
        return max(0, total)

    def _open_long_planned_loss_risk(self) -> int:
        total = 0
        for raw_symbol, position in list((self.positions or {}).items()):
            symbol = str(raw_symbol or getattr(position, "symbol", "") or "").strip()
            if not symbol:
                continue
            qty = max(0, int(getattr(position, "quantity", 0) or 0))
            if qty <= 0:
                continue
            planned = int(getattr(position, "planned_risk_net_loss_abs", 0) or 0)
            if planned <= 0:
                planned = int(getattr(position, "planned_stop_net_loss_abs", 0) or 0)
            if planned <= 0:
                invested = int(getattr(position, "invested_amount", 0) or 0)
                if invested <= 0:
                    invested = int(getattr(position, "buy_price", 0) or 0) * qty
                if invested <= 0:
                    continue
                planned = min(
                    int(self.config.long_stop_loss_cap_amount),
                    int(invested * float(self.config.long_stop_loss_notional_pct)),
                )
            total += max(0, planned)
        return max(0, total)

    def _walk_forward_calibrated_prediction(
        self,
        prediction: ShortHorizonPrediction,
        *,
        strategy_name: str,
    ) -> Tuple[ShortHorizonPrediction, float]:
        confidence = self._clip_float(
            float(getattr(prediction, "confidence", 0.0) or 0.0),
            0.0,
            1.0,
        )
        direction = self._clip_float(
            float(getattr(prediction, "direction_score", 0.0) or 0.0),
            0.0,
            1.0,
        )
        raw_win_probability = self._clip_float(
            0.50 + (direction - 0.50) * confidence,
            0.0,
            1.0,
        )
        raw_return = float(getattr(prediction, "predicted_return_pct", 0.0) or 0.0)
        calibration = self._forecast_outcomes.calibrate(
            as_of=self._now(),
            raw_win_probability=raw_win_probability,
            raw_return_pct=raw_return,
            horizon_seconds=int(prediction.horizon_seconds),
            round_trip_cost_pct=self._round_trip_execution_cost_pct(),
            strategy_name=strategy_name,
        )
        shift = float(calibration.return_shift_pct)
        features = dict(getattr(prediction, "features", {}) or {})
        features.update(
            {
                "raw_predicted_return_pct": raw_return,
                "raw_lower_bound_return_pct": float(prediction.lower_bound_return_pct),
                "raw_upper_bound_return_pct": float(prediction.upper_bound_return_pct),
                "raw_win_probability": raw_win_probability,
                "calibrated_win_probability": float(calibration.calibrated_win_probability),
                "walk_forward_sample_count": float(calibration.sample_count),
                "walk_forward_effective_sample_size": float(calibration.effective_sample_size),
                "walk_forward_return_shift_pct": shift,
            }
        )
        calibrated_prediction = replace(
            prediction,
            predicted_return_pct=self._clip_float(
                float(calibration.calibrated_return_pct),
                -5.0,
                5.0,
            ),
            lower_bound_return_pct=self._clip_float(
                float(prediction.lower_bound_return_pct) + shift,
                -5.0,
                5.0,
            ),
            upper_bound_return_pct=self._clip_float(
                float(prediction.upper_bound_return_pct) + shift,
                -5.0,
                5.0,
            ),
            features=features,
        )
        return calibrated_prediction, float(calibration.calibrated_win_probability)

    def _build_expected_value_trade_plan(
        self,
        quote: Quote,
        entry_meta: Dict[str, Any],
        *,
        pending_orders: Sequence[Order],
    ) -> ExpectedValueTradePlan:
        current_price = max(0, int(quote.current_price or 0))
        if current_price <= 0:
            return ExpectedValueTradePlan(False, "ev_invalid_price")

        prediction = self._price_prediction_for_entry(
            quote,
            entry_meta,
            min_samples=max(3, int(getattr(self.config, "price_prediction_min_samples", 5) or 5)),
        )
        if not bool(getattr(prediction, "ready", False)):
            return ExpectedValueTradePlan(False, "ev_prediction_not_ready", prediction=prediction)
        prediction, win_probability = self._walk_forward_calibrated_prediction(
            prediction,
            strategy_name=str(entry_meta.get("strategy_name") or ""),
        )

        total_budget = min(
            self._remaining_long_exposure_budget(pending_orders),
            self._remaining_long_seed_exposure_budget(pending_orders),
        )
        if total_budget < current_price:
            return ExpectedValueTradePlan(
                False,
                "ev_budget_too_small",
                budget=max(0, int(total_budget)),
                win_probability=float(win_probability),
                prediction=prediction,
            )
        max_quantity = max(0, int(total_budget // current_price))
        if max_quantity <= 0:
            return ExpectedValueTradePlan(
                False,
                "ev_quantity_zero",
                budget=max(0, int(total_budget)),
                win_probability=float(win_probability),
                prediction=prediction,
            )

        pending_long_risk = self._pending_long_planned_loss_risk(pending_orders)
        open_long_risk = self._open_long_planned_loss_risk()
        committed_long_risk = int(pending_long_risk) + int(open_long_risk)
        if open_long_risk > 0:
            entry_meta["open_long_planned_loss_risk"] = int(open_long_risk)
        if committed_long_risk > 0:
            entry_meta["committed_long_planned_loss_risk"] = int(committed_long_risk)

        remaining_loss_room = max(0, int(self._daily_loss_room()) - committed_long_risk)
        execution_reserve = max(0, int(getattr(self.config, "daily_loss_near_stop_buffer", 0) or 0))
        usable_loss_room = max(0, remaining_loss_room - execution_reserve)
        if usable_loss_room <= 0:
            return ExpectedValueTradePlan(
                False,
                "ev_daily_loss_room_exhausted",
                budget=max(0, int(total_budget)),
                win_probability=float(win_probability),
                prediction=prediction,
            )

        predicted_return = float(getattr(prediction, "predicted_return_pct", 0.0) or 0.0)
        if predicted_return <= 0.0:
            return ExpectedValueTradePlan(
                False,
                "ev_prediction_non_positive",
                budget=max(0, int(total_budget)),
                win_probability=float(win_probability),
                prediction=prediction,
            )

        lower_return = float(getattr(prediction, "lower_bound_return_pct", 0.0) or 0.0)
        upper_return = float(getattr(prediction, "upper_bound_return_pct", 0.0) or 0.0)
        entry_slippage_rate = self._entry_execution_slippage_rate(prediction)
        entry_price = max(1, int(round(current_price * (1.0 + entry_slippage_rate))))
        predicted_exit_price = max(
            1,
            int(
                round(
                    current_price
                    * (
                        1.0
                        + (predicted_return / 100.0)
                        - float(self.config.exit_market_slippage_rate)
                    )
                )
            ),
        )
        lower_exit_price = max(
            1,
            int(
                round(
                    current_price
                    * (
                        1.0
                        + (lower_return / 100.0)
                        - float(self.config.exit_market_slippage_rate)
                    )
                )
            ),
        )
        zero_return_exit_price = max(
            1,
            int(
                round(
                    current_price
                    * (1.0 - float(self.config.exit_market_slippage_rate))
                )
            ),
        )
        remaining_target = max(0, int(self._daily_target_remaining_net()))
        target_room = remaining_target if remaining_target > 0 else max(1, int(total_budget))

        best: Optional[ExpectedValueTradePlan] = None
        best_key: Tuple[float, float, int, int] = (-float("inf"), -float("inf"), 0, 0)
        best_rejected: Optional[ExpectedValueTradePlan] = None
        best_rejected_key: Tuple[float, int, int] = (-float("inf"), 0, 0)

        def remember_rejected(
            detail: str,
            *,
            quantity: int,
            predicted_net: int = 0,
            lower_net: int = 0,
            upper_net: int = 0,
            expected_net: float = 0.0,
            break_even_probability: float = 1.0,
            planned_target: int = 0,
            planned_stop: int = 0,
            planned_risk: int = 0,
        ) -> None:
            nonlocal best_rejected, best_rejected_key
            key = (float(expected_net), int(predicted_net), -int(planned_risk))
            if key <= best_rejected_key:
                return
            best_rejected_key = key
            quantity = max(0, int(quantity))
            best_rejected = ExpectedValueTradePlan(
                allowed=False,
                reject_reason="ev_no_positive_quantity",
                quantity=quantity,
                budget=min(int(total_budget), quantity * current_price),
                expected_net=float(expected_net),
                predicted_net=int(predicted_net),
                lower_net=int(lower_net),
                upper_net=int(upper_net),
                win_probability=float(win_probability),
                break_even_probability=float(break_even_probability),
                planned_target_net=int(planned_target),
                planned_stop_net_loss_abs=int(planned_stop),
                planned_risk_net_loss_abs=int(planned_risk),
                prediction=prediction,
                reject_detail=str(detail),
            )

        for quantity in range(1, max_quantity + 1):
            predicted_net = self._net_for_return_pct(
                quote,
                quantity=quantity,
                return_pct=predicted_return,
                entry_slippage_rate=entry_slippage_rate,
            )
            if predicted_net <= 0:
                remember_rejected(
                    "predicted_net_non_positive",
                    quantity=quantity,
                    predicted_net=int(predicted_net),
                )
                continue

            predicted_net_unrounded = estimate_trade_net_pnl_unrounded(
                entry_price=entry_price,
                exit_price=predicted_exit_price,
                quantity=quantity,
                commission_rate=float(self.config.commission_rate),
                tax_slippage_rate=float(self.config.tax_slippage_rate),
            )
            if predicted_net_unrounded <= 0.0:
                remember_rejected(
                    "predicted_net_non_positive",
                    quantity=quantity,
                    predicted_net=int(predicted_net),
                )
                continue

            lower_net = self._net_for_return_pct(
                quote,
                quantity=quantity,
                return_pct=lower_return,
                entry_slippage_rate=entry_slippage_rate,
            )
            upper_net = self._net_for_return_pct(
                quote,
                quantity=quantity,
                return_pct=upper_return,
                entry_slippage_rate=entry_slippage_rate,
            )
            zero_return_net = self._net_for_return_pct(
                quote,
                quantity=quantity,
                return_pct=0.0,
                entry_slippage_rate=entry_slippage_rate,
            )
            lower_net_unrounded = estimate_trade_net_pnl_unrounded(
                entry_price=entry_price,
                exit_price=lower_exit_price,
                quantity=quantity,
                commission_rate=float(self.config.commission_rate),
                tax_slippage_rate=float(self.config.tax_slippage_rate),
            )
            zero_return_net_unrounded = estimate_trade_net_pnl_unrounded(
                entry_price=entry_price,
                exit_price=zero_return_exit_price,
                quantity=quantity,
                commission_rate=float(self.config.commission_rate),
                tax_slippage_rate=float(self.config.tax_slippage_rate),
            )
            cost_floor = abs(min(0, int(zero_return_net))) + 1
            cost_floor_unrounded = abs(min(0.0, float(zero_return_net_unrounded)))
            configured_stop = max(
                1,
                min(
                    int(self.config.long_stop_loss_cap_amount),
                    int(entry_price * quantity * float(self.config.long_stop_loss_notional_pct)),
                ),
            )
            model_loss = abs(min(0, int(lower_net)))
            configured_stop_unrounded = max(
                1.0,
                min(
                    float(self.config.long_stop_loss_cap_amount),
                    float(entry_price * quantity)
                    * float(self.config.long_stop_loss_notional_pct),
                ),
            )
            model_loss_unrounded = abs(min(0.0, float(lower_net_unrounded)))
            planned_stop = max(
                cost_floor,
                min(configured_stop, model_loss if model_loss > 0 else configured_stop),
            )
            planned_stop_unrounded = max(
                cost_floor_unrounded,
                min(
                    configured_stop_unrounded,
                    model_loss_unrounded if model_loss_unrounded > 0.0 else configured_stop_unrounded,
                ),
            )
            execution_buffer = max(
                1,
                int(
                    round(
                        float(cost_floor) * 0.10
                        + float(entry_price * quantity)
                        * max(0.0, float(self.config.exit_market_slippage_rate))
                        * 0.25
                    )
                ),
            )
            execution_buffer_unrounded = max(
                1.0,
                cost_floor_unrounded * 0.10
                + float(entry_price * quantity)
                * max(0.0, float(self.config.exit_market_slippage_rate))
                * 0.25,
            )
            planned_risk = max(planned_stop, model_loss) + execution_buffer
            planned_risk_unrounded = (
                max(planned_stop_unrounded, model_loss_unrounded)
                + execution_buffer_unrounded
            )
            planned_stop = max(planned_stop, int(ceil(planned_stop_unrounded)))
            execution_buffer = max(execution_buffer, int(ceil(execution_buffer_unrounded)))
            planned_risk = max(
                max(planned_stop, model_loss) + execution_buffer,
                planned_risk,
                int(ceil(planned_risk_unrounded)),
            )
            planned_target = max(1, min(int(predicted_net), int(target_room)))
            break_even_probability = float(planned_risk_unrounded) / max(
                1.0,
                float(predicted_net_unrounded + planned_risk_unrounded),
            )
            # predicted_return_pct is the calibrated expected return, not a
            # conditional win payoff. Multiplying its net value by the win
            # probability again would discount the same uncertainty twice.
            expected_net = float(predicted_net_unrounded)

            if planned_risk > usable_loss_room:
                remember_rejected(
                    f"risk_room_exceeded:need={planned_risk} room={usable_loss_room}",
                    quantity=quantity,
                    predicted_net=int(predicted_net),
                    lower_net=int(lower_net),
                    upper_net=int(upper_net),
                    expected_net=float(expected_net),
                    break_even_probability=float(break_even_probability),
                    planned_target=int(planned_target),
                    planned_stop=int(planned_stop),
                    planned_risk=int(planned_risk),
                )
                continue
            if expected_net <= 0.0:
                remember_rejected(
                    f"expected_net_non_positive:be={break_even_probability:.3f}",
                    quantity=quantity,
                    predicted_net=int(predicted_net),
                    lower_net=int(lower_net),
                    upper_net=int(upper_net),
                    expected_net=float(expected_net),
                    break_even_probability=float(break_even_probability),
                    planned_target=int(planned_target),
                    planned_stop=int(planned_stop),
                    planned_risk=int(planned_risk),
                )
                continue

            planned_take_profit = self._take_profit_pct_for_net_target(
                quote,
                quantity=quantity,
                target_net=planned_target,
                return_hint_pct=max(predicted_return, float(self.config.take_profit_pct or 0.0)),
                entry_slippage_rate=entry_slippage_rate,
            )
            if planned_take_profit <= 0.0:
                planned_take_profit = max(0.01, predicted_return)
            planned_stop_loss_pct = (
                float(planned_stop) / max(1.0, float(entry_price * quantity))
            ) * 100.0
            plan = ExpectedValueTradePlan(
                allowed=True,
                quantity=quantity,
                budget=min(int(total_budget), quantity * current_price),
                expected_net=float(expected_net),
                predicted_net=int(predicted_net),
                lower_net=int(lower_net),
                upper_net=int(upper_net),
                win_probability=float(win_probability),
                break_even_probability=float(break_even_probability),
                planned_target_net=int(planned_target),
                planned_stop_net_loss_abs=int(planned_stop),
                planned_risk_net_loss_abs=int(planned_risk),
                planned_take_profit_pct=float(planned_take_profit),
                planned_stop_loss_pct=float(planned_stop_loss_pct),
                prediction=prediction,
            )
            key = (
                float(plan.expected_net),
                float(plan.expected_net) / max(1.0, float(plan.planned_risk_net_loss_abs)),
                int(plan.predicted_net),
                int(plan.quantity),
            )
            if key > best_key:
                best_key = key
                best = plan

        if best is not None:
            return best
        if best_rejected is not None:
            entry_meta["ev_reject_detail"] = str(best_rejected.reject_detail)
            entry_meta["ev_reject_best_quantity"] = int(best_rejected.quantity)
            entry_meta["ev_reject_best_expected_net"] = round(float(best_rejected.expected_net), 2)
            return best_rejected
        return ExpectedValueTradePlan(
            False,
            "ev_no_positive_quantity",
            budget=max(0, int(total_budget)),
            win_probability=float(win_probability),
            prediction=prediction,
        )
    def _apply_expected_value_trade_plan_metadata(
        self,
        entry_meta: Dict[str, Any],
        plan: ExpectedValueTradePlan,
    ) -> None:
        prediction = plan.prediction
        entry_meta.update(
            {
                "execution_mode": "live",
                "entry_reason": "expected_value",
                "entry_ev_decision": 1.0,
                "historical_entry_ev": float(entry_meta.get("entry_ev", 0.0) or 0.0),
                "historical_entry_ev_confidence": str(entry_meta.get("entry_ev_confidence", "")),
                "historical_entry_ev_closed_trades": int(entry_meta.get("entry_ev_closed_trades", 0) or 0),
                "entry_ev": round(float(plan.expected_net), 2),
                "entry_ev_confidence": "live_plan",
                "entry_expected_net_pnl": round(float(plan.expected_net), 2),
                "entry_prediction_net_pnl": int(plan.predicted_net),
                "entry_prediction_lower_net_pnl": int(plan.lower_net),
                "entry_prediction_win_probability": round(float(plan.win_probability), 6),
                "entry_prediction_break_even_probability": round(float(plan.break_even_probability), 6),
                "planned_target_net_pnl": int(plan.planned_target_net),
                "planned_stop_net_loss_abs": int(plan.planned_stop_net_loss_abs),
                "planned_risk_net_loss_abs": int(plan.planned_risk_net_loss_abs),
                "planned_take_profit_pct": float(plan.planned_take_profit_pct),
                "planned_stop_loss_pct": float(plan.planned_stop_loss_pct),
                "adaptive_take_profit_pct": float(plan.planned_take_profit_pct),
                "adaptive_stop_loss_pct": -abs(float(plan.planned_stop_loss_pct)),
                "price_prediction_net_pnl": int(plan.predicted_net),
                "price_prediction_lower_net_pnl": int(plan.lower_net),
                "price_prediction_upper_net_pnl": int(plan.upper_net),
            }
        )
        if prediction is None:
            return
        entry_meta.update(
            {
                "price_prediction_ready": 1.0 if bool(prediction.ready) else 0.0,
                "price_prediction_reason": str(prediction.reason),
                "price_prediction_horizon_seconds": int(prediction.horizon_seconds),
                "price_prediction_sample_count": int(prediction.sample_count),
                "price_prediction_return_pct": float(prediction.predicted_return_pct),
                "price_prediction_lower_pct": float(prediction.lower_bound_return_pct),
                "price_prediction_upper_pct": float(prediction.upper_bound_return_pct),
                "price_prediction_confidence": float(prediction.confidence),
                "price_prediction_direction_score": float(prediction.direction_score),
                "price_prediction_volatility_pct": float(prediction.volatility_pct),
            }
        )
        for key, value in dict(getattr(prediction, "features", {}) or {}).items():
            if isinstance(value, (int, float)):
                entry_meta[f"price_prediction_{key}"] = float(value)











    def _estimate_trade_net_pnl_from_prices(
        self,
        *,
        entry_price: int,
        exit_price: int,
        quantity: int,
    ) -> int:
        return estimate_trade_net_pnl_from_prices(
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            commission_rate=float(self.config.commission_rate),
            tax_slippage_rate=float(self.config.tax_slippage_rate),
        )


    def _price_prediction_for_entry(
        self,
        quote: Quote,
        entry_meta: Dict[str, Any],
        *,
        horizon_seconds: Optional[int] = None,
        min_samples: Optional[int] = None,
    ) -> ShortHorizonPrediction:
        leader = self._leader_signal_for_quote(quote)
        if entry_meta:
            leader = LeaderSignal(
                symbol=str(getattr(quote, "symbol", "") or ""),
                leader_score=float(entry_meta.get("leader_score", leader.leader_score) or 0.0),
                leader_percentile=float(entry_meta.get("leader_percentile", leader.leader_percentile) or 0.0),
                entry_grade=str(entry_meta.get("entry_grade", leader.entry_grade) or "C"),
                change_rate=float(getattr(quote, "change_rate", 0.0) or 0.0),
                trade_amount=int(getattr(quote, "trade_amount", 0) or 0),
                vs_open_pct=float(entry_meta.get("vs_open_pct", leader.vs_open_pct) or 0.0),
                high_proximity=float(entry_meta.get("high_proximity", leader.high_proximity) or 0.0),
                volume_vs_avg=float(entry_meta.get("volume_vs_avg", leader.volume_vs_avg) or 0.0),
                reclaim_speed_ticks=int(getattr(leader, "reclaim_speed_ticks", 99) or 99),
                recent_acceleration_pct=float(entry_meta.get("recent_accel", leader.recent_acceleration_pct) or 0.0),
                effective_leader_score=float(
                    entry_meta.get("effective_leader_score", leader.effective_leader_score or leader.leader_score) or 0.0
                ),
            )
        return predict_short_horizon_return(
            quote,
            recent_quotes=list(self._recent_quotes.get(quote.symbol, [])),
            leader=leader,
            market_state=dict(getattr(self, "_adaptive_market_state", {}) or {}),
            horizon_seconds=int(
                horizon_seconds
                if horizon_seconds is not None
                else getattr(self.config, "price_prediction_horizon_seconds", 180) or 180
            ),
            min_samples=int(
                min_samples
                if min_samples is not None
                else getattr(self.config, "price_prediction_min_samples", 5) or 5
            ),
        )
















    # ------------------------------------------------------------------
    # 실전 주문 평가
    # ------------------------------------------------------------------



    def _log_ev_reject(
        self,
        quote: Quote,
        strategy_name: str,
        reason: str,
        meta: Dict[str, Any],
        plan: Optional[ExpectedValueTradePlan] = None,
    ) -> None:
        prediction = plan.prediction if plan is not None else None
        reject_detail = ""
        if plan is not None:
            raw_detail = str(getattr(plan, "reject_detail", "") or "").strip()
            if raw_detail:
                reject_detail = f" detail={raw_detail[:160]}"
        logger.info(
            "EV 진입 거부[%s]: %s route=%s source=%s rank=%s score=%.4f "
            "price=%d vs_open=%.2f hp=%.3f pred=%.3f lower=%.3f conf=%.3f "
            "win=%.3f exp=%.1f pnet=%s lnet=%s qty=%s target=%s stop=%s risk=%s model=%s%s",
            reason,
            quote.symbol,
            strategy_name,
            meta.get("queue_source", ""),
            int(meta.get("conviction_rank", 0) or 0),
            float(meta.get("conviction_score", 0.0) or 0.0),
            int(quote.current_price or 0),
            float(meta.get("vs_open_pct", 0.0) or 0.0),
            float(meta.get("high_proximity", 0.0) or 0.0),
            float(getattr(prediction, "predicted_return_pct", 0.0) or 0.0),
            float(getattr(prediction, "lower_bound_return_pct", 0.0) or 0.0),
            float(getattr(prediction, "confidence", 0.0) or 0.0),
            float(getattr(plan, "win_probability", 0.0) or 0.0),
            float(getattr(plan, "expected_net", 0.0) or 0.0),
            int(getattr(plan, "predicted_net", 0) or 0) if plan is not None else "-",
            int(getattr(plan, "lower_net", 0) or 0) if plan is not None else "-",
            int(getattr(plan, "quantity", 0) or 0) if plan is not None else "-",
            int(getattr(plan, "planned_target_net", 0) or 0) if plan is not None else "-",
            int(getattr(plan, "planned_stop_net_loss_abs", 0) or 0) if plan is not None else "-",
            int(getattr(plan, "planned_risk_net_loss_abs", 0) or 0) if plan is not None else "-",
            str(getattr(prediction, "reason", "-") or "-"),
            reject_detail,
        )

    def _log_ev_buy_signal(self, quote: Quote, plan: ExpectedValueTradePlan, meta: Dict[str, Any]) -> None:
        prediction = plan.prediction
        logger.info(
            "EV 매수 신호: %s %d주 @ %d원 budget=%s route=%s source=%s rank=%s score=%.4f "
            "pred=%.3f lower=%.3f conf=%.3f win=%.3f exp=%.1f pnet=%d lnet=%d target=%d stop=%d risk=%d model=%s",
            quote.symbol,
            int(plan.quantity),
            int(quote.current_price or 0),
            f"{int(plan.budget):,}",
            meta.get("strategy_name", ""),
            meta.get("queue_source", ""),
            int(meta.get("conviction_rank", 0) or 0),
            float(meta.get("conviction_score", 0.0) or 0.0),
            float(getattr(prediction, "predicted_return_pct", 0.0) or 0.0),
            float(getattr(prediction, "lower_bound_return_pct", 0.0) or 0.0),
            float(getattr(prediction, "confidence", 0.0) or 0.0),
            float(plan.win_probability),
            float(plan.expected_net),
            int(plan.predicted_net),
            int(plan.lower_net),
            int(plan.planned_target_net),
            int(plan.planned_stop_net_loss_abs),
            int(plan.planned_risk_net_loss_abs),
            str(getattr(prediction, "reason", "-") or "-"),
        )

    def _build_expected_value_candidate(
        self,
        quote: Quote,
        *,
        pending_orders: Sequence[Order],
    ) -> Optional[ExpectedValueCandidate]:
        strategy_name = self._long_ev_strategy_name_for_quote(quote)
        if not strategy_name:
            return None
        leader_signal = self._leader_signal_for_quote(quote)
        meta = self._build_entry_metadata(
            quote.symbol,
            "expected_value",
            "entry_reason=expected_value",
            strategy_name=strategy_name,
            quote=quote,
        )
        meta.update(
            {
                "candidate_class": (
                    "opening_conviction"
                    if strategy_name == OPENING_STRATEGY
                    else "intraday_conviction"
                ),
                "execution_mode": "evaluated",
                "live_route": strategy_name,
                "entry_style": "expected_value",
                "conviction_score": float(
                    leader_signal.effective_leader_score or leader_signal.leader_score
                ),
                "recent_accel": float(leader_signal.recent_acceleration_pct),
            }
        )
        meta["setup_name"] = "expected_value"
        meta["entry_reason"] = "expected_value"
        meta["payload"] = "entry_reason=expected_value"
        meta["size_multiplier"] = 1.0
        plan = self._build_expected_value_trade_plan(quote, meta, pending_orders=pending_orders)
        if plan.allowed:
            operational_reject_reason = self._long_ev_precheck_reject_reason(
                quote,
                pending_orders=pending_orders,
                strategy_name_override=strategy_name,
                entry_meta=meta,
            )
            if operational_reject_reason:
                plan = replace(
                    plan,
                    allowed=False,
                    reject_reason=operational_reject_reason,
                    reject_detail="operational_constraint",
                )
        return ExpectedValueCandidate(
            quote=quote,
            strategy_name=strategy_name,
            metadata=meta,
            plan=plan,
        )

    @staticmethod
    def _expected_value_candidate_sort_key(
        candidate: ExpectedValueCandidate,
    ) -> Tuple[float, float, int, str]:
        plan = candidate.plan
        risk_adjusted_ev = float(plan.expected_net) / max(
            1.0,
            float(plan.planned_risk_net_loss_abs),
        )
        return (
            -float(plan.expected_net),
            -risk_adjusted_ev,
            -int(plan.predicted_net),
            candidate.quote.symbol,
        )

    def _rank_expected_value_candidates(
        self,
        candidates: Sequence[ExpectedValueCandidate],
    ) -> List[ExpectedValueCandidate]:
        ranked = sorted(candidates, key=self._expected_value_candidate_sort_key)
        for rank, candidate in enumerate(ranked, start=1):
            candidate.metadata["conviction_rank"] = rank
            candidate.metadata["ev_rank"] = rank
        return ranked

    def _record_expected_value_forecast(
        self,
        candidate: ExpectedValueCandidate,
        *,
        selected: bool,
    ) -> None:
        prediction = candidate.plan.prediction
        if prediction is None or not bool(prediction.ready):
            return
        features = dict(getattr(prediction, "features", {}) or {})
        signal_at = getattr(candidate.quote, "timestamp", None) or self._now()
        round_trip_cost_pct = self._round_trip_execution_cost_pct()
        payload = {
            "symbol": candidate.quote.symbol,
            "signal_timestamp": signal_at.isoformat(timespec="seconds"),
            "signal_price": int(candidate.quote.current_price or 0),
            "horizon_seconds": int(prediction.horizon_seconds),
            "round_trip_cost_pct": round(round_trip_cost_pct, 6),
            "strategy_name": candidate.strategy_name,
            "queue_source": str(candidate.metadata.get("queue_source", "") or ""),
            "rank": int(candidate.metadata.get("ev_rank", 0) or 0),
            "allowed": bool(candidate.plan.allowed),
            "reject_reason": str(candidate.plan.reject_reason or ""),
            "quantity": int(candidate.plan.quantity),
            "expected_net": round(float(candidate.plan.expected_net), 6),
            "predicted_net": int(candidate.plan.predicted_net),
            "planned_risk_net_loss_abs": int(candidate.plan.planned_risk_net_loss_abs),
            "raw_predicted_return_pct": float(
                features.get("raw_predicted_return_pct", prediction.predicted_return_pct)
            ),
            "calibrated_predicted_return_pct": float(prediction.predicted_return_pct),
            "lower_bound_return_pct": float(prediction.lower_bound_return_pct),
            "upper_bound_return_pct": float(prediction.upper_bound_return_pct),
            "confidence": float(prediction.confidence),
            "raw_win_probability": float(
                features.get("raw_win_probability", candidate.plan.win_probability)
            ),
            "calibrated_win_probability": float(candidate.plan.win_probability),
            "walk_forward_sample_count": int(
                features.get("walk_forward_sample_count", 0) or 0
            ),
            "walk_forward_effective_sample_size": float(
                features.get("walk_forward_effective_sample_size", 0.0) or 0.0
            ),
            "raw_lower_bound_return_pct": float(
                features.get("raw_lower_bound_return_pct", prediction.lower_bound_return_pct)
            ),
            "raw_upper_bound_return_pct": float(
                features.get("raw_upper_bound_return_pct", prediction.upper_bound_return_pct)
            ),
            "prediction_features": {
                key: float(features[key])
                for key in (
                    "short_return_pct",
                    "full_return_pct",
                    "positive_move_support",
                    "tail_positive_support",
                    "single_tick_impulse_risk",
                    "late_extension_risk",
                    "flow_confirmation",
                    "quote_gap_risk",
                    "rejection_risk_score",
                    "deceleration_pressure",
                    "confirmed_high_hold_continuation_score",
                    "confirmed_opening_launch_score",
                    "directional_evidence_pct",
                    "directional_reversal_penalty_pct",
                )
                if key in features
            },
        }
        try:
            self._forecast_outcomes.record(payload, now=self._now(), selected=selected)
        except OSError as exc:
            logger.warning("예측 결과 원장 기록 실패: %s", exc)

    def _order_from_expected_value_candidate(
        self,
        candidate: ExpectedValueCandidate,
    ) -> Order:
        quote = candidate.quote
        plan = candidate.plan
        meta = candidate.metadata
        self._apply_expected_value_trade_plan_metadata(meta, plan)

        quantity = int(plan.quantity)
        budget = int(plan.budget)
        meta["entry_signal_price"] = int(quote.current_price or 0)
        meta["entry_signal_timestamp"] = self._now().isoformat(timespec="seconds")
        meta["pending_order_quantity"] = quantity
        meta["pending_order_reference_price"] = int(quote.current_price or 0)
        meta["pending_order_budget"] = budget
        meta["pending_order_created_at"] = meta["entry_signal_timestamp"]
        self._pending_entry_meta[quote.symbol] = meta
        self._log_ev_buy_signal(quote, plan, meta)
        return Order(
            symbol=quote.symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=0,
            reference_price=max(0, int(quote.current_price or 0)),
            requested_reason="expected_value",
        )


    # ------------------------------------------------------------------
    # 청산
    # ------------------------------------------------------------------
    def _estimated_exit_net_pnl(
        self,
        pos: PositionState,
        quote: Quote,
        *,
        quantity: Optional[int] = None,
    ) -> int:
        sell_qty = max(0, min(int(quantity or pos.quantity or 0), int(pos.quantity or 0)))
        if sell_qty <= 0:
            return 0
        current_price = max(0, int(getattr(quote, "current_price", 0) or 0))
        buy_price = max(0, int(getattr(pos, "buy_price", 0) or 0))
        return estimate_trade_net_pnl_from_prices(
            entry_price=buy_price,
            exit_price=current_price,
            quantity=sell_qty,
            commission_rate=float(self.config.commission_rate),
            tax_slippage_rate=float(self.config.tax_slippage_rate),
        )

    def _paper_mode_active(self) -> bool:
        client = getattr(self.market_data, "client", None)
        cfg = getattr(client, "config", None)
        return bool(getattr(cfg, "is_paper", False))

    def _make_sell_order(
        self,
        pos: PositionState,
        quantity: int,
        *,
        reason: str,
        reference_price: int = 0,
    ) -> Optional[Order]:
        requested_quantity = max(0, min(int(quantity or 0), int(pos.quantity or 0)))
        reserved_quantity = 0
        if self._position_pending_exit_reconcile_pending(pos):
            reserved_quantity = min(
                int(pos.quantity or 0),
                max(0, int(getattr(pos, "pending_exit_quantity", 0) or 0)),
            )
        sell_quantity = min(requested_quantity, max(0, int(pos.quantity or 0) - reserved_quantity))
        if sell_quantity <= 0:
            return None
        stop_cap_amount = int(self.config.long_stop_loss_cap_amount)
        protective_exit_mode = "" if self._paper_mode_active() else "limit_then_market"
        protective_limit_price = 0
        protective_fallback_polls = 0
        if protective_exit_mode == "limit_then_market":
            protective_limit_price = max(1, int(pos.high_since_buy or pos.buy_price))
            protective_fallback_polls = max(1, int(self.config.protective_stop_fallback_ticks))
        order_reference_price = max(0, int(reference_price or 0))
        if order_reference_price <= 0:
            quote = self._quotes_cache.get(pos.symbol)
            if quote is None:
                recent_quotes = list(self._recent_quotes.get(pos.symbol, []))
                quote = recent_quotes[-1] if recent_quotes else None
            if quote is not None:
                order_reference_price = max(0, int(getattr(quote, "current_price", 0) or 0))
        if order_reference_price <= 0:
            order_reference_price = max(0, int(pos.buy_price or 0))
        return Order(
            symbol=pos.symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=sell_quantity,
            price=0,
            reference_price=order_reference_price,
            protective_exit_mode=protective_exit_mode,
            protective_limit_price=protective_limit_price,
            protective_fallback_polls=protective_fallback_polls,
            stop_reference_amount_krw=max(0, stop_cap_amount),
            requested_reason=reason,
        )





    def _paper_exit_grace_active(self, pos: PositionState) -> bool:
        client = getattr(self.market_data, "client", None)
        cfg = getattr(client, "config", None)
        if not bool(getattr(cfg, "is_paper", False)):
            return False
        grace = max(0, int(self.config.paper_position_exit_grace_seconds))
        if grace <= 0:
            return False
        return (self._now() - pos.buy_time).total_seconds() < grace

    def _resolve_fill_price_with_quote_fallback(
        self,
        symbol: str,
        *,
        broker_price: int,
        requested_price: int = 0,
        reference_price: int = 0,
        position: Optional[PositionState] = None,
    ) -> int:
        price = max(0, int(broker_price or 0))
        if price > 0:
            return price

        requested = max(0, int(requested_price or 0))
        if requested > 0:
            logger.debug("체결가가 비어 있어 요청가로 대체합니다: %s @ %d원", symbol, requested)
            return requested

        reference = max(0, int(reference_price or 0))
        if reference > 0:
            logger.debug("체결가가 비어 있어 주문 기준가로 대체합니다: %s @ %d원", symbol, reference)
            return reference

        cached_quote = self._quotes_cache.get(symbol)
        cached_quote_price = max(0, int(getattr(cached_quote, "current_price", 0) or 0))
        if cached_quote_price > 0:
            logger.debug("체결가가 비어 있어 최근 캐시 시세로 대체합니다: %s @ %d원", symbol, cached_quote_price)
            return cached_quote_price

        recent_quotes = list(self._recent_quotes.get(symbol, []))
        if recent_quotes:
            fallback_quote_price = max(0, int(getattr(recent_quotes[-1], "current_price", 0) or 0))
            if fallback_quote_price > 0:
                logger.debug("체결가가 비어 있어 최근 시세로 대체합니다: %s @ %d원", symbol, fallback_quote_price)
                return fallback_quote_price

        if position is not None:
            position_price = max(0, int(getattr(position, "buy_price", 0) or 0))
            if position_price > 0:
                logger.debug("체결가가 비어 있어 보유 평균단가로 대체합니다: %s @ %d원", symbol, position_price)
                return position_price

        return 0

    def _default_long_exit(self, quote: Quote) -> Optional[Order]:
        pos = self.positions.get(quote.symbol)
        if pos is None:
            return None
        pos.high_since_buy = max(int(pos.high_since_buy or 0), int(quote.current_price or 0))
        estimated_net_pnl = self._estimated_exit_net_pnl(pos, quote)
        planned_stop = int(getattr(pos, "planned_stop_net_loss_abs", 0) or 0)
        if planned_stop > 0 and estimated_net_pnl <= -planned_stop:
            return self._make_sell_order(
                pos,
                int(pos.quantity or 0),
                reason="ev_planned_stop_net",
                reference_price=int(quote.current_price or 0),
            )
        if getattr(pos, "pending_entry_started_at", None) is not None:
            return None
        if self._paper_exit_grace_active(pos):
            return None
        if pos.is_restored and pos.restored_at is not None:
            grace = int(self.config.restored_position_grace_seconds)
            if (self._now() - pos.restored_at).total_seconds() < grace:
                return None
        held_seconds = max(0.0, (self._now() - pos.buy_time).total_seconds())
        held_minutes = held_seconds / 60.0
        gain_pct = ((int(quote.current_price or 0) - int(pos.buy_price or 0)) / max(1, int(pos.buy_price or 0))) * 100.0
        trail_drawdown = (
            ((int(quote.current_price or 0) - int(pos.high_since_buy or pos.buy_price)) / max(1, int(pos.high_since_buy or pos.buy_price))) * 100.0
        )
        stop_amount = min(
            int(self.config.long_stop_loss_cap_amount),
            int(pos.invested_amount * float(self.config.long_stop_loss_notional_pct)),
        )
        net_stop_amount = max(1, stop_amount)
        planned_target = int(getattr(pos, "planned_target_net_pnl", 0) or 0)
        if planned_target > 0 and estimated_net_pnl >= planned_target:
            return self._make_sell_order(
                pos,
                int(pos.quantity or 0),
                reason="ev_planned_target_net",
                reference_price=int(quote.current_price or 0),
            )
        take_profit = float(pos.adaptive_take_profit_pct or self.config.take_profit_pct)
        adaptive_target = float(pos.adaptive_take_profit_pct or 0.0) > 0.0
        hold_profile = long_exit_hold_profile(
            self,
            pos,
            quote,
            estimated_net_pnl=estimated_net_pnl,
            gain_pct=gain_pct,
            take_profit_pct=take_profit,
        )
        max_hold_minutes = max(1.0, float(hold_profile.get("max_hold_minutes", self.config.max_position_holding_minutes)))
        min_trailing_seconds = max(0.0, float(hold_profile.get("min_trailing_seconds", 0.0)))
        trailing_activation = float(pos.adaptive_trailing_activation_pct or self.config.trailing_stop_activation_gain_pct)
        trailing_stop = -abs(float(pos.adaptive_trailing_stop_pct or self.config.trailing_stop_pct))
        exit_decision = decide_long_exit(
            LongExitSnapshot(
                quantity=int(pos.quantity or 0),
                held_minutes=held_minutes,
                held_seconds=held_seconds,
                gain_pct=gain_pct,
                trail_drawdown_pct=trail_drawdown,
                unrealized_pnl=int((int(quote.current_price or 0) - int(pos.buy_price or 0)) * int(pos.quantity or 0)),
                estimated_net_pnl=estimated_net_pnl,
                net_stop_amount=net_stop_amount,
                take_profit_pct=take_profit,
                adaptive_target=adaptive_target,
                max_hold_minutes=max_hold_minutes,
                min_trailing_seconds=min_trailing_seconds,
                trailing_activation_pct=trailing_activation,
                trailing_stop_pct=trailing_stop,
                adaptive_stop_loss_pct=float(getattr(pos, "adaptive_stop_loss_pct", 0.0) or 0.0),
            ),
            partial_exit_done=bool(pos.partial_exit_done),
            partial_exit_ratio=float(self.config.bull_partial_exit_ratio),
        )
        if not exit_decision.should_exit:
            return None
        return self._make_sell_order(
            pos,
            exit_decision.quantity,
            reason=exit_decision.reason,
            reference_price=int(quote.current_price or 0),
        )

    def _liquidate_all(self) -> List[Order]:
        orders: List[Order] = []
        for pos in list(self.positions.values()):
            order = self._make_sell_order(pos, pos.quantity, reason="liquidate_all")
            if order is not None:
                orders.append(order)
        return orders

    def _fresh_market_state_quotes(self, incoming_quotes: Sequence[Quote], *, max_age_seconds: float = 75.0) -> List[Quote]:
        ordered_symbols: List[str] = [quote.symbol for quote in incoming_quotes if getattr(quote, "symbol", "")]
        ordered_symbols.extend(self._latest_math_queue_symbols)
        ordered_symbols.extend(self._latest_math_backfill_symbols)
        ordered_symbols.extend(sorted(self._latest_opening_fast_symbols))
        ordered_symbols.extend(sorted(self._latest_opening_hot_symbols))
        ordered_symbols.extend(self._pool)
        ordered_symbols.extend(self.get_watchlist())
        ordered_symbols.extend(list(self.positions.keys()))

        now = self._now()
        seen = set()
        fresh_quotes: List[Quote] = []
        for symbol in ordered_symbols:
            normalized = str(symbol or "").strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            quote = self._quotes_cache.get(normalized)
            if quote is None:
                continue
            quote_ts = getattr(quote, "timestamp", None)
            if quote_ts is not None:
                age_seconds = max(0.0, (now - quote_ts).total_seconds())
                if age_seconds > max_age_seconds:
                    continue
            fresh_quotes.append(quote)
        return fresh_quotes

    # ------------------------------------------------------------------
    # 이벤트 루프
    # ------------------------------------------------------------------
    def on_tick(self, quote: Quote) -> List[Order]:
        return self.on_batch_tick([quote])

    def _update_market_state(self, quotes: Sequence[Quote]):
        if not quotes:
            return
        market_quotes = self._fresh_market_state_quotes(quotes)
        if not market_quotes:
            return
        long_universe = [
            quote
            for quote in market_quotes
            if self._is_supported_long_symbol(quote.symbol)
        ]
        if not long_universe:
            return
        market_basis = [quote for quote in long_universe if quote.symbol not in self._inverse_symbols] or long_universe
        avg_change = mean(float(quote.change_rate or 0.0) for quote in market_basis)
        decliner_ratio = sum(1 for quote in market_basis if float(quote.change_rate or 0.0) < 0.0) / max(1, len(market_basis))

        leader_signals = build_leader_signals(
            long_universe,
            avg_volumes=self._avg_volumes,
            recent_quotes_by_symbol=self._recent_quotes,
            regime_score=int(self._bear_score),
        )
        self._latest_math_leader_signals.update(leader_signals)
        market_basis_signals = {
            quote.symbol: leader_signals[quote.symbol]
            for quote in market_basis
            if quote.symbol in leader_signals
        }
        strong_leaders = [
            signal
            for signal in market_basis_signals.values()
            if float(signal.effective_leader_score or signal.leader_score) >= float(self.config.strong_leader_min_change_rate) / 4.0
        ]
        strong_leader_avg = mean(
            float(signal.effective_leader_score or signal.leader_score) for signal in strong_leaders
        ) if strong_leaders else 0.0

        probs = compute_regime_probabilities(
            index_gap_ma20_pct=avg_change,
            index_gap_ma5_pct=avg_change * 0.6,
            avg_change=avg_change,
            decliner_ratio=decliner_ratio,
            strong_leader_count=len(strong_leaders),
            strong_leader_avg_score=strong_leader_avg,
        )
        self._latest_regime_probabilities = probs
        if probs.bear_prob >= max(probs.bull_prob, probs.neutral_prob, probs.soft_bear_prob):
            self._bear_score = 3
        elif probs.soft_bear_prob >= max(probs.bull_prob, probs.neutral_prob):
            self._bear_score = 2
        elif probs.bull_prob >= 0.45:
            self._bear_score = 0
        else:
            self._bear_score = 1
        self._strong_bull_override_active = probs.bull_prob >= 0.60
        if self._resolve_regime_profile_name() == "bull":
            self._bull_market_context = "fragile_bull" if (self._bear_score > 0 or probs.bull_prob < 0.70) else "broad_bull"
        else:
            self._bull_market_context = self._resolve_regime_profile_name()

        inverse_quotes = [quote for quote in market_quotes if quote.symbol in self._inverse_symbols]
        inverse_leader_count = sum(1 for quote in inverse_quotes if float(quote.change_rate or 0.0) > 0.0)
        self._latest_market_shock_signal = compute_market_shock_signal(
            minutes_since_open=self._minutes_since_market_open(),
            crash_window_minutes=int(self.config.market_shock_window_minutes_after_open),
            index_gap_open_pct=avg_change,
            index_gap_ma5_pct=avg_change * 0.7,
            index_gap_ma20_pct=avg_change * 1.1,
            avg_change=avg_change,
            decliner_ratio=decliner_ratio,
            falling_speed_pct=max(0.0, -avg_change),
            inverse_leader_count=inverse_leader_count,
        )
        self._refresh_adaptive_market_state(
            market_basis,
            market_basis_signals,
            avg_change=avg_change,
            decliner_ratio=decliner_ratio,
            probs=probs,
        )

        self._refresh_runtime_math_candidate_queue(long_universe)

    def on_batch_tick(self, quotes: List[Quote]) -> List[Order]:
        if not quotes:
            return []
        now = max((quote.timestamp for quote in quotes), default=self._now())
        self.set_simulated_now(now)
        if self._active_day != self._today() or self._session_start_at is None:
            self.initialize()

        for quote in quotes:
            self._quotes_cache[quote.symbol] = quote
            window = self._ensure_recent_quote_window(quote.symbol)
            window.append(quote)

        try:
            settled_forecasts = self._forecast_outcomes.settle(quotes, now=self._now())
        except OSError as exc:
            settled_forecasts = []
            logger.warning("예측 결과 원장 정산 실패: %s", exc)
        if settled_forecasts:
            profitable_count = sum(
                1 for item in settled_forecasts if bool(item.get("profitable"))
            )
            logger.info(
                "180초 예측 정산: total=%d profitable=%d",
                len(settled_forecasts),
                profitable_count,
            )

        self._update_market_state(quotes)
        self._update_daily_breakers()
        self._last_long_shortlist_symbols = []
        if self._halted:
            return self._liquidate_all()

        orders: List[Order] = []
        entry_quotes = self._fresh_market_state_quotes(quotes)
        quote_by_symbol = {quote.symbol: quote for quote in quotes}

        # 보유 포지션 청산 우선
        for symbol, pos in list(self.positions.items()):
            quote = quote_by_symbol.get(symbol)
            if quote is None:
                continue
            exit_order = self._regime_router.evaluate_long_exit(self, quote)
            if exit_order is not None:
                orders.append(exit_order)
        if orders:
            return orders

        if self._profit_halt_confirmation_pending():
            return []

        long_shortlist = self._long_entry_shortlist(entry_quotes)
        self._last_long_shortlist_symbols = [quote.symbol for quote in long_shortlist]
        candidates = [
            candidate
            for candidate in (
                self._build_expected_value_candidate(quote, pending_orders=[])
                for quote in long_shortlist
            )
            if candidate is not None
        ]
        ranked = self._rank_expected_value_candidates(candidates)
        selected = next(
            (candidate for candidate in ranked if candidate.plan.allowed),
            None,
        )
        for candidate in ranked:
            self._record_expected_value_forecast(
                candidate,
                selected=candidate is selected,
            )
        if selected is None:
            if ranked:
                top = ranked[0]
                self._log_ev_reject(
                    top.quote,
                    top.strategy_name,
                    top.plan.reject_reason,
                    top.metadata,
                    top.plan,
                )
            return []

        logger.info(
            "EV 후보 선택: symbol=%s evaluated=%d viable=%d rank=%d exp=%.1f risk=%d",
            selected.quote.symbol,
            len(ranked),
            sum(1 for candidate in ranked if candidate.plan.allowed),
            int(selected.metadata.get("ev_rank", 0) or 0),
            float(selected.plan.expected_net),
            int(selected.plan.planned_risk_net_loss_abs),
        )
        return [self._order_from_expected_value_candidate(selected)]

    # ------------------------------------------------------------------
    # 체결 처리
    # ------------------------------------------------------------------
    def _position_counts_for_daily_breaker(self, pos: Optional[PositionState]) -> bool:
        if pos is None:
            return True
        if bool(self.config.use_restored_pnl_for_daily_breaker):
            return True
        if bool(getattr(pos, "is_restored", False)):
            return False
        if str(getattr(pos, "entry_setup_name", "") or "") == "restored_position":
            return False
        if str(getattr(pos, "queue_source", "") or "") == "account_restore":
            return False
        return True

    def _position_realized_counts_for_daily_breaker(self, pos: Optional[PositionState]) -> bool:
        if pos is None:
            return True
        return True

    def _position_counts_for_strategy_stats(self, pos: Optional[PositionState]) -> bool:
        return self._position_counts_for_daily_breaker(pos)

    def _normalize_sell_fill_breaker_flags(self) -> None:
        for entry in self._sell_fill_ledger or []:
            if not isinstance(entry, dict):
                continue
            if not bool(entry.get("counts_for_daily_breaker", True)):
                entry["counts_for_daily_breaker"] = True
                entry["daily_breaker_flag_migrated"] = "realized_sell_fill"

    def _realized_net_pnl_for_daily_breaker(self) -> int:
        realized = int(self.daily_pnl.realized_net_pnl)
        if not bool(self.config.use_restored_pnl_for_daily_breaker):
            realized -= int(self._breaker_excluded_realized_net_pnl)
        return realized

    def _daily_loss_floor(self) -> int:
        return int(
            self.config.daily_total_loss_limit
            if self.config.daily_total_loss_limit is not None
            else self.config.daily_loss_limit
        )

    def _daily_loss_room(self, realized_net_pnl: Optional[int] = None) -> int:
        realized = self._realized_net_pnl_for_daily_breaker() if realized_net_pnl is None else int(realized_net_pnl)
        return max(0, realized - self._daily_loss_floor())

    def _total_net_pnl_for_daily_breaker(self, realized_net_pnl: Optional[int] = None) -> int:
        total = self._realized_net_pnl_for_daily_breaker() if realized_net_pnl is None else int(realized_net_pnl)
        if bool(self.config.enable_unrealized_loss_guard):
            total += self._unrealized_net_pnl_for_daily_breaker()
        return int(total)

    def _restored_loss_halt_is_stale(self) -> bool:
        reason = str(getattr(self, "_halt_reason", "") or "")
        if not self._halted or reason not in {"daily_loss_limit", "daily_loss_near_limit", "daily_total_loss_limit"}:
            return False
        realized_net_pnl = self._realized_net_pnl_for_daily_breaker()
        if realized_net_pnl <= int(self.config.daily_loss_limit):
            return False

        near_stop_buffer = max(0, int(getattr(self.config, "daily_loss_near_stop_buffer", 0) or 0))
        near_stop_floor = int(self.config.daily_loss_limit) + near_stop_buffer
        if (
            reason == "daily_loss_near_limit"
            and near_stop_buffer > 0
            and realized_net_pnl < 0
            and realized_net_pnl <= near_stop_floor
            and not self.positions
        ):
            return False

        total_limit = self.config.daily_total_loss_limit
        if total_limit is not None and self._total_net_pnl_for_daily_breaker(realized_net_pnl) <= int(total_limit):
            return False
        return True


    def _daily_profit_lock_reached(self, realized_net_pnl: Optional[int] = None) -> bool:
        target = max(0, int(self.config.daily_profit_target))
        if target <= 0:
            return False
        realized = self._realized_net_pnl_for_daily_breaker() if realized_net_pnl is None else int(realized_net_pnl)
        protect_threshold = max(0, int(self.config.profit_protect_threshold))
        lock_buffer = max(0, int(getattr(self.config, "daily_profit_lock_buffer", 0) or 0))
        if lock_buffer <= 0:
            return False
        lock_threshold = max(protect_threshold, target - lock_buffer)
        return realized >= lock_threshold

    def _has_unconfirmed_daily_breaker_sell_fills(self) -> bool:
        for entry in self._sell_fill_ledger:
            if not bool(entry.get("price_estimated")):
                continue
            if bool(entry.get("counts_for_daily_breaker", True)):
                return True
        return False


    def _profit_halt_confirmation_pending(self, realized_net_pnl: Optional[int] = None) -> bool:
        if not self._has_unconfirmed_daily_breaker_sell_fills():
            return False
        realized = self._realized_net_pnl_for_daily_breaker() if realized_net_pnl is None else int(realized_net_pnl)
        if realized >= int(self.config.daily_profit_target):
            return True
        return self._daily_profit_lock_reached(realized)





    def _unrealized_net_pnl_for_daily_breaker(self) -> int:
        total = 0
        for pos in self.positions.values():
            if not self._position_counts_for_daily_breaker(pos):
                continue
            quantity = max(0, int(pos.quantity or 0))
            if self._position_pending_exit_reconcile_pending(pos):
                quantity -= min(
                    quantity,
                    max(0, int(getattr(pos, "pending_exit_quantity", 0) or 0)),
                )
            if quantity <= 0:
                continue
            quote = self._quotes_cache.get(pos.symbol)
            if quote is None:
                recent_quotes = list(self._recent_quotes.get(pos.symbol, []))
                quote = recent_quotes[-1] if recent_quotes else None
            exit_price = int(getattr(quote, "current_price", 0) or 0) if quote is not None else 0
            if exit_price <= 0:
                continue
            pnl = calculate_trade_pnl_from_prices(
                entry_price=int(pos.buy_price or 0),
                exit_price=int(exit_price),
                quantity=quantity,
                commission_rate=float(self.config.commission_rate),
                tax_slippage_rate=float(self.config.tax_slippage_rate),
            )
            total += int(pnl.net_pnl)
        return int(total)

    def _update_daily_breakers(self):
        was_halted = bool(self._halted)
        realized_net_pnl = self._realized_net_pnl_for_daily_breaker()
        breaker_net_pnl = realized_net_pnl
        halt_reason = ""
        if not self.positions and self._restored_loss_halt_is_stale():
            previous_reason = str(getattr(self, "_halt_reason", "") or "")
            self._halted = False
            self._halt_reason = ""
            was_halted = False
            logger.warning(
                "확정 체결손익이 손실한도 안으로 복구되어 당일 하드스탑을 해제합니다: "
                "previous_reason=%s realized_net_pnl=%d",
                previous_reason,
                realized_net_pnl,
            )
        if (
            self._halted
            and str(getattr(self, "_halt_reason", "") or "") in {"daily_profit_target", "daily_profit_lock"}
            and realized_net_pnl < int(self.config.daily_profit_target)
            and not self._daily_profit_lock_reached(realized_net_pnl)
        ):
            self._halted = False
            self._halt_reason = ""
            was_halted = False
        if realized_net_pnl <= int(self.config.daily_loss_limit):
            self._halted = True
            halt_reason = "daily_loss_limit"
        near_stop_buffer = max(0, int(getattr(self.config, "daily_loss_near_stop_buffer", 0) or 0))
        near_stop_floor = int(self.config.daily_loss_limit) + near_stop_buffer
        if (
            not self._halted
            and near_stop_buffer > 0
            and realized_net_pnl < 0
            and realized_net_pnl <= near_stop_floor
            and not self.positions
        ):
            self._halted = True
            halt_reason = "daily_loss_near_limit"
        total_limit = self.config.daily_total_loss_limit
        if total_limit is not None:
            total_net_pnl = self._total_net_pnl_for_daily_breaker(realized_net_pnl)
            if total_net_pnl <= int(total_limit):
                self._halted = True
                halt_reason = "daily_total_loss_limit"
                breaker_net_pnl = int(total_net_pnl)
        profit_confirmation_pending = self._profit_halt_confirmation_pending(realized_net_pnl)
        if (
            not profit_confirmation_pending
            and realized_net_pnl >= int(self.config.daily_profit_target)
        ):
            self._halted = True
            halt_reason = "daily_profit_target"
        elif not profit_confirmation_pending and self._daily_profit_lock_reached(realized_net_pnl):
            self._halted = True
            halt_reason = "daily_profit_lock"
        if self._halted and halt_reason:
            self._halt_reason = halt_reason
        if self._halted and not was_halted:
            if halt_reason == "daily_profit_target":
                logger.warning("일일 총손익 목표 달성! realized_net_pnl=%d", realized_net_pnl)
            elif halt_reason == "daily_profit_lock":
                logger.warning("일일 수익 보호 잠금 도달: realized_net_pnl=%d", realized_net_pnl)
            elif halt_reason == "daily_loss_near_limit":
                logger.warning(
                    "일일 손실한도 근접으로 신규거래를 종료합니다: realized_net_pnl=%d limit=%d buffer=%d",
                    realized_net_pnl,
                    int(self.config.daily_loss_limit),
                    near_stop_buffer,
                )
            elif halt_reason in {"daily_loss_limit", "daily_total_loss_limit"}:
                logger.warning(
                    "일일 총손익 하드스탑 도달! realized_net_pnl=%d breaker_net_pnl=%d reason=%s",
                    realized_net_pnl,
                    breaker_net_pnl,
                    halt_reason,
                )
            self._save_daily_state_if_due(force=True)

    @staticmethod
    def _is_pending_fill_result(result: OrderResult) -> bool:
        if result is None or not getattr(result, "success", False):
            return False
        fill_mode = str(getattr(result, "fill_mode", "") or "")
        if fill_mode in {"market_pending", "limit_then_market_pending", "partial_fill_pending"}:
            return True
        requested_quantity = max(0, int(getattr(result, "requested_quantity", 0) or 0))
        if requested_quantity > 0 and int(getattr(result, "quantity", 0) or 0) < requested_quantity:
            return True
        return int(getattr(result, "quantity", 0) or 0) <= 0 and getattr(result, "side", None) in {
            OrderSide.BUY,
            OrderSide.SELL,
        }

    @staticmethod
    def _account_positions_by_symbol(account_positions: List[Position]) -> Dict[str, Position]:
        mapped: Dict[str, Position] = {}
        for item in account_positions or []:
            symbol = str(getattr(item, "symbol", "") or "").strip()
            if symbol:
                mapped[symbol] = item
        return mapped

    def _reconciled_order_result(
        self,
        source: OrderResult,
        *,
        quantity: int,
        price: int,
        fill_mode: str = "account_reconciled",
    ) -> OrderResult:
        return OrderResult(
            success=True,
            order_no=str(getattr(source, "order_no", "") or ""),
            message="account_reconciled_pending_fill",
            symbol=str(getattr(source, "symbol", "") or ""),
            side=getattr(source, "side", None),
            quantity=max(0, int(quantity or 0)),
            price=max(0, int(price or 0)),
            requested_price=int(getattr(source, "requested_price", 0) or 0),
            reference_price=int(getattr(source, "reference_price", 0) or 0),
            fill_mode=str(fill_mode or "account_reconciled"),
            protective_exit_mode=str(getattr(source, "protective_exit_mode", "") or ""),
            protective_fallback_used=bool(getattr(source, "protective_fallback_used", False)),
            stop_reference_amount_krw=max(0, int(getattr(source, "stop_reference_amount_krw", 0) or 0)),
            requested_reason=str(getattr(source, "requested_reason", "") or ""),
            timestamp=getattr(source, "timestamp", None) or self._now(),
            requested_quantity=max(0, int(getattr(source, "requested_quantity", 0) or 0)),
        )

    def reconcile_pending_fills_from_account(
        self,
        results: List[OrderResult],
        account_positions: List[Position],
    ) -> List[OrderResult]:
        """체결조회 제한으로 pending 처리된 주문을 계좌 변화로 확정 보정한다."""
        if not results:
            return []
        account_by_symbol = self._account_positions_by_symbol(account_positions)
        reconciled: List[OrderResult] = []
        for result in results:
            if not self._is_pending_fill_result(result):
                continue
            symbol = str(getattr(result, "symbol", "") or "").strip()
            if not symbol:
                continue
            account_pos = account_by_symbol.get(symbol)
            if result.side == OrderSide.BUY:
                before_pos = self.positions.get(symbol)
                before_qty = int(getattr(before_pos, "quantity", 0) or 0) if before_pos is not None else 0
                after_qty = int(getattr(account_pos, "quantity", 0) or 0) if account_pos is not None else before_qty
                inferred_qty = max(0, after_qty - before_qty)
                if inferred_qty <= 0:
                    if (
                        str(getattr(result, "fill_mode", "") or "") == "order_result_pending"
                        and symbol in self._pending_entry_meta
                        and account_pos is None
                    ):
                        meta = self._pending_entry_meta.get(symbol)
                        if isinstance(meta, dict) and self._pending_entry_meta_in_reconcile_grace(meta):
                            if not meta.get("pending_order_created_at"):
                                meta["pending_order_created_at"] = self._serialize_datetime(
                                    getattr(result, "timestamp", None) or self._now()
                                )
                            if not meta.get("pending_order_reference_price"):
                                reference_price = int(
                                    getattr(result, "reference_price", 0)
                                    or getattr(result, "requested_price", 0)
                                    or 0
                                )
                                if reference_price > 0:
                                    meta["pending_order_reference_price"] = reference_price
                            logger.warning(
                                "주문 결과 미확정 매수가 아직 계좌 보유에서 확인되지 않아 pending 진입 메타를 유지합니다: %s",
                                symbol,
                            )
                            self._save_daily_state_if_due(force=True)
                        else:
                            self._pending_entry_meta.pop(symbol, None)
                            logger.warning(
                                "주문 결과 미확정 매수가 유예시간 이후에도 계좌 보유에서 확인되지 않아 pending 진입 메타를 정리합니다: %s",
                                symbol,
                            )
                            self._save_daily_state_if_due(force=True)
                    continue
                account_avg_price = int(round(float(getattr(account_pos, "avg_price", 0) or 0))) if account_pos is not None else 0
                buy_price = self._resolve_fill_price_with_quote_fallback(
                    symbol,
                    broker_price=account_avg_price,
                    requested_price=int(getattr(result, "requested_price", 0) or 0),
                    reference_price=int(getattr(result, "reference_price", 0) or 0),
                )
                synthetic = self._reconciled_order_result(
                    result,
                    quantity=inferred_qty,
                    price=buy_price,
                    fill_mode="account_reconciled",
                )
                logger.debug(
                    "pending 매수 체결을 계좌 재동기화로 보정합니다: %s %d주 @ %d원",
                    symbol,
                    inferred_qty,
                    buy_price,
                )
                self.on_order_filled(synthetic)
                reconciled.append(synthetic)
                continue

            if result.side == OrderSide.SELL:
                before_pos = self.positions.get(symbol)
                if before_pos is None:
                    continue
                before_qty = int(getattr(before_pos, "quantity", 0) or 0)
                after_qty = int(getattr(account_pos, "quantity", 0) or 0) if account_pos is not None else 0
                inferred_qty = max(0, min(before_qty, before_qty - after_qty))
                if inferred_qty <= 0:
                    continue
                sell_price = self._resolve_fill_price_with_quote_fallback(
                    symbol,
                    broker_price=int(getattr(result, "price", 0) or 0),
                    requested_price=int(getattr(result, "requested_price", 0) or 0),
                    reference_price=int(getattr(result, "reference_price", 0) or 0),
                    position=before_pos,
                )
                synthetic = self._reconciled_order_result(
                    result,
                    quantity=inferred_qty,
                    price=sell_price,
                    fill_mode="account_reconciled_estimated",
                )
                logger.debug(
                    "pending 매도 체결을 계좌 재동기화로 보정합니다: %s %d주 @ %d원 (계좌수량 %d→%d)",
                    symbol,
                    inferred_qty,
                    sell_price,
                    before_qty,
                    after_qty,
                )
                self.on_order_filled(synthetic)
                reconciled.append(synthetic)
        return reconciled

    def reconcile_no_holding_sell_failures_from_account(
        self,
        results: List[OrderResult],
        account_positions: List[Position],
    ) -> List[OrderResult]:
        """매도 무보유 응답이 계좌 무보유와 일치하면 로컬 포지션을 추정 청산으로 보정한다."""
        if not results:
            return []
        account_symbols = set(self._account_positions_by_symbol(account_positions).keys())
        reconciled: List[OrderResult] = []
        for result in results:
            if (
                getattr(result, "success", False)
                or getattr(result, "side", None) != OrderSide.SELL
                or str(getattr(result, "error_category", "") or "") != "no_holding"
            ):
                continue
            symbol = str(getattr(result, "symbol", "") or "").strip()
            if not symbol or symbol in account_symbols:
                continue
            pos = self.positions.get(symbol)
            if pos is None:
                continue
            if self._position_pending_entry_reconcile_pending(pos):
                logger.warning(
                    "pending 매수 확정 전 무보유 매도 응답이라 추정청산을 보류합니다: %s %d주 @ %d원 "
                    "(계좌 확정 전 신규 리스크 계산에 포함)",
                    symbol,
                    int(getattr(pos, "quantity", 0) or 0),
                    int(getattr(pos, "buy_price", 0) or 0),
                )
                continue
            if getattr(pos, "pending_entry_started_at", None) is not None:
                self.positions.pop(symbol, None)
                logger.warning(
                    "유예시간이 지난 pending 매수가 계좌에도 없고 매도 무보유로 확인되어 "
                    "임시 포지션을 손익 없이 제거합니다: %s %d주 @ %d원",
                    symbol,
                    int(getattr(pos, "quantity", 0) or 0),
                    int(getattr(pos, "buy_price", 0) or 0),
                )
                self._save_daily_state_if_due(force=True)
                continue
            sell_qty = max(0, int(getattr(pos, "quantity", 0) or 0))
            sell_price = self._resolve_fill_price_with_quote_fallback(
                symbol,
                broker_price=0,
                requested_price=int(getattr(result, "requested_price", 0) or 0),
                reference_price=int(getattr(result, "reference_price", 0) or 0),
                position=pos,
            )
            if sell_qty <= 0 or sell_price <= 0:
                continue
            synthetic = self._reconciled_order_result(
                result,
                quantity=sell_qty,
                price=sell_price,
                fill_mode="account_reconciled_estimated",
            )
            logger.warning(
                "매도 무보유 응답과 계좌 무보유가 일치해 로컬 포지션을 추정 청산으로 보정합니다: %s %d주 @ %d원",
                symbol,
                sell_qty,
                sell_price,
            )
            self.on_order_filled(synthetic)
            reconciled.append(synthetic)
        return reconciled

    def on_order_filled(self, result: OrderResult):
        return handle_order_filled(self, result)
