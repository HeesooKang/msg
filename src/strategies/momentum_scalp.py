from __future__ import annotations

import json
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, fields
from datetime import datetime, timedelta
from math import ceil, floor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from src.analytics.forecast_outcomes import ForecastOutcomeLedger
from src.analytics.price_prediction import (
    PREDICTION_HORIZONS_SECONDS,
    Prediction,
    build_feature_rows,
    fit_model,
    predict_batch,
)
from src.models import Order, OrderResult, OrderSide, OrderType, Position, Quote
from src.strategy import BaseStrategy
from src.strategies.momentum_scalp_pnl import (
    TradePnL,
    calculate_trade_pnl_from_prices,
    estimate_trade_net_pnl_unrounded,
)
from src.strategies.momentum_scalp_types import (
    DEFAULT_INVERSE_ETFS,
    DEFAULT_STATIC_WATCHLIST,
    DailyPnL,
    MomentumScalpConfig,
    PositionState,
    TradePlan,
)


logger = logging.getLogger("kis_trader.strategy.momentum_scalp")
order_logger = logging.getLogger("kis_trader.orders")

STATE_SCHEMA_VERSION = 4
KRX_CLOSING_AUCTION_START = (15, 20)


@dataclass
class ExpectedValueCandidate:
    quote: Quote
    prediction: Prediction
    plan: TradePlan
    forecast_payload: Dict[str, Any]
    forecast_id: str = ""


class MomentumScalpStrategy(BaseStrategy):
    """One execution-price prediction and one EV plan for every symbol."""

    def __init__(
        self,
        market_data,
        config: MomentumScalpConfig,
        fixed_pool: Optional[List[str]] = None,
    ):
        self.market_data = market_data
        self.config = config
        self.fixed_pool = self._normalize_symbols(fixed_pool or [])

        self.positions: Dict[str, PositionState] = {}
        self.daily_pnl = DailyPnL()
        self._sell_fill_ledger: List[Dict[str, Any]] = []
        self._pending_entry_meta: Dict[str, Dict[str, Any]] = {}
        self._pending_sell_fills: Dict[str, Dict[str, Any]] = {}

        self._pool: List[str] = list(self.fixed_pool)
        self._quotes_cache: Dict[str, Quote] = {}
        self._recent_quotes: Dict[str, deque[Quote]] = {}
        self._last_evaluation_count = 0
        self._entry_window_closed_log_day = ""
        self._prediction_model: Any = None
        self._prediction_training_signature: tuple[int, str] | None = None
        self._last_batch_summary_at: Optional[datetime] = None

        self._forecast_outcomes = ForecastOutcomeLedger(
            Path(self.config.forecast_outcome_root)
        )
        self._simulated_now: Optional[datetime] = None
        self._active_day = ""
        self._last_daily_state_save_at: Optional[datetime] = None
        self._halted = False
        self._halt_reason = ""
        self._loss_guard_realized_baseline = 0
        self._hard_stop_bypass_active = False
        self._real_money_stage = 0
        self._capital_scale = 1.0

    # ------------------------------------------------------------------
    # Lifecycle and universe
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_symbols(symbols: Iterable[str]) -> List[str]:
        normalized: List[str] = []
        seen = set()
        for symbol in symbols:
            value = str(symbol or "").strip()
            if len(value) != 6 or not value.isdigit() or value in seen:
                continue
            seen.add(value)
            normalized.append(value)
        return normalized

    def set_simulated_now(self, now: Optional[datetime]) -> None:
        self._simulated_now = now

    def _now(self) -> datetime:
        return self._simulated_now or datetime.now()

    def _today(self) -> str:
        return self._now().strftime("%Y%m%d")

    def _reset_for_new_day(self) -> None:
        carried_positions = list(self.positions.values())
        for position in carried_positions:
            position.plan = None
            position.pending_exit_started_at = None
            position.pending_exit_quantity = 0
        self.positions = {position.symbol: position for position in carried_positions}
        self.daily_pnl = DailyPnL()
        self._sell_fill_ledger = []
        self._pending_entry_meta = {}
        self._pending_sell_fills = {}
        self._quotes_cache = {}
        self._recent_quotes = {}
        self._entry_window_closed_log_day = ""
        self._last_batch_summary_at = None
        self._prediction_model = None
        self._prediction_training_signature = None
        self._pool = list(self.fixed_pool)
        self._halted = False
        self._halt_reason = ""
        self._loss_guard_realized_baseline = 0
        self._hard_stop_bypass_active = False
        self._last_daily_state_save_at = None
        self._active_day = self._today()

    def _ensure_active_day(self) -> None:
        if not self._active_day:
            self._active_day = self._today()
        elif self._active_day != self._today():
            self._reset_for_new_day()

    def initialize(self) -> None:
        today = self._today()
        if self._active_day and self._active_day != today:
            self._reset_for_new_day()
        else:
            self._active_day = today
        self._load_daily_state()
        self._rebuild_daily_pnl()
        self._configure_restart_loss_baseline()
        self._update_daily_halt_state()
        if self._hard_stop_bypass_active:
            self._save_daily_state_if_due(force=True)
        logger.info(
            "EV 전략 초기화: schema=%d 고정=%d 동적=%d 보유=%d 누적실현=%+d원 세션실현=%+d원 손실한도=%+d원",
            STATE_SCHEMA_VERSION,
            len(self.config.static_watchlist),
            len(self._pool),
            len(self.positions),
            self.daily_pnl.realized_net_pnl,
            self._realized_net_pnl_for_loss_guard(),
            int(self.config.daily_loss_limit),
        )

    def _configure_restart_loss_baseline(self) -> None:
        self._loss_guard_realized_baseline = 0
        self._hard_stop_bypass_active = False
        if not bool(self.config.allow_hard_stop_bypass_for_day):
            return
        realized = int(self.daily_pnl.realized_net_pnl)
        loss_halt_restored = self._halt_reason == "daily_total_loss_limit"
        if not loss_halt_restored and realized > int(self.config.daily_loss_limit):
            return
        self._loss_guard_realized_baseline = realized
        self._hard_stop_bypass_active = True
        if loss_halt_restored or self._halted:
            self._halted = False
            self._halt_reason = ""
        logger.warning(
            "당일 손실 하드스톱 재시작 우회 활성화: 누적실현=%+d원 재시작기준=%+d원 추가손실한도=%+d원",
            realized,
            self._loss_guard_realized_baseline,
            int(self.config.daily_loss_limit),
        )

    def _realized_net_pnl_for_loss_guard(self) -> int:
        return (
            int(self.daily_pnl.realized_net_pnl)
            - int(self._loss_guard_realized_baseline)
        )

    def get_watchlist(self) -> List[str]:
        symbols = list(self.positions)
        symbols.extend(self._pool)
        symbols.extend(self.config.static_watchlist or DEFAULT_STATIC_WATCHLIST)
        symbols.extend(self.config.inverse_etfs or DEFAULT_INVERSE_ETFS)
        return self._normalize_symbols(symbols)

    def update_runtime_pool(self, symbols: List[str]) -> None:
        discovered = self._normalize_symbols(symbols)
        if self.fixed_pool:
            self._pool = list(self.fixed_pool)
        else:
            fixed = set(self.config.static_watchlist or DEFAULT_STATIC_WATCHLIST)
            fixed.update(self.config.inverse_etfs or DEFAULT_INVERSE_ETFS)
            self._pool = [
                symbol for symbol in discovered if symbol not in fixed
            ][: max(1, int(self.config.dynamic_pool_size))]
        self._save_daily_state_if_due()

    def should_continue(self) -> bool:
        self._ensure_active_day()
        self._update_daily_halt_state()
        return not self._halted

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------
    @staticmethod
    def _serialize_datetime(value: Optional[datetime]) -> str:
        return value.isoformat(timespec="seconds") if value is not None else ""

    @staticmethod
    def _deserialize_datetime(value: Any) -> Optional[datetime]:
        if isinstance(value, datetime):
            return value
        try:
            return datetime.fromisoformat(str(value or ""))
        except ValueError:
            return None

    def _daily_state_path(self) -> Path:
        return Path(self.config.daily_state_path)

    def _serialize_plan(self, plan: Optional[TradePlan]) -> Optional[Dict[str, Any]]:
        if plan is None:
            return None
        return {
            "symbol": plan.symbol,
            "allowed": plan.allowed,
            "reason": plan.reason,
            "quantity": plan.quantity,
            "entry_limit_price": plan.entry_limit_price,
            "expected_exit_bid": plan.expected_exit_bid,
            "lower_exit_bid": plan.lower_exit_bid,
            "expected_net": plan.expected_net,
            "lower_net": plan.lower_net,
            "committed_risk_net_abs": plan.committed_risk_net_abs,
            "signal_at": self._serialize_datetime(plan.signal_at),
            "expires_at": self._serialize_datetime(plan.expires_at),
        }

    def _deserialize_plan(
        self,
        payload: Mapping[str, Any] | None,
    ) -> Optional[TradePlan]:
        if not isinstance(payload, Mapping):
            return None
        signal_at = self._deserialize_datetime(payload.get("signal_at"))
        expires_at = self._deserialize_datetime(payload.get("expires_at"))
        if signal_at is None or expires_at is None:
            return None
        try:
            plan = TradePlan(
                symbol=str(payload["symbol"]),
                allowed=bool(payload["allowed"]),
                reason=str(payload["reason"]),
                quantity=int(payload["quantity"]),
                entry_limit_price=int(payload["entry_limit_price"]),
                expected_exit_bid=int(payload["expected_exit_bid"]),
                lower_exit_bid=int(payload["lower_exit_bid"]),
                expected_net=float(payload["expected_net"]),
                lower_net=int(payload["lower_net"]),
                committed_risk_net_abs=int(payload["committed_risk_net_abs"]),
                signal_at=signal_at,
                expires_at=expires_at,
            )
        except (KeyError, TypeError, ValueError):
            return None
        if not plan.symbol or plan.quantity <= 0:
            return None
        return plan

    def _serialize_position(self, position: PositionState) -> Dict[str, Any]:
        return {
            "symbol": position.symbol,
            "buy_price": position.buy_price,
            "quantity": position.quantity,
            "buy_time": self._serialize_datetime(position.buy_time),
            "plan": self._serialize_plan(position.plan),
            "trade_key": position.trade_key,
            "forecast_id": position.forecast_id,
            "pending_exit_started_at": self._serialize_datetime(
                position.pending_exit_started_at
            ),
            "pending_exit_quantity": position.pending_exit_quantity,
            "pending_exit_reason": position.pending_exit_reason,
        }

    def _deserialize_position(self, payload: Mapping[str, Any]) -> Optional[PositionState]:
        symbol = str(payload.get("symbol") or "").strip()
        quantity = max(0, int(payload.get("quantity", 0) or 0))
        buy_price = max(0, int(float(payload.get("buy_price", 0) or 0)))
        if len(symbol) != 6 or quantity <= 0 or buy_price <= 0:
            return None
        buy_time = self._deserialize_datetime(payload.get("buy_time")) or self._now()
        plan = self._deserialize_plan(payload.get("plan"))
        if plan is not None and (plan.symbol != symbol or plan.quantity != quantity):
            plan = None
        return PositionState(
            symbol=symbol,
            buy_price=buy_price,
            quantity=quantity,
            buy_time=buy_time,
            plan=plan,
            trade_key=str(payload.get("trade_key") or ""),
            forecast_id=str(payload.get("forecast_id") or ""),
            pending_exit_started_at=self._deserialize_datetime(
                payload.get("pending_exit_started_at")
            ),
            pending_exit_quantity=max(
                0, int(payload.get("pending_exit_quantity", 0) or 0)
            ),
            pending_exit_reason=str(payload.get("pending_exit_reason") or ""),
        )

    def _serialize_pending_entry(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "plan": self._serialize_plan(payload.get("plan")),
            "forecast_id": str(payload.get("forecast_id") or ""),
            "signal_price": int(payload.get("signal_price", 0) or 0),
            "created_at": self._serialize_datetime(payload.get("created_at")),
            "order_no": str(payload.get("order_no") or ""),
        }

    def _deserialize_pending_entry(
        self,
        symbol: str,
        payload: Mapping[str, Any],
    ) -> Optional[Dict[str, Any]]:
        plan_payload = payload.get("plan")
        if not isinstance(plan_payload, Mapping):
            return None
        created_at = self._deserialize_datetime(payload.get("created_at")) or self._now()
        plan = self._deserialize_plan(plan_payload)
        if plan is None:
            return None
        return {
            "plan": plan,
            "forecast_id": str(payload.get("forecast_id") or ""),
            "signal_price": int(payload.get("signal_price", 0) or 0),
            "created_at": created_at,
            "order_no": str(payload.get("order_no") or ""),
        }

    def _save_daily_state_if_due(self, *, force: bool = False) -> None:
        now = self._now()
        interval = max(0, int(self.config.state_save_interval_seconds))
        if (
            not force
            and self._last_daily_state_save_at is not None
            and (now - self._last_daily_state_save_at).total_seconds() < interval
        ):
            return
        self._save_daily_state()

    def _save_daily_state(self) -> None:
        path = self._daily_state_path()
        now = self._now()
        pending_sells = {}
        for symbol, payload in self._pending_sell_fills.items():
            snapshot = payload.get("position")
            if not isinstance(snapshot, PositionState):
                continue
            pending_sells[symbol] = {
                "position": self._serialize_position(snapshot),
                "order_no": str(payload.get("order_no") or ""),
                "requested_quantity": int(payload.get("requested_quantity", 0) or 0),
                "requested_at": self._serialize_datetime(payload.get("requested_at")),
                "reference_price": int(payload.get("reference_price", 0) or 0),
                "reason": str(payload.get("reason") or ""),
                "account_remaining_quantity": payload.get("account_remaining_quantity"),
                "confirmed_quantity": int(payload.get("confirmed_quantity", 0) or 0),
                "confirmed_amount": int(payload.get("confirmed_amount", 0) or 0),
            }
        payload = {
            "schema_version": STATE_SCHEMA_VERSION,
            "date": self._today(),
            "halted": self._halted,
            "halt_reason": self._halt_reason,
            "pool": list(self._pool),
            "positions": [
                self._serialize_position(position)
                for position in self.positions.values()
            ],
            "pending_entry_meta": {
                symbol: self._serialize_pending_entry(item)
                for symbol, item in self._pending_entry_meta.items()
            },
            "pending_sell_fills": pending_sells,
            "sell_fill_ledger": list(self._sell_fill_ledger),
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(path)
            self._last_daily_state_save_at = now
        except OSError:
            logger.exception("일일 상태 저장 실패")

    def _load_daily_state(self) -> None:
        path = self._daily_state_path()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if str(payload.get("date") or "") != self._today():
            return

        self._halted = bool(payload.get("halted", False))
        self._halt_reason = str(payload.get("halt_reason") or "")
        if not self.fixed_pool:
            self._pool = self._normalize_symbols(payload.get("pool") or [])[
                : max(1, int(self.config.dynamic_pool_size))
            ]

        restored_positions: Dict[str, PositionState] = {}
        for item in payload.get("positions") or []:
            if not isinstance(item, Mapping):
                continue
            position = self._deserialize_position(item)
            if position is not None:
                restored_positions[position.symbol] = position
        self.positions = restored_positions

        self._pending_entry_meta = {}
        for symbol, item in (payload.get("pending_entry_meta") or {}).items():
            if not isinstance(item, Mapping):
                continue
            restored = self._deserialize_pending_entry(str(symbol), item)
            if restored is not None:
                self._pending_entry_meta[str(symbol)] = restored

        self._pending_sell_fills = {}
        for symbol, item in (payload.get("pending_sell_fills") or {}).items():
            if not isinstance(item, Mapping) or not isinstance(item.get("position"), Mapping):
                continue
            snapshot = self._deserialize_position(item["position"])
            if snapshot is None:
                continue
            self._pending_sell_fills[str(symbol)] = {
                "position": snapshot,
                "order_no": str(item.get("order_no") or ""),
                "requested_quantity": int(item.get("requested_quantity", 0) or 0),
                "requested_at": self._deserialize_datetime(item.get("requested_at"))
                or self._now(),
                "reference_price": int(item.get("reference_price", 0) or 0),
                "reason": str(item.get("reason") or ""),
                "account_remaining_quantity": item.get("account_remaining_quantity"),
                "confirmed_quantity": int(item.get("confirmed_quantity", 0) or 0),
                "confirmed_amount": int(item.get("confirmed_amount", 0) or 0),
            }

        ledger: List[Dict[str, Any]] = []
        seen_fill_ids = set()
        discarded_other_day = 0
        repriced_fills = 0
        for item in payload.get("sell_fill_ledger") or []:
            if not isinstance(item, Mapping):
                continue
            fill_at = self._deserialize_datetime(item.get("timestamp"))
            if fill_at is not None and fill_at.strftime("%Y%m%d") != self._today():
                discarded_other_day += 1
                continue
            if bool(item.get("price_estimated")):
                continue
            quantity = max(0, int(item.get("quantity", 0) or 0))
            sell_price = max(0, int(item.get("sell_price", 0) or 0))
            if quantity <= 0 or sell_price <= 0:
                continue
            fill_id = str(item.get("fill_id") or item.get("order_no") or "").strip()
            if fill_id and fill_id in seen_fill_ids:
                continue
            if fill_id:
                seen_fill_ids.add(fill_id)
            normalized = dict(item)
            normalized["fill_id"] = fill_id or (
                f"legacy:{normalized.get('symbol', '')}:{len(ledger) + 1}"
            )
            if self._recalculate_sell_fill_pnl(normalized):
                repriced_fills += 1
            ledger.append(normalized)
        self._sell_fill_ledger = ledger
        if discarded_other_day:
            logger.warning(
                "전일 체결원장 혼입 제거: %d건 (오늘=%s)",
                discarded_other_day,
                self._today(),
            )
        if repriced_fills:
            logger.info(
                "확정 매도체결 원장 비용 재계산: %d건 수수료율=%.6f%% 매도비용률=%.4f%%",
                repriced_fills,
                float(self.config.commission_rate) * 100.0,
                float(self.config.sell_tax_rate) * 100.0,
            )

    # ------------------------------------------------------------------
    # Confirmed fill ledger and daily PnL
    # ------------------------------------------------------------------
    def _recalculate_sell_fill_pnl(self, fill: Dict[str, Any]) -> bool:
        buy_price = max(0, int(fill.get("buy_price", 0) or 0))
        sell_price = max(0, int(fill.get("sell_price", 0) or 0))
        quantity = max(0, int(fill.get("quantity", 0) or 0))
        if buy_price <= 0 or sell_price <= 0 or quantity <= 0:
            return False
        pnl = calculate_trade_pnl_from_prices(
            entry_price=buy_price,
            exit_price=sell_price,
            quantity=quantity,
            commission_rate=self.config.commission_rate,
            sell_tax_rate=self.config.sell_tax_rate,
        )
        recalculated = {
            "gross_pnl": pnl.gross_pnl,
            "net_pnl": pnl.net_pnl,
            "fees": pnl.fees,
            "taxes": pnl.taxes,
        }
        changed = any(
            int(fill.get(key, 0) or 0) != value
            for key, value in recalculated.items()
        )
        fill.update(recalculated)
        return changed

    def _rebuild_daily_pnl(self) -> None:
        self._sell_fill_ledger = [
            fill
            for fill in self._sell_fill_ledger
            if (
                (fill_at := self._deserialize_datetime(fill.get("timestamp")))
                is None
                or fill_at.strftime("%Y%m%d") == self._today()
            )
        ]
        pnl = DailyPnL()
        grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for fill in self._sell_fill_ledger:
            pnl.realized_gross_pnl += int(fill.get("gross_pnl", 0) or 0)
            pnl.realized_net_pnl += int(fill.get("net_pnl", 0) or 0)
            pnl.fees_paid += int(fill.get("fees", 0) or 0)
            pnl.taxes_paid += int(fill.get("taxes", 0) or 0)
            grouped[str(fill.get("trade_key") or fill.get("fill_id") or "")].append(fill)

        for fills_for_trade in grouped.values():
            sold_quantity = sum(
                max(0, int(item.get("quantity", 0) or 0))
                for item in fills_for_trade
            )
            entry_quantity = max(
                [
                    max(0, int(item.get("entry_quantity", 0) or 0))
                    for item in fills_for_trade
                ]
                or [0]
            )
            if entry_quantity <= 0 or sold_quantity < entry_quantity:
                continue
            net_pnl = sum(int(item.get("net_pnl", 0) or 0) for item in fills_for_trade)
            pnl.trade_count += 1
            if net_pnl > 0:
                pnl.win_count += 1
                pnl.winning_net_pnl_sum += net_pnl
                pnl.largest_win_net = max(pnl.largest_win_net, net_pnl)
            elif net_pnl < 0:
                pnl.loss_count += 1
                pnl.losing_net_pnl_sum += net_pnl
                pnl.largest_loss_net = min(pnl.largest_loss_net, net_pnl)
            else:
                pnl.breakeven_count += 1
        self.daily_pnl = pnl

    def _trade_is_closed(self, trade_key: str) -> bool:
        fills = [
            item
            for item in self._sell_fill_ledger
            if str(item.get("trade_key") or "") == str(trade_key or "")
        ]
        if not fills:
            return False
        sold = sum(max(0, int(item.get("quantity", 0) or 0)) for item in fills)
        entered = max(max(0, int(item.get("entry_quantity", 0) or 0)) for item in fills)
        return entered > 0 and sold >= entered

    def _record_execution_outcome_if_closed(self, trade_key: str) -> None:
        fills = [
            item
            for item in self._sell_fill_ledger
            if str(item.get("trade_key") or "") == str(trade_key or "")
        ]
        if not fills or not self._trade_is_closed(trade_key):
            return
        if any(bool(item.get("execution_outcome_recorded")) for item in fills):
            return
        quantity = sum(max(0, int(item.get("quantity", 0) or 0)) for item in fills)
        sell_notional = sum(
            max(0, int(item.get("sell_price", 0) or 0))
            * max(0, int(item.get("quantity", 0) or 0))
            for item in fills
        )
        first = fills[0]
        if quantity <= 0 or sell_notional <= 0:
            return
        recorded = self._forecast_outcomes.record_execution_outcome(
            symbol=str(first.get("symbol") or ""),
            forecast_id=str(first.get("forecast_id") or ""),
            entry_price=max(0, int(first.get("buy_price", 0) or 0)),
            exit_price=int(round(sell_notional / quantity)),
            quantity=quantity,
            net_pnl=sum(int(item.get("net_pnl", 0) or 0) for item in fills),
            now=self._now(),
        )
        if recorded:
            for item in fills:
                item["execution_outcome_recorded"] = True

    def _append_confirmed_sell_fill(
        self,
        *,
        result: OrderResult,
        position: PositionState,
        quantity: int,
        sell_price: int,
        apply_to_position: bool,
        clear_pending: bool = True,
        fill_id: str = "",
    ) -> Optional[Dict[str, Any]]:
        resolved_quantity = max(0, min(int(quantity or 0), int(position.quantity or 0)))
        resolved_price = max(0, int(sell_price or 0))
        if resolved_quantity <= 0 or resolved_price <= 0:
            return None
        order_no = str(result.order_no or "").strip()
        resolved_fill_id = str(fill_id or "").strip() or order_no or (
            f"{position.symbol}:{self._now().isoformat(timespec='microseconds')}"
        )
        if any(
            str(item.get("fill_id") or "") == resolved_fill_id
            for item in self._sell_fill_ledger
        ):
            return None
        pnl = calculate_trade_pnl_from_prices(
            entry_price=position.buy_price,
            exit_price=resolved_price,
            quantity=resolved_quantity,
            commission_rate=self.config.commission_rate,
            sell_tax_rate=self.config.sell_tax_rate,
        )
        previously_sold = sum(
            max(0, int(item.get("quantity", 0) or 0))
            for item in self._sell_fill_ledger
            if str(item.get("trade_key") or "") == position.trade_key
        )
        entry_quantity = max(
            resolved_quantity,
            int(position.quantity or 0),
            previously_sold + resolved_quantity,
        )
        record = {
            "fill_id": resolved_fill_id,
            "order_no": order_no,
            "symbol": position.symbol,
            "trade_key": position.trade_key,
            "quantity": resolved_quantity,
            "entry_quantity": entry_quantity,
            "buy_price": position.buy_price,
            "sell_price": resolved_price,
            "gross_pnl": pnl.gross_pnl,
            "net_pnl": pnl.net_pnl,
            "fees": pnl.fees,
            "taxes": pnl.taxes,
            "timestamp": self._serialize_datetime(self._now()),
            "requested_reason": str(result.requested_reason or position.pending_exit_reason),
            "forecast_id": position.forecast_id,
            "planned_expected_net": float(position.plan.expected_net if position.plan else 0.0),
            "planned_expected_exit_bid": int(position.plan.expected_exit_bid if position.plan else 0),
            "planned_lower_net": int(position.plan.lower_net if position.plan else 0),
        }
        self._sell_fill_ledger.append(record)

        if apply_to_position:
            current = self.positions.get(position.symbol)
            if current is not None and current.trade_key == position.trade_key:
                current.quantity = max(0, current.quantity - resolved_quantity)
                if clear_pending:
                    current.pending_exit_started_at = None
                    current.pending_exit_quantity = 0
                    current.pending_exit_reason = ""
                else:
                    current.pending_exit_quantity = current.quantity
                if current.quantity <= 0:
                    self.positions.pop(current.symbol, None)

        if clear_pending:
            self._pending_sell_fills.pop(position.symbol, None)
        self._rebuild_daily_pnl()
        self._record_execution_outcome_if_closed(position.trade_key)
        self._update_daily_halt_state()
        self._save_daily_state_if_due(force=True)
        fill_label = "체결" if clear_pending else "부분체결"
        logger.info(
            "매도 %s 확정: %s %d주 @ %d원 순손익=%+d원 일실현=%+d원",
            fill_label,
            position.symbol,
            resolved_quantity,
            resolved_price,
            pnl.net_pnl,
            self.daily_pnl.realized_net_pnl,
        )
        order_logger.info(
            "매도 %s: sell %s MARKET %d주 @ 체결가 %d원 (순손익 %+d원)",
            fill_label,
            position.symbol,
            resolved_quantity,
            resolved_price,
            pnl.net_pnl,
        )
        return record

    # ------------------------------------------------------------------
    # Quotes, prices and cost model
    # ------------------------------------------------------------------
    def _recent_window(self, symbol: str) -> deque[Quote]:
        window = self._recent_quotes.get(symbol)
        if window is None:
            window = deque(
                maxlen=max(
                    240,
                    PREDICTION_HORIZONS_SECONDS[-1] + 30,
                )
            )
            self._recent_quotes[symbol] = window
        return window

    def _ingest_quotes(self, quotes: Sequence[Quote]) -> None:
        now = self._now()
        history_seconds = max(
            240,
            PREDICTION_HORIZONS_SECONDS[-1] + 60,
        )
        oldest_allowed = now - timedelta(seconds=history_seconds)
        newest_allowed = now + timedelta(seconds=1)
        for quote in quotes:
            symbol = str(quote.symbol or "").strip()
            if (
                len(symbol) != 6
                or int(quote.current_price or 0) <= 0
                or quote.timestamp.date() != now.date()
                or quote.timestamp < oldest_allowed
                or quote.timestamp > newest_allowed
            ):
                continue
            self._quotes_cache[symbol] = quote
            window = self._recent_window(symbol)
            while window and window[0].timestamp < oldest_allowed:
                window.popleft()
            if window and window[-1].timestamp == quote.timestamp:
                window[-1] = quote
            else:
                window.append(quote)

    def _fresh_quotes(self) -> List[Quote]:
        now = self._now()
        maximum_age = max(1, int(self.config.quote_freshness_seconds))
        fresh = []
        for quote in self._quotes_cache.values():
            age = (now - quote.timestamp).total_seconds()
            if age < -1.0 or age > maximum_age:
                continue
            if int(quote.current_price or 0) > 0:
                fresh.append(quote)
        return sorted(fresh, key=lambda item: str(item.symbol))

    @staticmethod
    def _closing_auction_start(now: datetime) -> datetime:
        return now.replace(
            hour=KRX_CLOSING_AUCTION_START[0],
            minute=KRX_CLOSING_AUCTION_START[1],
            second=0,
            microsecond=0,
        )

    def _available_prediction_horizons(self, now: datetime) -> tuple[int, ...]:
        closing_auction = self._closing_auction_start(now)
        return tuple(
            horizon
            for horizon in PREDICTION_HORIZONS_SECONDS
            if now + timedelta(seconds=horizon) < closing_auction
        )

    def _continuous_quote_phase(self, now: datetime) -> bool:
        return now < self._closing_auction_start(now)

    @staticmethod
    def _krx_tick_size(price: int) -> int:
        value = max(1, int(price or 0))
        if value < 2_000:
            return 1
        if value < 5_000:
            return 5
        if value < 20_000:
            return 10
        if value < 50_000:
            return 50
        if value < 200_000:
            return 100
        if value < 500_000:
            return 500
        return 1_000

    def _observed_tick_size(self, quote: Quote) -> int:
        current = max(1, int(quote.current_price or 0))
        ask = max(0, int(quote.ask_price or 0))
        bid = max(0, int(quote.bid_price or 0))
        candidates = [
            value
            for value in (ask - bid, abs(ask - current), abs(current - bid))
            if value > 0
        ]
        observed = min(candidates) if candidates else 0
        if observed > 0 and observed <= max(1, int(current * 0.01)):
            return observed
        return self._krx_tick_size(current)

    def _entry_limit_price(self, quote: Quote) -> int:
        current = max(1, int(quote.current_price or 0))
        ask = max(0, int(quote.ask_price or 0))
        if ask > 0:
            return ask
        if bool(getattr(quote, "book_available", False)):
            return 0
        return current + self._observed_tick_size(quote)

    def _current_exit_price(self, quote: Quote) -> int:
        current = max(1, int(quote.current_price or 0))
        bid = max(0, int(quote.bid_price or 0))
        return bid if bid > 0 else max(1, current - self._observed_tick_size(quote))

    def _trade_pnl(self, entry_price: int, exit_price: int, quantity: int) -> TradePnL:
        return calculate_trade_pnl_from_prices(
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            commission_rate=self.config.commission_rate,
            sell_tax_rate=self.config.sell_tax_rate,
        )

    # ------------------------------------------------------------------
    # Risk, prediction and one TradePlan
    # ------------------------------------------------------------------
    def _open_capital(self, *, exclude_symbol: str = "") -> int:
        return sum(
            max(0, int(position.buy_price or 0) * int(position.quantity or 0))
            for symbol, position in self.positions.items()
            if symbol != exclude_symbol
        )

    def _pending_entry_capital(self, *, exclude_symbol: str = "") -> int:
        return sum(
            max(0, int(item["plan"].entry_limit_price) * int(item["plan"].quantity))
            for symbol, item in self._pending_entry_meta.items()
            if symbol != exclude_symbol and isinstance(item.get("plan"), TradePlan)
        )

    def _open_committed_risk(self, *, exclude_symbol: str = "") -> int:
        total = 0
        hard_limit = abs(int(self.config.daily_loss_limit))
        for symbol, position in self.positions.items():
            if symbol == exclude_symbol:
                continue
            if position.plan is None:
                total += hard_limit
            else:
                total += max(0, int(position.plan.committed_risk_net_abs))
        for symbol, payload in self._pending_sell_fills.items():
            if symbol == exclude_symbol or symbol in self.positions:
                continue
            snapshot = payload.get("position")
            if isinstance(snapshot, PositionState) and snapshot.plan is not None:
                total += max(0, int(snapshot.plan.committed_risk_net_abs))
        return total

    def _pending_entry_risk(self, *, exclude_symbol: str = "") -> int:
        return sum(
            max(0, int(item["plan"].committed_risk_net_abs))
            for symbol, item in self._pending_entry_meta.items()
            if symbol != exclude_symbol and isinstance(item.get("plan"), TradePlan)
        )

    def _available_capital(self, *, exclude_symbol: str = "") -> int:
        return max(
            0,
            int(self.config.seed_money)
            - self._open_capital(exclude_symbol=exclude_symbol)
            - self._pending_entry_capital(exclude_symbol=exclude_symbol),
        )

    def _remaining_daily_loss_room(self, *, exclude_symbol: str = "") -> int:
        return max(
            0,
            abs(int(self.config.daily_loss_limit))
            + self._realized_net_pnl_for_loss_guard()
            - self._open_committed_risk(exclude_symbol=exclude_symbol)
            - self._pending_entry_risk(exclude_symbol=exclude_symbol),
        )

    def _rejected_plan(
        self,
        *,
        symbol: str,
        reason: str,
        now: datetime,
        entry_price: int = 0,
        prediction: Optional[Prediction] = None,
        evaluated_quantity: int = 0,
        values: Optional[Mapping[str, float | int]] = None,
    ) -> TradePlan:
        evaluated = values or {}
        horizon = (
            max(30, int(prediction.horizon_seconds))
            if prediction is not None
            else PREDICTION_HORIZONS_SECONDS[-1]
        )
        return TradePlan(
            symbol=symbol,
            allowed=False,
            reason=reason,
            quantity=max(0, int(evaluated_quantity)),
            entry_limit_price=max(0, int(entry_price or 0)),
            expected_exit_bid=max(0, int(evaluated.get("expected_exit_bid", 0) or 0)),
            lower_exit_bid=max(0, int(evaluated.get("lower_exit_bid", 0) or 0)),
            expected_net=round(float(evaluated.get("expected_net", 0.0)), 2),
            lower_net=int(evaluated.get("lower_net", 0) or 0),
            committed_risk_net_abs=max(
                0,
                int(evaluated.get("committed_risk_net_abs", 0) or 0),
            ),
            signal_at=now,
            expires_at=now + timedelta(seconds=horizon),
        )

    def _exit_price_for_net_return(self, entry_price: int, net_return_pct: float) -> int:
        entry = max(1, int(entry_price or 0))
        commission = max(0.0, float(self.config.commission_rate))
        tax = max(0.0, float(self.config.sell_tax_rate))
        denominator = max(1e-9, 1.0 - commission - tax)
        desired_net = float(entry) * float(net_return_pct) / 100.0
        raw_exit = (desired_net + float(entry) * (1.0 + commission)) / denominator
        tick = self._krx_tick_size(max(1, int(raw_exit)))
        return max(tick, int(floor(raw_exit / tick) * tick))

    def _plan_values_for_quantity(
        self,
        *,
        quote: Quote,
        entry_price: int,
        prediction: Prediction,
        quantity: int,
        expected_exit_bid: int = 0,
        lower_exit_bid: int = 0,
    ) -> Dict[str, float | int]:
        expected_exit = max(
            1,
            int(expected_exit_bid or 0)
            or self._exit_price_for_net_return(
                entry_price,
                prediction.expected_net_return_pct,
            ),
        )
        lower_exit = max(
            1,
            int(lower_exit_bid or 0)
            or self._exit_price_for_net_return(
                entry_price,
                prediction.lower_net_return_pct,
            ),
        )
        flat_exit = self._current_exit_price(quote)
        expected_net = estimate_trade_net_pnl_unrounded(
            entry_price=entry_price,
            exit_price=expected_exit,
            quantity=quantity,
            commission_rate=self.config.commission_rate,
            sell_tax_rate=self.config.sell_tax_rate,
        )
        lower_net = self._trade_pnl(entry_price, lower_exit, quantity).net_pnl
        flat_net = self._trade_pnl(entry_price, flat_exit, quantity).net_pnl
        committed_risk = max(0, -lower_net, -flat_net)
        return {
            "expected_exit_bid": expected_exit,
            "lower_exit_bid": lower_exit,
            "expected_net": expected_net,
            "lower_net": lower_net,
            "flat_net": flat_net,
            "committed_risk_net_abs": committed_risk,
        }

    def _build_trade_plan(
        self,
        quote: Quote,
        prediction: Prediction,
        *,
        now: datetime,
        fixed_quantity: Optional[int] = None,
        exclude_symbol: str = "",
    ) -> TradePlan:
        symbol = str(quote.symbol)
        entry_price = self._entry_limit_price(quote)
        if entry_price <= 0:
            return self._rejected_plan(
                symbol=symbol,
                reason="no_executable_ask",
                now=now,
                entry_price=0,
                prediction=prediction,
            )
        if not prediction.ready:
            return self._rejected_plan(
                symbol=symbol,
                reason=prediction.reason,
                now=now,
                entry_price=entry_price,
                prediction=prediction,
            )
        available_capital = self._available_capital(exclude_symbol=exclude_symbol)
        if fixed_quantity is None and entry_price > available_capital:
            return self._rejected_plan(
                symbol=symbol,
                reason="unaffordable_one_share",
                now=now,
                entry_price=entry_price,
                prediction=prediction,
            )

        per_share = self._plan_values_for_quantity(
            quote=quote,
            entry_price=entry_price,
            prediction=prediction,
            quantity=1,
        )
        loss_room = self._remaining_daily_loss_room(exclude_symbol=exclude_symbol)
        per_share_risk = max(1.0, float(per_share["committed_risk_net_abs"]))
        if fixed_quantity is None:
            max_by_capital = available_capital // max(1, entry_price)
            max_by_risk = int(floor(float(loss_room) / per_share_risk))
            quantity = max(0, min(max_by_capital, max_by_risk))
        else:
            quantity = max(0, int(fixed_quantity))
        if quantity <= 0:
            return self._rejected_plan(
                symbol=symbol,
                reason="insufficient_capital_or_loss_room",
                now=now,
                entry_price=entry_price,
                prediction=prediction,
            )

        values = self._plan_values_for_quantity(
            quote=quote,
            entry_price=entry_price,
            prediction=prediction,
            quantity=quantity,
        )
        if float(values["expected_net"]) <= 0.0:
            return self._rejected_plan(
                symbol=symbol,
                reason="non_positive_expected_net",
                now=now,
                entry_price=entry_price,
                prediction=prediction,
                evaluated_quantity=quantity,
                values=values,
            )
        committed_risk = int(values["committed_risk_net_abs"])
        if committed_risk > loss_room:
            if fixed_quantity is not None:
                return self._rejected_plan(
                    symbol=symbol,
                    reason="risk_exceeds_remaining_loss_room",
                    now=now,
                    entry_price=entry_price,
                    prediction=prediction,
                    evaluated_quantity=quantity,
                    values=values,
                )
            overrun = committed_risk - loss_room
            quantity = max(0, quantity - int(ceil(overrun / per_share_risk)))
            if quantity <= 0:
                return self._rejected_plan(
                    symbol=symbol,
                    reason="risk_rounding_exceeds_loss_room",
                    now=now,
                    entry_price=entry_price,
                    prediction=prediction,
                )
            values = self._plan_values_for_quantity(
                quote=quote,
                entry_price=entry_price,
                prediction=prediction,
                quantity=quantity,
            )
            committed_risk = int(values["committed_risk_net_abs"])
        if float(values["expected_net"]) <= 0.0 or committed_risk > loss_room:
            return self._rejected_plan(
                symbol=symbol,
                reason="final_plan_not_positive_or_safe",
                now=now,
                entry_price=entry_price,
                prediction=prediction,
                evaluated_quantity=quantity,
                values=values,
            )

        signal_at = quote.timestamp
        return TradePlan(
            symbol=symbol,
            allowed=True,
            reason="positive_expected_value",
            quantity=quantity,
            entry_limit_price=entry_price,
            expected_exit_bid=max(1, int(values["expected_exit_bid"])),
            lower_exit_bid=max(1, int(values["lower_exit_bid"])),
            expected_net=round(float(values["expected_net"]), 2),
            lower_net=int(values["lower_net"]),
            committed_risk_net_abs=committed_risk,
            signal_at=signal_at,
            expires_at=signal_at + timedelta(seconds=prediction.horizon_seconds),
        )

    def _build_predictions(
        self,
        quotes: Sequence[Quote],
        *,
        now: datetime,
        evaluated_candidate_count: int,
        horizons: Sequence[int],
    ) -> Dict[int, Dict[str, Prediction]]:
        recent_by_symbol = {
            str(quote.symbol): list(self._recent_window(str(quote.symbol)))
            for quote in quotes
        }
        feature_rows = build_feature_rows(
            quotes,
            recent_quotes_by_symbol=recent_by_symbol,
        )
        outcomes = self._forecast_outcomes.training_outcomes(as_of=now)
        resolved_horizons = tuple(sorted({max(30, int(value)) for value in horizons}))
        joint_candidate_count = max(
            1,
            int(evaluated_candidate_count) * max(1, len(resolved_horizons)),
        )
        last_outcome = outcomes[-1] if outcomes else {}
        training_signature = (
            len(outcomes),
            f"{last_outcome.get('forecast_id', '')}:"
            f"{last_outcome.get('outcome_timestamp', '')}",
        )
        if self._prediction_training_signature != training_signature:
            self._prediction_model = fit_model(
                outcomes,
                as_of=now,
            )
            self._prediction_training_signature = training_signature
        predictions: Dict[int, Dict[str, Prediction]] = {}
        for horizon in resolved_horizons:
            predictions[horizon] = predict_batch(
                feature_rows,
                model=self._prediction_model,
                evaluated_candidate_count=joint_candidate_count,
                horizon_seconds=horizon,
            )
        return predictions

    def _forecast_payload(
        self,
        candidate: ExpectedValueCandidate,
    ) -> Dict[str, Any]:
        prediction = candidate.prediction
        plan = candidate.plan
        return {
            "symbol": candidate.quote.symbol,
            "signal_timestamp": self._serialize_datetime(plan.signal_at),
            "signal_entry_ask": plan.entry_limit_price,
            "horizon_seconds": prediction.horizon_seconds,
            "sample_count": prediction.sample_count,
            "ready": prediction.ready,
            "reason": prediction.reason,
            "compact_features": dict(prediction.compact_features),
            "unavailable_features": list(prediction.unavailable_features),
            "expected_net_return_pct": prediction.expected_net_return_pct,
            "lower_net_return_pct": prediction.lower_net_return_pct,
            "commission_rate": float(self.config.commission_rate),
            "sell_tax_rate": float(self.config.sell_tax_rate),
            "allowed": plan.allowed,
            "decision_reason": plan.reason,
            "quantity": plan.quantity,
            "entry_limit_price": plan.entry_limit_price,
            "expected_exit_bid": plan.expected_exit_bid,
            "lower_exit_bid": plan.lower_exit_bid,
            "expected_net": plan.expected_net,
            "lower_net": plan.lower_net,
            "committed_risk_net_abs": plan.committed_risk_net_abs,
        }

    @staticmethod
    def _candidate_sort_key(candidate: ExpectedValueCandidate) -> tuple[float, float, str]:
        risk = max(1, int(candidate.plan.committed_risk_net_abs))
        return (
            -float(candidate.plan.expected_net),
            -(float(candidate.plan.expected_net) / float(risk)),
            str(candidate.quote.symbol),
        )

    # ------------------------------------------------------------------
    # Position adoption and planned exits
    # ------------------------------------------------------------------
    def _reprice_plan_for_fill(
        self,
        *,
        plan: TradePlan,
        quote: Quote,
        fill_price: int,
        quantity: int,
    ) -> TradePlan:
        resolved_entry = max(1, int(fill_price))
        resolved_quantity = max(1, int(quantity))
        expected_net = estimate_trade_net_pnl_unrounded(
            entry_price=resolved_entry,
            exit_price=max(1, int(plan.expected_exit_bid)),
            quantity=resolved_quantity,
            commission_rate=self.config.commission_rate,
            sell_tax_rate=self.config.sell_tax_rate,
        )
        lower_net = self._trade_pnl(
            resolved_entry,
            max(1, int(plan.lower_exit_bid)),
            resolved_quantity,
        ).net_pnl
        flat_net = self._trade_pnl(
            resolved_entry,
            self._current_exit_price(quote),
            resolved_quantity,
        ).net_pnl
        committed_risk = max(0, -lower_net, -flat_net)
        return TradePlan(
            symbol=plan.symbol,
            allowed=expected_net > 0.0,
            reason="recalculated_for_fill" if expected_net > 0.0 else "unfavorable_fill_non_positive_ev",
            quantity=resolved_quantity,
            entry_limit_price=resolved_entry,
            expected_exit_bid=plan.expected_exit_bid,
            lower_exit_bid=plan.lower_exit_bid,
            expected_net=round(expected_net, 2),
            lower_net=lower_net,
            committed_risk_net_abs=committed_risk,
            signal_at=plan.signal_at,
            expires_at=plan.expires_at,
        )

    def _adopt_unplanned_position(
        self,
        position: PositionState,
        quote: Quote,
    ) -> Optional[Order]:
        return self._make_market_sell(
            position,
            quote,
            reason="unplanned_position",
        )

    def _position_exit_order(
        self,
        position: PositionState,
        quote: Quote,
        *,
        now: datetime,
    ) -> Optional[Order]:
        if position.pending_exit_started_at is not None:
            return None
        plan = position.plan
        if plan is None:
            return None
        if now >= plan.expires_at:
            return self._make_market_sell(position, quote, reason="horizon_expiry")
        return None

    def _make_market_sell(
        self,
        position: PositionState,
        quote: Optional[Quote],
        *,
        reason: str,
    ) -> Optional[Order]:
        if position.quantity <= 0 or position.pending_exit_started_at is not None:
            return None
        reference_price = (
            self._current_exit_price(quote)
            if quote is not None
            else max(1, int(position.buy_price or 0))
        )
        now = self._now()
        position.pending_exit_started_at = now
        position.pending_exit_quantity = position.quantity
        position.pending_exit_reason = reason
        self._pending_sell_fills[position.symbol] = {
            "position": self._deserialize_position(self._serialize_position(position)) or position,
            "order_no": "",
            "requested_quantity": position.quantity,
            "requested_at": now,
            "reference_price": reference_price,
            "reason": reason,
            "account_remaining_quantity": position.quantity,
            "confirmed_quantity": 0,
            "confirmed_amount": 0,
        }
        logger.info(
            "계획 청산: %s reason=%s %d주 기준가=%d원",
            position.symbol,
            reason,
            position.quantity,
            reference_price,
        )
        return Order(
            symbol=position.symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=position.quantity,
            price=0,
            reference_price=reference_price,
            requested_reason=reason,
        )

    def _liquidate_all(self) -> List[Order]:
        orders = []
        for symbol, position in list(self.positions.items()):
            quote = self._quotes_cache.get(symbol)
            order = self._make_market_sell(
                position,
                quote,
                reason=self._halt_reason or "session_halt",
            )
            if order is not None:
                orders.append(order)
        return orders

    # ------------------------------------------------------------------
    # Daily breakers and execution state
    # ------------------------------------------------------------------
    def _unrealized_net_pnl(self) -> int:
        total = 0
        for symbol, position in self.positions.items():
            quote = self._quotes_cache.get(symbol)
            if quote is None:
                continue
            total += self._trade_pnl(
                position.buy_price,
                self._current_exit_price(quote),
                position.quantity,
            ).net_pnl
        return total

    def _update_daily_halt_state(self) -> None:
        realized = int(self.daily_pnl.realized_net_pnl)
        if realized >= int(self.config.daily_profit_target):
            self._halted = True
            self._halt_reason = "daily_profit_target"
            return
        session_realized = self._realized_net_pnl_for_loss_guard()
        unrealized = self._unrealized_net_pnl()
        total = session_realized + unrealized
        if total <= int(self.config.daily_loss_limit):
            if not (
                self._halted
                and self._halt_reason == "daily_total_loss_limit"
            ):
                logger.warning(
                    "당일 손실 하드스톱: 세션실현=%+d원 미실현=%+d원 합계=%+d원 한도=%+d원",
                    session_realized,
                    unrealized,
                    total,
                    int(self.config.daily_loss_limit),
                )
            self._halted = True
            self._halt_reason = "daily_total_loss_limit"

    def _api_cooldown_remaining(self) -> float:
        try:
            return max(0.0, float(self.market_data.client.rate_limit_cooldown_remaining()))
        except Exception:
            return 0.0

    def _entry_submission_available(self) -> bool:
        if self._halted:
            return False
        if self._pending_entry_meta:
            return False
        if any(
            position.pending_exit_started_at is not None
            for position in self.positions.values()
        ):
            return False
        return self._api_cooldown_remaining() <= 0.0

    # ------------------------------------------------------------------
    # Single batch route
    # ------------------------------------------------------------------
    def on_batch_tick(self, quotes: List[Quote]) -> List[Order]:
        self._ensure_active_day()
        now = self._now()

        if self._continuous_quote_phase(now):
            self._forecast_outcomes.settle(quotes, now=now)
        self._ingest_quotes(quotes)
        fresh_quotes = self._fresh_quotes()

        self._update_daily_halt_state()
        if self._halted:
            orders = self._liquidate_all()
            self._save_daily_state_if_due(force=True)
            return orders

        available_capital = self._available_capital()
        purchasable_quotes = []
        for quote in fresh_quotes:
            entry_price = self._entry_limit_price(quote)
            if (
                str(quote.symbol) not in self.positions
                and 0 < entry_price <= available_capital
            ):
                purchasable_quotes.append(quote)
        quote_by_symbol = {str(quote.symbol): quote for quote in fresh_quotes}
        exit_orders: List[Order] = []
        for symbol, position in list(self.positions.items()):
            quote = quote_by_symbol.get(symbol)
            if quote is None:
                continue
            if position.plan is None:
                adoption_exit = self._adopt_unplanned_position(
                    position,
                    quote,
                )
                if adoption_exit is not None:
                    exit_orders.append(adoption_exit)
                continue
            order = self._position_exit_order(position, quote, now=now)
            if order is not None:
                exit_orders.append(order)

        available_horizons = self._available_prediction_horizons(now)
        if not available_horizons:
            self._last_evaluation_count = 0
            day_key = now.date().isoformat()
            if self._entry_window_closed_log_day != day_key:
                self._entry_window_closed_log_day = day_key
                logger.info(
                    "신규 진입 종료: 최단 %d초 예측 만료가 KRX 마감 동시호가 시작 "
                    "15:20을 넘습니다. 보유 포지션 청산만 계속합니다.",
                    PREDICTION_HORIZONS_SECONDS[0],
                )
            self._save_daily_state_if_due(force=bool(exit_orders))
            return exit_orders

        predictions_by_horizon = self._build_predictions(
            fresh_quotes,
            now=now,
            evaluated_candidate_count=len(purchasable_quotes),
            horizons=available_horizons,
        )
        candidates: List[ExpectedValueCandidate] = []
        for quote in purchasable_quotes:
            symbol = str(quote.symbol)
            for horizon in available_horizons:
                prediction = predictions_by_horizon[horizon][symbol]
                plan = self._build_trade_plan(quote, prediction, now=now)
                candidate = ExpectedValueCandidate(
                    quote=quote,
                    prediction=prediction,
                    plan=plan,
                    forecast_payload={},
                )
                candidate.forecast_payload = self._forecast_payload(candidate)
                candidates.append(candidate)

        allowed = sorted(
            (candidate for candidate in candidates if candidate.plan.allowed),
            key=self._candidate_sort_key,
        )
        selected = allowed[0] if allowed else None
        selected_for_execution = None
        if not exit_orders and self._entry_submission_available():
            selected_for_execution = next(
                (
                    candidate
                    for candidate in allowed
                    if not self._forecast_outcomes.has_pending_selected(
                        candidate.quote.symbol,
                        now=now,
                    )
                ),
                None,
            )
        if selected_for_execution is not None:
            selected_for_execution.forecast_id = self._forecast_outcomes.record(
                selected_for_execution.forecast_payload,
                now=now,
                selected=True,
            ) or ""
        self._forecast_outcomes.record_observation_sets(
            [candidate.forecast_payload for candidate in candidates],
            now=now,
        )

        self._last_evaluation_count = len(purchasable_quotes)
        if (
            self._last_batch_summary_at is None
            or (now - self._last_batch_summary_at).total_seconds() >= 30.0
        ):
            positive_symbols = len(
                {candidate.quote.symbol for candidate in allowed}
            )
            logger.info(
                "EV 배치: 최신=%d 종목=%d 계획=%d 양수종목=%d 보유=%d best=%s",
                len(fresh_quotes),
                len(purchasable_quotes),
                len(candidates),
                positive_symbols,
                len(self.positions),
                (
                    f"{selected.quote.symbol}/{selected.prediction.horizon_seconds}s"
                    if selected is not None
                    else "-"
                ),
            )
            self._last_batch_summary_at = now

        if exit_orders:
            self._save_daily_state_if_due(force=True)
            return exit_orders
        if selected_for_execution is None:
            self._save_daily_state_if_due()
            return []

        selected = selected_for_execution
        plan = selected.plan
        self._pending_entry_meta[plan.symbol] = {
            "plan": plan,
            "forecast_id": selected.forecast_id,
            "signal_price": int(selected.quote.current_price or 0),
            "created_at": now,
            "order_no": "",
        }
        logger.info(
            "EV 매수 선택: %s %d주 @ %d원 horizon=%d초 exit_bid=%d "
            "EV=%+.0f원 risk=%d원",
            plan.symbol,
            plan.quantity,
            plan.entry_limit_price,
            selected.prediction.horizon_seconds,
            plan.expected_exit_bid,
            plan.expected_net,
            plan.committed_risk_net_abs,
        )
        self._save_daily_state_if_due(force=True)
        return [
            Order(
                symbol=plan.symbol,
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=plan.quantity,
                price=plan.entry_limit_price,
                reference_price=int(selected.quote.current_price or plan.entry_limit_price),
                requested_reason="expected_value",
            )
        ]

    # ------------------------------------------------------------------
    # Fills and account reconciliation
    # ------------------------------------------------------------------
    @staticmethod
    def _copy_position(position: PositionState) -> PositionState:
        copied = PositionState(
            **{
                field.name: getattr(position, field.name)
                for field in fields(PositionState)
            }
        )
        return copied

    def _apply_buy_fill(
        self,
        result: OrderResult,
        *,
        quantity: int,
        price: int,
    ) -> Optional[PositionState]:
        symbol = str(result.symbol or "")
        meta = self._pending_entry_meta.get(symbol)
        if meta is None or not isinstance(meta.get("plan"), TradePlan):
            return None
        plan: TradePlan = meta["plan"]
        quote = self._quotes_cache.get(symbol)
        if quote is None:
            return None
        resolved_quantity = max(0, int(quantity or 0))
        resolved_price = max(0, int(price or 0))
        if resolved_quantity <= 0 or resolved_price <= 0:
            return None
        repriced = self._reprice_plan_for_fill(
            plan=plan,
            quote=quote,
            fill_price=resolved_price,
            quantity=resolved_quantity,
        )
        existing = self.positions.get(symbol)
        if existing is not None:
            if str(result.fill_mode or "") == "account_reconciled":
                combined_quantity = resolved_quantity
                combined_price = resolved_price
            else:
                combined_quantity = existing.quantity + resolved_quantity
                combined_price = int(
                    round(
                        (
                            existing.buy_price * existing.quantity
                            + resolved_price * resolved_quantity
                        )
                        / max(1, combined_quantity)
                    )
                )
            existing.buy_price = combined_price
            existing.quantity = combined_quantity
            existing.plan = self._reprice_plan_for_fill(
                plan=plan,
                quote=quote,
                fill_price=combined_price,
                quantity=combined_quantity,
            )
            position = existing
        else:
            position = PositionState(
                symbol=symbol,
                buy_price=resolved_price,
                quantity=resolved_quantity,
                buy_time=self._now(),
                plan=repriced,
                forecast_id=str(meta.get("forecast_id") or ""),
            )
            self.positions[symbol] = position
        self._pending_entry_meta.pop(symbol, None)
        self._save_daily_state_if_due(force=True)
        logger.info(
            "매수 체결 확정: %s %d주 @ %d원 EV=%+.0f원 expiry=%s",
            symbol,
            resolved_quantity,
            resolved_price,
            position.plan.expected_net if position.plan is not None else 0.0,
            position.plan.expires_at.strftime("%H:%M:%S") if position.plan is not None else "-",
        )
        if str(result.fill_mode or "") == "account_reconciled":
            order_logger.info(
                "매수 체결: buy %s %d주 @ 체결가 %d원 (계좌 재동기화)",
                symbol,
                resolved_quantity,
                resolved_price,
            )
        return position

    def on_order_filled(self, result: OrderResult) -> None:
        symbol = str(result.symbol or "").strip()
        side = result.side
        if not symbol or side is None:
            return
        if not result.success:
            if side == OrderSide.BUY:
                self._pending_entry_meta.pop(symbol, None)
            elif side == OrderSide.SELL:
                self._pending_sell_fills.pop(symbol, None)
                position = self.positions.get(symbol)
                if position is not None:
                    position.pending_exit_started_at = None
                    position.pending_exit_quantity = 0
            self._save_daily_state_if_due(force=True)
            return

        if side == OrderSide.BUY:
            meta = self._pending_entry_meta.get(symbol)
            if meta is not None:
                meta["order_no"] = str(result.order_no or "")
            price = max(0, int(result.price or result.requested_price or 0))
            quantity = max(0, int(result.quantity or 0))
            if quantity > 0 and price > 0:
                self._apply_buy_fill(result, quantity=quantity, price=price)
            else:
                self._save_daily_state_if_due(force=True)
            return

        pending = self._pending_sell_fills.get(symbol)
        position = self.positions.get(symbol)
        if pending is not None:
            pending["order_no"] = str(result.order_no or "")
        requested = max(0, int(result.requested_quantity or 0))
        quantity = max(0, int(result.quantity or 0))
        price = max(0, int(result.price or 0))
        if position is not None and quantity > 0 and price > 0:
            complete = requested <= 0 or quantity >= requested
            self._append_confirmed_sell_fill(
                result=result,
                position=self._copy_position(position),
                quantity=quantity,
                sell_price=price,
                apply_to_position=True,
                clear_pending=complete,
                fill_id=(
                    str(result.order_no or "")
                    if complete
                    else f"{str(result.order_no or symbol)}:{quantity}"
                ),
            )
            if not complete and pending is not None:
                pending["confirmed_quantity"] = quantity
                pending["confirmed_amount"] = quantity * price
                current = self.positions.get(symbol)
                pending["account_remaining_quantity"] = max(
                    0,
                    int(current.quantity if current is not None else requested - quantity),
                )
                self._save_daily_state_if_due(force=True)
        else:
            self._save_daily_state_if_due(force=True)

    def sync_positions_from_account(self, account_positions: List[Position]) -> None:
        account_map = {
            str(position.symbol): position
            for position in account_positions
            if max(0, int(position.quantity or 0)) > 0
        }
        now = self._now()

        for symbol, meta in list(self._pending_entry_meta.items()):
            account_position = account_map.get(symbol)
            if account_position is None:
                created_at = meta.get("created_at")
                if isinstance(created_at, datetime) and (
                    now - created_at
                ).total_seconds() > max(30, int(self.config.pending_order_block_seconds)):
                    self._pending_entry_meta.pop(symbol, None)
                continue

        for symbol, position in list(self.positions.items()):
            account_position = account_map.get(symbol)
            if account_position is None:
                if position.pending_exit_started_at is not None:
                    pending = self._pending_sell_fills.get(symbol)
                    if pending is not None:
                        pending["account_remaining_quantity"] = 0
                else:
                    self._pending_sell_fills[symbol] = {
                        "position": self._copy_position(position),
                        "order_no": "",
                        "requested_quantity": position.quantity,
                        "requested_at": now,
                        "reference_price": position.buy_price,
                        "reason": "external_or_reconciled_sell",
                        "account_remaining_quantity": 0,
                    }
                self.positions.pop(symbol, None)
                continue
            account_quantity = max(0, int(account_position.quantity or 0))
            account_price = max(1, int(round(float(account_position.avg_price or 0))))
            if position.pending_exit_started_at is not None:
                pending = self._pending_sell_fills.get(symbol)
                if pending is not None:
                    pending["account_remaining_quantity"] = account_quantity
                    position.pending_exit_quantity = account_quantity
            if account_quantity != position.quantity or account_price != position.buy_price:
                position.quantity = account_quantity
                position.buy_price = account_price
                quote = self._quotes_cache.get(symbol)
                if position.plan is not None and quote is not None:
                    position.plan = self._reprice_plan_for_fill(
                        plan=position.plan,
                        quote=quote,
                        fill_price=account_price,
                        quantity=account_quantity,
                    )

        for symbol, account_position in account_map.items():
            if symbol in self.positions:
                continue
            if symbol in self._pending_entry_meta:
                continue
            buy_price = max(1, int(round(float(account_position.avg_price or 0))))
            self.positions[symbol] = PositionState(
                symbol=symbol,
                buy_price=buy_price,
                quantity=max(1, int(account_position.quantity or 0)),
                buy_time=now,
                plan=None,
            )
        self._save_daily_state_if_due(force=True)

    def reconcile_pending_fills_from_account(
        self,
        results: List[OrderResult],
        account_positions: List[Position],
    ) -> List[OrderResult]:
        account_map = {str(position.symbol): position for position in account_positions}
        inferred: List[OrderResult] = []
        for result in results:
            symbol = str(result.symbol or "")
            account_position = account_map.get(symbol)
            if result.side == OrderSide.BUY and symbol in self._pending_entry_meta and account_position is not None:
                inferred_result = OrderResult(
                    success=True,
                    order_no=str(result.order_no or ""),
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=max(0, int(account_position.quantity or 0)),
                    price=max(1, int(round(float(account_position.avg_price or 0)))),
                    requested_quantity=max(0, int(result.requested_quantity or 0)),
                    fill_mode="account_reconciled",
                    requested_reason="expected_value",
                )
                applied = self._apply_buy_fill(
                    inferred_result,
                    quantity=inferred_result.quantity,
                    price=inferred_result.price,
                )
                if applied is not None:
                    inferred.append(inferred_result)
            elif result.side == OrderSide.SELL and symbol in self._pending_sell_fills:
                self._pending_sell_fills[symbol]["account_remaining_quantity"] = (
                    max(0, int(account_position.quantity or 0))
                    if account_position is not None
                    else 0
                )
        return inferred

    def reconcile_no_holding_sell_failures_from_account(
        self,
        results: List[OrderResult],
        account_positions: List[Position],
    ) -> List[OrderResult]:
        account_symbols = {
            str(position.symbol)
            for position in account_positions
            if max(0, int(position.quantity or 0)) > 0
        }
        for result in results:
            symbol = str(result.symbol or "")
            if (
                result.side == OrderSide.SELL
                and str(result.error_category or "") == "no_holding"
                and symbol not in account_symbols
                and symbol in self._pending_sell_fills
            ):
                self._pending_sell_fills[symbol]["account_remaining_quantity"] = 0
        return []

    def _sell_history_match(
        self,
        *,
        symbol: str,
        pending: Mapping[str, Any],
        rows: Sequence[Mapping[str, Any]],
        account_api,
    ) -> Optional[Mapping[str, Any]]:
        order_no = str(pending.get("order_no") or "").strip()
        symbol_rows = [
            row
            for row in rows
            if str(row.get("pdno") or "").strip() == symbol
        ]
        if order_no:
            matches = [
                row
                for row in symbol_rows
                if str(row.get("odno") or "").strip() == order_no
            ]
            if not matches:
                return None
            return max(
                matches,
                key=lambda row: (
                    account_api._coerce_int(row.get("tot_ccld_qty")) or 0,
                    account_api._coerce_int(row.get("tot_ccld_amt")) or 0,
                    str(row.get("ord_tmd") or ""),
                ),
            )

        if pending.get("account_remaining_quantity") != 0:
            return None
        snapshot = pending.get("position")
        requested_at = pending.get("requested_at")
        if not isinstance(snapshot, PositionState) or not isinstance(requested_at, datetime):
            return None
        lower_bound = max(
            snapshot.buy_time,
            requested_at
            - timedelta(seconds=max(30, int(self.config.pending_order_block_seconds))),
        )
        upper_bound = self._now() + timedelta(seconds=5)
        requested_quantity = max(0, int(pending.get("requested_quantity", 0) or 0))
        used_order_nos = {
            str(item.get("order_no") or "").strip()
            for item in self._sell_fill_ledger
            if str(item.get("order_no") or "").strip()
        }
        dated_matches = []
        for row in symbol_rows:
            row_order_no = str(row.get("odno") or "").strip()
            if not row_order_no or row_order_no in used_order_nos:
                continue
            quantity = account_api._coerce_int(row.get("tot_ccld_qty")) or 0
            if quantity != requested_quantity:
                continue
            order_date = "".join(character for character in str(row.get("ord_dt") or "") if character.isdigit())
            order_time = "".join(character for character in str(row.get("ord_tmd") or "") if character.isdigit())
            if len(order_date) != 8 or len(order_time) < 6:
                continue
            try:
                filled_at = datetime.strptime(order_date + order_time[:6], "%Y%m%d%H%M%S")
            except ValueError:
                continue
            if lower_bound <= filled_at <= upper_bound:
                dated_matches.append((filled_at, row))
        if not dated_matches:
            return None
        return max(dated_matches, key=lambda item: item[0])[1]

    def confirm_reconciled_sell_fills(
        self,
        account_api,
        *,
        results: Optional[List[OrderResult]] = None,
    ) -> List[Dict[str, Any]]:
        if not self._pending_sell_fills:
            return []
        history = account_api.get_order_history(self._today(), self._today(), side="01")
        if history.empty:
            return []
        rows = [item for item in history.to_dict("records") if isinstance(item, dict)]
        corrections: List[Dict[str, Any]] = []
        for symbol, pending in list(self._pending_sell_fills.items()):
            requested_quantity = max(0, int(pending.get("requested_quantity", 0) or 0))
            if requested_quantity <= 0:
                continue
            row = self._sell_history_match(
                symbol=symbol,
                pending=pending,
                rows=rows,
                account_api=account_api,
            )
            if row is None:
                continue
            order_no = str(pending.get("order_no") or "").strip()
            resolved_order_no = str(row.get("odno") or order_no).strip()
            quantity = account_api._coerce_int(row.get("tot_ccld_qty")) or 0
            price = account_api._coerce_int(row.get("avg_prvs")) or 0
            amount = account_api._coerce_int(row.get("tot_ccld_amt")) or 0
            if quantity > 0 and price <= 0 and amount > 0:
                price = int(round(amount / quantity))
            account_remaining = pending.get("account_remaining_quantity")
            cumulative_quantity = min(requested_quantity, max(0, int(quantity)))
            confirmed_quantity = min(
                cumulative_quantity,
                max(0, int(pending.get("confirmed_quantity", 0) or 0)),
            )
            incremental_quantity = cumulative_quantity - confirmed_quantity
            if incremental_quantity <= 0:
                continue
            confirmed_amount = max(0, int(pending.get("confirmed_amount", 0) or 0))
            if amount > confirmed_amount:
                incremental_amount = amount - confirmed_amount
                incremental_price = int(round(incremental_amount / incremental_quantity))
            else:
                incremental_amount = incremental_quantity * price
                incremental_price = price
            if incremental_price <= 0:
                continue
            snapshot = pending.get("position")
            if not isinstance(snapshot, PositionState):
                continue
            complete = account_remaining == 0 or cumulative_quantity >= requested_quantity
            fill_id = (
                resolved_order_no
                if complete and confirmed_quantity == 0
                else f"{resolved_order_no or symbol}:{cumulative_quantity}"
            )
            confirmed_result = OrderResult(
                success=True,
                order_no=resolved_order_no,
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=incremental_quantity,
                price=incremental_price,
                requested_quantity=requested_quantity,
                fill_mode=(
                    "account_reconciled_confirmed"
                    if complete
                    else "account_reconciled_partial"
                ),
                requested_reason=str(pending.get("reason") or ""),
                fill_id=fill_id,
                remaining_quantity=max(0, requested_quantity - cumulative_quantity),
            )
            record = self._append_confirmed_sell_fill(
                result=confirmed_result,
                position=snapshot,
                quantity=incremental_quantity,
                sell_price=incremental_price,
                apply_to_position=False,
                clear_pending=complete,
                fill_id=fill_id,
            )
            if record is None:
                continue
            if not complete:
                pending["confirmed_quantity"] = cumulative_quantity
                pending["confirmed_amount"] = max(
                    confirmed_amount + incremental_amount,
                    amount,
                )
                self._save_daily_state_if_due(force=True)
            order_records = [
                item
                for item in self._sell_fill_ledger
                if str(item.get("order_no") or "") == resolved_order_no
            ]
            cumulative_net_pnl = sum(
                int(item.get("net_pnl", 0) or 0)
                for item in order_records
            )
            alert_price = (
                int(round(amount / cumulative_quantity))
                if amount > 0 and cumulative_quantity > 0
                else price
            )
            corrections.append(
                {
                    "order_no": resolved_order_no,
                    "symbol": symbol,
                    "quantity": incremental_quantity,
                    "previous_price": int(pending.get("reference_price", 0) or 0),
                    "corrected_price": incremental_price,
                    "delta_net_pnl": int(record.get("net_pnl", 0) or 0),
                    "net_pnl": int(record.get("net_pnl", 0) or 0),
                    "fill_id": fill_id,
                    "remaining_quantity": max(0, requested_quantity - cumulative_quantity),
                    "partial": not complete,
                    "notify": confirmed_quantity == 0 or complete,
                    "alert_quantity": cumulative_quantity,
                    "alert_price": alert_price,
                    "alert_net_pnl": cumulative_net_pnl,
                    "buy_price": int(record.get("buy_price", 0) or 0),
                    "requested_reason": str(record.get("requested_reason") or ""),
                }
            )
        return corrections


__all__ = [
    "MomentumScalpConfig",
    "MomentumScalpStrategy",
    "PositionState",
    "TradePlan",
]
