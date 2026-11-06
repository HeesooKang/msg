from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict, deque
from dataclasses import fields, replace
from datetime import datetime, timedelta
from math import ceil, floor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from src.analytics.quote_archive import QuoteArchive
from src.models import Order, OrderResult, OrderSide, OrderType, Position, Quote
from src.instruments import is_stock_symbol, tick_size, transaction_tax_rate
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

STATE_SCHEMA_VERSION = 5
BREAKOUT_LOOKBACK_SECONDS = 60
MAX_HOLD_SECONDS = 180
KRX_CLOSING_AUCTION_START = (15, 20)


class MomentumScalpStrategy(BaseStrategy):
    """One breakout entry with a frozen drawdown stop, target and expiry."""

    def __init__(
        self,
        market_data,
        config: MomentumScalpConfig,
        fixed_pool: Optional[List[str]] = None,
    ):
        self.market_data = market_data
        groups = getattr(market_data, "instrument_groups", {})
        self.instrument_groups = groups if isinstance(groups, dict) else {}
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
        self._last_submitted_signal_at: Dict[str, datetime] = {}
        self._last_batch_summary_at: Optional[datetime] = None

        self._quote_archive = QuoteArchive(self.config.quote_archive_root)
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
            if not is_stock_symbol(value) or value in seen:
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
        self._last_submitted_signal_at = {}
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
            "가격 돌파 전략 초기화: schema=%d 고정=%d 동적=%d 보유=%d 누적실현=%+d원 세션실현=%+d원 손실한도=%+d원",
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
        return value.isoformat() if value is not None else ""

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
            "breakout_price": plan.breakout_price,
            "committed_risk_net_abs": plan.committed_risk_net_abs,
            "signal_at": self._serialize_datetime(plan.signal_at),
            "expires_at": self._serialize_datetime(plan.expires_at),
            "target_price": plan.target_price,
            "stop_price": plan.stop_price,
            "exit_price_allowance": plan.exit_price_allowance,
        }

    def _deserialize_plan(
        self,
        payload: Mapping[str, Any] | None,
        *,
        fill_price: Optional[int] = None,
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
                # A legacy forecast is not a price-breakout plan. Keep its
                # order identity for reconciliation, then liquidate on a quote.
                breakout_price=max(0, int(payload.get("breakout_price", 0))),
                committed_risk_net_abs=int(payload["committed_risk_net_abs"]),
                signal_at=signal_at,
                expires_at=expires_at,
                target_price=max(0, int(payload.get("target_price", 0))),
                stop_price=max(0, int(payload.get("stop_price", 0))),
                exit_price_allowance=max(0, int(payload.get("exit_price_allowance", 0))),
            )
        except (KeyError, TypeError, ValueError):
            return None
        if not plan.symbol or plan.quantity <= 0:
            return None
        if "target_price" not in payload and plan.breakout_price > 0:
            # Preserve the old risk-sized profit exit only when restoring an old plan.
            rate = 1 - self.config.commission_rate - self._sell_tax_rate_for_symbol(plan.symbol)
            if rate <= 0:
                return None
            price = ((plan.entry_limit_price if fill_price is None else fill_price)
                     * (1 + self.config.commission_rate)
                     + plan.committed_risk_net_abs / plan.quantity) / rate
            tick = tick_size(price, self.instrument_groups.get(plan.symbol, ""))
            plan = replace(plan, target_price=int(ceil(price / tick) * tick))
        return plan

    def _serialize_position(self, position: PositionState) -> Dict[str, Any]:
        return {
            "symbol": position.symbol,
            "buy_price": position.buy_price,
            "quantity": position.quantity,
            "buy_time": self._serialize_datetime(position.buy_time),
            "plan": self._serialize_plan(position.plan),
            "trade_key": position.trade_key,
            "pending_exit_started_at": self._serialize_datetime(
                position.pending_exit_started_at
            ),
            "pending_exit_quantity": position.pending_exit_quantity,
            "pending_exit_reason": position.pending_exit_reason,
            "stop_triggered": position.stop_triggered,
        }

    def _deserialize_position(self, payload: Mapping[str, Any]) -> Optional[PositionState]:
        symbol = str(payload.get("symbol") or "").strip()
        quantity = max(0, int(payload.get("quantity", 0) or 0))
        buy_price = max(0, int(float(payload.get("buy_price", 0) or 0)))
        if len(symbol) != 6 or quantity <= 0 or buy_price <= 0:
            return None
        buy_time = self._deserialize_datetime(payload.get("buy_time")) or self._now()
        plan = self._deserialize_plan(payload.get("plan"), fill_price=buy_price)
        if plan is not None and (plan.symbol != symbol or plan.quantity != quantity):
            plan = None
        return PositionState(
            symbol=symbol,
            buy_price=buy_price,
            quantity=quantity,
            buy_time=buy_time,
            plan=plan,
            trade_key=str(payload.get("trade_key") or ""),
            pending_exit_started_at=self._deserialize_datetime(
                payload.get("pending_exit_started_at")
            ),
            pending_exit_quantity=max(
                0, int(payload.get("pending_exit_quantity", 0) or 0)
            ),
            pending_exit_reason=str(payload.get("pending_exit_reason") or ""),
            stop_triggered=bool(payload.get("stop_triggered", False)),
        )

    def _serialize_pending_entry(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "plan": self._serialize_plan(payload.get("plan")),
            "signal_price": int(payload.get("signal_price", 0) or 0),
            "created_at": self._serialize_datetime(payload.get("created_at")),
            "order_no": str(payload.get("order_no") or ""),
            "confirmed_quantity": int(payload.get("confirmed_quantity", 0) or 0),
            "confirmed_amount": int(payload.get("confirmed_amount", 0) or 0),
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
            "signal_price": int(payload.get("signal_price", 0) or 0),
            "created_at": created_at,
            "order_no": str(payload.get("order_no") or ""),
            "confirmed_quantity": int(payload.get("confirmed_quantity", 0) or 0),
            "confirmed_amount": int(payload.get("confirmed_amount", 0) or 0),
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
                position = self.positions.get(str(symbol))
                if "confirmed_quantity" not in item and position is not None:
                    restored["confirmed_quantity"] = position.quantity
                    restored["confirmed_amount"] = position.quantity * position.buy_price
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
                "확정 매도체결 원장 비용 재계산: %d건 수수료율=%.6f%% 종목별 매도비용 적용",
                repriced_fills,
                float(self.config.commission_rate) * 100.0,
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
            sell_tax_rate=self._sell_tax_rate_for_symbol(
                str(fill.get("symbol") or "")
            ),
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

    def _append_confirmed_sell_fill(
        self,
        *,
        result: OrderResult,
        position: PositionState,
        quantity: int,
        sell_price: int,
        remaining_quantity: Optional[int] = None,
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
            sell_tax_rate=self._sell_tax_rate_for_symbol(position.symbol),
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
            "order_complete": clear_pending,
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
            "breakout_price": position.plan.breakout_price if position.plan else 0,
            "planned_risk": position.plan.committed_risk_net_abs if position.plan else 0,
            "target_price": position.plan.target_price if position.plan else 0,
            "stop_price": (position.plan.stop_price or position.plan.breakout_price) if position.plan else 0,
        }
        self._sell_fill_ledger.append(record)

        current = self.positions.get(position.symbol)
        if current is not None and current.trade_key == position.trade_key:
            # History is cumulative; balance may already include some or all
            # of the fill. Never subtract the same sold shares twice.
            current.quantity = (
                max(0, current.quantity - resolved_quantity)
                if remaining_quantity is None
                else min(current.quantity, max(0, int(remaining_quantity)))
            )
            if clear_pending:
                current.pending_exit_started_at = None
                current.pending_exit_quantity = 0
                current.pending_exit_reason = ""
            else:
                current.pending_exit_quantity = current.quantity
            if current.quantity <= 0:
                self.positions.pop(current.symbol, None)
            elif current.plan is not None and current.plan.quantity > 0:
                current.plan = replace(
                    current.plan,
                    quantity=current.quantity,
                    committed_risk_net_abs=int(ceil(
                        current.plan.committed_risk_net_abs
                        * current.quantity / current.plan.quantity
                    )),
                )

        if clear_pending:
            self._pending_sell_fills.pop(position.symbol, None)
        self._rebuild_daily_pnl()
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
            window = deque()
            self._recent_quotes[symbol] = window
        return window

    def _ingest_quotes(self, quotes: Sequence[Quote]) -> None:
        now = self._now()
        # Retain the anchor before the lookback plus the latest quote's allowed
        # delivery lag. Neither allowance extends executable quote freshness.
        history_seconds = BREAKOUT_LOOKBACK_SECONDS + 2 * self.config.quote_freshness_seconds
        oldest_allowed = now - timedelta(seconds=history_seconds)
        newest_allowed = now
        for incoming in sorted(quotes, key=lambda item: (item.timestamp, item.symbol)):
            quote = replace(incoming)
            symbol = str(quote.symbol or "").strip()
            if (
                len(symbol) != 6
                or int(quote.current_price or 0) <= 0
                or quote.timestamp.date() != now.date()
                or quote.timestamp < oldest_allowed
                or quote.timestamp > newest_allowed
            ):
                continue
            previous = self._quotes_cache.get(symbol)
            if previous is not None:
                if quote.timestamp < previous.timestamp:
                    previous.observe_price_peaks(quote)
                else:
                    quote.observe_price_peaks(previous)
            if previous is None or quote.timestamp >= previous.timestamp:
                self._quotes_cache[symbol] = quote
            window = self._recent_window(symbol)
            while window and window[0].timestamp < oldest_allowed:
                window.popleft()
            if not window or window[-1].timestamp < quote.timestamp:
                window.append(quote)
                continue
            # A late REST/stream observation belongs in history, not in the
            # current execution book. Never manufacture the missing timestamps.
            for index in range(len(window) - 1, -1, -1):
                if window[index].timestamp == quote.timestamp:
                    quote.observe_price_peaks(window[index])
                    window[index] = quote
                    break
                if window[index].timestamp < quote.timestamp:
                    window.insert(index + 1, quote)
                    break
            else:
                window.appendleft(quote)

    def _fresh_quotes(self) -> List[Quote]:
        now = self._now()
        maximum_age = max(1, int(self.config.quote_freshness_seconds))
        fresh = []
        for quote in self._quotes_cache.values():
            age = (now - quote.timestamp).total_seconds()
            if age < 0.0 or age > maximum_age:
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

    def _observed_tick_size(self, quote: Quote) -> int:
        return tick_size(quote.current_price, self.instrument_groups.get(quote.symbol, quote.security_group))

    def _current_exit_price(self, quote: Quote) -> int:
        current = max(1, int(quote.current_price or 0))
        bid = max(0, int(quote.bid_price or 0))
        return bid if bid > 0 else max(1, current - self._observed_tick_size(quote))

    def _sell_tax_rate_for_symbol(self, symbol: str) -> float:
        return transaction_tax_rate(self.instrument_groups.get(symbol, ""), self.config.sell_tax_rate)

    def _trade_pnl(
        self,
        symbol: str,
        entry_price: int,
        exit_price: int,
        quantity: int,
    ) -> TradePnL:
        return calculate_trade_pnl_from_prices(
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            commission_rate=self.config.commission_rate,
            sell_tax_rate=self._sell_tax_rate_for_symbol(symbol),
        )

    # ------------------------------------------------------------------
    # Observed breakout and cost-based position risk
    # ------------------------------------------------------------------
    def _open_capital(self) -> int:
        return sum(
            max(0, int(position.buy_price or 0) * int(position.quantity or 0))
            for position in self.positions.values()
        )

    def _pending_entry_capital(self) -> int:
        return sum(
            max(0, int(item["plan"].entry_limit_price) *
                (int(item["plan"].quantity) - int(item.get("confirmed_quantity", 0))))
            for item in self._pending_entry_meta.values()
            if isinstance(item.get("plan"), TradePlan)
        )

    def _open_committed_risk(self) -> int:
        total = 0
        hard_limit = abs(int(self.config.daily_loss_limit))
        for position in self.positions.values():
            if position.plan is None or position.plan.breakout_price <= 0:
                total += hard_limit
            else:
                total += max(0, int(position.plan.committed_risk_net_abs))
        for symbol, payload in self._pending_sell_fills.items():
            if symbol in self.positions:
                continue
            snapshot = payload.get("position")
            if isinstance(snapshot, PositionState):
                total += (max(0, snapshot.plan.committed_risk_net_abs)
                          if snapshot.plan is not None and snapshot.plan.breakout_price > 0 else hard_limit)
        return total

    def _pending_entry_risk(self) -> int:
        return sum(
            max(0, int(ceil(item["plan"].committed_risk_net_abs *
                max(0, item["plan"].quantity - int(item.get("confirmed_quantity", 0))) /
                max(1, item["plan"].quantity))))
            for item in self._pending_entry_meta.values()
            if isinstance(item.get("plan"), TradePlan)
        )

    def _available_capital(self) -> int:
        return max(
            0,
            int(self.config.seed_money)
            - self._open_capital()
            - self._pending_entry_capital(),
        )

    def _remaining_daily_loss_room(self) -> int:
        return max(
            0,
            abs(int(self.config.daily_loss_limit))
            + self._realized_net_pnl_for_loss_guard()
            - self._open_committed_risk()
            - self._pending_entry_risk(),
        )

    def _breakout_window(self, quote: Quote) -> List[Quote]:
        cutoff = quote.timestamp - timedelta(seconds=BREAKOUT_LOOKBACK_SECONDS)
        history = [q for q in self._recent_window(quote.symbol)
                   if q.timestamp < quote.timestamp and q.has_valid_prices
                   and (q.ask_high_price > 0 or
                        (q.book_available and 0 < q.bid_price <= q.ask_price))]
        anchors = [q for q in history if q.timestamp <= cutoff]
        if not anchors or (cutoff - anchors[-1].timestamp).total_seconds() > self.config.quote_freshness_seconds:
            return []
        window = [q for q in history if q.timestamp >= anchors[-1].timestamp]
        # A fresh last quote cannot fill a missing interval in the lookback.
        if any((right.timestamp - left.timestamp).total_seconds() > self.config.quote_freshness_seconds
               for left, right in zip(window, window[1:] + [quote])):
            return []
        return window

    def _exit_price_allowance(self, quote: Quote, stop_price: int,
                              *, window: Optional[Sequence[Quote]] = None) -> int:
        tick = tick_size(stop_price, self.instrument_groups.get(quote.symbol, quote.security_group))
        cutoff = quote.timestamp - timedelta(seconds=BREAKOUT_LOOKBACK_SECONDS)
        window = [q for q in (self._recent_window(quote.symbol) if window is None else window)
                  if cutoff <= q.timestamp < quote.timestamp and q.has_valid_prices
                  and q.book_available and 0 < q.bid_price <= q.ask_price] + [quote]
        # A tight signal book must not erase recent spreads or adverse bid steps.
        # This remains a planning allowance, not a guaranteed market fill.
        adverse_step = max((left.bid_price - right.bid_price
                            for left, right in zip(window, window[1:])
                            if 0 < (right.timestamp - left.timestamp).total_seconds()
                            <= self.config.quote_freshness_seconds), default=0)
        observed_spread = max(q.ask_price - q.bid_price for q in window)
        return max(tick, observed_spread, adverse_step)

    def _risk_per_share(self, quote: Quote, entry_price: int, stop_price: int,
                        *, window: Optional[Sequence[Quote]] = None,
                        exit_allowance: Optional[int] = None) -> float:
        tick = tick_size(stop_price, self.instrument_groups.get(quote.symbol, quote.security_group))
        allowance = (self._exit_price_allowance(quote, stop_price, window=window)
                     if exit_allowance is None else exit_allowance)
        stop_fill = max(tick, int(floor((stop_price - allowance) / tick) * tick))
        return max(0.0, -estimate_trade_net_pnl_unrounded(
            entry_price=entry_price, exit_price=min(stop_fill, self._current_exit_price(quote)),
            quantity=1, commission_rate=self.config.commission_rate,
            sell_tax_rate=self._sell_tax_rate_for_symbol(quote.symbol)))

    def _target_net_pnl(self, plan: TradePlan) -> int:
        # A target bid is a sell trigger, not a guaranteed market execution.
        price = max(1, plan.target_price - plan.exit_price_allowance)
        tick = tick_size(price, self.instrument_groups.get(plan.symbol, ""))
        price = max(tick, int(price // tick * tick))
        return self._trade_pnl(plan.symbol, plan.entry_limit_price, price, plan.quantity).net_pnl

    def _build_trade_plan(self, quote: Quote, *, now: datetime) -> TradePlan:
        window = self._breakout_window(quote)
        # Keep observed trade/ask peaks for the signal, executable bids for the target.
        level = max((max(q.current_price, q.trade_high_price, q.ask_high_price,
                         q.ask_price if q.book_available and 0 < q.bid_price <= q.ask_price else 0)
                     for q in window), default=0)
        entry = max(0, int(quote.ask_price or 0))
        plan = TradePlan(quote.symbol, False, "no_breakout", 0, entry, level, 0,
                         quote.timestamp, quote.timestamp + timedelta(seconds=MAX_HOLD_SECONDS))
        if not 0 <= (now - quote.timestamp).total_seconds() <= self.config.quote_freshness_seconds:
            return replace(plan, reason="stale_quote")
        if (not quote.book_available or not 0 < quote.bid_price <= quote.ask_price
                or (quote.book_depth_available and min(quote.ask_size, quote.bid_size) <= 0)):
            return replace(plan, reason="no_executable_book")
        if entry * (1 + self.config.commission_rate) > self._available_capital():
            return replace(plan, reason="unaffordable_one_share")
        if not level:
            return replace(plan, reason="price_window_incomplete")
        if not quote.is_price_breakout(level):
            return plan
        books = [q for q in window if q.book_available and 0 < q.bid_price <= q.ask_price]
        cutoff = quote.timestamp - timedelta(seconds=BREAKOUT_LOOKBACK_SECONDS)
        weighted_bid = duration = 0.0
        for left, right in zip(books, books[1:] + [quote]):
            if (right.timestamp - left.timestamp).total_seconds() > self.config.quote_freshness_seconds:
                return replace(plan, reason="price_window_incomplete")
            seconds = max(0.0, (right.timestamp - max(cutoff, left.timestamp)).total_seconds())
            weighted_bid += left.bid_price * seconds
            duration += seconds
        if not books or books[0].timestamp > cutoff or duration <= 0:
            return replace(plan, reason="price_window_incomplete")
        # Project one displacement from the time-weighted price, not the full high/low range.
        # The current spike is excluded; an isolated low has only its observed time weight.
        target = max(0.0, 2 * books[-1].bid_price - weighted_bid / duration)
        tick = tick_size(target, self.instrument_groups.get(quote.symbol, quote.security_group))
        target = int(target // tick * tick)
        # Separate observed pullbacks from the completed rise; freeze this exit at entry.
        # The drawdown is a stop policy, not a bound on future market losses.
        peak = drawdown = 0
        for book in books:
            peak = max(peak, book.bid_price)
            drawdown = max(drawdown, peak - book.bid_price)
        group = self.instrument_groups.get(quote.symbol, quote.security_group)
        stop = quote.bid_price - max(drawdown, tick_size(quote.bid_price, group))
        stop_tick = tick_size(stop, group)
        stop = max(stop_tick, int(stop // stop_tick * stop_tick))
        allowance = self._exit_price_allowance(quote, stop, window=window)
        plan = replace(plan, target_price=target, stop_price=stop, exit_price_allowance=allowance)
        risk = self._risk_per_share(quote, entry, stop, window=window, exit_allowance=allowance)
        capital = int(self._available_capital() // (entry * (1 + self.config.commission_rate)))
        loss_room = self._remaining_daily_loss_room()
        # Size the limit order by capital and risk; confirmed fills determine inventory.
        quantity = min(capital, int(loss_room / risk)) if risk > 0 else capital
        if quantity <= 0:
            return replace(plan, reason="insufficient_capital_or_loss_room")
        # Round risk up, rather than rounding a one-share fee down.
        committed = int(ceil(risk * quantity))
        if committed > loss_room:
            return replace(plan, reason="risk_exceeds_remaining_loss_room")
        plan = replace(plan, quantity=quantity, committed_risk_net_abs=committed)
        # Judge the whole risk-sized order, not a positive one-share projection.
        target_net = self._target_net_pnl(plan)
        if target_net <= 0:
            return replace(plan, reason="target_not_profitable")
        return replace(plan, allowed=True, reason="price_breakout")

    def _reprice_plan_for_fill(
        self, *, plan: TradePlan, fill_price: int, quantity: int,
        previous_entry_price: Optional[int] = None,
    ) -> TradePlan:
        if plan.breakout_price <= 0:
            return replace(plan, allowed=False, reason="legacy_plan", quantity=quantity)
        entry_basis = plan.entry_limit_price if previous_entry_price is None else previous_entry_price
        # Keep the signal's exit allowance. Later spreads must not move risk or the profit target.
        risk = (plan.committed_risk_net_abs / max(1, plan.quantity)
                + (fill_price - entry_basis) * (1 + self.config.commission_rate))
        return replace(plan, allowed=fill_price <= plan.entry_limit_price,
                       reason="confirmed_fill", quantity=quantity,
                       committed_risk_net_abs=max(0, int(ceil(risk * quantity))))

    def _adopt_unplanned_position(
        self,
        position: PositionState,
        quote: Optional[Quote],
    ) -> Optional[Order]:
        return self._make_market_sell(
            position,
            quote,
            reason="unplanned_position",
        )

    def _position_exit_order(
        self,
        position: PositionState,
        quote: Optional[Quote],
        *,
        now: datetime,
        observed_quotes: Sequence[Quote] = (),
    ) -> Optional[Order]:
        if position.pending_exit_started_at is not None:
            return None
        plan = position.plan
        if plan is None:
            return None
        if plan.breakout_price <= 0 or not plan.allowed:
            return self._make_market_sell(position, quote, reason="invalid_entry_plan")
        stop = plan.stop_price or plan.breakout_price
        # A rebound later in the same batch must not erase an observed stop.
        # Old/pre-fill observations cannot trigger a new holding's stop.
        observed_stop = any(
            item.symbol == position.symbol and item.book_available
            and 0 < item.bid_price <= min(item.ask_price, stop)
            and position.buy_time <= item.timestamp
            and 0 <= (now - item.timestamp).total_seconds() <= self.config.quote_freshness_seconds
            for item in observed_quotes
        )
        if (position.stop_triggered or observed_stop
                or (quote is not None and quote.book_available and 0 < quote.bid_price <= stop)):
            # Preserve the exit decision across submission failures and restarts;
            # pending_exit_started_at still means an actually submitted order.
            position.stop_triggered = True
            return self._make_market_sell(position, quote, reason="breakout_failed")
        if (
            self.market_data is not None and self.market_data.client.config.is_paper
            and quote is not None and quote.book_available and quote.book_depth_available
            and 0 < quote.bid_price <= quote.ask_price
            and quote.bid_size >= position.quantity
            and 0 <= (now - quote.timestamp).total_seconds() <= self.config.quote_freshness_seconds
            and plan.target_price > 0 and quote.bid_price >= plan.target_price
            and self._trade_pnl(position.symbol, position.buy_price, quote.bid_price,
                                position.quantity).net_pnl > 0
        ):
            return self._make_market_sell(position, quote, reason="planned_profit")
        if now >= plan.expires_at:
            return self._make_market_sell(position, quote, reason="holding_expiry")
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
                symbol,
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
        if self._pending_entry_meta or self._pending_sell_fills:
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
    def _entry_plan_priority(self, plan: TradePlan) -> tuple:
        return (-self._target_net_pnl(plan), plan.committed_risk_net_abs, plan.symbol)

    def on_batch_tick(self, quotes: List[Quote]) -> List[Order]:
        self._ensure_active_day()
        now = self._now()
        quotes = [q for q in quotes if q.has_valid_prices and q.timestamp <= now]
        for quote in quotes:
            if quote.security_group:
                self.instrument_groups[quote.symbol] = quote.security_group
            else:
                quote.security_group = self.instrument_groups.get(quote.symbol, "")
        self._ingest_quotes(quotes)
        self._quote_archive.record(quotes, now=now, universe=self.get_watchlist())
        fresh = self._fresh_quotes()
        self._last_evaluation_count = 0
        self._update_daily_halt_state()
        if self._halted:
            orders = self._liquidate_all()
            self._save_daily_state_if_due(force=True)
            return orders

        by_symbol = {q.symbol: q for q in fresh}
        # A realizable daily target is an exit, not a realized-profit event.
        # Keep the session running until confirmed fills actually meet it.
        take_daily_profit = (
            bool(self.positions)
            and not self._pending_entry_meta
            and not self._pending_sell_fills
            and all(
                (quote := by_symbol.get(symbol)) is not None
                and quote.book_available and 0 < quote.bid_price <= quote.ask_price
                and quote.book_depth_available and quote.bid_size >= position.quantity
                for symbol, position in self.positions.items()
            )
            and self.daily_pnl.realized_net_pnl + self._unrealized_net_pnl()
            >= int(self.config.daily_profit_target)
        )
        exits = []
        for symbol, position in self.positions.items():
            quote = by_symbol.get(symbol)
            if position.plan is None:
                order = self._adopt_unplanned_position(position, quote)
            elif take_daily_profit:
                order = self._make_market_sell(position, quote, reason="daily_profit_target")
            else:
                order = self._position_exit_order(position, quote, now=now, observed_quotes=quotes)
            if order is not None:
                exits.append(order)
        if exits:
            self._save_daily_state_if_due(force=True)
            return exits

        if now + timedelta(seconds=MAX_HOLD_SECONDS) >= self._closing_auction_start(now):
            day = now.date().isoformat()
            if self._entry_window_closed_log_day != day:
                self._entry_window_closed_log_day = day
                logger.info("신규 진입 종료: 최대 보유 %d초가 15:20 동시호가를 넘습니다.", MAX_HOLD_SECONDS)
            self._save_daily_state_if_due()
            return []

        watchlist = set(self.get_watchlist())
        candidates = [(q, self._build_trade_plan(q, now=now)) for q in fresh
                      if q.symbol in watchlist and q.symbol not in self.positions]
        self._last_evaluation_count = len(candidates)
        # Use the same execution-adjusted proceeds as the plan's cost check.
        allowed = sorted(((q, p) for q, p in candidates if p.allowed),
                         key=lambda item: self._entry_plan_priority(item[1]))
        if self._last_batch_summary_at is None or (now - self._last_batch_summary_at).total_seconds() >= 30:
            reasons = Counter(p.reason for _, p in candidates)
            breakouts = sum(q.is_price_breakout(p.breakout_price) for q, p in candidates)
            logger.info("돌파 배치: 평가=%d 돌파=%d 계획=%d 보유=%d 손실여유=%d원 사유=%s",
                        len(candidates), breakouts, len(allowed), len(self.positions),
                        self._remaining_daily_loss_room(),
                        ",".join(f"{k}:{v}" for k, v in sorted(reasons.items())) or "-")
            self._last_batch_summary_at = now
        selected = None
        if self._entry_submission_available():
            fresh_symbols = {q.symbol for q in self._fresh_quotes()}
            selected = next(((q, p) for q, p in allowed
                             if q.symbol in fresh_symbols
                             and self._last_submitted_signal_at.get(q.symbol) != q.timestamp), None)
        if selected is None:
            self._save_daily_state_if_due()
            return []

        quote, plan = selected
        self._last_submitted_signal_at[plan.symbol] = plan.signal_at
        self._pending_entry_meta[plan.symbol] = dict(
            plan=plan, signal_price=quote.current_price, created_at=now, order_no="")
        self._save_daily_state_if_due(force=True)
        logger.info("돌파 매수 선택: %s %d주 @ %d원 돌파선=%d원 손절선=%d원 계획위험=%d원 만료=%s 체결시세=%d bid=%d 목표가=%d 목표순익=%d원 체결여유=%d원/주",
                    plan.symbol, plan.quantity, plan.entry_limit_price, plan.breakout_price,
                    plan.stop_price, plan.committed_risk_net_abs, plan.expires_at.strftime("%H:%M:%S"),
                    quote.current_price, quote.bid_price, plan.target_price,
                    self._target_net_pnl(plan), plan.exit_price_allowance)
        return [Order(symbol=plan.symbol, side=OrderSide.BUY, order_type=OrderType.LIMIT,
                      quantity=plan.quantity, price=plan.entry_limit_price,
                      reference_price=quote.current_price, requested_reason="price_breakout",
                      submit_before=quote.timestamp.timestamp() + self.config.quote_freshness_seconds,
                      breakout_price=plan.breakout_price)]

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

    def _track_submitted_sell(
        self,
        result: OrderResult,
        position: PositionState,
    ) -> Dict[str, Any]:
        existing = self._pending_sell_fills.get(position.symbol)
        if existing is not None:
            if result.order_no:
                existing["order_no"] = str(result.order_no)
            return existing

        requested_quantity = min(
            position.quantity,
            max(1, int(result.requested_quantity or position.quantity)),
        )
        requested_at = self._now()
        reason = str(result.requested_reason or "sell")
        reference_price = max(
            1,
            int(result.reference_price or result.requested_price or position.buy_price),
        )
        snapshot = self._copy_position(position)
        for target in (position, snapshot):
            target.pending_exit_started_at = requested_at
            target.pending_exit_quantity = requested_quantity
            target.pending_exit_reason = reason
        pending = {
            "position": snapshot,
            "order_no": str(result.order_no or ""),
            "requested_quantity": requested_quantity,
            "requested_at": requested_at,
            "reference_price": reference_price,
            "reason": reason,
            "account_remaining_quantity": position.quantity,
            "confirmed_quantity": 0,
            "confirmed_amount": 0,
        }
        self._pending_sell_fills[position.symbol] = pending
        return pending

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
        resolved_quantity = max(0, int(quantity or 0))
        resolved_price = max(0, int(price or 0))
        if resolved_quantity <= 0 or resolved_price <= 0:
            return None
        repriced = self._reprice_plan_for_fill(
            plan=plan,
            fill_price=resolved_price,
            quantity=resolved_quantity,
        )
        existing = self.positions.get(symbol)
        if existing is not None:
            combined_quantity = existing.quantity + resolved_quantity
            combined_price = int(
                round(
                    (existing.buy_price * existing.quantity + resolved_price * resolved_quantity)
                    / max(1, combined_quantity)
                )
            )
            existing.buy_price = combined_price
            existing.quantity = combined_quantity
            existing.plan = self._reprice_plan_for_fill(
                plan=plan,
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
            )
            self.positions[symbol] = position
        meta["confirmed_quantity"] = int(meta.get("confirmed_quantity", 0)) + resolved_quantity
        meta["confirmed_amount"] = int(meta.get("confirmed_amount", 0)) + resolved_price * resolved_quantity
        if result.fill_mode != "partial_fill_pending":
            self._pending_entry_meta.pop(symbol, None)
        self._save_daily_state_if_due(force=True)
        logger.info(
            "매수 체결 확정: %s %d주 @ %d원 돌파선=%d원 만료=%s",
            symbol,
            resolved_quantity,
            resolved_price,
            position.plan.breakout_price if position.plan is not None else 0,
            position.plan.expires_at.strftime("%H:%M:%S") if position.plan is not None else "-",
        )
        order_logger.info(
            "매수 체결: buy %s %d주 @ 체결가 %d원 order_no=%s",
            symbol, resolved_quantity, resolved_price, result.order_no or "-",
        )
        return position

    def on_order_filled(self, result: OrderResult) -> bool:
        symbol = str(result.symbol or "").strip()
        side = result.side
        if not symbol or side is None:
            return False
        pending = (self._pending_entry_meta.get(symbol) if side == OrderSide.BUY
                   else self._pending_sell_fills.get(symbol))
        active_order_no = str((pending or {}).get("order_no") or "")
        if active_order_no and result.order_no and active_order_no != str(result.order_no):
            logger.warning("다른 주문 결과 무시: %s %s order_no=%s active=%s",
                           side.value, symbol, result.order_no, active_order_no)
            return False
        if side == OrderSide.SELL and result.order_no and any(
            str(item.get("order_no") or "") == str(result.order_no)
            and bool(item.get("order_complete", item.get("fill_id") == result.order_no))
            for item in self._sell_fill_ledger
        ):
            return False
        if not result.success:
            if side == OrderSide.BUY:
                if pending is None:
                    return False
                self._pending_entry_meta.pop(symbol, None)
                if result.fill_mode == "limit_cancelled" and result.error_category == "not_filled":
                    order_logger.info(
                        "매수 미체결 취소: buy %s %d주 @ 주문가 %d원 order_no=%s",
                        symbol, result.requested_quantity, result.requested_price, result.order_no or "-",
                    )
            elif side == OrderSide.SELL:
                if result.error_category == "no_holding":
                    # No inventory is not proof of a failed sell: its execution may already exist.
                    position = self.positions.get(symbol)
                    if pending is None and position is not None:
                        self._track_submitted_sell(replace(result, order_no=""), position)
                    self._save_daily_state_if_due(force=True)
                    return True
                self._pending_sell_fills.pop(symbol, None)
                position = self.positions.get(symbol)
                if position is not None:
                    position.pending_exit_started_at = None
                    position.pending_exit_quantity = 0
                    position.pending_exit_reason = ""
            self._save_daily_state_if_due(force=True)
            return True

        if side == OrderSide.BUY:
            meta = self._pending_entry_meta.get(symbol)
            if meta is None:
                return False
            identity_changed = bool(result.order_no and str(result.order_no) != active_order_no)
            if result.order_no:
                meta["order_no"] = str(result.order_no)
            price = max(0, int(result.price or 0))
            cumulative_quantity = max(0, int(result.quantity or 0))
            if cumulative_quantity <= 0 or price <= 0:
                self._save_daily_state_if_due(force=True)
                return True
            previous_quantity = max(0, int(meta.get("confirmed_quantity", 0)))
            terminal = (result.fill_mode not in {"partial_fill_pending", "entry_reconciled_partial",
                                                 "account_reconciled"}
                        or cumulative_quantity >= meta["plan"].quantity)
            # History and account recovery both supply cumulative quantities and costs.
            result.quantity = max(0, cumulative_quantity - previous_quantity)
            amount = cumulative_quantity * price
            if result.quantity > 0:
                result.price = int(round((amount - int(meta.get("confirmed_amount", 0))) / result.quantity))
            result.fill_id = f"{result.order_no or symbol}:{cumulative_quantity}"
            result.fill_mode = "limit" if terminal else "partial_fill_pending"
            if result.quantity == 0:
                if terminal:
                    self._pending_entry_meta.pop(symbol, None)
                self._save_daily_state_if_due(force=True)
                applied = terminal or identity_changed
            else:
                applied = self._apply_buy_fill(result, quantity=result.quantity, price=result.price) is not None
            if applied and terminal and result.requested_quantity > cumulative_quantity:
                order_logger.info(
                    "매수 잔량 취소 확정: buy %s order_no=%s 취소=%d주 누적체결=%d주",
                    symbol, result.order_no or "-", result.requested_quantity - cumulative_quantity,
                    cumulative_quantity,
                )
            return applied

        position = self.positions.get(symbol)
        pending = self._pending_sell_fills.get(symbol)
        requested = max(0, int(result.requested_quantity or 0))
        quantity = max(0, int(result.quantity or 0))
        price = max(0, int(result.price or 0))
        complete = requested <= 0 or quantity >= requested
        fill_id = str(result.fill_id or "") or (str(result.order_no or "") if complete
                                               else f"{str(result.order_no or symbol)}:{quantity}")
        if quantity > 0 and any(str(item.get("fill_id") or "") == fill_id
                                for item in self._sell_fill_ledger):
            return False
        if position is not None and pending is None:
            pending = self._track_submitted_sell(result, position)
        elif pending is not None and result.order_no:
            pending["order_no"] = str(result.order_no)
        if position is not None and quantity > 0 and price > 0:
            record = self._append_confirmed_sell_fill(
                result=result,
                position=self._copy_position(position),
                quantity=quantity,
                sell_price=price,
                clear_pending=complete,
                fill_id=fill_id,
            )
            if record is None:
                return False
            result.fill_id = str(record["fill_id"])
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
            return pending is not None
        return True

    def sync_positions_from_account(self, account_positions: List[Position]) -> None:
        account_map = {
            str(position.symbol): position
            for position in account_positions
            if max(0, int(position.quantity or 0)) > 0
        }
        now = self._now()

        for symbol, position in list(self.positions.items()):
            account_position = account_map.get(symbol)
            if account_position is None:
                if position.pending_exit_started_at is not None:
                    pending = self._pending_sell_fills.get(symbol)
                    if pending is not None:
                        pending["account_remaining_quantity"] = 0
                elif position.buy_time.date() == now.date():
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
                if pending is None:
                    logger.warning(
                        "청산 주문 정보 없는 대기 상태 해제: %s %d주 "
                        "(계좌 보유 확인, 다음 틱 재주문)",
                        symbol,
                        account_quantity,
                    )
                    position.pending_exit_started_at = None
                    position.pending_exit_quantity = 0
                    position.pending_exit_reason = ""
                else:
                    pending["account_remaining_quantity"] = account_quantity
                    position.pending_exit_quantity = account_quantity
                    confirmed_quantity = max(
                        0,
                        int(pending.get("confirmed_quantity", 0) or 0),
                    )
                    snapshot = pending.get("position")
                    if (
                        isinstance(snapshot, PositionState)
                        and snapshot.trade_key == position.trade_key
                    ):
                        account_quantity = min(
                            account_quantity,
                            max(0, snapshot.quantity - confirmed_quantity),
                        )
                        position.pending_exit_quantity = account_quantity
            if account_quantity != position.quantity:
                # A holding's aggregate average is not a correction to this buy fill.
                # Reductions keep the remaining shares' cost; only additions rebase it.
                previous_entry_price = position.buy_price
                if account_quantity > position.quantity:
                    position.buy_price = account_price
                position.quantity = account_quantity
                if position.plan is not None:
                    position.plan = self._reprice_plan_for_fill(
                        plan=position.plan,
                        fill_price=position.buy_price,
                        quantity=account_quantity,
                        previous_entry_price=previous_entry_price,
                    )

        for symbol, pending in list(self._pending_sell_fills.items()):
            if symbol in self.positions or symbol in account_map:
                continue
            snapshot = pending.get("position")
            if (
                isinstance(snapshot, PositionState)
                and snapshot.buy_time.date() < now.date()
            ):
                logger.info(
                    "전일 미보유 청산 원장 정리: %s %d주",
                    symbol,
                    max(0, int(pending.get("requested_quantity", 0) or 0)),
                )
                self._pending_sell_fills.pop(symbol, None)

        for symbol, account_position in account_map.items():
            if symbol in self.positions:
                continue
            if symbol in self._pending_entry_meta:
                continue
            pending = self._pending_sell_fills.get(symbol)
            snapshot = pending.get("position") if pending is not None else None
            if isinstance(snapshot, PositionState):
                quantity = min(
                    max(0, int(account_position.quantity or 0)),
                    max(0, snapshot.quantity - int(pending.get("confirmed_quantity", 0) or 0)),
                )
                if quantity > 0:
                    restored = self._copy_position(snapshot)
                    restored.quantity = quantity
                    restored.pending_exit_started_at = pending.get("requested_at") or now
                    restored.pending_exit_quantity = quantity
                    restored.pending_exit_reason = str(pending.get("reason") or "")
                    self.positions[symbol] = restored
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
                previous_quantity = int(self._pending_entry_meta[symbol].get("confirmed_quantity", 0))
                if account_position.quantity <= previous_quantity:
                    continue
                inferred_result = OrderResult(
                    success=True,
                    order_no=str(result.order_no or ""),
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=max(0, int(account_position.quantity or 0)),
                    price=max(1, int(round(float(account_position.avg_price or 0)))),
                    requested_quantity=max(0, int(result.requested_quantity or 0)),
                    fill_mode="account_reconciled",
                    requested_reason="price_breakout",
                )
                if self.on_order_filled(inferred_result):
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
        history=None,
    ) -> List[Dict[str, Any]]:
        if not self._pending_sell_fills:
            return []
        if history is None:
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
            cumulative_quantity = min(requested_quantity, max(0, int(quantity)))
            confirmed_quantity = min(
                cumulative_quantity,
                max(0, int(pending.get("confirmed_quantity", 0) or 0)),
            )
            complete = cumulative_quantity >= requested_quantity or (
                str(row.get("rmn_qty", "")).strip() == "0" and (
                    str(row.get("cncl_yn", "")).strip().upper() == "Y"
                    or cumulative_quantity + (account_api._coerce_int(row.get("cnc_cfrm_qty")) or 0)
                    + (account_api._coerce_int(row.get("rjct_qty")) or 0) >= requested_quantity
                )
            )
            incremental_quantity = cumulative_quantity - confirmed_quantity
            if incremental_quantity <= 0:
                if complete:
                    for fill in reversed(self._sell_fill_ledger):
                        if str(fill.get("order_no") or "") == resolved_order_no:
                            fill["order_complete"] = True
                            break
                    self._pending_sell_fills.pop(symbol, None)
                    position = self.positions.get(symbol)
                    if position is not None:
                        position.pending_exit_started_at = None
                        position.pending_exit_quantity = 0
                        position.pending_exit_reason = ""
                    self._save_daily_state_if_due(force=True)
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
            # An empty balance can arrive before the final execution price.
            # Keep the order until every requested share is in fill history.
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
                remaining_quantity=0 if complete else max(0, requested_quantity - cumulative_quantity),
            )
            record = self._append_confirmed_sell_fill(
                result=confirmed_result,
                position=snapshot,
                quantity=incremental_quantity,
                sell_price=incremental_price,
                remaining_quantity=max(0, snapshot.quantity - cumulative_quantity),
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
                    "remaining_quantity": 0 if complete else max(0, requested_quantity - cumulative_quantity),
                    "requested_quantity": requested_quantity,
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
