from __future__ import annotations

import logging
from typing import Any

from src.models import OrderResult, OrderSide
from src.strategies.momentum_scalp_pnl import calculate_trade_pnl_from_prices
from src.strategies.momentum_scalp_types import INTRADAY_STRATEGY, OPENING_STRATEGY, PositionState


logger = logging.getLogger("kis_trader.strategy.momentum_scalp")


def _apply_planned_ev_fields(pos: PositionState, meta: dict[str, Any]) -> None:
    planned_target = int(meta.get("planned_target_net_pnl", 0) or 0)
    if planned_target > 0:
        pos.planned_target_net_pnl = planned_target
    planned_stop = int(meta.get("planned_stop_net_loss_abs", 0) or 0)
    if planned_stop > 0:
        pos.planned_stop_net_loss_abs = planned_stop
    planned_risk = int(meta.get("planned_risk_net_loss_abs", meta.get("planned_stop_net_loss_abs", 0)) or 0)
    if planned_risk > 0:
        pos.planned_risk_net_loss_abs = planned_risk
    pos.entry_expected_net_pnl = float(meta.get("entry_expected_net_pnl", pos.entry_expected_net_pnl) or 0.0)
    pos.entry_prediction_net_pnl = int(
        meta.get("price_prediction_net_pnl", meta.get("entry_prediction_net_pnl", pos.entry_prediction_net_pnl)) or 0
    )
    pos.entry_prediction_lower_net_pnl = int(
        meta.get(
            "price_prediction_lower_net_pnl",
            meta.get("entry_prediction_lower_net_pnl", pos.entry_prediction_lower_net_pnl),
        )
        or 0
    )
    pos.entry_prediction_upper_net_pnl = int(
        meta.get(
            "price_prediction_upper_net_pnl",
            meta.get("entry_prediction_upper_net_pnl", pos.entry_prediction_upper_net_pnl),
        )
        or 0
    )
    pos.entry_prediction_win_probability = float(
        meta.get("entry_prediction_win_probability", pos.entry_prediction_win_probability) or 0.0
    )
    signal_price = int(
        meta.get(
            "entry_signal_price",
            meta.get("pending_order_reference_price", meta.get("price_prediction_signal_price", 0)),
        )
        or 0
    )
    if signal_price > 0:
        pos.entry_signal_price = signal_price
    for attr_name, meta_key in (
        ("entry_prediction_return_pct", "price_prediction_return_pct"),
        ("entry_prediction_lower_pct", "price_prediction_lower_pct"),
        ("entry_prediction_upper_pct", "price_prediction_upper_pct"),
    ):
        if meta_key in meta:
            setattr(pos, attr_name, float(meta.get(meta_key, 0.0) or 0.0))


def handle_order_filled(strategy: Any, result: OrderResult):
    self = strategy
    now = result.timestamp or self._now()
    self.set_simulated_now(now)

    if not result.success:
        if result.side == OrderSide.BUY:
            symbol = str(result.symbol or "").strip()
            self._pending_entry_meta.pop(symbol, None)
            if str(result.error_category or "") == "symbol_order_unavailable":
                mark_unavailable = getattr(self, "_mark_symbol_order_unavailable", None)
                if callable(mark_unavailable):
                    mark_unavailable(
                        symbol,
                        reason=str(result.message or result.error_category or ""),
                        error_code=str(result.error_code or ""),
                    )
            self._save_daily_state_if_due(force=True)
        if result.side == OrderSide.SELL:
            if str(result.error_category or "") == "no_holding":
                logger.warning(
                    "매도 실패 응답이 계좌 무보유를 가리켰지만 전략 상태는 유지합니다: %s "
                    "(계좌 재동기화가 최종 보유 여부를 확정)",
                    result.symbol,
                )
        return

    if int(result.quantity or 0) <= 0:
        symbol = str(result.symbol or "").strip()
        fill_mode = str(result.fill_mode or "")
        reference_price = int(
            getattr(result, "reference_price", 0)
            or getattr(result, "requested_price", 0)
            or getattr(result, "price", 0)
            or 0
        )
        if (
            result.side == OrderSide.BUY
            and fill_mode in {"market_pending", "limit_then_market_pending"}
            and symbol in self._pending_entry_meta
        ):
            meta = dict(self._pending_entry_meta.get(symbol) or {})
            planned_quantity = self._coerce_pending_meta_int(
                meta,
                "pending_order_quantity",
                "entry_ev_live_model_quantity",
            )
            provisional_price = self._resolve_fill_price_with_quote_fallback(
                symbol,
                broker_price=0,
                requested_price=int(getattr(result, "requested_price", 0) or 0),
                reference_price=reference_price,
            )
            if planned_quantity > 0 and provisional_price > 0:
                self._pending_entry_meta.pop(symbol, None)
                existing = self.positions.get(symbol)
                if existing is not None:
                    total_qty = int(existing.quantity) + int(planned_quantity)
                    if total_qty > 0:
                        existing.buy_price = int(
                            round(
                                (
                                    (int(existing.buy_price) * int(existing.quantity))
                                    + (int(provisional_price) * int(planned_quantity))
                                )
                                / total_qty
                            )
                        )
                    existing.quantity = total_qty
                    existing.invested_amount = max(0, int(existing.buy_price) * int(existing.quantity))
                    existing.high_since_buy = max(existing.high_since_buy, int(provisional_price))
                    _apply_planned_ev_fields(existing, meta)
                    if not str(getattr(existing, "trade_key", "") or "").strip():
                        existing.trade_key = self._make_trade_key(symbol, existing.buy_time)
                    existing.pending_entry_started_at = now
                    existing.pending_entry_reference_price = int(provisional_price)
                    existing.pending_entry_fill_mode = fill_mode
                else:
                    self.positions[symbol] = self._position_from_entry_meta(
                        symbol=symbol,
                        buy_price=int(provisional_price),
                        quantity=int(planned_quantity),
                        invested_amount=int(provisional_price) * int(planned_quantity),
                        buy_time=now,
                        meta=meta,
                    )
                    self.positions[symbol].pending_entry_started_at = now
                    self.positions[symbol].pending_entry_reference_price = int(provisional_price)
                    self.positions[symbol].pending_entry_fill_mode = fill_mode
                logger.warning(
                    "시장가 pending 매수를 기준가로 임시 포지션에 즉시 반영합니다: %s %d주 @ %d원 "
                    "(fill_mode=%s, 계좌 동기화 시 평균가 보정)",
                    symbol,
                    int(planned_quantity),
                    int(provisional_price),
                    fill_mode,
                )
                self._save_daily_state_if_due(force=True)
                return
        if result.side == OrderSide.SELL and reference_price > 0:
            pos = self.positions.get(symbol)
            if pos is not None:
                requested_quantity = max(
                    0,
                    int(getattr(result, "requested_quantity", 0) or 0),
                ) or int(pos.quantity or 0)
                existing_pending_quantity = 0
                if (
                    getattr(pos, "pending_exit_started_at", None) is not None
                    and str(getattr(pos, "pending_exit_fill_mode", "") or "")
                    in {"market_pending", "limit_then_market_pending", "order_result_pending"}
                ):
                    existing_pending_quantity = max(0, int(getattr(pos, "pending_exit_quantity", 0) or 0))
                pos.pending_exit_started_at = now
                pos.pending_exit_order_no = str(getattr(result, "order_no", "") or "")
                pos.pending_exit_reference_price = int(reference_price)
                pos.pending_exit_quantity = min(
                    int(pos.quantity or 0),
                    existing_pending_quantity + requested_quantity,
                )
                pos.pending_exit_reason = str(getattr(result, "requested_reason", "") or "")
                pos.pending_exit_fill_mode = fill_mode
                logger.warning(
                    "시장가 pending 매도를 포지션에 기록합니다: %s %d주 @ 기준가 %d원 "
                    "(fill_mode=%s, reason=%s)",
                    symbol,
                    int(pos.pending_exit_quantity),
                    int(reference_price),
                    fill_mode,
                    pos.pending_exit_reason or "-",
                )
                self._save_daily_state_if_due(force=True)
                return
        if reference_price > 0:
            logger.warning(
                "체결 수량이 확정되지 않아 전략 상태 반영을 보류합니다: %s %s 기준가=%d원 (fill_mode=%s)",
                result.side.value if result.side is not None else "unknown",
                result.symbol,
                reference_price,
                str(result.fill_mode or ""),
            )
        else:
            logger.warning(
                "체결 수량이 확정되지 않아 전략 상태 반영을 보류합니다: %s %s (fill_mode=%s)",
                result.side.value if result.side is not None else "unknown",
                result.symbol,
                str(result.fill_mode or ""),
            )
        if result.side == OrderSide.BUY and symbol in self._pending_entry_meta:
            self._save_daily_state_if_due(force=True)
        return

    if result.side == OrderSide.BUY and str(getattr(result, "fill_mode", "") or "") == "partial_fill_pending":
        symbol = str(result.symbol or "").strip()
        meta = dict(self._pending_entry_meta.get(symbol) or {})
        buy_price = self._resolve_fill_price_with_quote_fallback(
            symbol,
            broker_price=int(result.price or 0),
            requested_price=int(result.requested_price or 0),
            reference_price=int(getattr(result, "reference_price", 0) or 0),
        )
        confirmed_quantity = max(0, int(result.quantity or 0))
        if not symbol or confirmed_quantity <= 0 or buy_price <= 0:
            return
        existing = self.positions.get(symbol)
        if existing is None:
            existing = self._position_from_entry_meta(
                symbol=symbol,
                buy_price=buy_price,
                quantity=confirmed_quantity,
                invested_amount=buy_price * confirmed_quantity,
                buy_time=now,
                meta=meta,
            )
            self.positions[symbol] = existing
        else:
            existing.buy_price = buy_price
            existing.quantity = confirmed_quantity
            existing.invested_amount = buy_price * confirmed_quantity
            existing.high_since_buy = max(existing.high_since_buy, buy_price)
            _apply_planned_ev_fields(existing, meta)
        existing.pending_entry_started_at = now
        existing.pending_entry_reference_price = int(
            meta.get("entry_signal_price", meta.get("pending_order_reference_price", result.reference_price)) or buy_price
        )
        existing.pending_entry_fill_mode = "partial_fill_pending"
        logger.warning(
            "매수 부분체결을 계좌 확정 대기 상태로 유지합니다: %s %d/%d주 @ %d원",
            symbol,
            confirmed_quantity,
            max(confirmed_quantity, int(getattr(result, "requested_quantity", 0) or 0)),
            buy_price,
        )
        self._save_daily_state_if_due(force=True)
        return

    if result.side == OrderSide.BUY:
        meta = self._pending_entry_meta.pop(result.symbol, {})
        buy_price = self._resolve_fill_price_with_quote_fallback(
            result.symbol,
            broker_price=int(result.price or 0),
            requested_price=int(result.requested_price or 0),
            reference_price=int(getattr(result, "reference_price", 0) or 0),
        )
        signal_price = int(meta.get("entry_signal_price", meta.get("price_prediction_signal_price", 0)) or 0)
        if signal_price > 0 and buy_price > 0:
            fill_slippage_pct = ((buy_price - signal_price) / max(1.0, float(signal_price))) * 100.0
            meta["entry_actual_fill_slippage_pct"] = round(fill_slippage_pct, 6)
            planned_buffer_pct = float(meta.get("price_prediction_adverse_fill_buffer_pct", 0.0) or 0.0)
            if (
                str(meta.get("entry_style", "") or "") == "queue_value_scout"
                and planned_buffer_pct > 0.0
                and fill_slippage_pct > planned_buffer_pct
            ):
                logger.warning(
                    "queue_scout 매수 체결가가 평가가보다 불리합니다: %s signal=%d fill=%d slip=%.3f%% buffer=%.3f%% "
                    "pred_net=%s lower_net=%s adverse_lower=%s",
                    result.symbol,
                    signal_price,
                    buy_price,
                    fill_slippage_pct,
                    planned_buffer_pct,
                    meta.get("price_prediction_net_pnl", "-"),
                    meta.get("price_prediction_lower_net_pnl", "-"),
                    meta.get("price_prediction_adverse_lower_net_pnl", "-"),
                )
                self._mark_symbol_entry_cooldown(
                    result.symbol,
                    seconds=max(60, int(getattr(self.config, "loss_symbol_cooldown_seconds", 300) or 300) // 2),
                )
        pos = self.positions.get(result.symbol)
        if pos is not None:
            was_pending_entry = getattr(pos, "pending_entry_started_at", None) is not None
            if getattr(pos, "pending_entry_started_at", None) is not None and int(result.quantity or 0) <= int(pos.quantity or 0):
                pos.buy_price = buy_price
                pos.quantity = int(result.quantity or pos.quantity)
                pos.invested_amount = pos.buy_price * pos.quantity
                pos.high_since_buy = max(pos.high_since_buy, buy_price)
                self._clear_pending_entry_position_state(pos)
                _apply_planned_ev_fields(pos, meta)
                reprice_after_fill = getattr(self, "_reprice_position_ev_after_confirmed_entry", None)
                if callable(reprice_after_fill):
                    reprice_after_fill(pos, signal_price=signal_price)
                if not str(getattr(pos, "trade_key", "") or "").strip():
                    pos.trade_key = self._make_trade_key(result.symbol, pos.buy_time)
                self._save_daily_state()
                return
            total_qty = int(pos.quantity) + int(result.quantity)
            if total_qty > 0:
                pos.buy_price = int(round(((pos.buy_price * pos.quantity) + (buy_price * result.quantity)) / total_qty))
            pos.quantity = total_qty
            pos.invested_amount = pos.buy_price * pos.quantity
            pos.high_since_buy = max(pos.high_since_buy, buy_price)
            planned_take_profit = float(meta.get("adaptive_take_profit_pct", meta.get("planned_take_profit_pct", 0.0)) or 0.0)
            if planned_take_profit > 0.0:
                pos.adaptive_take_profit_pct = planned_take_profit
            planned_stop_loss = float(meta.get("adaptive_stop_loss_pct", 0.0) or 0.0)
            if planned_stop_loss != 0.0:
                pos.adaptive_stop_loss_pct = planned_stop_loss
            planned_trailing_activation = float(meta.get("adaptive_trailing_activation_pct", 0.0) or 0.0)
            if planned_trailing_activation > 0.0:
                pos.adaptive_trailing_activation_pct = planned_trailing_activation
            planned_trailing_stop = float(meta.get("adaptive_trailing_stop_pct", 0.0) or 0.0)
            if planned_trailing_stop != 0.0:
                pos.adaptive_trailing_stop_pct = planned_trailing_stop
            planned_max_hold = int(meta.get("adaptive_max_hold_minutes", 0) or 0)
            if planned_max_hold > 0:
                pos.adaptive_max_hold_minutes = planned_max_hold
            _apply_planned_ev_fields(pos, meta)
            reprice_after_fill = getattr(self, "_reprice_position_ev_after_confirmed_entry", None)
            if callable(reprice_after_fill):
                reprice_after_fill(pos, signal_price=signal_price)
            if was_pending_entry:
                self._clear_pending_entry_position_state(pos)
            if not str(getattr(pos, "trade_key", "") or "").strip():
                pos.trade_key = self._make_trade_key(result.symbol, pos.buy_time)
            self._save_daily_state()
            return

        self.positions[result.symbol] = PositionState(
            symbol=result.symbol,
            buy_price=buy_price,
            quantity=int(result.quantity),
            invested_amount=buy_price * int(result.quantity),
            buy_time=now,
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
            trade_key=self._make_trade_key(result.symbol, now),
        )
        reprice_after_fill = getattr(self, "_reprice_position_ev_after_confirmed_entry", None)
        if callable(reprice_after_fill):
            reprice_after_fill(self.positions[result.symbol], signal_price=signal_price)
        self._save_daily_state()
        return

    if result.side == OrderSide.SELL:
        pos = self.positions.get(result.symbol)
        if pos is None:
            return
        sell_price = self._resolve_fill_price_with_quote_fallback(
            result.symbol,
            broker_price=int(result.price or 0),
            requested_price=int(result.requested_price or 0),
            reference_price=int(getattr(result, "reference_price", 0) or 0),
            position=pos,
        )
        sell_qty = max(0, min(int(result.quantity or 0), int(pos.quantity or 0)))
        if sell_qty <= 0:
            return
        pending_exit_started_at = getattr(pos, "pending_exit_started_at", None)
        pending_exit_order_no = str(getattr(pos, "pending_exit_order_no", "") or "")
        pending_exit_reference_price = int(getattr(pos, "pending_exit_reference_price", 0) or 0)
        pending_exit_quantity = int(getattr(pos, "pending_exit_quantity", 0) or 0)
        pending_exit_reason = str(getattr(pos, "pending_exit_reason", "") or "")
        pending_exit_fill_mode = str(getattr(pos, "pending_exit_fill_mode", "") or "")
        pending_exit_mode_active = bool(
            pending_exit_started_at is not None
            and pending_exit_fill_mode in {"market_pending", "limit_then_market_pending", "order_result_pending"}
        )
        sell_pnl = calculate_trade_pnl_from_prices(
            entry_price=int(pos.buy_price),
            exit_price=int(sell_price or 0),
            quantity=sell_qty,
            commission_rate=float(self.config.commission_rate),
            tax_slippage_rate=float(self.config.tax_slippage_rate),
        )
        gross_pnl = int(sell_pnl.gross_pnl)
        net_pnl = int(sell_pnl.net_pnl)
        fees = int(sell_pnl.fees)
        taxes = int(sell_pnl.taxes)
        is_full_exit = sell_qty >= int(pos.quantity)
        counts_for_daily_breaker = self._position_realized_counts_for_daily_breaker(pos)
        counts_for_strategy_stats = self._position_counts_for_strategy_stats(pos)
        price_estimated = str(getattr(result, "fill_mode", "") or "") == "account_reconciled_estimated"
        self._append_sell_fill_record(
            result=result,
            position=pos,
            quantity=sell_qty,
            sell_price=int(sell_price or 0),
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            fees=fees,
            taxes=taxes,
            count_as_closed_trade=False,
            counts_for_daily_breaker=counts_for_daily_breaker,
            price_estimated=price_estimated,
        )
        pos.realized_gross_pnl_so_far += int(gross_pnl)
        pos.realized_net_pnl_so_far += int(net_pnl)
        pos.realized_fees_paid_so_far += int(fees)
        pos.realized_taxes_paid_so_far += int(taxes)

        cumulative_trade_gross_pnl = int(pos.realized_gross_pnl_so_far)
        cumulative_trade_net_pnl = int(pos.realized_net_pnl_so_far)
        if is_full_exit:
            self._upsert_closed_trade_record(
                trade_key=str(getattr(pos, "trade_key", "") or ""),
                symbol=pos.symbol,
                strategy_name=pos.entry_strategy_name,
                setup_name=pos.entry_setup_name,
            )
            self._rebuild_daily_pnl_from_ledgers()

        log_prefix = "매도 체결" if is_full_exit else "부분매도 체결"

        logger.info(
            "%s: %s %d주 @ %d원 (총손익: %d원, 순손익: %d원, 누적순손익: %d원, order_no=%s, fill_mode=%s, strategy_name=%s, setup_name=%s, regime_label=%s, entry_grade=%s, entry_grade_math=%s, leader_score=%.4f, leader_pct=%.4f, entry_ev=%.2f, entry_ev_conf=%s, conviction_tier=%s, bull_risk_mode=%s, post_loss_admission_class=%s, candidate_class=%s, execution_mode=%s, live_route=%s, queue_source=%s, exit_reason=%s, size_multiplier=%.4f, conviction_score=%.4f, conviction_rank=%s)",
            log_prefix,
            result.symbol,
            sell_qty,
            int(sell_price or 0),
            cumulative_trade_gross_pnl if is_full_exit else gross_pnl,
            cumulative_trade_net_pnl if is_full_exit else net_pnl,
            int(self.daily_pnl.realized_net_pnl),
            str(getattr(result, "order_no", "") or "-"),
            str(getattr(result, "fill_mode", "") or ""),
            pos.entry_strategy_name,
            pos.entry_setup_name,
            pos.regime_label,
            pos.entry_grade,
            pos.entry_grade_math,
            float(pos.leader_score),
            float(pos.leader_percentile),
            float(pos.entry_ev),
            pos.entry_ev_confidence,
            pos.conviction_tier,
            pos.bull_risk_mode,
            pos.post_loss_admission_class,
            pos.candidate_class,
            pos.execution_mode,
            pos.live_route,
            pos.queue_source,
            str(getattr(result, "requested_reason", "") or ""),
            float(pos.size_multiplier),
            float(pos.conviction_score),
            int(pos.conviction_rank),
        )

        if is_full_exit:
            self.positions.pop(result.symbol, None)
        else:
            remaining_quantity = max(0, int(pos.quantity) - int(sell_qty))
            remaining_pending_exit_quantity = max(0, pending_exit_quantity - int(sell_qty))
            if pending_exit_mode_active and remaining_pending_exit_quantity > 0:
                pos.pending_exit_started_at = pending_exit_started_at
                pos.pending_exit_order_no = pending_exit_order_no
                pos.pending_exit_reference_price = pending_exit_reference_price
                pos.pending_exit_quantity = max(0, min(remaining_quantity, int(remaining_pending_exit_quantity)))
                pos.pending_exit_reason = pending_exit_reason
                pos.pending_exit_fill_mode = pending_exit_fill_mode
            else:
                pos.pending_exit_started_at = None
                pos.pending_exit_order_no = ""
                pos.pending_exit_reference_price = 0
                pos.pending_exit_quantity = 0
                pos.pending_exit_reason = ""
                pos.pending_exit_fill_mode = ""
            pos.quantity -= sell_qty
            pos.invested_amount = pos.buy_price * pos.quantity
            pos.partial_exit_done = True

        if (
            is_full_exit
            and not pos.partial_exit_done
            and cumulative_trade_net_pnl < 0
            and counts_for_strategy_stats
            and pos.entry_strategy_name in {OPENING_STRATEGY, INTRADAY_STRATEGY}
        ):
            self._bull_loss_count_today += 1
            self._mark_symbol_entry_cooldown(
                result.symbol,
                seconds=int(self.config.loss_symbol_cooldown_seconds),
            )
        elif (
            is_full_exit
            and cumulative_trade_net_pnl > 0
            and counts_for_strategy_stats
            and pos.entry_strategy_name in {OPENING_STRATEGY, INTRADAY_STRATEGY}
        ):
            self._recover_bull_loss_count_after_win(cumulative_trade_net_pnl)

        self._update_daily_breakers()
        self._save_daily_state()


__all__ = ["handle_order_filled"]
