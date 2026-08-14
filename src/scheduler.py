from __future__ import annotations

import copy
import logging
import math
import time
import json
from datetime import datetime, time as clock_time, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import requests

from src.account import AccountAPI
from src.api_client import is_kis_rate_limited_message
from src.executor import OrderExecutor, RiskManager
from src.logger_setup import setup_logger
from src.market_calendar import is_krx_regular_trading_day, normalize_trading_date
from src.market_stream import MarketQuoteStream
from src.performance_reporting import update_performance_reports
from src.notifications import AlertManager
from src.models import Order, OrderResult, OrderSide
from src.trading import TradingAPI

if TYPE_CHECKING:
    from src.strategies.momentum_scalp import MomentumScalpStrategy

logger = logging.getLogger("kis_trader.scheduler")

# 장 시간
MARKET_OPEN = (9, 0)    # 09:00
MARKET_CLOSE = (15, 30)  # 15:30
PRE_OPEN = (8, 50)       # 장 시작 10분 전 준비


class TradingScheduler:
    """매일 장 시간에 맞춰 전략을 자동 실행하는 스케줄러."""

    def __init__(self, strategy: MomentumScalpStrategy):
        self.strategy = strategy
        self.market_data = strategy.market_data
        self.client = self.market_data.client
        self.config = self.client.config
        self._shutdown = False

        self.trading = TradingAPI(self.client)
        self.account = AccountAPI(self.client)
        self.executor = OrderExecutor(self.trading, RiskManager())
        self._alerts = AlertManager()
        quote_freshness = float(strategy.config.quote_freshness_seconds or 5)
        self._quote_stream = MarketQuoteStream(
            self.config,
            stale_seconds=quote_freshness,
        )

        self._balance_retry_attempts = 3
        self._balance_retry_delay_seconds = 2
        self._last_session_end_reason = "session_end"
        self._last_halt_alert_day = None
        self._last_runtime_pool_refresh_at = None
        self._session_started_at = None
        self._last_stream_stale_log_at = None
        self._last_emergency_position_refresh_at = None
        self._last_supplemental_quote_refresh_at = None
        self._last_tick_activity_log_at = None
        self._last_loop_error_alert_at = None
        self._market_closed_session_date = None
        self._account_order_blocked_session_date = None
        self._pending_order_blocks = {}
        self._last_pending_entry_reconcile_at = None
        self._last_pending_exit_reconcile_at = None

    def stop(self):
        self._shutdown = True
        self._quote_stream.stop()

    def run(self, tick_interval: int = 1):
        """스케줄러를 시작한다. Ctrl+C로 종료.

        Args:
            tick_interval: 전략 틱 간격 (초). 실시간 경로 기본 1초.
        """
        setup_logger(self.config.log_level)
        logger.info("=" * 50)
        logger.info("스케줄러 시작 [%s 모드]", self.config.trading_mode.upper())
        logger.info("틱 간격: %d초", tick_interval)
        logger.info("장외 체크 간격: %d초", self.config.off_hours_check_interval)
        logger.info("=" * 50)

        try:
            while not self._shutdown:
                try:
                    now = datetime.now()

                    if self._is_trading_time(now):
                        halted_for_day = self._run_trading_session(tick_interval)
                        if halted_for_day and not self._shutdown:
                            if self._last_session_end_reason == "daily_hard_stop":
                                logger.info("당일 하드스탑 감지: 다음 장 준비 시각까지 대기합니다.")
                            elif self._last_session_end_reason == "daily_profit_target":
                                logger.info("당일 목표 달성 종료: 다음 장 준비 시각까지 대기합니다.")
                            else:
                                logger.info("당일 세션 종료 감지: 다음 장 준비 시각까지 대기합니다.")
                            self._sleep_until_preopen()
                    else:
                        wait = self._seconds_until_preopen(now)
                        if wait > 0:
                            next_open = now + timedelta(seconds=wait)
                            logger.info(
                                "장 시간이 아닙니다. 다음 준비 시각: %s (%d분 후)",
                                next_open.strftime("%H:%M"),
                                wait // 60,
                            )
                            self._interruptible_sleep(min(wait, self.config.off_hours_check_interval))
                        else:
                            # 오늘 장 끝남, 내일까지 대기
                            logger.info("오늘 장이 종료되었습니다. 내일까지 대기합니다.")
                            self._interruptible_sleep(self.config.off_hours_check_interval)
                except requests.exceptions.RequestException as e:
                    if self._shutdown:
                        break
                    logger.warning(
                        "스케줄러 네트워크 오류: %s (30초 후 재시도)",
                        e,
                    )
                    self._interruptible_sleep(min(30, self.config.off_hours_check_interval))
                except Exception:
                    if self._shutdown:
                        break
                    logger.exception("스케줄러 루프 오류 (30초 후 재시도)")
                    self._send_loop_error_alert(scope="스케줄러 루프", message="자세한 내용은 trading.log를 확인해 주십시오.")
                    self._interruptible_sleep(min(30, self.config.off_hours_check_interval))

        except KeyboardInterrupt:
            logger.info("Ctrl+C — 스케줄러를 종료합니다.")
        finally:
            self._quote_stream.stop()
            logger.info("스케줄러 종료")

    def _is_trading_time(self, now: datetime) -> bool:
        """현재가 거래 가능 시간인지 확인."""
        open_time = clock_time(hour=MARKET_OPEN[0], minute=MARKET_OPEN[1])
        close_time = clock_time(hour=MARKET_CLOSE[0], minute=MARKET_CLOSE[1])
        return self._is_regular_trading_day(now) and open_time <= now.time() < close_time

    def _is_regular_trading_day(self, now: datetime) -> bool:
        date_key = normalize_trading_date(now)
        if self._market_closed_session_date == date_key:
            return False
        return is_krx_regular_trading_day(now)

    def _seconds_until_preopen(self, now: datetime) -> int:
        """다음 준비 시각까지 남은 초."""
        today_preopen = now.replace(hour=PRE_OPEN[0], minute=PRE_OPEN[1], second=0, microsecond=0)
        today_open = now.replace(hour=MARKET_OPEN[0], minute=MARKET_OPEN[1], second=0, microsecond=0)

        if self._is_regular_trading_day(now):
            if now < today_preopen:
                return max(0, math.ceil((today_preopen - now).total_seconds()))
            if today_preopen <= now < today_open:
                # 프리오픈(08:50~09:00) 구간은 당일 09:00 장 시작까지 대기
                return max(0, math.ceil((today_open - now).total_seconds()))

        # 오늘 이미 지남 → 다음 평일 계산
        days_ahead = 1
        next_day = now + timedelta(days=days_ahead)
        while not self._is_regular_trading_day(next_day):
            days_ahead += 1
            next_day = now + timedelta(days=days_ahead)

        next_preopen = next_day.replace(hour=PRE_OPEN[0], minute=PRE_OPEN[1], second=0, microsecond=0)
        return max(0, math.ceil((next_preopen - now).total_seconds()))

    def _next_preopen_after(self, now: datetime) -> datetime:
        """지정 시각 이후 가장 가까운 다음 준비 시각을 계산한다."""
        today_preopen = now.replace(hour=PRE_OPEN[0], minute=PRE_OPEN[1], second=0, microsecond=0)

        if self._is_regular_trading_day(now) and now < today_preopen:
            return today_preopen

        days_ahead = 1
        next_day = now + timedelta(days=days_ahead)
        while not self._is_regular_trading_day(next_day):
            days_ahead += 1
            next_day = now + timedelta(days=days_ahead)

        return next_day.replace(hour=PRE_OPEN[0], minute=PRE_OPEN[1], second=0, microsecond=0)

    def _interruptible_sleep(self, seconds: int):
        """shutdown 체크하면서 대기."""
        end = time.time() + seconds
        while time.time() < end and not self._shutdown:
            time.sleep(1)

    def _reset_strategy_simulated_clock(self) -> None:
        try:
            self.strategy.set_simulated_now(None)
        except Exception:
            logger.exception("전략 시뮬레이션 시각 초기화 실패")

    @staticmethod
    def _is_rate_limited_message(message: str) -> bool:
        return is_kis_rate_limited_message(message)

    def _client_rate_limit_cooldown(self) -> float:
        try:
            return max(0.0, float(self.client.rate_limit_cooldown_remaining()))
        except Exception:
            return 0.0

    def _multi_price_backoff_remaining(self) -> float:
        try:
            return max(0.0, float(self.market_data.multi_price_backoff_remaining()))
        except Exception:
            return 0.0

    @staticmethod
    def _normalize_symbols(symbols: List[str]) -> List[str]:
        seen = set()
        ordered: List[str] = []
        for symbol in symbols:
            normalized = str(symbol or "").strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(normalized)
        return ordered

    def _session_age_seconds(self, now: datetime):
        started_at = self._session_started_at
        if started_at is None:
            return None
        return max(0.0, float((now - started_at).total_seconds()))

    def _market_data_pressure_active(self) -> bool:
        fluctuation_backoff = 0.0
        try:
            fluctuation_backoff = max(
                0.0,
                float(self.market_data.fluctuation_backoff_remaining()),
            )
        except Exception:
            fluctuation_backoff = 0.0
        return (
            self._client_rate_limit_cooldown() > 0
            or self._multi_price_backoff_remaining() > 0
            or fluctuation_backoff > 0
        )

    def _session_start_balance_delay_seconds(self) -> float:
        base_delay = 2.0 if self.config.is_paper else 0.25
        return max(base_delay, self._client_rate_limit_cooldown())

    def _refresh_runtime_pool_if_needed(self, now: datetime, *, force: bool = False) -> bool:
        strategy = self.strategy
        if strategy.fixed_pool:
            return False
        cfg = strategy.config
        if self._market_data_pressure_active():
            return False
        session_age_seconds = self._session_age_seconds(now)
        if not force and session_age_seconds is not None and session_age_seconds < 2.0:
            return False

        min_change_rate = float(cfg.dynamic_pool_quote_min_change_rate or 0.0)
        refresh_interval = max(30, int(cfg.pool_refresh_interval or 120))
        fetch_count = max(30, int(cfg.dynamic_pool_ranking_fetch_count or 60))

        last_refreshed = self._last_runtime_pool_refresh_at
        if last_refreshed is not None and (now - last_refreshed).total_seconds() < refresh_interval:
            return False

        # Discovery should be wider than entry judgment: do not exclude low-price
        # or already-fast movers before the EV planner can evaluate them.
        symbols = self.market_data.get_fluctuation_symbols(
            count=fetch_count,
            min_change_rate=min_change_rate,
        )
        if not symbols:
            self._last_runtime_pool_refresh_at = now
            logger.info("동적풀 갱신 생략: 등락률 후보 없음, 기존 감시풀 유지")
            return False

        strategy.update_runtime_pool(symbols)
        self._last_runtime_pool_refresh_at = now
        active_dynamic_pool = strategy._pool
        logger.info(
            "동적풀 갱신: %d개 (%s)",
            len(active_dynamic_pool),
            ", ".join(active_dynamic_pool[:5]),
        )
        return True

    def _log_tick_activity(self, now: datetime, watchlist: List[str], refresh_batch: List[str], quotes: List) -> None:
        last_log_at = self._last_tick_activity_log_at
        should_log = (
            last_log_at is None
            or (now - last_log_at).total_seconds() >= 60
        )
        if not should_log:
            return
        positions = self.strategy.positions
        long_candidates = int(self.strategy._last_evaluation_count)
        logger.info(
            "틱 상태: 감시=%d 배치=%d 수신=%d 롱후보=%d 보유=%d",
            len(watchlist),
            len(refresh_batch),
            len(quotes),
            long_candidates,
            len(positions),
        )
        self._last_tick_activity_log_at = now

    def _prioritized_stream_symbols(self, watchlist: List[str]) -> List[str]:
        strategy = self.strategy
        cfg = strategy.config
        positions = strategy.positions
        return self._normalize_symbols(
            list(positions)
            + list(strategy._pool)
            + list(cfg.inverse_etfs)
            + list(cfg.static_watchlist)
            + list(watchlist)
        )

    def _supplemental_watchlist_quotes(
        self,
        now: datetime,
        prioritized_symbols: List[str],
        subscribed_symbols: List[str],
    ) -> List:
        last_refresh = self._last_supplemental_quote_refresh_at
        refresh_interval = max(
            2.0,
            float(self.strategy.config.quote_freshness_seconds or 5),
        )
        if (
            last_refresh is not None
            and (now - last_refresh).total_seconds()
            < refresh_interval
        ):
            return []

        subscribed = set(subscribed_symbols)
        held = set(self.strategy.positions)
        stale_stream = [
            symbol
            for symbol in self._quote_stream.stale_symbols(
                subscribed_symbols,
                now=now,
            )
            if symbol not in held
        ]
        excluded = [
            symbol
            for symbol in prioritized_symbols
            if symbol not in subscribed and symbol not in held
        ]
        symbols = self._normalize_symbols(stale_stream + excluded)[:30]
        if not symbols:
            return []

        self._last_supplemental_quote_refresh_at = now
        return self.market_data.get_multi_price(symbols)

    def _emergency_position_quotes(self, now: datetime) -> List:
        positions = self.strategy.positions
        if not positions:
            return []
        symbols = self._quote_stream.stale_symbols(list(positions), now=now)
        if not symbols:
            return []
        last_refresh = self._last_emergency_position_refresh_at
        if last_refresh is not None and (now - last_refresh).total_seconds() < 5.0:
            return []
        self._last_emergency_position_refresh_at = now
        quotes = []
        for index in range(0, len(symbols), 30):
            quotes.extend(
                self.market_data.get_multi_price(
                    symbols[index:index + 30],
                )
            )
        should_log = (
            self._last_stream_stale_log_at is None
            or (now - self._last_stream_stale_log_at).total_seconds() >= 60
        )
        if should_log:
            logger.warning(
                "실시간 시세 stale: 보유종목 %d개만 REST 비상 조회했습니다.",
                len(symbols),
            )
            self._last_stream_stale_log_at = now
        return quotes

    @staticmethod
    def _order_block_side_value(side) -> str:
        if isinstance(side, OrderSide):
            return side.value
        return str(side or "").strip()

    @classmethod
    def _order_block_key(cls, item) -> Optional[tuple]:
        symbol = str(item.symbol or "").strip()
        side_value = cls._order_block_side_value(item.side)
        if not symbol or side_value not in {OrderSide.BUY.value, OrderSide.SELL.value}:
            return None
        return (symbol, side_value)

    def _pending_order_block_seconds(self) -> int:
        return max(30, int(self.strategy.config.pending_order_block_seconds))

    def _purge_expired_pending_order_blocks(self, now: datetime) -> None:
        for key, payload in list(self._pending_order_blocks.items()):
            expires_at = payload.get("expires_at") if isinstance(payload, dict) else payload
            if expires_at is not None and expires_at <= now:
                self._pending_order_blocks.pop(key, None)

    def _filter_pending_duplicate_orders(self, orders: List[Order], now: Optional[datetime] = None) -> List[Order]:
        if not orders:
            return []
        now = now or datetime.now()
        self._purge_expired_pending_order_blocks(now)
        seen = set()
        filtered: List[Order] = []
        for order in orders:
            key = self._order_block_key(order)
            if key is None:
                filtered.append(order)
                continue
            if key in seen:
                logger.warning(
                    "중복 주문 차단: %s %s %d주 (같은 배치)",
                    self._order_block_side_value(order.side),
                    str(order.symbol or ""),
                    int(order.quantity or 0),
                )
                continue
            if key in self._pending_order_blocks:
                allow_unreserved_sell = False
                if order.side == OrderSide.SELL:
                    symbol = str(order.symbol or "").strip()
                    position = self.strategy.positions.get(symbol)
                    if position is not None:
                        position_quantity = max(0, int(position.quantity or 0))
                        pending_quantity = max(0, int(position.pending_exit_quantity or 0))
                        unreserved_quantity = max(0, position_quantity - min(position_quantity, pending_quantity))
                        order_quantity = max(0, int(order.quantity or 0))
                        allow_unreserved_sell = 0 < order_quantity <= unreserved_quantity
                if not allow_unreserved_sell:
                    logger.warning(
                        "pending 주문 중복 차단: %s %s %d주 (이전 주문 체결 확인 대기)",
                        self._order_block_side_value(order.side),
                        str(order.symbol or ""),
                        int(order.quantity or 0),
                    )
                    continue
            seen.add(key)
            filtered.append(order)
        return filtered

    @staticmethod
    def _result_requires_balance_reconcile(result: OrderResult) -> bool:
        if (
            not result.success
            and result.side == OrderSide.SELL
            and str(result.error_category or "") == "no_holding"
        ):
            return True
        if not result.success:
            return False
        fill_mode = str(result.fill_mode or "")
        quantity = int(result.quantity or 0)
        price = int(result.price or 0)
        requested_price = int(result.requested_price or 0)
        if fill_mode in {"market_pending", "partial_fill_pending"}:
            return True
        requested_quantity = max(0, int(result.requested_quantity or 0))
        if (
            requested_quantity > 0
            and quantity < requested_quantity
            and fill_mode != "limit_partial"
        ):
            return True
        if quantity <= 0:
            return True
        if price <= 0 and requested_price <= 0:
            return True
        return False

    def _remember_pending_order_results(self, results: List[OrderResult], now: Optional[datetime] = None) -> None:
        if not results:
            return
        now = now or datetime.now()
        self._purge_expired_pending_order_blocks(now)
        for result in results:
            key = self._order_block_key(result)
            if key is None:
                continue
            if result.success and self._result_requires_balance_reconcile(result):
                self._pending_order_blocks[key] = {
                    "expires_at": now + timedelta(seconds=self._pending_order_block_seconds()),
                    "order_no": str(result.order_no or ""),
                    "fill_mode": str(result.fill_mode or ""),
                    "requested_quantity": max(0, int(result.requested_quantity or 0)),
                }
            else:
                self._pending_order_blocks.pop(key, None)

    def _clear_pending_order_blocks_for_results(self, results: List[OrderResult]) -> None:
        if not results:
            return
        for result in results:
            key = self._order_block_key(result)
            if key is not None and not self._result_requires_balance_reconcile(result):
                self._pending_order_blocks.pop(key, None)

    def _pending_sell_block_symbols_for_positions(self, positions, now: Optional[datetime] = None) -> List[str]:
        if not positions:
            return []
        now = now or datetime.now()
        self._purge_expired_pending_order_blocks(now)
        position_symbols = {str(symbol or "").strip() for symbol in positions}
        position_symbols.discard("")
        if not position_symbols:
            return []
        blocked_symbols = []
        for key in self._pending_order_blocks:
            if not isinstance(key, tuple) or len(key) < 2:
                continue
            symbol, side_value = key[0], key[1]
            if side_value == OrderSide.SELL.value and symbol in position_symbols:
                blocked_symbols.append(symbol)
        return sorted(set(blocked_symbols))

    def _reconcile_pending_exit_positions(self) -> bool:
        positions = self.strategy.positions
        position_symbols = [
            str(symbol or "").strip()
            for symbol, position in positions.items()
            if position.pending_exit_started_at is not None
            and int(position.pending_exit_quantity or 0) > 0
        ]
        pending_fills = self.strategy._pending_sell_fills
        pending_symbols = sorted(
            {
                symbol
                for symbol in position_symbols + [str(symbol or "").strip() for symbol in pending_fills]
                if symbol
            }
        )
        if not pending_symbols:
            self._last_pending_exit_reconcile_at = None
            return False

        now = datetime.now()
        last_checked = self._last_pending_exit_reconcile_at
        if last_checked is not None and (now - last_checked).total_seconds() < 5.0:
            return False
        self._last_pending_exit_reconcile_at = now

        balance = self._fetch_balance_with_retry(
            "pending 매도 잔고 확인",
            max_attempts=1,
            base_delay_seconds=1,
        )
        if balance is None:
            return False
        self.strategy.sync_positions_from_account(balance.positions)
        self._confirm_reconciled_sell_prices()
        logger.debug(
            "pending 매도 잔고 재동기화 완료: %s",
            ",".join(sorted(pending_symbols)),
        )
        return True

    def _reconcile_pending_entry_positions(self) -> bool:
        pending = self.strategy._pending_entry_meta
        if not pending:
            self._last_pending_entry_reconcile_at = None
            return False

        now = datetime.now()
        created_times = [
            payload.get("created_at")
            for payload in pending.values()
            if isinstance(payload, dict) and isinstance(payload.get("created_at"), datetime)
        ]
        if created_times and (now - min(created_times)).total_seconds() < 2.0:
            return False
        last_checked = self._last_pending_entry_reconcile_at
        if last_checked is not None and (now - last_checked).total_seconds() < 30.0:
            return False
        self._last_pending_entry_reconcile_at = now

        balance = self._fetch_balance_with_retry(
            "pending 매수 잔고 확인",
            max_attempts=1,
            base_delay_seconds=1,
        )
        if balance is None:
            return False

        pending_results: List[OrderResult] = []
        for symbol, payload in list(pending.items()):
            if not isinstance(payload, dict):
                continue
            plan = payload.get("plan")
            pending_results.append(
                OrderResult(
                    success=True,
                    order_no=str(payload.get("order_no") or ""),
                    symbol=str(symbol),
                    side=OrderSide.BUY,
                    quantity=0,
                    price=0,
                    requested_price=max(0, int(plan.entry_limit_price or 0)),
                    reference_price=max(0, int(payload.get("signal_price", 0) or 0)),
                    requested_quantity=max(0, int(plan.quantity or 0)),
                    fill_mode="order_result_pending",
                    requested_reason="expected_value",
                    timestamp=now,
                )
            )

        inferred_results: List[OrderResult] = []
        inferred_results = list(
            self.strategy.reconcile_pending_fills_from_account(
                pending_results,
                balance.positions,
            ) or []
        )
        self.strategy.sync_positions_from_account(balance.positions)

        remaining = self.strategy._pending_entry_meta
        blocks = self._pending_order_blocks
        for result in pending_results:
            if result.symbol not in remaining and isinstance(blocks, dict):
                blocks.pop((result.symbol, OrderSide.BUY.value), None)
        for inferred in inferred_results:
            after_position = self.strategy.positions.get(inferred.symbol)
            self._send_order_result_alert(inferred, after_position=after_position)
        if inferred_results:
            logger.info(
                "pending 매수 체결 확정: %s",
                ",".join(sorted(result.symbol for result in inferred_results)),
            )
        elif any(
            str(position.symbol or "") in remaining
            for position in balance.positions
        ):
            # A quote is needed to reprice the restored plan. Retry next tick
            # instead of waiting the normal 30-second account poll interval.
            self._last_pending_entry_reconcile_at = None
        return True

    def _reconcile_positions_after_order_failures(self, results: List[OrderResult], before_positions=None) -> List[OrderResult]:
        if not results:
            return []
        has_sell_rate_limit_failure = any(
            (not result.success)
            and result.side == OrderSide.SELL
            and self._is_rate_limited_message(result.message)
            for result in results
        )
        has_sell_no_position_failure = any(
            (not result.success)
            and result.side == OrderSide.SELL
            and str(result.error_category or "") == "no_holding"
            for result in results
        )
        has_uncertain_fill = any(self._result_requires_balance_reconcile(result) for result in results)
        if not has_sell_rate_limit_failure and not has_sell_no_position_failure and not has_uncertain_fill:
            return []
        context = "주문 결과 확인 후 잔고 재동기화"
        if has_sell_rate_limit_failure or has_sell_no_position_failure:
            context = "매도 실패 후 잔고 재동기화"
        balance = self._fetch_balance_with_retry(
            context,
            max_attempts=2,
            base_delay_seconds=1,
        )
        if balance is None:
            logger.warning("%s 실패: 기존 포지션 상태를 유지합니다.", context)
            return []
        inferred_results = []
        try:
            inferred_results = list(
                self.strategy.reconcile_pending_fills_from_account(
                    results,
                    balance.positions,
                ) or []
            )
        except Exception:
            logger.exception("pending 체결 계좌 보정 실패")
        try:
            inferred_results.extend(
                list(
                    self.strategy.reconcile_no_holding_sell_failures_from_account(
                        results,
                        balance.positions,
                    ) or []
                )
            )
        except Exception:
            logger.exception("무보유 매도 실패 계좌 보정 실패")
        self.strategy.sync_positions_from_account(balance.positions)
        logger.debug(
            "계좌 기준 재동기화 완료: 보유 %d종목",
            len(balance.positions),
        )
        self._confirm_reconciled_sell_prices(inferred_results)
        for inferred in inferred_results:
            symbol = str(inferred.symbol or "")
            before_position = None
            if inferred.side == OrderSide.SELL and before_positions:
                before_position = before_positions.get(symbol)
            after_position = self.strategy.positions.get(symbol)
            self._send_order_result_alert(
                inferred,
                before_position=before_position,
                after_position=after_position,
            )
        return inferred_results

    def _stop_session_on_market_closed_order_failure(self, results: List[OrderResult]) -> bool:
        if not any(
            (not result.success)
            and str(result.error_category or "") == "market_closed"
            for result in (results or [])
        ):
            return False

        date_key = datetime.now().strftime("%Y%m%d")
        self._market_closed_session_date = date_key
        self._last_session_end_reason = "market_holiday"
        logger.warning("주문 API가 비영업일/휴장 응답을 반환했습니다. date=%s 세션을 중단합니다.", date_key)
        self._alerts.send(
            event_key=f"market_closed_order_{date_key}",
            title="휴장일 주문 차단",
            message=f"{date_key} 주문 API가 비영업일을 반환해 오늘 거래 세션을 중단했습니다.",
            level="warning",
            cooldown_seconds=0,
        )
        return True

    def _stop_session_on_account_order_unavailable(self, results: List[OrderResult]) -> bool:
        if not any(
            (not result.success)
            and str(result.error_category or "") == "account_order_unavailable"
            for result in (results or [])
        ):
            return False

        date_key = datetime.now().strftime("%Y%m%d")
        self._account_order_blocked_session_date = date_key
        self._last_session_end_reason = "account_order_unavailable"
        logger.error(
            "계좌/API 키 조합이 주문 불가 상태입니다. date=%s 오늘 세션의 추가 주문을 중단합니다.",
            date_key,
        )
        self._alerts.send(
            event_key=f"account_order_unavailable_{date_key}",
            title="계좌 주문 불가",
            message=(
                f"{date_key} 모의투자 계좌/API 키 조합이 주문 불가로 응답했습니다.\n"
                "PAPER_ACCOUNT_NUMBER, PAPER_API_KEY, PAPER_API_SECRET 연결을 확인해 주십시오."
            ),
            level="error",
            cooldown_seconds=0,
        )
        return True

    def _confirm_reconciled_sell_prices(self, results: Optional[List[OrderResult]] = None) -> None:
        try:
            corrections = list(
                self.strategy.confirm_reconciled_sell_fills(
                    self.account,
                    results=results,
                ) or []
            )
        except Exception:
            logger.exception("계좌 재동기화 매도 체결가 재확정 실패")
            return
        for correction in corrections:
            partial = bool(correction.get("partial"))
            logger.info(
                "계좌 재동기화 매도 %s 확정: %s order_no=%s %d주 @ %d원 (기존 %d원, 순손익 %+d원 보정, 잔여 %d주)",
                "부분체결" if partial else "체결가",
                str(correction.get("symbol", "") or ""),
                str(correction.get("order_no", "") or "-"),
                int(correction.get("quantity", 0) or 0),
                int(correction.get("corrected_price", 0) or 0),
                int(correction.get("previous_price", 0) or 0),
                int(correction.get("delta_net_pnl", 0) or 0),
                int(correction.get("remaining_quantity", 0) or 0),
            )
            if not bool(correction.get("notify", True)):
                continue
            confirmed_result = OrderResult(
                success=True,
                order_no=str(correction.get("order_no", "") or ""),
                symbol=str(correction.get("symbol", "") or ""),
                side=OrderSide.SELL,
                quantity=int(correction.get("alert_quantity", correction.get("quantity", 0)) or 0),
                price=int(correction.get("alert_price", correction.get("corrected_price", 0)) or 0),
                requested_quantity=int(correction.get("alert_quantity", correction.get("quantity", 0)) or 0),
                fill_mode=(
                    "account_reconciled_partial"
                    if partial
                    else "account_reconciled_confirmed"
                ),
                requested_reason=str(correction.get("requested_reason", "") or ""),
                fill_id=str(correction.get("fill_id", "") or ""),
                remaining_quantity=int(correction.get("remaining_quantity", 0) or 0),
            )
            self._send_order_result_alert(
                confirmed_result,
                confirmed_trade_net_pnl=int(
                    correction.get("alert_net_pnl", correction.get("net_pnl", 0)) or 0
                ),
            )

    def _handle_order_results(self, results: List[OrderResult]) -> None:
        if not results:
            return
        self._remember_pending_order_results(results, datetime.now())
        before_positions = {}
        for result in results:
            before_position = self.strategy.positions.get(result.symbol)
            if before_position is not None:
                before_positions.setdefault(result.symbol, copy.copy(before_position))
            self.strategy.on_order_filled(result)
            after_position = self.strategy.positions.get(result.symbol)
            self._send_order_result_alert(result, before_position=before_position, after_position=after_position)
        inferred_results = self._reconcile_positions_after_order_failures(results, before_positions=before_positions)
        self._clear_pending_order_blocks_for_results(inferred_results)
        self._send_daily_halt_alert_if_needed()

    def _alert_daily_net_pnl(self) -> int:
        return int(self.strategy.daily_pnl.realized_net_pnl)

    def _confirmed_sell_fill_net_pnl(self, result: OrderResult) -> Optional[int]:
        ledger = self.strategy._sell_fill_ledger
        fill_id = str(result.fill_id or "").strip()
        order_no = str(result.order_no or "").strip()
        quantity = max(0, int(result.quantity or 0))
        price = max(0, int(result.price or 0))
        for fill in reversed(ledger):
            if not isinstance(fill, dict):
                continue
            ledger_fill_id = str(fill.get("fill_id") or "").strip()
            exact_fill = bool(fill_id) and ledger_fill_id == fill_id
            same_order_fill = (
                bool(order_no)
                and str(fill.get("order_no") or "").strip() == order_no
                and int(fill.get("quantity", 0) or 0) == quantity
                and int(fill.get("sell_price", 0) or 0) == price
            )
            if exact_fill or same_order_fill:
                return int(fill.get("net_pnl", 0) or 0)
        return None

    def _send_order_result_alert(
        self,
        result: OrderResult,
        before_position=None,
        after_position=None,
        *,
        confirmed_trade_net_pnl: Optional[int] = None,
    ) -> None:
        symbol = str(result.symbol or "").strip()
        if not symbol:
            return

        price = int(result.price or 0)
        qty = int(result.quantity or 0)
        side = result.side
        daily_net_pnl = self._alert_daily_net_pnl()

        if result.success:
            if qty <= 0 or price <= 0:
                return
            if side == OrderSide.BUY:
                title = "매수 체결"
                message = (
                    f"{symbol} {qty}주 @ {price:,}원\n"
                    f"매수금액: {(price * qty):,}원\n"
                    f"당일 누적순손익: {daily_net_pnl:,}원"
                )
                level = "info"
            elif side == OrderSide.SELL:
                trade_net_pnl = (
                    int(confirmed_trade_net_pnl)
                    if confirmed_trade_net_pnl is not None
                    else self._confirmed_sell_fill_net_pnl(result)
                )
                if trade_net_pnl is None:
                    logger.error(
                        "확정 매도체결 알림 원장 조회 실패: %s order_no=%s fill_id=%s",
                        symbol,
                        str(result.order_no or "-"),
                        str(result.fill_id or "-"),
                    )
                    return
                message = (
                    f"{symbol} {qty}주 @ {price:,}원\n"
                    f"거래 순손익: {trade_net_pnl:,}원\n"
                    f"당일 누적순손익: {daily_net_pnl:,}원"
                )
                remaining_quantity = max(0, int(result.remaining_quantity or 0))
                if remaining_quantity > 0:
                    message += f"\n미체결 잔량: {remaining_quantity}주"
                    title = "매도 부분체결"
                else:
                    title = "매도 체결"
                level = "info"
            else:
                return
            fill_identity = str(result.fill_id or "").strip()
            if not fill_identity:
                fill_identity = str(result.order_no or "").strip()
            if not fill_identity:
                fill_identity = str(
                    int(result.timestamp.timestamp()) if result.timestamp else 0
                )
            self._alerts.send(
                event_key=f"order_fill_{side.value}_{symbol}_{fill_identity}",
                title=title,
                message=message,
                level=level,
                cooldown_seconds=0,
            )
            return

        # Submission failures are operational log events, not fills. Session-
        # stopping account/market errors and loop failures have dedicated alerts.
        return

    def _send_daily_halt_alert_if_needed(self) -> None:
        if self.strategy.should_continue():
            return
        day_key = datetime.now().date().isoformat()
        if self._last_halt_alert_day == day_key:
            return
        self._last_halt_alert_day = day_key
        realized = int(self.strategy.daily_pnl.realized_net_pnl)
        self._alerts.send(
            event_key=f"strategy_halted_{day_key}",
            title="당일 거래 중지",
            message=f"전략이 당일 거래를 중지했습니다.\n누적 순손익: {realized:,}원",
            level="warning",
            cooldown_seconds=0,
        )

    def _send_loop_error_alert(self, *, scope: str, message: str) -> None:
        now = datetime.now()
        last_sent_at = self._last_loop_error_alert_at
        if last_sent_at is not None and (now - last_sent_at).total_seconds() < 300:
            return
        self._last_loop_error_alert_at = now
        self._alerts.send(
            event_key=f"{scope}_loop_error",
            title="봇 오류 감지",
            message=f"{scope} 오류가 발생했습니다.\n{message}",
            level="error",
            cooldown_seconds=300,
        )

    def _send_session_summary_alert(self, balance, session_pnl: int) -> None:
        if balance is None:
            return
        daily_pnl = self.strategy.daily_pnl
        trade_count = int(daily_pnl.trade_count)
        win_count = int(daily_pnl.win_count)
        loss_count = int(daily_pnl.loss_count)
        positions = self.strategy.positions
        self._alerts.send(
            event_key=f"trading_session_summary_{datetime.now().date().isoformat()}_{self._last_session_end_reason}",
            title="트레이딩 세션 종료",
            message=(
                f"금일 실현손익: {int(session_pnl):,}원\n"
                f"평가금액: {int(balance.total_eval_amount):,}원\n"
                f"거래수: {trade_count}건 (승 {win_count} / 패 {loss_count})\n"
                f"잔여 보유: {len(positions)}종목\n"
                f"종료 사유: {self._last_session_end_reason}"
            ),
            level="info",
            cooldown_seconds=0,
        )

    def _sleep_until_preopen(self):
        """다음 장 준비 시각까지 대기한다."""
        target_preopen = self._next_preopen_after(datetime.now())

        while not self._shutdown:
            now = datetime.now()
            if now >= target_preopen:
                return

            wait = max(0, math.ceil((target_preopen - now).total_seconds()))
            logger.info(
                "장 시간이 아닙니다. 다음 준비 시각: %s (%d분 후)",
                target_preopen.strftime("%H:%M"),
                wait // 60,
            )
            self._interruptible_sleep(min(wait, self.config.off_hours_check_interval))

    def _run_trading_session(self, tick_interval: int) -> bool:
        """장 시간 동안 전략을 실행한다.

        Returns:
            True이면 전략이 오늘 거래 중지를 요청한 것 (재시작 금지).
        """
        logger.info("--- 트레이딩 세션 시작 ---")
        self._session_started_at = datetime.now()
        self._last_stream_stale_log_at = None
        self._last_emergency_position_refresh_at = None
        self._last_tick_activity_log_at = None
        self._last_pending_entry_reconcile_at = None
        self._last_pending_exit_reconcile_at = None

        # 휴장일 체크
        today_key = datetime.now().strftime("%Y%m%d")
        if not self.market_data.is_market_open():
            self._session_started_at = None
            self._market_closed_session_date = today_key
            self._last_session_end_reason = "market_holiday"
            logger.info("오늘은 휴장일입니다. date=%s", today_key)
            self._interruptible_sleep(3600)  # 1시간 뒤 재확인
            return False

        # 전략 초기화
        self._reset_strategy_simulated_clock()
        self.strategy.initialize()
        session_start_delay = self._session_start_balance_delay_seconds()
        if session_start_delay > 0:
            self._interruptible_sleep(session_start_delay)
        balance = self._fetch_balance_with_retry("세션 시작 잔고 조회")
        if balance:
            self.strategy.sync_positions_from_account(balance.positions)
            logger.info(
                "예수금: %s원 | 보유: %d종목",
                f"{balance.total_deposit:,}",
                len(balance.positions),
            )
        else:
            self._session_started_at = None
            self._reset_strategy_simulated_clock()
            logger.error("잔고 조회 실패로 이번 세션의 주문 진행을 중단합니다.")
            self._interruptible_sleep(60)
            logger.info("--- 트레이딩 세션 종료 ---")
            return False

        # A saved same-day pool is only a fallback. Every new process/session
        # starts by discovering the current movers again.
        self._refresh_runtime_pool_if_needed(datetime.now(), force=True)
        watchlist = self.strategy.get_watchlist()
        prioritized_symbols = self._prioritized_stream_symbols(watchlist)
        self._quote_stream.start(prioritized_symbols)
        stream_symbols = self._quote_stream.subscribed_symbols()
        if not isinstance(stream_symbols, (list, tuple, set)):
            stream_symbols = prioritized_symbols
        stream_symbols = list(stream_symbols)
        logger.info(
            "감시 종목: 전체=%d WebSocket=%d KIS멀티보완=%d",
            len(watchlist),
            len(stream_symbols),
            max(0, len(prioritized_symbols) - len(stream_symbols)),
        )
        self._alerts.send(
            event_key=f"trading_session_started_{datetime.now().date().isoformat()}",
            title="트레이딩 세션 시작",
            message=(
                f"{self.config.trading_mode.upper()} 모드로 장중 세션을 시작했습니다.\n"
                f"감시 종목 {len(watchlist)}개"
            ),
            level="info",
            cooldown_seconds=0,
        )

        halted_for_day = False
        strategy_halt_observed = False
        self._last_session_end_reason = "session_end"

        # 틱 루프
        while not self._shutdown and self._is_trading_time(datetime.now()):
            try:
                self._reconcile_pending_entry_positions()
                self._reconcile_pending_exit_positions()
                if not self.strategy.should_continue():
                    halted_for_day = True
                    strategy_halt_observed = True
                    positions = self.strategy.positions
                    if positions:
                        logger.warning(
                            "전략 중지 상태지만 보유 포지션 %d종목이 남아 즉시 청산을 시도합니다.",
                            len(positions),
                        )
                        halt_orders = self.strategy._liquidate_all()
                        if halt_orders:
                            halt_orders = self._filter_pending_duplicate_orders(halt_orders, datetime.now())
                        if halt_orders:
                            results = self.executor.submit_orders(halt_orders)
                            self._handle_order_results(results)
                            if self._stop_session_on_market_closed_order_failure(results):
                                break
                            if self._stop_session_on_account_order_unavailable(results):
                                halted_for_day = True
                                break
                            self._interruptible_sleep(max(2, tick_interval))
                            continue
                        pending_exit_symbols = self._pending_sell_block_symbols_for_positions(positions, datetime.now())
                        pending_balance = self._fetch_balance_with_retry(
                            "전략 중지 청산 잔고 확인",
                            max_attempts=1,
                            base_delay_seconds=1,
                        )
                        if pending_balance is not None:
                            self.strategy.sync_positions_from_account(pending_balance.positions)
                        remaining_positions = self.strategy.positions
                        if remaining_positions:
                            waiting_symbols = pending_exit_symbols or sorted(str(symbol) for symbol in remaining_positions)
                            logger.warning(
                                "전략 중지 청산 잔고 확인 대기: %s",
                                ",".join(waiting_symbols),
                            )
                            self._interruptible_sleep(max(2, tick_interval))
                            continue
                    logger.info("전략이 종료를 요청했습니다.")
                    break

                # 동적 발견은 KIS 순위를, 예측 시세는 WebSocket과 저빈도
                # KIS 멀티시세 보완을 함께 사용한다.
                self._refresh_runtime_pool_if_needed(datetime.now())
                watchlist = self.strategy.get_watchlist()
                prioritized_symbols = self._prioritized_stream_symbols(watchlist)
                self._quote_stream.update_symbols(prioritized_symbols)
                stream_symbols = self._quote_stream.subscribed_symbols()
                if not isinstance(stream_symbols, (list, tuple, set)):
                    stream_symbols = prioritized_symbols
                stream_symbols = list(stream_symbols)
                all_quotes = self._quote_stream.drain_quotes()
                tick_now = datetime.now()
                all_quotes.extend(
                    self._supplemental_watchlist_quotes(
                        tick_now,
                        prioritized_symbols,
                        stream_symbols,
                    )
                )
                all_quotes.extend(self._emergency_position_quotes(tick_now))

                if all_quotes:
                    for q in all_quotes:
                        logger.debug(
                            "[%s] %s원",
                            q.symbol,
                            f"{q.current_price:,}",
                        )

                    orders = self.strategy.on_batch_tick(all_quotes)
                    self._log_tick_activity(datetime.now(), watchlist, stream_symbols, all_quotes)
                    if orders:
                        orders = self._filter_pending_duplicate_orders(orders, datetime.now())
                    if orders:
                        logger.info("주문 %d건 제출", len(orders))
                        results = self.executor.submit_orders(orders)
                        self._handle_order_results(results)
                        if self._stop_session_on_market_closed_order_failure(results):
                            break
                        if self._stop_session_on_account_order_unavailable(results):
                            halted_for_day = True
                            break
                else:
                    self._log_tick_activity(datetime.now(), watchlist, stream_symbols, all_quotes)

                self._interruptible_sleep(1)
            except requests.exceptions.RequestException as e:
                if self._shutdown:
                    break
                logger.warning(
                    "장중 네트워크 오류: %s (10초 후 재시도)",
                    e,
                )
                self._interruptible_sleep(max(10, tick_interval))
            except Exception:
                if self._shutdown:
                    break
                logger.exception("장중 루프 오류 (10초 후 재시도)")
                self._send_loop_error_alert(scope="장중 루프", message="자세한 내용은 trading.log를 확인해 주십시오.")
                self._interruptible_sleep(max(10, tick_interval))

        self._quote_stream.stop()
        logger.info("--- 트레이딩 세션 종료 ---")
        self._session_started_at = None

        self._confirm_reconciled_sell_prices()
        if strategy_halt_observed and halted_for_day:
            try:
                strategy_resumed = bool(self.strategy.should_continue())
            except Exception:
                strategy_resumed = False
                logger.exception("확정 체결손익 반영 후 전략 지속 여부 확인 실패")
            account_blocked_today = (
                self._account_order_blocked_session_date
                == datetime.now().strftime("%Y%m%d")
            )
            if strategy_resumed and not account_blocked_today:
                halted_for_day = False
                logger.warning("확정 체결손익 반영 후 당일 정지가 해제되어 장중 평가를 재개합니다.")

        # 종료 시 잔고 요약
        balance = self._fetch_balance_with_retry("세션 종료 잔고 조회", max_attempts=2, base_delay_seconds=1)
        if balance:
            session_pnl = self._resolve_session_profit_loss(balance)
            logger.info(
                "최종 잔고 — 평가금액: %s원 | 금일 실현손익: %s원",
                f"{balance.total_eval_amount:,}",
                f"{session_pnl:,}",
            )
            self._write_performance_reports(balance, session_pnl)
            self._send_session_summary_alert(balance, session_pnl)
        else:
            logger.warning("세션 종료 잔고 조회 실패: 오늘 요약 로그를 생략합니다.")
        self._reset_strategy_simulated_clock()
        return halted_for_day

    def _write_performance_reports(self, balance, session_pnl: int):
        try:
            paths = update_performance_reports(
                strategy=self.strategy,
                balance=balance,
                session_pnl=session_pnl,
                trading_mode=self.config.trading_mode,
            )
            logger.info(
                "성과 리포트 갱신: score=%s, gate=%s",
                paths["scorecard"]["json"],
                paths["readiness"]["json"],
            )
            self._last_session_end_reason = self._resolve_session_end_reason(paths["scorecard"]["json"])
        except Exception:
            logger.exception("성과 리포트 생성 실패")
            self._last_session_end_reason = "session_end"

    def _resolve_session_end_reason(self, scorecard_path) -> str:
        try:
            payload = json.loads(Path(scorecard_path).read_text(encoding="utf-8"))
        except Exception:
            return "session_end"

        halt_reason = str(payload.get("strategy", {}).get("halt_reason") or "")
        if halt_reason == "daily_total_loss_limit":
            return "daily_hard_stop"
        if halt_reason == "daily_profit_target":
            return "daily_profit_target"
        return "session_end"

    def _resolve_session_profit_loss(self, balance) -> int:
        """세션 종료 시 금일 누적 실현손익을 결정한다.

        우선순위:
        1) 계좌 API 당일 실현손익
        2) 확정 매도체결 원장에서 만든 전략 당일 누적 순손익
        3) 잔고 API 평가손익 (fallback)
        """
        realized_pnl = self.account.get_realized_profit_loss()
        if realized_pnl is not None:
            return realized_pnl

        strategy_pnl = self._extract_strategy_realized_pnl()
        if strategy_pnl is not None:
            logger.warning(
                "실현손익 API 조회 실패로 확정 매도체결 원장의 금일 누적 순손익을 사용합니다: %s원",
                f"{strategy_pnl:,}",
            )
            return strategy_pnl

        logger.warning(
            "실현손익 집계 실패로 평가손익 fallback을 사용합니다: %s원",
            f"{balance.total_profit_loss:,}",
        )
        return int(balance.total_profit_loss)

    def _extract_strategy_realized_pnl(self):
        """전략 객체에서 누적 순손익을 추출한다."""
        try:
            return int(self.strategy.daily_pnl.realized_net_pnl)
        except (AttributeError, TypeError, ValueError):
            return None

    def _fetch_balance_with_retry(
        self,
        context: str,
        max_attempts: int = None,
        base_delay_seconds: int = None,
    ):
        """잔고 조회를 실패 시 재시도한다."""
        attempts = max_attempts or self._balance_retry_attempts
        delay = base_delay_seconds or self._balance_retry_delay_seconds

        for i in range(1, attempts + 1):
            failure_detail = ""
            try:
                balance = self.account.get_balance()
            except requests.exceptions.RequestException as exc:
                balance = None
                failure_detail = f" 네트워크 오류: {exc}"
            if balance is not None:
                return balance

            if i < attempts and not self._shutdown:
                wait = delay * (2 ** (i - 1))
                cooldown_wait = self._client_rate_limit_cooldown()
                if cooldown_wait > 0:
                    wait = max(wait, int(math.ceil(cooldown_wait)))
                logger.warning(
                    "%s 실패 (%d/%d):%s %d초 후 재시도",
                    context,
                    i,
                    attempts,
                    failure_detail,
                    wait,
                )
                self._interruptible_sleep(wait)

        return None
