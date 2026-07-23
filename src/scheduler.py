import copy
import logging
import math
import time
import json
from datetime import datetime, time as clock_time, timedelta
from pathlib import Path
from typing import List, Optional

import requests

from src.account import AccountAPI
from src.api_client import KISClient, is_kis_rate_limited_message
from src.auth import TokenManager
from src.config import Config
from src.executor import OrderExecutor, RiskManager
from src.logger_setup import setup_logger
from src.market_calendar import is_krx_regular_trading_day, normalize_trading_date
from src.market_data import MarketDataAPI
from src.performance_reporting import update_performance_reports
from src.notifications import AlertManager
from src.models import Order, OrderResult, OrderSide
from src.strategy import BaseStrategy
from src.trading import TradingAPI

logger = logging.getLogger("kis_trader.scheduler")

# 장 시간
MARKET_OPEN = (9, 0)    # 09:00
MARKET_CLOSE = (15, 30)  # 15:30
PRE_OPEN = (8, 50)       # 장 시작 10분 전 준비


class TradingScheduler:
    """매일 장 시간에 맞춰 전략을 자동 실행하는 스케줄러."""

    def __init__(self, strategy: BaseStrategy, config: Config = None):
        self.strategy = strategy
        shared_market_data = getattr(strategy, "market_data", None)
        shared_client = getattr(shared_market_data, "client", None)
        shared_config = getattr(shared_client, "config", None)
        if shared_client is not None and config is not None and shared_config is not None and config is not shared_config:
            logger.warning("전략 런타임 config와 스케줄러 전달 config가 달라 전략 쪽 config를 우선 사용합니다.")
        self.config = shared_config or config or Config.load()
        self._shutdown = False

        # 런타임 컴포넌트는 가능한 한 전략과 공유해 유량 제한/캐시 상태를 일치시킨다.
        if shared_client is not None:
            self.client = shared_client
            self.token_mgr = getattr(self.client, "token_manager", None)
            self.market_data = shared_market_data
        else:
            self.token_mgr = TokenManager(self.config)
            self.client = KISClient(self.config, self.token_mgr)
            self.market_data = MarketDataAPI(self.client)
            if hasattr(self.strategy, "market_data"):
                self.strategy.market_data = self.market_data

        self.trading = TradingAPI(self.client)
        self.account = AccountAPI(self.client)
        self.executor = OrderExecutor(self.trading, RiskManager())
        self._alerts = AlertManager()

        self._balance_retry_attempts = 3
        self._balance_retry_delay_seconds = 2
        self._last_session_end_reason = "session_end"
        self._exit_priority_until = None
        self._last_exit_priority_log_at = None
        self._last_rate_limit_watchlist_log_at = None
        self._last_halt_alert_day = None
        self._watchlist_rotation_cursor = 0
        self._last_runtime_pool_refresh_at = None
        self._session_started_at = None
        self._last_cached_quote_fallback_log_at = None
        self._last_empty_quote_batch_log_at = None
        self._last_tick_activity_log_at = None
        self._last_loop_error_alert_at = None
        self._market_closed_session_date = None
        self._account_order_blocked_session_date = None
        self._pending_order_blocks = {}

    def stop(self):
        self._shutdown = True

    def run(self, tick_interval: int = 10):
        """스케줄러를 시작한다. Ctrl+C로 종료.

        Args:
            tick_interval: 전략 틱 간격 (초). 기본 10초.
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
            logger.info("스케줄러 종료")

    def _is_trading_time(self, now: datetime) -> bool:
        """현재가 거래 가능 시간인지 확인."""
        open_time = clock_time(hour=MARKET_OPEN[0], minute=MARKET_OPEN[1])
        close_time = clock_time(hour=MARKET_CLOSE[0], minute=MARKET_CLOSE[1])
        return self._is_regular_trading_day(now) and open_time <= now.time() < close_time

    def _is_preopen_time(self, now: datetime) -> bool:
        """장 시작 전 준비 시간인지 확인."""
        preopen_time = clock_time(hour=PRE_OPEN[0], minute=PRE_OPEN[1])
        open_time = clock_time(hour=MARKET_OPEN[0], minute=MARKET_OPEN[1])
        return self._is_regular_trading_day(now) and preopen_time <= now.time() < open_time

    def _is_regular_trading_day(self, now: datetime) -> bool:
        date_key = normalize_trading_date(now)
        if getattr(self, "_market_closed_session_date", None) == date_key:
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
        reset_fn = getattr(self.strategy, "set_simulated_now", None)
        if not callable(reset_fn):
            return
        try:
            reset_fn(None)
        except Exception:
            logger.exception("전략 시뮬레이션 시각 초기화 실패")

    @staticmethod
    def _is_rate_limited_message(message: str) -> bool:
        return is_kis_rate_limited_message(message)

    def _client_rate_limit_cooldown(self) -> float:
        cooldown_fn = getattr(self.client, "rate_limit_cooldown_remaining", None)
        if not callable(cooldown_fn):
            return 0.0
        try:
            return max(0.0, float(cooldown_fn()))
        except Exception:
            return 0.0

    def _multi_price_backoff_remaining(self) -> float:
        market_data = getattr(self, "market_data", None)
        remaining_fn = getattr(market_data, "multi_price_backoff_remaining", None)
        if not callable(remaining_fn):
            return 0.0
        try:
            return max(0.0, float(remaining_fn()))
        except Exception:
            return 0.0

    def _mark_exit_priority(self, now: datetime, *, seconds: int = 120) -> None:
        until = now + timedelta(seconds=max(10, seconds))
        if self._exit_priority_until is None or until > self._exit_priority_until:
            self._exit_priority_until = until

    def _exit_priority_active(self, now: datetime) -> bool:
        positions = getattr(self.strategy, "positions", {}) or {}
        if not positions:
            self._exit_priority_until = None
            return False
        if self._exit_priority_until and now < self._exit_priority_until:
            return True
        if self._exit_priority_until and now >= self._exit_priority_until:
            self._exit_priority_until = None
        return False

    def _effective_watchlist(self, strategy_watchlist: List[str], now: datetime) -> List[str]:
        positions = getattr(self.strategy, "positions", {}) or {}
        client_cooldown_remaining = self._client_rate_limit_cooldown()
        multi_price_backoff_remaining = self._multi_price_backoff_remaining()
        cooldown_remaining = max(client_cooldown_remaining, multi_price_backoff_remaining)
        held_symbols = list(dict.fromkeys(str(symbol) for symbol in positions.keys() if symbol))

        if not positions:
            self._exit_priority_until = None
            if client_cooldown_remaining > 0:
                last_log_at = getattr(self, "_last_rate_limit_watchlist_log_at", None)
                should_log = (
                    last_log_at is None
                    or (now - last_log_at).total_seconds() >= 30
                )
                if should_log:
                    logger.warning(
                        "API 냉각 모드: 신규 감시를 %.1fs 동안 건너뜁니다. (대상 %d종목)",
                        client_cooldown_remaining,
                        len(strategy_watchlist),
                    )
                    self._last_rate_limit_watchlist_log_at = now
                return []
            if multi_price_backoff_remaining > 0:
                should_log = (
                    self._last_rate_limit_watchlist_log_at is None
                    or (now - self._last_rate_limit_watchlist_log_at).total_seconds() >= 30
                )
                if should_log:
                    logger.warning(
                        "멀티시세 냉각 모드: %.1fs 동안 캐시 우선 감시를 유지합니다.",
                        multi_price_backoff_remaining,
                    )
                    self._last_rate_limit_watchlist_log_at = now
                return strategy_watchlist
            return strategy_watchlist

        if cooldown_remaining > 0:
            self._mark_exit_priority(now)
        if not self._exit_priority_active(now):
            return strategy_watchlist
        if not held_symbols:
            return strategy_watchlist
        should_log = (
            self._last_exit_priority_log_at is None
            or (now - self._last_exit_priority_log_at).total_seconds() >= 30
        )
        if should_log:
            logger.warning(
                "청산 우선 모드: 유량 제한 회복까지 보유 종목 %d개만 감시합니다.",
                len(held_symbols),
            )
            self._last_exit_priority_log_at = now
        return held_symbols

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

    def _quote_cache_age_seconds(self, symbol: str):
        market_data = getattr(self, "market_data", None)
        age_fn = getattr(market_data, "quote_cache_age_seconds", None)
        if not callable(age_fn):
            return None
        try:
            return age_fn(symbol)
        except Exception:
            return None

    def _session_age_seconds(self, now: datetime):
        started_at = getattr(self, "_session_started_at", None)
        if started_at is None:
            return None
        return max(0.0, float((now - started_at).total_seconds()))

    def _market_data_pressure_active(self) -> bool:
        fluctuation_backoff_fn = getattr(self.market_data, "fluctuation_backoff_remaining", None)
        fluctuation_backoff = 0.0
        if callable(fluctuation_backoff_fn):
            try:
                fluctuation_backoff = max(0.0, float(fluctuation_backoff_fn()))
            except Exception:
                fluctuation_backoff = 0.0
        return (
            self._client_rate_limit_cooldown() > 0
            or self._multi_price_backoff_remaining() > 0
            or fluctuation_backoff > 0
        )

    def _session_start_balance_delay_seconds(self) -> float:
        cfg = getattr(self, "config", None)
        is_paper = bool(getattr(cfg, "is_paper", False))
        base_delay = 2.0 if is_paper else 0.25
        return max(base_delay, self._client_rate_limit_cooldown())

    @staticmethod
    def _minutes_since_market_open(now: datetime) -> int:
        market_open = now.replace(hour=MARKET_OPEN[0], minute=MARKET_OPEN[1], second=0, microsecond=0)
        return max(0, int((now - market_open).total_seconds() // 60))

    def _refresh_runtime_pool_if_needed(self, now: datetime, *, force: bool = False) -> bool:
        strategy = getattr(self, "strategy", None)
        if strategy is None or not hasattr(strategy, "update_runtime_pool"):
            return False
        if getattr(strategy, "pool_override", None):
            return False
        cfg = getattr(strategy, "config", None)
        if cfg is None:
            return False
        if self._market_data_pressure_active():
            return False
        minutes_since_open = self._minutes_since_market_open(now)
        opening_window = int(getattr(cfg, "opening_candidate_window_minutes", 20) or 20)
        opening_fast_window = int(getattr(cfg, "opening_fast_window_minutes", 3) or 3)
        opening_fast_mode = minutes_since_open < opening_fast_window
        session_age_seconds = self._session_age_seconds(now)
        if not force and session_age_seconds is not None:
            warmup_seconds = 10.0
            if opening_fast_mode:
                warmup_seconds = float(getattr(cfg, "opening_fast_pool_warmup_seconds", 2.0) or 2.0)
            if session_age_seconds < warmup_seconds:
                return False

        min_change_rate = float(getattr(cfg, "dynamic_pool_quote_min_change_rate", 0.0) or 0.0)
        if minutes_since_open < opening_fast_window:
            refresh_interval = max(10, int(getattr(cfg, "opening_fast_pool_refresh_interval_seconds", 20) or 20))
            fetch_count = int(getattr(cfg, "opening_fast_fetch_count", getattr(cfg, "dynamic_pool_ranking_fetch_count", 30)) or 30)
            min_change_rate = min(
                min_change_rate,
                float(getattr(cfg, "opening_fast_min_change_rate", min_change_rate) or 0.0),
            )
        elif minutes_since_open < opening_window:
            refresh_interval = max(20, int(getattr(cfg, "opening_candidate_pool_refresh_interval_seconds", 45) or 45))
            fetch_count = int(getattr(cfg, "opening_candidate_fetch_count", getattr(cfg, "dynamic_pool_ranking_fetch_count", 30)) or 30)
            min_change_rate = min(
                min_change_rate,
                float(getattr(cfg, "opening_candidate_min_change_rate", min_change_rate) or 0.0),
            )
        else:
            refresh_interval = max(120, int(getattr(cfg, "pool_refresh_interval", 300) or 300))
            fetch_count = int(getattr(cfg, "dynamic_pool_ranking_fetch_count", 30) or 30)

        last_refreshed = getattr(self, "_last_runtime_pool_refresh_at", None)
        if last_refreshed is not None and (now - last_refreshed).total_seconds() < refresh_interval:
            return False

        # Discovery should be wider than entry judgment: do not exclude low-price
        # or already-fast movers before the EV planner can evaluate them.
        ranking = self.market_data.get_fluctuation_ranking(
            count=fetch_count,
            min_change_rate=min_change_rate,
            max_change_rate=30.0,
            min_price=0,
            min_volume=0,
        )
        symbols = [item.symbol for item in ranking if str(getattr(item, "symbol", "") or "").strip()]
        if not symbols:
            self._last_runtime_pool_refresh_at = now
            logger.info("동적풀 갱신 생략: 등락률 후보 없음, 기존 감시풀 유지")
            return False

        strategy.update_runtime_pool(symbols)
        self._last_runtime_pool_refresh_at = now
        logger.info(
            "동적풀 갱신: %d개 (%s)",
            len(symbols),
            ", ".join(symbols[:5]),
        )
        return True

    def _log_tick_activity(self, now: datetime, watchlist: List[str], refresh_batch: List[str], quotes: List) -> None:
        last_log_at = getattr(self, "_last_tick_activity_log_at", None)
        should_log = (
            last_log_at is None
            or (now - last_log_at).total_seconds() >= 60
        )
        if not should_log:
            return
        positions = getattr(self.strategy, "positions", {}) or {}
        long_candidates = 0
        cached_shortlist = getattr(self.strategy, "_last_long_shortlist_symbols", None)
        shortlist_fn = getattr(self.strategy, "_long_entry_shortlist", None)
        fresh_market_quotes_fn = getattr(self.strategy, "_fresh_market_state_quotes", None)
        if quotes and isinstance(cached_shortlist, list):
            long_candidates = len(cached_shortlist)
        elif callable(shortlist_fn) and quotes:
            try:
                shortlist_input = quotes
                if callable(fresh_market_quotes_fn):
                    expanded_quotes = fresh_market_quotes_fn(quotes)
                    if expanded_quotes:
                        shortlist_input = expanded_quotes
                long_candidates = len(shortlist_fn(shortlist_input))
            except Exception:
                long_candidates = 0
        logger.info(
            "틱 상태: 감시=%d 배치=%d 수신=%d 롱후보=%d 보유=%d",
            len(watchlist),
            len(refresh_batch),
            len(quotes),
            long_candidates,
            len(positions),
        )
        self._last_tick_activity_log_at = now

    def _round_robin_symbols(self, symbols: List[str], limit: int) -> List[str]:
        normalized = self._normalize_symbols(symbols)
        if limit <= 0 or not normalized:
            return []
        cursor = int(getattr(self, "_watchlist_rotation_cursor", 0) or 0)
        start = cursor % len(normalized)
        selected = [
            normalized[(start + offset) % len(normalized)]
            for offset in range(min(limit, len(normalized)))
        ]
        self._watchlist_rotation_cursor = (start + len(selected)) % len(normalized)
        return selected

    def _priority_refresh_symbols(self, strategy_watchlist: List[str]) -> List[str]:
        positions = getattr(self.strategy, "positions", {}) or {}
        strategy = getattr(self, "strategy", None)
        opening_fast = sorted(getattr(strategy, "_latest_opening_fast_symbols", set()) or [])
        opening_hot = sorted(getattr(strategy, "_latest_opening_hot_symbols", set()) or [])
        math_queue = list(getattr(strategy, "_latest_math_queue_symbols", []) or [])[:4]
        math_backfill = list(getattr(strategy, "_latest_math_backfill_symbols", []) or [])[:2]
        runtime_pool = list(getattr(strategy, "_pool", []) or [])[:6]
        valid_symbols = set(self._normalize_symbols(strategy_watchlist)) | set(str(symbol) for symbol in positions.keys())
        priority = self._normalize_symbols(
            list(positions.keys()) + opening_fast + opening_hot + math_queue + math_backfill + runtime_pool
        )
        return [symbol for symbol in priority if symbol in valid_symbols]

    def _build_quote_refresh_batch(self, strategy_watchlist: List[str], now: datetime) -> List[str]:
        watchlist = self._normalize_symbols(strategy_watchlist)
        if not watchlist:
            return []

        positions = getattr(self.strategy, "positions", {}) or {}
        has_positions = bool(positions)
        session_age_seconds = self._session_age_seconds(now)
        minutes_since_open = self._minutes_since_market_open(now)
        opening_mode = minutes_since_open < 5
        cfg = getattr(getattr(self, "strategy", None), "config", None)
        opening_fast_window = int(getattr(cfg, "opening_fast_window_minutes", 3) or 3)
        opening_fast_mode = minutes_since_open < opening_fast_window
        if not has_positions and session_age_seconds is not None:
            warmup_seconds = 8.0
            if opening_fast_mode:
                warmup_seconds = float(getattr(cfg, "opening_fast_quote_warmup_seconds", 2.0) or 2.0)
            if session_age_seconds < warmup_seconds:
                return []
        total_budget = 18 if opening_mode else 12
        if has_positions:
            total_budget = 14 if opening_mode else 10
        elif session_age_seconds is not None and session_age_seconds < 60.0:
            if opening_fast_mode:
                total_budget = min(
                    total_budget,
                    max(1, int(getattr(cfg, "opening_fast_initial_quote_budget", 14) or 14)),
                )
            elif opening_mode:
                total_budget = min(
                    total_budget,
                    max(1, int(getattr(cfg, "opening_candidate_initial_quote_budget", 10) or 10)),
                )
            else:
                total_budget = min(total_budget, 6)
        elif self._client_rate_limit_cooldown() > 0:
            total_budget = min(total_budget, 6)
        priority_ttl = 12.0 if opening_mode else 18.0
        background_ttl = 30.0 if opening_mode else 60.0

        refresh_batch: List[str] = []
        for symbol in self._priority_refresh_symbols(watchlist):
            is_position = symbol in positions
            if is_position:
                refresh_batch.append(symbol)
            else:
                age_seconds = self._quote_cache_age_seconds(symbol)
                if age_seconds is None or age_seconds >= priority_ttl:
                    refresh_batch.append(symbol)
            if len(refresh_batch) >= total_budget:
                return refresh_batch[:30]

        remaining = [symbol for symbol in watchlist if symbol not in refresh_batch]
        background_slots = max(0, total_budget - len(refresh_batch))
        stale_background = []
        for symbol in remaining:
            age_seconds = self._quote_cache_age_seconds(symbol)
            if age_seconds is None or float(age_seconds) >= background_ttl:
                stale_background.append(symbol)
        selected_background = self._round_robin_symbols(
            stale_background if stale_background else remaining,
            background_slots,
        )
        for symbol in selected_background:
            if symbol not in refresh_batch:
                refresh_batch.append(symbol)
        return refresh_batch[:30]

    def _load_quotes_for_refresh_batch(self, refresh_batch: List[str], now: datetime) -> List:
        if not refresh_batch:
            return []

        all_quotes = []
        for i in range(0, len(refresh_batch), 30):
            chunk = refresh_batch[i:i + 30]
            chunk_quotes = self.market_data.get_multi_price(chunk)
            all_quotes.extend(chunk_quotes)

        if all_quotes:
            return all_quotes

        cached_quotes = []
        cached_getter = getattr(self.market_data, "get_cached_quotes", None)
        if callable(cached_getter):
            positions = getattr(self.strategy, "positions", {}) or {}
            ttl_seconds = 20.0 if positions else 30.0
            cached_quotes = cached_getter(refresh_batch, ttl_seconds=ttl_seconds)

        if cached_quotes:
            last_log_at = getattr(self, "_last_cached_quote_fallback_log_at", None)
            should_log = (
                last_log_at is None
                or (now - last_log_at).total_seconds() >= 60
            )
            if should_log:
                logger.warning(
                    "실시간 시세 배치 실패로 최근 캐시 %d종목을 사용합니다.",
                    len(cached_quotes),
                )
                self._last_cached_quote_fallback_log_at = now
            return list(cached_quotes)

        last_log_at = getattr(self, "_last_empty_quote_batch_log_at", None)
        should_log = (
            last_log_at is None
            or (now - last_log_at).total_seconds() >= 60
        )
        if should_log:
            logger.warning(
                "시세 배치를 확보하지 못해 이번 틱의 신규 평가를 건너뜁니다. (대상 %d종목)",
                len(refresh_batch),
            )
            self._last_empty_quote_batch_log_at = now
        return []

    def _update_exit_priority_from_results(self, results: List[OrderResult], now: datetime) -> None:
        if not results:
            return
        if any(
            (not result.success)
            and result.side == OrderSide.SELL
            and self._is_rate_limited_message(result.message)
            for result in results
        ):
            self._mark_exit_priority(now, seconds=180)
            return
        if any(
            (not result.success)
            and self._is_rate_limited_message(result.message)
            for result in results
        ):
            self._mark_exit_priority(now, seconds=90)

    @staticmethod
    def _order_block_side_value(side) -> str:
        if isinstance(side, OrderSide):
            return side.value
        return str(side or "").strip()

    @classmethod
    def _order_block_key(cls, item) -> Optional[tuple]:
        symbol = str(getattr(item, "symbol", "") or "").strip()
        side_value = cls._order_block_side_value(getattr(item, "side", None))
        if not symbol or side_value not in {OrderSide.BUY.value, OrderSide.SELL.value}:
            return None
        return (symbol, side_value)

    def _pending_order_block_seconds(self, item) -> int:
        return max(30, int(getattr(self.config, "pending_order_block_seconds", 180) or 180))

    def _purge_expired_pending_order_blocks(self, now: datetime) -> None:
        blocks = getattr(self, "_pending_order_blocks", None)
        if not isinstance(blocks, dict):
            self._pending_order_blocks = {}
            return
        for key, payload in list(blocks.items()):
            expires_at = payload.get("expires_at") if isinstance(payload, dict) else payload
            if expires_at is not None and expires_at <= now:
                blocks.pop(key, None)

    def _filter_pending_duplicate_orders(self, orders: List[Order], now: Optional[datetime] = None) -> List[Order]:
        if not orders:
            return []
        now = now or datetime.now()
        self._purge_expired_pending_order_blocks(now)
        blocks = getattr(self, "_pending_order_blocks", {})
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
                    self._order_block_side_value(getattr(order, "side", None)),
                    str(getattr(order, "symbol", "") or ""),
                    int(getattr(order, "quantity", 0) or 0),
                )
                continue
            if key in blocks:
                allow_unreserved_sell = False
                if getattr(order, "side", None) == OrderSide.SELL:
                    symbol = str(getattr(order, "symbol", "") or "").strip()
                    position = getattr(self.strategy, "positions", {}).get(symbol)
                    if position is not None:
                        position_quantity = max(0, int(getattr(position, "quantity", 0) or 0))
                        pending_quantity = max(0, int(getattr(position, "pending_exit_quantity", 0) or 0))
                        unreserved_quantity = max(0, position_quantity - min(position_quantity, pending_quantity))
                        order_quantity = max(0, int(getattr(order, "quantity", 0) or 0))
                        allow_unreserved_sell = 0 < order_quantity <= unreserved_quantity
                if not allow_unreserved_sell:
                    logger.warning(
                        "pending 주문 중복 차단: %s %s %d주 (이전 주문 체결 확인 대기)",
                        self._order_block_side_value(getattr(order, "side", None)),
                        str(getattr(order, "symbol", "") or ""),
                        int(getattr(order, "quantity", 0) or 0),
                    )
                    continue
            seen.add(key)
            filtered.append(order)
        return filtered

    @staticmethod
    def _result_requires_balance_reconcile(result: OrderResult) -> bool:
        if result is None:
            return False
        if (
            not getattr(result, "success", False)
            and getattr(result, "side", None) == OrderSide.SELL
            and str(getattr(result, "error_category", "") or "") == "no_holding"
        ):
            return True
        if not getattr(result, "success", False):
            return False
        fill_mode = str(getattr(result, "fill_mode", "") or "")
        quantity = int(getattr(result, "quantity", 0) or 0)
        price = int(getattr(result, "price", 0) or 0)
        requested_price = int(getattr(result, "requested_price", 0) or 0)
        if fill_mode in {"market_pending", "limit_then_market_pending", "partial_fill_pending"}:
            return True
        requested_quantity = max(0, int(getattr(result, "requested_quantity", 0) or 0))
        if requested_quantity > 0 and quantity < requested_quantity:
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
        blocks = getattr(self, "_pending_order_blocks", None)
        if not isinstance(blocks, dict):
            self._pending_order_blocks = {}
            blocks = self._pending_order_blocks
        for result in results:
            key = self._order_block_key(result)
            if key is None:
                continue
            if getattr(result, "success", False) and self._result_requires_balance_reconcile(result):
                blocks[key] = {
                    "expires_at": now + timedelta(seconds=self._pending_order_block_seconds(result)),
                    "order_no": str(getattr(result, "order_no", "") or ""),
                    "fill_mode": str(getattr(result, "fill_mode", "") or ""),
                    "requested_quantity": max(0, int(getattr(result, "requested_quantity", 0) or 0)),
                }
            else:
                blocks.pop(key, None)

    def _clear_pending_order_blocks_for_results(self, results: List[OrderResult]) -> None:
        if not results:
            return
        blocks = getattr(self, "_pending_order_blocks", None)
        if not isinstance(blocks, dict):
            return
        for result in results:
            key = self._order_block_key(result)
            if key is not None and not self._result_requires_balance_reconcile(result):
                blocks.pop(key, None)

    def _pending_sell_block_symbols_for_positions(self, positions, now: Optional[datetime] = None) -> List[str]:
        if not positions:
            return []
        now = now or datetime.now()
        self._purge_expired_pending_order_blocks(now)
        if isinstance(positions, dict):
            position_symbols = {
                str(symbol or getattr(position, "symbol", "") or "").strip()
                for symbol, position in positions.items()
            }
        else:
            position_symbols = {str(getattr(position, "symbol", "") or "").strip() for position in positions}
        position_symbols.discard("")
        if not position_symbols:
            return []
        blocks = getattr(self, "_pending_order_blocks", {})
        blocked_symbols = []
        for key in blocks.keys():
            if not isinstance(key, tuple) or len(key) < 2:
                continue
            symbol, side_value = key[0], key[1]
            if side_value == OrderSide.SELL.value and symbol in position_symbols:
                blocked_symbols.append(symbol)
        return sorted(set(blocked_symbols))

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
            and str(getattr(result, "error_category", "") or "") == "no_holding"
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
        reconcile_fn = getattr(self.strategy, "reconcile_pending_fills_from_account", None)
        if callable(reconcile_fn):
            try:
                inferred_results = list(reconcile_fn(results, balance.positions) or [])
            except Exception:
                logger.exception("pending 체결 계좌 보정 실패")
        no_holding_reconcile_fn = getattr(self.strategy, "reconcile_no_holding_sell_failures_from_account", None)
        if callable(no_holding_reconcile_fn):
            try:
                inferred_results.extend(list(no_holding_reconcile_fn(results, balance.positions) or []))
            except Exception:
                logger.exception("무보유 매도 실패 계좌 보정 실패")
        self._confirm_reconciled_sell_prices(inferred_results)
        for inferred in inferred_results:
            symbol = str(getattr(inferred, "symbol", "") or "")
            before_position = None
            if getattr(inferred, "side", None) == OrderSide.SELL and before_positions:
                before_position = before_positions.get(symbol)
            after_position = getattr(self.strategy, "positions", {}).get(symbol)
            self._send_order_result_alert(
                inferred,
                before_position=before_position,
                after_position=after_position,
            )
        if hasattr(self.strategy, "sync_positions_from_account"):
            self.strategy.sync_positions_from_account(balance.positions)
            logger.debug(
                "계좌 기준 재동기화 완료: 보유 %d종목",
                len(balance.positions),
            )
        return inferred_results

    def _stop_session_on_market_closed_order_failure(self, results: List[OrderResult]) -> bool:
        if not any(
            (not result.success)
            and str(getattr(result, "error_category", "") or "") == "market_closed"
            for result in (results or [])
        ):
            return False

        date_key = datetime.now().strftime("%Y%m%d")
        self._market_closed_session_date = date_key
        self._last_session_end_reason = "market_holiday"
        logger.warning("주문 API가 비영업일/휴장 응답을 반환했습니다. date=%s 세션을 중단합니다.", date_key)
        if getattr(self, "_alerts", None):
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
            and str(getattr(result, "error_category", "") or "") == "account_order_unavailable"
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
        if getattr(self, "_alerts", None):
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
        confirm_fn = getattr(self.strategy, "confirm_reconciled_sell_fills", None)
        if not callable(confirm_fn):
            return
        try:
            corrections = list(confirm_fn(self.account, results=results) or [])
        except Exception:
            logger.exception("계좌 재동기화 매도 체결가 재확정 실패")
            return
        for correction in corrections:
            logger.info(
                "계좌 재동기화 매도 체결가 확정: %s order_no=%s %d주 @ %d원 (기존 %d원, 순손익 %+d원 보정)",
                str(correction.get("symbol", "") or ""),
                str(correction.get("order_no", "") or "-"),
                int(correction.get("quantity", 0) or 0),
                int(correction.get("corrected_price", 0) or 0),
                int(correction.get("previous_price", 0) or 0),
                int(correction.get("delta_net_pnl", 0) or 0),
            )

    def _handle_order_results(self, results: List[OrderResult]) -> None:
        if not results:
            return
        self._update_exit_priority_from_results(results, datetime.now())
        self._remember_pending_order_results(results, datetime.now())
        before_positions = {}
        for result in results:
            before_position = getattr(self.strategy, "positions", {}).get(result.symbol)
            if before_position is not None:
                before_positions.setdefault(result.symbol, copy.copy(before_position))
            self.strategy.on_order_filled(result)
            after_position = getattr(self.strategy, "positions", {}).get(result.symbol)
            self._send_order_result_alert(result, before_position=before_position, after_position=after_position)
        inferred_results = self._reconcile_positions_after_order_failures(results, before_positions=before_positions)
        self._clear_pending_order_blocks_for_results(inferred_results)
        self._send_daily_halt_alert_if_needed()

    @staticmethod
    def _alert_fill_price(result: OrderResult, before_position=None, after_position=None) -> int:
        price = int(
            getattr(result, "price", 0)
            or getattr(result, "requested_price", 0)
            or getattr(result, "reference_price", 0)
            or 0
        )
        if price > 0:
            return price
        if after_position is not None:
            price = int(getattr(after_position, "buy_price", 0) or 0)
            if price > 0:
                return price
        if before_position is not None:
            price = int(getattr(before_position, "buy_price", 0) or 0)
            if price > 0:
                return price
        return 0

    def _alert_daily_net_pnl(self) -> int:
        daily_pnl = getattr(self.strategy, "daily_pnl", None)
        if daily_pnl is None:
            return 0
        return int(getattr(daily_pnl, "realized_net_pnl", 0) or 0)

    def _alert_sell_net_pnl(self, result: OrderResult, before_position=None) -> int:
        if before_position is None:
            return 0
        qty = int(getattr(result, "quantity", 0) or 0)
        fill_price = self._alert_fill_price(result, before_position=before_position)
        buy_price = int(getattr(before_position, "buy_price", 0) or 0)
        if qty <= 0 or fill_price <= 0 or buy_price <= 0:
            return 0
        commission_rate = float(getattr(getattr(self.strategy, "config", None), "commission_rate", 0.0) or 0.0)
        tax_rate = float(getattr(getattr(self.strategy, "config", None), "tax_slippage_rate", 0.0) or 0.0)
        gross_buy = buy_price * qty
        gross_sell = fill_price * qty
        fees = int((gross_buy + gross_sell) * commission_rate)
        taxes = int(gross_sell * tax_rate)
        return int(gross_sell - gross_buy - fees - taxes)

    def _send_order_result_alert(self, result: OrderResult, before_position=None, after_position=None) -> None:
        if not getattr(self, "_alerts", None):
            return

        symbol = str(getattr(result, "symbol", "") or "").strip()
        if not symbol:
            return

        price = self._alert_fill_price(result, before_position=before_position, after_position=after_position)
        qty = int(getattr(result, "quantity", 0) or 0)
        side = getattr(result, "side", None)
        fill_mode = str(getattr(result, "fill_mode", "") or "")
        daily_net_pnl = self._alert_daily_net_pnl()

        if result.success:
            if (
                qty <= 0
                and side == OrderSide.BUY
                and after_position is not None
                and fill_mode in {"market_pending", "limit_then_market_pending"}
            ):
                qty = int(getattr(after_position, "quantity", 0) or 0)
                if price <= 0:
                    price = int(getattr(after_position, "buy_price", 0) or 0)
            if qty <= 0:
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
                trade_net_pnl = self._alert_sell_net_pnl(result, before_position=before_position)
                message = (
                    f"{symbol} {qty}주 @ {price:,}원\n"
                    f"거래 순손익: {trade_net_pnl:,}원\n"
                    f"당일 누적순손익: {daily_net_pnl:,}원"
                )
                title = "매도 체결"
                level = "info"
            else:
                return
            self._alerts.send(
                event_key=f"order_fill_{side.value}_{symbol}_{int(result.timestamp.timestamp()) if result.timestamp else 0}",
                title=title,
                message=message,
                level=level,
                cooldown_seconds=0,
            )
            return

        side_label = side.value if side is not None else "unknown"
        title = "주문 실패"
        message = f"{symbol} {side_label} {qty}주\n사유: {result.message or '-'}"
        if str(getattr(result, "error_category", "") or "") == "account_order_unavailable":
            self._alerts.send(
                event_key=f"order_fail_account_unavailable_{datetime.now().date().isoformat()}",
                title="계좌 주문 불가",
                message=(
                    "모의투자 계좌/API 키 조합이 주문 불가로 응답했습니다.\n"
                    "세션을 중단하고 추가 주문 스팸을 막습니다."
                ),
                level="error",
                cooldown_seconds=300,
            )
            return
        self._alerts.send(
            event_key=f"order_fail_{side_label}_{symbol}_{int(result.timestamp.timestamp()) if result.timestamp else 0}",
            title=title,
            message=message,
            level="warning",
            cooldown_seconds=0,
        )

    def _send_daily_halt_alert_if_needed(self) -> None:
        if not getattr(self, "_alerts", None):
            return
        if not hasattr(self.strategy, "should_continue") or self.strategy.should_continue():
            return
        day_key = datetime.now().date().isoformat()
        if self._last_halt_alert_day == day_key:
            return
        self._last_halt_alert_day = day_key
        daily_pnl = getattr(self.strategy, "daily_pnl", None)
        realized = int(getattr(daily_pnl, "realized_net_pnl", 0) or 0) if daily_pnl is not None else 0
        self._alerts.send(
            event_key=f"strategy_halted_{day_key}",
            title="당일 거래 중지",
            message=f"전략이 당일 거래를 중지했습니다.\n누적 순손익: {realized:,}원",
            level="warning",
            cooldown_seconds=0,
        )

    def _send_loop_error_alert(self, *, scope: str, message: str) -> None:
        if not getattr(self, "_alerts", None):
            return
        now = datetime.now()
        last_sent_at = getattr(self, "_last_loop_error_alert_at", None)
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
        if not getattr(self, "_alerts", None) or balance is None:
            return
        daily_pnl = getattr(self.strategy, "daily_pnl", None)
        trade_count = int(getattr(daily_pnl, "trade_count", 0) or 0) if daily_pnl is not None else 0
        win_count = int(getattr(daily_pnl, "win_count", 0) or 0) if daily_pnl is not None else 0
        loss_count = int(getattr(daily_pnl, "loss_count", 0) or 0) if daily_pnl is not None else 0
        positions = getattr(self.strategy, "positions", {}) or {}
        self._alerts.send(
            event_key=f"trading_session_summary_{datetime.now().date().isoformat()}_{self._last_session_end_reason}",
            title="트레이딩 세션 종료",
            message=(
                f"손익: {int(session_pnl):,}원\n"
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
        self._last_cached_quote_fallback_log_at = None
        self._last_empty_quote_batch_log_at = None
        self._last_tick_activity_log_at = None

        # 휴장일 체크
        today_key = datetime.now().strftime("%Y%m%d")
        if not self.market_data.is_market_open():
            self._session_started_at = None
            self._market_closed_session_date = today_key
            self._last_session_end_reason = "market_holiday"
            logger.info("오늘은 휴장일입니다. date=%s", today_key)
            self._interruptible_sleep(3600)  # 1시간 뒤 재확인
            return False

        # 전략이 스케줄러와 같은 client를 쓰도록 주입 (rate limit 공유)
        if hasattr(self.strategy, 'market_data'):
            self.strategy.market_data = self.market_data

        # 전략 초기화
        self._reset_strategy_simulated_clock()
        self.strategy.initialize()
        session_start_delay = self._session_start_balance_delay_seconds()
        if session_start_delay > 0:
            self._interruptible_sleep(session_start_delay)
        balance = self._fetch_balance_with_retry("세션 시작 잔고 조회")
        if balance:
            if hasattr(self.strategy, "sync_positions_from_account"):
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

        strategy_pnl_baseline = self._extract_strategy_realized_pnl()
        has_runtime_state_snapshot = False
        runtime_snapshot_fn = getattr(self.strategy, "has_runtime_state_snapshot", None)
        if callable(runtime_snapshot_fn):
            try:
                has_runtime_state_snapshot = bool(runtime_snapshot_fn())
            except Exception:
                has_runtime_state_snapshot = False
        force_runtime_pool_refresh = self._minutes_since_market_open(datetime.now()) < 30 and not has_runtime_state_snapshot
        self._refresh_runtime_pool_if_needed(datetime.now(), force=force_runtime_pool_refresh)
        watchlist = self.strategy.get_watchlist()
        logger.info("감시 종목: %d개", len(watchlist))
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
                if not self.strategy.should_continue():
                    halted_for_day = True
                    strategy_halt_observed = True
                    positions = getattr(self.strategy, "positions", {}) or {}
                    liquidate_fn = getattr(self.strategy, "_liquidate_all", None)
                    if positions and callable(liquidate_fn):
                        logger.warning(
                            "전략 중지 상태지만 보유 포지션 %d종목이 남아 즉시 청산을 시도합니다.",
                            len(positions),
                        )
                        halt_orders = liquidate_fn()
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
                        if pending_balance is not None and hasattr(self.strategy, "sync_positions_from_account"):
                            self.strategy.sync_positions_from_account(pending_balance.positions)
                        remaining_positions = getattr(self.strategy, "positions", {}) or {}
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

                # 동적 watchlist 갱신
                runtime_pool_refreshed = self._refresh_runtime_pool_if_needed(datetime.now())
                watchlist = self.strategy.get_watchlist()
                watchlist = self._effective_watchlist(watchlist, datetime.now())
                positions = getattr(self.strategy, "positions", {}) or {}
                if runtime_pool_refreshed and not positions:
                    self._log_tick_activity(datetime.now(), watchlist, [], [])
                    self._interruptible_sleep(1)
                    continue
                refresh_batch = self._build_quote_refresh_batch(watchlist, datetime.now())

                # 배치 시세 조회: 신규 조회가 막혀도 최근 캐시로 평가 루프를 유지한다.
                all_quotes = self._load_quotes_for_refresh_batch(refresh_batch, datetime.now())

                if all_quotes:
                    for q in all_quotes:
                        logger.debug(
                            "[%s] %s %s원 (%+.2f%%)",
                            q.symbol, q.name,
                            f"{q.current_price:,}", q.change_rate,
                        )

                    orders = self.strategy.on_batch_tick(all_quotes)
                    self._log_tick_activity(datetime.now(), watchlist, refresh_batch, all_quotes)
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
                    self._log_tick_activity(datetime.now(), watchlist, refresh_batch, all_quotes)

                self._interruptible_sleep(tick_interval)
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
                getattr(self, "_account_order_blocked_session_date", None)
                == datetime.now().strftime("%Y%m%d")
            )
            if strategy_resumed and not account_blocked_today:
                halted_for_day = False
                logger.warning("확정 체결손익 반영 후 당일 정지가 해제되어 장중 평가를 재개합니다.")

        # 종료 시 잔고 요약
        balance = self._fetch_balance_with_retry("세션 종료 잔고 조회", max_attempts=2, base_delay_seconds=1)
        if balance:
            session_pnl = self._resolve_session_profit_loss(
                balance,
                strategy_pnl_baseline=strategy_pnl_baseline,
            )
            logger.info(
                "최종 잔고 — 평가금액: %s원 | 손익: %s원",
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

        risk_events = payload.get("log_analysis", {}).get("risk_events", {})
        if bool(risk_events.get("daily_hard_stop_triggered")):
            return "daily_hard_stop"
        if bool(risk_events.get("daily_profit_target_triggered")):
            return "daily_profit_target"
        return "session_end"

    def _resolve_session_profit_loss(self, balance, strategy_pnl_baseline=None) -> int:
        """세션 종료 손익 값을 결정한다.

        우선순위:
        1) 계좌 API 당일 실현손익
        2) 전략 세션 증분 순손익(누적 - 세션 시작 시점)
        3) 잔고 API 평가손익 (fallback)
        """
        realized_pnl = self.account.get_realized_profit_loss()
        if realized_pnl is not None:
            return realized_pnl

        strategy_pnl = self._extract_strategy_realized_pnl()
        if strategy_pnl is not None:
            if strategy_pnl_baseline is not None:
                try:
                    baseline = int(strategy_pnl_baseline)
                    session_delta = int(strategy_pnl) - baseline
                    logger.warning(
                        "실현손익 API 조회 실패로 전략 세션 증분 순손익을 사용합니다: %s원 "
                        "(세션시작=%s원, 현재누적=%s원)",
                        f"{session_delta:,}",
                        f"{baseline:,}",
                        f"{int(strategy_pnl):,}",
                    )
                    return session_delta
                except (TypeError, ValueError):
                    pass
            logger.warning(
                "실현손익 API 조회 실패로 전략 누적 순손익을 사용합니다: %s원",
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
        daily_pnl = getattr(self.strategy, "daily_pnl", None)
        if daily_pnl is None:
            return None

        try:
            return int(getattr(daily_pnl, "realized_net_pnl"))
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
