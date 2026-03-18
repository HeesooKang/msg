import logging
import math
import time
import json
from datetime import datetime, time as clock_time, timedelta
from pathlib import Path

import requests

from src.account import AccountAPI
from src.api_client import KISClient
from src.auth import TokenManager
from src.config import Config
from src.executor import OrderExecutor, RiskManager
from src.logger_setup import setup_logger
from src.market_data import MarketDataAPI
from src.performance_reporting import update_performance_reports
from src.notifications import AlertManager
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
        self.config = config or Config.load()
        self.strategy = strategy
        self._shutdown = False

        # 컴포넌트 초기화
        self.token_mgr = TokenManager(self.config)
        self.client = KISClient(self.config, self.token_mgr)
        self.market_data = MarketDataAPI(self.client)
        self.trading = TradingAPI(self.client)
        self.account = AccountAPI(self.client)
        self.executor = OrderExecutor(self.trading, RiskManager())
        self._alerts = AlertManager()

        self._balance_retry_attempts = 3
        self._balance_retry_delay_seconds = 2
        self._last_session_end_reason = "session_end"

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
                    self._interruptible_sleep(min(30, self.config.off_hours_check_interval))

        except KeyboardInterrupt:
            logger.info("Ctrl+C — 스케줄러를 종료합니다.")
        finally:
            logger.info("스케줄러 종료")

    def _is_trading_time(self, now: datetime) -> bool:
        """현재가 거래 가능 시간인지 확인."""
        open_time = clock_time(hour=MARKET_OPEN[0], minute=MARKET_OPEN[1])
        close_time = clock_time(hour=MARKET_CLOSE[0], minute=MARKET_CLOSE[1])
        return now.weekday() < 5 and open_time <= now.time() < close_time

    def _is_preopen_time(self, now: datetime) -> bool:
        """장 시작 전 준비 시간인지 확인."""
        preopen_time = clock_time(hour=PRE_OPEN[0], minute=PRE_OPEN[1])
        open_time = clock_time(hour=MARKET_OPEN[0], minute=MARKET_OPEN[1])
        return now.weekday() < 5 and preopen_time <= now.time() < open_time

    def _seconds_until_preopen(self, now: datetime) -> int:
        """다음 준비 시각까지 남은 초."""
        today_preopen = now.replace(hour=PRE_OPEN[0], minute=PRE_OPEN[1], second=0, microsecond=0)
        today_open = now.replace(hour=MARKET_OPEN[0], minute=MARKET_OPEN[1], second=0, microsecond=0)

        if now.weekday() < 5:
            if now < today_preopen:
                return max(0, math.ceil((today_preopen - now).total_seconds()))
            if today_preopen <= now < today_open:
                # 프리오픈(08:50~09:00) 구간은 당일 09:00 장 시작까지 대기
                return max(0, math.ceil((today_open - now).total_seconds()))

        # 오늘 이미 지남 → 다음 평일 계산
        days_ahead = 1
        next_day = now + timedelta(days=days_ahead)
        while next_day.weekday() >= 5:  # 토(5), 일(6) 건너뛰기
            days_ahead += 1
            next_day = now + timedelta(days=days_ahead)

        next_preopen = next_day.replace(hour=PRE_OPEN[0], minute=PRE_OPEN[1], second=0, microsecond=0)
        return max(0, math.ceil((next_preopen - now).total_seconds()))

    def _next_preopen_after(self, now: datetime) -> datetime:
        """지정 시각 이후 가장 가까운 다음 준비 시각을 계산한다."""
        today_preopen = now.replace(hour=PRE_OPEN[0], minute=PRE_OPEN[1], second=0, microsecond=0)

        if now.weekday() < 5 and now < today_preopen:
            return today_preopen

        days_ahead = 1
        next_day = now + timedelta(days=days_ahead)
        while next_day.weekday() >= 5:  # 토(5), 일(6) 건너뛰기
            days_ahead += 1
            next_day = now + timedelta(days=days_ahead)

        return next_day.replace(hour=PRE_OPEN[0], minute=PRE_OPEN[1], second=0, microsecond=0)

    def _interruptible_sleep(self, seconds: int):
        """shutdown 체크하면서 대기."""
        end = time.time() + seconds
        while time.time() < end and not self._shutdown:
            time.sleep(1)

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

        # 휴장일 체크
        if not self.market_data.is_market_open():
            logger.info("오늘은 휴장일입니다.")
            self._interruptible_sleep(3600)  # 1시간 뒤 재확인
            return False

        # 전략이 스케줄러와 같은 client를 쓰도록 주입 (rate limit 공유)
        if hasattr(self.strategy, 'market_data'):
            self.strategy.market_data = self.market_data

        # 전략 초기화
        self.strategy.initialize()
        strategy_pnl_baseline = self._extract_strategy_realized_pnl()
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

        # 잔고 확인
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
            logger.error("잔고 조회 실패로 이번 세션의 주문 진행을 중단합니다.")
            self._interruptible_sleep(60)
            logger.info("--- 트레이딩 세션 종료 ---")
            return False

        halted_for_day = False
        self._last_session_end_reason = "session_end"

        # 틱 루프
        while not self._shutdown and self._is_trading_time(datetime.now()):
            try:
                if not self.strategy.should_continue():
                    logger.info("전략이 종료를 요청했습니다.")
                    halted_for_day = True
                    break

                # 동적 watchlist 갱신
                watchlist = self.strategy.get_watchlist()

                # 배치 시세 조회 (30종목씩)
                all_quotes = []
                for i in range(0, len(watchlist), 30):
                    chunk = watchlist[i:i+30]
                    chunk_quotes = self.market_data.get_multi_price(chunk)
                    all_quotes.extend(chunk_quotes)

                if all_quotes:
                    for q in all_quotes:
                        logger.debug(
                            "[%s] %s %s원 (%+.2f%%)",
                            q.symbol, q.name,
                            f"{q.current_price:,}", q.change_rate,
                        )

                    orders = self.strategy.on_batch_tick(all_quotes)
                    if orders:
                        logger.info("주문 %d건 제출", len(orders))
                        results = self.executor.submit_orders(orders)
                        for r in results:
                            self.strategy.on_order_filled(r)

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
                self._interruptible_sleep(max(10, tick_interval))

        logger.info("--- 트레이딩 세션 종료 ---")

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
        else:
            logger.warning("세션 종료 잔고 조회 실패: 오늘 요약 로그를 생략합니다.")
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

        for field in ("realized_net_pnl", "realized_pnl", "total_pnl"):
            if not hasattr(daily_pnl, field):
                continue
            try:
                return int(getattr(daily_pnl, field))
            except (TypeError, ValueError):
                continue

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
            balance = self.account.get_balance()
            if balance is not None:
                return balance

            if i < attempts and not self._shutdown:
                wait = delay * (2 ** (i - 1))
                logger.warning(
                    "%s 실패 (%d/%d): %d초 후 재시도",
                    context,
                    i,
                    attempts,
                    wait,
                )
                self._interruptible_sleep(wait)

        return None
