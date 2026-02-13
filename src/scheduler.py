import logging
import time
from datetime import datetime, timedelta

from src.account import AccountAPI
from src.api_client import KISClient
from src.auth import TokenManager
from src.config import Config
from src.executor import OrderExecutor, RiskManager
from src.logger_setup import setup_logger
from src.market_data import MarketDataAPI
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
        logger.info("=" * 50)

        try:
            while not self._shutdown:
                now = datetime.now()

                if self._is_trading_time(now):
                    self._run_trading_session(tick_interval)
                else:
                    wait = self._seconds_until_preopen(now)
                    if wait > 0:
                        next_open = now + timedelta(seconds=wait)
                        logger.info(
                            "장 시간이 아닙니다. 다음 준비 시각: %s (%d분 후)",
                            next_open.strftime("%H:%M"),
                            wait // 60,
                        )
                        # 긴 대기 시간은 60초 단위로 쪼개서 shutdown 체크
                        self._interruptible_sleep(min(wait, 300))
                    else:
                        # 오늘 장 끝남, 내일까지 대기
                        logger.info("오늘 장이 종료되었습니다. 내일까지 대기합니다.")
                        self._interruptible_sleep(60)

        except KeyboardInterrupt:
            logger.info("Ctrl+C — 스케줄러를 종료합니다.")
        finally:
            logger.info("스케줄러 종료")

    def _is_trading_time(self, now: datetime) -> bool:
        """현재가 거래 가능 시간인지 확인."""
        current = (now.hour, now.minute)
        return MARKET_OPEN <= current < MARKET_CLOSE and now.weekday() < 5

    def _is_preopen_time(self, now: datetime) -> bool:
        """장 시작 전 준비 시간인지 확인."""
        current = (now.hour, now.minute)
        return PRE_OPEN <= current < MARKET_OPEN and now.weekday() < 5

    def _seconds_until_preopen(self, now: datetime) -> int:
        """다음 준비 시각까지 남은 초."""
        today_preopen = now.replace(hour=PRE_OPEN[0], minute=PRE_OPEN[1], second=0)

        if now < today_preopen and now.weekday() < 5:
            return int((today_preopen - now).total_seconds())

        # 오늘 이미 지남 → 다음 평일 계산
        days_ahead = 1
        next_day = now + timedelta(days=days_ahead)
        while next_day.weekday() >= 5:  # 토(5), 일(6) 건너뛰기
            days_ahead += 1
            next_day = now + timedelta(days=days_ahead)

        next_preopen = next_day.replace(hour=PRE_OPEN[0], minute=PRE_OPEN[1], second=0, microsecond=0)
        return int((next_preopen - now).total_seconds())

    def _interruptible_sleep(self, seconds: int):
        """shutdown 체크하면서 대기."""
        end = time.time() + seconds
        while time.time() < end and not self._shutdown:
            time.sleep(1)

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
            return

        # 전략이 스케줄러와 같은 client를 쓰도록 주입 (rate limit 공유)
        if hasattr(self.strategy, 'market_data'):
            self.strategy.market_data = self.market_data

        # 전략 초기화
        self.strategy.initialize()
        watchlist = self.strategy.get_watchlist()
        logger.info("감시 종목: %d개", len(watchlist))

        # 잔고 확인
        balance = self.account.get_balance()
        if balance:
            logger.info(
                "예수금: %s원 | 보유: %d종목",
                f"{balance.total_deposit:,}",
                len(balance.positions),
            )

        # 틱 루프
        while not self._shutdown and self._is_trading_time(datetime.now()):
            if not self.strategy.should_continue():
                logger.info("전략이 종료를 요청했습니다.")
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

        logger.info("--- 트레이딩 세션 종료 ---")

        # 종료 시 잔고 요약
        balance = self.account.get_balance()
        if balance:
            logger.info(
                "최종 잔고 — 평가금액: %s원 | 손익: %s원",
                f"{balance.total_eval_amount:,}",
                f"{balance.total_profit_loss:,}",
            )
