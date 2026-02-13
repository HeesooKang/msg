import logging
import signal
import sys
import time

from src.account import AccountAPI
from src.api_client import KISClient
from src.auth import TokenManager
from src.config import Config
from src.executor import OrderExecutor, RiskManager
from src.logger_setup import setup_logger
from src.market_data import MarketDataAPI
from src.strategy import BaseStrategy
from src.trading import TradingAPI

logger = logging.getLogger("kis_trader.main")

# 전역 종료 플래그
_shutdown = False


def _signal_handler(signum, frame):
    global _shutdown
    _shutdown = True
    logger.info("종료 신호 수신 (signal=%d), 안전하게 종료합니다...", signum)


def create_components(config: Config):
    """핵심 컴포넌트들을 생성한다."""
    token_mgr = TokenManager(config)
    client = KISClient(config, token_mgr)
    market_data = MarketDataAPI(client)
    trading = TradingAPI(client)
    account = AccountAPI(client)
    return client, market_data, trading, account


def run(strategy: BaseStrategy, config: Config = None):
    """트레이딩 봇의 메인 루프를 실행한다."""
    global _shutdown
    _shutdown = False

    if config is None:
        config = Config.load()

    setup_logger(config.log_level)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    logger.info("=" * 50)
    logger.info("KIS 자동매매 봇 시작 [%s 모드]", config.trading_mode.upper())
    logger.info("=" * 50)

    client, market_data, trading, account = create_components(config)
    risk = RiskManager()
    executor = OrderExecutor(trading, risk)

    # 전략 초기화
    strategy.initialize()
    watchlist = strategy.get_watchlist()
    logger.info("감시 종목: %s", watchlist)

    # 초기 잔고 확인
    balance = account.get_balance()
    if balance:
        logger.info("예수금: %s원, 보유종목: %d개", f"{balance.total_deposit:,}", len(balance.positions))

    # 메인 루프
    while not _shutdown and strategy.should_continue():
        try:
            # 장 시간 체크
            if not market_data.is_market_open():
                logger.info("오늘은 휴장일입니다. 종료합니다.")
                break

            for symbol in watchlist:
                if _shutdown:
                    break

                quote = market_data.get_current_price(symbol)
                if quote is None:
                    continue

                logger.info(
                    "[%s] %s: %s원 (%+d, %+.2f%%)",
                    quote.symbol,
                    quote.name,
                    f"{quote.current_price:,}",
                    quote.change,
                    quote.change_rate,
                )

                # 전략에서 주문 생성
                orders = strategy.on_tick(quote)
                if orders:
                    results = executor.submit_orders(orders)
                    for result in results:
                        strategy.on_order_filled(result)

            # 다음 틱까지 대기
            time.sleep(5)

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error("예상치 못한 오류: %s", e, exc_info=True)
            time.sleep(10)

    logger.info("봇 종료 완료")


def run_scheduled(strategy: BaseStrategy, config: Config = None, tick_interval: int = 10):
    """스케줄러를 통해 장 시간에 맞춰 자동 실행한다."""
    from src.scheduler import TradingScheduler

    if config is None:
        config = Config.load()

    scheduler = TradingScheduler(strategy, config)

    signal.signal(signal.SIGINT, lambda s, f: scheduler.stop())
    signal.signal(signal.SIGTERM, lambda s, f: scheduler.stop())

    scheduler.run(tick_interval=tick_interval)


if __name__ == "__main__":
    # 기본 실행: 전략 없이 잔고/시세만 확인하는 테스트 모드
    config = Config.load()
    setup_logger(config.log_level)

    logger.info("=== 연결 테스트 모드 ===")
    client, market_data, trading, account = create_components(config)

    # 삼성전자 현재가 조회
    quote = market_data.get_current_price("005930")
    if quote:
        logger.info("삼성전자: %s원", f"{quote.current_price:,}")

    # 잔고 조회
    balance = account.get_balance()
    if balance:
        logger.info("예수금: %s원", f"{balance.total_deposit:,}")
        for pos in balance.positions:
            logger.info("  %s(%s): %d주, 수익률 %.2f%%", pos.name, pos.symbol, pos.quantity, pos.profit_rate)

    logger.info("=== 테스트 완료 ===")
