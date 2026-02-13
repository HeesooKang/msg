"""KIS API 연결 테스트.

사용법:
    ./dev test tests/test_api_connection.py -v -s
    또는
    ./dev py tests/test_api_connection.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.auth import TokenManager
from src.api_client import KISClient
from src.market_data import MarketDataAPI
from src.trading import TradingAPI
from src.account import AccountAPI
from src.logger_setup import setup_logger


def test_connection():
    """기본 API 연결 및 토큰 발급 테스트."""
    config = Config.load()
    setup_logger(config.log_level)

    token_mgr = TokenManager(config)
    token = token_mgr.get_token()
    assert token, "토큰 발급 실패"
    print(f"[PASS] 토큰 발급 성공: {token[:20]}...")


def test_current_price():
    """삼성전자(005930) 현재가 조회 테스트."""
    config = Config.load()
    client = KISClient(config, TokenManager(config))
    market = MarketDataAPI(client)

    quote = market.get_current_price("005930")
    assert quote is not None, "현재가 조회 실패"
    assert quote.current_price > 0, "현재가가 0"
    print(f"[PASS] 삼성전자 현재가: {quote.current_price:,}원 ({quote.change_rate:+.2f}%)")


def test_balance():
    """잔고 조회 테스트."""
    config = Config.load()
    client = KISClient(config, TokenManager(config))
    account = AccountAPI(client)

    balance = account.get_balance()
    assert balance is not None, "잔고 조회 실패"
    print(f"[PASS] 예수금: {balance.total_deposit:,}원, 보유종목: {len(balance.positions)}개")


def test_buying_power():
    """매수가능금액 조회 테스트."""
    config = Config.load()
    client = KISClient(config, TokenManager(config))
    account = AccountAPI(client)

    power = account.get_buying_power()
    print(f"[PASS] 매수가능금액: {power:,}원")


if __name__ == "__main__":
    print("=" * 50)
    print("KIS API 연결 테스트")
    print("=" * 50)

    tests = [test_connection, test_current_price, test_balance, test_buying_power]
    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1

    print(f"\n결과: {passed} passed, {failed} failed / {len(tests)} total")
