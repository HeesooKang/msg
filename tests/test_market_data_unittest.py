import unittest
import time
from types import SimpleNamespace

import requests

from src.market_data import MarketDataAPI
from src.models import Quote, RankingItem


class _DummyClient:
    def __init__(self):
        self.calls = 0
        self.config = SimpleNamespace(is_paper=False)

    def rate_limit_cooldown_remaining(self):
        return 1.5

    def get(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(success=False, error_message='{"message":"EGW00201"}', output=[])


class _TimeoutClient(_DummyClient):
    def get(self, **kwargs):
        self.calls += 1
        raise requests.exceptions.Timeout("simulated timeout")


class MarketDataTests(unittest.TestCase):
    def test_rate_limit_detection_includes_ledger_quota_error(self):
        api = MarketDataAPI(_DummyClient())

        self.assertTrue(
            api._is_rate_limited_message(
                '{"rt_cd":"1","msg1":"원장에서 허용 가능한 초당 거래건수를 초과하였습니다.",'
                '"msg_cd":"EGW00215"}'
            )
        )

    def test_is_market_open_skips_holiday_api_in_paper_mode(self):
        client = _DummyClient()
        client.config = SimpleNamespace(is_paper=True)
        api = MarketDataAPI(client)

        is_open = api.is_market_open("20260410")

        self.assertTrue(is_open)
        self.assertEqual(client.calls, 0)

    def test_is_market_open_blocks_known_holiday_in_paper_mode(self):
        client = _DummyClient()
        client.config = SimpleNamespace(is_paper=True)
        api = MarketDataAPI(client)

        is_open = api.is_market_open("20260501")

        self.assertFalse(is_open)
        self.assertEqual(client.calls, 0)

    def test_is_market_open_blocks_known_holiday_before_api_call(self):
        client = _DummyClient()
        api = MarketDataAPI(client)

        is_open = api.is_market_open("20260501")

        self.assertFalse(is_open)
        self.assertEqual(client.calls, 0)

    def test_get_multi_price_uses_recent_cache_during_rate_limit_cooldown(self):
        client = _DummyClient()
        api = MarketDataAPI(client)
        cached_quotes = [
            Quote(
                symbol="005930",
                name="삼성전자",
                current_price=70_000,
                change=500,
                change_rate=0.72,
                open_price=69_500,
                high_price=70_100,
                low_price=69_400,
                volume=1_000_000,
                trade_amount=70_000_000_000,
            )
        ]
        api._multi_price_cache[("005930",)] = (__import__("time").time(), cached_quotes)

        quotes = api.get_multi_price(["005930"])

        self.assertEqual(client.calls, 0)
        self.assertEqual(len(quotes), 1)
        self.assertEqual(quotes[0].symbol, "005930")

    def test_get_multi_price_skips_immediate_retry_during_endpoint_backoff(self):
        client = _DummyClient()
        client.rate_limit_cooldown_remaining = lambda: 0.0
        api = MarketDataAPI(client)

        first = api.get_multi_price(["005930"])
        second = api.get_multi_price(["005930"])

        self.assertEqual(first, [])
        self.assertEqual(second, [])
        self.assertEqual(client.calls, 1)

    def test_get_multi_price_uses_symbol_cache_during_endpoint_backoff(self):
        client = _DummyClient()
        client.rate_limit_cooldown_remaining = lambda: 0.0
        api = MarketDataAPI(client)
        cached_quote = Quote(
            symbol="005930",
            name="삼성전자",
            current_price=70_000,
            change=500,
            change_rate=0.72,
            open_price=69_500,
            high_price=70_100,
            low_price=69_400,
            volume=1_000_000,
            trade_amount=70_000_000_000,
        )
        now_ts = __import__("time").time()
        api._multi_price_symbol_cache["005930"] = (now_ts, cached_quote)
        api._multi_price_backoff_until = now_ts + 10.0

        quotes = api.get_multi_price(["005930"])

        self.assertEqual(client.calls, 0)
        self.assertEqual(len(quotes), 1)
        self.assertEqual(quotes[0].symbol, "005930")

    def test_get_multi_price_sets_backoff_on_transport_timeout(self):
        client = _TimeoutClient()
        client.rate_limit_cooldown_remaining = lambda: 0.0
        api = MarketDataAPI(client)

        first = api.get_multi_price(["005930"])
        second = api.get_multi_price(["005930"])

        self.assertEqual(first, [])
        self.assertEqual(second, [])
        self.assertEqual(client.calls, 1)
        self.assertGreater(api.multi_price_backoff_remaining(), 0.0)
        self.assertLessEqual(api.multi_price_backoff_remaining(), 2.0)

    def test_get_multi_price_uses_symbol_cache_on_transport_timeout(self):
        client = _TimeoutClient()
        client.rate_limit_cooldown_remaining = lambda: 0.0
        api = MarketDataAPI(client)
        cached_quote = Quote(
            symbol="005930",
            name="삼성전자",
            current_price=70_000,
            change=500,
            change_rate=0.72,
            open_price=69_500,
            high_price=70_100,
            low_price=69_400,
            volume=1_000_000,
            trade_amount=70_000_000_000,
        )
        api._multi_price_symbol_cache["005930"] = (time.time(), cached_quote)

        quotes = api.get_multi_price(["005930"])

        self.assertEqual(client.calls, 1)
        self.assertEqual([quote.symbol for quote in quotes], ["005930"])
        self.assertGreater(api.multi_price_backoff_remaining(), 0.0)
        self.assertLessEqual(api.multi_price_backoff_remaining(), 2.0)

    def test_get_cached_quotes_returns_only_fresh_symbols_in_order(self):
        client = _DummyClient()
        api = MarketDataAPI(client)
        now_ts = __import__("time").time()
        fresh = Quote(
            symbol="005930",
            name="삼성전자",
            current_price=70_000,
            change=500,
            change_rate=0.72,
            open_price=69_500,
            high_price=70_100,
            low_price=69_400,
            volume=1_000_000,
            trade_amount=70_000_000_000,
        )
        stale = Quote(
            symbol="000660",
            name="SK하이닉스",
            current_price=120_000,
            change=3_600,
            change_rate=3.09,
            open_price=116_400,
            high_price=120_200,
            low_price=116_000,
            volume=300_000,
            trade_amount=36_000_000_000,
        )
        api._multi_price_symbol_cache["005930"] = (now_ts, fresh)
        api._multi_price_symbol_cache["000660"] = (now_ts - 100.0, stale)

        quotes = api.get_cached_quotes(["005930", "000660"], ttl_seconds=5.0)

        self.assertEqual([quote.symbol for quote in quotes], ["005930"])

    def test_get_fluctuation_ranking_uses_stale_cache_on_transport_timeout(self):
        client = _TimeoutClient()
        api = MarketDataAPI(client)
        api._fluctuation_cache[(3, 0.0, 30.0, 0, 0, 0)] = (
            time.time() - 300.0,
            [
                RankingItem(
                    symbol="005930",
                    name="삼성전자",
                    current_price=70_000,
                    change_rate=2.1,
                    volume=1_000_000,
                    rank=1,
                )
            ],
        )

        items = api.get_fluctuation_ranking(count=3)

        self.assertEqual(client.calls, 1)
        self.assertEqual([item.symbol for item in items], ["005930"])
        self.assertGreater(api.fluctuation_backoff_remaining(), 0.0)

    def test_get_fluctuation_ranking_skips_api_during_backoff_without_cache(self):
        client = _DummyClient()
        api = MarketDataAPI(client)
        api._fluctuation_backoff_until = time.time() + 30.0

        items = api.get_fluctuation_ranking(count=3)

        self.assertEqual(items, [])
        self.assertEqual(client.calls, 0)


if __name__ == "__main__":
    unittest.main()
