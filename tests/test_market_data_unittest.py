import time
import unittest
from types import SimpleNamespace

import requests

from src.market_data import MarketDataAPI


class _DummyClient:
    def __init__(self):
        self.calls = 0
        self.config = SimpleNamespace(is_paper=False)
        self.cooldown = 0.0

    def rate_limit_cooldown_remaining(self):
        return self.cooldown

    def get(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(
            success=False,
            error_message='{"message":"EGW00201"}',
            output=[],
        )


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

        self.assertTrue(api.is_market_open("20260410"))
        self.assertEqual(client.calls, 0)

    def test_is_market_open_blocks_known_holiday_before_api_call(self):
        client = _DummyClient()
        api = MarketDataAPI(client)

        self.assertFalse(api.is_market_open("20260501"))
        self.assertEqual(client.calls, 0)

    def test_get_multi_price_skips_client_and_endpoint_cooldowns(self):
        client = _DummyClient()
        client.cooldown = 1.5
        api = MarketDataAPI(client)

        self.assertEqual(api.get_multi_price(["005930"]), [])
        self.assertEqual(client.calls, 0)

        client.cooldown = 0.0
        self.assertEqual(api.get_multi_price(["005930"]), [])
        self.assertEqual(api.get_multi_price(["005930"]), [])
        self.assertEqual(client.calls, 1)

    def test_get_multi_price_sets_backoff_on_transport_timeout(self):
        client = _TimeoutClient()
        api = MarketDataAPI(client)

        self.assertEqual(api.get_multi_price(["005930"]), [])
        self.assertEqual(api.get_multi_price(["005930"]), [])
        self.assertEqual(client.calls, 1)
        self.assertGreater(api.multi_price_backoff_remaining(), 0.0)

    def test_get_fluctuation_symbols_uses_stale_cache_on_transport_timeout(self):
        client = _TimeoutClient()
        api = MarketDataAPI(client)
        api._fluctuation_cache[(3, 0.0)] = (
            time.time() - 300.0,
            ["005930"],
        )

        symbols = api.get_fluctuation_symbols(count=3)

        self.assertEqual(client.calls, 1)
        self.assertEqual(symbols, ["005930"])
        self.assertGreater(api.fluctuation_backoff_remaining(), 0.0)

    def test_get_fluctuation_symbols_skips_api_during_backoff_without_cache(self):
        client = _DummyClient()
        api = MarketDataAPI(client)
        api._fluctuation_backoff_until = time.time() + 30.0

        self.assertEqual(api.get_fluctuation_symbols(count=3), [])
        self.assertEqual(client.calls, 0)


if __name__ == "__main__":
    unittest.main()
