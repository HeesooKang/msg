import gzip
import json
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests

from src.api_client import KISClient
from src.analytics.quote_archive import QuoteArchive
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


class _SuccessClient(_DummyClient):
    def get(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(
            success=True,
            error_message="",
            output=[
                {
                    "inter_shrn_iscd": "005930",
                    "inter2_prpr": "80,000",
                    "inter2_askp": "80,100",
                    "inter2_bidp": "79,900",
                    "acml_vol": "1,234",
                    "seln_rsqn": "100",
                    "shnu_rsqn": "300",
                    "total_askp_rsqn": "1,000",
                    "total_bidp_rsqn": "3,000",
                    "acml_tr_pbmn": "98,765,432",
                    "inter2_oprc": "78,000",
                    "inter2_hgpr": "81,000",
                    "inter2_lwpr": "77,500",
                    "prdy_ctrt": "2.50",
                }
            ],
        )


class MarketDataTests(unittest.TestCase):
    def test_fluctuation_uses_close_rank_and_limits_after_code_validation(self):
        rows = [
            {"stck_shrn_iscd": "Q500001", "prdy_ctrt": "40.00"},
            {"stck_shrn_iscd": "18064K", "prdy_ctrt": "30.00", "data_rank": "2"},
            {"stck_shrn_iscd": "18064K", "prdy_ctrt": "30.00"},
            {"stck_shrn_iscd": "invalid"},
            {"stck_shrn_iscd": "020400", "prdy_ctrt": "29.85"},
            {"stck_shrn_iscd": "0000Z0", "prdy_ctrt": "29.50"},
            {"stck_shrn_iscd": "199430", "prdy_ctrt": "20.65"},
        ]
        for paper in (True, False):
            with self.subTest(paper=paper), tempfile.TemporaryDirectory() as root:
                client = _DummyClient()
                client.config.is_paper = paper
                client.get = Mock(return_value=SimpleNamespace(success=True, output=rows))
                archive = QuoteArchive(root)
                api = MarketDataAPI(client, discovery_archive=archive)
                expected = ["18064K", "020400", "0000Z0"]
                self.assertEqual(api.get_fluctuation_symbols(count=3), expected)
                self.assertEqual(api.get_fluctuation_symbols(count=3), expected)
                client.get.assert_called_once()
                request = client.get.call_args.kwargs
                self.assertEqual(request["api_url"], "/uapi/domestic-stock/v1/ranking/fluctuation")
                self.assertEqual(request["tr_id"], "FHPST01700000")
                self.assertEqual(request["params"]["fid_prc_cls_code"], "1")
                self.assertEqual(request["params"]["fid_input_cnt_1"], "0")
                self.assertEqual(request["params"]["fid_rank_sort_cls_code"], "0")
                self.assertEqual(request["params"]["fid_rsfl_rate2"], "")
                self.assertEqual(request["params"]["fid_trgt_exls_cls_code"], "0000000100")
                paths = list(Path(root).rglob("ranking-snapshots.*.jsonl.gz"))
                self.assertEqual(len(paths), 1)
                with gzip.open(paths[0], "rt") as handle:
                    records = [json.loads(line) for line in handle]
                self.assertEqual(len(records), 1)  # Cache hits are not new responses.
                record = records[0]
                self.assertEqual(record["rows"], rows)
                self.assertEqual(record["symbols"], expected)
                self.assertEqual(record["environment"], "paper" if paper else "real")
                self.assertEqual(record["params"], request["params"])
                archive.record([], now=datetime.now(), universe=expected)
                with gzip.open(next(Path(root).rglob("quote-batches.*.jsonl.gz")), "rt") as handle:
                    self.assertTrue(json.loads(handle.readline())["reset_history"])

    def test_discovery_archive_failure_does_not_drop_api_candidates(self):
        with tempfile.TemporaryDirectory() as root:
            invalid_root = Path(root) / "not-a-directory"
            invalid_root.touch()
            client = _DummyClient()
            client.get = Mock(return_value=SimpleNamespace(success=True,
                output=[{"stck_shrn_iscd": "18064K", "prdy_ctrt": "29.98"}]))
            api = MarketDataAPI(client, discovery_archive=QuoteArchive(invalid_root))
            with self.assertLogs("kis_trader.quote_archive", level="WARNING"):
                self.assertEqual(api.get_fluctuation_symbols(), ["18064K"])

    def test_multi_price_age_includes_response_delay_for_every_symbol(self):
        started = datetime(2026, 10, 2, 14, 42, 3)
        for wait, delay in ((0, 1), (0, 10), (8, 1), (8, 10)):
            with self.subTest(wait=wait, delay=delay), patch("src.market_data.datetime") as clock:
                clock.now.return_value = started
                client = _SuccessClient()
                response = client.get()
                response.output.append(dict(response.output[0], inter_shrn_iscd="105560"))
                transmitted = started + timedelta(seconds=wait)
                if wait:
                    response.request_started_at = transmitted

                def get(**kwargs):
                    clock.now.return_value = transmitted + timedelta(seconds=delay)
                    return response

                client.get = Mock(side_effect=get)
                quotes = MarketDataAPI(client).get_multi_price(["005930", "105560"])
                self.assertEqual([q.timestamp for q in quotes], [transmitted, transmitted])
                self.assertEqual([q.current_price for q in quotes], [80000, 80000])
                self.assertEqual([(clock.now() - q.timestamp).total_seconds() for q in quotes],
                                 [delay, delay])
                client.get.assert_called_once()

    def test_multi_price_carries_day_price_state_into_quote(self):
        api = MarketDataAPI(_SuccessClient())

        quote = api.get_multi_price(["005930"])[0]

        self.assertEqual(quote.current_price, 80_000)
        self.assertEqual(quote.cumulative_volume, 1_234)
        self.assertEqual(quote.cumulative_trade_amount, 98_765_432)
        self.assertEqual(quote.open_price, 78_000)
        self.assertEqual(quote.high_price, 81_000)
        self.assertEqual(quote.low_price, 77_500)
        self.assertEqual(quote.day_change_pct, 2.5)
        self.assertTrue(quote.day_summary_available)

    def test_multi_price_depth_does_not_fabricate_trade_flow(self):
        quote = MarketDataAPI(_SuccessClient()).get_multi_price(["005930"])[0]

        self.assertEqual((quote.ask_size, quote.bid_size), (100, 300))
        self.assertEqual((quote.total_ask_size, quote.total_bid_size), (1_000, 3_000))
        self.assertTrue(quote.book_depth_available)
        self.assertFalse(quote.flow_available)

    def test_multi_price_depth_requires_a_complete_nonnegative_pair(self):
        for invalid in (None, "", "invalid", "-1"):
            with self.subTest(invalid=invalid):
                client = _SuccessClient()
                response = client.get()
                response.output[0].update({
                    "seln_rsqn": invalid, "shnu_rsqn": "300",
                    "total_askp_rsqn": invalid, "total_bidp_rsqn": "3,000",
                })
                client.get = Mock(return_value=response)
                quote = MarketDataAPI(client).get_multi_price(["005930"])[0]
                self.assertFalse(quote.book_depth_available)

        client = _SuccessClient()
        response = client.get()
        response.output[0].update({
            "seln_rsqn": "0", "shnu_rsqn": "300",
            "total_askp_rsqn": "", "total_bidp_rsqn": "",
        })
        client.get = Mock(return_value=response)
        quote = MarketDataAPI(client).get_multi_price(["005930"])[0]
        self.assertTrue(quote.book_depth_available)
        self.assertEqual((quote.ask_size, quote.bid_size), (0, 300))

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
        client = _SuccessClient()
        client.cooldown = 1.5
        api = MarketDataAPI(client)

        self.assertEqual(api.get_multi_price(["005930"]), [])
        self.assertEqual(client.calls, 0)

        client.cooldown = 0.0
        api._multi_price_backoff_until = time.time() + 4
        self.assertEqual(api.get_multi_price(["005930"]), [])
        self.assertEqual(client.calls, 0)
        api._multi_price_backoff_until = 0
        self.assertEqual(len(api.get_multi_price(["005930"])), 1)
        self.assertEqual(client.calls, 1)

    def test_multi_price_uses_shared_quota_cooldown_without_another_endpoint_wait(self):
        for code, cooldown in (("EGW00201", 1.25), ("EGW00215", 3.0)):
            for status in (200, 500):
                with self.subTest(code=code, status=status):
                    now = [100.0]
                    sent = []
                    config = SimpleNamespace(is_paper=True, rate_limit_interval=0.0,
                        base_url="https://invalid.example", api_key="test", api_secret="test",
                        request_connect_timeout=3.05, request_read_timeout=10.0)
                    client = KISClient(config, SimpleNamespace(
                        get_token=lambda: "test", last_token_issue_at=0.0))
                    client._session.close()

                    def get(*args, **kwargs):
                        sent.append(now[0])
                        failed = len(sent) <= 2
                        data = ({"rt_cd": "1", "msg_cd": code, "msg1": code} if failed else
                                {"rt_cd": "0", "output": _SuccessClient().get().output})
                        return Mock(status_code=status if failed else 200, headers={},
                                    text=code, json=Mock(return_value=data))

                    client._session = Mock(get=Mock(side_effect=get))
                    api = MarketDataAPI(client)
                    with patch("time.time", side_effect=lambda: now[0]), patch(
                            "time.sleep", side_effect=lambda seconds: now.__setitem__(0, now[0] + seconds)):
                        for attempt in range(2):
                            self.assertEqual(api.get_multi_price(["005930"]), [])
                            self.assertAlmostEqual(client.rate_limit_cooldown_remaining(), cooldown)
                            self.assertEqual(api.get_multi_price(["005930"]), [])
                            self.assertEqual(len(sent), attempt + 1)
                            now[0] += cooldown
                        quotes = api.get_multi_price(["005930"])
                    self.assertEqual(len(quotes), 1)
                    self.assertEqual(sent, [100.0, 100.0 + cooldown, 100.0 + 2 * cooldown])
                    self.assertEqual(api.multi_price_backoff_remaining(), 0.0)

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
