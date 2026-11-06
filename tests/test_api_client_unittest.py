import json
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from threading import Event
from unittest.mock import Mock, patch

import requests

from src.api_client import KISClient
from src.config import Config


def _config() -> Config:
    return Config(
        trading_mode="paper",
        is_paper=True,
        api_key="paper-key",
        api_secret="paper-secret",
        account_number="12345678",
        account_product_code="01",
        base_url="https://openapivts.koreainvestment.com:29443",
        ws_url="ws://ops.koreainvestment.com:31000",
        rate_limit_interval=0.0,
        request_connect_timeout=3.05,
        request_read_timeout=10.0,
        log_level="INFO",
    )


class DummyHTTPResponse:
    def __init__(self, payload, *, status_code=200, text=None):
        self.status_code = status_code
        self.headers = {}
        self._payload = payload
        self.text = text if text is not None else json.dumps(payload)

    def json(self):
        return self._payload


class FakeTokenManager:
    def __init__(self):
        self.invalidated = 0
        self._token = "expired-token"
        self.last_token_issue_at = 0.0

    def get_token(self):
        return self._token

    def invalidate_token(self, *, remove_cache=False):
        self.invalidated += 1
        self.remove_cache = remove_cache
        self._token = "fresh-token"


class FakeSession:
    def __init__(self):
        self.post_calls = []

    def post(self, url, headers, data, timeout):
        self.post_calls.append(
            {"url": url, "headers": headers, "data": data, "timeout": timeout, "at": time.time()}
        )
        if len(self.post_calls) == 1:
            return DummyHTTPResponse(
                {
                    "rt_cd": "1",
                    "msg_cd": "EGW00123",
                    "msg1": "기간이 만료된 token 입니다.",
                }
            )
        return DummyHTTPResponse({"rt_cd": "0", "output": {"ok": "1"}})


class TimeoutSession:
    def __init__(self):
        self.closed = False

    def get(self, url, headers, params, timeout):
        raise requests.exceptions.Timeout("simulated timeout")

    def close(self):
        self.closed = True


class StaticGetSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def get(self, url, headers, params, timeout):
        self.calls.append({"url": url, "headers": headers, "params": params, "timeout": timeout})
        return self.response

    def close(self):
        pass


class KISClientTokenRetryTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        clock = patch("src.api_client.time.time", side_effect=lambda: self.now)
        sleep = patch("src.api_client.time.sleep", side_effect=self._advance_time)
        clock.start()
        sleep.start()
        self.addCleanup(clock.stop)
        self.addCleanup(sleep.stop)

    def _advance_time(self, seconds):
        self.now += seconds

    def test_get_quote_time_is_final_transmission_after_auth_and_quota_waits(self):
        for retry in (False, True):
            with self.subTest(retry=retry):
                self.now = 100.0
                tokens = FakeTokenManager()
                client = KISClient(_config(), tokens)
                client._rate_limit_backoff_until = 110.0
                original_get_token = tokens.get_token

                def token():
                    self._advance_time(3.0)
                    return original_get_token()

                tokens.get_token = token
                sent = []

                def get(*args, **kwargs):
                    sent.append(self.now)
                    self._advance_time(2.0)
                    if retry and len(sent) == 1:
                        return DummyHTTPResponse({"rt_cd": "1", "msg_cd": "EGW00123"})
                    return DummyHTTPResponse({"rt_cd": "0", "output": []})

                client._session = Mock(get=Mock(side_effect=get))
                response = client.get(api_url="/quote", tr_id="FHKST11300006", params={})
                self.assertTrue(response.success)
                self.assertEqual(len(sent), 2 if retry else 1)
                self.assertGreaterEqual(sent[0], 110.0)
                self.assertEqual(response.request_started_at, datetime.fromtimestamp(sent[-1]))
                self.assertEqual((datetime.fromtimestamp(self.now) - response.request_started_at).total_seconds(), 2)

    def test_post_refreshes_token_once_when_api_reports_expired_token(self):
        token_manager = FakeTokenManager()
        client = KISClient(_config(), token_manager)
        fake_session = FakeSession()
        client._session = fake_session

        response = client.post(
            api_url="/uapi/test",
            tr_id="TTTC0012U",
            body={"CANO": "", "ACNT_PRDT_CD": ""},
        )

        self.assertTrue(response.success)
        self.assertEqual(token_manager.invalidated, 1)
        self.assertTrue(token_manager.remove_cache)
        self.assertEqual(len(fake_session.post_calls), 2)
        self.assertEqual(fake_session.post_calls[0]["headers"]["authorization"], "Bearer expired-token")
        self.assertEqual(fake_session.post_calls[1]["headers"]["authorization"], "Bearer fresh-token")
        self.assertAlmostEqual(
            fake_session.post_calls[1]["at"] - fake_session.post_calls[0]["at"], 1.1
        )

    def test_get_transport_timeout_does_not_set_global_rate_limit_cooldown(self):
        token_manager = FakeTokenManager()
        client = KISClient(_config(), token_manager)
        client._session = TimeoutSession()

        with self.assertRaises(requests.exceptions.Timeout):
            client.get(
                api_url="/uapi/domestic-stock/v1/quotations/intstock-multprice",
                tr_id="FHKST11300006",
                params={},
                log_timeout=False,
            )

        self.assertEqual(client.rate_limit_cooldown_remaining(), 0.0)

    def test_order_deadline_is_checked_after_rate_wait_and_token_retry(self):
        for expire_on_retry in (False, True):
            with self.subTest(expire_on_retry=expire_on_retry):
                self.now = 100.0
                client = KISClient(_config(), FakeTokenManager())
                session = FakeSession()
                client._session = session
                if not expire_on_retry:
                    client._last_call_time = self.now
                result = client.post("/uapi/test", "TTTC0012U", {}, submit_before=100.5)
                self.assertFalse(result.success)
                self.assertEqual(result.error_code, "LOCAL_ORDER_EXPIRED")
                self.assertEqual(len(session.post_calls), int(expire_on_retry))

    def test_order_accepted_before_deadline_is_not_discarded_by_slow_response(self):
        client = KISClient(_config(), FakeTokenManager())

        def accepted(*args, **kwargs):
            self.now += 10
            return DummyHTTPResponse({"rt_cd": "0", "output": {"ODNO": "confirmed"}})

        client._session = Mock(post=Mock(side_effect=accepted))
        check = Mock(return_value=None)
        result = client.post("/uapi/test", "TTTC0012U", {}, submit_before=105.0, pre_submit=check)
        self.assertTrue(result.success)
        self.assertEqual(result.output["ODNO"], "confirmed")
        client._session.post.assert_called_once()
        check.assert_called_once()

    def test_order_submission_check_runs_after_wait_and_again_after_token_retry(self):
        for reject_on_retry in (False, True):
            with self.subTest(reject_on_retry=reject_on_retry):
                self.now = 100.0
                client = KISClient(_config(), FakeTokenManager())
                session = FakeSession()
                client._session = session
                client._last_call_time = self.now
                checks = []

                def check():
                    checks.append(self.now)
                    return "돌파선 복귀" if not reject_on_retry or len(checks) == 2 else None

                result = client.post("/uapi/test", "TTTC0012U", {},
                                     submit_before=105.0, pre_submit=check)
                self.assertFalse(result.success)
                self.assertEqual(result.error_code, "LOCAL_ORDER_INVALIDATED")
                self.assertEqual(len(session.post_calls), int(reject_on_retry))
                self.assertGreater(checks[0], 100.0)
                if reject_on_retry:
                    self.assertGreater(checks[1], checks[0])

    def test_failed_submission_check_is_definitively_unsent_not_unknown(self):
        client = KISClient(_config(), FakeTokenManager())
        client._session = Mock()
        result = client.post("/uapi/test", "TTTC0012U", {},
                             pre_submit=Mock(side_effect=RuntimeError("quote unavailable")))
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "LOCAL_ORDER_INVALIDATED")
        client._session.post.assert_not_called()

    def test_get_timeout_is_request_scoped_and_preserved_on_token_retry(self):
        client = KISClient(_config(), FakeTokenManager())
        success = DummyHTTPResponse({"rt_cd": "0"})
        client._session = Mock(
            get=Mock(side_effect=[
                DummyHTTPResponse({"rt_cd": "1", "msg_cd": "EGW00123"}),
                success,
                success,
                success,
            ]),
            post=Mock(return_value=success),
        )
        balance_url = "/uapi/domestic-stock/v1/trading/inquire-balance"

        response = client.get(balance_url, "TEST", {}, timeout=(10.0, 60.0))
        client.get(balance_url, "TEST", {})
        client.get("/uapi/domestic-stock/v1/quotations/inquire-price", "TEST", {})
        client.post("/uapi/domestic-stock/v1/trading/order-cash", "TEST", {})

        self.assertTrue(response.success)
        self.assertEqual(
            [call.kwargs["timeout"] for call in client._session.get.call_args_list],
            [(10.0, 60.0), (10.0, 60.0), (3.05, 10.0), (3.05, 10.0)],
        )
        self.assertEqual(client._session.post.call_args.kwargs["timeout"], (3.05, 10.0))

    def test_get_ledger_rate_limit_sets_global_cooldown(self):
        token_manager = FakeTokenManager()
        client = KISClient(_config(), token_manager)
        payload_text = (
            '{"rt_cd":"1","msg1":"원장에서 허용 가능한 초당 거래건수를 초과하였습니다.",'
            '"msg_cd":"EGW00215"}'
        )
        client._session = StaticGetSession(
            DummyHTTPResponse({}, status_code=500, text=payload_text)
        )

        response = client.get(
            api_url="/uapi/domestic-stock/v1/trading/inquire-balance",
            tr_id="TTTC8434R",
            params={"CANO": "", "ACNT_PRDT_CD": ""},
        )

        self.assertFalse(response.success)
        self.assertGreaterEqual(client.rate_limit_cooldown_remaining(), 2.5)
        self.assertEqual(len(client._session.calls), 1)
        rate_limited_at = self.now
        client._session.response = DummyHTTPResponse({"rt_cd": "0"})

        client.get("/uapi/domestic-stock/v1/quotations/inquire-price", "TEST", {})

        self.assertAlmostEqual(self.now - rate_limited_at, 3.0)
        self.assertEqual(client.rate_limit_cooldown_remaining(), 0.0)

    def test_paper_requests_share_one_second_quota_even_with_legacy_or_zero_config(self):
        for configured_interval in (0.0, 0.5, 1.1, 2.0):
            with self.subTest(interval=configured_interval):
                config = _config()
                config.rate_limit_interval = configured_interval
                client = KISClient(config, FakeTokenManager())
                call_times = []

                def record_request(*args, **kwargs):
                    call_times.append(self.now)
                    return DummyHTTPResponse({"rt_cd": "0"})

                client._session = Mock(
                    get=Mock(side_effect=record_request), post=Mock(side_effect=record_request)
                )
                for endpoint in (
                    "quotations/inquire-price",
                    "ranking/fluctuation",
                    "trading/inquire-balance",
                ):
                    client.get(f"/uapi/domestic-stock/v1/{endpoint}", "TEST", {})
                for endpoint in ("order-cash", "order-rvsecncl"):
                    client.post(f"/uapi/domestic-stock/v1/trading/{endpoint}", "TEST", {})

                self.assertEqual(len(call_times), 5)
                for before, after in zip(call_times, call_times[1:]):
                    self.assertAlmostEqual(after - before, max(1.1, configured_interval))

                client._parse_response(DummyHTTPResponse({
                    "rt_cd": "1", "msg_cd": "EGW00201", "msg1": "Rate limit reached",
                }), "/quote")
                self.assertAlmostEqual(client.rate_limit_cooldown_remaining(), 1.25)

    def test_fresh_token_issuance_delays_first_request(self):
        token_manager = FakeTokenManager()
        token_manager.last_token_issue_at = self.now
        client = KISClient(_config(), token_manager)
        client._session = StaticGetSession(DummyHTTPResponse({"rt_cd": "0"}))

        client.get("/uapi/domestic-stock/v1/trading/inquire-balance", "TEST", {})

        self.assertAlmostEqual(self.now - token_manager.last_token_issue_at, 1.1)
        self.assertEqual(len(client._session.calls), 1)

    def test_real_mode_keeps_configured_interval(self):
        config = _config()
        config.is_paper = False
        config.trading_mode = "real"
        config.rate_limit_interval = 0.05
        client = KISClient(config, FakeTokenManager())
        started_at = self.now
        client._last_call_time = started_at

        client._rate_limit()

        self.assertAlmostEqual(self.now - started_at, 0.05)

    def test_waiting_request_rechecks_cooldown_from_an_inflight_response(self):
        client = KISClient(_config(), FakeTokenManager())
        client._last_call_time = self.now
        waiting, release = Event(), Event()
        requests_at = []

        def wait_once(seconds):
            if not waiting.is_set():
                waiting.set()
                if not release.wait(3):
                    raise AssertionError("test rate wait was not released")
            self._advance_time(seconds)

        def request():
            client._session = Mock(get=Mock(side_effect=lambda *a, **k: (
                requests_at.append(self.now) or DummyHTTPResponse({"rt_cd": "0"}))))
            return client.get("/quote", "TEST", {})

        with patch("src.api_client.time.sleep", side_effect=wait_once):
            with ThreadPoolExecutor(max_workers=2) as workers:
                pending = workers.submit(request)
                try:
                    self.assertTrue(waiting.wait(2))
                    cooldown = workers.submit(client._parse_response,
                        DummyHTTPResponse({"rt_cd": "1", "msg_cd": "EGW00215",
                                           "msg1": "원장에서 허용 가능한 초당 거래건수를 초과하였습니다."}),
                        "/balance")
                    # A sleeping caller must not prevent a completed response
                    # from publishing a longer cooldown to all queued callers.
                    cooldown.result(timeout=1)
                finally:
                    release.set()
                self.assertTrue(pending.result(timeout=2).success)
        self.assertEqual(requests_at, [103.0])

    def test_slow_quote_request_and_timeout_do_not_block_or_close_order_transport(self):
        entered, release = Event(), Event()
        requests_at = []

        def slow_get(*args, **kwargs):
            requests_at.append(self.now)
            entered.set()
            if not release.wait(3):
                raise AssertionError("test quote request was not released")
            raise requests.exceptions.Timeout("quote timeout")

        def order_post(*args, **kwargs):
            requests_at.append(self.now)
            self.assertFalse(release.is_set())
            return DummyHTTPResponse({"rt_cd": "0"})

        order_session = Mock(post=Mock(side_effect=order_post))
        quote_session = Mock(get=Mock(side_effect=slow_get))
        replacement = Mock()
        with patch("src.api_client.requests.Session", side_effect=[order_session, quote_session, replacement]):
            client = KISClient(_config(), FakeTokenManager())
            with ThreadPoolExecutor(max_workers=1) as worker:
                future = worker.submit(client.get, "/quote", "TEST", {}, log_timeout=False)
                try:
                    self.assertTrue(entered.wait(2))
                    response = client.post("/order", "TEST", {})
                    self.assertTrue(response.success)
                finally:
                    release.set()
                with self.assertRaises(requests.exceptions.Timeout):
                    future.result(timeout=2)
                self.assertIs(worker.submit(lambda: client._session).result(timeout=2), replacement)

        self.assertIs(client._session, order_session)
        self.assertAlmostEqual(requests_at[1] - requests_at[0], 1.1)
        order_session.close.assert_not_called()
        quote_session.close.assert_called_once()

    def test_delayed_old_token_failure_does_not_invalidate_replacement_token(self):
        tokens = FakeTokenManager()
        client = KISClient(_config(), tokens)
        old = client._prepare_request("TEST", "")
        fresh = client._prepare_request("TEST", "", rejected_token=old["authorization"])
        delayed_retry = client._prepare_request("TEST", "", rejected_token=old["authorization"])

        self.assertEqual(tokens.invalidated, 1)
        self.assertEqual(fresh["authorization"], "Bearer fresh-token")
        self.assertEqual(delayed_retry["authorization"], fresh["authorization"])


if __name__ == "__main__":
    unittest.main()
