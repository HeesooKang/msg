import json
import unittest

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
        hts_id="",
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
        self.post_calls.append({"url": url, "headers": headers, "data": data, "timeout": timeout})
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
        self.assertGreater(client.transport_cooldown_remaining(), 0.0)
        self.assertLessEqual(client.transport_cooldown_remaining(), 2.0)
        self.assertEqual(client._transport_error_streak, 1)

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


if __name__ == "__main__":
    unittest.main()
