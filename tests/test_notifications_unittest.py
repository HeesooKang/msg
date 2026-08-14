import unittest
from unittest.mock import patch

from src.notifications import AlertConfig, AlertManager


class DummyResp:
    def __init__(self, status_code=200, text="ok", json_body=None):
        self.status_code = status_code
        self.text = text
        self._json_body = json_body or {}

    def json(self):
        return self._json_body


class NotificationTests(unittest.TestCase):
    def test_kakao_send_success(self):
        cfg = AlertConfig(
            enabled=True,
            channel="kakao",
            kakao_rest_api_key="rest-key",
            kakao_client_secret="client-secret",
            kakao_refresh_token="refresh-token",
            kakao_message_web_url="https://example.com",
            min_interval_seconds=0,
        )
        mgr = AlertManager(cfg=cfg)

        with patch(
            "src.notifications.requests.post",
            side_effect=[
                DummyResp(200, json_body={"access_token": "access-token", "expires_in": 3600}),
                DummyResp(200, json_body={"result_code": 0}),
            ],
        ) as mock_post:
            ok = mgr.send(
                event_key="k1",
                title="제목",
                message="내용",
                level="warning",
            )

        self.assertTrue(ok)
        self.assertEqual(mock_post.call_count, 2)
        token_kwargs = mock_post.call_args_list[0].kwargs
        send_kwargs = mock_post.call_args_list[1].kwargs
        self.assertEqual(token_kwargs["timeout"], 5)
        self.assertEqual(token_kwargs["data"]["grant_type"], "refresh_token")
        self.assertEqual(send_kwargs["timeout"], 5)
        self.assertEqual(send_kwargs["headers"]["Authorization"], "Bearer access-token")
        self.assertIn("제목", send_kwargs["data"]["template_object"])

    def test_cooldown_blocks_duplicate(self):
        cfg = AlertConfig(
            enabled=True,
            channel="kakao",
            kakao_rest_api_key="rest-key",
            kakao_client_secret="client-secret",
            kakao_refresh_token="refresh-token",
            kakao_message_web_url="https://example.com",
            min_interval_seconds=60,
        )
        mgr = AlertManager(cfg=cfg)

        with patch(
            "src.notifications.requests.post",
            side_effect=[
                DummyResp(200, json_body={"access_token": "access-token", "expires_in": 3600}),
                DummyResp(200, json_body={"result_code": 0}),
            ],
        ) as mock_post:
            first = mgr.send(event_key="same", title="a", message="b")
            second = mgr.send(event_key="same", title="a", message="b")

        self.assertTrue(first)
        self.assertFalse(second)
        self.assertEqual(mock_post.call_count, 2)

    def test_transient_kakao_send_failure_is_retried(self):
        cfg = AlertConfig(
            enabled=True,
            channel="kakao",
            kakao_rest_api_key="rest-key",
            kakao_refresh_token="refresh-token",
            kakao_message_web_url="https://example.com",
            min_interval_seconds=0,
        )
        mgr = AlertManager(cfg=cfg)

        with patch(
            "src.notifications.requests.post",
            side_effect=[
                DummyResp(200, json_body={"access_token": "access-token", "expires_in": 3600}),
                DummyResp(503, text="temporary"),
                DummyResp(200, json_body={"result_code": 0}),
            ],
        ) as mock_post, patch("src.notifications.time.sleep") as mock_sleep:
            ok = mgr.send(event_key="retry", title="t", message="m")

        self.assertTrue(ok)
        self.assertEqual(mock_post.call_count, 3)
        mock_sleep.assert_called_once_with(0.5)

    def test_disabled_alert_returns_false(self):
        cfg = AlertConfig(
            enabled=False,
            channel="kakao",
            kakao_rest_api_key="rest-key",
            kakao_refresh_token="refresh-token",
            kakao_message_web_url="https://example.com",
            min_interval_seconds=0,
        )
        mgr = AlertManager(cfg=cfg)

        with patch("src.notifications.requests.post") as mock_post:
            ok = mgr.send(event_key="k", title="t", message="m")

        self.assertFalse(ok)
        mock_post.assert_not_called()

    def test_missing_kakao_config_returns_false(self):
        cfg = AlertConfig(
            enabled=True,
            channel="kakao",
            kakao_rest_api_key="",
            kakao_refresh_token="",
            kakao_message_web_url="",
            min_interval_seconds=0,
        )
        mgr = AlertManager(cfg=cfg)

        with patch("src.notifications.requests.post") as mock_post:
            ok = mgr.send(event_key="k", title="t", message="m")

        self.assertFalse(ok)
        mock_post.assert_not_called()

    def test_invalid_kakao_refresh_token_disables_repeated_refresh_attempts(self):
        cfg = AlertConfig(
            enabled=True,
            channel="kakao",
            kakao_rest_api_key="rest-key",
            kakao_client_secret="client-secret",
            kakao_refresh_token="expired-refresh-token",
            kakao_message_web_url="https://example.com",
            min_interval_seconds=0,
        )
        mgr = AlertManager(cfg=cfg)

        with patch(
            "src.notifications.requests.post",
            return_value=DummyResp(
                400,
                text='{"error":"invalid_grant","error_description":"expired_or_invalid_refresh_token","error_code":"KOE322"}',
            ),
        ) as mock_post:
            first = mgr.send(event_key="k1", title="t1", message="m1")
            second = mgr.send(event_key="k2", title="t2", message="m2")

        self.assertFalse(first)
        self.assertFalse(second)
        self.assertEqual(mock_post.call_count, 1)
        self.assertEqual(mgr._kakao_auth_disabled_reason, "invalid_refresh_token")


if __name__ == "__main__":
    unittest.main()
