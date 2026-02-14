import unittest
from unittest.mock import patch

from src.notifications import AlertConfig, AlertManager


class DummyResp:
    def __init__(self, status_code=200, text="ok"):
        self.status_code = status_code
        self.text = text


class NotificationTests(unittest.TestCase):
    def test_slack_send_success(self):
        cfg = AlertConfig(
            enabled=True,
            channel="slack",
            slack_webhook_url="https://hooks.slack.com/services/test",
            min_interval_seconds=0,
        )
        mgr = AlertManager(cfg=cfg)

        with patch("src.notifications.requests.post", return_value=DummyResp(200)) as mock_post:
            ok = mgr.send(
                event_key="k1",
                title="제목",
                message="내용",
                level="warning",
            )

        self.assertTrue(ok)
        mock_post.assert_called_once()
        kwargs = mock_post.call_args.kwargs
        self.assertEqual(kwargs["timeout"], 5)
        self.assertIn("제목", kwargs["json"]["text"])

    def test_cooldown_blocks_duplicate(self):
        cfg = AlertConfig(
            enabled=True,
            channel="slack",
            slack_webhook_url="https://hooks.slack.com/services/test",
            min_interval_seconds=60,
        )
        mgr = AlertManager(cfg=cfg)

        with patch("src.notifications.requests.post", return_value=DummyResp(200)) as mock_post:
            first = mgr.send(event_key="same", title="a", message="b")
            second = mgr.send(event_key="same", title="a", message="b")

        self.assertTrue(first)
        self.assertFalse(second)
        self.assertEqual(mock_post.call_count, 1)

    def test_disabled_alert_returns_false(self):
        cfg = AlertConfig(
            enabled=False,
            channel="slack",
            slack_webhook_url="https://hooks.slack.com/services/test",
            min_interval_seconds=0,
        )
        mgr = AlertManager(cfg=cfg)

        with patch("src.notifications.requests.post") as mock_post:
            ok = mgr.send(event_key="k", title="t", message="m")

        self.assertFalse(ok)
        mock_post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
