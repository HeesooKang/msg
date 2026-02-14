import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict

import requests

logger = logging.getLogger("kis_trader.notifications")


@dataclass
class AlertConfig:
    enabled: bool = False
    channel: str = "slack"  # slack|none
    slack_webhook_url: str = ""
    min_interval_seconds: int = 300

    @classmethod
    def from_env(cls) -> "AlertConfig":
        enabled = os.getenv("ALERTS_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")
        channel = os.getenv("ALERT_CHANNEL", "slack").strip().lower() or "slack"
        webhook = os.getenv("SLACK_WEBHOOK_URL", "").strip()
        min_interval = int(os.getenv("ALERT_MIN_INTERVAL_SECONDS", "300"))
        return cls(
            enabled=enabled,
            channel=channel,
            slack_webhook_url=webhook,
            min_interval_seconds=max(0, min_interval),
        )


@dataclass
class AlertManager:
    cfg: AlertConfig = field(default_factory=AlertConfig.from_env)
    _last_sent: Dict[str, float] = field(default_factory=dict)

    def send(
        self,
        event_key: str,
        title: str,
        message: str,
        level: str = "info",
        cooldown_seconds: int = 0,
    ) -> bool:
        if not self.cfg.enabled:
            return False

        if self.cfg.channel in ("none", ""):
            return False

        if self.cfg.channel != "slack":
            logger.warning("지원하지 않는 ALERT_CHANNEL: %s", self.cfg.channel)
            return False

        if not self.cfg.slack_webhook_url:
            logger.warning("SLACK_WEBHOOK_URL 미설정: 알림 전송 생략")
            return False

        min_interval = max(self.cfg.min_interval_seconds, cooldown_seconds)
        now = time.time()
        last_ts = self._last_sent.get(event_key, 0.0)
        if min_interval > 0 and now - last_ts < min_interval:
            return False

        ok = self._send_slack(level=level, title=title, message=message)
        if ok:
            self._last_sent[event_key] = now
        return ok

    def _send_slack(self, level: str, title: str, message: str) -> bool:
        icon = {
            "info": ":large_blue_circle:",
            "warning": ":large_orange_circle:",
            "error": ":red_circle:",
        }.get(level, ":white_circle:")
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text = f"{icon} *{title}*\n{message}\n`{ts}`"
        payload = {"text": text}

        try:
            res = requests.post(self.cfg.slack_webhook_url, json=payload, timeout=5)
            if 200 <= res.status_code < 300:
                return True
            logger.warning("슬랙 알림 실패 HTTP %d: %s", res.status_code, res.text)
            return False
        except Exception as e:
            logger.warning("슬랙 알림 예외: %s", e)
            return False
