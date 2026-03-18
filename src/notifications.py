import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
import json
from typing import Dict

import requests

logger = logging.getLogger("kis_trader.notifications")

KAKAO_TOKEN_URL = "https://kauth.kakao.com/oauth/token"
KAKAO_MEMO_SEND_URL = "https://kapi.kakao.com/v2/api/talk/memo/default/send"


@dataclass
class AlertConfig:
    enabled: bool = False
    channel: str = "kakao"  # kakao|none
    kakao_rest_api_key: str = ""
    kakao_client_secret: str = ""
    kakao_refresh_token: str = ""
    kakao_message_web_url: str = ""
    kakao_message_mobile_web_url: str = ""
    kakao_message_button_title: str = "상세 보기"
    min_interval_seconds: int = 300

    @classmethod
    def from_env(cls) -> "AlertConfig":
        enabled = os.getenv("ALERTS_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")
        channel = os.getenv("ALERT_CHANNEL", "kakao").strip().lower() or "kakao"
        min_interval = int(os.getenv("ALERT_MIN_INTERVAL_SECONDS", "300"))
        return cls(
            enabled=enabled,
            channel=channel,
            kakao_rest_api_key=os.getenv("KAKAO_REST_API_KEY", "").strip(),
            kakao_client_secret=os.getenv("KAKAO_CLIENT_SECRET", "").strip(),
            kakao_refresh_token=os.getenv("KAKAO_REFRESH_TOKEN", "").strip(),
            kakao_message_web_url=os.getenv("KAKAO_MESSAGE_WEB_URL", "").strip(),
            kakao_message_mobile_web_url=os.getenv("KAKAO_MESSAGE_MOBILE_WEB_URL", "").strip(),
            kakao_message_button_title=os.getenv("KAKAO_MESSAGE_BUTTON_TITLE", "상세 보기").strip() or "상세 보기",
            min_interval_seconds=max(0, min_interval),
        )


@dataclass
class AlertManager:
    cfg: AlertConfig = field(default_factory=AlertConfig.from_env)
    _last_sent: Dict[str, float] = field(default_factory=dict)
    _kakao_access_token: str = ""
    _kakao_access_token_expires_at: float = 0.0

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

        if self.cfg.channel != "kakao":
            logger.warning("지원하지 않는 ALERT_CHANNEL: %s", self.cfg.channel)
            return False

        if not self.cfg.kakao_rest_api_key:
            logger.warning("KAKAO_REST_API_KEY 미설정: 알림 전송 생략")
            return False

        if not self.cfg.kakao_refresh_token:
            logger.warning("KAKAO_REFRESH_TOKEN 미설정: 알림 전송 생략")
            return False

        if not self.cfg.kakao_message_web_url:
            logger.warning("KAKAO_MESSAGE_WEB_URL 미설정: 알림 전송 생략")
            return False

        min_interval = max(self.cfg.min_interval_seconds, cooldown_seconds)
        now = time.time()
        last_ts = self._last_sent.get(event_key, 0.0)
        if min_interval > 0 and now - last_ts < min_interval:
            return False

        ok = self._send_kakao(level=level, title=title, message=message)
        if ok:
            self._last_sent[event_key] = now
        return ok

    def _get_kakao_access_token(self) -> str:
        now = time.time()
        if self._kakao_access_token and now < self._kakao_access_token_expires_at - 60:
            return self._kakao_access_token
        return self._refresh_kakao_access_token()

    def _refresh_kakao_access_token(self) -> str:
        payload = {
            "grant_type": "refresh_token",
            "client_id": self.cfg.kakao_rest_api_key,
            "refresh_token": self.cfg.kakao_refresh_token,
        }
        if self.cfg.kakao_client_secret:
            payload["client_secret"] = self.cfg.kakao_client_secret

        try:
            res = requests.post(KAKAO_TOKEN_URL, data=payload, timeout=5)
            if not 200 <= res.status_code < 300:
                logger.warning("카카오 토큰 갱신 실패 HTTP %d: %s", res.status_code, res.text)
                return ""

            body = res.json()
            access_token = str(body.get("access_token", "")).strip()
            if not access_token:
                logger.warning("카카오 토큰 갱신 실패: access_token 누락")
                return ""

            expires_in = int(body.get("expires_in", 0) or 0)
            self._kakao_access_token = access_token
            self._kakao_access_token_expires_at = time.time() + max(expires_in, 300)

            new_refresh_token = str(body.get("refresh_token", "")).strip()
            if new_refresh_token and new_refresh_token != self.cfg.kakao_refresh_token:
                self.cfg.kakao_refresh_token = new_refresh_token
                logger.warning(
                    "카카오 refresh_token이 재발급되었습니다. .env의 KAKAO_REFRESH_TOKEN 값을 갱신해 주십시오."
                )
            return access_token
        except Exception as e:
            logger.warning("카카오 토큰 갱신 예외: %s", e)
            return ""

    def _send_kakao(self, level: str, title: str, message: str) -> bool:
        access_token = self._get_kakao_access_token()
        if not access_token:
            return False

        prefix = {
            "info": "[INFO]",
            "warning": "[WARN]",
            "error": "[ERROR]",
        }.get(level, "[INFO]")
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text = f"{prefix} {title}\n{message}\n{ts}"
        web_url = self.cfg.kakao_message_web_url
        mobile_web_url = self.cfg.kakao_message_mobile_web_url or web_url
        template_object = {
            "object_type": "text",
            "text": text,
            "link": {
                "web_url": web_url,
                "mobile_web_url": mobile_web_url,
            },
        }
        if self.cfg.kakao_message_button_title:
            template_object["button_title"] = self.cfg.kakao_message_button_title

        try:
            res = requests.post(
                KAKAO_MEMO_SEND_URL,
                headers={"Authorization": f"Bearer {access_token}"},
                data={"template_object": json.dumps(template_object, ensure_ascii=False)},
                timeout=5,
            )
            if 200 <= res.status_code < 300:
                body = res.json()
                if int(body.get("result_code", 0)) != 0:
                    logger.warning("카카오 알림 실패 result_code=%s: %s", body.get("result_code"), body)
                    return False
                return True
            logger.warning("카카오 알림 실패 HTTP %d: %s", res.status_code, res.text)
            return False
        except Exception as e:
            logger.warning("카카오 알림 예외: %s", e)
            return False
