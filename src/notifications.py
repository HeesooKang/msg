import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Dict

import requests

logger = logging.getLogger("kis_trader.notifications")

KAKAO_TOKEN_URL = "https://kauth.kakao.com/oauth/token"
KAKAO_MEMO_SEND_URL = "https://kapi.kakao.com/v2/api/talk/memo/default/send"
KAKAO_REFRESH_TOKEN_STATE_PATH = (
    Path(__file__).resolve().parents[1] / "state" / "kakao_refresh_token.json"
)


def _load_persisted_kakao_refresh_token() -> str:
    path = KAKAO_REFRESH_TOKEN_STATE_PATH
    if not path.is_file():
        return ""
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
        return str(body.get("refresh_token", "") or "").strip()
    except Exception as exc:
        logger.warning("카카오 refresh_token 상태 파일을 읽지 못했습니다: %s", exc)
        return ""


def _persist_kakao_refresh_token(refresh_token: str) -> bool:
    token = str(refresh_token or "").strip()
    if not token:
        return False

    path = KAKAO_REFRESH_TOKEN_STATE_PATH
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"refresh_token": token}, handle, ensure_ascii=True)
            handle.write("\n")
        os.replace(temp_path, path)
        os.chmod(path, 0o600)
        return True
    except Exception as exc:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass
        logger.warning("카카오 refresh_token 상태 파일 저장 실패: %s", exc)
        return False


@dataclass
class AlertConfig:
    enabled: bool = False
    channel: str = "kakao"  # kakao|none
    kakao_rest_api_key: str = ""
    kakao_client_secret: str = ""
    kakao_refresh_token: str = field(default="", repr=False)
    kakao_refresh_token_fallback: str = field(default="", repr=False)
    kakao_message_web_url: str = ""
    kakao_message_mobile_web_url: str = ""
    kakao_message_button_title: str = "상세 보기"
    min_interval_seconds: int = 300

    @classmethod
    def from_env(cls) -> "AlertConfig":
        enabled = os.getenv("ALERTS_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")
        channel = os.getenv("ALERT_CHANNEL", "kakao").strip().lower() or "kakao"
        min_interval = int(os.getenv("ALERT_MIN_INTERVAL_SECONDS", "300"))
        env_refresh_token = os.getenv("KAKAO_REFRESH_TOKEN", "").strip()
        persisted_refresh_token = _load_persisted_kakao_refresh_token()
        return cls(
            enabled=enabled,
            channel=channel,
            kakao_rest_api_key=os.getenv("KAKAO_REST_API_KEY", "").strip(),
            kakao_client_secret=os.getenv("KAKAO_CLIENT_SECRET", "").strip(),
            kakao_refresh_token=persisted_refresh_token or env_refresh_token,
            kakao_refresh_token_fallback=(
                env_refresh_token
                if persisted_refresh_token and env_refresh_token != persisted_refresh_token
                else ""
            ),
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
    _kakao_auth_disabled_reason: str = ""
    _using_kakao_refresh_token_fallback: bool = False

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

        if self._kakao_auth_disabled_reason:
            logger.debug(
                "카카오 인증이 비활성화되어 알림 전송을 생략합니다: %s",
                self._kakao_auth_disabled_reason,
            )
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

        ok = False
        for attempt in range(3):
            ok = self._send_kakao(level=level, title=title, message=message)
            if ok or self._kakao_auth_disabled_reason:
                break
            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))
        if ok:
            self._last_sent[event_key] = now
            logger.info("알림 전송 성공: event=%s title=%s", event_key, title)
        else:
            logger.warning("알림 전송 실패: event=%s title=%s", event_key, title)
        return ok

    def _get_kakao_access_token(self) -> str:
        now = time.time()
        if self._kakao_access_token and now < self._kakao_access_token_expires_at - 60:
            return self._kakao_access_token
        return self._refresh_kakao_access_token()

    def _refresh_kakao_access_token(self) -> str:
        refresh_token = self.cfg.kakao_refresh_token
        payload = {
            "grant_type": "refresh_token",
            "client_id": self.cfg.kakao_rest_api_key,
            "refresh_token": refresh_token,
        }
        if self.cfg.kakao_client_secret:
            payload["client_secret"] = self.cfg.kakao_client_secret

        try:
            res = requests.post(KAKAO_TOKEN_URL, data=payload, timeout=5)
            if not 200 <= res.status_code < 300:
                body_text = str(res.text or "")
                if (
                    res.status_code == 400
                    and (
                        "invalid_grant" in body_text
                        or "expired_or_invalid_refresh_token" in body_text
                        or "KOE322" in body_text
                    )
                ):
                    fallback = self.cfg.kakao_refresh_token_fallback
                    if fallback and fallback != refresh_token:
                        self.cfg.kakao_refresh_token = fallback
                        self.cfg.kakao_refresh_token_fallback = ""
                        self._using_kakao_refresh_token_fallback = True
                        logger.warning(
                            "저장된 카카오 refresh_token이 무효하여 환경 설정 토큰으로 한 번 재시도합니다."
                        )
                        return self._refresh_kakao_access_token()
                    self._kakao_auth_disabled_reason = "invalid_refresh_token"
                    logger.error(
                        "카카오 refresh_token이 만료/무효입니다. 새 KAKAO_REFRESH_TOKEN으로 갱신 후 재시작해야 알림이 재개됩니다: %s",
                        body_text,
                    )
                    return ""
                logger.warning("카카오 토큰 갱신 실패 HTTP %d: %s", res.status_code, body_text)
                return ""

            body = res.json()
            access_token = str(body.get("access_token", "")).strip()
            if not access_token:
                logger.warning("카카오 토큰 갱신 실패: access_token 누락")
                return ""

            expires_in = int(body.get("expires_in", 0) or 0)
            self._kakao_access_token = access_token
            self._kakao_access_token_expires_at = time.time() + max(expires_in, 300)
            self._kakao_auth_disabled_reason = ""

            new_refresh_token = str(body.get("refresh_token", "")).strip()
            token_to_persist = ""
            if new_refresh_token and new_refresh_token != self.cfg.kakao_refresh_token:
                self.cfg.kakao_refresh_token = new_refresh_token
                token_to_persist = new_refresh_token
            elif self._using_kakao_refresh_token_fallback:
                token_to_persist = self.cfg.kakao_refresh_token

            self._using_kakao_refresh_token_fallback = False
            if token_to_persist:
                if _persist_kakao_refresh_token(token_to_persist):
                    logger.info("카카오 refresh_token 갱신 상태를 안전하게 저장했습니다.")
                else:
                    logger.warning(
                        "카카오 refresh_token은 갱신됐지만 재시작용 상태 저장에 실패했습니다."
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
            if res.status_code == 401:
                self._kakao_access_token = ""
                self._kakao_access_token_expires_at = 0.0
            logger.warning("카카오 알림 실패 HTTP %d: %s", res.status_code, res.text)
            return False
        except Exception as e:
            logger.warning("카카오 알림 예외: %s", e)
            return False
