import json
import logging
import os
from datetime import datetime
from typing import Optional

import requests
import yaml

from src.config import Config

logger = logging.getLogger("kis_trader.auth")

TOKEN_DIR = os.path.join(os.path.expanduser("~"), "KIS", "config")


class TokenManager:
    """KIS API 접근 토큰을 발급/저장/갱신한다."""

    def __init__(self, config: Config):
        self.config = config
        self._token: str = ""
        self._token_expired: datetime = datetime.min
        os.makedirs(TOKEN_DIR, exist_ok=True)

    @property
    def _token_file(self) -> str:
        mode = "paper" if self.config.is_paper else "real"
        date_str = datetime.today().strftime("%Y%m%d")
        return os.path.join(TOKEN_DIR, f"KIS_{mode}_{date_str}")

    def get_token(self) -> str:
        """유효한 토큰을 반환한다. 만료됐으면 자동 갱신."""
        if self._token and self._is_valid():
            return self._token

        # 파일에서 로드 시도
        saved = self._load_token()
        if saved:
            self._token = saved
            return self._token

        # 새로 발급
        self._issue_token()
        return self._token

    def _is_valid(self) -> bool:
        return datetime.now() < self._token_expired

    def _load_token(self) -> Optional[str]:
        """파일에서 토큰을 읽어온다."""
        try:
            with open(self._token_file, encoding="UTF-8") as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
            if not data or "token" not in data:
                return None

            valid_date = data["valid-date"]
            if isinstance(valid_date, str):
                valid_date = datetime.strptime(valid_date, "%Y-%m-%d %H:%M:%S")

            if valid_date > datetime.now():
                self._token_expired = valid_date
                logger.info("저장된 토큰 로드 완료 (만료: %s)", valid_date)
                return data["token"]
            return None
        except (FileNotFoundError, TypeError, KeyError):
            return None

    def _save_token(self, token: str, expired: str):
        """토큰을 파일에 저장한다."""
        valid_date = datetime.strptime(expired, "%Y-%m-%d %H:%M:%S")
        self._token_expired = valid_date
        with open(self._token_file, "w", encoding="utf-8") as f:
            f.write(f"token: {token}\n")
            f.write(f"valid-date: {valid_date}\n")
        logger.info("토큰 저장 완료 (만료: %s)", valid_date)

    def _issue_token(self):
        """새 토큰을 발급받는다."""
        url = f"{self.config.base_url}/oauth2/tokenP"
        body = {
            "grant_type": "client_credentials",
            "appkey": self.config.api_key,
            "appsecret": self.config.api_secret,
        }
        headers = {"Content-Type": "application/json"}

        logger.info("토큰 발급 요청: %s", url)
        res = requests.post(url, data=json.dumps(body), headers=headers)

        if res.status_code != 200:
            raise RuntimeError(f"토큰 발급 실패 (HTTP {res.status_code}): {res.text}")

        data = res.json()
        self._token = data["access_token"]
        expired = data["access_token_token_expired"]
        self._save_token(self._token, expired)
        logger.info("토큰 발급 성공")
