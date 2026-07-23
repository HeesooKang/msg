import os
from dataclasses import dataclass
import logging

from dotenv import load_dotenv

logger = logging.getLogger("kis_trader.config")

@dataclass
class Config:
    # 트레이딩 모드
    trading_mode: str  # "paper" or "real"
    is_paper: bool

    # API 인증
    api_key: str
    api_secret: str

    # 계좌
    account_number: str  # 8자리
    account_product_code: str  # 2자리 (01:종합, 03:선물옵션 등)
    hts_id: str

    # 도메인
    base_url: str
    ws_url: str

    # 레이트 리미팅
    rate_limit_interval: float  # 초
    request_connect_timeout: float
    request_read_timeout: float

    # 로깅
    log_level: str
    off_hours_check_interval: int = 1800  # 초, 장외 체크 주기

    @classmethod
    def load(cls, env_path: str = None) -> "Config":
        """`.env` 파일에서 설정을 로드한다."""
        if env_path:
            load_dotenv(env_path)
        else:
            load_dotenv()

        trading_mode = os.getenv("TRADING_MODE", "paper").lower()
        is_paper = trading_mode != "real"

        if is_paper:
            api_key = os.getenv("PAPER_API_KEY", "")
            api_secret = os.getenv("PAPER_API_SECRET", "")
            account_number = os.getenv("PAPER_ACCOUNT_NUMBER", "")
            base_url = "https://openapivts.koreainvestment.com:29443"
            ws_url = "ws://ops.koreainvestment.com:31000"
            rate_limit_interval = 0.5
        else:
            api_key = os.getenv("REAL_API_KEY", "")
            api_secret = os.getenv("REAL_API_SECRET", "")
            account_number = os.getenv("REAL_ACCOUNT_NUMBER", "")
            base_url = "https://openapi.koreainvestment.com:9443"
            ws_url = "ws://ops.koreainvestment.com:21000"
            rate_limit_interval = 0.05

        account_number = account_number.strip()
        api_key = api_key.strip()
        api_secret = api_secret.strip()
        account_product_code = os.getenv("ACCOUNT_PRODUCT_CODE", "01").strip()
        hts_id = os.getenv("HTS_ID", "").strip()
        log_level = os.getenv("LOG_LEVEL", "INFO")
        request_connect_timeout = float(os.getenv("REQUEST_CONNECT_TIMEOUT_SECONDS", "3.05"))
        request_read_timeout = float(os.getenv("REQUEST_READ_TIMEOUT_SECONDS", "10"))
        off_hours_check_interval = int(os.getenv("OFF_HOURS_CHECK_INTERVAL_SECONDS", "1800"))

        config = cls(
            trading_mode=trading_mode,
            is_paper=is_paper,
            api_key=api_key,
            api_secret=api_secret,
            account_number=account_number,
            account_product_code=account_product_code,
            hts_id=hts_id,
            base_url=base_url,
            ws_url=ws_url,
            rate_limit_interval=rate_limit_interval,
            request_connect_timeout=max(0.5, request_connect_timeout),
            request_read_timeout=max(1.0, request_read_timeout),
            log_level=log_level,
            off_hours_check_interval=max(60, off_hours_check_interval),
        )
        config.validate()
        logger.info(
            "설정 로드: mode=%s, account=%s, product=%s, hts_id=%s",
            config.trading_mode.upper(),
            config.account_number[:-3] + "***" if len(config.account_number) >= 3 else "***",
            config.account_product_code,
            config.hts_id,
        )
        return config

    def validate(self):
        if not self.api_key:
            raise ValueError("API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
        if not self.api_secret:
            raise ValueError("API_SECRET이 설정되지 않았습니다. .env 파일을 확인하세요.")
        if not self.account_number:
            raise ValueError("ACCOUNT_NUMBER가 설정되지 않았습니다. .env 파일을 확인하세요.")
        if not self.account_number.isdigit() or len(self.account_number) != 8:
            raise ValueError("ACCOUNT_NUMBER는 8자리 숫자여야 합니다. .env 파일을 확인하세요.")
        if not self.account_product_code.isdigit() or len(self.account_product_code) != 2:
            raise ValueError("ACCOUNT_PRODUCT_CODE는 2자리 숫자여야 합니다. (예: 01)")
