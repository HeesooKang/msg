import os
from dataclasses import dataclass

from dotenv import load_dotenv


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

    # 로깅
    log_level: str

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

        account_product_code = os.getenv("ACCOUNT_PRODUCT_CODE", "01")
        hts_id = os.getenv("HTS_ID", "")
        log_level = os.getenv("LOG_LEVEL", "INFO")

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
            log_level=log_level,
        )
        config.validate()
        return config

    def validate(self):
        if not self.api_key:
            raise ValueError("API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
        if not self.api_secret:
            raise ValueError("API_SECRET이 설정되지 않았습니다. .env 파일을 확인하세요.")
        if not self.account_number:
            raise ValueError("ACCOUNT_NUMBER가 설정되지 않았습니다. .env 파일을 확인하세요.")
