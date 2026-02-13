import json
import logging
import time
from typing import Any, Dict, Optional

import requests

from src.auth import TokenManager
from src.config import Config

logger = logging.getLogger("kis_trader.api")


class APIResponse:
    """KIS API 응답을 래핑한다."""

    def __init__(self, status_code: int, data: dict, headers: dict):
        self.status_code = status_code
        self.data = data
        self.headers = headers

    @property
    def success(self) -> bool:
        return self.data.get("rt_cd") == "0"

    @property
    def error_code(self) -> str:
        return self.data.get("msg_cd", "")

    @property
    def error_message(self) -> str:
        return self.data.get("msg1", "")

    @property
    def output(self) -> Any:
        return self.data.get("output")

    @property
    def output1(self) -> Any:
        return self.data.get("output1")

    @property
    def output2(self) -> Any:
        return self.data.get("output2")

    @property
    def has_next(self) -> bool:
        """연속 조회 데이터가 있는지 확인한다."""
        return self.headers.get("tr_cont", "") == "M"


class KISClient:
    """KIS Open API HTTP 클라이언트."""

    def __init__(self, config: Config, token_manager: TokenManager):
        self.config = config
        self.token_manager = token_manager
        self._session = requests.Session()
        self._last_call_time: float = 0

    def _rate_limit(self):
        """API 호출 간 최소 간격을 유지한다."""
        elapsed = time.time() - self._last_call_time
        if elapsed < self.config.rate_limit_interval:
            sleep_time = self.config.rate_limit_interval - elapsed
            logger.debug("레이트 리미팅: %.2fs 대기", sleep_time)
            time.sleep(sleep_time)
        self._last_call_time = time.time()

    def _build_headers(self, tr_id: str, tr_cont: str = "") -> dict:
        """API 호출에 필요한 공통 헤더를 구성한다."""
        # 모의투자: T/J/C로 시작하는 TR ID를 V로 변환
        if self.config.is_paper and tr_id[0] in ("T", "J", "C"):
            tr_id = "V" + tr_id[1:]

        token = self.token_manager.get_token()
        return {
            "Content-Type": "application/json",
            "Accept": "text/plain",
            "charset": "UTF-8",
            "authorization": f"Bearer {token}",
            "appkey": self.config.api_key,
            "appsecret": self.config.api_secret,
            "tr_id": tr_id,
            "tr_cont": tr_cont,
            "custtype": "P",
        }

    def _inject_account(self, params: dict) -> dict:
        """계좌번호와 상품코드를 params에 자동 주입한다."""
        if "CANO" in params:
            params["CANO"] = self.config.account_number
        if "ACNT_PRDT_CD" in params:
            params["ACNT_PRDT_CD"] = self.config.account_product_code
        return params

    def get(
        self,
        api_url: str,
        tr_id: str,
        params: Dict[str, str],
        tr_cont: str = "",
    ) -> APIResponse:
        """GET 요청을 보낸다."""
        self._rate_limit()
        url = f"{self.config.base_url}{api_url}"
        headers = self._build_headers(tr_id, tr_cont)
        params = self._inject_account(params)

        logger.debug("GET %s tr_id=%s", api_url, headers["tr_id"])
        res = self._session.get(url, headers=headers, params=params)
        return self._parse_response(res, api_url)

    def post(
        self,
        api_url: str,
        tr_id: str,
        body: Dict[str, str],
        tr_cont: str = "",
    ) -> APIResponse:
        """POST 요청을 보낸다."""
        self._rate_limit()
        url = f"{self.config.base_url}{api_url}"
        headers = self._build_headers(tr_id, tr_cont)
        body = self._inject_account(body)

        logger.debug("POST %s tr_id=%s", api_url, headers["tr_id"])
        res = self._session.post(url, headers=headers, data=json.dumps(body))
        return self._parse_response(res, api_url)

    def _parse_response(self, res: requests.Response, api_url: str) -> APIResponse:
        """응답을 파싱하고 에러를 로깅한다."""
        resp_headers = {k: v for k, v in res.headers.items() if k.islower()}

        if res.status_code != 200:
            logger.error("HTTP %d: %s (%s)", res.status_code, res.text, api_url)
            return APIResponse(res.status_code, {"rt_cd": "-1", "msg1": res.text}, resp_headers)

        data = res.json()
        api_resp = APIResponse(res.status_code, data, resp_headers)

        if not api_resp.success:
            logger.warning(
                "API 실패 [%s]: %s - %s",
                api_url,
                api_resp.error_code,
                api_resp.error_message,
            )

        return api_resp
