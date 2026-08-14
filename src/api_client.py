import json
import logging
import time
from typing import Any, Dict

import requests

from src.auth import TokenManager
from src.config import Config

logger = logging.getLogger("kis_trader.api")


def is_kis_rate_limited_message(message: str) -> bool:
    normalized = str(message or "")
    if "EGW00201" in normalized or "EGW00215" in normalized:
        return True
    compact = normalized.replace(" ", "")
    return "초당거래건수" in compact and "초과" in compact


def kis_rate_limit_cooldown_seconds(message: str, *, is_paper: bool) -> float:
    if not is_kis_rate_limited_message(message):
        return 0.0
    normalized = str(message or "")
    if "EGW00215" in normalized or "원장" in normalized:
        return 3.0 if is_paper else 0.8
    return 1.25 if is_paper else 0.35


class APIResponse:
    """KIS API 응답을 래핑한다."""

    def __init__(self, data: dict, headers: dict):
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
        return self.headers.get("tr_cont", "") in {"M", "F"}


class KISClient:
    """KIS Open API HTTP 클라이언트."""

    def __init__(self, config: Config, token_manager: TokenManager):
        self.config = config
        self.token_manager = token_manager
        self._session = requests.Session()
        self._last_call_time: float = 0
        self._rate_limit_backoff_until: float = 0.0

    def _request_timeout(self) -> tuple[float, float]:
        return (
            float(self.config.request_connect_timeout or 3.05),
            float(self.config.request_read_timeout or 10.0),
        )

    def _reset_session(self) -> None:
        try:
            self._session.close()
        except Exception:
            pass
        self._session = requests.Session()

    @staticmethod
    def _is_rate_limited_message(message: str) -> bool:
        return is_kis_rate_limited_message(message)

    @staticmethod
    def _is_token_invalid_response(response: APIResponse) -> bool:
        code = str(response.error_code or "")
        message = str(response.error_message or "")
        return (
            code == "EGW00123"
            or "기간이 만료된 token" in message
            or "만료된 token" in message
            or "invalid token" in message.lower()
        )

    def _invalidate_token_cache(self) -> None:
        self.token_manager.invalidate_token(remove_cache=True)

    def _note_rate_limit(self, message: str):
        cooldown_seconds = kis_rate_limit_cooldown_seconds(
            message,
            is_paper=bool(self.config.is_paper),
        )
        if cooldown_seconds <= 0:
            return
        self._rate_limit_backoff_until = max(
            self._rate_limit_backoff_until,
            time.time() + cooldown_seconds,
        )

    def rate_limit_cooldown_remaining(self) -> float:
        return max(0.0, self._rate_limit_backoff_until - time.time())

    def _rate_limit(self, api_url: str = ""):
        """API 호출 간 최소 간격을 유지한다."""
        cooldown_remaining = self.rate_limit_cooldown_remaining()
        if cooldown_remaining > 0:
            logger.debug("API 유량 제한 냉각 대기: %.2fs", cooldown_remaining)
            time.sleep(cooldown_remaining)
        configured_interval = max(0.0, float(self.config.rate_limit_interval or 0.0))
        request_interval = configured_interval
        if bool(self.config.is_paper) and configured_interval > 0.0:
            # The paper server enforces its two-per-second window at the gateway.
            # Calling exactly every 0.5s repeatedly lands on that boundary.
            request_interval = max(0.60, configured_interval)
            if "/trading/" in str(api_url or ""):
                # Account/order endpoints were still rejected 0.8s after the
                # previous request. Keep market-data polling responsive while
                # giving the lower paper-account quota a separate margin.
                request_interval = max(1.10, request_interval)
        elapsed = time.time() - self._last_call_time
        if elapsed < request_interval:
            sleep_time = request_interval - elapsed
            logger.debug("레이트 리미팅: %.2fs 대기", sleep_time)
            time.sleep(sleep_time)
        self._last_call_time = time.time()

    def _sync_token_issue_time(self) -> None:
        """Treat a fresh OAuth issuance as an API call before the first account request."""
        issued_at = float(self.token_manager.last_token_issue_at or 0.0)
        if issued_at > self._last_call_time:
            self._last_call_time = issued_at

    def _build_headers(self, tr_id: str, tr_cont: str = "") -> dict:
        """API 호출에 필요한 공통 헤더를 구성한다."""
        # 모의투자: T/J/C로 시작하는 TR ID를 V로 변환
        # 단, CTCA0903R(국내휴장일조회)는 변환 시 미지원 오류가 발생할 수 있어 예외 처리
        if self.config.is_paper and tr_id and tr_id[0] in ("T", "J", "C") and tr_id != "CTCA0903R":
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
        log_timeout: bool = True,
    ) -> APIResponse:
        """GET 요청을 보낸다."""
        url = f"{self.config.base_url}{api_url}"
        headers = self._build_headers(tr_id, tr_cont)
        self._sync_token_issue_time()
        self._rate_limit(api_url)
        params = self._inject_account(params)

        logger.debug("GET %s tr_id=%s", api_url, headers["tr_id"])
        try:
            res = self._session.get(
                url,
                headers=headers,
                params=params,
                timeout=self._request_timeout(),
            )
        except requests.exceptions.Timeout:
            self._reset_session()
            if log_timeout:
                logger.warning(
                    "GET timeout [%s]: connect=%.2fs read=%.2fs",
                    api_url,
                    self._request_timeout()[0],
                    self._request_timeout()[1],
                )
            raise
        except requests.exceptions.RequestException:
            self._reset_session()
            raise
        parsed = self._parse_response(res, api_url)
        if self._is_token_invalid_response(parsed):
            logger.warning("KIS 토큰이 API에서 만료/무효로 거부되어 캐시 삭제 후 1회 재발급합니다.")
            self._invalidate_token_cache()
            headers = self._build_headers(tr_id, tr_cont)
            self._sync_token_issue_time()
            self._rate_limit(api_url)
            try:
                res = self._session.get(
                    url,
                    headers=headers,
                    params=params,
                    timeout=self._request_timeout(),
                )
            except requests.exceptions.Timeout:
                self._reset_session()
                if log_timeout:
                    logger.warning(
                        "GET timeout [%s]: connect=%.2fs read=%.2fs",
                        api_url,
                        self._request_timeout()[0],
                        self._request_timeout()[1],
                    )
                raise
            except requests.exceptions.RequestException:
                self._reset_session()
                raise
            parsed = self._parse_response(res, api_url)
        return parsed

    def post(
        self,
        api_url: str,
        tr_id: str,
        body: Dict[str, str],
        tr_cont: str = "",
        log_timeout: bool = True,
    ) -> APIResponse:
        """POST 요청을 보낸다."""
        url = f"{self.config.base_url}{api_url}"
        headers = self._build_headers(tr_id, tr_cont)
        self._sync_token_issue_time()
        self._rate_limit(api_url)
        body = self._inject_account(body)

        logger.debug("POST %s tr_id=%s", api_url, headers["tr_id"])
        try:
            res = self._session.post(
                url,
                headers=headers,
                data=json.dumps(body),
                timeout=self._request_timeout(),
            )
        except requests.exceptions.Timeout:
            self._reset_session()
            if log_timeout:
                logger.warning(
                    "POST timeout [%s]: connect=%.2fs read=%.2fs",
                    api_url,
                    self._request_timeout()[0],
                    self._request_timeout()[1],
                )
            raise
        except requests.exceptions.RequestException:
            self._reset_session()
            raise
        parsed = self._parse_response(res, api_url)
        if self._is_token_invalid_response(parsed):
            logger.warning("KIS 토큰이 API에서 만료/무효로 거부되어 캐시 삭제 후 1회 재발급합니다.")
            self._invalidate_token_cache()
            headers = self._build_headers(tr_id, tr_cont)
            self._sync_token_issue_time()
            self._rate_limit(api_url)
            try:
                res = self._session.post(
                    url,
                    headers=headers,
                    data=json.dumps(body),
                    timeout=self._request_timeout(),
                )
            except requests.exceptions.Timeout:
                self._reset_session()
                if log_timeout:
                    logger.warning(
                        "POST timeout [%s]: connect=%.2fs read=%.2fs",
                        api_url,
                        self._request_timeout()[0],
                        self._request_timeout()[1],
                    )
                raise
            except requests.exceptions.RequestException:
                self._reset_session()
                raise
            parsed = self._parse_response(res, api_url)
        return parsed

    def _parse_response(self, res: requests.Response, api_url: str) -> APIResponse:
        """응답을 파싱하고 에러를 로깅한다."""
        resp_headers = {k: v for k, v in res.headers.items() if k.islower()}

        if res.status_code != 200:
            logger.error("HTTP %d: %s (%s)", res.status_code, res.text, api_url)
            self._note_rate_limit(res.text)
            return APIResponse({"rt_cd": "-1", "msg1": res.text}, resp_headers)

        data = res.json()
        api_resp = APIResponse(data, resp_headers)

        if not api_resp.success:
            self._note_rate_limit(api_resp.error_message)
            logger.warning(
                "API 실패 [%s]: %s - %s",
                api_url,
                api_resp.error_code,
                api_resp.error_message,
            )

        return api_resp
