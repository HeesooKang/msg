import logging
import time
from datetime import datetime
from typing import List

import pandas as pd
import requests

from src.api_client import KISClient, is_kis_rate_limited_message
from src.market_calendar import is_known_krx_closed_date, normalize_trading_date
from src.models import Quote

logger = logging.getLogger("kis_trader.market")


class MarketDataAPI:
    """국내주식 시세 조회 API."""

    def __init__(self, client: KISClient):
        self.client = client
        self._market_open_cache: dict[str, bool] = {}
        self._holiday_warned_dates: set[str] = set()
        self._multi_price_backoff_until: float = 0.0
        self._multi_price_rate_limit_streak: int = 0
        self._multi_price_transport_error_streak: int = 0
        self._fluctuation_cache: dict[tuple[int, float], tuple[float, list[str]]] = {}
        self._fluctuation_backoff_until: float = 0.0
        self._last_multi_price_backoff_skip_log_at: float = 0.0
        self._last_multi_price_rate_limit_log_at: float = 0.0

    @staticmethod
    def _is_rate_limited_message(message: str) -> bool:
        return is_kis_rate_limited_message(message)

    @staticmethod
    def _cache_valid(cached_at: float, ttl_seconds: float) -> bool:
        return (time.time() - cached_at) <= ttl_seconds

    def _multi_price_backoff_remaining(self) -> float:
        return max(0.0, self._multi_price_backoff_until - time.time())

    def multi_price_backoff_remaining(self) -> float:
        return self._multi_price_backoff_remaining()

    def fluctuation_backoff_remaining(self) -> float:
        return max(0.0, self._fluctuation_backoff_until - time.time())

    def _client_cooldown(self) -> float:
        try:
            return max(0.0, float(self.client.rate_limit_cooldown_remaining()))
        except Exception:
            return 0.0

    def _note_fluctuation_rate_limit(self) -> float:
        is_paper = self.client.config.is_paper
        client_cooldown = self._client_cooldown()
        base_backoff = 30.0 if is_paper else 10.0
        backoff = max(base_backoff, client_cooldown + 1.0)
        self._fluctuation_backoff_until = max(self._fluctuation_backoff_until, time.time() + backoff)
        return backoff

    def _note_fluctuation_transport_error(self) -> float:
        is_paper = self.client.config.is_paper
        client_cooldown = self._client_cooldown()
        base_backoff = 6.0 if is_paper else 2.0
        backoff_cap = 8.0 if is_paper else 4.0
        backoff = max(base_backoff, min(backoff_cap, client_cooldown + 1.0))
        self._fluctuation_backoff_until = max(self._fluctuation_backoff_until, time.time() + backoff)
        return backoff

    def _note_multi_price_rate_limit(self, *, minimum_cooldown: float = 0.0) -> float:
        is_paper = self.client.config.is_paper
        self._multi_price_rate_limit_streak = min(3, int(self._multi_price_rate_limit_streak) + 1)
        if is_paper:
            staged_backoff = (8.0, 15.0, 30.0)
        else:
            staged_backoff = (2.0, 4.0, 8.0)
        base_cooldown = staged_backoff[self._multi_price_rate_limit_streak - 1]
        client_cooldown = self._client_cooldown()
        backoff = max(base_cooldown, minimum_cooldown, client_cooldown + 1.0)
        self._multi_price_backoff_until = max(
            self._multi_price_backoff_until,
            time.time() + backoff,
        )
        return backoff

    def _note_multi_price_transport_error(self) -> float:
        is_paper = self.client.config.is_paper
        self._multi_price_transport_error_streak = min(3, int(self._multi_price_transport_error_streak or 0) + 1)
        staged_backoff = (4.0, 6.0, 8.0) if is_paper else (1.5, 2.5, 4.0)
        backoff = staged_backoff[self._multi_price_transport_error_streak - 1]
        self._multi_price_backoff_until = max(
            self._multi_price_backoff_until,
            time.time() + backoff,
        )
        return backoff

    def get_daily_prices(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        period: str = "D",
        adjusted: bool = True,
    ) -> pd.DataFrame:
        """기간별 시세(일/주/월/년)를 조회한다.

        Args:
            symbol: 종목코드
            start_date: 시작일 (YYYYMMDD)
            end_date: 종료일 (YYYYMMDD)
            period: D(일), W(주), M(월), Y(년)
            adjusted: True면 수정주가
        """
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": symbol,
            "FID_INPUT_DATE_1": start_date,
            "FID_INPUT_DATE_2": end_date,
            "FID_PERIOD_DIV_CODE": period,
            "FID_ORG_ADJ_PRC": "0" if adjusted else "1",
        }

        res = None
        for attempt in range(4):
            res = self.client.get(
                api_url="/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice",
                tr_id="FHKST03010100",
                params=params,
            )
            if res.success:
                break

            error_message = res.error_message or ""
            is_rate_limited = self._is_rate_limited_message(error_message)
            if not is_rate_limited or attempt == 3:
                logger.error("기간별 시세 조회 실패 [%s]: %s", symbol, error_message)
                return pd.DataFrame()

            backoff = 1.0 + (attempt * 0.7)
            logger.warning(
                "기간별 시세 조회 재시도 [%s]: 호출 제한 감지 (%d/4, %.1fs 후 재시도)",
                symbol,
                attempt + 1,
                backoff,
            )
            time.sleep(backoff)

        data = res.output2
        if not data:
            return pd.DataFrame()
        return pd.DataFrame(data)

    def get_daily_minute_prices(
        self,
        symbol: str,
        target_date: str,
        end_time: str = "153000",
        include_past: bool = True,
        include_fake_ticks: bool = False,
    ) -> pd.DataFrame:
        """특정 거래일의 분봉 데이터를 조회한다.

        주식일별분봉조회(v1_국내주식-213)를 사용한다.
        한 번의 호출에 최대 120건까지만 반환되므로, 호출자는 마지막 분봉의
        날짜/시간을 이용해 추가 페이지네이션을 수행해야 한다.
        """
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": symbol,
            "FID_INPUT_HOUR_1": end_time,
            "FID_INPUT_DATE_1": target_date,
            "FID_PW_DATA_INCU_YN": "Y" if include_past else "N",
            "FID_FAKE_TICK_INCU_YN": "Y" if include_fake_ticks else "",
        }

        res = None
        for attempt in range(4):
            res = self.client.get(
                api_url="/uapi/domestic-stock/v1/quotations/inquire-time-dailychartprice",
                tr_id="FHKST03010230",
                params=params,
            )
            if res.success:
                break

            error_message = res.error_message or ""
            is_rate_limited = self._is_rate_limited_message(error_message)
            if not is_rate_limited or attempt == 3:
                logger.error("일별 분봉 조회 실패 [%s/%s]: %s", symbol, target_date, error_message)
                return pd.DataFrame()

            backoff = 1.0 + (attempt * 0.7)
            logger.warning(
                "일별 분봉 조회 재시도 [%s/%s]: 호출 제한 감지 (%d/4, %.1fs 후 재시도)",
                symbol,
                target_date,
                attempt + 1,
                backoff,
            )
            time.sleep(backoff)

        data = res.output2
        if not data:
            return pd.DataFrame()
        return pd.DataFrame(data)

    def get_fluctuation_symbols(
        self,
        count: int = 20,
        min_change_rate: float = 0.0,
    ) -> List[str]:
        """등락률 상위 응답에서 감시할 종목코드만 반환한다."""
        cache_key = (int(count), float(min_change_rate))
        cached = self._fluctuation_cache.get(cache_key)
        if cached and self._cache_valid(cached[0], 45.0):
            return list(cached[1])[:count]

        if self.fluctuation_backoff_remaining() > 0:
            if cached:
                return list(cached[1])[:count]
            return []

        try:
            res = self.client.get(
                api_url="/uapi/domestic-stock/v1/ranking/fluctuation",
                tr_id="FHPST01700000",
                params={
                    "fid_cond_mrkt_div_code": "J",
                    "fid_cond_scr_div_code": "20170",
                    "fid_input_iscd": "0000",
                    "fid_rank_sort_cls_code": "0",
                    "fid_input_cnt_1": str(max(1, int(count))),
                    "fid_prc_cls_code": "0",
                    "fid_input_price_1": "",
                    "fid_input_price_2": "",
                    "fid_vol_cnt": "",
                    "fid_trgt_cls_code": "0",
                    # KIS exclusion bitmap position 8 is ETN. Ranking responses
                    # otherwise contain Q-prefixed issue identifiers that cannot
                    # be sent to the six-character domestic-stock quote route.
                    "fid_trgt_exls_cls_code": "0000000100",
                    "fid_div_cls_code": "0",
                    "fid_rsfl_rate1": str(min_change_rate) if min_change_rate else "",
                    "fid_rsfl_rate2": "30",
                },
                log_timeout=False,
            )
        except requests.exceptions.RequestException as exc:
            backoff = self._note_fluctuation_transport_error()
            if cached:
                logger.warning("등락률 순위 조회 네트워크 오류: 직전 캐시 사용 (%.0fs 냉각): %s", backoff, exc)
                return list(cached[1])[:count]
            logger.warning("등락률 순위 조회 네트워크 오류: %.0fs 냉각 후 재시도: %s", backoff, exc)
            return []
        if not res.success:
            if self._is_rate_limited_message(res.error_message):
                self._note_fluctuation_rate_limit()
                if cached:
                    logger.warning("등락률 순위 조회 제한 감지: 직전 캐시 사용")
                    return list(cached[1])[:count]
            logger.error("등락률 순위 조회 실패: %s", res.error_message)
            return []

        symbols = []
        for row in (res.output or [])[:count]:
            symbol = str(row.get("stck_shrn_iscd", "") or "").strip()
            if len(symbol) == 6 and symbol.isdigit() and symbol not in symbols:
                symbols.append(symbol)
        if symbols:
            self._fluctuation_cache[cache_key] = (time.time(), list(symbols))
            self._fluctuation_backoff_until = 0.0
        return symbols

    def get_multi_price(self, symbols: List[str]) -> List[Quote]:
        """WebSocket 미수신 종목을 최대 30개까지 한 번에 보완한다."""
        if not symbols:
            return []
        symbols = symbols[:30]
        cooldown_remaining = max(0.0, float(self.client.rate_limit_cooldown_remaining()))
        if cooldown_remaining > 0:
            return []

        endpoint_backoff = self._multi_price_backoff_remaining()
        if endpoint_backoff > 0:
            now_ts = time.time()
            if now_ts - self._last_multi_price_backoff_skip_log_at >= 30.0:
                logger.warning(
                    "멀티시세 조회 냉각 유지: 신규 조회를 잠시 건너뜁니다. (%d종목, %.1fs 남음)",
                    len(symbols),
                    endpoint_backoff,
                )
                self._last_multi_price_backoff_skip_log_at = now_ts
            return []

        params = {}
        for i, sym in enumerate(symbols, 1):
            params[f"FID_COND_MRKT_DIV_CODE_{i}"] = "J"
            params[f"FID_INPUT_ISCD_{i}"] = sym

        try:
            res = self.client.get(
                api_url="/uapi/domestic-stock/v1/quotations/intstock-multprice",
                tr_id="FHKST11300006",
                params=params,
                log_timeout=False,
            )
        except requests.exceptions.RequestException as exc:
            backoff = self._note_multi_price_transport_error()
            now_ts = time.time()
            if now_ts - self._last_multi_price_rate_limit_log_at >= 30.0:
                logger.warning(
                    "멀티시세 조회 네트워크 오류: %.0fs 냉각 후 다음 틱에서 다시 확인합니다. (%d종목): %s",
                    backoff,
                    len(symbols),
                    exc,
                )
                self._last_multi_price_rate_limit_log_at = now_ts
            return []
        if not res.success:
            error_message = res.error_message or ""
            if self._is_rate_limited_message(error_message):
                backoff = self._note_multi_price_rate_limit()
                now_ts = time.time()
                if now_ts - self._last_multi_price_rate_limit_log_at >= 30.0:
                    logger.warning(
                        "멀티시세 조회 제한 감지: %.1fs 냉각 후 다음 틱에서 다시 확인합니다. (%d종목)",
                        backoff,
                        len(symbols),
                    )
                    self._last_multi_price_rate_limit_log_at = now_ts
                return []
            logger.error("멀티시세 조회 실패: %s", error_message)
            return []

        quotes = []
        for row in (res.output or []):
            sym = row.get("inter_shrn_iscd", "")
            if not sym:
                continue
            try:
                quotes.append(Quote(
                    symbol=sym,
                    current_price=int(row.get("inter2_prpr", 0)),
                    ask_price=int(row.get("inter2_askp", 0) or 0),
                    bid_price=int(row.get("inter2_bidp", 0) or 0),
                    cumulative_volume=int(row.get("inter2_acml_vol", 0) or 0),
                    book_available=True,
                    flow_available=False,
                    book_depth_available=False,
                ))
            except (ValueError, TypeError):
                continue
        if quotes:
            self._multi_price_backoff_until = 0.0
            self._multi_price_rate_limit_streak = 0
            self._multi_price_transport_error_streak = 0
        return quotes

    def is_market_open(self, date: str = None) -> bool:
        """오늘(또는 지정일)이 거래일인지 확인한다."""
        try:
            date = normalize_trading_date(date)
        except ValueError:
            date = datetime.today().strftime("%Y%m%d")
        if date in self._market_open_cache:
            return self._market_open_cache[date]
        if is_known_krx_closed_date(date):
            if date not in self._holiday_warned_dates:
                logger.info("로컬 KRX 휴장 캘린더로 휴장 처리: date=%s", date)
                self._holiday_warned_dates.add(date)
            self._market_open_cache[date] = False
            return False
        if self.client.config.is_paper:
            is_open = self._weekday_fallback_open(date)
            self._market_open_cache[date] = is_open
            return is_open

        res = self.client.get(
            api_url="/uapi/domestic-stock/v1/quotations/chk-holiday",
            tr_id="CTCA0903R",
            params={
                "BASS_DT": date,
                "CTX_AREA_FK": "",
                "CTX_AREA_NK": "",
            },
        )
        if not res.success:
            is_open = self._weekday_fallback_open(date)
            if res.error_code == "OPSQ0002":
                if date not in self._holiday_warned_dates:
                    logger.warning(
                        "휴장일 조회 미지원(OPSQ0002): 주중 fallback 사용 (date=%s)",
                        date,
                    )
                    self._holiday_warned_dates.add(date)
            else:
                logger.warning("휴장일 조회 실패, 주중 fallback 사용: %s", res.error_message)
            self._market_open_cache[date] = is_open
            return is_open

        output = res.output
        if isinstance(output, list) and len(output) > 0:
            # opnd_yn == "Y"이면 거래일
            today_info = output[0]
            is_open = today_info.get("opnd_yn", "Y") == "Y"
            self._market_open_cache[date] = is_open
            return is_open

        is_open = self._weekday_fallback_open(date)
        self._market_open_cache[date] = is_open
        return is_open

    def _weekday_fallback_open(self, date: str) -> bool:
        try:
            return datetime.strptime(date, "%Y%m%d").weekday() < 5
        except ValueError:
            return datetime.today().weekday() < 5
