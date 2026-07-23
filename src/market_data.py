import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

from src.api_client import KISClient, is_kis_rate_limited_message
from src.market_calendar import is_known_krx_closed_date, normalize_trading_date
from src.models import Quote, RankingItem

logger = logging.getLogger("kis_trader.market")


class MarketDataAPI:
    """국내주식 시세 조회 API."""

    def __init__(self, client: KISClient):
        self.client = client
        self._market_open_cache: dict[str, bool] = {}
        self._holiday_warned_dates: set[str] = set()
        self._multi_price_cache: Dict[Tuple[str, ...], Tuple[float, List[Quote]]] = {}
        self._multi_price_symbol_cache: Dict[str, Tuple[float, Quote]] = {}
        self._multi_price_backoff_until: float = 0.0
        self._multi_price_rate_limit_streak: int = 0
        self._multi_price_transport_error_streak: int = 0
        self._fluctuation_cache: Dict[Tuple[int, float, float, int, int, int], Tuple[float, List[RankingItem]]] = {}
        self._fluctuation_backoff_until: float = 0.0
        self._index_daily_cache: Dict[Tuple[str, str, str], Tuple[float, pd.DataFrame]] = {}
        self._last_multi_price_cooldown_cache_log_at: float = 0.0
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

    def _note_fluctuation_rate_limit(self) -> float:
        cfg = getattr(self.client, "config", None)
        is_paper = bool(getattr(cfg, "is_paper", False))
        cooldown_fn = getattr(self.client, "rate_limit_cooldown_remaining", None)
        client_cooldown = 0.0
        if callable(cooldown_fn):
            try:
                client_cooldown = max(0.0, float(cooldown_fn()))
            except Exception:
                client_cooldown = 0.0
        base_backoff = 30.0 if is_paper else 10.0
        backoff = max(base_backoff, client_cooldown + 1.0)
        self._fluctuation_backoff_until = max(self._fluctuation_backoff_until, time.time() + backoff)
        return backoff

    def _note_fluctuation_transport_error(self) -> float:
        cfg = getattr(self.client, "config", None)
        is_paper = bool(getattr(cfg, "is_paper", False))
        cooldown_fn = getattr(self.client, "rate_limit_cooldown_remaining", None)
        client_cooldown = 0.0
        if callable(cooldown_fn):
            try:
                client_cooldown = max(0.0, float(cooldown_fn()))
            except Exception:
                client_cooldown = 0.0
        base_backoff = 6.0 if is_paper else 2.0
        backoff_cap = 8.0 if is_paper else 4.0
        backoff = max(base_backoff, min(backoff_cap, client_cooldown + 1.0))
        self._fluctuation_backoff_until = max(self._fluctuation_backoff_until, time.time() + backoff)
        return backoff

    def _cached_multi_price_quotes_for_symbols(
        self,
        symbols: List[str],
        *,
        ttl_seconds: float,
    ) -> List[Quote]:
        quotes: List[Quote] = []
        for symbol in symbols:
            cached = self._multi_price_symbol_cache.get(str(symbol))
            if not cached or not self._cache_valid(cached[0], ttl_seconds):
                return []
            quotes.append(cached[1])
        return quotes

    def get_cached_quotes(self, symbols: List[str], *, ttl_seconds: float) -> List[Quote]:
        quotes: List[Quote] = []
        seen = set()
        for symbol in symbols:
            normalized = str(symbol or "").strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            cached = self._multi_price_symbol_cache.get(normalized)
            if not cached or not self._cache_valid(cached[0], ttl_seconds):
                continue
            quotes.append(cached[1])
        return quotes

    def quote_cache_age_seconds(self, symbol: str) -> Optional[float]:
        cached = self._multi_price_symbol_cache.get(str(symbol or "").strip())
        if not cached:
            return None
        return max(0.0, time.time() - cached[0])

    def _note_multi_price_rate_limit(self, *, minimum_cooldown: float = 0.0) -> float:
        cfg = getattr(self.client, "config", None)
        is_paper = bool(getattr(cfg, "is_paper", False))
        self._multi_price_rate_limit_streak = min(3, int(self._multi_price_rate_limit_streak) + 1)
        if is_paper:
            staged_backoff = (8.0, 15.0, 30.0)
        else:
            staged_backoff = (2.0, 4.0, 8.0)
        base_cooldown = staged_backoff[self._multi_price_rate_limit_streak - 1]
        client_cooldown = 0.0
        cooldown_fn = getattr(self.client, "rate_limit_cooldown_remaining", None)
        if callable(cooldown_fn):
            try:
                client_cooldown = max(0.0, float(cooldown_fn()))
            except Exception:
                client_cooldown = 0.0
        backoff = max(base_cooldown, minimum_cooldown, client_cooldown + 1.0)
        self._multi_price_backoff_until = max(
            self._multi_price_backoff_until,
            time.time() + backoff,
        )
        return backoff

    def _note_multi_price_transport_error(self) -> float:
        cfg = getattr(self.client, "config", None)
        is_paper = bool(getattr(cfg, "is_paper", False))
        self._multi_price_transport_error_streak = min(3, int(self._multi_price_transport_error_streak or 0) + 1)
        staged_backoff = (4.0, 6.0, 8.0) if is_paper else (1.5, 2.5, 4.0)
        backoff = staged_backoff[self._multi_price_transport_error_streak - 1]
        self._multi_price_backoff_until = max(
            self._multi_price_backoff_until,
            time.time() + backoff,
        )
        return backoff

    def get_current_price(self, symbol: str) -> Optional[Quote]:
        """주식 현재가를 조회한다.

        Args:
            symbol: 종목코드 (예: "005930")
        """
        res = self.client.get(
            api_url="/uapi/domestic-stock/v1/quotations/inquire-price",
            tr_id="FHKST01010100",
            params={
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": symbol,
            },
        )
        if not res.success:
            logger.error("현재가 조회 실패 [%s]: %s", symbol, res.error_message)
            return None

        o = res.output
        return Quote(
            symbol=symbol,
            name=o.get("hts_kor_isnm", ""),
            current_price=int(o.get("stck_prpr", 0)),
            change=int(o.get("prdy_vrss", 0)),
            change_rate=float(o.get("prdy_ctrt", 0)),
            open_price=int(o.get("stck_oprc", 0)),
            high_price=int(o.get("stck_hgpr", 0)),
            low_price=int(o.get("stck_lwpr", 0)),
            volume=int(o.get("acml_vol", 0)),
            trade_amount=int(o.get("acml_tr_pbmn", 0)),
        )

    def get_orderbook(self, symbol: str) -> Optional[dict]:
        """호가/예상체결을 조회한다."""
        res = self.client.get(
            api_url="/uapi/domestic-stock/v1/quotations/inquire-asking-price-exp-ccn",
            tr_id="FHKST01010200",
            params={
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": symbol,
            },
        )
        if not res.success:
            logger.error("호가 조회 실패 [%s]: %s", symbol, res.error_message)
            return None

        return {"orderbook": res.output1, "expected": res.output2}

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

    def get_fluctuation_ranking(
        self,
        count: int = 20,
        min_change_rate: float = 0.0,
        max_change_rate: float = 30.0,
        min_price: int = 0,
        max_price: int = 0,
        min_volume: int = 0,
    ) -> List[RankingItem]:
        """등락률 상위 종목을 조회한다."""
        cache_key = (int(count), float(min_change_rate), float(max_change_rate), int(min_price), int(max_price), int(min_volume))
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
                    "fid_input_cnt_1": "0",
                    "fid_prc_cls_code": "0",
                    "fid_input_price_1": str(min_price) if min_price else "",
                    "fid_input_price_2": str(max_price) if max_price else "",
                    "fid_vol_cnt": str(min_volume) if min_volume else "",
                    "fid_trgt_cls_code": "0",
                    "fid_trgt_exls_cls_code": "0",
                    "fid_div_cls_code": "0",
                    "fid_rsfl_rate1": str(min_change_rate) if min_change_rate else "",
                    "fid_rsfl_rate2": str(max_change_rate) if max_change_rate else "",
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

        items = []
        for row in (res.output or [])[:count]:
            try:
                items.append(RankingItem(
                    symbol=row.get("stck_shrn_iscd", ""),
                    name=row.get("hts_kor_isnm", ""),
                    current_price=int(row.get("stck_prpr", 0)),
                    change_rate=float(row.get("prdy_ctrt", 0)),
                    volume=int(row.get("acml_vol", 0)),
                    rank=int(row.get("data_rank", 0)),
                ))
            except (ValueError, TypeError):
                continue
        if items:
            self._fluctuation_cache[cache_key] = (time.time(), list(items))
            self._fluctuation_backoff_until = 0.0
        return items

    def get_market_cap_ranking(self, count: int = 30) -> List[RankingItem]:
        """시가총액 상위 종목을 조회한다."""
        res = self.client.get(
            api_url="/uapi/domestic-stock/v1/ranking/market-cap",
            tr_id="FHPST01740000",
            params={
                "fid_cond_mrkt_div_code": "J",
                "fid_cond_scr_div_code": "20174",
                "fid_div_cls_code": "0",
                "fid_input_iscd": "0000",
                "fid_trgt_cls_code": "0",
                "fid_trgt_exls_cls_code": "0",
                "fid_input_price_1": "",
                "fid_input_price_2": "",
                "fid_vol_cnt": "",
            },
        )
        if not res.success:
            logger.error("시가총액 순위 조회 실패: %s", res.error_message)
            return []

        items = []
        for row in (res.output or [])[:count]:
            try:
                items.append(RankingItem(
                    symbol=row.get("mksc_shrn_iscd", ""),
                    name=row.get("hts_kor_isnm", ""),
                    current_price=int(row.get("stck_prpr", 0)),
                    change_rate=float(row.get("prdy_ctrt", 0)),
                    volume=int(row.get("acml_vol", 0)),
                    rank=int(row.get("data_rank", 0)),
                ))
            except (ValueError, TypeError):
                continue
        return items

    def get_multi_price(self, symbols: List[str]) -> List[Quote]:
        """최대 30종목의 시세를 한번에 조회한다."""
        if not symbols:
            return []
        symbols = symbols[:30]
        cache_key = tuple(symbols)
        cached = self._multi_price_cache.get(cache_key)
        cooldown_remaining = max(0.0, float(self.client.rate_limit_cooldown_remaining()))
        if (
            cooldown_remaining > 0
            and cached
            and self._cache_valid(cached[0], 5.0)
        ):
            now_ts = time.time()
            if now_ts - self._last_multi_price_cooldown_cache_log_at >= 30.0:
                logger.warning(
                    "멀티시세 조회 냉각 중 직전 캐시 사용 (%d종목, %.1fs 남음)",
                    len(symbols),
                    cooldown_remaining,
                )
                self._last_multi_price_cooldown_cache_log_at = now_ts
            return list(cached[1])
        if cooldown_remaining > 0:
            symbol_cached_quotes = self._cached_multi_price_quotes_for_symbols(symbols, ttl_seconds=5.0)
            if symbol_cached_quotes:
                now_ts = time.time()
                if now_ts - self._last_multi_price_cooldown_cache_log_at >= 30.0:
                    logger.warning(
                        "멀티시세 조회 냉각 중 심볼 캐시 사용 (%d종목, %.1fs 남음)",
                        len(symbols),
                        cooldown_remaining,
                    )
                    self._last_multi_price_cooldown_cache_log_at = now_ts
                return list(symbol_cached_quotes)

        endpoint_backoff = self._multi_price_backoff_remaining()
        if endpoint_backoff > 0:
            now_ts = time.time()
            if cached and self._cache_valid(cached[0], 20.0):
                if now_ts - self._last_multi_price_cooldown_cache_log_at >= 30.0:
                    logger.warning(
                        "멀티시세 조회 냉각 중 직전 캐시 사용 (%d종목, %.1fs 남음)",
                        len(symbols),
                        endpoint_backoff,
                    )
                    self._last_multi_price_cooldown_cache_log_at = now_ts
                return list(cached[1])
            symbol_cached_quotes = self._cached_multi_price_quotes_for_symbols(symbols, ttl_seconds=20.0)
            if symbol_cached_quotes:
                if now_ts - self._last_multi_price_cooldown_cache_log_at >= 30.0:
                    logger.warning(
                        "멀티시세 조회 냉각 중 심볼 캐시 사용 (%d종목, %.1fs 남음)",
                        len(symbols),
                        endpoint_backoff,
                    )
                    self._last_multi_price_cooldown_cache_log_at = now_ts
                return list(symbol_cached_quotes)
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
            if cached and self._cache_valid(cached[0], 20.0):
                if now_ts - self._last_multi_price_rate_limit_log_at >= 30.0:
                    logger.warning(
                        "멀티시세 조회 네트워크 오류: %.0fs 냉각 동안 직전 캐시를 사용합니다. (%d종목): %s",
                        backoff,
                        len(symbols),
                        exc,
                    )
                    self._last_multi_price_rate_limit_log_at = now_ts
                return list(cached[1])
            symbol_cached_quotes = self._cached_multi_price_quotes_for_symbols(symbols, ttl_seconds=20.0)
            if symbol_cached_quotes:
                if now_ts - self._last_multi_price_rate_limit_log_at >= 30.0:
                    logger.warning(
                        "멀티시세 조회 네트워크 오류: %.0fs 냉각 동안 심볼 캐시를 사용합니다. (%d종목): %s",
                        backoff,
                        len(symbols),
                        exc,
                    )
                    self._last_multi_price_rate_limit_log_at = now_ts
                return list(symbol_cached_quotes)
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
                if cached and self._cache_valid(cached[0], 20.0):
                    if now_ts - self._last_multi_price_rate_limit_log_at >= 30.0:
                        logger.warning(
                            "멀티시세 조회 제한 감지: %.1fs 냉각 동안 직전 캐시를 재사용합니다. (%d종목)",
                            backoff,
                            len(symbols),
                        )
                        self._last_multi_price_rate_limit_log_at = now_ts
                    return list(cached[1])
                symbol_cached_quotes = self._cached_multi_price_quotes_for_symbols(symbols, ttl_seconds=20.0)
                if symbol_cached_quotes:
                    if now_ts - self._last_multi_price_rate_limit_log_at >= 30.0:
                        logger.warning(
                            "멀티시세 조회 제한 감지: %.1fs 냉각 동안 심볼 캐시를 재사용합니다. (%d종목)",
                            backoff,
                            len(symbols),
                        )
                        self._last_multi_price_rate_limit_log_at = now_ts
                    return list(symbol_cached_quotes)
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
                    name=row.get("inter_kor_isnm", ""),
                    current_price=int(row.get("inter2_prpr", 0)),
                    change=int(row.get("inter2_prdy_vrss", 0)),
                    change_rate=float(row.get("prdy_ctrt", 0)),
                    open_price=int(row.get("inter2_oprc", 0)),
                    high_price=int(row.get("inter2_hgpr", 0)),
                    low_price=int(row.get("inter2_lwpr", 0)),
                    volume=int(row.get("acml_vol", 0)),
                    trade_amount=int(row.get("acml_tr_pbmn", 0)),
                ))
            except (ValueError, TypeError):
                continue
        if quotes:
            cached_at = time.time()
            self._multi_price_cache[cache_key] = (cached_at, list(quotes))
            for quote in quotes:
                self._multi_price_symbol_cache[str(quote.symbol)] = (cached_at, quote)
            self._multi_price_backoff_until = 0.0
            self._multi_price_rate_limit_streak = 0
            self._multi_price_transport_error_streak = 0
        return quotes

    def get_index_daily_prices(
        self,
        index_code: str = "0001",
        start_date: str = "",
        end_date: str = "",
    ) -> pd.DataFrame:
        """업종(인덱스) 일봉 시세를 조회한다.

        Args:
            index_code: "0001"=KOSPI, "1001"=KOSDAQ
            start_date: 시작일 (YYYYMMDD)
            end_date: 종료일 (YYYYMMDD)
        """
        if not end_date:
            end_date = datetime.today().strftime("%Y%m%d")
        if not start_date:
            from datetime import timedelta
            start_date = (datetime.today() - timedelta(days=45)).strftime("%Y%m%d")
        cache_key = (str(index_code), str(start_date), str(end_date))
        cached = self._index_daily_cache.get(cache_key)
        if cached and self._cache_valid(cached[0], 180.0):
            return cached[1].copy()

        res = None
        params = {
            "FID_COND_MRKT_DIV_CODE": "U",
            "FID_INPUT_ISCD": index_code,
            "FID_INPUT_DATE_1": start_date,
            "FID_INPUT_DATE_2": end_date,
            "FID_PERIOD_DIV_CODE": "D",
        }
        for attempt in range(3):
            res = self.client.get(
                api_url="/uapi/domestic-stock/v1/quotations/inquire-daily-indexchartprice",
                tr_id="FHKUP03500100",
                params=params,
            )
            if res.success:
                break
            error_message = res.error_message or ""
            if not self._is_rate_limited_message(error_message) or attempt == 2:
                if self._is_rate_limited_message(error_message) and cached:
                    logger.warning("인덱스 일봉 조회 제한 감지 [%s]: 직전 캐시 사용", index_code)
                    return cached[1].copy()
                logger.error("인덱스 일봉 조회 실패 [%s]: %s", index_code, error_message)
                return pd.DataFrame()
            backoff = max(1.0, float(self.client.rate_limit_cooldown_remaining()))
            logger.warning(
                "인덱스 일봉 조회 재시도 [%s]: 호출 제한 감지 (%d/3, %.1fs 후 재시도)",
                index_code,
                attempt + 1,
                backoff,
            )
            time.sleep(backoff)

        data = res.output2
        if not data:
            return pd.DataFrame()
        frame = pd.DataFrame(data)
        self._index_daily_cache[cache_key] = (time.time(), frame.copy())
        return frame

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
        cfg = getattr(self.client, "config", None)
        if bool(getattr(cfg, "is_paper", False)):
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
