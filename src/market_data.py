import logging
from datetime import datetime
from typing import List, Optional

import pandas as pd

from src.api_client import KISClient
from src.models import Quote, RankingItem

logger = logging.getLogger("kis_trader.market")


class MarketDataAPI:
    """국내주식 시세 조회 API."""

    def __init__(self, client: KISClient):
        self.client = client
        self._market_open_cache: dict[str, bool] = {}
        self._holiday_warned_dates: set[str] = set()

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
        res = self.client.get(
            api_url="/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice",
            tr_id="FHKST03010100",
            params={
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": symbol,
                "FID_INPUT_DATE_1": start_date,
                "FID_INPUT_DATE_2": end_date,
                "FID_PERIOD_DIV_CODE": period,
                "FID_ORG_ADJ_PRC": "0" if adjusted else "1",
            },
        )
        if not res.success:
            logger.error("기간별 시세 조회 실패 [%s]: %s", symbol, res.error_message)
            return pd.DataFrame()

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
        )
        if not res.success:
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

        params = {}
        for i, sym in enumerate(symbols, 1):
            params[f"FID_COND_MRKT_DIV_CODE_{i}"] = "J"
            params[f"FID_INPUT_ISCD_{i}"] = sym

        res = self.client.get(
            api_url="/uapi/domestic-stock/v1/quotations/intstock-multprice",
            tr_id="FHKST11300006",
            params=params,
        )
        if not res.success:
            logger.error("멀티시세 조회 실패: %s", res.error_message)
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

        res = self.client.get(
            api_url="/uapi/domestic-stock/v1/quotations/inquire-daily-indexchartprice",
            tr_id="FHKUP03500100",
            params={
                "FID_COND_MRKT_DIV_CODE": "U",
                "FID_INPUT_ISCD": index_code,
                "FID_INPUT_DATE_1": start_date,
                "FID_INPUT_DATE_2": end_date,
                "FID_PERIOD_DIV_CODE": "D",
            },
        )
        if not res.success:
            logger.error("인덱스 일봉 조회 실패 [%s]: %s", index_code, res.error_message)
            return pd.DataFrame()

        data = res.output2
        if not data:
            return pd.DataFrame()
        return pd.DataFrame(data)

    def is_market_open(self, date: str = None) -> bool:
        """오늘(또는 지정일)이 거래일인지 확인한다."""
        if date is None:
            date = datetime.today().strftime("%Y%m%d")
        if date in self._market_open_cache:
            return self._market_open_cache[date]

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
