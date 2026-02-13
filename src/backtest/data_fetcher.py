"""과거 일봉 데이터를 다운로드하고 캐싱한다."""

import logging
import os
import time
from typing import Dict, List, Optional

import pandas as pd

from src.market_data import MarketDataAPI

logger = logging.getLogger("kis_trader.backtest.data")

DEFAULT_CACHE_DIR = "data/daily"


class HistoricalDataFetcher:
    """KIS API에서 일봉 데이터를 가져오고 Parquet으로 캐싱한다."""

    def __init__(self, market_data: MarketDataAPI, cache_dir: str = DEFAULT_CACHE_DIR):
        self.market_data = market_data
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def fetch_and_cache(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """일봉 데이터를 가져오고 캐싱한다. 캐시가 있으면 재사용."""
        cache_path = os.path.join(self.cache_dir, f"{symbol}.parquet")

        # 캐시 확인
        cached = self._load_cached(cache_path, start_date, end_date)
        if cached is not None:
            return cached

        # API에서 가져오기 (페이지네이션)
        logger.info("데이터 다운로드: %s (%s ~ %s)", symbol, start_date, end_date)
        all_data = []
        current_end = end_date

        for page in range(10):  # 최대 10페이지
            df = self.market_data.get_daily_prices(symbol, start_date, current_end)
            if df.empty:
                break

            all_data.append(df)

            # 가장 오래된 날짜 확인 (데이터는 최신순)
            dates = df.get("stck_bsop_date", pd.Series())
            if dates.empty:
                break

            oldest = dates.iloc[-1]
            if oldest <= start_date:
                break  # 요청 범위 도달

            # 다음 페이지: 가장 오래된 날짜 하루 전까지
            current_end = str(int(oldest) - 1).zfill(8)
            time.sleep(0.6)  # rate limit

        if not all_data:
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=True).drop_duplicates(subset=["stck_bsop_date"])

        # 날짜 범위 필터
        if "stck_bsop_date" in result.columns:
            result = result[
                (result["stck_bsop_date"] >= start_date) &
                (result["stck_bsop_date"] <= end_date)
            ]

        # 캐싱
        if not result.empty:
            result.to_parquet(cache_path, index=False)
            logger.info("캐시 저장: %s (%d rows)", cache_path, len(result))

        return result

    def fetch_pool(
        self, symbols: List[str], start_date: str, end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """여러 종목의 일봉 데이터를 가져온다."""
        data = {}
        for i, symbol in enumerate(symbols):
            logger.info("[%d/%d] %s 데이터 로드 중...", i + 1, len(symbols), symbol)
            df = self.fetch_and_cache(symbol, start_date, end_date)
            if not df.empty:
                data[symbol] = df
            time.sleep(0.3)  # rate limit 여유
        logger.info("총 %d/%d 종목 데이터 로드 완료", len(data), len(symbols))
        return data

    def _load_cached(
        self, cache_path: str, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """캐시 파일이 있고 범위를 커버하면 로드한다."""
        if not os.path.exists(cache_path):
            return None

        try:
            df = pd.read_parquet(cache_path)
            if "stck_bsop_date" not in df.columns or df.empty:
                return None

            cached_min = df["stck_bsop_date"].min()
            cached_max = df["stck_bsop_date"].max()

            if cached_min <= start_date and cached_max >= end_date:
                filtered = df[
                    (df["stck_bsop_date"] >= start_date) &
                    (df["stck_bsop_date"] <= end_date)
                ]
                logger.debug("캐시 사용: %s (%d rows)", cache_path, len(filtered))
                return filtered

        except Exception:
            pass

        return None
