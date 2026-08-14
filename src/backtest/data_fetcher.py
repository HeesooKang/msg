"""과거 일봉/분봉 데이터를 다운로드하고 캐싱한다."""

import logging
import os
import time
from datetime import timedelta
from typing import Dict, List, Optional

import pandas as pd

from src.market_data import MarketDataAPI

logger = logging.getLogger("kis_trader.backtest.data")

DEFAULT_CACHE_DIR = "data/daily"
DEFAULT_MINUTE_CACHE_DIR = "data/minute"


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
        partial_cached = self._load_cached_slice(cache_path, start_date, end_date)

        # API에서 가져오기 (페이지네이션)
        logger.info("데이터 다운로드: %s (%s ~ %s)", symbol, start_date, end_date)
        all_data = []
        current_end = end_date

        try:
            for _ in range(10):  # 최대 10페이지
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
        except Exception as exc:
            if partial_cached is not None and not partial_cached.empty:
                logger.warning(
                    "데이터 다운로드 실패, 부분 캐시로 대체: %s (%s)",
                    symbol,
                    exc,
                )
                return partial_cached
            raise

        if not all_data:
            if partial_cached is not None and not partial_cached.empty:
                logger.warning("부분 캐시 사용: %s", symbol)
                return partial_cached
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
            try:
                df = self.fetch_and_cache(symbol, start_date, end_date)
            except Exception as exc:
                logger.warning("데이터 로드 실패 [%s]: %s", symbol, exc)
                continue
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

    def _load_cached_slice(
        self, cache_path: str, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """캐시가 일부만 있어도 요청 구간과 겹치는 데이터는 반환한다."""
        if not os.path.exists(cache_path):
            return None

        try:
            df = pd.read_parquet(cache_path)
            if "stck_bsop_date" not in df.columns or df.empty:
                return None
            filtered = df[
                (df["stck_bsop_date"] >= start_date) &
                (df["stck_bsop_date"] <= end_date)
            ]
            if filtered.empty:
                return None
            return filtered
        except Exception:
            return None


class MinuteHistoricalDataFetcher:
    """KIS API에서 거래일별 1분봉을 가져오고 캐싱한다."""

    def __init__(self, market_data: MarketDataAPI, cache_dir: str = DEFAULT_MINUTE_CACHE_DIR):
        self.market_data = market_data
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def fetch_and_cache(self, symbol: str, trade_date: str) -> pd.DataFrame:
        cache_path = os.path.join(self.cache_dir, f"{symbol}_{trade_date}_1m.parquet")
        cached = self._load_cached(cache_path)
        if cached is not None:
            return cached

        logger.info("1분봉 다운로드: %s (%s)", symbol, trade_date)
        all_rows: List[pd.DataFrame] = []
        query_date = trade_date
        query_time = "153000"

        for _ in range(12):
            df = self.market_data.get_daily_minute_prices(
                symbol=symbol,
                target_date=query_date,
                end_time=query_time,
                include_past=True,
            )
            if df.empty:
                break

            normalized = self._normalize_minute_bars(symbol, df)
            if normalized.empty:
                break
            all_rows.append(normalized)

            last_row = normalized.iloc[-1]
            last_dt = last_row["timestamp"] - timedelta(minutes=1)
            if last_row["trade_date"] < trade_date:
                break
            if last_dt.strftime("%Y%m%d") < trade_date:
                break
            query_date = last_dt.strftime("%Y%m%d")
            query_time = last_dt.strftime("%H%M%S")
            time.sleep(0.35)

        if not all_rows:
            return pd.DataFrame()

        result = (
            pd.concat(all_rows, ignore_index=True)
            .drop_duplicates(subset=["timestamp"])
            .sort_values("timestamp")
        )
        result = result[result["trade_date"] == trade_date].reset_index(drop=True)
        if not result.empty:
            result.to_parquet(cache_path, index=False)
            logger.info("1분봉 캐시 저장: %s (%d rows)", cache_path, len(result))
        return result

    def fetch_pool(self, symbols: List[str], trade_dates: List[str]) -> Dict[str, pd.DataFrame]:
        data: Dict[str, pd.DataFrame] = {}
        total = max(1, len(symbols) * max(1, len(trade_dates)))
        step = 0
        for symbol in symbols:
            frames = []
            for trade_date in trade_dates:
                step += 1
                logger.info("[%d/%d] %s %s 1분봉 로드 중...", step, total, symbol, trade_date)
                df = self.fetch_and_cache(symbol, trade_date)
                if not df.empty:
                    frames.append(df)
                time.sleep(0.2)
            if frames:
                data[symbol] = pd.concat(frames, ignore_index=True).sort_values("timestamp")
        logger.info("총 %d종목 1분봉 로드 완료", len(data))
        return data

    def _load_cached(self, cache_path: str) -> Optional[pd.DataFrame]:
        if not os.path.exists(cache_path):
            return None
        try:
            df = pd.read_parquet(cache_path)
            if df.empty or "timestamp" not in df.columns:
                return None
            return df.sort_values("timestamp").reset_index(drop=True)
        except Exception:
            return None

    @staticmethod
    def _normalize_minute_bars(symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        work = df.copy()
        date_col = "stck_bsop_date" if "stck_bsop_date" in work.columns else None
        time_col = "stck_cntg_hour" if "stck_cntg_hour" in work.columns else None
        if date_col is None or time_col is None:
            return pd.DataFrame()

        for col in ("stck_oprc", "stck_hgpr", "stck_lwpr", "stck_prpr", "cntg_vol", "acml_vol"):
            if col in work.columns:
                work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0).astype(int)

        work["trade_date"] = work[date_col].astype(str).str.zfill(8)
        work["trade_time"] = work[time_col].astype(str).str.zfill(6)
        work["timestamp"] = pd.to_datetime(
            work["trade_date"] + work["trade_time"],
            format="%Y%m%d%H%M%S",
            errors="coerce",
        )
        work = work.dropna(subset=["timestamp"]).copy()
        if work.empty:
            return pd.DataFrame()

        work["symbol"] = symbol
        work["open"] = work["stck_oprc"] if "stck_oprc" in work.columns else 0
        work["high"] = work["stck_hgpr"] if "stck_hgpr" in work.columns else work["open"]
        work["low"] = work["stck_lwpr"] if "stck_lwpr" in work.columns else work["open"]
        work["close"] = work["stck_prpr"] if "stck_prpr" in work.columns else work["open"]
        if "acml_vol" in work.columns and work["acml_vol"].max() > 0:
            work["cumulative_volume"] = work["acml_vol"].astype(int)
            work["volume"] = work["cntg_vol"].astype(int) if "cntg_vol" in work.columns else 0
        else:
            work["volume"] = work["cntg_vol"].astype(int) if "cntg_vol" in work.columns else 0
            work["cumulative_volume"] = work.groupby("trade_date")["volume"].cumsum()

        return work[
            [
                "symbol",
                "trade_date",
                "trade_time",
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "cumulative_volume",
            ]
        ].sort_values("timestamp")
