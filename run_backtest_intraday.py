#!/usr/bin/env python3
"""Run the one-minute replay backtest for the expected-value strategy."""

import os
import sys
import tempfile
from datetime import datetime, timedelta
from typing import Optional


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from src.api_client import KISClient
from src.auth import TokenManager
from src.backtest.data_fetcher import HistoricalDataFetcher, MinuteHistoricalDataFetcher
from src.backtest.intraday_engine import IntradayBacktestEngine
from src.backtest.report import print_report
from src.config import Config
from src.market_data import MarketDataAPI
from src.strategies.momentum_scalp import (
    DEFAULT_INVERSE_ETFS,
    DEFAULT_STATIC_WATCHLIST,
    MomentumScalpConfig,
    MomentumScalpStrategy,
)


def build_strategy_config(directory: str = "") -> MomentumScalpConfig:
    state_root = directory or tempfile.gettempdir()
    return MomentumScalpConfig(
        seed_money=1_000_000,
        daily_profit_target=10_000,
        daily_loss_limit=-5_000,
        commission_rate=0.00015,
        sell_tax_rate=0.002,
        daily_state_path=os.path.join(state_root, "msg-intraday-backtest-state.json"),
        forecast_outcome_root=os.path.join(state_root, "msg-intraday-backtest-forecasts"),
    )


def _resolve_recent_trade_dates(
    daily_data: dict[str, object],
    limit: int = 20,
) -> list[str]:
    dates = set()
    for frame in daily_data.values():
        if "stck_bsop_date" in frame.columns:
            dates.update(frame["stck_bsop_date"].tolist())
    return sorted(dates)[-limit:]


def _validate_intraday_backtest_mode(config: Config) -> Optional[str]:
    if config.is_paper:
        return (
            "paper 모드에서는 KIS 1분봉 조회를 공식 검증 소스로 사용할 수 없습니다. "
            "일봉 smoke 백테스트와 실거래 forecast 원장을 사용해주십시오."
        )
    return None


def main() -> None:
    runtime_config = Config.load()
    unsupported_reason = _validate_intraday_backtest_mode(runtime_config)
    if unsupported_reason:
        print(unsupported_reason)
        raise SystemExit(1)

    client = KISClient(runtime_config, TokenManager(runtime_config))
    market_data = MarketDataAPI(client)
    symbols = DEFAULT_STATIC_WATCHLIST + DEFAULT_INVERSE_ETFS
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=45)).strftime("%Y%m%d")

    print(f"거래일 계산용 일봉 다운로드: {len(symbols)}종목 ({start_date}~{end_date})...")
    daily_data = HistoricalDataFetcher(market_data).fetch_pool(
        symbols,
        start_date,
        end_date,
    )
    trade_dates = _resolve_recent_trade_dates(daily_data)
    if not trade_dates:
        print("최근 거래일을 계산하지 못했습니다.")
        return

    print(
        f"1분봉 다운로드: {len(symbols)}종목 "
        f"({trade_dates[0]}~{trade_dates[-1]}, {len(trade_dates)}거래일)..."
    )
    minute_data = MinuteHistoricalDataFetcher(market_data).fetch_pool(
        symbols,
        trade_dates,
    )
    print(f"→ {len(minute_data)}종목 1분봉 로드 완료")
    if not minute_data:
        return

    with tempfile.TemporaryDirectory(prefix="msg-intraday-backtest-") as directory:
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=build_strategy_config(directory),
            fixed_pool=list(minute_data),
        )
        engine = IntradayBacktestEngine(
            strategy=strategy,
            data=minute_data,
            initial_capital=1_000_000,
            commission_rate=0.00015,
            tax_rate=0.002,
        )
        result = engine.run(trade_dates[0], trade_dates[-1])
    print_report(result)


if __name__ == "__main__":
    main()
