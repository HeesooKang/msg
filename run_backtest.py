#!/usr/bin/env python3
"""Run the daily-bar smoke backtest for the expected-value strategy."""

import argparse
import os
import sys
import tempfile
from datetime import datetime, timedelta


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from src.api_client import KISClient
from src.auth import TokenManager
from src.backtest.data_fetcher import HistoricalDataFetcher
from src.backtest.engine import BacktestEngine
from src.backtest.report import print_report
from src.config import Config
from src.market_data import MarketDataAPI
from src.strategies.momentum_scalp import (
    DEFAULT_INVERSE_ETFS,
    DEFAULT_STATIC_WATCHLIST,
    MomentumScalpConfig,
    MomentumScalpStrategy,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="최근 N일 기준 일봉 백테스트 실행")
    parser.add_argument("--days", type=int, default=40)
    parser.add_argument("--end-date", help="종료 기준 일자(YYYYMMDD)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    os.environ["ALERTS_ENABLED"] = "false"

    runtime_config = Config.load()
    client = KISClient(runtime_config, TokenManager(runtime_config))
    market_data = MarketDataAPI(client)
    symbols = DEFAULT_STATIC_WATCHLIST + DEFAULT_INVERSE_ETFS

    end_at = (
        datetime.strptime(args.end_date, "%Y%m%d")
        if args.end_date
        else datetime.now() - timedelta(days=1)
    )
    start_at = end_at - timedelta(days=max(1, int(args.days * 1.5)))
    start_date = start_at.strftime("%Y%m%d")
    end_date = end_at.strftime("%Y%m%d")

    print(f"데이터 다운로드: {len(symbols)}종목 ({start_date}~{end_date})...")
    data = HistoricalDataFetcher(market_data).fetch_pool(
        symbols,
        start_date,
        end_date,
    )
    print(f"→ {len(data)}종목 로드 완료")
    if not data:
        return

    with tempfile.TemporaryDirectory(prefix="msg-daily-backtest-") as directory:
        strategy_config = MomentumScalpConfig(
            seed_money=1_000_000,
            daily_profit_target=10_000,
            daily_loss_limit=-5_000,
            commission_rate=0.00015,
            sell_tax_rate=0.002,
            static_watchlist=list(data),
            inverse_etfs=[symbol for symbol in DEFAULT_INVERSE_ETFS if symbol in data],
            daily_state_path=os.path.join(directory, "state.json"),
            forecast_outcome_root=os.path.join(directory, "forecasts"),
        )
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=strategy_config,
            fixed_pool=list(data),
        )
        engine = BacktestEngine(
            strategy=strategy,
            data=data,
            initial_capital=1_000_000,
            commission_rate=0.00015,
            tax_rate=0.002,
        )
        result = engine.run(start_date, end_date)
    print_report(result)


if __name__ == "__main__":
    main()
