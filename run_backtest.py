#!/usr/bin/env python3
"""백테스트 실행 스크립트."""

import sys
import os
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from src.config import Config
from src.auth import TokenManager
from src.api_client import KISClient
from src.market_data import MarketDataAPI
from src.backtest.data_fetcher import HistoricalDataFetcher
from src.backtest.engine import BacktestEngine
from src.backtest.report import print_report
from src.strategies.momentum_scalp import (
    MomentumScalpStrategy,
    MomentumScalpConfig,
    DEFAULT_STATIC_WATCHLIST,
    DEFAULT_INVERSE_ETFS,
)


def main():
    # 1. API 클라이언트 (데이터 다운로드용)
    config = Config.load()
    client = KISClient(config, TokenManager(config))
    market_data = MarketDataAPI(client)

    # 2. 백테스트 대상 종목 (시총 상위 30개 + 인버스 ETF)
    symbols = DEFAULT_STATIC_WATCHLIST + DEFAULT_INVERSE_ETFS

    # 3. 데이터 다운로드 (최근 2개월)
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=60)).strftime("%Y%m%d")

    fetcher = HistoricalDataFetcher(market_data)
    print(f"데이터 다운로드: {len(symbols)}종목 ({start_date}~{end_date})...")
    data = fetcher.fetch_pool(symbols, start_date, end_date)
    print(f"→ {len(data)}종목 로드 완료")

    # 4. 전략 설정 (백테스트 모드)
    strategy_config = MomentumScalpConfig(
        seed_money=1_000_000,
        max_position_count=5,
        per_stock_amount=200_000,
        daily_profit_target=10_000,
        daily_loss_limit=-5_000,
        per_position_stop_loss=-5_000,
        take_profit_pct=1.5,
        trailing_stop_pct=-0.7,
        static_watchlist=list(data.keys()),
        # 인버스 ETF 활성화
        inverse_enabled=True,
        inverse_max_positions=2,
        bearish_threshold=2,
    )

    strategy = MomentumScalpStrategy(
        market_data=None,
        config=strategy_config,
        pool_override=list(data.keys()),
    )

    # 5. 백테스트 실행 (수수료 0.015% + 세금/슬리피지 0.20%)
    engine = BacktestEngine(
        strategy=strategy,
        data=data,
        initial_capital=1_000_000,
        commission_rate=0.00015,
        tax_rate=0.002,
    )

    result = engine.run(start_date, end_date)

    # 6. 결과 출력
    print_report(result)


if __name__ == "__main__":
    main()
