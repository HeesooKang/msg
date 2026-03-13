#!/usr/bin/env python3
"""일봉 백테스트 실행 스크립트.

권장 실행:
    ./dev py run_backtest.py
    ./dev py run_backtest.py --days 40
"""

import argparse
import os
import sys
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="최근 N일 기준 일봉 백테스트 실행")
    parser.add_argument(
        "--days",
        type=int,
        default=40,
        help="최근 거래일 기준 백테스트 구간입니다. 기본값은 40일입니다.",
    )
    parser.add_argument(
        "--end-date",
        help="종료 기준 일자(YYYYMMDD). 미지정 시 전일을 사용합니다.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)

    # 1. API 클라이언트 (데이터 다운로드용)
    config = Config.load()
    client = KISClient(config, TokenManager(config))
    market_data = MarketDataAPI(client)

    # 2. 백테스트 대상 종목 (시총 상위 30개 + 인버스 ETF)
    symbols = DEFAULT_STATIC_WATCHLIST + DEFAULT_INVERSE_ETFS

    # 3. 데이터 다운로드
    # 거래일 40일은 대략 달력일 60일 수준이므로 여유 구간으로 내려받는다.
    end_dt = datetime.strptime(args.end_date, "%Y%m%d") if args.end_date else (datetime.now() - timedelta(days=1))
    calendar_lookback_days = max(1, int(args.days * 1.5))
    end_date = end_dt.strftime("%Y%m%d")
    start_date = (end_dt - timedelta(days=calendar_lookback_days)).strftime("%Y%m%d")

    fetcher = HistoricalDataFetcher(market_data)
    print(f"데이터 다운로드: {len(symbols)}종목 ({start_date}~{end_date})...")
    data = fetcher.fetch_pool(symbols, start_date, end_date)
    print(f"→ {len(data)}종목 로드 완료")

    # 4. 전략 설정 (백테스트 모드)
    strategy_config = MomentumScalpConfig(
        seed_money=1_000_000,
        max_position_count=2,
        bull_max_position_count=2,
        neutral_max_position_count=1,
        soft_bear_max_position_count=0,
        bear_max_position_count=0,
        per_stock_amount=200_000,
        max_per_stock_amount=350_000,
        capital_utilization_pct=0.70,
        bull_capital_utilization_pct=0.70,
        neutral_capital_utilization_pct=0.35,
        soft_bear_capital_utilization_pct=0.00,
        bear_capital_utilization_pct=0.20,
        max_single_position_pct=0.35,
        bull_max_single_position_pct=0.35,
        neutral_max_single_position_pct=0.25,
        soft_bear_max_single_position_pct=0.20,
        bear_max_single_position_pct=0.20,
        enable_pyramiding=False,
        daily_profit_target=10_000,
        profit_protect_threshold=8_000,
        stage1_loss_threshold=-3_000,
        daily_loss_limit=-5_000,
        daily_total_loss_limit=-5_000,
        per_position_stop_loss=-2_500,
        long_stop_loss_notional_pct=0.007,
        long_stop_loss_cap_amount=2_500,
        inverse_stop_loss_notional_pct=0.006,
        inverse_stop_loss_cap_amount=1_800,
        take_profit_pct=1.6,
        trailing_stop_pct=-0.55,
        min_momentum_score=2.8,
        min_change_rate=0.6,
        min_volume=250_000,
        min_price=5_000,
        enable_expected_net_filter=False,
        expected_move_pct=1.2,
        min_expected_net_profit=200,
        min_expected_rr_ratio=0.40,
        # 일봉 OHLC 기반 백테스트에서는 장중 전용 필터를 완화한다.
        enable_volume_spike_filter=False,
        enable_entry_confirmation=False,
        enable_pullback_entry_filter=False,
        enable_pool_persistence_gate=False,
        enable_early_session_guard=False,
        neutral_pullback_min_drop_pct=0.35,
        neutral_pullback_max_drop_pct=1.2,
        neutral_min_runup_from_open_pct=1.0,
        neutral_reclaim_buffer_pct=0.10,
        neutral_chase_block_proximity_pct=0.25,
        soft_bear_inverse_pullback_min_drop_pct=0.12,
        soft_bear_inverse_pullback_max_drop_pct=0.8,
        soft_bear_inverse_min_runup_from_open_pct=0.4,
        soft_bear_inverse_reclaim_buffer_pct=0.05,
        soft_bear_inverse_min_change_rate=0.9,
        soft_bear_inverse_min_momentum=2.2,
        static_watchlist=list(data.keys()),
        # 인버스 ETF 활성화
        inverse_enabled=True,
        inverse_max_positions=1,
        soft_bear_inverse_max_positions=0,
        bearish_threshold=2,
        inverse_min_momentum=4.5,
        inverse_min_change_rate=2.5,
        inverse_min_bear_score=3,
        inverse_take_profit_pct=0.9,
        inverse_trailing_stop_activation_gain_pct=0.5,
        cooldown_seconds=900,
        bear_market_mode='A',
        min_bear_score_for_new_long=2,
        bear_market_entry_score=3.8,
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
