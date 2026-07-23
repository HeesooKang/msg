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

    # 일봉 smoke 백테스트에서는 외부 알림을 비활성화한다.
    os.environ["ALERTS_ENABLED"] = "false"

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
        neutral_capital_utilization_pct=0.20,
        soft_bear_capital_utilization_pct=0.30,
        bear_capital_utilization_pct=0.20,
        max_single_position_pct=0.35,
        bull_max_single_position_pct=0.35,
        neutral_max_single_position_pct=0.15,
        soft_bear_max_single_position_pct=0.20,
        bear_max_single_position_pct=0.20,
        enable_pyramiding=False,
        daily_profit_target=10_000,
        profit_protect_threshold=8_000,
        stage1_loss_threshold=-2_000,
        daily_loss_limit=-5_000,
        daily_total_loss_limit=-5_000,
        per_position_stop_loss=-2_500,
        long_stop_loss_notional_pct=0.007,
        long_stop_loss_cap_amount=2_500,
        take_profit_pct=1.6,
        trailing_stop_pct=-0.55,
        min_momentum_score=2.8,
        min_change_rate=0.6,
        min_volume=250_000,
        min_price=5_000,
        dynamic_pool_ranking_fetch_count=30,
        dynamic_pool_turnover_slots=6,
        dynamic_pool_quote_trade_amount_slots=4,
        dynamic_pool_quote_min_change_rate=0.8,
        enable_expected_net_filter=False,
        expected_move_pct=1.2,
        min_expected_net_profit=200,
        min_expected_rr_ratio=0.40,
        bull_bias_avg_change_rate_threshold=0.70,
        bull_bias_max_decliner_ratio=0.45,
        index_support_bull_bias_index_gap_pct=1.0,
        index_support_bull_bias_avg_change_rate_threshold=1.0,
        index_support_bull_bias_max_decliner_ratio=0.55,
        index_support_bull_bias_min_quote_count=8,
        strong_bull_override_index_gap_pct=1.5,
        strong_bull_override_avg_change_rate_threshold=2.0,
        strong_bull_override_max_decliner_ratio=0.25,
        strong_bull_override_min_quote_count=8,
        strong_leader_min_change_rate=2.5,
        strong_leader_min_trade_amount=1_000_000_000,
        strong_leader_top_rank=6,
        leader_support_bull_bias_min_count=1,
        leader_support_bull_bias_min_change_rate=4.0,
        leader_support_bull_bias_min_trade_amount=2_000_000_000,
        leader_support_bull_bias_max_decliner_ratio=0.70,
        bull_partial_exit_ratio=0.50,
        allow_expensive_single_share_override=True,
        expensive_single_share_min_price=50_000,
        expensive_single_share_cap_multiplier=1.5,
        bull_live_only_priority_sources=("opening_fast_queue", "opening_hot_queue", "math_queue", "math_queue_live"),
        conviction_low_ev_size_cap=1.00,
        broad_bull_avg_change_rate_threshold=2.0,
        broad_bull_max_decliner_ratio=0.25,
        fragile_bull_index_score_floor=2,
        # 일봉 OHLC 기반 백테스트에서는 장중 전용 필터를 완화한다.
        enable_volume_spike_filter=False,
        enable_entry_confirmation=False,
        enable_pool_persistence_gate=False,
        opening_candidate_window_minutes=20,
        opening_candidate_fetch_count=60,
        opening_hot_top_n=8,
        opening_hot_percentile_floor=0.85,
        enable_opening_fast_lane=True,
        opening_fast_window_minutes=3,
        opening_fast_fetch_count=80,
        opening_fast_live_top_n=2,
        enable_opening_conviction_lane=True,
        opening_conviction_window_minutes=5,
        opening_conviction_live_top_n=2,
        opening_conviction_initial_entry_scale=0.85,
        opening_conviction_max_single_position_pct=0.50,
        opening_conviction_max_total_exposure_pct=0.70,
        opening_conviction_soft_bear_max_single_position_pct=0.35,
        opening_conviction_min_effective_score=0.90,
        opening_conviction_min_recent_accel=0.10,
        opening_conviction_min_vs_open_pct=0.80,
        enable_intraday_conviction_lane=True,
        intraday_conviction_live_top_n=2,
        intraday_conviction_min_effective_score=0.90,
        intraday_conviction_min_recent_accel=0.05,
        intraday_conviction_min_vs_open_pct=0.30,
        conviction_extension_vs_open_threshold_pct=6.0,
        conviction_extension_low_accel_threshold=0.22,
        conviction_extension_penalty_scale_krw=650.0,
        fragile_bull_max_long_positions=0,
        enable_early_session_guard=False,
        strategy_gate_window_days=5,
        strategy_gate_min_closed_trades=4,
        enable_strategy_gate_enforcement=False,
        enable_math_shadow_layer=True,
        enable_math_live_layer=True,
        ev_window_days=5,
        conviction_ev_window_days=20,
        ev_min_samples=4,
        math_queue_top_n=12,
        math_queue_percentile_floor=0.80,
        math_queue_backfill_slots=6,
        math_gate_regime_margin=0.08,
        math_gate_min_leader_percentile=0.80,
        math_gate_positive_ev_required=True,
        math_size_min_multiplier=0.70,
        math_size_max_multiplier=1.50,
        math_size_bull_a_max_multiplier=1.65,
        math_ev_scale_krw=2500,
        protective_stop_hybrid_enabled=True,
        protective_stop_fallback_ticks=1,
        protective_stop_emergency_gap_multiple=1.5,
        enable_math_late_entry_override=True,
        late_entry_min_time_capture_ratio=0.55,
        static_watchlist=list(data.keys()),
        market_shock_window_minutes_after_open=45,
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
