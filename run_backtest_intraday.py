#!/usr/bin/env python3
"""1분봉 리플레이 백테스트 실행 스크립트.

권장 실행:
    ./dev py run_backtest_intraday.py
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from src.auth import TokenManager
from src.backtest.data_fetcher import HistoricalDataFetcher, MinuteHistoricalDataFetcher
from src.backtest.intraday_engine import IntradayBacktestEngine
from src.backtest.report import print_report
from src.config import Config
from src.api_client import KISClient
from src.market_data import MarketDataAPI
from src.strategies.momentum_scalp import (
    DEFAULT_INVERSE_ETFS,
    DEFAULT_STATIC_WATCHLIST,
    MomentumScalpConfig,
    MomentumScalpStrategy,
)


def build_strategy_config() -> MomentumScalpConfig:
    return MomentumScalpConfig(
        seed_money=1_000_000,
        max_position_count=2,
        bull_max_position_count=2,
        neutral_max_position_count=1,
        soft_bear_max_position_count=0,
        bear_max_position_count=0,
        per_stock_amount=250_000,
        max_per_stock_amount=450_000,
        capital_utilization_pct=0.75,
        bull_capital_utilization_pct=0.90,
        neutral_capital_utilization_pct=0.75,
        soft_bear_capital_utilization_pct=0.30,
        bear_capital_utilization_pct=0.45,
        max_single_position_pct=0.40,
        bull_max_single_position_pct=0.45,
        neutral_max_single_position_pct=0.40,
        soft_bear_max_single_position_pct=0.20,
        bear_max_single_position_pct=0.25,
        enable_pyramiding=False,
        daily_profit_target=10_000,
        profit_protect_threshold=8_000,
        stage1_loss_threshold=-3_000,
        daily_loss_limit=-5_000,
        daily_total_loss_limit=-5_000,
        enable_unrealized_loss_guard=True,
        loss_stage_exposure_scale=0.5,
        profit_protect_exposure_scale=0.6,
        per_position_stop_loss=-2_500,
        long_stop_loss_notional_pct=0.007,
        long_stop_loss_cap_amount=2_500,
        take_profit_pct=1.6,
        trailing_stop_pct=-0.55,
        min_momentum_score=3.2,
        min_change_rate=0.8,
        min_volume=180_000,
        min_price=2_000,
        enable_expected_net_filter=False,
        expected_move_pct=1.4,
        min_expected_net_profit=300,
        min_expected_rr_ratio=0.45,
        enable_volume_spike_filter=True,
        volume_spike_min_history=2,
        volume_spike_ratio=1.7,
        volume_spike_ratio_min=1.1,
        volume_spike_abs_min=3_500,
        bullish_min_change_rate=0.45,
        bullish_min_momentum_score=3.0,
        bullish_min_momentum_score_floor=3.0,
        bullish_volume_spike_ratio_adjustment=0.30,
        bullish_volume_spike_abs_min_ratio=0.6,
        enable_entry_confirmation=True,
        entry_confirmation_ticks=2,
        scale_in_confirmation_ticks=1,
        enable_pool_persistence_gate=False,
        bear_market_mode="A",
        min_bear_score_for_new_long=2,
        bear_market_entry_score=4.0,
        cooldown_seconds=900,
        loss_trade_cooldown_seconds=900,
        trailing_stop_activation_gain_pct=0.8,
        max_position_holding_minutes=45,
        enable_regime_adaptive=True,
        market_shock_window_minutes_after_open=45,
        block_new_entry_windows=["15:00-15:21"],
        enable_dynamic_entry_block_windows=False,
    )


def _resolve_recent_trade_dates(daily_data: dict[str, object], limit: int = 20) -> list[str]:
    all_dates = set()
    for df in daily_data.values():
        if "stck_bsop_date" not in df.columns:
            continue
        all_dates.update(df["stck_bsop_date"].tolist())
    return sorted(all_dates)[-limit:]


def _validate_intraday_backtest_mode(config: Config) -> Optional[str]:
    if config.is_paper:
        return (
            "paper 모드에서는 KIS 1분봉/일별분봉 조회를 공식 검증 소스로 사용할 수 없습니다. "
            "공식 검증은 ./dev py run_backtest.py 와 실거래 로그 리포트를 사용해주십시오."
        )
    return None


def main():
    config = Config.load()
    unsupported_reason = _validate_intraday_backtest_mode(config)
    if unsupported_reason:
        print(unsupported_reason)
        raise SystemExit(1)

    client = KISClient(config, TokenManager(config))
    market_data = MarketDataAPI(client)

    symbols = DEFAULT_STATIC_WATCHLIST + DEFAULT_INVERSE_ETFS
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=45)).strftime("%Y%m%d")

    daily_fetcher = HistoricalDataFetcher(market_data)
    minute_fetcher = MinuteHistoricalDataFetcher(market_data)

    print(f"거래일 계산용 일봉 다운로드: {len(symbols)}종목 ({start_date}~{end_date})...")
    daily_data = daily_fetcher.fetch_pool(symbols, start_date, end_date)
    trade_dates = _resolve_recent_trade_dates(daily_data, limit=20)
    if not trade_dates:
        print("최근 거래일을 계산하지 못했습니다.")
        return

    print(f"1분봉 다운로드: {len(symbols)}종목 ({trade_dates[0]}~{trade_dates[-1]}, {len(trade_dates)}거래일)...")
    minute_data = minute_fetcher.fetch_pool(symbols, trade_dates)
    print(f"→ {len(minute_data)}종목 1분봉 로드 완료")
    if not minute_data:
        print("1분봉 데이터가 없어 백테스트를 중단합니다.")
        return

    strategy = MomentumScalpStrategy(
        market_data=None,
        config=build_strategy_config(),
        pool_override=list(minute_data.keys()),
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
