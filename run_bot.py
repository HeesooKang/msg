#!/usr/bin/env python3
"""Launch the scheduled KIS expected-value trading bot."""

import json
import os
import subprocess
import sys


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

READINESS_REPORT_PATH = os.path.join(
    PROJECT_ROOT,
    "reports",
    "real-trade-readiness.json",
)
REAL_MONEY_STAGE_RULES = {
    1: {
        "capital_scale": 0.25,
        "daily_loss_limit": -1_250,
        "daily_profit_target": 2_500,
    },
    2: {
        "capital_scale": 0.50,
        "daily_loss_limit": -2_500,
        "daily_profit_target": 5_000,
    },
    3: {
        "capital_scale": 1.00,
        "daily_loss_limit": -5_000,
        "daily_profit_target": 10_000,
    },
}

from src.config import Config
from src.main import run_scheduled


def _load_allowed_real_money_stage() -> int:
    try:
        with open(READINESS_REPORT_PATH, encoding="utf-8") as file_pointer:
            payload = json.load(file_pointer)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "실계좌 모드를 시작하려면 reports/real-trade-readiness.json이 필요합니다."
        ) from exc
    stage = int((payload.get("promotion") or {}).get("current_stage_allowed", 0) or 0)
    if stage < 1:
        raise RuntimeError("paper gate를 통과하기 전에는 실계좌 모드를 시작할 수 없습니다.")
    return min(stage, 3)


def _apply_real_money_stage(strategy_config, stage: int) -> tuple[int, float]:
    rules = REAL_MONEY_STAGE_RULES[int(stage)]
    scale = float(rules["capital_scale"])
    strategy_config.seed_money = max(
        100_000,
        int(strategy_config.seed_money * scale),
    )
    strategy_config.daily_loss_limit = int(rules["daily_loss_limit"])
    strategy_config.daily_profit_target = int(rules["daily_profit_target"])
    return stage, scale


def get_strategy(runtime_config: Config):
    from src.api_client import KISClient
    from src.auth import TokenManager
    from src.market_data import MarketDataAPI
    from src.strategies.momentum_scalp import (
        MomentumScalpConfig,
        MomentumScalpStrategy,
    )

    client = KISClient(runtime_config, TokenManager(runtime_config))
    market_data = MarketDataAPI(client)
    if runtime_config.is_paper:
        commission_rate = 0.000142
        sell_tax_rate = 0.0015
    else:
        commission_rate = 0.00015
        sell_tax_rate = 0.002
    strategy_config = MomentumScalpConfig(
        seed_money=1_000_000,
        daily_profit_target=10_000,
        daily_loss_limit=-5_000,
        commission_rate=commission_rate,
        sell_tax_rate=sell_tax_rate,
        dynamic_pool_size=30,
        dynamic_pool_ranking_fetch_count=60,
        dynamic_pool_quote_min_change_rate=0.0,
        pool_refresh_interval=120,
        quote_freshness_seconds=5,
        forecast_outcome_root=os.path.join(
            PROJECT_ROOT,
            "reports",
            "forecast-outcomes",
        ),
        daily_state_path=os.path.join(
            PROJECT_ROOT,
            "state",
            "momentum_scalp_daily_state.json",
        ),
        state_save_interval_seconds=5,
        pending_order_block_seconds=180,
        allow_hard_stop_bypass_for_day=os.getenv(
            "ALLOW_DAILY_HARD_STOP_BYPASS",
            "false",
        ).strip().lower() in {"1", "true", "yes", "on"},
    )

    real_money_stage = 0
    capital_scale = 1.0
    if not runtime_config.is_paper:
        real_money_stage, capital_scale = _apply_real_money_stage(
            strategy_config,
            _load_allowed_real_money_stage(),
        )

    strategy = MomentumScalpStrategy(market_data, strategy_config)
    strategy._real_money_stage = real_money_stage
    strategy._capital_scale = capital_scale
    return strategy


def main() -> None:
    caffeinate_process = None
    try:
        caffeinate_process = subprocess.Popen(
            ["caffeinate", "-dims"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        pass

    try:
        runtime_config = Config.load()
        strategy = get_strategy(runtime_config)
        run_scheduled(strategy, tick_interval=1)
    finally:
        if caffeinate_process:
            caffeinate_process.terminate()


if __name__ == "__main__":
    main()
