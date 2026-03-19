#!/usr/bin/env python3
"""KIS 자동매매 봇 실행 스크립트.

사용법:
    직접 실행:   ./dev py run_bot.py
    launchd:     자동 실행됨
"""

import json
import subprocess
import sys
import os

# 프로젝트 루트로 이동
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
READINESS_REPORT_PATH = os.path.join(PROJECT_ROOT, "reports", "real-trade-readiness.json")
REAL_MONEY_STAGE_RULES = {
    1: {
        "capital_scale": 0.25,
        "daily_loss_limit": -1_250,
        "daily_total_loss_limit": -1_250,
        "profit_protect_threshold": 2_000,
        "daily_profit_target": 2_500,
    },
    2: {
        "capital_scale": 0.50,
        "daily_loss_limit": -2_500,
        "daily_total_loss_limit": -2_500,
        "profit_protect_threshold": 4_000,
        "daily_profit_target": 5_000,
    },
    3: {
        "capital_scale": 1.00,
        "daily_loss_limit": -5_000,
        "daily_total_loss_limit": -5_000,
        "profit_protect_threshold": 8_000,
        "daily_profit_target": 10_000,
    },
}

from src.config import Config
from src.logger_setup import setup_logger
from src.main import run_scheduled


def _load_allowed_real_money_stage() -> int:
    try:
        with open(READINESS_REPORT_PATH, encoding="utf-8") as fp:
            payload = json.load(fp)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "실계좌 모드를 시작하려면 paper gate 리포트가 먼저 필요합니다. "
            "reports/real-trade-readiness.json 파일이 없습니다."
        ) from exc

    promotion = payload.get("promotion", {})
    stage = int(promotion.get("current_stage_allowed", 0) or 0)
    if stage < 1:
        raise RuntimeError(
            "paper gate를 통과하기 전에는 실계좌 모드가 자동 보류됩니다. "
            "최근 reports/real-trade-readiness.json을 확인해주십시오."
        )
    return min(stage, 3)


def _apply_real_money_stage(strategy_config, stage: int) -> tuple[int, float]:
    rules = REAL_MONEY_STAGE_RULES[int(stage)]
    scale = float(rules["capital_scale"])

    strategy_config.seed_money = max(100_000, int(strategy_config.seed_money * scale))
    strategy_config.per_stock_amount = max(40_000, int(strategy_config.per_stock_amount * scale))
    strategy_config.max_per_stock_amount = max(
        strategy_config.per_stock_amount,
        int(strategy_config.max_per_stock_amount * scale),
    )
    strategy_config.long_stop_loss_cap_amount = max(700, int(strategy_config.long_stop_loss_cap_amount * scale))
    strategy_config.inverse_stop_loss_cap_amount = max(500, int(strategy_config.inverse_stop_loss_cap_amount * scale))
    strategy_config.per_position_stop_loss = -abs(strategy_config.long_stop_loss_cap_amount)
    strategy_config.daily_loss_limit = int(rules["daily_loss_limit"])
    strategy_config.daily_total_loss_limit = int(rules["daily_total_loss_limit"])
    strategy_config.profit_protect_threshold = int(rules["profit_protect_threshold"])
    strategy_config.daily_profit_target = int(rules["daily_profit_target"])
    return stage, scale


def get_strategy():
    """사용할 전략을 반환한다."""
    from src.auth import TokenManager
    from src.api_client import KISClient
    from src.market_data import MarketDataAPI
    from src.strategies.momentum_scalp import MomentumScalpStrategy, MomentumScalpConfig

    config = Config.load()
    client = KISClient(config, TokenManager(config))
    market_data = MarketDataAPI(client)

    # === 전략 설정 (여기서 수정) ===
    strategy_config = MomentumScalpConfig(
        seed_money=1_000_000,             # 시드 100만원
        max_position_count=2,             # 강세 최대 2개, 그 외 레짐별 제한
        bull_max_position_count=2,        # 강세장: 롱 최대 2종목
        neutral_max_position_count=1,     # 중립장: 롱 최대 1종목
        soft_bear_max_position_count=0,   # 완만약세(점수 2): 롱 기본 비활성화
        bear_max_position_count=0,        # 강한 약세(점수 3): 롱 비활성화
        per_stock_amount=200_000,         # 기본 슬롯 예산 20만원
        max_per_stock_amount=350_000,     # 단일 포지션 최대 35만원
        capital_utilization_pct=0.70,     # 기본 총노출 예산 70%
        bull_capital_utilization_pct=0.70,      # 강세장: 총노출 70%
        neutral_capital_utilization_pct=0.25,   # 중립장: 총노출 25%
        soft_bear_capital_utilization_pct=0.30, # 완만약세: 인버스 총노출 30%
        bear_capital_utilization_pct=0.20,      # 강한 약세: 총노출 20% (인버스 중심)
        max_single_position_pct=0.35,           # 기본 단일 종목 최대 35%
        bull_max_single_position_pct=0.35,      # 강세장: 단일 종목 최대 35%
        neutral_max_single_position_pct=0.15,   # 중립장: 단일 종목 최대 15%
        soft_bear_max_single_position_pct=0.20, # 완만약세: 단일 종목 최대 20%
        bear_max_single_position_pct=0.20,      # 강한 약세: 단일 종목 최대 20%
        enable_pyramiding=False,          # 장중 추격 추가매수 비활성화
        daily_profit_target=10_000,       # 일일 목표 +1만원
        profit_protect_threshold=8_000,   # +8천원부터 신규 진입 축소
        stage1_loss_threshold=-2_000,     # -2천원부터 손실 1단계
        daily_loss_limit=-5_000,          # 일일 하드스탑 -5천원
        daily_total_loss_limit=-5_000,    # 실현+미실현 합산 하드스탑 -5천원
        enable_unrealized_loss_guard=True,
        loss_stage_exposure_scale=0.5,    # 손실 1단계: 총노출 절반
        profit_protect_exposure_scale=0.6,  # 수익 보호 단계: 신규 노출 축소
        per_position_stop_loss=-2_500,    # 로그 표시용 기준 손절
        long_stop_loss_notional_pct=0.007,
        long_stop_loss_cap_amount=2_500,
        inverse_stop_loss_notional_pct=0.006,
        inverse_stop_loss_cap_amount=1_800,
        take_profit_pct=1.6,              # 강세장 기본 익절값
        trailing_stop_pct=-0.55,          # 강세장 기본 추적손절
        min_momentum_score=3.2,           # 기본 모멘텀 점수
        min_change_rate=0.8,              # 기본 등락률 하한
        min_volume=180_000,               # 거래량 하한
        min_price=5_000,                  # 저가주 제외 강화
        dynamic_pool_ranking_fetch_count=30,
        dynamic_pool_turnover_slots=6,
        dynamic_pool_quote_trade_amount_slots=4,
        dynamic_pool_direct_rank_slots=4,
        dynamic_pool_direct_turnover_slots=3,
        dynamic_pool_direct_quote_leader_slots=2,
        dynamic_pool_quote_min_change_rate=0.8,
        enable_expected_net_filter=False,  # 고정 기대순익 필터는 비활성화
        expected_move_pct=1.4,
        min_expected_net_profit=300,
        min_expected_rr_ratio=0.45,
        # 거래량 스파이크 강제
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
        bull_leader_top_n=4,
        bull_leader_relative_strength_pp=0.45,
        bull_partial_exit_ratio=0.50,
        bull_priority_turnover_rank_max=2,
        bull_priority_per_stock_amount_multiplier=3.0,
        bull_priority_max_per_stock_amount_multiplier=3.0,
        bull_priority_max_single_position_pct=0.65,
        bull_priority_effective_slots=1,
        bull_priority_initial_entry_scale=0.85,
        bull_breakout_late_entry_start_minutes_after_open=255,
        bull_breakout_late_entry_score_bonus=0.35,
        bull_breakout_late_entry_change_rate_bonus=0.20,
        bull_breakout_initial_entry_scale=0.65,
        bull_post_loss_score_bonus=0.30,
        bull_post_loss_change_rate_bonus=0.20,
        bull_post_loss_breakout_buffer_bonus_pct=0.05,
        allow_expensive_single_share_override=True,
        expensive_single_share_min_price=50_000,
        expensive_single_share_cap_multiplier=1.5,
        # 모멘텀 진입 보강
        enable_entry_confirmation=True,       # 연속 재확인 후 진입
        entry_confirmation_ticks=2,
        scale_in_confirmation_ticks=1,
        bullish_fast_entry_score_bonus=0.9,
        bullish_fast_entry_change_rate_bonus=0.6,
        entry_confirmation_window_seconds=240,
        entry_confirmation_min_score_tolerance=0.4,
        entry_confirmation_max_pullback_pct=-0.6,   # 최초 대비 -0.6% 허용
        enable_pullback_entry_filter=False,
        pullback_activation_change_rate=1.8,
        pullback_required_min_drop_pct=0.2,
        pullback_allowed_max_drop_pct=1.4,
        pullback_min_vs_open_pct=0.25,
        overheated_jump_change_pct=3.5,
        overheated_retrace_ratio=0.9,
        # 동적 풀 지속성 게이트
        enable_pool_persistence_gate=True,
        momentum_pool_persistence_window=3,
        momentum_pool_min_appearances=2,
        neutral_pullback_min_drop_pct=0.45,
        neutral_pullback_max_drop_pct=1.8,
        neutral_pullback_min_ticks=1,
        neutral_min_runup_from_open_pct=1.1,
        neutral_reclaim_buffer_pct=0.03,        # 이전 고점 0.03% 상향 재돌파면 통과
        neutral_chase_block_proximity_pct=0.20,
        neutral_entry_start_minutes_after_open=60,
        neutral_entry_confirmation_ticks=2,
        neutral_max_losses_per_day=2,
        neutral_post_loss_cooldown_minutes=15,
        neutral_post_loss_reentry_limit=1,
        neutral_post_loss_min_drop_bonus_pct=0.15,
        neutral_post_loss_min_runup_bonus_pct=0.25,
        neutral_post_loss_reclaim_buffer_bonus_pct=0.02,
        neutral_post_loss_score_bonus=0.20,
        neutral_post_loss_change_rate_bonus=0.10,
        neutral_post_loss_extra_pullback_ticks=0,
        neutral_leader_top_n=8,
        neutral_leader_relative_strength_pp=0.5,
        neutral_leader_max_reclaim_ticks=6,
        neutral_first_entry_score_bonus=0.45,
        neutral_first_entry_change_rate_bonus=0.20,
        neutral_first_entry_min_drop_bonus_pct=0.15,
        neutral_first_entry_min_runup_bonus_pct=0.35,
        neutral_first_entry_reclaim_buffer_bonus_pct=0.02,
        neutral_first_entry_max_turnover_rank=4,
        neutral_first_entry_max_reclaim_ticks=3,
        neutral_strategy_cooldown_minutes=10,
        soft_bear_inverse_min_runup_pct=0.6,
        soft_bear_inverse_min_drop_pct=0.15,
        soft_bear_inverse_max_drop_pct=0.8,
        soft_bear_inverse_reclaim_buffer_pct=0.03,
        soft_bear_inverse_min_change_rate=0.9,
        soft_bear_inverse_min_momentum=2.2,
        soft_bear_strategy_cooldown_minutes=8,
        enable_soft_bear_strong_leader_longs=True,
        soft_bear_strong_leader_max_positions=1,
        soft_bear_strong_leader_min_change_rate=3.2,
        soft_bear_strong_leader_min_momentum=2.5,
        soft_bear_strong_leader_min_trade_amount=1_500_000_000,
        stage1_neutral_score_bonus=0.60,
        stage1_neutral_change_rate_bonus=0.25,
        stage1_neutral_min_drop_bonus_pct=0.10,
        stage1_neutral_min_runup_bonus_pct=0.25,
        stage1_neutral_reclaim_buffer_bonus_pct=0.02,
        stage1_neutral_max_turnover_rank=4,
        stage1_neutral_max_reclaim_ticks=3,
        strategy_gate_window_days=5,
        strategy_gate_min_closed_trades=4,
        strategy_gate_path=os.path.join(PROJECT_ROOT, "reports", "strategy-gates.json"),
        enable_math_shadow_layer=True,
        enable_math_live_layer=True,
        quote_tape_enabled=True,
        quote_tape_root=os.path.join(PROJECT_ROOT, "data", "intraday_tape"),
        ev_window_days=5,
        ev_min_samples=4,
        bear_market_mode='A',             # 점수 2는 제한적 대응, 점수 3부터는 예외 통과형
        min_bear_score_for_new_long=2,
        bear_market_entry_score=4.0,      # 강한 약세장 예외 진입은 더 강한 모멘텀만 허용
        cooldown_seconds=900,             # 매도 후 재매수 쿨다운(레짐별 보정)
        loss_trade_cooldown_seconds=900,  # 손실 체결 후 재진입 대기 확대
        trailing_stop_activation_gain_pct=0.8,
        bullish_trailing_stop_activation_gain_pct_floor=1.1,
        max_position_holding_minutes=45,
        block_new_entry_windows=["15:00-15:21"],  # 장마감 직전만 신규진입 차단
        enable_dynamic_entry_block_windows=True,   # 약세장에서는 차단 시간대 자동 해제
        dynamic_entry_block_disable_bear_score=2,  # 약세점수 2 이상이면 차단 해제
        use_restored_pnl_for_daily_breaker=False,  # 재시작 복구 손익은 하드스탑 기준에서 제외
        allow_hard_stop_bypass_for_day=os.getenv("ALLOW_DAILY_HARD_STOP_BYPASS", "false").strip().lower() in {"1", "true", "yes", "on"},
        commission_rate=0.00015,          # 수수료 0.015%
        tax_slippage_rate=0.002,         # 세금+슬리피지 0.20%
        entry_market_slippage_rate=0.001, # 매수 기대가격 상향 0.10%
        exit_market_slippage_rate=0.001,  # 매도 기대가격 하향 0.10%
        enable_regime_adaptive=True,      # 시장 레짐 자동 전환(상승/약세/중립)
        # 인버스 ETF 설정
        inverse_enabled=True,             # 인버스 진입 활성화
        inverse_max_positions=1,          # 강한 약세 인버스 최대 1종목
        soft_bear_inverse_max_positions=1,  # 완만약세(점수 2): 인버스 최대 1종목
        bearish_threshold=2,              # 약세 점수 2 이상 시 진입
        inverse_min_bear_score=3,
        inverse_min_change_rate=2.5,
        inverse_min_momentum=4.5,
        inverse_take_profit_pct=0.9,
        inverse_trailing_stop_activation_gain_pct=0.5,
    )

    real_money_stage = 0
    capital_scale = 1.0
    if not config.is_paper:
        real_money_stage, capital_scale = _apply_real_money_stage(
            strategy_config,
            _load_allowed_real_money_stage(),
        )

    strategy = MomentumScalpStrategy(market_data, strategy_config)
    strategy._real_money_stage = real_money_stage
    strategy._capital_scale = capital_scale
    return strategy


def main():
    # macOS 절전 방지: caffeinate을 백그라운드로 실행
    caffeinate_proc = None
    try:
        caffeinate_proc = subprocess.Popen(
            ["caffeinate", "-dims"],  # display, idle, disk, system 절전 방지
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        pass  # caffeinate이 없는 환경 (Linux 등)

    try:
        strategy = get_strategy()
        run_scheduled(strategy, tick_interval=10)
    finally:
        if caffeinate_proc:
            caffeinate_proc.terminate()


if __name__ == "__main__":
    main()
