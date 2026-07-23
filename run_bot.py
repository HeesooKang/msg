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
        max_position_count=3,             # 포지션 수가 아니라 총노출/총손절예산으로 제어
        bull_max_position_count=3,        # 강세장: 목표형 단일 집중 또는 상위 후보 분산
        neutral_max_position_count=2,     # 중립장: 검증된 후보 2개까지
        soft_bear_max_position_count=0,   # 완만약세(점수 2): 롱 기본 비활성화
        bear_max_position_count=0,        # 강한 약세(점수 3): 롱 비활성화
        per_stock_amount=900_000,         # 목표형 후보는 시드 대부분으로 일 +1만원을 노림
        max_per_stock_amount=1_000_000,   # 100만원 시드 기준 단일 포지션 상한
        capital_utilization_pct=0.98,     # 총노출은 별도 예산 캡으로 관리
        bull_capital_utilization_pct=1.00,      # 강세장: 100만원 안에서 분산 운용 가능
        neutral_capital_utilization_pct=0.95,   # 중립장도 검증된 후보는 목표형 집중 허용
        soft_bear_capital_utilization_pct=0.30, # 완만약세: 인버스 총노출 30%
        bear_capital_utilization_pct=0.20,      # 강한 약세: 총노출 20% (인버스 중심)
        max_single_position_pct=0.98,           # 목표형 단일 종목은 시드 대부분 허용
        bull_max_single_position_pct=0.98,      # 강세장: 상위 후보 1개 집중 가능
        neutral_max_single_position_pct=0.95,   # 중립장: 검증된 후보만 집중
        soft_bear_max_single_position_pct=0.20, # 완만약세: 단일 종목 최대 20%
        bear_max_single_position_pct=0.20,      # 강한 약세: 단일 종목 최대 20%
        enable_daily_target_position_sizing=True,
        daily_target_focus_min_seed_pct=0.90,
        daily_target_focus_max_seed_pct=0.98,
        daily_target_focus_min_leader_percentile=0.90,
        daily_target_focus_min_effective_score=0.82,
        daily_target_focus_min_recent_accel=0.02,
        daily_target_focus_min_volume_vs_avg=0.80,
        daily_target_focus_min_micro_net_score=0.08,
        daily_target_focus_min_precision_score=0.42,
        daily_target_focus_max_signal_conflict=0.70,
        daily_target_focus_min_entry_ev_krw=0.0,
        daily_target_focus_min_expected_net_ratio=0.55,
        daily_target_focus_partial_min_expected_net_ratio=0.30,
        daily_target_focus_partial_min_expected_net_krw=1_500,
        daily_target_focus_enable_incremental_accumulation=True,
        daily_target_focus_incremental_min_expected_net_krw=700,
        daily_target_focus_incremental_min_remaining_ratio=0.08,
        daily_target_focus_incremental_target_ratio=0.25,
        daily_target_focus_incremental_max_target_ratio=0.45,
        daily_target_focus_target_buffer_krw=300,
        daily_target_focus_min_take_profit_pct=0.65,
        daily_target_focus_max_take_profit_pct=2.40,
        daily_target_focus_stop_budget_share=0.75,
        daily_target_focus_all_market_min_leader_percentile=0.97,
        daily_target_focus_all_market_min_effective_score=1.00,
        daily_target_focus_all_market_min_recent_accel=0.08,
        daily_target_focus_all_market_min_volume_vs_avg=1.05,
        daily_target_focus_all_market_min_high_proximity=0.94,
        daily_target_focus_all_market_min_vs_open_pct=0.80,
        daily_target_focus_all_market_max_vs_open_pct=12.50,
        daily_target_focus_guarded_min_loss_room=3_000,
        enable_daily_target_fast_breakout_lane=True,
        daily_target_fast_breakout_min_leader_percentile=0.85,
        daily_target_fast_breakout_min_effective_score=0.70,
        daily_target_fast_breakout_min_recent_accel=0.20,
        daily_target_fast_breakout_min_volume_vs_avg=0.90,
        daily_target_fast_breakout_min_high_proximity=0.90,
        daily_target_fast_breakout_min_vs_open_pct=1.0,
        daily_target_fast_breakout_max_vs_open_pct=15.5,
        daily_target_fast_breakout_min_score=0.85,
        daily_target_fast_breakout_min_entry_ev_krw=0.0,
        daily_target_fast_breakout_negative_ev_floor_krw=-700.0,
        daily_target_fast_breakout_negative_ev_min_score=0.50,
        daily_target_fast_breakout_min_quality_score=0.68,
        daily_target_fast_breakout_negative_ev_min_quality_score=0.76,
        daily_target_fast_breakout_guarded_min_quality_score=0.78,
        daily_target_fast_breakout_min_micro_net_score=0.25,
        daily_target_fast_breakout_min_precision_score=0.52,
        daily_target_fast_breakout_max_signal_conflict=0.50,
        daily_target_fast_breakout_max_failed_breakout=0.45,
        daily_target_fast_breakout_max_downside_score=0.55,
        daily_target_fast_breakout_score_bonus=0.18,
        daily_target_fast_breakout_stop_budget_overrun_tolerance=1.20,
        daily_target_fast_breakout_stop_override_min_micro_net_score=0.35,
        daily_target_fast_breakout_stop_override_min_precision_score=0.52,
        daily_target_fast_breakout_stop_override_max_signal_conflict=0.45,
        daily_target_fast_breakout_stop_override_max_failed_breakout=0.40,
        daily_target_fast_breakout_stop_override_max_downside_score=0.45,
        daily_profit_target=10_000,       # 일일 목표 +1만원
        profit_protect_threshold=8_000,   # +8천원부터 신규 진입 축소
        daily_profit_lock_buffer=0,       # 목표 전 조기 잠금 대신 +1만원 실현 기준으로 종료
        daily_loss_limit=-5_000,          # 일일 하드스탑 -5천원
        daily_total_loss_limit=-5_000,    # 실현+미실현 합산 하드스탑 -5천원
        daily_loss_near_stop_buffer=250,  # 손실한도 근처에서는 재진입하지 않고 일중 종료
        enable_unrealized_loss_guard=True,
        daily_entry_loss_room_stop_ratio=0.90,
        daily_entry_min_loss_room=750,
        daily_entry_loss_room_reserve=500,  # 목표형 진입도 다음 판단을 위해 손실여력을 남긴다
        entry_stop_budget_share=0.75,     # 집중 진입이어도 1회 실패가 하루 손실한도를 거의 다 쓰지 않게 제한
        opening_entry_stop_budget_share=0.70,
        post_loss_entry_stop_budget_share=0.45,  # 손실 이후에는 더 강한 우위만 재진입
        recovery_entry_min_loss_room=800,  # 손실 후에도 이 정도 여력이 있으면 소량 복구 진입 허용
        recovery_entry_loss_room_stop_ratio=0.70,  # 복구 진입 손절예상액은 남은 손실여력의 70% 이내
        long_stop_loss_notional_pct=0.007,
        long_stop_loss_cap_amount=3_500,  # 100만원 집중 진입도 net 기준 손절이 일손실 한도 안에 오도록 제한
        protective_stop_fallback_ticks=1,
        take_profit_pct=1.8,              # 70만원 슬롯 기준 순익 +1만원 근접
        trailing_stop_pct=-0.55,          # 강세장 기본 추적손절
        min_volume=180_000,               # 거래량 하한
        min_price=5_000,                  # 저가주 제외 강화
        dynamic_pool_ranking_fetch_count=30,
        dynamic_pool_quote_min_change_rate=0.8,
        enable_expected_net_filter=True,   # 기대순익/예상손실 기준 진입 필터 활성화
        min_expected_net_profit=500,
        min_expected_rr_ratio=1.0,
        expected_stop_loss_daily_budget_ratio=0.80,
        expected_stop_uses_net_cap=True,
        enable_price_prediction_gate=True,
        price_prediction_horizon_seconds=180,
        price_prediction_min_samples=5,
        price_prediction_min_return_pct=0.45,
        price_prediction_min_lower_bound_pct=-0.45,
        price_prediction_min_confidence=0.54,
        price_prediction_min_direction_score=0.58,
        price_prediction_min_net_pnl=400,
        price_prediction_stop_reward_ratio=0.30,
        price_prediction_daily_target_min_ratio=0.08,
        price_prediction_max_lower_bound_loss_ratio=0.78,
        price_prediction_max_reversal_risk_score=0.58,
        price_prediction_max_chase_risk_score=0.66,
        price_prediction_max_trap_risk_score=0.62,
        price_prediction_extended_vs_open_pct=6.0,
        price_prediction_extended_min_continuation_quality=0.48,
        price_prediction_extended_min_follow_through_score=0.46,
        enable_live_entry_ev_prediction_blend=True,
        enable_price_prediction_queue_scout_ev_override=True,
        price_prediction_queue_scout_ev_floor_krw=-3_500.0,
        price_prediction_queue_scout_deep_ev_floor_krw=-20_000.0,
        price_prediction_queue_scout_min_return_pct=0.65,
        price_prediction_queue_scout_min_lower_bound_pct=-0.03,
        price_prediction_queue_scout_min_confidence=0.58,
        price_prediction_queue_scout_min_direction_score=0.60,
        price_prediction_queue_scout_min_continuation_quality=0.54,
        price_prediction_queue_scout_min_follow_through_score=0.52,
        price_prediction_queue_scout_max_reversal_risk_score=0.48,
        price_prediction_queue_scout_max_chase_risk_score=0.56,
        price_prediction_queue_scout_max_trap_risk_score=0.50,
        price_prediction_queue_scout_min_leader_percentile=0.92,
        price_prediction_queue_scout_min_effective_score=0.50,
        price_prediction_queue_scout_min_recent_accel=0.08,
        price_prediction_queue_scout_min_micro_net_score=0.34,
        price_prediction_queue_scout_min_precision_score=0.48,
        enable_queue_scout_value_reclaim_model=True,
        queue_scout_value_reclaim_min_score=0.60,
        queue_scout_value_reclaim_min_leader_percentile=0.80,
        queue_scout_value_reclaim_min_effective_score=0.22,
        queue_scout_value_reclaim_min_volume_vs_avg=0.70,
        queue_scout_value_reclaim_min_recent_accel=-0.03,
        queue_scout_value_reclaim_min_discount_pct=0.40,
        queue_scout_value_reclaim_min_rebound_pct=0.70,
        queue_scout_value_reclaim_max_range_position=0.97,
        queue_scout_value_reclaim_min_headroom_pct=0.20,
        queue_scout_value_reclaim_min_high_proximity=0.88,
        queue_scout_value_reclaim_max_vs_open_pct=13.5,
        queue_scout_value_reclaim_fresh_max_vs_open_pct=10.5,
        queue_scout_value_reclaim_min_prediction_return_pct=0.42,
        queue_scout_value_reclaim_min_prediction_lower_pct=-0.16,
        queue_scout_value_reclaim_min_prediction_confidence=0.48,
        queue_scout_value_reclaim_min_direction_score=0.56,
        queue_scout_value_reclaim_min_continuation_quality=0.42,
        queue_scout_value_reclaim_min_follow_through_score=0.40,
        queue_scout_value_reclaim_max_risk_score=0.60,
        enable_overextension_chase_guard=True,
        overextension_guard_min_vs_open_pct=6.0,
        overextension_guard_min_day_range_position=0.88,
        overextension_guard_min_high_proximity=0.94,
        overextension_guard_min_discount_pct=0.22,
        overextension_guard_min_pullback_pct=0.22,
        overextension_guard_min_headroom_pct=0.20,
        overextension_guard_min_rebound_pct=0.08,
        overextension_guard_min_prediction_return_pct=1.10,
        overextension_guard_min_prediction_lower_pct=0.20,
        overextension_guard_min_prediction_confidence=0.68,
        overextension_guard_min_micro_net_score=0.55,
        overextension_guard_min_precision_score=0.62,
        overextension_guard_max_downside_score=0.45,
        overextension_guard_max_failed_breakout_score=0.40,
        strong_leader_min_change_rate=2.5,
        bull_partial_exit_ratio=0.50,
        allow_expensive_single_share_override=True,
        expensive_single_share_min_price=30_000,
        expensive_single_share_cap_multiplier=2.0,
        expensive_single_share_max_seed_pct=0.98,
        expensive_single_share_max_loss_budget_share=0.95,
        single_share_stop_budget_overrun_tolerance=1.55,
        allow_intraday_expensive_single_share_override=True,
        intraday_expensive_single_share_max_single_position_pct=0.98,
        opening_candidate_window_minutes=20,
        opening_candidate_fetch_count=60,
        opening_hot_top_n=8,
        opening_hot_percentile_floor=0.85,
        enable_opening_fast_lane=True,
        opening_fast_window_minutes=3,
        opening_fast_fetch_count=80,
        opening_fast_live_top_n=2,
        opening_fast_pool_warmup_seconds=2.0,
        opening_fast_pool_refresh_interval_seconds=20,
        opening_fast_min_change_rate=0.2,
        opening_fast_min_volume=30_000,
        opening_fast_quote_warmup_seconds=2.0,
        opening_fast_initial_quote_budget=14,
        opening_candidate_pool_refresh_interval_seconds=45,
        opening_candidate_min_change_rate=0.4,
        opening_candidate_min_volume=80_000,
        opening_candidate_initial_quote_budget=10,
        enable_opening_conviction_lane=True,
        opening_conviction_window_minutes=5,
        opening_conviction_live_top_n=3,
        opening_conviction_provisional_min_consecutive_ticks=1,
        opening_conviction_provisional_score_min=0.78,
        opening_conviction_provisional_percentile_min=0.96,
        opening_conviction_provisional_recent_accel_min=0.04,
        opening_conviction_provisional_hold_seconds=60,
        opening_conviction_initial_entry_scale=1.00,
        opening_conviction_max_single_position_pct=0.98,
        opening_conviction_max_total_exposure_pct=0.98,
        opening_conviction_soft_bear_max_single_position_pct=0.35,
        opening_conviction_min_effective_score=0.90,
        opening_conviction_min_vs_open_pct=0.80,
        opening_conviction_full_size_min_seconds_after_open=45,
        opening_conviction_require_micro_ready_for_full_size=True,
        opening_conviction_full_size_min_micro_net_score=0.05,
        opening_conviction_full_size_min_precision_score=0.34,
        opening_conviction_full_size_max_signal_conflict=0.72,
        opening_conviction_early_predictor_enabled=True,
        opening_conviction_early_predictor_min_score=0.68,
        opening_conviction_early_predictor_min_seconds_after_open=8,
        opening_conviction_early_predictor_min_micro_samples=4,
        opening_conviction_early_predictor_no_micro_min_score=0.92,
        opening_conviction_early_predictor_no_micro_min_recent_accel=0.35,
        opening_conviction_unconfirmed_max_single_position_pct=0.0,
        opening_conviction_unconfirmed_max_total_exposure_pct=0.0,
        enable_intraday_conviction_lane=True,
        intraday_conviction_live_top_n=3,
        intraday_conviction_min_effective_score=0.90,
        intraday_conviction_min_recent_accel=0.05,
        intraday_conviction_min_vs_open_pct=0.30,
        conviction_candidate_memory_seconds=180,
        intraday_queue_leader_provisional_top_n=2,
        intraday_queue_leader_provisional_score_min=0.74,
        intraday_queue_leader_provisional_percentile_min=0.90,
        intraday_queue_leader_provisional_recent_accel_min=0.01,
        intraday_queue_leader_provisional_effective_score_min=0.82,
        intraday_queue_leader_provisional_volume_vs_avg_min=0.75,
        intraday_queue_leader_provisional_min_vs_open_pct=0.15,
        intraday_conviction_capacity_extra_slots=0,  # 설정된 포지션/노출 한도 밖 추가 오버플로우는 금지
        enable_value_reentry_filter=True,
        opening_value_min_pullback_pct=0.10,
        opening_value_min_rebound_pct=0.05,
        opening_value_min_headroom_pct=0.15,
        opening_value_max_rebound_ratio=0.90,
        intraday_value_min_pullback_pct=0.14,
        intraday_value_min_rebound_pct=0.07,
        intraday_value_min_headroom_pct=0.18,
        intraday_value_max_rebound_ratio=0.88,
        intraday_value_relaxed_pullback_multiplier=0.55,
        intraday_value_relaxed_rebound_multiplier=0.50,
        intraday_value_relaxed_headroom_multiplier=0.50,
        intraday_value_relaxed_max_rebound_ratio=0.95,
        enable_all_weather_value_lane=True,
        all_weather_value_min_day_discount_pct=0.18,
        all_weather_value_min_rebound_pct=0.08,
        all_weather_value_max_day_range_position=0.82,
        all_weather_value_min_recent_accel=0.02,
        all_weather_value_min_effective_score=0.82,
        all_weather_value_min_leader_percentile=0.78,
        all_weather_value_min_volume_vs_avg=0.85,
        all_weather_value_min_vs_open_pct=0.15,
        all_weather_value_max_rebound_ratio=0.92,
        all_weather_value_score_bonus=0.08,
        intraday_queue_value_min_day_discount_pct=0.12,
        intraday_queue_value_min_rebound_pct=0.05,
        intraday_queue_value_max_day_range_position=0.95,
        intraday_queue_value_relaxed_discount_multiplier=0.70,
        intraday_queue_value_relaxed_rebound_multiplier=0.60,
        intraday_queue_value_relaxed_max_day_range_position=0.98,
        intraday_all_weather_negative_ev_soft_floor_krw=-450.0,
        intraday_all_weather_negative_ev_soft_max_trades=20,
        enable_intraday_queue_scout_lane=True,
        intraday_queue_scout_min_leader_percentile=0.86,
        intraday_queue_scout_min_effective_score=0.78,
        intraday_queue_scout_min_volume_vs_avg=0.70,
        intraday_queue_scout_min_recent_accel=-0.02,
        intraday_queue_scout_min_vs_open_pct=0.10,
        intraday_queue_scout_min_discount_pct=0.04,
        intraday_queue_scout_min_rebound_pct=0.02,
        intraday_queue_scout_max_day_range_position=0.995,
        intraday_queue_scout_max_vs_open_pct=16.0,
        post_full_exit_queue_scout_cooldown_seconds=90,
        intraday_queue_scout_negative_ev_floor_krw=-750.0,
        intraday_queue_scout_require_micro_ready=True,
        intraday_queue_scout_min_micro_samples=4,
        intraday_queue_scout_min_micro_net_score=0.08,
        intraday_queue_scout_min_precision_score=0.42,
        intraday_queue_scout_max_signal_conflict=0.58,
        intraday_queue_scout_max_failed_breakout=0.55,
        intraday_queue_scout_max_downside_score=0.68,
        enable_intraday_queue_scout_continuation_bypass=True,
        intraday_queue_scout_continuation_min_leader_percentile=0.92,
        intraday_queue_scout_continuation_min_effective_score=0.42,
        intraday_queue_scout_continuation_min_recent_accel=0.08,
        intraday_queue_scout_continuation_min_high_proximity=0.88,
        intraday_queue_scout_continuation_min_micro_net_score=0.32,
        intraday_queue_scout_continuation_min_precision_score=0.50,
        intraday_queue_scout_continuation_max_signal_conflict=0.45,
        intraday_queue_scout_continuation_max_failed_breakout=0.40,
        intraday_queue_scout_continuation_max_downside_score=0.45,
        intraday_queue_scout_continuation_max_vs_open_pct=15.5,
        enable_adaptive_market_thresholds=True,
        adaptive_market_min_quote_count=8,
        enable_broad_bull_live_edge_rescue=True,
        broad_bull_live_edge_negative_ev_floor_krw=-1_200.0,
        broad_bull_live_edge_min_leader_percentile=0.95,
        broad_bull_live_edge_min_effective_score=0.55,
        broad_bull_live_edge_min_recent_accel=0.04,
        broad_bull_live_edge_min_volume_vs_avg=0.85,
        broad_bull_live_edge_max_vs_open_pct=14.5,
        broad_bull_live_edge_min_discount_pct=0.10,
        broad_bull_live_edge_min_rebound_pct=0.04,
        broad_bull_live_edge_min_micro_net_score=0.28,
        broad_bull_live_edge_min_precision_score=0.45,
        broad_bull_live_edge_max_signal_conflict=0.55,
        broad_bull_live_edge_max_failed_breakout=0.50,
        enable_individual_reclaim_lane=True,
        individual_reclaim_negative_ev_floor_krw=-500.0,
        individual_reclaim_min_leader_percentile=0.88,
        individual_reclaim_min_effective_score=0.85,
        individual_reclaim_min_recent_accel=0.03,
        individual_reclaim_min_volume_vs_avg=0.75,
        individual_reclaim_min_vs_open_pct=0.35,
        individual_reclaim_max_vs_open_pct=12.5,
        individual_reclaim_min_micro_net_score=0.55,
        individual_reclaim_min_precision_score=0.62,
        individual_reclaim_max_signal_conflict=0.45,
        individual_reclaim_max_failed_breakout=0.40,
        individual_reclaim_min_rebound_pct=0.12,
        individual_reclaim_max_recent_drawdown_pct=0.65,
        enable_symbol_micro_edge=True,
        symbol_micro_edge_min_history=4,
        symbol_micro_edge_score_weight=0.18,
        symbol_micro_edge_downside_block_score=0.78,
        symbol_micro_edge_downside_block_net_max=-0.24,
        symbol_micro_edge_weak_launch_score=0.22,
        symbol_micro_failed_breakout_score=0.62,
        symbol_micro_precision_entry_floor=0.42,
        symbol_micro_signal_conflict_block_score=0.66,
        enable_intraday_continuation_filter=True,
        intraday_continuation_min_percentile=0.97,
        intraday_continuation_min_effective_score=1.0,
        intraday_continuation_min_recent_accel=0.08,
        intraday_continuation_min_volume_vs_avg=1.10,
        intraday_continuation_min_high_proximity=0.95,
        intraday_continuation_min_vs_open_pct=1.2,
        intraday_continuation_max_vs_open_pct=8.5,
        intraday_continuation_score_penalty=0.20,
        conviction_extension_vs_open_threshold_pct=6.0,
        conviction_extension_low_accel_threshold=0.22,
        conviction_extension_penalty_scale_krw=650.0,
        fragile_bull_max_long_positions=0,
        strategy_gate_window_days=5,
        strategy_gate_min_closed_trades=4,
        strategy_gate_path=os.path.join(PROJECT_ROOT, "reports", "strategy-gates.json"),
        forecast_outcome_root=os.path.join(PROJECT_ROOT, "reports", "forecast-outcomes"),
        enable_strategy_gate_enforcement=False,
        ev_window_days=5,
        conviction_ev_window_days=20,
        ev_scorecard_max_age_days=8,
        conviction_ev_scorecard_max_age_days=14,
        ev_min_samples=4,
        negative_ev_hard_block_min_trades=8,
        intraday_negative_ev_hard_block_min_trades=8,
        math_queue_top_n=12,
        math_queue_percentile_floor=0.80,
        math_queue_backfill_slots=6,
        math_gate_positive_ev_required=True,
        math_size_min_multiplier=0.70,
        math_size_max_multiplier=1.50,
        math_size_bull_a_max_multiplier=1.65,
        math_ev_scale_krw=2500,
        opening_confirmed_entry_ev_min_krw=250.0,
        intraday_confirmed_entry_ev_min_krw=120.0,
        negative_ev_symbol_cooldown_seconds=900,
        loss_symbol_cooldown_seconds=1800,
        structural_reject_symbol_cooldown_seconds=180,
        loss_trade_cooldown_seconds=900,  # 손실 체결 후 재진입 대기 확대
        trailing_stop_activation_gain_pct=0.8,
        max_position_holding_minutes=45,
        intraday_conviction_end_minutes_after_open=381,  # 15:21 전까지 강한 후보는 평가
        late_entry_guard_start_minutes_after_open=360,   # 15:00 이후는 강한 후보만 예외 통과
        late_entry_strong_min_ev_krw=800.0,
        late_entry_strong_min_score=0.95,
        late_entry_strong_min_leader_percentile=0.95,
        late_entry_strong_min_effective_score=0.60,
        late_entry_strong_min_recent_accel=0.02,
        late_entry_strong_min_volume_vs_avg=0.85,
        late_entry_strong_min_micro_net_score=0.10,
        late_entry_strong_min_precision_score=0.44,
        late_entry_strong_max_signal_conflict=0.70,
        late_entry_strong_max_failed_breakout=0.60,
        block_new_entry_windows=["15:21-15:30"],  # 장마감 직전 신규진입 차단
        enable_dynamic_entry_block_windows=False,  # 장마감 직전 차단은 레짐과 무관하게 유지
        dynamic_entry_block_disable_bear_score=2,
        enable_math_late_entry_override=True,      # 장후반에도 EV가 유의미하면 신규 진입 예외 허용
        late_entry_min_time_capture_ratio=0.30,
        use_restored_pnl_for_daily_breaker=False,  # 재시작 복구 손익은 하드스탑 기준에서 제외
        allow_hard_stop_bypass_for_day=os.getenv("ALLOW_DAILY_HARD_STOP_BYPASS", "false").strip().lower() in {"1", "true", "yes", "on"},
        commission_rate=0.00015,          # 수수료 0.015%
        tax_slippage_rate=0.002,         # 세금+슬리피지 0.20%
        entry_market_slippage_rate=0.001, # 매수 기대가격 상향 0.10%
        exit_market_slippage_rate=0.001,  # 매도 기대가격 하향 0.10%
        # 인버스 ETF는 별도 라우트 없이 inverse_etfs watchlist를 통해 일반 EV 후보로 평가한다.
        market_shock_window_minutes_after_open=45,
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
    strategy._runtime_config = config
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
        run_scheduled(strategy, config=getattr(strategy, "_runtime_config", None), tick_interval=10)
    finally:
        if caffeinate_proc:
            caffeinate_proc.terminate()


if __name__ == "__main__":
    main()
