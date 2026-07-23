"""momentum_scalp 전략의 설정/상태 타입 정의."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

# 시가총액 상위 30종목 + 한국시장 추종 ETF (하드코딩)
DEFAULT_STATIC_WATCHLIST = [
    "005930",  # 삼성전자
    "000660",  # SK하이닉스
    "373220",  # LG에너지솔루션
    "207940",  # 삼성바이오로직스
    "005490",  # POSCO홀딩스
    "006400",  # 삼성SDI
    "051910",  # LG화학
    "035420",  # NAVER
    "000270",  # 기아
    "005380",  # 현대차
    "035720",  # 카카오
    "105560",  # KB금융
    "055550",  # 신한지주
    "012330",  # 현대모비스
    "066570",  # LG전자
    "003670",  # 포스코퓨처엠
    "028260",  # 삼성물산
    "032830",  # 삼성생명
    "003550",  # LG
    "086790",  # 하나금융지주
    "034730",  # SK
    "015760",  # 한국전력
    "017670",  # SK텔레콤
    "009150",  # 삼성전기
    "010130",  # 고려아연
    "033780",  # KT&G
    "018260",  # 삼성에스디에스
    "011200",  # HMM
    "138930",  # BNK금융지주
    "024110",  # 기업은행
    "152100",  # PLUS 200 (KOSPI 200 추종)
    "292190",  # KODEX KRX300 (코스피·코스닥 전반 추종)
]

# 인버스 ETF 유니버스
DEFAULT_INVERSE_ETFS = [
    "114800",  # KODEX 인버스 (KOSPI 200 역방향)
    "123310",  # TIGER 인버스 (KOSPI 200 역방향)
    "251340",  # KODEX 코스닥150 인버스
    "464930",  # TIGER 2차전지TOP10 인버스
]

OPENING_STRATEGY = "opening_conviction_long_strategy"
INTRADAY_STRATEGY = "intraday_conviction_long_strategy"
LIVE_LONG_QUEUE_SOURCES = frozenset({"math_queue", "math_queue_live", "opening_hot_queue", "opening_fast_queue"})
OPENING_LONG_QUEUE_SOURCES = frozenset({"opening_fast_queue", "opening_hot_queue", "math_queue", "math_queue_live"})
INTRADAY_BASE_QUEUE_SOURCES = frozenset({"math_queue", "math_queue_live", "math_backfill"})


@dataclass
class MomentumScalpConfig:
    """모멘텀 스캘핑 전략 설정."""

    seed_money: int = 1_000_000
    max_position_count: int = 0
    bull_max_position_count: Optional[int] = None
    neutral_max_position_count: Optional[int] = None
    soft_bear_max_position_count: Optional[int] = None
    bear_max_position_count: Optional[int] = None
    per_stock_amount: int = 180_000
    max_per_stock_amount: int = 500_000
    capital_utilization_pct: float = 1.0
    bull_capital_utilization_pct: Optional[float] = None
    neutral_capital_utilization_pct: Optional[float] = None
    soft_bear_capital_utilization_pct: Optional[float] = None
    bear_capital_utilization_pct: Optional[float] = None
    max_single_position_pct: float = 1.0
    bull_max_single_position_pct: Optional[float] = None
    neutral_max_single_position_pct: Optional[float] = None
    soft_bear_max_single_position_pct: Optional[float] = None
    bear_max_single_position_pct: Optional[float] = None
    enable_daily_target_position_sizing: bool = False
    daily_target_focus_min_seed_pct: float = 0.90
    daily_target_focus_max_seed_pct: float = 0.98
    daily_target_focus_min_leader_percentile: float = 0.90
    daily_target_focus_min_effective_score: float = 0.82
    daily_target_focus_min_recent_accel: float = 0.02
    daily_target_focus_min_volume_vs_avg: float = 0.80
    daily_target_focus_min_micro_net_score: float = 0.08
    daily_target_focus_min_precision_score: float = 0.42
    daily_target_focus_max_signal_conflict: float = 0.70
    daily_target_focus_min_entry_ev_krw: float = 0.0
    daily_target_focus_min_expected_net_ratio: float = 0.80
    daily_target_focus_partial_min_expected_net_ratio: float = 0.35
    daily_target_focus_partial_min_expected_net_krw: int = 1_500
    daily_target_focus_enable_incremental_accumulation: bool = True
    daily_target_focus_incremental_min_expected_net_krw: int = 700
    daily_target_focus_incremental_min_remaining_ratio: float = 0.08
    daily_target_focus_incremental_target_ratio: float = 0.25
    daily_target_focus_incremental_max_target_ratio: float = 0.45
    daily_target_focus_target_buffer_krw: int = 700
    daily_target_focus_min_take_profit_pct: float = 0.65
    daily_target_focus_max_take_profit_pct: float = 2.20
    daily_target_focus_stop_budget_share: float = 0.95
    daily_target_focus_all_market_min_leader_percentile: float = 0.97
    daily_target_focus_all_market_min_effective_score: float = 1.00
    daily_target_focus_all_market_min_recent_accel: float = 0.08
    daily_target_focus_all_market_min_volume_vs_avg: float = 1.05
    daily_target_focus_all_market_min_high_proximity: float = 0.94
    daily_target_focus_all_market_min_vs_open_pct: float = 0.80
    daily_target_focus_all_market_max_vs_open_pct: float = 12.50
    daily_target_focus_guarded_min_loss_room: int = 3_000
    enable_daily_target_fast_breakout_lane: bool = False
    daily_target_fast_breakout_min_leader_percentile: float = 0.85
    daily_target_fast_breakout_min_effective_score: float = 0.30
    daily_target_fast_breakout_min_recent_accel: float = 0.20
    daily_target_fast_breakout_min_volume_vs_avg: float = 0.90
    daily_target_fast_breakout_min_high_proximity: float = 0.90
    daily_target_fast_breakout_min_vs_open_pct: float = 1.0
    daily_target_fast_breakout_max_vs_open_pct: float = 15.5
    daily_target_fast_breakout_min_score: float = 0.85
    daily_target_fast_breakout_min_entry_ev_krw: float = 0.0
    daily_target_fast_breakout_negative_ev_floor_krw: float = -700.0
    daily_target_fast_breakout_negative_ev_min_score: float = 0.50
    daily_target_fast_breakout_min_quality_score: float = 0.55
    daily_target_fast_breakout_negative_ev_min_quality_score: float = 0.66
    daily_target_fast_breakout_guarded_min_quality_score: float = 0.68
    daily_target_fast_breakout_min_micro_net_score: float = 0.05
    daily_target_fast_breakout_min_precision_score: float = 0.34
    daily_target_fast_breakout_max_signal_conflict: float = 0.72
    daily_target_fast_breakout_max_failed_breakout: float = 0.58
    daily_target_fast_breakout_max_downside_score: float = 0.70
    daily_target_fast_breakout_score_bonus: float = 0.18
    daily_target_fast_breakout_stop_budget_overrun_tolerance: float = 1.20
    daily_target_fast_breakout_stop_override_min_micro_net_score: float = 0.35
    daily_target_fast_breakout_stop_override_min_precision_score: float = 0.52
    daily_target_fast_breakout_stop_override_max_downside_score: float = 0.45
    daily_target_fast_breakout_stop_override_max_failed_breakout: float = 0.40
    daily_target_fast_breakout_stop_override_max_signal_conflict: float = 0.45
    daily_profit_target: int = 10_000
    daily_loss_limit: int = -5_000
    enable_unrealized_loss_guard: bool = False
    daily_total_loss_limit: Optional[int] = None
    daily_loss_near_stop_buffer: int = 250
    take_profit_pct: float = 2.5
    trailing_stop_pct: float = -1.1
    trailing_stop_activation_gain_pct: float = 1.0
    max_position_holding_minutes: int = 50
    allow_hard_stop_bypass_for_day: bool = False
    commission_rate: float = 0.00015
    tax_slippage_rate: float = 0.002
    entry_market_slippage_rate: float = 0.001
    exit_market_slippage_rate: float = 0.001
    static_watchlist: List[str] = field(default_factory=lambda: DEFAULT_STATIC_WATCHLIST)
    dynamic_pool_size: int = 15
    dynamic_pool_ranking_fetch_count: int = 30
    dynamic_pool_quote_min_change_rate: float = 0.8
    pool_refresh_interval: int = 300
    max_change_rate: float = 10.0
    min_volume: int = 180_000
    min_price: int = 2_000
    strong_leader_min_change_rate: float = 2.5
    bull_partial_exit_ratio: float = 0.5
    bull_risk_mode_loss_thresholds: Tuple[int, int, int] = (1, 2, 3)
    bull_risk_mode_profit_recovery_enabled: bool = True
    bull_risk_mode_profit_recovery_min_net: int = 1_200
    bull_risk_mode_stop_soften_enabled: bool = True
    bull_risk_mode_stop_soften_min_net_pnl: int = -2_500
    allow_expensive_single_share_override: bool = True
    expensive_single_share_min_price: int = 50_000
    expensive_single_share_cap_multiplier: float = 1.5
    expensive_single_share_max_seed_pct: float = 0.60
    expensive_single_share_max_loss_budget_share: float = 0.45
    single_share_stop_budget_overrun_tolerance: float = 1.75
    allow_intraday_expensive_single_share_override: bool = True
    intraday_expensive_single_share_max_single_position_pct: float = 0.35
    enable_expected_net_filter: bool = True
    min_expected_net_profit: int = 1_200
    min_expected_rr_ratio: float = 0.85
    expected_stop_loss_daily_budget_ratio: float = 0.90
    expected_stop_uses_net_cap: bool = False
    enable_price_prediction_gate: bool = False
    price_prediction_horizon_seconds: int = 180
    price_prediction_min_samples: int = 5
    price_prediction_min_return_pct: float = 0.45
    price_prediction_min_lower_bound_pct: float = -0.35
    price_prediction_min_confidence: float = 0.52
    price_prediction_min_direction_score: float = 0.56
    price_prediction_min_net_pnl: int = 500
    price_prediction_stop_reward_ratio: float = 0.35
    price_prediction_daily_target_min_ratio: float = 0.10
    price_prediction_max_lower_bound_loss_ratio: float = 0.70
    price_prediction_max_reversal_risk_score: float = 0.62
    price_prediction_max_chase_risk_score: float = 0.68
    price_prediction_max_trap_risk_score: float = 0.64
    price_prediction_extended_vs_open_pct: float = 6.0
    price_prediction_extended_min_continuation_quality: float = 0.45
    price_prediction_extended_min_follow_through_score: float = 0.48
    enable_live_entry_ev_prediction_blend: bool = False
    enable_price_prediction_queue_scout_ev_override: bool = True
    price_prediction_queue_scout_ev_floor_krw: float = -3_000.0
    price_prediction_queue_scout_deep_ev_floor_krw: float = -20_000.0
    price_prediction_queue_scout_min_return_pct: float = 0.80
    price_prediction_queue_scout_min_lower_bound_pct: float = 0.02
    price_prediction_queue_scout_min_confidence: float = 0.58
    price_prediction_queue_scout_min_direction_score: float = 0.62
    price_prediction_queue_scout_min_continuation_quality: float = 0.58
    price_prediction_queue_scout_min_follow_through_score: float = 0.56
    price_prediction_queue_scout_max_reversal_risk_score: float = 0.45
    price_prediction_queue_scout_max_chase_risk_score: float = 0.55
    price_prediction_queue_scout_max_trap_risk_score: float = 0.48
    price_prediction_queue_scout_min_leader_percentile: float = 0.92
    price_prediction_queue_scout_min_effective_score: float = 0.50
    price_prediction_queue_scout_min_recent_accel: float = 0.08
    price_prediction_queue_scout_min_micro_net_score: float = 0.42
    price_prediction_queue_scout_min_precision_score: float = 0.52
    enable_queue_scout_tape_ignition_override: bool = True
    queue_scout_tape_ignition_min_leader_percentile: float = 0.84
    queue_scout_tape_ignition_min_effective_score: float = 0.24
    queue_scout_tape_ignition_min_recent_accel: float = 1.20
    queue_scout_tape_ignition_min_micro_net_score: float = 0.68
    queue_scout_tape_ignition_min_precision_score: float = 0.70
    queue_scout_tape_ignition_max_downside_score: float = 0.20
    queue_scout_tape_ignition_max_failed_breakout_score: float = 0.22
    queue_scout_tape_ignition_max_signal_conflict_score: float = 0.12
    queue_scout_tape_ignition_min_return_pct: float = 0.95
    queue_scout_tape_ignition_min_positive_lower_pct: float = 0.35
    queue_scout_tape_ignition_min_positive_confidence: float = 0.82
    queue_scout_tape_ignition_impulse_min_return_pct: float = 1.80
    queue_scout_tape_ignition_impulse_min_lower_pct: float = -0.20
    queue_scout_tape_ignition_impulse_min_confidence: float = 0.50
    enable_queue_scout_value_reclaim_model: bool = True
    queue_scout_value_reclaim_min_score: float = 0.62
    queue_scout_value_reclaim_min_leader_percentile: float = 0.78
    queue_scout_value_reclaim_min_effective_score: float = 0.18
    queue_scout_value_reclaim_min_volume_vs_avg: float = 0.70
    queue_scout_value_reclaim_min_recent_accel: float = -0.05
    queue_scout_value_reclaim_min_discount_pct: float = 0.35
    queue_scout_value_reclaim_min_rebound_pct: float = 0.65
    queue_scout_value_reclaim_max_range_position: float = 0.97
    queue_scout_value_reclaim_min_headroom_pct: float = 0.18
    queue_scout_value_reclaim_min_high_proximity: float = 0.88
    queue_scout_value_reclaim_max_vs_open_pct: float = 13.5
    queue_scout_value_reclaim_fresh_max_vs_open_pct: float = 10.5
    queue_scout_value_reclaim_min_prediction_return_pct: float = 0.40
    queue_scout_value_reclaim_min_prediction_lower_pct: float = -0.18
    queue_scout_value_reclaim_min_prediction_confidence: float = 0.46
    queue_scout_value_reclaim_min_direction_score: float = 0.56
    queue_scout_value_reclaim_min_continuation_quality: float = 0.42
    queue_scout_value_reclaim_min_follow_through_score: float = 0.40
    queue_scout_value_reclaim_max_risk_score: float = 0.62
    enable_overextension_chase_guard: bool = False
    overextension_guard_min_vs_open_pct: float = 6.0
    overextension_guard_min_day_range_position: float = 0.88
    overextension_guard_min_high_proximity: float = 0.94
    overextension_guard_min_discount_pct: float = 0.20
    overextension_guard_min_pullback_pct: float = 0.20
    overextension_guard_min_headroom_pct: float = 0.20
    overextension_guard_min_rebound_pct: float = 0.08
    overextension_guard_min_prediction_return_pct: float = 1.10
    overextension_guard_min_prediction_lower_pct: float = 0.20
    overextension_guard_min_prediction_confidence: float = 0.68
    overextension_guard_min_micro_net_score: float = 0.55
    overextension_guard_min_precision_score: float = 0.62
    overextension_guard_max_downside_score: float = 0.45
    overextension_guard_max_failed_breakout_score: float = 0.40
    setup_recent_quote_window: int = 15
    strategy_gate_window_days: int = 5
    strategy_gate_min_closed_trades: int = 4
    strategy_gate_disable_expectancy_threshold: float = -150.0
    strategy_gate_path: str = "reports/strategy-gates.json"
    enable_strategy_gate_enforcement: bool = False
    ev_window_days: int = 5
    conviction_ev_window_days: int = 20
    ev_scorecard_max_age_days: int = 8
    conviction_ev_scorecard_max_age_days: int = 14
    ev_min_samples: int = 4
    math_live_ev_min_trades: int = 4
    math_live_negative_ev_threshold: float = 0.0
    negative_ev_hard_block_min_trades: int = 8
    intraday_negative_ev_hard_block_min_trades: int = 8
    math_queue_top_n: int = 12
    math_queue_percentile_floor: float = 0.80
    math_queue_backfill_slots: int = 6
    math_gate_positive_ev_required: bool = True
    math_size_min_multiplier: float = 0.70
    math_size_max_multiplier: float = 1.50
    math_size_bull_a_max_multiplier: float = 1.65
    math_ev_scale_krw: float = 2500.0
    opening_confirmed_entry_ev_min_krw: float = 250.0
    intraday_confirmed_entry_ev_min_krw: float = 120.0
    negative_ev_symbol_cooldown_seconds: int = 900
    loss_symbol_cooldown_seconds: int = 1800
    structural_reject_symbol_cooldown_seconds: int = 180
    profit_protect_threshold: int = 8_000
    daily_profit_lock_buffer: int = 1_500
    daily_entry_loss_room_stop_ratio: float = 0.90
    daily_entry_min_loss_room: int = 1_500
    daily_entry_loss_room_reserve: int = 2_000
    entry_stop_budget_share: float = 0.60
    opening_entry_stop_budget_share: float = 0.55
    post_loss_entry_stop_budget_share: float = 0.45
    recovery_entry_min_loss_room: int = 800
    recovery_entry_loss_room_stop_ratio: float = 0.70
    long_stop_loss_notional_pct: float = 0.007
    long_stop_loss_cap_amount: int = 2_500
    protective_stop_fallback_ticks: int = 1
    opening_candidate_window_minutes: int = 20
    opening_candidate_fetch_count: int = 60
    opening_hot_top_n: int = 8
    opening_hot_percentile_floor: float = 0.85
    enable_opening_fast_lane: bool = True
    opening_fast_window_minutes: int = 3
    opening_fast_fetch_count: int = 80
    opening_fast_live_top_n: int = 2
    opening_fast_pool_warmup_seconds: float = 2.0
    opening_fast_pool_refresh_interval_seconds: int = 20
    opening_fast_min_change_rate: float = 0.2
    opening_fast_min_volume: int = 30_000
    opening_fast_quote_warmup_seconds: float = 2.0
    opening_fast_initial_quote_budget: int = 14
    opening_candidate_pool_refresh_interval_seconds: int = 45
    opening_candidate_min_change_rate: float = 0.4
    opening_candidate_min_volume: int = 80_000
    opening_candidate_initial_quote_budget: int = 10
    enable_opening_conviction_lane: bool = True
    opening_conviction_window_minutes: int = 5
    opening_conviction_live_top_n: int = 2
    opening_conviction_provisional_min_consecutive_ticks: int = 1
    opening_conviction_provisional_score_min: float = 0.78
    opening_conviction_provisional_percentile_min: float = 0.96
    opening_conviction_provisional_recent_accel_min: float = 0.04
    opening_conviction_provisional_hold_seconds: int = 60
    opening_conviction_initial_entry_scale: float = 0.85
    opening_conviction_max_single_position_pct: float = 0.50
    opening_conviction_max_total_exposure_pct: float = 0.70
    opening_conviction_soft_bear_max_single_position_pct: float = 0.35
    opening_conviction_min_effective_score: float = 0.90
    opening_conviction_min_vs_open_pct: float = 0.80
    opening_conviction_full_size_min_seconds_after_open: int = 45
    opening_conviction_require_micro_ready_for_full_size: bool = True
    opening_conviction_full_size_min_micro_net_score: float = 0.05
    opening_conviction_full_size_min_precision_score: float = 0.34
    opening_conviction_full_size_max_signal_conflict: float = 0.72
    opening_conviction_early_predictor_enabled: bool = True
    opening_conviction_early_predictor_min_score: float = 0.68
    opening_conviction_early_predictor_min_seconds_after_open: int = 8
    opening_conviction_early_predictor_min_micro_samples: int = 4
    opening_conviction_early_predictor_no_micro_min_score: float = 0.92
    opening_conviction_early_predictor_no_micro_min_recent_accel: float = 0.35
    opening_conviction_unconfirmed_max_single_position_pct: float = 0.0
    opening_conviction_unconfirmed_max_total_exposure_pct: float = 0.0
    enable_intraday_conviction_lane: bool = True
    intraday_conviction_live_top_n: int = 2
    intraday_conviction_min_effective_score: float = 0.90
    intraday_conviction_min_recent_accel: float = 0.05
    intraday_conviction_min_vs_open_pct: float = 0.30
    conviction_candidate_memory_seconds: int = 180
    intraday_conviction_provisional_min_consecutive_ticks: int = 2
    intraday_conviction_provisional_score_min: float = 0.82
    intraday_conviction_provisional_percentile_min: float = 0.965
    intraday_conviction_provisional_recent_accel_min: float = 0.03
    intraday_conviction_provisional_hold_seconds: int = 45
    intraday_queue_leader_provisional_top_n: int = 2
    intraday_queue_leader_provisional_score_min: float = 0.74
    intraday_queue_leader_provisional_percentile_min: float = 0.90
    intraday_queue_leader_provisional_recent_accel_min: float = 0.01
    intraday_queue_leader_provisional_effective_score_min: float = 0.82
    intraday_queue_leader_provisional_volume_vs_avg_min: float = 0.75
    intraday_queue_leader_provisional_min_vs_open_pct: float = 0.15
    intraday_conviction_capacity_overflow_score_margin: float = 0.02
    intraday_conviction_capacity_extra_slots: int = 1
    enable_value_reentry_filter: bool = True
    opening_value_min_pullback_pct: float = 0.15
    opening_value_min_rebound_pct: float = 0.08
    opening_value_min_headroom_pct: float = 0.20
    opening_value_max_rebound_ratio: float = 0.88
    intraday_value_min_pullback_pct: float = 0.20
    intraday_value_min_rebound_pct: float = 0.10
    intraday_value_min_headroom_pct: float = 0.25
    intraday_value_max_rebound_ratio: float = 0.84
    intraday_value_relaxed_pullback_multiplier: float = 0.65
    intraday_value_relaxed_rebound_multiplier: float = 0.60
    intraday_value_relaxed_headroom_multiplier: float = 0.60
    intraday_value_relaxed_max_rebound_ratio: float = 0.92
    enable_all_weather_value_lane: bool = True
    all_weather_value_min_day_discount_pct: float = 0.18
    all_weather_value_min_rebound_pct: float = 0.08
    all_weather_value_max_day_range_position: float = 0.82
    all_weather_value_min_recent_accel: float = 0.02
    all_weather_value_min_effective_score: float = 0.82
    all_weather_value_min_leader_percentile: float = 0.78
    all_weather_value_min_volume_vs_avg: float = 0.85
    all_weather_value_min_vs_open_pct: float = 0.15
    all_weather_value_max_rebound_ratio: float = 0.92
    all_weather_value_score_bonus: float = 0.08
    intraday_queue_value_min_day_discount_pct: float = 0.12
    intraday_queue_value_min_rebound_pct: float = 0.05
    intraday_queue_value_max_day_range_position: float = 0.95
    intraday_queue_value_relaxed_discount_multiplier: float = 0.70
    intraday_queue_value_relaxed_rebound_multiplier: float = 0.60
    intraday_queue_value_relaxed_max_day_range_position: float = 0.98
    intraday_all_weather_negative_ev_soft_floor_krw: float = -450.0
    intraday_all_weather_negative_ev_soft_max_trades: int = 20
    enable_intraday_queue_scout_lane: bool = True
    intraday_queue_scout_min_leader_percentile: float = 0.86
    intraday_queue_scout_min_effective_score: float = 0.78
    intraday_queue_scout_min_volume_vs_avg: float = 0.70
    intraday_queue_scout_min_recent_accel: float = -0.02
    intraday_queue_scout_min_vs_open_pct: float = 0.10
    intraday_queue_scout_min_discount_pct: float = 0.04
    intraday_queue_scout_min_rebound_pct: float = 0.02
    intraday_queue_scout_max_day_range_position: float = 0.995
    intraday_queue_scout_max_vs_open_pct: float = 16.0
    post_full_exit_queue_scout_cooldown_seconds: int = 90
    intraday_queue_scout_negative_ev_floor_krw: float = -450.0
    intraday_queue_scout_require_micro_ready: bool = True
    intraday_queue_scout_min_micro_samples: int = 4
    intraday_queue_scout_min_micro_net_score: float = 0.08
    intraday_queue_scout_min_precision_score: float = 0.42
    intraday_queue_scout_max_signal_conflict: float = 0.58
    intraday_queue_scout_max_failed_breakout: float = 0.55
    intraday_queue_scout_max_downside_score: float = 0.68
    enable_intraday_queue_scout_continuation_bypass: bool = True
    intraday_queue_scout_continuation_min_leader_percentile: float = 0.92
    intraday_queue_scout_continuation_min_effective_score: float = 0.42
    intraday_queue_scout_continuation_min_recent_accel: float = 0.08
    intraday_queue_scout_continuation_min_high_proximity: float = 0.88
    intraday_queue_scout_continuation_min_micro_net_score: float = 0.32
    intraday_queue_scout_continuation_min_precision_score: float = 0.50
    intraday_queue_scout_continuation_max_signal_conflict: float = 0.45
    intraday_queue_scout_continuation_max_failed_breakout: float = 0.40
    intraday_queue_scout_continuation_max_downside_score: float = 0.45
    intraday_queue_scout_continuation_max_vs_open_pct: float = 15.5
    enable_adaptive_market_thresholds: bool = True
    adaptive_market_min_quote_count: int = 8
    enable_broad_bull_live_edge_rescue: bool = True
    broad_bull_live_edge_negative_ev_floor_krw: float = -900.0
    broad_bull_live_edge_min_leader_percentile: float = 0.95
    broad_bull_live_edge_min_effective_score: float = 0.55
    broad_bull_live_edge_min_recent_accel: float = 0.04
    broad_bull_live_edge_min_volume_vs_avg: float = 0.85
    broad_bull_live_edge_max_vs_open_pct: float = 14.5
    broad_bull_live_edge_min_discount_pct: float = 0.10
    broad_bull_live_edge_min_rebound_pct: float = 0.04
    broad_bull_live_edge_min_micro_net_score: float = 0.28
    broad_bull_live_edge_min_precision_score: float = 0.45
    broad_bull_live_edge_max_signal_conflict: float = 0.55
    broad_bull_live_edge_max_failed_breakout: float = 0.50
    enable_individual_reclaim_lane: bool = True
    individual_reclaim_negative_ev_floor_krw: float = -500.0
    individual_reclaim_min_leader_percentile: float = 0.88
    individual_reclaim_min_effective_score: float = 0.72
    individual_reclaim_min_recent_accel: float = 0.03
    individual_reclaim_min_volume_vs_avg: float = 0.75
    individual_reclaim_min_vs_open_pct: float = 0.35
    individual_reclaim_max_vs_open_pct: float = 12.5
    individual_reclaim_min_micro_net_score: float = 0.10
    individual_reclaim_min_precision_score: float = 0.40
    individual_reclaim_max_signal_conflict: float = 0.72
    individual_reclaim_max_failed_breakout: float = 0.56
    individual_reclaim_min_rebound_pct: float = 0.08
    individual_reclaim_max_recent_drawdown_pct: float = 0.85
    enable_symbol_micro_edge: bool = True
    symbol_micro_edge_min_history: int = 4
    symbol_micro_edge_score_weight: float = 0.18
    symbol_micro_edge_downside_block_score: float = 0.78
    symbol_micro_edge_downside_block_net_max: float = -0.24
    symbol_micro_edge_weak_launch_score: float = 0.22
    symbol_micro_failed_breakout_score: float = 0.62
    symbol_micro_precision_entry_floor: float = 0.42
    symbol_micro_signal_conflict_block_score: float = 0.66
    enable_intraday_continuation_filter: bool = True
    intraday_continuation_min_percentile: float = 0.97
    intraday_continuation_min_effective_score: float = 1.0
    intraday_continuation_min_recent_accel: float = 0.08
    intraday_continuation_min_volume_vs_avg: float = 1.10
    intraday_continuation_min_high_proximity: float = 0.95
    intraday_continuation_min_vs_open_pct: float = 1.2
    intraday_continuation_max_vs_open_pct: float = 8.5
    intraday_continuation_score_penalty: float = 0.20
    conviction_extension_vs_open_threshold_pct: float = 6.0
    conviction_extension_low_accel_threshold: float = 0.22
    conviction_extension_penalty_scale_krw: float = 650.0
    fragile_bull_max_long_positions: int = 0
    loss_trade_cooldown_seconds: int = 420
    inverse_etfs: List[str] = field(default_factory=lambda: DEFAULT_INVERSE_ETFS)
    market_shock_window_minutes_after_open: int = 45
    paper_position_exit_grace_seconds: int = 20
    restored_position_grace_seconds: int = 30
    intraday_conviction_end_minutes_after_open: int = 381
    late_entry_guard_start_minutes_after_open: int = 360
    late_entry_strong_min_ev_krw: float = 800.0
    late_entry_strong_min_score: float = 0.95
    late_entry_strong_min_leader_percentile: float = 0.95
    late_entry_strong_min_effective_score: float = 0.60
    late_entry_strong_min_recent_accel: float = 0.02
    late_entry_strong_min_volume_vs_avg: float = 0.85
    late_entry_strong_min_micro_net_score: float = 0.10
    late_entry_strong_min_precision_score: float = 0.44
    late_entry_strong_max_signal_conflict: float = 0.70
    late_entry_strong_max_failed_breakout: float = 0.60
    block_new_entry_windows: List[str] = field(default_factory=list)
    enable_dynamic_entry_block_windows: bool = True
    dynamic_entry_block_disable_bear_score: int = 2
    enable_math_late_entry_override: bool = True
    late_entry_min_time_capture_ratio: float = 0.55
    use_restored_pnl_for_daily_breaker: bool = False
    forecast_outcome_root: str = "reports/forecast-outcomes"
    daily_state_path: str = "state/momentum_scalp_daily_state.json"


@dataclass
class PositionState:
    """보유 포지션 상태."""

    symbol: str
    buy_price: int
    quantity: int
    invested_amount: int = 0
    buy_time: datetime = field(default_factory=datetime.now)
    high_since_buy: int = 0
    is_restored: bool = False
    restored_at: Optional[datetime] = None
    entry_strategy_name: str = ""
    entry_setup_name: str = ""
    entry_reason: str = ""
    regime_label: str = ""
    bear_score: int = 0
    planned_risk_stage: str = ""
    entry_grade: str = ""
    leader_score: float = 0.0
    leader_percentile: float = 0.0
    entry_grade_math: str = ""
    entry_ev: float = 0.0
    entry_ev_confidence: str = ""
    conviction_tier: str = ""
    bull_risk_mode: str = "normal"
    post_loss_admission_class: str = "general"
    candidate_class: str = ""
    execution_mode: str = "live"
    live_route: str = ""
    queue_source: str = ""
    size_multiplier: float = 1.0
    conviction_score: float = 0.0
    conviction_rank: int = 0
    bull_prob: float = 0.0
    neutral_prob: float = 0.0
    soft_bear_prob: float = 0.0
    bear_prob: float = 0.0
    partial_exit_done: bool = False
    shock_score: float = 0.0
    shock_confidence: float = 0.0
    adaptive_take_profit_pct: float = 0.0
    adaptive_stop_loss_pct: float = 0.0
    adaptive_trailing_activation_pct: float = 0.0
    adaptive_trailing_stop_pct: float = 0.0
    adaptive_max_hold_minutes: int = 0
    planned_target_net_pnl: int = 0
    planned_stop_net_loss_abs: int = 0
    planned_risk_net_loss_abs: int = 0
    entry_expected_net_pnl: float = 0.0
    entry_prediction_net_pnl: int = 0
    entry_prediction_lower_net_pnl: int = 0
    entry_prediction_upper_net_pnl: int = 0
    entry_prediction_win_probability: float = 0.0
    entry_signal_price: int = 0
    entry_prediction_return_pct: float = 0.0
    entry_prediction_lower_pct: float = 0.0
    entry_prediction_upper_pct: float = 0.0
    pending_exit_started_at: Optional[datetime] = None
    pending_exit_order_no: str = ""
    pending_exit_reference_price: int = 0
    pending_exit_quantity: int = 0
    pending_exit_reason: str = ""
    pending_exit_fill_mode: str = ""
    pending_entry_started_at: Optional[datetime] = None
    pending_entry_reference_price: int = 0
    pending_entry_fill_mode: str = ""
    trade_key: str = ""
    realized_gross_pnl_so_far: int = 0
    realized_net_pnl_so_far: int = 0
    realized_fees_paid_so_far: int = 0
    realized_taxes_paid_so_far: int = 0

    def __post_init__(self):
        if self.invested_amount <= 0:
            self.invested_amount = self.buy_price * self.quantity
        if self.high_since_buy == 0:
            self.high_since_buy = self.buy_price
        if not self.trade_key:
            timestamp = self.buy_time.isoformat(timespec="seconds")
            self.trade_key = f"{self.symbol}:{timestamp}"


@dataclass
class DailyPnL:
    """일일 손익 추적."""

    realized_gross_pnl: int = 0
    realized_net_pnl: int = 0
    fees_paid: int = 0
    taxes_paid: int = 0
    trade_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    breakeven_count: int = 0
    winning_net_pnl_sum: int = 0
    losing_net_pnl_sum: int = 0
    largest_win_net: int = 0
    largest_loss_net: int = 0
