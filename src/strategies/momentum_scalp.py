"""모멘텀 스캘핑 전략.

매수: 모멘텀 점수(시가대비 상승, 등락률, 고가근접도, 거래량폭발) 기반
      + 시장 레짐 필터(KOSPI MA20)
매도: 익절(+2.5%) / 개별 손절(금액 기준 -2,000원) / 추적손절(고점 -0.7%) /
     고점 이익 0.8% 이상에서만 추적손절 / 장마감 청산
관리: 일일 목표 도달(순실현손익 ≥ +12,000원) → 전량 청산 후 거래 중지
      일일 최대손실(순실현손익 ≤ -3,500원) → 전량 청산 후 거래 중지
      미실현 추정 손실 컷은 기본 비활성화. 필요 시 설정으로 별도 활성화
인버스: 약세 점수 ≥ 2일 때 인버스 ETF 매수 (공매도 효과)
"""

import logging
import json
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from statistics import median
from typing import Dict, List, Optional

import pandas as pd

from src.market_data import MarketDataAPI
from src.models import Order, OrderResult, OrderSide, OrderType, Position, Quote
from src.notifications import AlertManager
from src.strategy import BaseStrategy
from src.strategies.regime_router import RegimeStrategyRouter

logger = logging.getLogger("kis_trader.strategy.momentum")

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


@dataclass
class MomentumScalpConfig:
    """모멘텀 스캘핑 전략 설정."""

    seed_money: int = 1_000_000
    max_position_count: int = 0  # 0이면 seed_money/per_stock_amount 기준으로 자동 계산
    bull_max_position_count: Optional[int] = None
    neutral_max_position_count: Optional[int] = None
    soft_bear_max_position_count: Optional[int] = None
    bear_max_position_count: Optional[int] = None
    per_stock_amount: int = 180_000      # 종목당 기본 할당액
    max_per_stock_amount: int = 500_000  # 종목당 최대 노출 (피라미딩 상한)
    capital_utilization_pct: float = 1.0  # 총 노출 한도 비율 (기준자본 대비)
    bull_capital_utilization_pct: Optional[float] = None
    neutral_capital_utilization_pct: Optional[float] = None
    soft_bear_capital_utilization_pct: Optional[float] = None
    bear_capital_utilization_pct: Optional[float] = None
    max_single_position_pct: float = 1.0  # 단일 종목 최대 노출 비율 (기준자본 대비)
    bull_max_single_position_pct: Optional[float] = None
    neutral_max_single_position_pct: Optional[float] = None
    soft_bear_max_single_position_pct: Optional[float] = None
    bear_max_single_position_pct: Optional[float] = None
    enable_pyramiding: bool = True
    scale_in_min_profit_pct: float = 0.3
    scale_in_score_bonus: float = 0.8

    # 일일 서킷 브레이커 (순실현손익 기준)
    daily_profit_target: int = 10_000    # 일일 목표 +1만원
    daily_loss_limit: int = -3_500       # 일일 최대손실 -3.5천원
    enable_unrealized_loss_guard: bool = False  # 미실현 포함 보조 손실컷(기본 비활성화)
    daily_total_loss_limit: Optional[int] = None  # None이면 daily_loss_limit 사용

    # 개별 포지션 손절 (금액 기준)
    per_position_stop_loss: int = -1_800  # 포지션당 -1,800원 즉시 청산

    # 익절 / 추적손절 (비율 기준)
    take_profit_pct: float = 2.5         # 익절 +2.5%
    trailing_stop_pct: float = -1.1      # 고점 대비 추적손절
    trailing_stop_activation_gain_pct: float = 1.0  # 추적손절은 이익이 최소 1.0% 발생 후 동작
    max_position_holding_minutes: int = 50  # 시간 기반 보수적 청산 제한

    # 재시작 복구: 보유 종목이 한도에 걸린 상태에서의 초기 재진입 억제를 위한 설정
    startup_full_position_recheck_ticks: int = 2
    startup_market_data_ready_ticks: int = 2
    startup_market_data_min_valid_quote_count: int = 8
    startup_market_data_wait_log_interval_seconds: int = 60

    # 시장 레짐 필터 (KOSPI + 실시간 후보군 조합)
    bear_market_mode: str = 'A'          # 'A'=약세 필터 보완 적용, 'B'=완전 차단
    min_bear_score_for_new_long: int = 2  # A 모드에서 이 점수 이상이면 신규 롱 차단
    bear_market_entry_score: float = 3.8   # 약세장 예외로 허용할 최소 모멘텀 점수

    # 디버그/실험 모드: 당일 하드스탑 상태를 재시작 시 무시할지 여부
    # 실거래에서는 기본적으로 False(보수적)로 운영 권장
    allow_hard_stop_bypass_for_day: bool = False

    # 거래 비용
    commission_rate: float = 0.00015     # 0.015% 수수료
    tax_slippage_rate: float = 0.002     # 0.20% 세금+슬리피지 (매도 시)
    entry_market_slippage_rate: float = 0.001   # 매수 시 시장가 비딩 손실(과도한 급등 구간 완화용)
    exit_market_slippage_rate: float = 0.001    # 매도 시 시장가 비딩 손실(슬리피지 가정치)

    # 레짐 적응형 가중치/필터 적용
    enable_regime_adaptive: bool = True

    # 종목 풀
    static_watchlist: List[str] = field(default_factory=lambda: DEFAULT_STATIC_WATCHLIST)
    dynamic_pool_size: int = 15
    dynamic_pool_ranking_fetch_count: int = 30
    dynamic_pool_turnover_slots: int = 6
    dynamic_pool_quote_trade_amount_slots: int = 4
    dynamic_pool_direct_rank_slots: int = 4
    dynamic_pool_direct_turnover_slots: int = 3
    dynamic_pool_direct_quote_leader_slots: int = 2
    dynamic_pool_quote_min_change_rate: float = 0.8
    opening_market_relief_minutes: int = 45
    opening_dynamic_pool_min_change_rate: float = 0.2
    opening_dynamic_pool_min_volume: int = 30_000
    opening_dynamic_pool_direct_rank_slots: int = 5
    opening_dynamic_pool_direct_turnover_slots: int = 5
    opening_dynamic_pool_direct_quote_leader_slots: int = 4
    dynamic_pool_log_symbol_count: int = 6
    pool_refresh_interval: int = 300     # 초

    # 필터
    min_change_rate: float = 1.0
    max_change_rate: float = 10.0
    min_volume: int = 180_000
    min_price: int = 2_000

    enable_volume_spike_filter: bool = True
    volume_spike_min_history: int = 2
    volume_spike_ratio: float = 1.8
    volume_spike_abs_min: int = 4_000
    volume_spike_ratio_min: float = 1.2
    bullish_min_change_rate: float = 0.5
    bullish_min_momentum_score: float = 2.6
    bullish_min_momentum_score_floor: float = 3.4
    bullish_volume_spike_ratio_adjustment: float = 0.30
    bullish_volume_spike_abs_min_ratio: float = 0.6
    bull_bias_avg_change_rate_threshold: float = 0.8
    bull_bias_max_decliner_ratio: float = 0.45
    index_support_bull_bias_index_gap_pct: float = 1.0
    index_support_bull_bias_avg_change_rate_threshold: float = 1.0
    index_support_bull_bias_max_decliner_ratio: float = 0.55
    index_support_bull_bias_min_quote_count: int = 8
    strong_bull_override_index_gap_pct: float = 1.5
    strong_bull_override_avg_change_rate_threshold: float = 2.0
    strong_bull_override_max_decliner_ratio: float = 0.25
    strong_bull_override_min_quote_count: int = 8
    strong_leader_min_change_rate: float = 2.5
    strong_leader_min_trade_amount: int = 1_000_000_000
    strong_leader_top_rank: int = 6
    leader_support_bull_bias_min_count: int = 1
    leader_support_bull_bias_min_change_rate: float = 4.0
    leader_support_bull_bias_min_trade_amount: int = 2_000_000_000
    leader_support_bull_bias_max_decliner_ratio: float = 0.7
    opening_leader_bull_bias_min_count: int = 2
    opening_leader_bull_bias_change_rate: float = 2.5
    opening_leader_bull_bias_min_trade_amount: int = 1_000_000_000
    bull_leader_top_n: int = 5
    bull_leader_relative_strength_pp: float = 0.4
    bull_partial_exit_ratio: float = 0.5
    bull_priority_turnover_rank_max: int = 2
    bull_priority_per_stock_amount_multiplier: float = 3.0
    bull_priority_max_per_stock_amount_multiplier: float = 3.0
    bull_priority_max_single_position_pct: float = 0.65
    bull_priority_effective_slots: int = 1
    bull_priority_initial_entry_scale: float = 0.85
    bull_breakout_late_entry_start_minutes_after_open: int = 255
    bull_breakout_late_entry_score_bonus: float = 0.35
    bull_breakout_late_entry_change_rate_bonus: float = 0.2
    bull_breakout_initial_entry_scale: float = 0.65
    bull_post_loss_score_bonus: float = 0.30
    bull_post_loss_change_rate_bonus: float = 0.20
    bull_post_loss_breakout_buffer_bonus_pct: float = 0.05
    allow_expensive_single_share_override: bool = True
    expensive_single_share_min_price: int = 50_000
    expensive_single_share_cap_multiplier: float = 1.5

    min_momentum_score: float = 3.5
    enable_expected_net_filter: bool = True
    expected_move_pct: float = 2.4
    min_expected_net_profit: int = 1_200
    min_expected_rr_ratio: float = 0.85
    enable_cost_aware_profit_exit: bool = True
    min_profit_exit_net_pnl: int = 1
    enable_setup_logging: bool = True
    enable_shadow_blocked_candidate_tracking: bool = True
    shadow_blocked_candidate_window_minutes: int = 20
    setup_recent_quote_window: int = 8
    bull_breakout_hold_ticks: int = 2
    bull_breakout_buffer_pct: float = 0.03
    neutral_pullback_min_drop_pct: float = 0.25
    neutral_pullback_max_drop_pct: float = 1.2
    neutral_pullback_min_ticks: int = 2
    neutral_min_runup_from_open_pct: float = 0.8
    neutral_reclaim_buffer_pct: float = 0.05
    neutral_chase_block_proximity_pct: float = 0.10
    neutral_entry_start_minutes_after_open: int = 35
    neutral_entry_confirmation_ticks: int = 3
    neutral_max_losses_per_day: int = 1
    neutral_post_loss_cooldown_minutes: int = 30
    neutral_post_loss_reentry_limit: int = 1
    neutral_post_loss_min_drop_bonus_pct: float = 0.30
    neutral_post_loss_min_runup_bonus_pct: float = 0.50
    neutral_post_loss_reclaim_buffer_bonus_pct: float = 0.05
    neutral_post_loss_score_bonus: float = 0.35
    neutral_post_loss_change_rate_bonus: float = 0.15
    neutral_post_loss_extra_pullback_ticks: int = 1
    enable_neutral_leader_filter: bool = True
    neutral_leader_top_n: int = 8
    neutral_leader_relative_strength_pp: float = 0.5
    neutral_leader_max_reclaim_ticks: int = 6
    neutral_first_entry_score_bonus: float = 0.35
    neutral_first_entry_change_rate_bonus: float = 0.15
    neutral_first_entry_min_drop_bonus_pct: float = 0.15
    neutral_first_entry_min_runup_bonus_pct: float = 0.30
    neutral_first_entry_reclaim_buffer_bonus_pct: float = 0.02
    neutral_first_entry_max_turnover_rank: int = 4
    neutral_first_entry_max_reclaim_ticks: int = 3
    neutral_strategy_cooldown_minutes: int = 10
    soft_bear_inverse_pullback_min_drop_pct: float = 0.12
    soft_bear_inverse_pullback_max_drop_pct: float = 0.8
    soft_bear_inverse_min_runup_from_open_pct: float = 0.4
    soft_bear_inverse_reclaim_buffer_pct: float = 0.03
    soft_bear_inverse_min_change_rate: float = 0.9
    soft_bear_inverse_min_momentum: float = 2.2
    soft_bear_inverse_min_runup_pct: float = 0.6
    soft_bear_inverse_min_drop_pct: float = 0.15
    soft_bear_inverse_max_drop_pct: float = 0.8
    soft_bear_strategy_cooldown_minutes: int = 8
    enable_soft_bear_strong_leader_longs: bool = True
    soft_bear_strong_leader_max_positions: int = 1
    soft_bear_strong_leader_min_change_rate: float = 3.2
    soft_bear_strong_leader_min_momentum: float = 2.5
    soft_bear_strong_leader_min_trade_amount: int = 1_500_000_000
    stage1_neutral_score_bonus: float = 0.55
    stage1_neutral_change_rate_bonus: float = 0.20
    stage1_neutral_min_drop_bonus_pct: float = 0.10
    stage1_neutral_min_runup_bonus_pct: float = 0.20
    stage1_neutral_reclaim_buffer_bonus_pct: float = 0.02
    stage1_neutral_max_turnover_rank: int = 4
    stage1_neutral_max_reclaim_ticks: int = 3
    strategy_gate_window_days: int = 5
    strategy_gate_min_closed_trades: int = 4
    strategy_gate_path: str = "reports/strategy-gates.json"
    enable_backtest_score_entry_fallback: bool = False
    stage1_loss_threshold: int = -3_000
    profit_protect_threshold: int = 8_000
    loss_stage_exposure_scale: float = 0.5
    profit_protect_exposure_scale: float = 0.6
    long_stop_loss_notional_pct: float = 0.007
    long_stop_loss_cap_amount: int = 2_500
    inverse_stop_loss_notional_pct: float = 0.006
    inverse_stop_loss_cap_amount: int = 1_800
    stage1_inverse_score_bonus: float = 0.6
    stage1_inverse_change_bonus: float = 0.2
    # 진입 보강
    enable_entry_confirmation: bool = True          # 1차 후보 후 재확인 대기
    entry_confirmation_ticks: int = 2               # 신규 진입 최소 확인 틱 수
    scale_in_confirmation_ticks: int = 1            # 스케일인 최소 확인 틱 수
    bullish_fast_entry_score_bonus: float = 0.9
    bullish_fast_entry_change_rate_bonus: float = 0.6
    entry_confirmation_window_seconds: int = 240     # 확인 후보 유효 시간(초)
    entry_confirmation_min_score_tolerance: float = 0.4
    entry_confirmation_max_pullback_pct: float = -0.6
    # 눌림목(리테스트) 진입: 급등 종목은 고점 추격 대신 조정 구간에서만 신규 진입
    enable_pullback_entry_filter: bool = True
    pullback_activation_change_rate: float = 1.8
    pullback_required_min_drop_pct: float = 0.2
    pullback_allowed_max_drop_pct: float = 1.4
    pullback_min_vs_open_pct: float = 0.25
    overheated_jump_change_pct: float = 3.5
    overheated_retrace_ratio: float = 0.9
    enable_pool_persistence_gate: bool = True
    momentum_pool_persistence_window: int = 3
    momentum_pool_min_appearances: int = 2

    # 재매수 쿨다운
    cooldown_seconds: int = 900          # 15분
    loss_trade_cooldown_seconds: int = 420  # 손실 체결 후 전역 진입 일시 중지 시간

    # === 인버스 ETF 설정 ===
    inverse_enabled: bool = False
    inverse_etfs: List[str] = field(default_factory=lambda: DEFAULT_INVERSE_ETFS)
    inverse_max_positions: int = 2           # 인버스 최대 보유 수
    soft_bear_inverse_max_positions: Optional[int] = None
    inverse_take_profit_pct: float = 1.0     # 인버스 익절 +1.0% (일반보다 빠르게)
    inverse_stop_loss_pct: float = -0.5      # 인버스 손절 -0.5% (타이트)
    inverse_trailing_stop_pct: float = -0.3  # 인버스 추적손절 (고점 -0.3%)
    inverse_trailing_stop_activation_gain_pct: float = 0.45  # 추적손절은 이익이 최소 0.45% 발생 후 동작
    inverse_max_hold_minutes: int = 120      # 최대 2시간 보유 (음의 복리 방지)
    bearish_threshold: int = 2               # 이 점수 이상일 때 인버스 진입
    inverse_min_momentum: float = 2.0        # 인버스 매수 최소 모멘텀 점수
    inverse_min_change_rate: float = 1.4     # 인버스 매수 최소 등락률
    inverse_min_bear_score: int = 3          # 인버스 매수 최소 약세 점수
    inverse_volume_spike_ratio_offset: float = 0.45  # 인버스는 거래량 스파이크 비율 임계를 완화
    inverse_volume_spike_abs_min_ratio: float = 0.7  # 인버스는 최소 1틱 거래량 임계를 완화

    # 초반 구간 과도한 진입 억제
    enable_early_session_guard: bool = True
    early_session_guard_minutes: int = 12
    early_session_min_change_rate_boost: float = 0.20
    early_session_min_score_boost: float = 0.55
    early_session_entry_confirmation_ticks: int = 2
    bullish_trailing_stop_activation_gain_pct_floor: float = 1.1
    restored_position_grace_seconds: int = 30
    block_new_entry_windows: List[str] = field(default_factory=list)  # 예: ["11:00-12:00", "15:00-15:21"]
    enable_dynamic_entry_block_windows: bool = True
    dynamic_entry_block_disable_bear_score: int = 2  # 약세 점수가 이 값 이상이면 차단 시간대 자동 해제
    # 재시작 시 복구된 누적 실현손익을 당일 브레이커 기준에 포함할지 여부.
    # False면 재시작 이후 발생한 손익만으로 일일 목표/손실 브레이커를 판단한다.
    use_restored_pnl_for_daily_breaker: bool = False
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
    partial_exit_done: bool = False

    def __post_init__(self):
        if self.invested_amount <= 0:
            self.invested_amount = self.buy_price * self.quantity
        if self.high_since_buy == 0:
            self.high_since_buy = self.buy_price


@dataclass
class ShadowBlockedCandidate:
    """실제 진입은 하지 않지만 결과를 추적할 차단 후보."""

    symbol: str
    blocked_at: datetime
    reject_reason: str
    regime_label: str
    strategy_name: str
    entry_price: int
    hypothetical_quantity: int
    notional: int
    target_pct: float
    stop_loss_pct: float
    max_price: int
    min_price: int
    last_price: int
    first_hit_outcome: str = ""
    first_hit_at: Optional[datetime] = None


@dataclass
class MomentumEntrySignal:
    """진입 재확인 후보 상태."""

    streak: int
    first_price: int
    best_score: float
    started_at: datetime
    last_seen_at: datetime


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

    @property
    def realized_pnl(self) -> int:
        """하위 호환용 alias: 순실현손익."""
        return self.realized_net_pnl

    @property
    def total_pnl(self) -> int:
        return self.realized_net_pnl


class MomentumScalpStrategy(BaseStrategy):
    """모멘텀 스캘핑 전략."""

    def __init__(
        self,
        market_data: MarketDataAPI,
        config: MomentumScalpConfig = None,
        pool_override: List[str] = None,
    ):
        self.market_data = market_data
        self.cfg = config or MomentumScalpConfig()
        self._pool_override = pool_override

        self.positions: Dict[str, PositionState] = {}
        self.daily_pnl = DailyPnL()
        self._pool: List[str] = []
        self._last_pool_refresh: Optional[datetime] = None
        self._pool_build_epoch = 0
        self._pool_appearance: Dict[str, deque] = {}
        self._entry_signals: Dict[str, MomentumEntrySignal] = {}
        self._last_cumulative_volumes: Dict[str, int] = {}
        self._recent_tick_volumes: Dict[str, deque] = {}
        self._latest_tick_volumes: Dict[str, int] = {}
        self._halted = False
        self._avg_volumes: Dict[str, int] = {}
        self._quotes_cache: Dict[str, Quote] = {}
        self._recent_quotes: Dict[str, deque] = {}
        self._sell_cooldown: Dict[str, datetime] = {}
        self._symbol_cooldown_until: Dict[str, datetime] = {}
        self._strategy_cooldown_until: Dict[str, datetime] = {}
        self._bull_loss_count_today: int = 0
        self._bull_last_loss_at: Optional[datetime] = None
        self._startup_rebalance_ticks: int = 0
        self._startup_rebalance_active: bool = False
        self._market_data_ready_for_entries: bool = market_data is None
        self._market_data_ready_streak: int = 0
        self._market_data_readiness_reason: str = ""
        self._last_market_data_wait_log_at: Optional[datetime] = None
        self._bear_score: int = 0
        self._bear_market = False
        self._inverse_symbols: set = set(self.cfg.inverse_etfs)
        self._halt_date: Optional[date] = None
        self._current_day: Optional[date] = None
        self._alerts = AlertManager()
        self._last_regime_check_at: Optional[datetime] = None
        self._last_index_regime_check_at: Optional[datetime] = None
        self._cached_index_regime_score: int = 0
        self._cached_index_regime_info: Optional[tuple[float, float, float]] = None
        self._cached_index_regime_error: Optional[Exception] = None
        self._opening_leader_bull_bias_active: bool = False
        self._leader_support_bull_bias_active: bool = False
        self._index_support_bull_bias_active: bool = False
        self._strong_bull_override_active: bool = False
        self._regime_profile_name: str = "neutral"
        self._session_start_at: Optional[datetime] = None
        self._state_path = Path(self.cfg.daily_state_path) if self.cfg.daily_state_path else None
        # 백테스트/오프라인 실행에서는 로컬 상태파일 복구를 비활성화해 손익 오염을 방지한다.
        if self.market_data is None:
            self._state_path = None
        self._hard_stop_bypass_for_day: bool = False
        self._entry_block_windows = self._parse_entry_block_windows(self.cfg.block_new_entry_windows)
        self._last_entry_block_log_key: Optional[str] = None
        self._last_entry_block_bypass_log_key: Optional[str] = None
        self._entry_filter_log_cache: Dict[str, str] = {}
        self._state_loaded_for_today: bool = False
        self._daily_breaker_pnl_offset: int = 0
        self._loaded_position_meta: Dict[str, dict] = {}
        self._simulated_now: Optional[datetime] = None
        self._risk_stage_label: str = "normal"
        self._pending_entry_meta: Dict[str, dict] = {}
        self._neutral_loss_count_today: int = 0
        self._neutral_last_loss_at: Optional[datetime] = None
        self._neutral_post_loss_reentries_today: int = 0
        self._shadow_blocked_candidates: Dict[str, ShadowBlockedCandidate] = {}
        self._latest_direct_dynamic_symbols: set[str] = set()
        self._latest_strong_leader_symbols: set[str] = set()
        self._latest_strong_leader_snapshot: Dict[str, dict] = {}
        self._regime_router = RegimeStrategyRouter()
        self._strategy_gate_state: Dict[str, dict] = {}

    def set_simulated_now(self, now: Optional[datetime]):
        """백테스트에서 사용할 시뮬레이션 시각을 주입한다."""
        self._simulated_now = now

    def _now(self) -> datetime:
        return self._simulated_now or datetime.now()

    def _today(self) -> date:
        return self._now().date()

    def _record_recent_quote(self, quote: Quote):
        window = max(4, int(self.cfg.setup_recent_quote_window))
        history = self._recent_quotes.setdefault(quote.symbol, deque(maxlen=window))
        history.append(quote)

    def _get_recent_quotes(self, symbol: str) -> List[Quote]:
        return list(self._recent_quotes.get(symbol, ()))

    def _current_total_net_pnl(self) -> int:
        realized_net = self._effective_realized_net_for_breaker()
        unrealized_net = self._estimate_unrealized_net_pnl()
        return int(realized_net + unrealized_net)

    def _is_loss_stage_active(self, total_net: Optional[int] = None) -> bool:
        if total_net is None:
            total_net = self._current_total_net_pnl()
        return total_net <= int(self.cfg.stage1_loss_threshold)

    def _is_profit_protect_active(self, total_net: Optional[int] = None) -> bool:
        if total_net is None:
            total_net = self._current_total_net_pnl()
        return total_net >= int(self.cfg.profit_protect_threshold)

    def _current_risk_stage(self, total_net: Optional[int] = None) -> str:
        if total_net is None:
            total_net = self._current_total_net_pnl()
        if total_net <= int(self.cfg.daily_total_loss_limit or self.cfg.daily_loss_limit):
            return "hard_stop"
        if total_net >= int(self.cfg.daily_profit_target):
            return "profit_target"
        if self._is_loss_stage_active(total_net):
            return "loss_stage1"
        if self._is_profit_protect_active(total_net):
            return "profit_protect"
        return "normal"

    def _risk_exposure_scale(self, total_net: Optional[int] = None) -> float:
        scale = 1.0
        if self._is_loss_stage_active(total_net):
            scale = min(scale, float(self.cfg.loss_stage_exposure_scale))
        if self._is_profit_protect_active(total_net):
            scale = min(scale, float(self.cfg.profit_protect_exposure_scale))
        return max(0.1, scale)

    def _current_daily_total_loss_limit(self) -> int:
        return int(
            self._get_regime_value(
                "daily_total_loss_limit",
                self.cfg.daily_total_loss_limit
                if self.cfg.daily_total_loss_limit is not None
                else self.cfg.daily_loss_limit,
            )
        )

    def _current_daily_profit_target(self) -> int:
        return int(self._get_regime_value("daily_profit_target", self.cfg.daily_profit_target))

    def _trigger_daily_hard_stop(
        self,
        realized_net: int,
        unrealized_net: int,
        total_net: int,
        *,
        liquidate: bool,
    ) -> List[Order]:
        logger.warning(
            "일일 총손익 하드스탑 도달! (순실현: %s원, 미실현추정: %s원, 합계: %s원) → 전량 청산 후 거래 중지",
            f"{realized_net:,}",
            f"{unrealized_net:,}",
            f"{total_net:,}",
        )
        self._alerts.send(
            event_key="daily_total_loss_limit_hit",
            title="일일 총손익 하드스탑",
            message=(
                f"순실현 {realized_net:,}원, 미실현추정 {unrealized_net:,}원, "
                f"합계 {total_net:,}원으로 하드스탑에 도달했습니다."
            ),
            level="error",
            cooldown_seconds=1800,
        )
        self._halted = True
        self._halt_date = self._today()
        if liquidate:
            return self._liquidate_all()
        return []

    def _trigger_daily_profit_target(
        self,
        total_net: int,
        *,
        liquidate: bool,
    ) -> List[Order]:
        logger.info(
            "일일 총손익 목표 달성! (총손익: %s원) → 전량 청산 후 거래 중지",
            f"{total_net:,}",
        )
        self._alerts.send(
            event_key="daily_profit_target_hit",
            title="일일 목표 달성",
            message=f"총손익 {total_net:,}원으로 목표를 달성했습니다. 전량 청산 후 거래를 중지합니다.",
            level="info",
            cooldown_seconds=1800,
        )
        self._halted = True
        self._halt_date = self._today()
        if liquidate:
            return self._liquidate_all()
        return []

    def _evaluate_daily_breakers(self, *, liquidate: bool) -> Optional[List[Order]]:
        if self._hard_stop_bypass_for_day:
            return None

        realized_net = self._effective_realized_net_for_breaker()
        unrealized_net = self._estimate_unrealized_net_pnl()
        total_net = realized_net + unrealized_net

        if total_net <= self._current_daily_total_loss_limit():
            return self._trigger_daily_hard_stop(
                realized_net,
                unrealized_net,
                total_net,
                liquidate=liquidate,
            )
        if total_net >= self._current_daily_profit_target():
            return self._trigger_daily_profit_target(total_net, liquidate=liquidate)
        return None

    def _log_setup_reject(self, quote: Quote, reject_reason: str, message: str, *args):
        if not self.cfg.enable_setup_logging:
            return
        self._log_entry_filter_once_per_minute(
            quote.symbol,
            reject_reason,
            f"진입 거부[{reject_reason}] reject_reason={reject_reason}: " + message,
            *args,
        )

    def _track_shadow_blocked_candidate(self, quote: Quote, reject_reason: str) -> None:
        if not self.cfg.enable_shadow_blocked_candidate_tracking:
            return
        if reject_reason != "neutral_loss_limit_block":
            return
        if quote.current_price <= 0:
            return
        if quote.symbol in self._shadow_blocked_candidates:
            return

        allocation = self._compute_buy_allocation(
            symbol=quote.symbol,
            current_price=quote.current_price,
        )
        quantity = allocation // quote.current_price
        if quantity <= 0:
            quantity = 1
        notional = max(quote.current_price, quantity * quote.current_price)
        target_pct = float(self._build_regime_profile("neutral")["take_profit_pct"])
        stop_loss_pct = abs(self._long_stop_loss_amount_for_notional(notional)) / max(1, notional) * 100

        self._shadow_blocked_candidates[quote.symbol] = ShadowBlockedCandidate(
            symbol=quote.symbol,
            blocked_at=self._now(),
            reject_reason=reject_reason,
            regime_label=self._resolve_regime_profile_name(),
            strategy_name=self._current_profile_entry_strategy_name(is_inverse=False),
            entry_price=quote.current_price,
            hypothetical_quantity=quantity,
            notional=notional,
            target_pct=target_pct,
            stop_loss_pct=stop_loss_pct,
            max_price=quote.current_price,
            min_price=quote.current_price,
            last_price=quote.current_price,
        )
        logger.info(
            "그림자 후보 추적 시작: %s shadow_reason=%s regime_label=%s strategy_name=%s entry=%s원 qty=%d "
            "target=%.2f%% stop=%.2f%% window=%d분",
            quote.symbol,
            reject_reason,
            self._resolve_regime_profile_name(),
            self._current_profile_entry_strategy_name(is_inverse=False),
            f"{quote.current_price:,}",
            quantity,
            target_pct,
            stop_loss_pct,
            max(1, int(self.cfg.shadow_blocked_candidate_window_minutes)),
        )

    def _finalize_shadow_blocked_candidate(self, candidate: ShadowBlockedCandidate) -> None:
        mfe_pct = self._pct_move(candidate.entry_price, candidate.max_price)
        mae_pct = self._pct_move(candidate.entry_price, candidate.min_price)
        close_return_pct = self._pct_move(candidate.entry_price, candidate.last_price)

        if candidate.first_hit_outcome:
            outcome = candidate.first_hit_outcome
        elif close_return_pct > 0:
            outcome = "close_up"
        elif close_return_pct < 0:
            outcome = "close_down"
        else:
            outcome = "flat"

        logger.info(
            "그림자 후보 종료: %s shadow_reason=%s regime_label=%s strategy_name=%s entry=%s원 last=%s원 max=%s원 "
            "min=%s원 MFE=%.2f%% MAE=%.2f%% close=%.2f%% outcome=%s",
            candidate.symbol,
            candidate.reject_reason,
            candidate.regime_label,
            candidate.strategy_name,
            f"{candidate.entry_price:,}",
            f"{candidate.last_price:,}",
            f"{candidate.max_price:,}",
            f"{candidate.min_price:,}",
            mfe_pct,
            mae_pct,
            close_return_pct,
            outcome,
        )

    def _update_shadow_blocked_candidates(
        self,
        quotes: List[Quote],
        now: Optional[datetime] = None,
    ) -> None:
        if not self.cfg.enable_shadow_blocked_candidate_tracking:
            return
        if not self._shadow_blocked_candidates:
            return
        if now is None:
            now = self._now()

        quote_map = {quote.symbol: quote for quote in quotes}
        expiry = timedelta(minutes=max(1, int(self.cfg.shadow_blocked_candidate_window_minutes)))
        finalize_symbols: List[str] = []

        for symbol, candidate in self._shadow_blocked_candidates.items():
            quote = quote_map.get(symbol) or self._quotes_cache.get(symbol)
            if quote and quote.current_price > 0:
                candidate.last_price = quote.current_price
                candidate.max_price = max(candidate.max_price, quote.current_price)
                candidate.min_price = min(candidate.min_price, quote.current_price)
                move_pct = self._pct_move(candidate.entry_price, quote.current_price)
                if not candidate.first_hit_outcome:
                    if move_pct >= candidate.target_pct:
                        candidate.first_hit_outcome = "take_profit_first"
                        candidate.first_hit_at = now
                    elif move_pct <= -candidate.stop_loss_pct:
                        candidate.first_hit_outcome = "stop_loss_first"
                        candidate.first_hit_at = now

            if now >= candidate.blocked_at + expiry or (now.hour > 15 or (now.hour == 15 and now.minute >= 15)):
                self._finalize_shadow_blocked_candidate(candidate)
                finalize_symbols.append(symbol)

        for symbol in finalize_symbols:
            self._shadow_blocked_candidates.pop(symbol, None)

    def _log_risk_stage_change(self, stage_label: str, total_net: int):
        if stage_label == self._risk_stage_label:
            return
        self._risk_stage_label = stage_label
        logger.info("리스크 단계 전환: %s (총손익 %s원)", stage_label, f"{total_net:,}")

    @staticmethod
    def _pct_move(base_price: int, price: int) -> float:
        if base_price <= 0:
            return 0.0
        return (price - base_price) / base_price * 100

    @staticmethod
    def _loss_pct(peak_price: int, trough_price: int) -> float:
        if peak_price <= 0:
            return 0.0
        return (peak_price - trough_price) / peak_price * 100

    @staticmethod
    def _extract_context_token(payload: str, key: str, default: str = "") -> str:
        marker = f"{key}="
        for chunk in str(payload or "").split():
            if chunk.startswith(marker):
                return chunk[len(marker):]
        return default

    def _extract_context_int(self, payload: str, key: str, default: int = 0) -> int:
        raw = MomentumScalpStrategy._extract_context_token(payload, key, "")
        if not raw:
            return default
        try:
            return int(float(raw))
        except (TypeError, ValueError):
            return default

    def _build_entry_metadata(
        self,
        symbol: str,
        setup_name: str,
        payload: str,
        strategy_name: str = "",
    ) -> dict:
        current_stage = self._current_risk_stage()
        return {
            "strategy_name": strategy_name or self._extract_context_token(payload, "strategy_name", "unknown"),
            "setup_name": setup_name or self._extract_context_token(payload, "setup_name", "unknown"),
            "entry_reason": self._extract_context_token(payload, "entry_reason", setup_name or "unknown"),
            "regime_label": self._resolve_regime_profile_name(),
            "bear_score": int(self._bear_score),
            "planned_risk_stage": current_stage,
            "is_inverse": symbol in self._inverse_symbols,
            "entry_grade": self._extract_context_token(payload, "entry_grade", ""),
            "turnover_rank": self._extract_context_int(payload, "turnover_rank", 0),
        }

    def _append_entry_context(self, payload: str, metadata: dict) -> str:
        base = str(payload or "").strip()
        extra = (
            f"strategy_name={metadata.get('strategy_name', '')} "
            f"regime_label={metadata.get('regime_label', '')} "
            f"bear_score={int(metadata.get('bear_score', 0))} "
            f"planned_risk_stage={metadata.get('planned_risk_stage', '')}"
        )
        if metadata.get("entry_grade"):
            extra += f" entry_grade={metadata.get('entry_grade', '')}"
        if metadata.get("turnover_rank"):
            extra += f" turnover_rank={int(metadata.get('turnover_rank', 0))}"
        return f"{base} {extra}".strip()

    def _current_profile_entry_strategy_name(self, is_inverse: bool) -> str:
        profile_name = self._resolve_regime_profile_name()
        if not is_inverse and profile_name == "neutral" and self._is_bull_bias_market():
            return "bull_breakout_strategy"
        if not is_inverse and profile_name == "soft_bear" and self._soft_bear_strong_leader_lane_active():
            return "bull_breakout_strategy"
        strategy_name = self._regime_router.strategy_for_profile(profile_name).name
        if is_inverse:
            return strategy_name
        if profile_name in {"soft_bear", "bear"}:
            return ""
        return strategy_name

    def _load_strategy_gates(self) -> None:
        self._strategy_gate_state = {}
        if self.market_data is None:
            return
        gate_path_raw = str(getattr(self.cfg, "strategy_gate_path", "") or "").strip()
        if not gate_path_raw:
            return
        gate_path = Path(gate_path_raw)
        if not gate_path.exists():
            return
        try:
            payload = json.loads(gate_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("전략 자동 게이트 로드 실패(무시): %s", exc)
            return

        strategies = payload.get("strategies")
        if not isinstance(strategies, dict):
            return
        self._strategy_gate_state = {
            str(name): dict(meta)
            for name, meta in strategies.items()
            if isinstance(meta, dict)
        }
        disabled = [
            f"{name}(exp={float(meta.get('expectancy', 0.0)):.2f}, trades={int(meta.get('closed_trades', 0) or 0)})"
            for name, meta in sorted(self._strategy_gate_state.items())
            if not bool(meta.get("enabled", True))
        ]
        if disabled:
            logger.info("전략 자동 게이트 비활성화 적용: %s", ", ".join(disabled))

    def _is_strategy_gate_enabled(self, strategy_name: str) -> bool:
        if not strategy_name:
            return True
        gate = self._strategy_gate_state.get(strategy_name)
        if not gate:
            return True
        return bool(gate.get("enabled", True))

    def _symbol_cooldown_remaining(self, symbol: str, now: Optional[datetime] = None) -> Optional[datetime]:
        if now is None:
            now = self._now()
        cooldown_until = self._symbol_cooldown_until.get(symbol)
        if cooldown_until and now < cooldown_until:
            return cooldown_until
        if cooldown_until and now >= cooldown_until:
            self._symbol_cooldown_until.pop(symbol, None)
        return None

    def _strategy_cooldown_remaining(self, strategy_name: str, now: Optional[datetime] = None) -> Optional[datetime]:
        if now is None:
            now = self._now()
        if not strategy_name:
            return None
        cooldown_until = self._strategy_cooldown_until.get(strategy_name)
        if cooldown_until and now < cooldown_until:
            return cooldown_until
        if cooldown_until and now >= cooldown_until:
            self._strategy_cooldown_until.pop(strategy_name, None)
        return None

    def _apply_loss_cooldowns(self, symbol: str, strategy_name: str) -> None:
        now = self._now()
        self._symbol_cooldown_until[symbol] = now + timedelta(
            seconds=max(10, self._regime_loss_cooldown_seconds())
        )
        if strategy_name == "neutral_pullback_strategy":
            minutes = max(1, int(self.cfg.neutral_strategy_cooldown_minutes))
            if self._neutral_loss_count_today >= 2:
                minutes *= 2
            self._strategy_cooldown_until[strategy_name] = now + timedelta(minutes=minutes)
        elif strategy_name == "soft_bear_inverse_strategy":
            minutes = max(1, int(self.cfg.soft_bear_strategy_cooldown_minutes))
            self._strategy_cooldown_until[strategy_name] = now + timedelta(minutes=minutes)
        elif strategy_name == "bull_breakout_strategy":
            minutes = max(1, int(round(self._regime_loss_cooldown_seconds() / 60)))
            self._strategy_cooldown_until[strategy_name] = now + timedelta(minutes=minutes)

    def _minutes_since_market_open(self, now: Optional[datetime] = None) -> int:
        if now is None:
            now = self._now()
        return (now.hour * 60 + now.minute) - (9 * 60)

    def _is_neutral_entry_window_open(self, now: Optional[datetime] = None) -> bool:
        return self._minutes_since_market_open(now) >= int(
            self.cfg.neutral_entry_start_minutes_after_open
        )

    def _is_bull_late_entry_window(self, now: Optional[datetime] = None) -> bool:
        return self._minutes_since_market_open(now) >= int(
            self.cfg.bull_breakout_late_entry_start_minutes_after_open
        )

    def _position_notional(self, pos: PositionState) -> int:
        return max(0, int(pos.invested_amount or (pos.buy_price * pos.quantity)))

    def _neutral_loss_limit(self) -> int:
        return max(1, int(self.cfg.neutral_max_losses_per_day))

    def _neutral_post_loss_cooldown_until(self) -> Optional[datetime]:
        if self._neutral_last_loss_at is None:
            return None
        return self._neutral_last_loss_at + timedelta(
            minutes=max(0, int(self.cfg.neutral_post_loss_cooldown_minutes))
        )

    def _is_neutral_post_loss_retry_available(self, now: Optional[datetime] = None) -> bool:
        if now is None:
            now = self._now()
        if self._neutral_loss_count_today != self._neutral_loss_limit():
            return False
        if int(self.cfg.neutral_post_loss_reentry_limit) <= 0:
            return False
        if self._neutral_post_loss_reentries_today >= int(self.cfg.neutral_post_loss_reentry_limit):
            return False
        cooldown_until = self._neutral_post_loss_cooldown_until()
        if cooldown_until is None:
            return False
        return now >= cooldown_until

    def _neutral_retry_thresholds(self) -> dict:
        return {
            "min_drop_pct": self.cfg.neutral_pullback_min_drop_pct + self.cfg.neutral_post_loss_min_drop_bonus_pct,
            "min_runup_pct": (
                self.cfg.neutral_min_runup_from_open_pct + self.cfg.neutral_post_loss_min_runup_bonus_pct
            ),
            "reclaim_buffer_pct": (
                self.cfg.neutral_reclaim_buffer_pct + self.cfg.neutral_post_loss_reclaim_buffer_bonus_pct
            ),
            "min_score": self._regime_min_momentum_score() + self.cfg.neutral_post_loss_score_bonus,
            "min_change_rate": self._regime_min_change_rate() + self.cfg.neutral_post_loss_change_rate_bonus,
            "min_pullback_ticks": max(
                1,
                int(self.cfg.neutral_pullback_min_ticks) + int(self.cfg.neutral_post_loss_extra_pullback_ticks),
            ),
        }

    def _is_neutral_first_entry_attempt(self) -> bool:
        if self.daily_pnl.trade_count > 0:
            return False
        return not any(not pos.is_inverse for pos in self.positions.values())

    def _long_stop_loss_amount_for_notional(self, notional: int) -> int:
        dynamic_stop = int(round(max(0, notional) * self.cfg.long_stop_loss_notional_pct))
        stop_amount = min(max(1, dynamic_stop), max(1, int(self.cfg.long_stop_loss_cap_amount)))
        return -stop_amount

    def _inverse_stop_loss_amount_for_notional(self, notional: int) -> int:
        dynamic_stop = int(round(max(0, notional) * self.cfg.inverse_stop_loss_notional_pct))
        stop_amount = min(max(1, dynamic_stop), max(1, int(self.cfg.inverse_stop_loss_cap_amount)))
        return -stop_amount

    def _long_stop_loss_amount(self, pos: PositionState) -> int:
        return self._long_stop_loss_amount_for_notional(self._position_notional(pos))

    def _inverse_stop_loss_amount(self, pos: PositionState) -> int:
        return self._inverse_stop_loss_amount_for_notional(self._position_notional(pos))

    def _entry_stop_risk_amount(self, symbol: str, quantity: int, entry_price: int) -> int:
        notional = max(0, int(quantity) * int(entry_price))
        if symbol in self._inverse_symbols:
            return abs(self._inverse_stop_loss_amount_for_notional(notional))
        return abs(self._long_stop_loss_amount_for_notional(notional))

    def _recent_price_path(self, symbol: str) -> List[int]:
        return [q.current_price for q in self._get_recent_quotes(symbol) if q.current_price > 0]

    def _recent_local_high(self, symbol: str) -> tuple[int, int]:
        prices = self._recent_price_path(symbol)
        if not prices:
            return 0, -1
        high_price = max(prices)
        high_idx = max(idx for idx, price in enumerate(prices) if price == high_price)
        return high_price, high_idx

    def _recent_local_low_after_index(self, symbol: str, start_idx: int) -> int:
        prices = self._recent_price_path(symbol)
        if not prices:
            return 0
        tail = prices[start_idx:] if start_idx < len(prices) else prices[-1:]
        if not tail:
            return prices[-1]
        return min(tail)

    def _active_pool_quotes(self, include_inverse: bool = False) -> List[Quote]:
        candidates: List[Quote] = []
        seen: set[str] = set()
        pool_symbols = list(self._pool) if self._pool else list(self._quotes_cache.keys())
        for symbol in pool_symbols:
            quote = self._quotes_cache.get(symbol)
            if quote is None or quote.current_price <= 0:
                continue
            if not include_inverse and symbol in self._inverse_symbols:
                continue
            if symbol in seen:
                continue
            seen.add(symbol)
            candidates.append(quote)
        return candidates

    @staticmethod
    def _ranking_trade_amount(item) -> int:
        return max(0, int(item.current_price)) * max(0, int(item.volume))

    @staticmethod
    def _quote_trade_amount(quote: Quote) -> int:
        raw = int(getattr(quote, "trade_amount", 0) or 0)
        if raw > 0:
            return raw
        return max(0, int(quote.current_price)) * max(0, int(quote.volume))

    def _format_symbol_sample(self, symbols) -> str:
        sample = sorted(str(symbol) for symbol in symbols if symbol)
        if not sample:
            return "-"
        limit = max(1, int(self.cfg.dynamic_pool_log_symbol_count))
        clipped = sample[:limit]
        if len(sample) > limit:
            clipped.append("...")
        return ", ".join(clipped)

    def _is_dynamic_strong_leader_candidate(
        self,
        *,
        symbol: str,
        current_price: int,
        change_rate: float,
        trade_amount: int,
        rank: Optional[int],
    ) -> bool:
        if not symbol or symbol in self._inverse_symbols:
            return False
        if current_price < self.cfg.min_price:
            return False
        if change_rate < float(self.cfg.strong_leader_min_change_rate):
            return False
        has_top_rank = rank is not None and int(rank) <= max(1, int(self.cfg.strong_leader_top_rank))
        has_turnover = trade_amount >= int(self.cfg.strong_leader_min_trade_amount)
        return has_top_rank or has_turnover

    def _soft_bear_strong_leader_lane_active(self) -> bool:
        return (
            self.cfg.enable_soft_bear_strong_leader_longs
            and self._resolve_regime_profile_name() == "soft_bear"
            and bool(self._latest_strong_leader_symbols)
        )

    def _is_soft_bear_strong_leader_long_candidate(
        self,
        quote: Quote,
        *,
        score: Optional[float] = None,
    ) -> bool:
        if not self.cfg.enable_soft_bear_strong_leader_longs:
            return False
        if self._resolve_regime_profile_name() != "soft_bear":
            return False
        if quote.symbol in self._inverse_symbols:
            return False
        if quote.current_price <= 0 or quote.open_price <= 0:
            return False
        if quote.current_price <= quote.open_price:
            return False
        if quote.symbol not in self._latest_strong_leader_symbols:
            return False
        if quote.change_rate < float(self.cfg.soft_bear_strong_leader_min_change_rate):
            return False
        if self._quote_trade_amount(quote) < int(self.cfg.soft_bear_strong_leader_min_trade_amount):
            return False
        if score is not None and score < float(self.cfg.soft_bear_strong_leader_min_momentum):
            return False
        leader_ok, _, _ = self._passes_bull_leader_filter(quote)
        return leader_ok

    def _passes_neutral_leader_filter(
        self,
        quote: Quote,
        reclaim_speed_ticks: int,
        *,
        max_turnover_rank: Optional[int] = None,
        max_reclaim_ticks: Optional[int] = None,
        relative_strength_bonus_pp: float = 0.0,
    ) -> tuple[bool, str]:
        active_quotes = self._active_pool_quotes(include_inverse=False)
        active_quotes_by_symbol = {item.symbol: item for item in active_quotes}
        active_quotes_by_symbol[quote.symbol] = quote
        active_quotes = list(active_quotes_by_symbol.values())

        if quote.current_price <= quote.open_price:
            return False, "neutral_non_leader"

        if not active_quotes:
            return False, "neutral_non_leader"

        traded_values = sorted(
            (
                item.symbol,
                max(0, int(item.current_price)) * max(0, int(item.volume)),
            )
            for item in active_quotes
        )
        ranked_turnover = sorted(traded_values, key=lambda item: (item[1], item[0]), reverse=True)
        turnover_rank = next(
            (idx for idx, (symbol, _) in enumerate(ranked_turnover, start=1) if symbol == quote.symbol),
            len(ranked_turnover) + 1,
        )
        turnover_rank_limit = max_turnover_rank
        if turnover_rank_limit is None:
            turnover_rank_limit = int(self.cfg.neutral_leader_top_n)
        if turnover_rank > max(1, int(turnover_rank_limit)):
            return False, "neutral_low_turnover_rank"

        if len(active_quotes) > 1:
            median_change_rate = float(median(item.change_rate for item in active_quotes))
            min_strength = (
                median_change_rate
                + float(self.cfg.neutral_leader_relative_strength_pp)
                + float(relative_strength_bonus_pp)
            )
            if quote.change_rate < min_strength:
                return False, "neutral_weak_relative_strength"

        reclaim_tick_limit = max_reclaim_ticks
        if reclaim_tick_limit is None:
            reclaim_tick_limit = int(self.cfg.neutral_leader_max_reclaim_ticks)
        if reclaim_speed_ticks > max(1, int(reclaim_tick_limit)):
            return False, "neutral_slow_reclaim"

        return True, ""

    def _passes_bull_leader_filter(self, quote: Quote) -> tuple[bool, str, str]:
        active_quotes = self._active_pool_quotes(include_inverse=False)
        active_quotes_by_symbol = {item.symbol: item for item in active_quotes}
        active_quotes_by_symbol[quote.symbol] = quote
        active_quotes = list(active_quotes_by_symbol.values())

        if quote.current_price <= quote.open_price:
            return False, "bull_non_leader", ""
        if not active_quotes:
            return False, "bull_non_leader", ""

        traded_values = sorted(
            (
                item.symbol,
                max(0, int(item.current_price)) * max(0, int(item.volume)),
            )
            for item in active_quotes
        )
        ranked_turnover = sorted(traded_values, key=lambda item: (item[1], item[0]), reverse=True)
        turnover_rank = next(
            (idx for idx, (symbol, _) in enumerate(ranked_turnover, start=1) if symbol == quote.symbol),
            len(ranked_turnover) + 1,
        )
        if turnover_rank > max(1, int(self.cfg.bull_leader_top_n)):
            return False, "bull_low_turnover_rank", ""

        median_change_rate = float(median(item.change_rate for item in active_quotes))
        min_strength = median_change_rate + float(self.cfg.bull_leader_relative_strength_pp)
        if quote.change_rate < min_strength:
            return False, "bull_weak_relative_strength", ""

        payload = (
            f"entry_grade=A turnover_rank={turnover_rank} "
            f"relative_strength_floor={min_strength:.2f}"
        )
        return True, "", payload

    def _passes_bull_breakout_setup(self, quote: Quote, score: float) -> tuple[bool, str, str]:
        soft_bear_leader_lane = self._is_soft_bear_strong_leader_long_candidate(quote, score=score)
        if not self._is_bullish_regime() and not soft_bear_leader_lane:
            return False, "", "bull_only"
        history = self._get_recent_quotes(quote.symbol)
        hold_ticks = max(2, int(self.cfg.bull_breakout_hold_ticks))
        if len(history) < hold_ticks + 1:
            return False, "", "bull_breakout_wait"
        required_change_rate = self._regime_bullish_min_change_rate()
        required_score = self._regime_bullish_min_momentum_score()
        if soft_bear_leader_lane:
            required_change_rate = float(self.cfg.soft_bear_strong_leader_min_change_rate)
            required_score = float(self.cfg.soft_bear_strong_leader_min_momentum)
        if quote.change_rate < required_change_rate:
            return False, "", "bull_change_rate"
        if score < required_score:
            return False, "", "bull_score"
        if not self._is_volume_spike(quote, score=score):
            return False, "", "bull_volume"
        leader_ok, leader_reject, leader_payload = self._passes_bull_leader_filter(quote)
        if not leader_ok:
            return False, "", leader_reject
        if self._is_bull_late_entry_window():
            late_day_min_score = self._regime_bullish_min_momentum_score() + float(
                self.cfg.bull_breakout_late_entry_score_bonus
            )
            if score < late_day_min_score:
                return False, "", "bull_late_day_score"
            late_day_min_change = self._regime_bullish_min_change_rate() + float(
                self.cfg.bull_breakout_late_entry_change_rate_bonus
            )
            if quote.change_rate < late_day_min_change:
                return False, "", "bull_late_day_change"
        if self._bull_loss_count_today > 0:
            post_loss_min_score = self._regime_bullish_min_momentum_score() + float(
                self.cfg.bull_post_loss_score_bonus
            )
            if score < post_loss_min_score:
                return False, "", "bull_post_loss_score"
            post_loss_min_change = self._regime_bullish_min_change_rate() + float(
                self.cfg.bull_post_loss_change_rate_bonus
            )
            if quote.change_rate < post_loss_min_change:
                return False, "", "bull_post_loss_change"

        prices = [item.current_price for item in history]
        prior_prices = prices[:-hold_ticks] if len(prices) > hold_ticks else prices[:-1]
        if not prior_prices:
            return False, "", "bull_breakout_wait"
        prior_high = max(prior_prices)
        breakout_buffer_pct = float(self.cfg.bull_breakout_buffer_pct)
        if self._bull_loss_count_today > 0:
            breakout_buffer_pct += float(self.cfg.bull_post_loss_breakout_buffer_bonus_pct)
        breakout_level = prior_high * (1 + (breakout_buffer_pct / 100))
        if quote.current_price < breakout_level:
            return False, "", "bull_breakout_wait"

        recent_hold = history[-hold_ticks:]
        if any(item.current_price < breakout_level for item in recent_hold):
            return False, "", "bull_breakout_wait"

        reason = (
            f"setup_name=bull_breakout entry_reason=local_high_breakout "
            f"prior_high={prior_high} hold_ticks={hold_ticks} "
            f"leader_lane={'soft_bear' if soft_bear_leader_lane else 'bull'} {leader_payload}"
        )
        return True, "bull_breakout", reason

    def _passes_neutral_pullback_reclaim_setup(self, quote: Quote, score: float) -> tuple[bool, str, str]:
        if self._resolve_regime_profile_name() != "neutral":
            return False, "", "neutral_only"
        if not self._is_neutral_entry_window_open():
            return False, "", "neutral_too_early"
        history = self._get_recent_quotes(quote.symbol)
        if len(history) < 4:
            return False, "", "pullback_missing"
        if quote.current_price <= 0 or quote.open_price <= 0:
            return False, "", "pullback_missing"

        prices = [item.current_price for item in history[:-1]]
        if not prices:
            return False, "", "pullback_missing"

        local_high = max(prices)
        local_high_idx = max(idx for idx, price in enumerate(prices) if price == local_high)
        pullback_window = prices[local_high_idx + 1:]
        runup_from_open = self._pct_move(quote.open_price, local_high)
        proximity_to_day_high = self._loss_pct(max(quote.high_price, quote.current_price), quote.current_price)
        min_pullback_ticks = max(1, int(self.cfg.neutral_pullback_min_ticks))
        min_pullback_drop_pct = self.cfg.neutral_pullback_min_drop_pct
        min_runup_from_open_pct = self.cfg.neutral_min_runup_from_open_pct
        reclaim_buffer_pct = self.cfg.neutral_reclaim_buffer_pct
        min_score = self._regime_min_momentum_score()
        min_change_rate = self._regime_min_change_rate()
        leader_turnover_rank_limit: Optional[int] = None
        leader_reclaim_tick_limit: Optional[int] = None
        leader_relative_strength_bonus_pp = 0.0

        if self._is_neutral_first_entry_attempt():
            min_pullback_drop_pct += float(self.cfg.neutral_first_entry_min_drop_bonus_pct)
            min_runup_from_open_pct += float(self.cfg.neutral_first_entry_min_runup_bonus_pct)
            reclaim_buffer_pct += float(self.cfg.neutral_first_entry_reclaim_buffer_bonus_pct)
            min_score += float(self.cfg.neutral_first_entry_score_bonus)
            min_change_rate += float(self.cfg.neutral_first_entry_change_rate_bonus)
            leader_turnover_rank_limit = int(self.cfg.neutral_first_entry_max_turnover_rank)
            leader_reclaim_tick_limit = int(self.cfg.neutral_first_entry_max_reclaim_ticks)
            leader_relative_strength_bonus_pp += 0.15

        if self._is_neutral_post_loss_retry_available():
            retry_thresholds = self._neutral_retry_thresholds()
            min_pullback_ticks = retry_thresholds["min_pullback_ticks"]
            min_pullback_drop_pct = retry_thresholds["min_drop_pct"]
            min_runup_from_open_pct = retry_thresholds["min_runup_pct"]
            reclaim_buffer_pct = retry_thresholds["reclaim_buffer_pct"]
            min_score = retry_thresholds["min_score"]
            min_change_rate = retry_thresholds["min_change_rate"]

        if self._is_loss_stage_active() and not self._is_neutral_post_loss_retry_available():
            min_pullback_drop_pct += float(self.cfg.stage1_neutral_min_drop_bonus_pct)
            min_runup_from_open_pct += float(self.cfg.stage1_neutral_min_runup_bonus_pct)
            reclaim_buffer_pct += float(self.cfg.stage1_neutral_reclaim_buffer_bonus_pct)
            min_score += float(self.cfg.stage1_neutral_score_bonus)
            min_change_rate += float(self.cfg.stage1_neutral_change_rate_bonus)
            leader_turnover_rank_limit = min(
                int(leader_turnover_rank_limit or self.cfg.neutral_leader_top_n),
                int(self.cfg.stage1_neutral_max_turnover_rank),
            )
            leader_reclaim_tick_limit = min(
                int(leader_reclaim_tick_limit or self.cfg.neutral_leader_max_reclaim_ticks),
                int(self.cfg.stage1_neutral_max_reclaim_ticks),
            )
            leader_relative_strength_bonus_pp += 0.10

        if runup_from_open < min_runup_from_open_pct:
            return False, "", "pullback_missing"
        if not pullback_window:
            if proximity_to_day_high <= self.cfg.neutral_chase_block_proximity_pct:
                return False, "", "neutral_chase_block"
            return False, "", "pullback_missing"
        if len(pullback_window) < min_pullback_ticks:
            if proximity_to_day_high <= self.cfg.neutral_chase_block_proximity_pct:
                return False, "", "neutral_chase_block"
            return False, "", "pullback_missing"

        pullback_low = min(pullback_window)
        pullback_drop = self._loss_pct(local_high, pullback_low)
        if pullback_drop < min_pullback_drop_pct:
            if proximity_to_day_high <= self.cfg.neutral_chase_block_proximity_pct:
                return False, "", "neutral_chase_block"
            return False, "", "pullback_missing"
        if pullback_drop > self.cfg.neutral_pullback_max_drop_pct:
            return False, "", "pullback_broken"
        if self._pct_move(quote.open_price, quote.current_price) <= 0:
            return False, "", "neutral_non_leader"

        pullback_low_idx = local_high_idx + 1 + min(
            range(len(pullback_window)),
            key=lambda idx: pullback_window[idx],
        )
        reclaim_speed_ticks = max(1, (len(history) - 1) - pullback_low_idx)
        if self.cfg.enable_neutral_leader_filter:
            leader_ok, leader_reject = self._passes_neutral_leader_filter(
                quote,
                reclaim_speed_ticks=reclaim_speed_ticks,
                max_turnover_rank=leader_turnover_rank_limit,
                max_reclaim_ticks=leader_reclaim_tick_limit,
                relative_strength_bonus_pp=leader_relative_strength_bonus_pp,
            )
            if not leader_ok:
                return False, "", leader_reject

        reclaim_level = int(round(local_high * (1 + reclaim_buffer_pct / 100)))
        if quote.current_price < reclaim_level:
            return False, "", "reclaim_failed"
        if score < min_score:
            return False, "", "neutral_score"
        if quote.change_rate < min_change_rate:
            return False, "", "neutral_change_rate"
        if not self._is_volume_spike(quote, score=score):
            return False, "", "neutral_volume"

        reason = (
            f"setup_name=neutral_pullback_reclaim entry_reason=pullback_reclaim "
            f"local_high={local_high} reclaim_level={reclaim_level} drop_pct={pullback_drop:.2f} "
            f"reclaim_speed_ticks={reclaim_speed_ticks}"
        )
        return True, "neutral_pullback_reclaim", reason

    def _passes_soft_bear_inverse_setup(self, quote: Quote, score: float) -> tuple[bool, str, str]:
        if self._resolve_regime_profile_name() != "soft_bear":
            return False, "", "soft_bear_only"
        if quote.symbol not in self._inverse_symbols:
            return False, "", "soft_bear_inverse_only"
        history = self._get_recent_quotes(quote.symbol)
        if len(history) < 4:
            return False, "", "soft_bear_pullback_missing"
        if quote.current_price <= 0 or quote.open_price <= 0:
            return False, "", "soft_bear_pullback_missing"

        prices = [item.current_price for item in history[:-1]]
        if not prices:
            return False, "", "soft_bear_pullback_missing"

        local_high = max(prices)
        local_high_idx = max(idx for idx, price in enumerate(prices) if price == local_high)
        pullback_window = prices[local_high_idx + 1:]
        if not pullback_window:
            return False, "", "soft_bear_pullback_missing"

        runup_from_open = self._pct_move(quote.open_price, local_high)
        if runup_from_open < self.cfg.soft_bear_inverse_min_runup_pct:
            return False, "", "soft_bear_runup_missing"

        pullback_low = min(pullback_window)
        pullback_drop = self._loss_pct(local_high, pullback_low)
        if pullback_drop < self.cfg.soft_bear_inverse_min_drop_pct:
            return False, "", "soft_bear_pullback_missing"
        if pullback_drop > self.cfg.soft_bear_inverse_max_drop_pct:
            return False, "", "soft_bear_pullback_missing"

        reclaim_level = int(round(local_high * (1 + self.cfg.soft_bear_inverse_reclaim_buffer_pct / 100)))
        if quote.current_price < reclaim_level:
            return False, "", "soft_bear_reclaim_failed"
        if quote.change_rate < self.cfg.soft_bear_inverse_min_change_rate:
            return False, "", "soft_bear_runup_missing"
        if score < self.cfg.soft_bear_inverse_min_momentum:
            return False, "", "soft_bear_runup_missing"
        if not self._is_volume_spike(quote, score=score):
            return False, "", "soft_bear_volume_failed"

        reason = (
            f"setup_name=soft_bear_inverse_breakdown entry_reason=weak_rebound_failure "
            f"local_high={local_high} reclaim_level={reclaim_level} drop_pct={pullback_drop:.2f}"
        )
        return True, "soft_bear_inverse_breakdown", reason

    def _passes_hard_bear_inverse_setup(self, quote: Quote, score: float) -> tuple[bool, str, str]:
        if self._resolve_regime_profile_name() != "bear":
            return False, "", "bear_only"
        min_change_rate = self._regime_inverse_min_change_rate()
        min_momentum = self._regime_inverse_min_momentum()
        if self._is_loss_stage_active():
            min_change_rate += self.cfg.stage1_inverse_change_bonus
            min_momentum += self.cfg.stage1_inverse_score_bonus
        if quote.change_rate < min_change_rate:
            return False, "", "bear_inverse_change_rate"
        if score < min_momentum:
            return False, "", "bear_inverse_score"
        if not self._is_volume_spike(quote, score=score):
            return False, "", "bear_inverse_volume"

        payload = (
            "setup_name=hard_bear_inverse_momentum "
            f"entry_reason=bear_momentum score={score:.2f}"
        )
        return True, "hard_bear_inverse_momentum", payload

    def initialize(self):
        today = self._today()
        self._state_loaded_for_today = False
        self._daily_breaker_pnl_offset = 0
        self._load_daily_state()
        self._load_strategy_gates()
        if self._halted and self._halt_date == today and self.cfg.allow_hard_stop_bypass_for_day:
            logger.warning(
                "당일 하드스탑 플래그 복구 무시(임시 모드): 오늘 전일 누적손실이 있어도 재개합니다."
            )
            self._halted = False
            self._halt_date = None
            self._hard_stop_bypass_for_day = True
        if self._current_day != today:
            self.daily_pnl = DailyPnL()
            self._halted = False
            self._halt_date = None
            self._sell_cooldown = {}
            self._symbol_cooldown_until = {}
            self._strategy_cooldown_until = {}
            self._startup_rebalance_active = False
            self._startup_rebalance_ticks = 0
            self._market_data_ready_for_entries = self.market_data is None
            self._market_data_ready_streak = 0
            self._market_data_readiness_reason = ""
            self._last_market_data_wait_log_at = None
            self._leader_support_bull_bias_active = False
            self._last_cumulative_volumes = {}
            self._entry_filter_log_cache = {}
            self._daily_breaker_pnl_offset = 0
            self._loaded_position_meta = {}
            self._recent_quotes = {}
            self._risk_stage_label = "normal"
            self._pending_entry_meta = {}
            self._neutral_loss_count_today = 0
            self._neutral_last_loss_at = None
            self._neutral_post_loss_reentries_today = 0
            self._bull_loss_count_today = 0
            self._bull_last_loss_at = None
            self._shadow_blocked_candidates = {}
            self._latest_direct_dynamic_symbols = set()
            self._latest_strong_leader_symbols = set()
            self._latest_strong_leader_snapshot = {}
        elif self._state_loaded_for_today and not self.cfg.use_restored_pnl_for_daily_breaker:
            restored_pnl = int(self.daily_pnl.realized_net_pnl)
            if restored_pnl != 0 or self.daily_pnl.trade_count > 0:
                self._daily_breaker_pnl_offset = restored_pnl
                logger.warning(
                    "재시작 복구 손익은 브레이커 기준에서 제외합니다 "
                    "(복구 순실현=%s원, 복구 매매=%d건, 브레이커 시작값=0원).",
                    f"{restored_pnl:,}",
                    self.daily_pnl.trade_count,
                )
        self._recent_tick_volumes = {}
        self._latest_tick_volumes = {}
        self._recent_quotes = {}
        self._shadow_blocked_candidates = {}
        self._current_day = today
        self._session_start_at = None
        self._build_pool()
        self._check_market_regime()
        regime_profile = self._build_regime_profile()
        self._log_regime_profiles()
        total_loss_limit = regime_profile["daily_total_loss_limit"]
        capital_base = self._allocation_capital_base()
        target_total_exposure = self._regime_target_total_exposure_amount()
        max_single_position_amount = self._regime_max_single_position_amount()
        target_long_slot_budget = int(
            target_total_exposure / max(1, self._effective_max_position_count())
        )
        blocked_windows = ", ".join(
            label for _, _, label in self._entry_block_windows
        ) if self._entry_block_windows else "없음"

        if self._halted and self._halt_date == today:
            logger.info("당일 하드스탑 상태 유지: 신규 거래 중지")
            if self._hard_stop_bypass_for_day:
                logger.info("당일 하드스탑 우회 모드가 적용되어 거래를 계속 진행합니다.")

        logger.info("전략 초기화: 모멘텀 스캘핑")
        if self.market_data is not None:
            logger.info(
                "  시작 워밍업: 시장 데이터 준비 후 신규 진입 허용 (%d틱 연속, 유효 시세 최소 %d개)",
                max(1, int(self.cfg.startup_market_data_ready_ticks)),
                max(1, int(self.cfg.startup_market_data_min_valid_quote_count)),
            )
        logger.info("  시드: %s원, 기준자본: %s원, 기본 종목당: %s원",
                     f"{self.cfg.seed_money:,}", f"{capital_base:,}", f"{self.cfg.per_stock_amount:,}")
        logger.info("  최대 동시 보유 수: %d종목", self._effective_max_position_count())
        logger.info("  익절: +%.1f%%, 개별손절: %s원, 추적손절: %.1f%%",
                     regime_profile["take_profit_pct"],
                     f"{regime_profile['per_position_stop_loss']:,}",
                     regime_profile["trailing_stop_pct"])
        logger.info("  일일 목표(순실현): +%s원, 최대손실(순실현): %s원",
                     f"{regime_profile['daily_profit_target']:,}",
                     f"{regime_profile['daily_loss_limit']:,}")
        if self.cfg.enable_expected_net_filter:
            logger.info(
                "  진입 필터: 기대상승 %.2f%%, 최소 기대순익 %s원, 최소 RR %.2f",
                regime_profile["expected_move_pct"],
                f"{regime_profile['min_expected_net_profit']:,}",
                regime_profile["min_expected_rr_ratio"],
            )
        if self.cfg.enable_unrealized_loss_guard:
            logger.info("  보조손실컷(순손익추정): %s원", f"{total_loss_limit:,}")
        if self._entry_block_windows:
            logger.info("  신규 진입 차단 시간대: %s", blocked_windows)
            if self.cfg.enable_dynamic_entry_block_windows:
                logger.info(
                    "  신규 진입 차단 시간대 자동해제: 약세점수 %d 이상",
                    self.cfg.dynamic_entry_block_disable_bear_score,
                )
        logger.info(
            "  오늘 적용값 요약: 총노출 %s원(%.0f%%), 종목당 목표 %s원, 종목당 최대 %s원(%.0f%%), "
            "개별손절 %s원, 일손실한도 %s원, 보조손실컷 %s원, 진입임계(점수 %.2f/등락률 %.2f%%/최소가 %s원), 차단시간 [%s]",
            f"{target_total_exposure:,}",
            regime_profile["capital_utilization_pct"] * 100,
            f"{target_long_slot_budget:,}",
            f"{max_single_position_amount:,}",
            regime_profile["max_single_position_pct"] * 100,
            f"{regime_profile['per_position_stop_loss']:,}",
            f"{regime_profile['daily_loss_limit']:,}",
            f"{total_loss_limit:,}",
            regime_profile["min_momentum_score"],
            regime_profile["min_change_rate"],
            f"{self.cfg.min_price:,}",
            blocked_windows,
        )
        if self.cfg.enable_pullback_entry_filter:
            logger.info(
                "  눌림목 필터: 등락률 %.2f%% 이상은 고점대비 조정 %.2f%%~%.2f%% + 시가대비 %.2f%% 이상에서만 신규진입",
                self.cfg.pullback_activation_change_rate,
                self.cfg.pullback_required_min_drop_pct,
                self.cfg.pullback_allowed_max_drop_pct,
                self.cfg.pullback_min_vs_open_pct,
            )
        if self.cfg.enable_volume_spike_filter:
            logger.info(
                "  거래량 스파이크 게이트: 최근 %d개 대비 %.1fx~%.1fx + 최소 1틱 %d주",
                self.cfg.volume_spike_min_history,
                regime_profile["volume_spike_ratio"],
                regime_profile["volume_spike_ratio_min"],
                regime_profile["volume_spike_abs_min"],
            )
        logger.info("  레짐 프로파일: %s (현재 약세점수=%d)", self._regime_profile_name, self._bear_score)
        logger.info(
            "  레짐 조정: min_change=%s, min점수=%s, 익절=%s%%, 손절=%s원, 추적손절=%s%%, 최대보유=%d분",
            f"{regime_profile['min_change_rate']:.2f}",
            f"{regime_profile['min_momentum_score']:.2f}",
            f"{regime_profile['take_profit_pct']:.2f}",
            f"{regime_profile['per_position_stop_loss']:,}",
            f"{regime_profile['trailing_stop_pct']:.2f}",
            regime_profile["max_position_holding_minutes"],
        )
        logger.info("  시장 레짐: 약세점수=%d (모드: %s)",
                     self._bear_score, self.cfg.bear_market_mode)
        if self.cfg.inverse_enabled:
            logger.info("  인버스: 활성화 (임계=%d, 최대%d종목)",
                         self.cfg.bearish_threshold, self._regime_inverse_max_positions())
            logger.info(
                "  인버스 진입 강화: 약세점수≥%d, 등락률≥%.2f%%, 모멘텀≥%.2f, "
                "추적손절 발동이익≥%.2f%%, 거래량스파이크 완화=-%.2f",
                max(self.cfg.bearish_threshold, self._regime_inverse_min_bear_score()),
                self._regime_inverse_min_change_rate(),
                self._regime_inverse_min_momentum(),
                self._regime_inverse_trailing_stop_activation_gain_pct(),
                self.cfg.inverse_volume_spike_ratio_offset,
            )
        logger.info("  풀 크기: %d종목", len(self._pool))

    def _log_regime_profiles(self):
        capital_base = self._allocation_capital_base()
        bull_profile = self._build_regime_profile(profile_name="bull")
        neutral_profile = self._build_regime_profile(profile_name="neutral")
        soft_bear_profile = self._build_regime_profile(profile_name="soft_bear")
        bear_profile = self._build_regime_profile(profile_name="bear")
        logger.info("레짐 프로파일 비교표 (활성=%s)", self._regime_profile_name)
        logger.info(
            "  [강세] long 예산=%s원/%s원 익절=%s%% 손절=%s원 추적=%s%% 보유=%d분 / "
            "inv 익절=%s%% inv손절=%s%% inv추적=%s%% inv보유=%d분",
            f"{int(capital_base * bull_profile['capital_utilization_pct']):,}",
            f"{int(capital_base * bull_profile['max_single_position_pct']):,}",
            f"{bull_profile['take_profit_pct']:.2f}",
            f"{bull_profile['per_position_stop_loss']:,}",
            f"{bull_profile['trailing_stop_pct']:.2f}",
            bull_profile["max_position_holding_minutes"],
            f"{bull_profile['inverse_take_profit_pct']:.2f}",
            f"{bull_profile['inverse_stop_loss_pct']:.2f}",
            f"{bull_profile['inverse_trailing_stop_pct']:.2f}",
            bull_profile["inverse_max_hold_minutes"],
        )
        logger.info(
            "  [중립] long 예산=%s원/%s원 익절=%s%% 손절=%s원 추적=%s%% 보유=%d분 / "
            "inv 익절=%s%% inv손절=%s%% inv추적=%s%% inv보유=%d분",
            f"{int(capital_base * neutral_profile['capital_utilization_pct']):,}",
            f"{int(capital_base * neutral_profile['max_single_position_pct']):,}",
            f"{neutral_profile['take_profit_pct']:.2f}",
            f"{neutral_profile['per_position_stop_loss']:,}",
            f"{neutral_profile['trailing_stop_pct']:.2f}",
            neutral_profile["max_position_holding_minutes"],
            f"{neutral_profile['inverse_take_profit_pct']:.2f}",
            f"{neutral_profile['inverse_stop_loss_pct']:.2f}",
            f"{neutral_profile['inverse_trailing_stop_pct']:.2f}",
            neutral_profile["inverse_max_hold_minutes"],
        )
        logger.info(
            "  [완만약세] long 예산=%s원/%s원 익절=%s%% 손절=%s원 추적=%s%% 보유=%d분 / "
            "inv 익절=%s%% inv손절=%s%% inv추적=%s%% inv보유=%d분",
            f"{int(capital_base * soft_bear_profile['capital_utilization_pct']):,}",
            f"{int(capital_base * soft_bear_profile['max_single_position_pct']):,}",
            f"{soft_bear_profile['take_profit_pct']:.2f}",
            f"{soft_bear_profile['per_position_stop_loss']:,}",
            f"{soft_bear_profile['trailing_stop_pct']:.2f}",
            soft_bear_profile["max_position_holding_minutes"],
            f"{soft_bear_profile['inverse_take_profit_pct']:.2f}",
            f"{soft_bear_profile['inverse_stop_loss_pct']:.2f}",
            f"{soft_bear_profile['inverse_trailing_stop_pct']:.2f}",
            soft_bear_profile["inverse_max_hold_minutes"],
        )
        logger.info(
            "  [약세] long 예산=%s원/%s원 익절=%s%% 손절=%s원 추적=%s%% 보유=%d분 / "
            "inv 익절=%s%% inv손절=%s%% inv추적=%s%% inv보유=%d분",
            f"{int(capital_base * bear_profile['capital_utilization_pct']):,}",
            f"{int(capital_base * bear_profile['max_single_position_pct']):,}",
            f"{bear_profile['take_profit_pct']:.2f}",
            f"{bear_profile['per_position_stop_loss']:,}",
            f"{bear_profile['trailing_stop_pct']:.2f}",
            bear_profile["max_position_holding_minutes"],
            f"{bear_profile['inverse_take_profit_pct']:.2f}",
            f"{bear_profile['inverse_stop_loss_pct']:.2f}",
            f"{bear_profile['inverse_trailing_stop_pct']:.2f}",
            bear_profile["inverse_max_hold_minutes"],
        )

    def _load_daily_state(self):
        if not self._state_path:
            return
        try:
            self._loaded_position_meta = {}
            if not self._state_path.exists():
                return
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
            raw_date = payload.get("date")
            if not raw_date:
                return

            state_day = datetime.strptime(raw_date, "%Y-%m-%d").date()
            if state_day != self._today():
                return

            self._current_day = state_day
            self.daily_pnl = DailyPnL(
                realized_gross_pnl=payload.get("realized_gross_pnl", 0),
                realized_net_pnl=payload.get("realized_net_pnl", 0),
                fees_paid=payload.get("fees_paid", 0),
                taxes_paid=payload.get("taxes_paid", 0),
                trade_count=payload.get("trade_count", 0),
                win_count=payload.get("win_count", 0),
                loss_count=payload.get("loss_count", 0),
                breakeven_count=payload.get("breakeven_count", 0),
                winning_net_pnl_sum=payload.get("winning_net_pnl_sum", 0),
                losing_net_pnl_sum=payload.get("losing_net_pnl_sum", 0),
                largest_win_net=payload.get("largest_win_net", 0),
                largest_loss_net=payload.get("largest_loss_net", 0),
            )
            self._neutral_loss_count_today = int(payload.get("neutral_loss_count", 0) or 0)
            self._neutral_post_loss_reentries_today = int(payload.get("neutral_post_loss_reentries", 0) or 0)
            self._neutral_last_loss_at = self._parse_state_datetime(payload.get("neutral_last_loss_at"))
            self._bull_loss_count_today = int(payload.get("bull_loss_count", 0) or 0)
            self._bull_last_loss_at = self._parse_state_datetime(payload.get("bull_last_loss_at"))
            raw_halt_date = payload.get("halt_date")
            if isinstance(raw_halt_date, str):
                try:
                    self._halt_date = datetime.strptime(raw_halt_date, "%Y-%m-%d").date()
                except ValueError:
                    self._halt_date = None
            else:
                self._halt_date = None
            self._halted = bool(payload.get("halted", False))
            raw_positions = payload.get("open_positions")
            if isinstance(raw_positions, dict):
                self._loaded_position_meta = raw_positions
            self._state_loaded_for_today = True
            logger.info(
                "일일 상태 복구 완료: %s (누적실현=%s원, 매매=%d건)",
                raw_date,
                f"{self.daily_pnl.realized_net_pnl:,}",
                self.daily_pnl.trade_count,
            )
        except Exception as e:
            logger.warning("일일 상태 복구 실패(무시): %s", e)

    def _save_daily_state(self):
        if not self._state_path:
            return
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "date": self._current_day.isoformat() if self._current_day else None,
                "realized_gross_pnl": self.daily_pnl.realized_gross_pnl,
                "realized_net_pnl": self.daily_pnl.realized_net_pnl,
                "fees_paid": self.daily_pnl.fees_paid,
                "taxes_paid": self.daily_pnl.taxes_paid,
                "trade_count": self.daily_pnl.trade_count,
                "win_count": self.daily_pnl.win_count,
                "loss_count": self.daily_pnl.loss_count,
                "breakeven_count": self.daily_pnl.breakeven_count,
                "winning_net_pnl_sum": self.daily_pnl.winning_net_pnl_sum,
                "losing_net_pnl_sum": self.daily_pnl.losing_net_pnl_sum,
                "largest_win_net": self.daily_pnl.largest_win_net,
                "largest_loss_net": self.daily_pnl.largest_loss_net,
                "neutral_loss_count": self._neutral_loss_count_today,
                "neutral_last_loss_at": self._neutral_last_loss_at.isoformat(timespec="seconds")
                if self._neutral_last_loss_at
                else None,
                "neutral_post_loss_reentries": self._neutral_post_loss_reentries_today,
                "bull_loss_count": self._bull_loss_count_today,
                "bull_last_loss_at": self._bull_last_loss_at.isoformat(timespec="seconds")
                if self._bull_last_loss_at
                else None,
                "halted": self._halted,
                "halt_date": self._halt_date.isoformat() if self._halt_date else None,
                "open_positions": self._serialize_open_positions_for_state(),
                "updated_at": self._now().isoformat(timespec="seconds"),
            }
            self._state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("일일 상태 저장 실패(무시): %s", e)

    def _serialize_open_positions_for_state(self) -> Dict[str, dict]:
        payload: Dict[str, dict] = {}
        for sym, pos in self.positions.items():
            payload[sym] = {
                "buy_price": pos.buy_price,
                "quantity": pos.quantity,
                "invested_amount": pos.invested_amount,
                "buy_time": pos.buy_time.isoformat(timespec="seconds"),
                "high_since_buy": pos.high_since_buy,
                "entry_strategy_name": pos.entry_strategy_name,
                "entry_setup_name": pos.entry_setup_name,
                "entry_reason": pos.entry_reason,
                "regime_label": pos.regime_label,
                "bear_score": pos.bear_score,
                "planned_risk_stage": pos.planned_risk_stage,
                "entry_grade": pos.entry_grade,
                "partial_exit_done": pos.partial_exit_done,
            }
        return payload

    @staticmethod
    def _parse_state_datetime(raw: Optional[str]) -> Optional[datetime]:
        if not isinstance(raw, str) or not raw:
            return None
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return None

    def _clear_restored_positions_after_grace(self, now: datetime):
        grace_seconds = max(0, int(self.cfg.restored_position_grace_seconds))
        if grace_seconds <= 0:
            for pos in self.positions.values():
                if pos.is_restored:
                    pos.is_restored = False
                    pos.restored_at = None
            return

        released = 0
        for pos in self.positions.values():
            if not pos.is_restored or pos.restored_at is None:
                continue
            elapsed = (now - pos.restored_at).total_seconds()
            if elapsed >= grace_seconds:
                pos.is_restored = False
                pos.restored_at = None
                released += 1

        if released > 0:
            logger.info(
                "재시작 복구 유예 종료: %d종목에 보유시간 규칙을 다시 적용합니다.",
                released,
            )

    def get_watchlist(self) -> List[str]:
        now = self._now()
        if (self._last_pool_refresh and
                (now - self._last_pool_refresh).total_seconds() >= self.cfg.pool_refresh_interval):
            self._build_pool()
        return self._pool

    def on_tick(self, quote: Quote) -> List[Order]:
        self._quotes_cache[quote.symbol] = quote
        self._record_recent_quote(quote)
        if self._session_start_at is None:
            self._session_start_at = self._now()
        orders = []

        if self._halted:
            return orders

        if quote.symbol in self.positions:
            order = self._evaluate_sell(quote)
            if order:
                orders.append(order)
            else:
                scale_in = self._evaluate_buy(quote)
                if scale_in:
                    orders.append(scale_in)

        else:
            long_count = sum(1 for s in self.positions if s not in self._inverse_symbols)
            max_long_count = self._effective_max_position_count()
            if long_count < max_long_count:
                order = self._evaluate_buy(quote)
                if order:
                    orders.append(order)

        return orders

    def on_batch_tick(self, quotes: List[Quote]) -> List[Order]:
        """배치 시세를 받아 전체적으로 판단한다."""
        now = self._now()
        if self._session_start_at is None:
            self._session_start_at = now
        for q in quotes:
            self._quotes_cache[q.symbol] = q
            self._record_recent_quote(q)
            self._update_tick_volume_state(q)
            self._refresh_entry_signal_if_stale(q.symbol, now)

        self._clear_restored_positions_after_grace(now)
        self._cleanup_stale_entry_signals(now)
        self._check_market_regime(quotes=quotes)
        self._update_shadow_blocked_candidates(quotes, now=now)
        market_data_ready = self._update_market_data_readiness(quotes=quotes, now=now)
        effective_max_position_count = self._effective_max_position_count()

        # 거래 중지 상태면 주문 없음
        if self._halted:
            return []

        # 미실현 손익 업데이트 + 고가 추적
        for sym, pos in self.positions.items():
            q = self._quotes_cache.get(sym)
            if q:
                if q.current_price > pos.high_since_buy:
                    pos.high_since_buy = q.current_price

        # 장마감 청산 (15:15 이후) — 반드시 halt 설정 (실거래 모드만)
        if self.market_data is not None:
            if now.hour >= 15 and now.minute >= 15:
                self._halted = True
                self._halt_date = now.date()
                if self.positions:
                    logger.info("장마감 임박 → 전량 청산")
                    self._alerts.send(
                        event_key="market_close_liquidation",
                        title="장마감 전량 청산",
                        message="15:15 이후 장마감 규칙에 따라 보유 포지션 전량 청산을 수행합니다.",
                        level="warning",
                        cooldown_seconds=1800,
                    )
                    return self._liquidate_all()
                return []

        # 단계형 리스크 관리: 실현 + 미실현 합산 총손익 기준
        realized_net = self._effective_realized_net_for_breaker()
        unrealized_net = self._estimate_unrealized_net_pnl()
        total_net = realized_net + unrealized_net
        risk_stage = self._current_risk_stage(total_net)
        self._log_risk_stage_change(risk_stage, total_net)
        if not self._hard_stop_bypass_for_day:
            breaker_orders = self._evaluate_daily_breakers(liquidate=True)
            if breaker_orders is not None:
                return breaker_orders
        else:
            logger.info("당일 하드스탑 무시 모드: 일일 손익 브레이크는 비활성화됩니다.")

        # 개별 종목 평가
        orders = []

        # 1) 매도 먼저 (일반 + 인버스 모두)
        for q in quotes:
            if q.symbol in self.positions:
                if q.symbol in self._inverse_symbols:
                    order = self._evaluate_inverse_sell(q)
                else:
                    order = self._evaluate_sell(q)
                if order:
                    orders.append(order)

        if self._startup_rebalance_active and len(self.positions) < effective_max_position_count:
            self._startup_rebalance_active = False
            for p in self.positions.values():
                p.is_restored = False
            logger.info("재시작 복구 모드 해제: 보유 슬롯이 확보되어 보유시간 규칙을 정상 적용합니다.")

        if self._startup_rebalance_active and len(self.positions) >= effective_max_position_count:
            self._startup_rebalance_ticks -= 1
            logger.info(
                "재시작 복구 모드: 기존 보유 %d종목 감시 중, 신규 진입 잠시 보류 (%d틱 남음)",
                len(self.positions),
                max(self._startup_rebalance_ticks, 0),
            )
            if self._startup_rebalance_ticks <= 0:
                self._startup_rebalance_active = False
                for p in self.positions.values():
                    p.is_restored = False
                logger.info("재시작 복구 모드 종료: 보유 포지션의 보유시간 규칙 정상 적용")
            return orders

        if not market_data_ready:
            return orders

        # 2) 일반 매수 후보를 모멘텀 점수 기준으로 정렬 후 진입 시도
        ranked_candidates = self._rank_long_entry_candidates(quotes)
        for score, q in ranked_candidates:
            if q.symbol in self._inverse_symbols:
                continue  # 인버스는 아래에서 별도 처리
            long_count = sum(
                1 for s in self.positions if s not in self._inverse_symbols
            )
            pending_long = sum(
                1 for o in orders
                if (
                    o.side == OrderSide.BUY and
                    o.symbol not in self._inverse_symbols and
                    o.symbol not in self.positions
                )
            )
            if q.symbol not in self.positions and long_count + pending_long >= effective_max_position_count:
                continue

            order = self._evaluate_buy(q, pending_orders=orders, score_hint=score)
            if order:
                orders.append(order)

        # 3) 인버스 매수 (인버스 포지션 카운트 기준)
        if self.cfg.inverse_enabled and self._bear_score >= self.cfg.bearish_threshold:
            regime_inverse_max_positions = self._regime_inverse_max_positions()
            if regime_inverse_max_positions <= 0:
                return orders
            for q in quotes:
                if q.symbol not in self._inverse_symbols:
                    continue
                inv_count = sum(
                    1 for s in self.positions if s in self._inverse_symbols
                )
                pending_inv = sum(
                    1 for o in orders
                    if o.side == OrderSide.BUY and o.symbol in self._inverse_symbols
                )
                if inv_count + pending_inv >= regime_inverse_max_positions:
                    break
                if q.symbol not in self.positions:
                    order = self._evaluate_inverse_buy(q, pending_orders=orders)
                    if order:
                        orders.append(order)

        return orders

    def _has_valid_index_regime_info(self) -> bool:
        index_info = self._cached_index_regime_info
        return bool(index_info) and all(float(value) > 0 for value in index_info)

    def _update_market_data_readiness(self, quotes: List[Quote], now: datetime) -> bool:
        if self.market_data is None:
            self._market_data_ready_for_entries = True
            self._market_data_ready_streak = 0
            self._market_data_readiness_reason = ""
            return True

        regular_quotes = [q for q in quotes if q.symbol not in self._inverse_symbols]
        valid_quotes = [
            q for q in regular_quotes
            if q.current_price > 0 and q.open_price > 0
        ]
        valid_index = self._has_valid_index_regime_info()
        required_valid_quotes = min(
            max(1, int(self.cfg.startup_market_data_min_valid_quote_count)),
            max(1, len(regular_quotes)),
        ) if regular_quotes else max(1, int(self.cfg.startup_market_data_min_valid_quote_count))
        valid_quote_count = len(valid_quotes)
        market_data_valid = (
            valid_index
            and bool(regular_quotes)
            and valid_quote_count >= required_valid_quotes
        )

        if market_data_valid:
            self._market_data_ready_streak += 1
            required_ticks = max(1, int(self.cfg.startup_market_data_ready_ticks))
            if not self._market_data_ready_for_entries and self._market_data_ready_streak >= required_ticks:
                self._market_data_ready_for_entries = True
                self._market_data_readiness_reason = ""
                logger.info(
                    "시장 데이터 준비 완료: 신규 진입 재개 (지수 유효, 유효 시세 %d/%d, 연속 %d틱)",
                    valid_quote_count,
                    len(regular_quotes),
                    self._market_data_ready_streak,
                )
            return self._market_data_ready_for_entries

        previous_ready = self._market_data_ready_for_entries
        self._market_data_ready_for_entries = False
        self._market_data_ready_streak = 0

        reason_parts: List[str] = []
        if not valid_index:
            reason_parts.append("지수 일봉 미준비")
        if not regular_quotes:
            reason_parts.append("일반 종목 시세 없음")
        elif valid_quote_count < required_valid_quotes:
            reason_parts.append(
                f"유효 시세 부족 {valid_quote_count}/{required_valid_quotes}"
            )
        self._market_data_readiness_reason = ", ".join(reason_parts) if reason_parts else "시장 데이터 미준비"

        log_interval = max(1, int(self.cfg.startup_market_data_wait_log_interval_seconds))
        should_log = previous_ready
        if not should_log:
            if self._last_market_data_wait_log_at is None:
                should_log = True
            else:
                should_log = (now - self._last_market_data_wait_log_at).total_seconds() >= log_interval
        if should_log:
            logger.warning(
                "시장 데이터 준비 대기: 신규 진입 보류 (%s, 유효 시세 %d/%d, 지수캐시=%s)",
                self._market_data_readiness_reason,
                valid_quote_count,
                len(regular_quotes),
                "OK" if valid_index else "NONE",
            )
            self._last_market_data_wait_log_at = now
        return False

    def on_order_filled(self, result: OrderResult):
        if result.side == OrderSide.BUY:
            if not result.success:
                self._pending_entry_meta.pop(result.symbol, None)
                return

            fill_price = result.price
            if fill_price <= 0:
                cached = self._quotes_cache.get(result.symbol)
                fill_price = cached.current_price if cached else 0
            if fill_price <= 0:
                self._pending_entry_meta.pop(result.symbol, None)
                return

            buy_notional = fill_price * result.quantity
            buy_fee = self._calc_commission_cost(buy_notional)
            if buy_fee > 0:
                self.daily_pnl.fees_paid += buy_fee
                # 매수 수수료는 체결 시점에 확정 비용으로 반영
                self.daily_pnl.realized_net_pnl -= buy_fee

            entry_meta = dict(self._pending_entry_meta.pop(result.symbol, {}))
            existing = self.positions.get(result.symbol)
            if existing:
                total_qty = existing.quantity + result.quantity
                total_invested = existing.invested_amount + (fill_price * result.quantity)
                existing.quantity = total_qty
                existing.invested_amount = total_invested
                existing.buy_price = int(round(total_invested / total_qty))
                existing.is_restored = False
                existing.restored_at = None
                existing.partial_exit_done = False
                if fill_price > existing.high_since_buy:
                    existing.high_since_buy = fill_price
                if entry_meta and not existing.entry_setup_name:
                    existing.entry_strategy_name = str(entry_meta.get("strategy_name", "") or "")
                    existing.entry_setup_name = str(entry_meta.get("setup_name", "") or "")
                    existing.entry_reason = str(entry_meta.get("entry_reason", "") or "")
                    existing.regime_label = str(entry_meta.get("regime_label", "") or "")
                    existing.bear_score = int(entry_meta.get("bear_score", 0) or 0)
                    existing.planned_risk_stage = str(entry_meta.get("planned_risk_stage", "") or "")
                    existing.entry_grade = str(entry_meta.get("entry_grade", "") or "")
                tag = "[INV] " if result.symbol in self._inverse_symbols else ""
                logger.info(
                    "%s추가매수 체결: %s +%d주 @ %s원 (평단 %s원, 총 %d주)",
                    tag,
                    result.symbol,
                    result.quantity,
                    f"{fill_price:,}",
                    f"{existing.buy_price:,}",
                    existing.quantity,
                )
                alert_key_suffix = self._now().strftime("%H%M%S%f")
                self._alerts.send(
                    event_key=f"buy_fill_add_{result.symbol}_{alert_key_suffix}",
                    title="추가매수 체결",
                    message=(
                        f"{result.symbol} +{result.quantity}주 @ {fill_price:,}원\n"
                        f"평단 {existing.buy_price:,}원, 총 {existing.quantity}주"
                    ),
                    level="info",
                    cooldown_seconds=0,
                )
                self._save_daily_state()
                return

            if bool(entry_meta.get("neutral_post_loss_retry")):
                self._neutral_post_loss_reentries_today += 1
                logger.info(
                    "중립장 손실 후 재도전 사용: %d/%d (%s)",
                    self._neutral_post_loss_reentries_today,
                    max(0, int(self.cfg.neutral_post_loss_reentry_limit)),
                    result.symbol,
                )

            self.positions[result.symbol] = PositionState(
                symbol=result.symbol,
                buy_price=fill_price,
                quantity=result.quantity,
                invested_amount=fill_price * result.quantity,
                is_restored=False,
                restored_at=None,
                entry_strategy_name=str(entry_meta.get("strategy_name", "") or ""),
                entry_setup_name=str(entry_meta.get("setup_name", "") or ""),
                entry_reason=str(entry_meta.get("entry_reason", "") or ""),
                regime_label=str(entry_meta.get("regime_label", "") or ""),
                bear_score=int(entry_meta.get("bear_score", 0) or 0),
                planned_risk_stage=str(entry_meta.get("planned_risk_stage", "") or ""),
                entry_grade=str(entry_meta.get("entry_grade", "") or ""),
            )
            tag = "[INV] " if result.symbol in self._inverse_symbols else ""
            logger.info(
                "%s매수 체결: %s %d주 @ %s원 "
                "(strategy_name=%s, setup_name=%s, regime_label=%s, bear_score=%d, planned_risk_stage=%s, entry_grade=%s)",
                tag,
                result.symbol,
                result.quantity,
                f"{fill_price:,}",
                self.positions[result.symbol].entry_strategy_name or "-",
                self.positions[result.symbol].entry_setup_name or "-",
                self.positions[result.symbol].regime_label or "-",
                self.positions[result.symbol].bear_score,
                self.positions[result.symbol].planned_risk_stage or "-",
                self.positions[result.symbol].entry_grade or "-",
            )
            alert_key_suffix = self._now().strftime("%H%M%S%f")
            self._alerts.send(
                event_key=f"buy_fill_{result.symbol}_{alert_key_suffix}",
                title="매수 체결",
                message=(
                    f"{result.symbol} {result.quantity}주 @ {fill_price:,}원\n"
                    f"전략 {self.positions[result.symbol].entry_strategy_name or '-'} / "
                    f"셋업 {self.positions[result.symbol].entry_setup_name or '-'} / "
                    f"레짐 {self.positions[result.symbol].regime_label or '-'}"
                ),
                level="info",
                cooldown_seconds=0,
            )
            self._save_daily_state()

        elif result.side == OrderSide.SELL:
            if not result.success:
                # 매도 실패 시 실제 보유는 유지되므로 포지션을 제거하면 안 된다.
                logger.warning("매도 실패(포지션 유지): %s", result.symbol)
                self._alerts.send(
                    event_key=f"sell_failed_{result.symbol}",
                    title="매도 실패 (포지션 유지)",
                    message=f"{result.symbol} 매도 주문이 실패했습니다. 포지션은 유지됩니다.",
                    level="warning",
                    cooldown_seconds=300,
                )
                return

            pos = self.positions.get(result.symbol)
            if pos:
                original_qty = int(pos.quantity)
                filled_qty = max(0, min(int(result.quantity or 0), original_qty))
                if filled_qty <= 0:
                    return
                sell_price = result.price
                if sell_price <= 0:
                    cached = self._quotes_cache.get(result.symbol)
                    sell_price = cached.current_price if cached else pos.buy_price

                gross_pnl = (sell_price - pos.buy_price) * filled_qty
                sell_notional = sell_price * filled_qty
                sell_fee = self._calc_commission_cost(sell_notional)
                sell_tax_slippage = self._calc_sell_tax_slippage_cost(sell_notional)
                net_pnl = gross_pnl - sell_fee - sell_tax_slippage

                self.daily_pnl.realized_gross_pnl += gross_pnl
                self.daily_pnl.realized_net_pnl += net_pnl
                self.daily_pnl.fees_paid += sell_fee
                self.daily_pnl.taxes_paid += sell_tax_slippage
                self.daily_pnl.trade_count += 1
                if net_pnl > 0:
                    self.daily_pnl.win_count += 1
                    self.daily_pnl.winning_net_pnl_sum += net_pnl
                    self.daily_pnl.largest_win_net = max(self.daily_pnl.largest_win_net, net_pnl)
                elif net_pnl < 0:
                    self.daily_pnl.loss_count += 1
                    self.daily_pnl.losing_net_pnl_sum += net_pnl
                    self.daily_pnl.largest_loss_net = min(self.daily_pnl.largest_loss_net, net_pnl)
                else:
                    self.daily_pnl.breakeven_count += 1

                is_partial_exit = filled_qty < original_qty

                if (
                    net_pnl < 0
                    and pos.regime_label == "neutral"
                    and result.symbol not in self._inverse_symbols
                    and not is_partial_exit
                ):
                    self._neutral_loss_count_today += 1
                    self._neutral_last_loss_at = self._now()
                    logger.info(
                        "중립장 손실 카운트 증가: %d/%d (%s)",
                        self._neutral_loss_count_today,
                        max(1, int(self.cfg.neutral_max_losses_per_day)),
                        result.symbol,
                    )

                if net_pnl < 0 and not is_partial_exit:
                    if pos.entry_strategy_name == "bull_breakout_strategy":
                        self._bull_loss_count_today += 1
                        self._bull_last_loss_at = self._now()
                        logger.info(
                            "bull 손실 카운트 증가: %d회 (%s)",
                            self._bull_loss_count_today,
                            result.symbol,
                        )
                    self._apply_loss_cooldowns(result.symbol, pos.entry_strategy_name)

                self._sell_cooldown[result.symbol] = self._now()

                if is_partial_exit:
                    remaining_qty = original_qty - filled_qty
                    pos.quantity = remaining_qty
                    pos.invested_amount = max(0, pos.buy_price * remaining_qty)
                    pos.partial_exit_done = True
                else:
                    self.positions.pop(result.symbol, None)

                tag = "[INV] " if result.symbol in self._inverse_symbols else ""
                if is_partial_exit:
                    logger.info(
                        "%s부분매도 체결: %s %d주 @ %s원 "
                        "(총손익: %s원, 순손익: %s원, 누적순손익: %s원, 잔여 %d주, "
                        "strategy_name=%s, setup_name=%s, regime_label=%s, entry_grade=%s)",
                        tag,
                        result.symbol,
                        filled_qty,
                        f"{sell_price:,}",
                        f"{gross_pnl:,}",
                        f"{net_pnl:,}",
                        f"{self.daily_pnl.realized_net_pnl:,}",
                        pos.quantity,
                        pos.entry_strategy_name or "-",
                        pos.entry_setup_name or "-",
                        pos.regime_label or "-",
                        pos.entry_grade or "-",
                    )
                else:
                    logger.info(
                        "%s매도 체결: %s %d주 @ %s원 "
                        "(총손익: %s원, 순손익: %s원, 누적순손익: %s원, "
                        "strategy_name=%s, setup_name=%s, regime_label=%s, entry_grade=%s)",
                        tag,
                        result.symbol,
                        filled_qty,
                        f"{sell_price:,}",
                        f"{gross_pnl:,}",
                        f"{net_pnl:,}",
                        f"{self.daily_pnl.realized_net_pnl:,}",
                        pos.entry_strategy_name or "-",
                        pos.entry_setup_name or "-",
                        pos.regime_label or "-",
                        pos.entry_grade or "-",
                    )
                alert_key_suffix = self._now().strftime("%H%M%S%f")
                self._alerts.send(
                    event_key=f"{'partial_' if is_partial_exit else ''}sell_fill_{result.symbol}_{alert_key_suffix}",
                    title="부분매도 체결" if is_partial_exit else "매도 체결",
                    message=(
                        f"{result.symbol} {filled_qty}주 @ {sell_price:,}원\n"
                        f"순손익 {net_pnl:,}원, 누적순손익 {self.daily_pnl.realized_net_pnl:,}원\n"
                        f"전략 {pos.entry_strategy_name or '-'} / "
                        f"셋업 {pos.entry_setup_name or '-'} / "
                        f"레짐 {pos.regime_label or '-'}"
                    ),
                    level="info" if net_pnl >= 0 else "warning",
                    cooldown_seconds=0,
                )

                self._save_daily_state()
                if not self._hard_stop_bypass_for_day:
                    self._evaluate_daily_breakers(liquidate=False)

    def should_continue(self) -> bool:
        if self._halted and not self.positions:
            return False
        return True

    def _base_max_position_count(self) -> int:
        """기본 동시 보유 가능한 최대 종목 수를 계산한다."""
        if self.cfg.max_position_count > 0:
            return self.cfg.max_position_count

        per_stock = max(1, self.cfg.per_stock_amount)
        auto_count = max(1, self.cfg.seed_money // per_stock)
        return auto_count

    def _resolve_profile_max_position_count(self, profile_name: str) -> int:
        base_count = self._base_max_position_count()
        override = None
        if profile_name == "bull":
            override = self.cfg.bull_max_position_count
        elif profile_name == "soft_bear":
            override = (
                self.cfg.soft_bear_max_position_count
                if self.cfg.soft_bear_max_position_count is not None
                else self.cfg.bear_max_position_count
            )
        elif profile_name == "bear":
            override = self.cfg.bear_max_position_count
        elif profile_name == "neutral":
            override = self.cfg.neutral_max_position_count

        if profile_name == "soft_bear" and self._soft_bear_strong_leader_lane_active():
            return max(0, int(self.cfg.soft_bear_strong_leader_max_positions))
        if override is None:
            return base_count
        return max(0, int(override))

    @staticmethod
    def _clamp_ratio(value: Optional[float], fallback: float) -> float:
        raw = fallback if value is None else value
        return max(0.05, min(1.0, float(raw)))

    def _resolve_profile_capital_utilization_pct(self, profile_name: str) -> float:
        override = None
        if profile_name == "bull":
            override = self.cfg.bull_capital_utilization_pct
        elif profile_name == "neutral":
            override = self.cfg.neutral_capital_utilization_pct
        elif profile_name == "soft_bear":
            override = self.cfg.soft_bear_capital_utilization_pct
        elif profile_name == "bear":
            override = self.cfg.bear_capital_utilization_pct
        return self._clamp_ratio(override, self.cfg.capital_utilization_pct)

    def _resolve_profile_max_single_position_pct(self, profile_name: str) -> float:
        override = None
        if profile_name == "bull":
            override = self.cfg.bull_max_single_position_pct
        elif profile_name == "neutral":
            override = self.cfg.neutral_max_single_position_pct
        elif profile_name == "soft_bear":
            override = self.cfg.soft_bear_max_single_position_pct
        elif profile_name == "bear":
            override = self.cfg.bear_max_single_position_pct
        return self._clamp_ratio(override, self.cfg.max_single_position_pct)

    def _effective_max_position_count(self) -> int:
        """현재 레짐 기준의 동시 보유 가능한 최대 종목 수를 계산한다."""
        profile = self._build_regime_profile()
        return int(profile.get("max_position_count", self._base_max_position_count()))

    def _resolve_regime_profile_name(self, bear_score: Optional[int] = None) -> str:
        if not self.cfg.enable_regime_adaptive:
            return "static"
        score = self._bear_score if bear_score is None else bear_score
        if score <= 0:
            return "bull"
        if score == 1 and (self._strong_bull_override_active or self._index_support_bull_bias_active):
            return "bull"
        if score == 1:
            return "neutral"
        if score == 2:
            return "soft_bear"
        if score >= 3:
            return "bear"
        return "neutral"

    def _is_early_session_guard_active(self) -> bool:
        if not self.cfg.enable_early_session_guard or self._session_start_at is None:
            return False
        elapsed = (self._now() - self._session_start_at).total_seconds()
        return elapsed <= self.cfg.early_session_guard_minutes * 60

    @staticmethod
    def _parse_hhmm_to_minutes(raw: str) -> Optional[int]:
        text = raw.strip()
        parts = text.split(":")
        if len(parts) != 2:
            return None
        try:
            hour = int(parts[0])
            minute = int(parts[1])
        except ValueError:
            return None
        if hour < 0 or hour > 23 or minute < 0 or minute > 59:
            return None
        return hour * 60 + minute

    def _parse_entry_block_windows(self, raw_windows: List[str]) -> List[tuple[int, int, str]]:
        parsed: List[tuple[int, int, str]] = []
        for raw in raw_windows or []:
            text = str(raw).strip()
            if not text:
                continue
            if "-" not in text:
                logger.warning("신규 진입 차단 시간 형식 오류(무시): %s", text)
                continue
            start_raw, end_raw = text.split("-", 1)
            start_min = self._parse_hhmm_to_minutes(start_raw)
            end_min = self._parse_hhmm_to_minutes(end_raw)
            if start_min is None or end_min is None:
                logger.warning("신규 진입 차단 시간 파싱 실패(무시): %s", text)
                continue
            if start_min >= end_min:
                logger.warning("신규 진입 차단 시간 범위 오류(무시): %s", text)
                continue
            parsed.append((start_min, end_min, f"{start_raw.strip()}-{end_raw.strip()}"))
        parsed.sort(key=lambda item: item[0])
        return parsed

    def _is_new_entry_window_blocked(self, now: datetime) -> bool:
        if not self._entry_block_windows:
            return False
        current_minute = now.hour * 60 + now.minute
        for start_min, end_min, label in self._entry_block_windows:
            if start_min <= current_minute < end_min:
                if (
                    self.cfg.enable_dynamic_entry_block_windows
                    and self._bear_score >= self.cfg.dynamic_entry_block_disable_bear_score
                ):
                    bypass_key = f"{now.strftime('%Y-%m-%d %H:%M')}-{label}"
                    if self._last_entry_block_bypass_log_key != bypass_key:
                        logger.info(
                            "신규 진입 차단 시간대 자동해제(%s): 현재 %02d:%02d, 약세점수=%d",
                            label,
                            now.hour,
                            now.minute,
                            self._bear_score,
                        )
                        self._last_entry_block_bypass_log_key = bypass_key
                    return False
                log_key = f"{now.strftime('%Y-%m-%d %H:%M')}-{label}"
                if self._last_entry_block_log_key != log_key:
                    logger.info(
                        "신규 진입 차단 시간대(%s): 현재 %02d:%02d",
                        label,
                        now.hour,
                        now.minute,
                    )
                    self._last_entry_block_log_key = log_key
                return True
        return False

    def _log_entry_filter_once_per_minute(
        self,
        symbol: str,
        reason: str,
        message: str,
        *args,
    ):
        minute_key = self._now().strftime("%Y-%m-%d %H:%M")
        key = f"{symbol}:{reason}"
        if self._entry_filter_log_cache.get(key) == minute_key:
            return
        self._entry_filter_log_cache[key] = minute_key
        logger.info(message, *args)

    def _build_regime_profile(self, profile_name: Optional[str] = None) -> dict:
        """현재 레짐 또는 특정 레짐을 기준으로 실시간 프로파일 값을 만든다."""
        is_preview = profile_name is not None

        if not self.cfg.enable_regime_adaptive:
            profile = {
                "max_position_count": self._base_max_position_count(),
                "min_change_rate": self.cfg.min_change_rate,
                "bullish_min_change_rate": self.cfg.bullish_min_change_rate,
                "min_momentum_score": self.cfg.min_momentum_score,
                "bullish_min_momentum_score": self.cfg.bullish_min_momentum_score,
                "volume_spike_ratio": self.cfg.volume_spike_ratio,
                "volume_spike_ratio_min": self.cfg.volume_spike_ratio_min,
                "volume_spike_abs_min": self.cfg.volume_spike_abs_min,
                "take_profit_pct": self.cfg.take_profit_pct,
                "per_position_stop_loss": -int(self.cfg.long_stop_loss_cap_amount),
                "trailing_stop_pct": self.cfg.trailing_stop_pct,
                "trailing_stop_activation_gain_pct": self.cfg.trailing_stop_activation_gain_pct,
                "max_position_holding_minutes": self.cfg.max_position_holding_minutes,
                "cooldown_seconds": self.cfg.cooldown_seconds,
                "loss_trade_cooldown_seconds": self.cfg.loss_trade_cooldown_seconds,
                "expected_move_pct": self.cfg.expected_move_pct,
                "min_expected_net_profit": self.cfg.min_expected_net_profit,
                "min_expected_rr_ratio": self.cfg.min_expected_rr_ratio,
                "per_stock_alloc_scale": 1.0,
                "max_stock_alloc_scale": 1.0,
                "capital_utilization_pct": self._resolve_profile_capital_utilization_pct("static"),
                "max_single_position_pct": self._resolve_profile_max_single_position_pct("static"),
                "daily_loss_limit": self.cfg.daily_loss_limit,
                "daily_profit_target": self.cfg.daily_profit_target,
                "daily_total_loss_limit": (
                    self.cfg.daily_total_loss_limit
                    if self.cfg.daily_total_loss_limit is not None
                    else self.cfg.daily_loss_limit
                ),
                "min_bear_score_for_new_long": self.cfg.min_bear_score_for_new_long,
                "bear_market_entry_score": self.cfg.bear_market_entry_score,
                "inverse_take_profit_pct": self.cfg.inverse_take_profit_pct,
                "inverse_stop_loss_pct": self.cfg.inverse_stop_loss_pct,
                "inverse_trailing_stop_pct": self.cfg.inverse_trailing_stop_pct,
                "inverse_trailing_stop_activation_gain_pct": self.cfg.inverse_trailing_stop_activation_gain_pct,
                "inverse_max_hold_minutes": self.cfg.inverse_max_hold_minutes,
                "inverse_max_positions": self.cfg.inverse_max_positions,
                "inverse_min_change_rate": self.cfg.inverse_min_change_rate,
                "inverse_min_momentum": self.cfg.inverse_min_momentum,
                "inverse_min_bear_score": self.cfg.inverse_min_bear_score,
            }
            if not is_preview:
                self._regime_profile_name = "static"
            return profile

        if profile_name is None:
            profile_name = self._resolve_regime_profile_name()
            self._regime_profile_name = profile_name
        else:
            if profile_name == "auto":
                profile_name = self._resolve_regime_profile_name()
        daily_total_loss_limit = (
            self.cfg.daily_total_loss_limit
            if self.cfg.daily_total_loss_limit is not None
            else self.cfg.daily_loss_limit
        )
        soft_inverse_max_positions = self.cfg.soft_bear_inverse_max_positions
        if soft_inverse_max_positions is None:
            soft_inverse_max_positions = min(self.cfg.inverse_max_positions, 1)

        base_profile = {
            "daily_loss_limit": int(self.cfg.daily_loss_limit),
            "daily_profit_target": int(self.cfg.daily_profit_target),
            "daily_total_loss_limit": int(daily_total_loss_limit),
            "min_bear_score_for_new_long": self.cfg.min_bear_score_for_new_long,
            "bear_market_entry_score": self.cfg.bear_market_entry_score,
            "inverse_take_profit_pct": 0.9,
            "inverse_stop_loss_pct": -0.6,
            "inverse_trailing_stop_pct": -0.3,
            "inverse_trailing_stop_activation_gain_pct": 0.5,
            "inverse_max_hold_minutes": max(30, self.cfg.inverse_max_hold_minutes),
        }

        profiles = {
            "bull": {
                "max_position_count": self._resolve_profile_max_position_count("bull"),
                "min_change_rate": max(0.45, self.cfg.bullish_min_change_rate),
                "bullish_min_change_rate": max(0.45, self.cfg.bullish_min_change_rate),
                "min_momentum_score": max(2.2, self.cfg.min_momentum_score * 0.8),
                "bullish_min_momentum_score": max(
                    self.cfg.bullish_min_momentum_score_floor,
                    self.cfg.bullish_min_momentum_score,
                ),
                "volume_spike_ratio": max(1.0, self.cfg.volume_spike_ratio * 0.72),
                "volume_spike_ratio_min": max(0.85, self.cfg.volume_spike_ratio_min * 0.82),
                "volume_spike_abs_min": int(max(1_000, self.cfg.volume_spike_abs_min * 0.60)),
                "take_profit_pct": 1.6,
                "per_position_stop_loss": -int(self.cfg.long_stop_loss_cap_amount),
                "trailing_stop_pct": -0.55,
                "trailing_stop_activation_gain_pct": 1.0,
                "max_position_holding_minutes": max(45, int(self.cfg.max_position_holding_minutes * 1.3)),
                "cooldown_seconds": max(120, int(self.cfg.cooldown_seconds * 0.7)),
                "loss_trade_cooldown_seconds": max(180, int(self.cfg.loss_trade_cooldown_seconds * 0.85)),
                "expected_move_pct": 1.6,
                "min_expected_net_profit": max(120, int(self.cfg.min_expected_net_profit * 0.75)),
                "min_expected_rr_ratio": max(0.65, self.cfg.min_expected_rr_ratio),
                "per_stock_alloc_scale": 1.0,
                "max_stock_alloc_scale": 1.0,
                "capital_utilization_pct": self._resolve_profile_capital_utilization_pct("bull"),
                "max_single_position_pct": self._resolve_profile_max_single_position_pct("bull"),
                "inverse_max_positions": self.cfg.inverse_max_positions,
                "inverse_min_change_rate": self.cfg.inverse_min_change_rate,
                "inverse_min_momentum": self.cfg.inverse_min_momentum,
                "inverse_min_bear_score": max(3, self.cfg.inverse_min_bear_score),
            },
            "neutral": {
                "max_position_count": self._resolve_profile_max_position_count("neutral"),
                "min_change_rate": max(0.35, self.cfg.min_change_rate * 0.45),
                "bullish_min_change_rate": max(0.45, self.cfg.bullish_min_change_rate),
                "min_momentum_score": max(1.8, self.cfg.min_momentum_score * 0.58),
                "bullish_min_momentum_score": max(2.2, self.cfg.bullish_min_momentum_score * 0.85),
                "volume_spike_ratio": max(0.95, self.cfg.volume_spike_ratio * 0.68),
                "volume_spike_ratio_min": max(0.90, self.cfg.volume_spike_ratio_min * 0.80),
                "volume_spike_abs_min": int(max(1_500, self.cfg.volume_spike_abs_min * 0.60)),
                "take_profit_pct": 1.0,
                "per_position_stop_loss": -int(self.cfg.long_stop_loss_cap_amount),
                "trailing_stop_pct": -0.35,
                "trailing_stop_activation_gain_pct": 0.7,
                "max_position_holding_minutes": max(18, int(self.cfg.max_position_holding_minutes * 0.65)),
                "cooldown_seconds": max(180, int(self.cfg.cooldown_seconds * 0.90)),
                "loss_trade_cooldown_seconds": max(240, int(self.cfg.loss_trade_cooldown_seconds * 1.05)),
                "expected_move_pct": 1.0,
                "min_expected_net_profit": max(80, int(self.cfg.min_expected_net_profit * 0.45)),
                "min_expected_rr_ratio": max(0.45, self.cfg.min_expected_rr_ratio * 0.8),
                "per_stock_alloc_scale": 0.9,
                "max_stock_alloc_scale": 0.9,
                "capital_utilization_pct": self._resolve_profile_capital_utilization_pct("neutral"),
                "max_single_position_pct": self._resolve_profile_max_single_position_pct("neutral"),
                "inverse_max_positions": 0,
                "inverse_min_change_rate": self.cfg.inverse_min_change_rate,
                "inverse_min_momentum": self.cfg.inverse_min_momentum,
                "inverse_min_bear_score": max(3, self.cfg.inverse_min_bear_score),
            },
            "soft_bear": {
                "max_position_count": self._resolve_profile_max_position_count("soft_bear"),
                "min_change_rate": max(0.50, self.cfg.min_change_rate),
                "bullish_min_change_rate": max(0.50, self.cfg.bullish_min_change_rate),
                "min_momentum_score": max(2.2, self.cfg.min_momentum_score * 0.75),
                "bullish_min_momentum_score": max(2.4, self.cfg.bullish_min_momentum_score * 0.9),
                "volume_spike_ratio": max(1.0, self.cfg.volume_spike_ratio * 0.72),
                "volume_spike_ratio_min": max(0.95, self.cfg.volume_spike_ratio_min * 0.85),
                "volume_spike_abs_min": int(max(2_000, self.cfg.volume_spike_abs_min * 0.72)),
                "take_profit_pct": 0.9,
                "per_position_stop_loss": -int(self.cfg.long_stop_loss_cap_amount),
                "trailing_stop_pct": -0.30,
                "trailing_stop_activation_gain_pct": 0.6,
                "max_position_holding_minutes": max(12, int(self.cfg.max_position_holding_minutes * 0.45)),
                "cooldown_seconds": max(240, int(self.cfg.cooldown_seconds * 1.05)),
                "loss_trade_cooldown_seconds": max(300, int(self.cfg.loss_trade_cooldown_seconds * 1.1)),
                "expected_move_pct": 0.9,
                "min_expected_net_profit": max(60, int(self.cfg.min_expected_net_profit * 0.35)),
                "min_expected_rr_ratio": max(0.40, self.cfg.min_expected_rr_ratio * 0.7),
                "per_stock_alloc_scale": 0.7,
                "max_stock_alloc_scale": 0.7,
                "capital_utilization_pct": self._resolve_profile_capital_utilization_pct("soft_bear"),
                "max_single_position_pct": self._resolve_profile_max_single_position_pct("soft_bear"),
                "inverse_max_positions": max(0, int(soft_inverse_max_positions)),
                "inverse_min_change_rate": self.cfg.soft_bear_inverse_min_change_rate,
                "inverse_min_momentum": self.cfg.soft_bear_inverse_min_momentum,
                "inverse_min_bear_score": max(2, self.cfg.bearish_threshold),
            },
            "bear": {
                "max_position_count": self._resolve_profile_max_position_count("bear"),
                "min_change_rate": max(0.60, self.cfg.min_change_rate),
                "bullish_min_change_rate": max(0.60, self.cfg.bullish_min_change_rate),
                "min_momentum_score": max(2.4, self.cfg.min_momentum_score * 0.78),
                "bullish_min_momentum_score": max(2.6, self.cfg.bullish_min_momentum_score * 0.92),
                "volume_spike_ratio": max(1.1, self.cfg.volume_spike_ratio * 0.82),
                "volume_spike_ratio_min": max(1.0, self.cfg.volume_spike_ratio_min * 0.90),
                "volume_spike_abs_min": int(max(2_400, self.cfg.volume_spike_abs_min * 0.82)),
                "take_profit_pct": 0.9,
                "per_position_stop_loss": -int(self.cfg.long_stop_loss_cap_amount),
                "trailing_stop_pct": -0.30,
                "trailing_stop_activation_gain_pct": 0.6,
                "max_position_holding_minutes": max(10, int(self.cfg.max_position_holding_minutes * 0.40)),
                "cooldown_seconds": max(300, int(self.cfg.cooldown_seconds * 1.10)),
                "loss_trade_cooldown_seconds": max(300, int(self.cfg.loss_trade_cooldown_seconds * 1.15)),
                "expected_move_pct": 0.9,
                "min_expected_net_profit": max(60, int(self.cfg.min_expected_net_profit * 0.35)),
                "min_expected_rr_ratio": max(0.40, self.cfg.min_expected_rr_ratio * 0.7),
                "per_stock_alloc_scale": 0.6,
                "max_stock_alloc_scale": 0.6,
                "capital_utilization_pct": self._resolve_profile_capital_utilization_pct("bear"),
                "max_single_position_pct": self._resolve_profile_max_single_position_pct("bear"),
                "inverse_max_positions": self.cfg.inverse_max_positions,
                "inverse_min_change_rate": self.cfg.inverse_min_change_rate,
                "inverse_min_momentum": self.cfg.inverse_min_momentum,
                "inverse_min_bear_score": max(3, self.cfg.inverse_min_bear_score),
            },
        }
        profile = dict(base_profile)
        profile.update(profiles.get(profile_name, profiles["neutral"]))
        return profile

    def _get_regime_value(self, key: str, default):
        profile = self._build_regime_profile()
        return profile.get(key, default)

    def _regime_min_change_rate(self) -> float:
        return float(self._get_regime_value("min_change_rate", self.cfg.min_change_rate))

    def _regime_bullish_min_change_rate(self) -> float:
        return float(self._get_regime_value("bullish_min_change_rate", self.cfg.bullish_min_change_rate))

    def _regime_min_momentum_score(self) -> float:
        return float(self._get_regime_value("min_momentum_score", self.cfg.min_momentum_score))

    def _regime_bullish_min_momentum_score(self) -> float:
        return float(self._get_regime_value("bullish_min_momentum_score", self.cfg.bullish_min_momentum_score))

    def _regime_take_profit_pct(self) -> float:
        return float(self._get_regime_value("take_profit_pct", self.cfg.take_profit_pct))

    def _regime_per_position_stop_loss(self) -> int:
        return int(self._get_regime_value("per_position_stop_loss", self.cfg.per_position_stop_loss))

    def _regime_trailing_stop_pct(self) -> float:
        return float(self._get_regime_value("trailing_stop_pct", self.cfg.trailing_stop_pct))

    def _regime_trailing_stop_activation_gain_pct(self) -> float:
        return float(
            self._get_regime_value(
                "trailing_stop_activation_gain_pct",
                self.cfg.trailing_stop_activation_gain_pct,
            )
        )

    def _regime_max_holding_minutes(self) -> int:
        return int(self._get_regime_value("max_position_holding_minutes", self.cfg.max_position_holding_minutes))

    def _regime_cooldown_seconds(self) -> int:
        return max(10, int(self._get_regime_value("cooldown_seconds", self.cfg.cooldown_seconds)))

    def _regime_loss_cooldown_seconds(self) -> int:
        return max(
            10,
            int(self._get_regime_value("loss_trade_cooldown_seconds", self.cfg.loss_trade_cooldown_seconds)),
        )

    def _regime_expected_move_pct(self) -> float:
        return float(self._get_regime_value("expected_move_pct", self.cfg.expected_move_pct))

    def _regime_min_expected_net_profit(self) -> int:
        return int(self._get_regime_value("min_expected_net_profit", self.cfg.min_expected_net_profit))

    def _regime_min_expected_rr_ratio(self) -> float:
        return float(self._get_regime_value("min_expected_rr_ratio", self.cfg.min_expected_rr_ratio))

    def _regime_inverse_take_profit_pct(self) -> float:
        return float(self._get_regime_value("inverse_take_profit_pct", self.cfg.inverse_take_profit_pct))

    def _regime_inverse_stop_loss_pct(self) -> float:
        return float(self._get_regime_value("inverse_stop_loss_pct", self.cfg.inverse_stop_loss_pct))

    def _regime_inverse_trailing_stop_pct(self) -> float:
        return float(self._get_regime_value("inverse_trailing_stop_pct", self.cfg.inverse_trailing_stop_pct))

    def _regime_inverse_trailing_stop_activation_gain_pct(self) -> float:
        return float(
            self._get_regime_value(
                "inverse_trailing_stop_activation_gain_pct",
                self.cfg.inverse_trailing_stop_activation_gain_pct,
            )
        )

    def _regime_inverse_max_hold_minutes(self) -> int:
        return int(self._get_regime_value("inverse_max_hold_minutes", self.cfg.inverse_max_hold_minutes))

    def _regime_inverse_max_positions(self) -> int:
        return max(0, int(self._get_regime_value("inverse_max_positions", self.cfg.inverse_max_positions)))

    def _regime_inverse_min_change_rate(self) -> float:
        return float(self._get_regime_value("inverse_min_change_rate", self.cfg.inverse_min_change_rate))

    def _regime_inverse_min_momentum(self) -> float:
        return float(self._get_regime_value("inverse_min_momentum", self.cfg.inverse_min_momentum))

    def _regime_inverse_min_bear_score(self) -> int:
        return int(self._get_regime_value("inverse_min_bear_score", self.cfg.inverse_min_bear_score))

    def _regime_per_stock_alloc_scale(self) -> float:
        return float(self._get_regime_value("per_stock_alloc_scale", 1.0))

    def _regime_max_stock_alloc_scale(self) -> float:
        return float(self._get_regime_value("max_stock_alloc_scale", 1.0))

    def _regime_capital_utilization_pct(self) -> float:
        return self._clamp_ratio(
            self._get_regime_value("capital_utilization_pct", self.cfg.capital_utilization_pct),
            self.cfg.capital_utilization_pct,
        )

    def _regime_max_single_position_pct(self) -> float:
        return self._clamp_ratio(
            self._get_regime_value("max_single_position_pct", self.cfg.max_single_position_pct),
            self.cfg.max_single_position_pct,
        )

    def _allocation_capital_base(self) -> int:
        realized_net = int(self.daily_pnl.realized_net_pnl)
        return max(0, self.cfg.seed_money + realized_net)

    def _regime_target_total_exposure_amount(self) -> int:
        capital_base = self._allocation_capital_base()
        return int(capital_base * self._regime_capital_utilization_pct() * self._risk_exposure_scale())

    def _regime_max_single_position_amount(self) -> int:
        capital_base = self._allocation_capital_base()
        return int(capital_base * self._regime_max_single_position_pct() * self._risk_exposure_scale())

    def _regime_volume_spike_ratio(self) -> float:
        return float(self._get_regime_value("volume_spike_ratio", self.cfg.volume_spike_ratio))

    def _regime_volume_spike_ratio_min(self) -> float:
        return float(self._get_regime_value("volume_spike_ratio_min", self.cfg.volume_spike_ratio_min))

    def _regime_volume_spike_abs_min(self) -> int:
        return int(self._get_regime_value("volume_spike_abs_min", self.cfg.volume_spike_abs_min))

    def _regime_bear_score_for_new_long(self) -> int:
        return int(self._get_regime_value("min_bear_score_for_new_long", self.cfg.min_bear_score_for_new_long))

    def _regime_bear_market_entry_score(self) -> float:
        return float(self._get_regime_value("bear_market_entry_score", self.cfg.bear_market_entry_score))

    def sync_positions_from_account(self, account_positions: List[Position]):
        """계좌 보유를 전략 포지션 상태와 동기화한다.

        장중 재시작 시 메모리 상태가 초기화되더라도 실제 보유 기준으로
        포지션 한도/추가매수 판단이 일관되게 동작하도록 한다.
        """
        synced: Dict[str, PositionState] = {}
        now = self._now()

        for p in account_positions or []:
            qty = int(p.quantity or 0)
            if qty <= 0:
                continue

            avg_price = int(round(float(p.avg_price or 0)))
            if avg_price <= 0:
                continue

            current_price = int(p.current_price or 0)
            snapshot = self._loaded_position_meta.get(p.symbol, {})
            restored_buy_time = self._parse_state_datetime(snapshot.get("buy_time"))
            if restored_buy_time is None:
                restored_buy_time = now
            restored_high = int(snapshot.get("high_since_buy", 0) or 0)
            restored_invested = int(snapshot.get("invested_amount", 0) or 0)
            synced[p.symbol] = PositionState(
                symbol=p.symbol,
                buy_price=avg_price,
                quantity=qty,
                invested_amount=restored_invested if restored_invested > 0 else avg_price * qty,
                buy_time=restored_buy_time,
                high_since_buy=max(
                    avg_price,
                    current_price if current_price > 0 else avg_price,
                    restored_high,
                ),
                is_restored=True,
                restored_at=now,
                entry_strategy_name=str(snapshot.get("entry_strategy_name", "") or ""),
                entry_setup_name=str(snapshot.get("entry_setup_name", "") or ""),
                entry_reason=str(snapshot.get("entry_reason", "") or ""),
                regime_label=str(snapshot.get("regime_label", "") or ""),
                bear_score=int(snapshot.get("bear_score", 0) or 0),
                planned_risk_stage=str(snapshot.get("planned_risk_stage", "") or ""),
                entry_grade=str(snapshot.get("entry_grade", "") or ""),
                partial_exit_done=bool(snapshot.get("partial_exit_done", False)),
            )

        self.positions = synced
        self._startup_rebalance_active = len(self.positions) >= self._effective_max_position_count()
        self._startup_rebalance_ticks = (
            self.cfg.startup_full_position_recheck_ticks if self._startup_rebalance_active else 0
        )
        logger.info(
            "계좌 보유 동기화 완료: %d종목 (최대 허용 %d, 재시작복구=%s, %d틱)",
            len(self.positions),
            self._effective_max_position_count(),
            self._startup_rebalance_active,
            self._startup_rebalance_ticks,
        )

    # --- 내부 로직 ---

    def _build_pool(self):
        """종목 풀을 구성한다."""
        if self._pool_override:
            self._pool = list(self._pool_override)
            # 인버스 ETF가 override에 없으면 추가
            if self.cfg.inverse_enabled:
                for sym in self.cfg.inverse_etfs:
                    if sym not in self._pool:
                        self._pool.append(sym)
            self._last_pool_refresh = self._now()
            return

        pool = set(self.cfg.static_watchlist)
        appeared: set[str] = set()
        direct_dynamic: set[str] = set()
        strong_leader_snapshot: Dict[str, dict] = {}

        # 인버스 ETF 추가
        if self.cfg.inverse_enabled:
            for sym in self.cfg.inverse_etfs:
                pool.add(sym)

        if self.market_data:
            try:
                rising = self.market_data.get_fluctuation_ranking(
                    count=max(self.cfg.dynamic_pool_size, self.cfg.dynamic_pool_ranking_fetch_count),
                    min_change_rate=self.cfg.min_change_rate,
                    max_change_rate=self.cfg.max_change_rate,
                    min_price=self.cfg.min_price,
                    min_volume=self.cfg.min_volume,
                )
                by_rank = rising[: max(0, int(self.cfg.dynamic_pool_size))]
                by_turnover = sorted(
                    rising,
                    key=lambda item: (
                        self._ranking_trade_amount(item),
                        float(item.change_rate),
                        -int(item.rank or 0),
                    ),
                    reverse=True,
                )[: max(0, int(self.cfg.dynamic_pool_turnover_slots))]
                cache_leaders = sorted(
                    [
                        quote
                        for quote in self._quotes_cache.values()
                        if (
                            quote.symbol not in self._inverse_symbols
                            and quote.current_price >= self.cfg.min_price
                            and quote.change_rate >= self.cfg.dynamic_pool_quote_min_change_rate
                            and quote.trade_amount > 0
                        )
                    ],
                    key=lambda quote: (
                        int(quote.trade_amount),
                        float(quote.change_rate),
                        int(quote.volume),
                    ),
                    reverse=True,
                )[: max(0, int(self.cfg.dynamic_pool_quote_trade_amount_slots))]
                for item in by_rank:
                    appeared.add(item.symbol)
                for item in by_turnover:
                    appeared.add(item.symbol)
                for quote in cache_leaders:
                    appeared.add(quote.symbol)
                for item in by_rank[: max(0, int(self.cfg.dynamic_pool_direct_rank_slots))]:
                    direct_dynamic.add(item.symbol)
                for item in by_turnover[: max(0, int(self.cfg.dynamic_pool_direct_turnover_slots))]:
                    direct_dynamic.add(item.symbol)
                for quote in cache_leaders[: max(0, int(self.cfg.dynamic_pool_direct_quote_leader_slots))]:
                    direct_dynamic.add(quote.symbol)
                for item in rising:
                    trade_amount = self._ranking_trade_amount(item)
                    if not self._is_dynamic_strong_leader_candidate(
                        symbol=item.symbol,
                        current_price=int(item.current_price),
                        change_rate=float(item.change_rate),
                        trade_amount=trade_amount,
                        rank=int(item.rank or 0),
                    ):
                        continue
                    strong_leader_snapshot[item.symbol] = {
                        "change_rate": float(item.change_rate),
                        "trade_amount": int(trade_amount),
                        "rank": int(item.rank or 0),
                    }
                    direct_dynamic.add(item.symbol)
                for quote in cache_leaders:
                    trade_amount = self._quote_trade_amount(quote)
                    if not self._is_dynamic_strong_leader_candidate(
                        symbol=quote.symbol,
                        current_price=int(quote.current_price),
                        change_rate=float(quote.change_rate),
                        trade_amount=trade_amount,
                        rank=None,
                    ):
                        continue
                    existing = strong_leader_snapshot.get(quote.symbol, {})
                    strong_leader_snapshot[quote.symbol] = {
                        "change_rate": max(float(existing.get("change_rate", 0.0)), float(quote.change_rate)),
                        "trade_amount": max(int(existing.get("trade_amount", 0)), int(trade_amount)),
                        "rank": int(existing.get("rank", 999)),
                    }
                    direct_dynamic.add(quote.symbol)
                logger.info(
                    "동적 풀 갱신: 등락률 %d개 + 거래대금 %d개 + 실시간 리더 %d개 (총 %d종목)",
                    len(by_rank),
                    len(by_turnover),
                    len(cache_leaders),
                    len(pool),
                )
            except Exception as e:
                logger.warning("등락률 순위 조회 실패, 정적 풀만 사용: %s", e)

        self._pool_build_epoch += 1
        if self.cfg.enable_pool_persistence_gate:
            self._record_pool_appearances(appeared)
            persistent = {
                sym
                for sym in appeared
                if self._is_pool_persistent(sym)
            }
            for sym in direct_dynamic:
                pool.add(sym)
            for sym in appeared:
                if sym in persistent:
                    pool.add(sym)
            if appeared:
                logger.info("풀 지속성 반영: %d개 동적 후보 중 %d개 채택",
                            len(appeared), len(persistent))
            if direct_dynamic:
                logger.info(
                    "강한 리더 즉시 편입: %d개 [%s]",
                    len(direct_dynamic),
                    self._format_symbol_sample(direct_dynamic),
                )
        else:
            pool.update(appeared)

        self._latest_direct_dynamic_symbols = set(direct_dynamic)
        self._latest_strong_leader_symbols = set(strong_leader_snapshot.keys())
        self._latest_strong_leader_snapshot = strong_leader_snapshot
        if strong_leader_snapshot:
            logger.info(
                "장중 강한 리더 후보: %d개 [%s]",
                len(strong_leader_snapshot),
                self._format_symbol_sample(strong_leader_snapshot.keys()),
            )
        self._pool = list(pool)[:55]  # 인버스 포함하여 여유 확보
        self._last_pool_refresh = self._now()

    def _record_pool_appearances(self, symbols: set[str]):
        if not self.cfg.enable_pool_persistence_gate:
            return

        max_window = max(1, self.cfg.momentum_pool_persistence_window)
        min_keep_epoch = self._pool_build_epoch - max_window + 1

        for dq in self._pool_appearance.values():
            while dq and dq[0] < min_keep_epoch:
                dq.popleft()
        for sym in list(self._pool_appearance.keys()):
            if not self._pool_appearance[sym]:
                self._pool_appearance.pop(sym, None)

        for sym in symbols:
            dq = self._pool_appearance.setdefault(sym, deque(maxlen=max_window))
            dq.append(self._pool_build_epoch)

    def _is_pool_persistent(self, symbol: str) -> bool:
        if not self.cfg.enable_pool_persistence_gate:
            return True
        if symbol in self.cfg.static_watchlist:
            return True
        dq = self._pool_appearance.get(symbol)
        if not dq:
            return False
        required = max(1, self.cfg.momentum_pool_min_appearances)
        return len(dq) >= required

    def _is_bullish_regime(self) -> bool:
        """약세점수 기반 완화 모드 판정."""
        return self._bear_score <= 0 or self._is_bull_bias_market()

    def _meets_leader_support_bull_bias_override(
        self,
        *,
        index_info: Optional[tuple[float, float, float]],
        quote_decline_ratio: Optional[float],
    ) -> bool:
        leaders = list(self._latest_strong_leader_snapshot.values())
        if len(leaders) < max(1, int(self.cfg.leader_support_bull_bias_min_count)):
            return False
        avg_change = sum(float(item.get("change_rate", 0.0)) for item in leaders) / len(leaders)
        avg_trade_amount = sum(int(item.get("trade_amount", 0)) for item in leaders) / len(leaders)
        if avg_change < float(self.cfg.leader_support_bull_bias_min_change_rate):
            return False
        if avg_trade_amount < float(self.cfg.leader_support_bull_bias_min_trade_amount):
            return False
        if (
            quote_decline_ratio is not None
            and quote_decline_ratio > float(self.cfg.leader_support_bull_bias_max_decliner_ratio)
        ):
            return False
        if index_info is None:
            return True
        current, ma20, ma5 = index_info
        if current <= 0 or ma20 <= 0 or ma5 <= 0:
            return False
        return current >= ma20 or current >= ma5

    def _meets_strong_bull_override(
        self,
        *,
        index_info: Optional[tuple[float, float, float]],
        quote_avg_change: Optional[float],
        quote_decline_ratio: Optional[float],
        quote_count: int,
    ) -> bool:
        if index_info is None:
            return False
        current, ma20, ma5 = index_info
        if current <= 0 or ma20 <= 0 or ma5 <= 0:
            return False
        if current <= ma20 or current <= ma5:
            return False
        gap_pct = ((current - ma20) / ma20) * 100 if ma20 > 0 else 0.0
        if gap_pct < float(self.cfg.strong_bull_override_index_gap_pct):
            return False
        if quote_count < max(1, int(self.cfg.strong_bull_override_min_quote_count)):
            return False
        if quote_avg_change is None or quote_avg_change < float(self.cfg.strong_bull_override_avg_change_rate_threshold):
            return False
        if quote_decline_ratio is None or quote_decline_ratio > float(self.cfg.strong_bull_override_max_decliner_ratio):
            return False
        return True

    def _meets_index_support_bull_bias_override(
        self,
        *,
        index_info: Optional[tuple[float, float, float]],
        quote_avg_change: Optional[float],
        quote_decline_ratio: Optional[float],
        quote_count: int,
    ) -> bool:
        if index_info is None:
            return False
        current, ma20, ma5 = index_info
        if current <= 0 or ma20 <= 0 or ma5 <= 0:
            return False
        if current <= ma20 or current <= ma5:
            return False
        gap_pct = ((current - ma20) / ma20) * 100 if ma20 > 0 else 0.0
        if gap_pct < float(self.cfg.index_support_bull_bias_index_gap_pct):
            return False
        if quote_count < max(1, int(self.cfg.index_support_bull_bias_min_quote_count)):
            return False
        if quote_avg_change is None or quote_avg_change < float(self.cfg.index_support_bull_bias_avg_change_rate_threshold):
            return False
        if quote_decline_ratio is None or quote_decline_ratio > float(self.cfg.index_support_bull_bias_max_decliner_ratio):
            return False
        return True

    def _is_bull_bias_market(self) -> bool:
        if self._leader_support_bull_bias_active:
            return True
        if self._index_support_bull_bias_active:
            return True
        if self._strong_bull_override_active:
            return True
        if self._bear_score != 1:
            return False
        index_info = self._cached_index_regime_info
        if index_info:
            current, ma20, ma5 = index_info
            if current <= 0 or current <= ma20 or current <= ma5:
                return False
        elif self.market_data is not None:
            return False

        active_quotes = self._active_pool_quotes(include_inverse=False)
        if len(active_quotes) < 4:
            return False

        avg_change = sum(item.change_rate for item in active_quotes) / len(active_quotes)
        declining_ratio = (
            sum(1 for item in active_quotes if item.change_rate < 0) / len(active_quotes)
        )
        return (
            avg_change >= float(self.cfg.bull_bias_avg_change_rate_threshold)
            and declining_ratio <= float(self.cfg.bull_bias_max_decliner_ratio)
        )

    def _check_market_regime(self, quotes: Optional[List[Quote]] = None, force: bool = False):
        """KOSPI + 실시간 후보군 추세를 결합해 약세 점수를 계산한다."""
        now = self._now()
        index_code = "0001"

        quote_estimate = (
            self._estimate_market_from_quotes(quotes)
            if quotes is not None else None
        )
        quote_score = quote_estimate[0] if quote_estimate is not None else None
        quote_avg_change = quote_estimate[1] if quote_estimate is not None else None
        quote_decline_ratio = quote_estimate[2] if quote_estimate is not None else None

        if (
            quotes is None
            and not force
            and self._last_regime_check_at is not None
            and (now - self._last_regime_check_at).total_seconds() < 120
        ):
            return

        if not self.market_data:
            self._index_support_bull_bias_active = False
            self._strong_bull_override_active = False
            fallback_score = 0 if quote_score is None else max(0, min(int(quote_score), 3))
            self._bear_score = fallback_score
            self._bear_market = fallback_score >= 2
            self._last_regime_check_at = now
            if quote_score is not None:
                logger.info(
                    "시장 레짐(백테스트): quote=%d(평균%.2f%%, 하락비율%.1f%%), 최종=%d",
                    quote_score,
                    quote_avg_change,
                    quote_decline_ratio * 100,
                    fallback_score,
                )
            return

        index_score = self._cached_index_regime_score
        index_info = self._cached_index_regime_info
        index_error = self._cached_index_regime_error
        should_refresh_index = (
            force
            or self._last_index_regime_check_at is None
            or (now - self._last_index_regime_check_at).total_seconds() >= 120
        )

        if should_refresh_index:
            previous_index_score = self._cached_index_regime_score
            previous_index_info = self._cached_index_regime_info
            index_score = 0
            index_info = None
            index_error = None
            try:
                end_date = now.strftime("%Y%m%d")
                start_date = (now - timedelta(days=45)).strftime("%Y%m%d")

                df = self.market_data.get_index_daily_prices(index_code, start_date, end_date)
                if df.empty or len(df) < 20:
                    index_info = None
                else:
                    close_col = None
                    for candidate in ("bstp_nmix_prpr", "bstp_nmix_oprc", "stck_clpr"):
                        if candidate in df.columns:
                            close_col = candidate
                            break

                    if close_col is None:
                        raise KeyError("일봉 close 컬럼 미검출")

                    close_series = pd.to_numeric(df[close_col], errors="coerce")
                    if "stck_bsop_date" in df.columns:
                        date_series = pd.to_numeric(df["stck_bsop_date"], errors="coerce")
                        closes = (
                            pd.DataFrame({"date": date_series, "close": close_series})
                            .dropna()
                            .sort_values("date")
                            ["close"]
                        )
                    else:
                        closes = close_series.dropna()

                    closes = closes[closes > 0]

                    if len(closes) < 20:
                        index_info = None
                    else:
                        score = 0

                        # 1. KOSPI < MA20: 중기 하락 추세
                        ma20 = closes.tail(20).mean()
                        current = closes.iloc[-1]
                        if current < ma20:
                            score += 1

                        # 2. MA5 < MA20: 단기 데드크로스
                        ma5 = closes.tail(5).mean()
                        if ma5 < ma20:
                            score += 1

                        # 3. MA10 < MA20: 중기 추세 약세 보강
                        ma10 = closes.tail(10).mean()
                        if ma10 < ma20:
                            score += 1

                        # 4. 3일 연속 하락
                        if len(closes) >= 4:
                            last3 = closes.iloc[-3:]
                            prev3 = closes.iloc[-4:-1]
                            if all(c < p for c, p in zip(last3, prev3)):
                                score += 1

                        if current <= 0 or ma20 <= 0 or ma5 <= 0:
                            raise ValueError("비정상 인덱스 응답(0 이하 값)")

                        index_score = score
                        index_info = (float(current), float(ma20), float(ma5))
                if (
                    index_info is None
                    and previous_index_info is not None
                    and all(float(value) > 0 for value in previous_index_info)
                ):
                    raise ValueError("비정상 인덱스 응답(유효 일봉 부족)")

                self._cached_index_regime_score = index_score
                self._cached_index_regime_info = index_info
                self._cached_index_regime_error = None
                self._last_index_regime_check_at = now
            except Exception as e:
                index_error = e
                self._cached_index_regime_error = e
                if (
                    previous_index_info is not None
                    and all(float(value) > 0 for value in previous_index_info)
                ):
                    index_score = previous_index_score or 0
                    index_info = previous_index_info
                    self._cached_index_regime_score = index_score
                    self._cached_index_regime_info = index_info
                    logger.warning("인덱스 일봉 비정상 응답 감지, 직전 캐시 유지: %s", e)
                else:
                    self._cached_index_regime_score = 0
                    self._cached_index_regime_info = None
                self._last_index_regime_check_at = now

        if index_score is None:
            index_score = 0

        index_weight = 0.7
        quote_weight = 0.3
        quote_score_value = quote_score or 0
        if quote_avg_change is None:
            quote_avg_change = 0.0
        if quote_decline_ratio is None:
            quote_decline_ratio = 0.0
        quote_count = (
            len([q for q in quotes if q.symbol not in self._inverse_symbols])
            if quotes is not None
            else len(self._active_pool_quotes(include_inverse=False))
        )

        raw_final_score = float(index_score)
        if quote_score is not None:
            raw_final_score = (
                index_score * index_weight
                + quote_score_value * quote_weight
            )
            # 후보군이 강세/약세로 급하게 치우쳐 있더라도 과잉 전환을 완화
            if index_score >= 3 and quote_score == 0:
                raw_final_score -= 1.0
            elif index_score <= 0 and quote_score >= 3:
                raw_final_score -= 0.5

        # 스무딩: 직전 판정이 있으면 30% 반영해 급격한 상태 역전 방지
        prev_score = self._bear_score if self._last_regime_check_at is not None else raw_final_score
        final_score = round((raw_final_score * 0.75) + (prev_score * 0.25))
        final_score = max(0, min(4, int(final_score)))

        index_support_bull_bias = self._meets_index_support_bull_bias_override(
            index_info=index_info,
            quote_avg_change=quote_avg_change,
            quote_decline_ratio=quote_decline_ratio,
            quote_count=quote_count,
        )
        leader_support_bull_bias = self._meets_leader_support_bull_bias_override(
            index_info=index_info,
            quote_decline_ratio=quote_decline_ratio,
        )
        strong_bull_override = self._meets_strong_bull_override(
            index_info=index_info,
            quote_avg_change=quote_avg_change,
            quote_decline_ratio=quote_decline_ratio,
            quote_count=quote_count,
        )
        if strong_bull_override and final_score >= 2:
            logger.info(
                "강세 오버라이드 적용: score %d -> 1 (KOSPI-MA20 %.2f%%, 평균등락 %.2f%%, 하락비율 %.1f%%, 표본 %d개)",
                final_score,
                ((index_info[0] - index_info[1]) / index_info[1]) * 100 if index_info[1] > 0 else 0.0,
                quote_avg_change,
                quote_decline_ratio * 100,
                quote_count,
            )
            final_score = 1
        elif index_support_bull_bias and final_score >= 2:
            logger.info(
                "지수 지지 bull-bias 적용: score %d -> 1 (KOSPI-MA20 %.2f%%, 평균등락 %.2f%%, 하락비율 %.1f%%, 표본 %d개)",
                final_score,
                ((index_info[0] - index_info[1]) / index_info[1]) * 100 if index_info[1] > 0 else 0.0,
                quote_avg_change,
                quote_decline_ratio * 100,
                quote_count,
            )
            final_score = 1
        elif leader_support_bull_bias and final_score >= 2:
            leader_count = len(self._latest_strong_leader_snapshot)
            avg_leader_change = (
                sum(float(item.get("change_rate", 0.0)) for item in self._latest_strong_leader_snapshot.values())
                / max(1, leader_count)
            )
            avg_leader_trade_amount = int(
                sum(int(item.get("trade_amount", 0)) for item in self._latest_strong_leader_snapshot.values())
                / max(1, leader_count)
            )
            logger.info(
                "강한 리더 bull-bias 적용: score %d -> 1 (리더 %d개, 평균등락 %.2f%%, 평균거래대금 %s원)",
                final_score,
                leader_count,
                avg_leader_change,
                f"{avg_leader_trade_amount:,}",
            )
            final_score = 1

        final_score = max(0, min(final_score, 3))
        self._bear_score = final_score
        self._bear_market = final_score >= 2
        self._leader_support_bull_bias_active = leader_support_bull_bias
        self._index_support_bull_bias_active = index_support_bull_bias
        self._strong_bull_override_active = strong_bull_override
        self._last_regime_check_at = now

        if index_info is not None and quote_score is not None:
            logger.info(
                "시장 레짐 계산: index=%d(가중 %.2f), quote=%d(평균%.2f%%, 하락비율%.1f%%), 최종=%d(이전=%s, 원시=%.2f), "
                "지수=%s, KOSPI: %.1f, MA20: %.1f, MA5: %.1f",
                index_score,
                index_weight,
                quote_score,
                quote_avg_change,
                quote_decline_ratio * 100,
                final_score,
                prev_score,
                raw_final_score,
                index_code,
                index_info[0],
                index_info[1],
                index_info[2],
            )
        elif index_info is not None:
            logger.info(
                "시장 레짐 계산: index=%d(가중 %.2f), quote=미사용, 최종=%d, 지수=%s, KOSPI: %.1f, MA20: %.1f, MA5: %.1f",
                index_score,
                raw_final_score,
                final_score,
                index_code,
                index_info[0],
                index_info[1],
                index_info[2],
            )
        else:
            if index_error is not None:
                logger.warning("시장 레짐 확인 실패(일봉): %s", index_error)
            logger.info(
                "시장 레짐: 약세점수=%d (index 데이터 미확보)",
                final_score,
            )

    def _cleanup_stale_entry_signals(self, now: datetime):
        if not self.cfg.enable_entry_confirmation:
            self._entry_signals.clear()
            return

        ttl = max(1, self.cfg.entry_confirmation_window_seconds)
        remove = [
            symbol
            for symbol, signal in self._entry_signals.items()
            if (now - signal.last_seen_at).total_seconds() > ttl
        ]
        for symbol in remove:
            self._entry_signals.pop(symbol, None)

    def _refresh_entry_signal_if_stale(self, symbol: str, now: datetime):
        if not self.cfg.enable_entry_confirmation:
            return
        signal = self._entry_signals.get(symbol)
        if not signal:
            return
        ttl = max(1, self.cfg.entry_confirmation_window_seconds)
        if (now - signal.last_seen_at).total_seconds() > ttl:
            self._entry_signals.pop(symbol, None)

    def _rank_long_entry_candidates(
        self,
        quotes: List[Quote],
    ) -> List[tuple[float, Quote]]:
        candidates: List[tuple[float, Quote]] = []
        profile_name = self._resolve_regime_profile_name()
        soft_bear_leader_lane = profile_name == "soft_bear" and self._soft_bear_strong_leader_lane_active()
        if profile_name == "bear":
            return candidates
        if profile_name == "soft_bear" and not soft_bear_leader_lane:
            return candidates
        is_opening_guard = self._is_early_session_guard_active()
        for q in quotes:
            if q.symbol in self.positions:
                continue
            if q.symbol in self._inverse_symbols:
                continue
            if q.current_price <= 0 or q.current_price < self.cfg.min_price:
                continue
            min_change_rate = self._regime_min_change_rate()
            if self._is_bullish_regime():
                min_change_rate = self._regime_bullish_min_change_rate()
            if soft_bear_leader_lane:
                min_change_rate = float(self.cfg.soft_bear_strong_leader_min_change_rate)
            if is_opening_guard and self._is_bullish_regime():
                min_change_rate += self.cfg.early_session_min_change_rate_boost
            if q.change_rate < min_change_rate:
                continue
            score = self._calc_momentum_score(q)
            min_score = self._regime_min_momentum_score()
            if self._is_bullish_regime():
                min_score = self._regime_bullish_min_momentum_score()
            if soft_bear_leader_lane:
                min_score = float(self.cfg.soft_bear_strong_leader_min_momentum)
            if is_opening_guard and self._is_bullish_regime():
                min_score += self.cfg.early_session_min_score_boost
            if score < min_score:
                logger.debug(
                    "진입 스킵(점수 미달): %s 점수=%.1f, 임계=%.1f",
                    q.symbol,
                    score,
                    min_score,
                )
                continue
            if soft_bear_leader_lane and not self._is_soft_bear_strong_leader_long_candidate(q, score=score):
                continue
            candidates.append((score, q))

        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates

    def _required_volume_spike_ratio(self, quote: Quote, score: float) -> float:
        required_ratio = self._regime_volume_spike_ratio()
        ratio_floor = self._regime_volume_spike_ratio_min()
        if self._is_bullish_regime():
            required_ratio -= self.cfg.bullish_volume_spike_ratio_adjustment

        # 강한 모멘텀/급등 구간은 스파이크 임계치를 완화해 기회를 살린다.
        if score >= 4.5:
            required_ratio -= 0.60
        elif score >= 4.0:
            required_ratio -= 0.45
        elif score >= 3.5:
            required_ratio -= 0.25

        if quote.change_rate >= 2.0:
            required_ratio -= 0.15
        if quote.change_rate >= 3.0:
            required_ratio -= 0.10

        if quote.symbol in self._inverse_symbols:
            required_ratio -= self.cfg.inverse_volume_spike_ratio_offset
            if self._bear_score >= max(self.cfg.bearish_threshold, self.cfg.inverse_min_bear_score):
                required_ratio -= 0.10
            ratio_floor = max(0.9, ratio_floor * 0.9)

        if required_ratio < ratio_floor:
            required_ratio = ratio_floor

        return max(0.0, required_ratio)

    def _update_tick_volume_state(self, quote: Quote):
        if quote.volume <= 0:
            self._latest_tick_volumes[quote.symbol] = 0
            return

        prev = self._last_cumulative_volumes.get(quote.symbol)
        if prev is None:
            # 초깃값은 누적량 기준으로부터의 실제 체결량이 아니므로 기준선 오염을 막기 위해 보류.
            self._last_cumulative_volumes[quote.symbol] = quote.volume
            self._latest_tick_volumes[quote.symbol] = 0
            return

        delta = quote.volume - prev
        if delta < 0:
            # 거래량 누적이 리셋되었거나 롤오버된 구간이면 기준선을 재동기화.
            self._latest_tick_volumes[quote.symbol] = 0
            self._last_cumulative_volumes[quote.symbol] = quote.volume
            return

        self._last_cumulative_volumes[quote.symbol] = quote.volume
        self._latest_tick_volumes[quote.symbol] = delta
        if delta <= 0:
            return

        dq = self._recent_tick_volumes.setdefault(
            quote.symbol,
            deque(maxlen=max(self.cfg.volume_spike_min_history * 3, 12)),
        )
        dq.append(delta)

    def _is_volume_spike(self, quote: Quote, score: Optional[float] = None) -> bool:
        if not self.cfg.enable_volume_spike_filter:
            return True

        if quote.volume <= 0:
            logger.debug("거래량 스파이크 탈락: %s 누적거래량 없음", quote.symbol)
            return False

        current_delta = self._latest_tick_volumes.get(quote.symbol, 0)
        abs_min = self._regime_volume_spike_abs_min()
        if self._is_bullish_regime():
            abs_min = int(max(1, abs_min * self.cfg.bullish_volume_spike_abs_min_ratio))
        if quote.symbol in self._inverse_symbols:
            abs_min = int(max(1, abs_min * self.cfg.inverse_volume_spike_abs_min_ratio))
        if current_delta < abs_min:
            logger.debug(
                "거래량 스파이크 탈락(절대량 부족): %s 현재 %d < %d",
                quote.symbol,
                current_delta,
                abs_min,
            )
            return False

        history = self._recent_tick_volumes.get(quote.symbol, deque())
        if len(history) < self.cfg.volume_spike_min_history:
            return True

        if score is None:
            score = self._calc_momentum_score(quote)
        required_ratio = self._required_volume_spike_ratio(quote=quote, score=score)
        if required_ratio <= 0:
            return True

        prev_deltas = list(history)[:-1]
        if not prev_deltas:
            return True

        prev_deltas.sort()
        mid_idx = len(prev_deltas) // 2
        if len(prev_deltas) % 2:
            baseline = prev_deltas[mid_idx]
        else:
            baseline = (prev_deltas[mid_idx - 1] + prev_deltas[mid_idx]) / 2
        if baseline <= 0:
            return True
        ratio = current_delta / baseline
        if ratio < required_ratio:
            logger.info(
                "거래량 스파이크 탈락: %s 현재=%d, 기준=%d, 비율=%.2f (임계 %.2f)",
                quote.symbol,
                current_delta,
                int(baseline),
                ratio,
                required_ratio,
            )
            return False
        return True

    def _can_open_new_long(
        self,
        quote: Quote,
        pending_orders: Optional[List[Order]] = None,
        score: Optional[float] = None,
        now: Optional[datetime] = None,
    ) -> bool:
        if now is None:
            now = self._now()
        profile_name = self._resolve_regime_profile_name()

        if self._is_new_entry_window_blocked(now):
            return False

        if any(
            o.side == OrderSide.BUY and o.symbol == quote.symbol
            for o in pending_orders or []
        ):
            return False

        strategy_name = self._current_profile_entry_strategy_name(is_inverse=False)
        if strategy_name and not self._is_strategy_gate_enabled(strategy_name):
            self._log_setup_reject(
                quote,
                "strategy_gate_disabled",
                "전략 자동 게이트 비활성화: %s (%s)",
                quote.symbol,
                strategy_name,
            )
            return False

        symbol_cooldown_until = self._symbol_cooldown_remaining(quote.symbol, now)
        if symbol_cooldown_until is not None:
            self._log_setup_reject(
                quote,
                "symbol_loss_cooldown",
                "종목 손실 쿨다운 중입니다: %s (재허용 %s)",
                quote.symbol,
                symbol_cooldown_until.strftime("%H:%M:%S"),
            )
            return False

        strategy_cooldown_until = self._strategy_cooldown_remaining(strategy_name, now)
        if strategy_cooldown_until is not None:
            self._log_setup_reject(
                quote,
                "strategy_loss_cooldown",
                "전략 손실 쿨다운 중입니다: %s (%s, 재허용 %s)",
                quote.symbol,
                strategy_name,
                strategy_cooldown_until.strftime("%H:%M:%S"),
            )
            return False

        soft_bear_leader_lane = profile_name == "soft_bear" and self._is_soft_bear_strong_leader_long_candidate(
            quote,
            score=score,
        )
        if profile_name == "bear":
            self._log_setup_reject(
                quote,
                "long_disabled",
                "%s 레짐에서는 신규 롱을 열지 않습니다: %s",
                profile_name,
                quote.symbol,
            )
            return False
        if profile_name == "soft_bear" and not soft_bear_leader_lane:
            self._log_setup_reject(
                quote,
                "long_disabled",
                "%s 레짐에서는 강한 리더만 예외적으로 신규 롱을 평가합니다: %s",
                profile_name,
                quote.symbol,
            )
            return False

        if profile_name == "neutral" and strategy_name == "neutral_pullback_strategy":
            neutral_loss_limit = self._neutral_loss_limit()
            if self._neutral_loss_count_today > neutral_loss_limit:
                self._track_shadow_blocked_candidate(quote, "neutral_loss_limit_block")
                self._log_setup_reject(
                    quote,
                    "neutral_loss_limit_block",
                    "중립장 손실 한도 초과로 신규 롱을 차단합니다: %s (손실 %d/%d회)",
                    quote.symbol,
                    self._neutral_loss_count_today,
                    neutral_loss_limit,
                )
                return False
            if self._neutral_loss_count_today == neutral_loss_limit:
                cooldown_until = self._neutral_post_loss_cooldown_until()
                if cooldown_until is None:
                    self._track_shadow_blocked_candidate(quote, "neutral_loss_limit_block")
                    self._log_setup_reject(
                        quote,
                        "neutral_loss_limit_block",
                        "중립장 손실 기준은 찼지만 손실 시각 정보가 없어 신규 롱을 차단합니다: %s",
                        quote.symbol,
                    )
                    return False
                if cooldown_until and now < cooldown_until:
                    self._log_setup_reject(
                        quote,
                        "neutral_loss_cooldown",
                        "중립장 손실 후 쿨다운 중입니다: %s (재허용 %s)",
                        quote.symbol,
                        cooldown_until.strftime("%H:%M:%S"),
                    )
                    return False
                if self._neutral_post_loss_reentries_today >= int(self.cfg.neutral_post_loss_reentry_limit):
                    self._track_shadow_blocked_candidate(quote, "neutral_loss_limit_block")
                    self._log_setup_reject(
                        quote,
                        "neutral_loss_limit_block",
                        "중립장 손실 후 재도전 한도를 모두 사용했습니다: %s (%d/%d회)",
                        quote.symbol,
                        self._neutral_post_loss_reentries_today,
                        int(self.cfg.neutral_post_loss_reentry_limit),
                    )
                    return False
                retry_thresholds = self._neutral_retry_thresholds()
                if score is not None and score < retry_thresholds["min_score"]:
                    self._log_setup_reject(
                        quote,
                        "neutral_post_loss_quality_block",
                        "중립장 재도전 A급 점수 미달: %s (점수 %.2f < %.2f)",
                        quote.symbol,
                        score,
                        retry_thresholds["min_score"],
                    )
                    return False
                if quote.change_rate < retry_thresholds["min_change_rate"]:
                    self._log_setup_reject(
                        quote,
                        "neutral_post_loss_quality_block",
                        "중립장 재도전 A급 등락률 미달: %s (등락률 %.2f%% < %.2f%%)",
                        quote.symbol,
                        quote.change_rate,
                        retry_thresholds["min_change_rate"],
                    )
                    return False

        if self._is_loss_stage_active() and strategy_name == "neutral_pullback_strategy":
            stage1_score_floor = self._regime_min_momentum_score() + float(self.cfg.stage1_neutral_score_bonus)
            stage1_change_floor = self._regime_min_change_rate() + float(self.cfg.stage1_neutral_change_rate_bonus)
            total_net = self._current_total_net_pnl()
            if score is not None and score < stage1_score_floor:
                self._log_setup_reject(
                    quote,
                    "risk_stage1_quality_block",
                    "손실 1단계에서는 A급 중립장 롱만 허용합니다: %s (점수 %.2f < %.2f, 총손익 %s원)",
                    quote.symbol,
                    score,
                    stage1_score_floor,
                    f"{total_net:,}",
                )
                return False
            if quote.change_rate < stage1_change_floor:
                self._log_setup_reject(
                    quote,
                    "risk_stage1_quality_block",
                    "손실 1단계에서는 A급 중립장 롱만 허용합니다: %s (등락률 %.2f%% < %.2f%%, 총손익 %s원)",
                    quote.symbol,
                    quote.change_rate,
                    stage1_change_floor,
                    f"{total_net:,}",
                )
                return False

        if self._bear_market and self.cfg.bear_market_mode == "B":
            logger.info("신규롱 차단: 약세 모드 B")
            return False

        if self._bear_market and self.cfg.bear_market_mode == "A":
            min_bear_score = self._regime_bear_score_for_new_long()
            if self._bear_score >= min_bear_score:
                if score is not None and score >= self._regime_bear_market_entry_score():
                    logger.info(
                        "약세 모드 예외 통과: %s (레짐점수=%d, 모멘텀점수=%.1f, 임계=%.1f)",
                        quote.symbol,
                        self._bear_score,
                        score,
                        self._regime_bear_market_entry_score(),
                    )
                    return True
                logger.info("신규롱 차단: 약세점수 높음 (%d >= %d)",
                            self._bear_score, min_bear_score)
                return False

        if (
            self.cfg.enable_pool_persistence_gate
            and quote.symbol not in self._latest_direct_dynamic_symbols
            and not self._is_pool_persistent(quote.symbol)
        ):
            logger.debug("신규롱 차단: 동적 풀 지속성 미충족 (%s)", quote.symbol)
            return False

        return True

    def _can_confirm_entry(
        self,
        quote: Quote,
        score: float,
        is_scale_in: bool,
        now: datetime,
    ) -> bool:
        if not self.cfg.enable_entry_confirmation:
            return True

        required_ticks = (
            self.cfg.scale_in_confirmation_ticks
            if is_scale_in
            else self.cfg.entry_confirmation_ticks
        )
        profile_name = self._resolve_regime_profile_name()
        if (
            self._is_bullish_regime()
            and not is_scale_in
            and self._is_early_session_guard_active()
        ):
            required_ticks = max(required_ticks, self.cfg.early_session_entry_confirmation_ticks)
        if (
            not is_scale_in
            and self._is_bullish_regime()
            and not self._is_early_session_guard_active()
            and required_ticks > 1
        ):
            fast_entry_min_score = (
                self._regime_bullish_min_momentum_score()
                + self.cfg.bullish_fast_entry_score_bonus
            )
            fast_entry_min_change = (
                self._regime_bullish_min_change_rate()
                + self.cfg.bullish_fast_entry_change_rate_bonus
            )
            if score >= fast_entry_min_score and quote.change_rate >= fast_entry_min_change:
                required_ticks = 1
        if (
            not is_scale_in
            and profile_name == "neutral"
        ):
            required_ticks = max(required_ticks, int(self.cfg.neutral_entry_confirmation_ticks))
        if required_ticks <= 1:
            self._entry_signals.pop(quote.symbol, None)
            return True

        if not quote.current_price:
            self._entry_signals.pop(quote.symbol, None)
            return False

        signal = self._entry_signals.get(quote.symbol)
        if not signal:
            self._entry_signals[quote.symbol] = MomentumEntrySignal(
                streak=1,
                first_price=quote.current_price,
                best_score=score,
                started_at=now,
                last_seen_at=now,
            )
            return False

        ttl = max(1, self.cfg.entry_confirmation_window_seconds)
        if (now - signal.last_seen_at).total_seconds() > ttl:
            self._entry_signals[quote.symbol] = MomentumEntrySignal(
                streak=1,
                first_price=quote.current_price,
                best_score=score,
                started_at=now,
                last_seen_at=now,
            )
            return False

        max_pullback_pct = self.cfg.entry_confirmation_max_pullback_pct
        score_tolerance = self.cfg.entry_confirmation_min_score_tolerance
        if profile_name == "neutral":
            max_pullback_pct = min(max_pullback_pct, -0.8)
            score_tolerance = max(score_tolerance, 0.55)
        elif profile_name == "soft_bear":
            max_pullback_pct = min(max_pullback_pct, -1.0)
            score_tolerance = max(score_tolerance, 0.65)

        pullback_pct = (quote.current_price - signal.first_price) / signal.first_price * 100
        if pullback_pct < max_pullback_pct:
            self._entry_signals[quote.symbol] = MomentumEntrySignal(
                streak=1,
                first_price=quote.current_price,
                best_score=score,
                started_at=now,
                last_seen_at=now,
            )
            logger.debug(
                "진입 재확인 리셋(과도한 되돌림): %s %.2f%% < %.2f%%",
                quote.symbol,
                pullback_pct,
                max_pullback_pct,
            )
            return False

        if score + score_tolerance < signal.best_score:
            self._entry_signals[quote.symbol] = MomentumEntrySignal(
                streak=1,
                first_price=quote.current_price,
                best_score=score,
                started_at=now,
                last_seen_at=now,
            )
            return False

        signal.streak += 1
        signal.best_score = max(signal.best_score, score)
        signal.last_seen_at = now
        return signal.streak >= required_ticks

    def _estimate_market_from_quotes(self, quotes: List[Quote]) -> Optional[tuple[int, float, float]]:
        """배치 시세에서 약세 점수를 추정한다 (백테스트용).

        인버스 ETF를 제외한 일반 종목의 등락률로 판단.
        """
        regular_quotes = [q for q in quotes if q.symbol not in self._inverse_symbols]
        if not regular_quotes:
            return None

        total = len(regular_quotes)
        avg_change = sum(q.change_rate for q in regular_quotes) / total
        declining = sum(1 for q in regular_quotes if q.change_rate < 0)
        decline_ratio = declining / total if total else 0.0

        score = 0
        if avg_change < -0.35:
            score += 1
        if avg_change < -0.9:
            score += 1
        if avg_change < -1.4:
            score += 1

        # 평균 등락률 < -0.5%
        if avg_change < -2.0:
            score += 1

        # 하락 종목 비율 > 70%
        if decline_ratio > 0.6:
            score += 1
        if decline_ratio > 0.78:
            score += 1

        return score, avg_change, decline_ratio

    def _passes_pullback_entry_filter(self, quote: Quote, is_scale_in: bool) -> bool:
        if is_scale_in or not self.cfg.enable_pullback_entry_filter:
            return True
        if quote.current_price <= 0 or quote.open_price <= 0 or quote.high_price <= 0:
            return True
        profile_name = self._resolve_regime_profile_name()
        activation_change_rate = self.cfg.pullback_activation_change_rate
        required_min_drop_pct = self.cfg.pullback_required_min_drop_pct
        allowed_max_drop_pct = self.cfg.pullback_allowed_max_drop_pct
        min_vs_open_pct = self.cfg.pullback_min_vs_open_pct
        if profile_name == "neutral":
            activation_change_rate = max(activation_change_rate, 2.3)
            required_min_drop_pct = min(required_min_drop_pct, 0.12)
            allowed_max_drop_pct = max(allowed_max_drop_pct, 1.6)
            min_vs_open_pct = min(min_vs_open_pct, 0.18)
        elif profile_name == "soft_bear":
            activation_change_rate = max(activation_change_rate, 2.6)
            required_min_drop_pct = min(required_min_drop_pct, 0.10)
            allowed_max_drop_pct = max(allowed_max_drop_pct, 1.8)
            min_vs_open_pct = min(min_vs_open_pct, 0.12)

        if quote.change_rate < activation_change_rate:
            return True

        pullback_drop_pct = (quote.high_price - quote.current_price) / quote.high_price * 100
        if pullback_drop_pct < required_min_drop_pct:
            self._log_entry_filter_once_per_minute(
                quote.symbol,
                "pullback_wait",
                "눌림목 대기: %s 고점대비 조정 %.2f%% < 최소 %.2f%%",
                quote.symbol,
                pullback_drop_pct,
                required_min_drop_pct,
            )
            return False
        if pullback_drop_pct > allowed_max_drop_pct:
            self._log_entry_filter_once_per_minute(
                quote.symbol,
                "pullback_broken",
                "눌림목 이탈: %s 고점대비 조정 %.2f%% > 최대 %.2f%%",
                quote.symbol,
                pullback_drop_pct,
                allowed_max_drop_pct,
            )
            return False

        vs_open = (quote.current_price - quote.open_price) / quote.open_price * 100
        if vs_open < min_vs_open_pct:
            self._log_entry_filter_once_per_minute(
                quote.symbol,
                "pullback_vs_open",
                "눌림목 취소: %s 시가대비 %.2f%% < 최소 %.2f%%",
                quote.symbol,
                vs_open,
                min_vs_open_pct,
            )
            return False
        return True

    def _evaluate_buy(
        self,
        quote: Quote,
        pending_orders: Optional[List[Order]] = None,
        score_hint: Optional[float] = None,
    ) -> Optional[Order]:
        """모멘텀 점수 기반 매수 판단 (일반 주식)."""
        # 인버스 ETF는 별도 로직
        if quote.symbol in self._inverse_symbols:
            return None

        now = self._now()
        is_opening_guard = self._is_early_session_guard_active()

        if quote.current_price <= 0 or quote.open_price <= 0:
            return None
        if quote.current_price < self.cfg.min_price:
            return None
        min_change_rate = self._regime_min_change_rate()
        if self._is_bullish_regime():
            min_change_rate = self._regime_bullish_min_change_rate()
        if is_opening_guard and self._is_bullish_regime():
            min_change_rate += self.cfg.early_session_min_change_rate_boost
        if quote.change_rate < min_change_rate:
            return None

        # 쿨다운 체크
        last_sold = self._sell_cooldown.get(quote.symbol)
        if last_sold:
            elapsed = (self._now() - last_sold).total_seconds()
            if elapsed < self._regime_cooldown_seconds():
                return None

        if any(
            o.side == OrderSide.BUY and o.symbol == quote.symbol
            for o in pending_orders or []
        ):
            return None

        position = self.positions.get(quote.symbol)
        is_scale_in = position is not None

        score = self._calc_momentum_score(quote) if score_hint is None else score_hint
        entry_reason = ""
        strategy_name = ""
        setup_name = ""
        if is_scale_in:
            if not self._is_volume_spike(quote, score=score):
                return None
            if not self.cfg.enable_pyramiding:
                return None
            pnl_pct = (quote.current_price - position.buy_price) / position.buy_price * 100
            if pnl_pct < self.cfg.scale_in_min_profit_pct:
                return None
            scale_threshold = (
                self._regime_min_momentum_score() + self.cfg.scale_in_score_bonus
            )
            if self._is_bullish_regime():
                scale_threshold = (
                    self._regime_bullish_min_momentum_score() + self.cfg.scale_in_score_bonus
            )
            if score < scale_threshold:
                return None
        else:
            if not self._can_open_new_long(
                quote,
                pending_orders=pending_orders,
                score=score,
                now=now,
            ):
                return None
            if self.market_data is None and self.cfg.enable_backtest_score_entry_fallback:
                fallback_min_score = self._regime_min_momentum_score()
                if self._is_bullish_regime():
                    fallback_min_score = self._regime_bullish_min_momentum_score()
                if score < fallback_min_score:
                    return None
                if not self._is_volume_spike(quote, score=score):
                    return None
                strategy_name = "backtest_score_fallback_strategy"
                setup_name = "backtest_score_entry"
                fallback_payload = (
                    "setup_name=backtest_score_entry "
                    "entry_reason=score_fallback"
                )
                entry_meta = self._build_entry_metadata(
                    quote.symbol,
                    setup_name,
                    fallback_payload,
                    strategy_name=strategy_name,
                )
                entry_reason = self._append_entry_context(fallback_payload, entry_meta)
            else:
                decision = self._regime_router.evaluate_long_entry(self, quote, score)
                if not decision.allowed:
                    self._entry_signals.pop(quote.symbol, None)
                    if decision.reject_reason:
                        self._log_setup_reject(
                            quote,
                            decision.reject_reason,
                            "%s (%s)",
                            quote.symbol,
                            decision.reject_reason,
                        )
                    return None
                strategy_name = decision.strategy_name
                setup_name = decision.setup_name
                entry_meta = self._build_entry_metadata(
                    quote.symbol,
                    setup_name,
                    decision.payload,
                    strategy_name=strategy_name,
                )
                if self._resolve_regime_profile_name() == "neutral" and self._is_neutral_post_loss_retry_available(now):
                    entry_meta["neutral_post_loss_retry"] = True
                entry_reason = self._append_entry_context(decision.payload, entry_meta)

        if quote.change_rate >= self.cfg.overheated_jump_change_pct:
            retrace_anchor = quote.open_price * (
                1
                + (self.cfg.overheated_jump_change_pct / 100)
                * self.cfg.overheated_retrace_ratio
            )
            if quote.current_price < retrace_anchor:
                logger.info(
                    "과열 회피: %s %s%% 급등 후 되돌림 %.2f%% < %.2f%%",
                    quote.symbol,
                    f"{quote.change_rate:.2f}",
                    (quote.current_price - quote.open_price) / quote.open_price * 100,
                    (retrace_anchor / quote.open_price - 1) * 100,
                )
                self._entry_signals.pop(quote.symbol, None)
                return None

        if is_scale_in and not self._can_confirm_entry(
            quote=quote,
            score=score,
            is_scale_in=is_scale_in,
            now=now,
        ):
            return None

        entry_grade = str(entry_meta.get("entry_grade", "") or "").upper()
        turnover_rank = int(entry_meta.get("turnover_rank", 0) or 0)
        bull_priority_entry = (
            not is_scale_in
            and strategy_name == "bull_breakout_strategy"
            and entry_grade == "A"
            and 0 < turnover_rank <= max(1, int(self.cfg.bull_priority_turnover_rank_max))
        )

        alloc = self._compute_buy_allocation(
            symbol=quote.symbol,
            current_price=quote.current_price,
            pending_orders=pending_orders,
            allow_expensive_single_share_override=(
                not is_scale_in
                and strategy_name == "bull_breakout_strategy"
                and entry_grade == "A"
            ),
            per_stock_amount_multiplier=(
                float(self.cfg.bull_priority_per_stock_amount_multiplier)
                if bull_priority_entry
                else 1.0
            ),
            max_per_stock_amount_multiplier=(
                float(self.cfg.bull_priority_max_per_stock_amount_multiplier)
                if bull_priority_entry
                else 1.0
            ),
            max_single_position_pct_override=(
                float(self.cfg.bull_priority_max_single_position_pct)
                if bull_priority_entry
                else None
            ),
            side_slot_override=(
                int(self.cfg.bull_priority_effective_slots)
                if bull_priority_entry
                else None
            ),
        )
        if (
            not is_scale_in
            and strategy_name == "bull_breakout_strategy"
            and entry_grade == "A"
        ):
            initial_scale = (
                float(self.cfg.bull_priority_initial_entry_scale)
                if bull_priority_entry
                else float(self.cfg.bull_breakout_initial_entry_scale)
            )
            initial_scale = min(1.0, max(0.1, initial_scale))
            alloc = min(alloc, max(quote.current_price, int(alloc * initial_scale)))
        quantity = alloc // quote.current_price

        if quantity <= 0:
            return None
        if not self._passes_expected_net_filter(
            symbol=quote.symbol,
            quantity=quantity,
            entry_price=quote.current_price,
        ):
            return None

        if is_scale_in:
            logger.info(
                "추가매수 신호: %s(%s) 점수=%.1f, %d주 @ %s원 (할당 %s원)",
                quote.name,
                quote.symbol,
                score,
                quantity,
                f"{quote.current_price:,}",
                f"{alloc:,}",
            )
        else:
            self._pending_entry_meta[quote.symbol] = dict(entry_meta)
            logger.info(
                "매수 신호: %s(%s) 점수=%.1f, %d주 @ %s원 (할당 %s원, %s)",
                quote.name,
                quote.symbol,
                score,
                quantity,
                f"{quote.current_price:,}",
                f"{alloc:,}",
                entry_reason,
            )

        self._entry_signals.pop(quote.symbol, None)

        return Order(
            symbol=quote.symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=0,
        )

    def _evaluate_inverse_buy(self, quote: Quote, pending_orders: Optional[List[Order]] = None) -> Optional[Order]:
        """인버스 ETF 매수 판단.

        약세 점수 >= bearish_threshold일 때만 진입.
        인버스 ETF는 시장 하락 시 상승하므로 모멘텀 점수가 자연스럽게 높아진다.
        """
        now = self._now()
        profile_name = self._resolve_regime_profile_name()
        if self._is_new_entry_window_blocked(now):
            return None
        if quote.current_price <= 0 or quote.open_price <= 0:
            return None
        required_bear_score = max(self.cfg.bearish_threshold, self._regime_inverse_min_bear_score())
        if self._bear_score < required_bear_score:
            return None

        # 쿨다운 체크
        last_sold = self._sell_cooldown.get(quote.symbol)
        if last_sold:
            elapsed = (now - last_sold).total_seconds()
            if elapsed < self._regime_cooldown_seconds():
                return None

        strategy_name = self._current_profile_entry_strategy_name(is_inverse=True)
        if strategy_name and not self._is_strategy_gate_enabled(strategy_name):
            self._log_setup_reject(
                quote,
                "strategy_gate_disabled",
                "%s (%s)",
                quote.symbol,
                strategy_name,
            )
            return None

        symbol_cooldown_until = self._symbol_cooldown_remaining(quote.symbol, now)
        if symbol_cooldown_until is not None:
            self._log_setup_reject(
                quote,
                "symbol_loss_cooldown",
                "%s (재허용 %s)",
                quote.symbol,
                symbol_cooldown_until.strftime("%H:%M:%S"),
            )
            return None

        strategy_cooldown_until = self._strategy_cooldown_remaining(strategy_name, now)
        if strategy_cooldown_until is not None:
            self._log_setup_reject(
                quote,
                "strategy_loss_cooldown",
                "%s (%s, 재허용 %s)",
                quote.symbol,
                strategy_name,
                strategy_cooldown_until.strftime("%H:%M:%S"),
            )
            return None

        score = self._calc_momentum_score(quote)
        decision = self._regime_router.evaluate_inverse_entry(self, quote, score)
        if not decision.allowed:
            if decision.reject_reason:
                self._log_setup_reject(
                    quote,
                    decision.reject_reason,
                    "%s (%s)",
                    quote.symbol,
                    decision.reject_reason,
                )
            return None

        entry_meta = self._build_entry_metadata(
            quote.symbol,
            decision.setup_name,
            decision.payload,
            strategy_name=decision.strategy_name,
        )
        entry_reason = self._append_entry_context(decision.payload, entry_meta)

        alloc = self._compute_buy_allocation(
            symbol=quote.symbol,
            current_price=quote.current_price,
            pending_orders=pending_orders,
        )
        quantity = alloc // quote.current_price
        if quantity <= 0:
            return None
        if not self._passes_expected_net_filter(
            symbol=quote.symbol,
            quantity=quantity,
            entry_price=quote.current_price,
        ):
            return None

        self._pending_entry_meta[quote.symbol] = dict(entry_meta)
        logger.info(
            "[INV] 매수 신호: %s 약세점수=%d, 모멘텀=%.1f, %d주 @ %s원 (%s)",
            quote.symbol,
            self._bear_score,
            score,
            quantity,
            f"{quote.current_price:,}",
            entry_reason,
        )

        return Order(
            symbol=quote.symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=0,
        )

    def _compute_buy_allocation(
        self,
        symbol: str,
        current_price: int,
        pending_orders: Optional[List[Order]] = None,
        allow_expensive_single_share_override: bool = False,
        per_stock_amount_multiplier: float = 1.0,
        max_per_stock_amount_multiplier: float = 1.0,
        max_single_position_pct_override: Optional[float] = None,
        side_slot_override: Optional[int] = None,
    ) -> int:
        is_inverse = symbol in self._inverse_symbols
        capital_base = self._allocation_capital_base()
        target_total_exposure = self._regime_target_total_exposure_amount()
        total_exposure = self._get_total_exposure()
        stock_exposure = self._get_stock_exposure(symbol)
        side_exposure = sum(
            pos.buy_price * pos.quantity
            for sym, pos in self.positions.items()
            if (sym in self._inverse_symbols) == is_inverse
        )

        pending_total = 0
        pending_stock = 0
        pending_side = 0
        pending_side_count = 0
        for order in pending_orders or []:
            if order.side != OrderSide.BUY:
                continue
            q = self._quotes_cache.get(order.symbol)
            if q is None or q.current_price <= 0:
                continue
            amount = q.current_price * order.quantity
            pending_total += amount
            if order.symbol == symbol:
                pending_stock += amount
            if (order.symbol in self._inverse_symbols) == is_inverse:
                pending_side += amount
                if order.symbol not in self.positions:
                    pending_side_count += 1

        if current_price <= 0:
            return 0

        total_room = min(
            capital_base - (total_exposure + pending_total),
            target_total_exposure - (total_exposure + pending_total),
        )
        per_stock_amount = int(
            self.cfg.per_stock_amount
            * self._regime_per_stock_alloc_scale()
            * max(0.1, float(per_stock_amount_multiplier))
        )
        max_stock_amount = int(
            self.cfg.max_per_stock_amount
            * self._regime_max_stock_alloc_scale()
            * max(0.1, float(max_per_stock_amount_multiplier))
        )
        if max_single_position_pct_override is None:
            max_stock_amount = min(max_stock_amount, self._regime_max_single_position_amount())
        else:
            override_cap = int(
                capital_base
                * max(0.0, float(max_single_position_pct_override))
                * self._risk_exposure_scale()
            )
            max_stock_amount = min(max_stock_amount, override_cap)
        stock_room = max_stock_amount - (stock_exposure + pending_stock)

        side_slot_limit = (
            self._regime_inverse_max_positions()
            if is_inverse
            else self._effective_max_position_count()
        )
        if side_slot_limit <= 0:
            return 0

        open_side_count = sum(
            1 for sym in self.positions if (sym in self._inverse_symbols) == is_inverse
        )
        remaining_side_slots = max(1, side_slot_limit - open_side_count - pending_side_count)
        if side_slot_override is not None:
            remaining_side_slots = max(1, min(remaining_side_slots, int(side_slot_override)))
        side_room_target = target_total_exposure - (side_exposure + pending_side)
        if side_room_target <= 0:
            return 0
        side_budget_target = max(
            per_stock_amount,
            int(side_room_target / remaining_side_slots),
        )

        alloc = min(side_budget_target, total_room, stock_room)
        expensive_single_share_allowed = (
            allow_expensive_single_share_override
            and self.cfg.allow_expensive_single_share_override
            and current_price >= int(self.cfg.expensive_single_share_min_price)
            and total_room >= current_price
            and (stock_exposure + pending_stock) <= 0
            and current_price <= int(max_stock_amount * float(self.cfg.expensive_single_share_cap_multiplier))
        )
        if alloc <= 0:
            if expensive_single_share_allowed:
                return current_price
            return 0

        # 비싼 종목(할당액 < 현재가)도 1주라도 살 수 있으면 최소 1주 주문을 허용한다.
        if alloc < current_price:
            if expensive_single_share_allowed:
                return current_price
            if total_room >= current_price and stock_room >= current_price:
                return current_price
            return 0

        return alloc

    def _get_total_exposure(self) -> int:
        return sum(pos.buy_price * pos.quantity for pos in self.positions.values())

    def _get_stock_exposure(self, symbol: str) -> int:
        pos = self.positions.get(symbol)
        if pos is None:
            return 0
        return pos.buy_price * pos.quantity

    def _default_long_exit(self, quote: Quote) -> Optional[Order]:
        """익절/손절/추적손절 판단 (일반 주식)."""
        pos = self.positions.get(quote.symbol)
        if not pos:
            return None

        pnl_pct = (quote.current_price - pos.buy_price) / pos.buy_price * 100
        pnl_amount = (quote.current_price - pos.buy_price) * pos.quantity
        stop_amount = self._long_stop_loss_amount(pos)

        holding_minutes = (self._now() - pos.buy_time).total_seconds() / 60
        if holding_minutes >= self._regime_max_holding_minutes() and not pos.is_restored:
            logger.info("보유시간 초과 청산: %s (%.1f분)",
                        quote.symbol, holding_minutes)
            return self._make_sell_order(pos)

        # 익절
        if pnl_pct >= self._regime_take_profit_pct():
            if (
                pos.entry_strategy_name == "bull_breakout_strategy"
                and pos.entry_grade == "A"
                and not pos.partial_exit_done
                and pos.quantity >= 2
            ):
                partial_ratio = min(0.9, max(0.1, float(self.cfg.bull_partial_exit_ratio)))
                partial_qty = max(1, int(pos.quantity * partial_ratio))
                if partial_qty >= pos.quantity:
                    partial_qty = pos.quantity - 1
                if partial_qty > 0:
                    logger.info(
                        "bull A급 부분익절: %s %.2f%% (%s원) %d/%d주",
                        quote.symbol,
                        pnl_pct,
                        f"{pnl_amount:,}",
                        partial_qty,
                        pos.quantity,
                    )
                    return self._make_sell_order(pos, quantity=partial_qty)
            elif (
                pos.entry_strategy_name == "bull_breakout_strategy"
                and pos.entry_grade == "A"
                and pos.partial_exit_done
            ):
                pass
            elif not self._should_defer_profit_exit(
                pos=pos,
                exit_price=quote.current_price,
                reason="익절",
            ):
                logger.info("익절: %s %.2f%% (%s원)",
                            quote.symbol, pnl_pct, f"{pnl_amount:,}")
                return self._make_sell_order(pos)

        # 개별 포지션 손절 (포지션 노출 연동 금액 기준)
        if pnl_amount <= stop_amount:
            logger.info("개별손절: %s %s원 (한도 %s원)",
                        quote.symbol, f"{pnl_amount:,}",
                        f"{stop_amount:,}")
            return self._make_sell_order(pos)

        # 추적손절 (고점 대비)
        if pos.high_since_buy > pos.buy_price:
            drop_from_high = (quote.current_price - pos.high_since_buy) / pos.high_since_buy * 100
            gain_from_entry = (pos.high_since_buy - pos.buy_price) / pos.buy_price * 100
            if (
                gain_from_entry >= self._regime_trailing_stop_activation_gain_pct()
                and drop_from_high <= self._regime_trailing_stop_pct()
            ):
                if self._should_defer_profit_exit(
                    pos=pos,
                    exit_price=quote.current_price,
                    reason="추적손절",
                ):
                    return None
                logger.info("추적손절: %s 고점 %s → 현재 %s (%.2f%%)",
                            quote.symbol, f"{pos.high_since_buy:,}",
                            f"{quote.current_price:,}", drop_from_high)
                return self._make_sell_order(pos)

        return None

    def _evaluate_sell(self, quote: Quote) -> Optional[Order]:
        return self._regime_router.evaluate_long_exit(self, quote)

    def _effective_realized_net_for_breaker(self) -> int:
        """일일 브레이커 판단에 사용할 순실현손익을 반환한다."""
        return int(self.daily_pnl.realized_net_pnl - self._daily_breaker_pnl_offset)

    def _default_inverse_exit(self, quote: Quote) -> Optional[Order]:
        """인버스 ETF 매도 판단 (타이트한 리스크 관리).

        인버스 ETF는 음의 복리 위험이 있으므로:
        - 익절/손절 기준이 일반보다 타이트
        - 시간 기반 강제 청산 (최대 2시간)
        - 시장 반등 시(약세 점수 하락) 즉시 청산
        """
        pos = self.positions.get(quote.symbol)
        if not pos:
            return None

        pnl_pct = (quote.current_price - pos.buy_price) / pos.buy_price * 100
        pnl_amount = (quote.current_price - pos.buy_price) * pos.quantity
        stop_amount = self._inverse_stop_loss_amount(pos)

        # 1. 익절
        if pnl_pct >= self._regime_inverse_take_profit_pct():
            if self._should_defer_profit_exit(
                pos=pos,
                exit_price=quote.current_price,
                reason="[INV] 익절",
            ):
                return None
            logger.info("[INV] 익절: %s %.2f%%", quote.symbol, pnl_pct)
            return self._make_sell_order(pos)

        # 2. 손절 (포지션 노출 연동 금액 기준)
        if pnl_amount <= stop_amount:
            logger.info("[INV] 손절: %s %s원 (한도 %s원)", quote.symbol, f"{pnl_amount:,}", f"{stop_amount:,}")
            return self._make_sell_order(pos)

        # 3. 시간 초과 청산 (음의 복리 방지, 실거래 모드만)
        if self.market_data is not None:
            hold_minutes = (self._now() - pos.buy_time).total_seconds() / 60
            if hold_minutes >= self._regime_inverse_max_hold_minutes() and not pos.is_restored:
                logger.info("[INV] 시간초과 청산: %s (%.0f분 보유)", quote.symbol, hold_minutes)
                return self._make_sell_order(pos)

        # 4. 시장 반등 청산 (약세 점수가 임계 미만으로 떨어지면)
        if self._bear_score < self.cfg.bearish_threshold:
            logger.info("[INV] 시장반등 청산: %s (약세점수: %d)", quote.symbol, self._bear_score)
            return self._make_sell_order(pos)

        # 5. 추적손절 (고점 -0.3%, 일반 -0.7%보다 타이트)
        if pos.high_since_buy > pos.buy_price:
            drop_from_high = (quote.current_price - pos.high_since_buy) / pos.high_since_buy * 100
            gain_from_entry = (pos.high_since_buy - pos.buy_price) / pos.buy_price * 100
            if (
                gain_from_entry >= self._regime_inverse_trailing_stop_activation_gain_pct()
                and drop_from_high <= self._regime_inverse_trailing_stop_pct()
            ):
                if self._should_defer_profit_exit(
                    pos=pos,
                    exit_price=quote.current_price,
                    reason="[INV] 추적손절",
                ):
                    return None
                logger.info("[INV] 추적손절: %s 고점 %s → 현재 %s (%.2f%%)",
                            quote.symbol, f"{pos.high_since_buy:,}",
                            f"{quote.current_price:,}", drop_from_high)
                return self._make_sell_order(pos)

        return None

    def _evaluate_inverse_sell(self, quote: Quote) -> Optional[Order]:
        return self._regime_router.evaluate_inverse_exit(self, quote)

    def _calc_momentum_score(self, quote: Quote) -> float:
        """모멘텀 점수를 계산한다 (0~5)."""
        score = 0.0

        # 1. 시가 대비 상승폭 (0~1.5)
        if quote.open_price > 0:
            vs_open = (quote.current_price - quote.open_price) / quote.open_price * 100
            if vs_open >= 2.0:
                score += 1.5
            elif vs_open >= 1.0:
                score += 1.0
            elif vs_open >= 0.5:
                score += 0.5

        # 2. 전일 대비 등락률 (0~1.0)
        if quote.change_rate >= 2.0:
            score += 1.0
        elif quote.change_rate >= 1.0:
            score += 0.6
        elif quote.change_rate >= 0.5:
            score += 0.3

        # 3. 고가 근접도 (0~1.0)
        price_range = quote.high_price - quote.low_price
        if price_range > 0 and self._is_bullish_regime():
            proximity = (quote.current_price - quote.low_price) / price_range
            if proximity >= 0.9:
                score += 1.0
            elif proximity >= 0.7:
                score += 0.5

        # 4. 거래량 폭발 (0~1.5) — 5일 평균 대비
        avg_vol = self._avg_volumes.get(quote.symbol)
        if avg_vol and avg_vol > 0:
            vol_ratio = quote.volume / avg_vol
            if vol_ratio >= 3.0:
                score += 1.5
            elif vol_ratio >= 2.0:
                score += 1.0
            elif vol_ratio >= 1.5:
                score += 0.5

        return score

    def _make_sell_order(self, pos: PositionState, quantity: Optional[int] = None) -> Order:
        return Order(
            symbol=pos.symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=max(1, int(quantity if quantity is not None else pos.quantity)),
            price=0,
        )

    def _calc_commission_cost(self, notional: int) -> int:
        if notional <= 0:
            return 0
        return int(round(notional * self.cfg.commission_rate))

    def _calc_sell_tax_slippage_cost(self, sell_notional: int) -> int:
        if sell_notional <= 0:
            return 0
        return int(round(sell_notional * self.cfg.tax_slippage_rate))

    def _estimate_round_trip_net_pnl(self, pos: PositionState, exit_price: int) -> int:
        if exit_price <= 0 or pos.quantity <= 0:
            return 0
        gross = (exit_price - pos.buy_price) * pos.quantity
        buy_notional = max(0, int(pos.invested_amount))
        sell_notional = exit_price * pos.quantity
        buy_fee = self._calc_commission_cost(buy_notional)
        sell_fee = self._calc_commission_cost(sell_notional)
        sell_tax_slippage = self._calc_sell_tax_slippage_cost(sell_notional)
        return gross - buy_fee - sell_fee - sell_tax_slippage

    def _should_defer_profit_exit(
        self,
        pos: PositionState,
        exit_price: int,
        reason: str,
    ) -> bool:
        if not self.cfg.enable_cost_aware_profit_exit:
            return False

        estimated_net = self._estimate_round_trip_net_pnl(pos, exit_price)
        min_net = int(self.cfg.min_profit_exit_net_pnl)
        if estimated_net >= min_net:
            return False

        self._log_entry_filter_once_per_minute(
            pos.symbol,
            f"profit-exit:{reason}",
            "%s 보류: %s 예상왕복순익 %s원 < 최소 %s원 (현재가 %s원)",
            reason,
            pos.symbol,
            f"{estimated_net:,}",
            f"{min_net:,}",
            f"{exit_price:,}",
        )
        return True

    def _passes_expected_net_filter(self, symbol: str, quantity: int, entry_price: int) -> bool:
        if not self.cfg.enable_expected_net_filter:
            return True
        if quantity <= 0 or entry_price <= 0:
            return False

        entry_price_with_slip = int(round(entry_price * (1 + self.cfg.entry_market_slippage_rate)))
        if entry_price_with_slip <= 0:
            return False

        buy_notional = entry_price_with_slip * quantity
        expected_exit_price = int(
            round(entry_price * (1 + self._regime_expected_move_pct() / 100))
        )
        if expected_exit_price <= 0:
            return False
        expected_exit_after_slip = int(
            round(expected_exit_price * (1 - self.cfg.exit_market_slippage_rate))
        )
        if expected_exit_after_slip <= 0:
            return False

        sell_notional = expected_exit_after_slip * quantity
        gross_expected = (expected_exit_after_slip - entry_price_with_slip) * quantity
        buy_fee = self._calc_commission_cost(buy_notional)
        sell_fee = self._calc_commission_cost(sell_notional)
        sell_tax_slippage = self._calc_sell_tax_slippage_cost(sell_notional)
        expected_net = gross_expected - buy_fee - sell_fee - sell_tax_slippage

        risk_amount = max(1, self._entry_stop_risk_amount(symbol, quantity, entry_price_with_slip))
        rr_ratio = expected_net / risk_amount
        passes = (
            expected_net >= self._regime_min_expected_net_profit()
            and rr_ratio >= self._regime_min_expected_rr_ratio()
        )
        if not passes:
            logger.info(
                "진입 필터 탈락: %s 기대순익 %s원, RR %.2f (기준: %s원 / %.2f), "
                "매입가=%s원, 매수예상=%s원, 매도예상=%s원, 수수료=%s원, 세금+슬리피지=%s원",
                symbol,
                f"{expected_net:,}",
                rr_ratio,
                f"{self._regime_min_expected_net_profit():,}",
                self._regime_min_expected_rr_ratio(),
                f"{entry_price:,}",
                f"{entry_price_with_slip:,}",
                f"{expected_exit_after_slip:,}",
                f"{(buy_fee + sell_fee):,}",
                f"{sell_tax_slippage:,}",
            )
        return passes

    def _estimate_unrealized_net_pnl(self) -> int:
        total = 0
        for sym, pos in self.positions.items():
            q = self._quotes_cache.get(sym)
            if not q or q.current_price <= 0:
                continue
            gross = (q.current_price - pos.buy_price) * pos.quantity
            sell_notional = q.current_price * pos.quantity
            exit_cost = (
                self._calc_commission_cost(sell_notional) +
                self._calc_sell_tax_slippage_cost(sell_notional)
            )
            total += (gross - exit_cost)
        return total

    def _liquidate_all(self) -> List[Order]:
        """전 포지션 청산 (일반 + 인버스 모두)."""
        orders = []
        for pos in self.positions.values():
            orders.append(self._make_sell_order(pos))
        return orders

    def load_avg_volumes(self, avg_volumes: Dict[str, int]):
        """5일 평균 거래량을 외부에서 주입한다 (백테스트/초기화 시)."""
        self._avg_volumes = avg_volumes
