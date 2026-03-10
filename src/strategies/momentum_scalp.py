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
from typing import Dict, List, Optional

import pandas as pd

from src.market_data import MarketDataAPI
from src.models import Order, OrderResult, OrderSide, OrderType, Position, Quote
from src.notifications import AlertManager
from src.strategy import BaseStrategy

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
    per_stock_amount: int = 180_000      # 종목당 기본 할당액
    max_per_stock_amount: int = 500_000  # 종목당 최대 노출 (피라미딩 상한)
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
    bullish_volume_spike_ratio_adjustment: float = 0.30
    bullish_volume_spike_abs_min_ratio: float = 0.6

    min_momentum_score: float = 3.5
    enable_expected_net_filter: bool = True
    expected_move_pct: float = 2.4
    min_expected_net_profit: int = 1_200
    min_expected_rr_ratio: float = 0.85
    # 진입 보강
    enable_entry_confirmation: bool = True          # 1차 후보 후 재확인 대기
    entry_confirmation_ticks: int = 2               # 신규 진입 최소 확인 틱 수
    scale_in_confirmation_ticks: int = 1            # 스케일인 최소 확인 틱 수
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

    def __post_init__(self):
        if self.invested_amount <= 0:
            self.invested_amount = self.buy_price * self.quantity
        if self.high_since_buy == 0:
            self.high_since_buy = self.buy_price


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
        self._sell_cooldown: Dict[str, datetime] = {}
        self._global_loss_cooldown_until: Optional[datetime] = None
        self._startup_rebalance_ticks: int = 0
        self._startup_rebalance_active: bool = False
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

    def initialize(self):
        today = datetime.now().date()
        self._state_loaded_for_today = False
        self._daily_breaker_pnl_offset = 0
        self._load_daily_state()
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
            self._global_loss_cooldown_until = None
            self._startup_rebalance_active = False
            self._startup_rebalance_ticks = 0
            self._last_cumulative_volumes = {}
            self._entry_filter_log_cache = {}
            self._daily_breaker_pnl_offset = 0
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
        self._current_day = today
        self._session_start_at = None
        self._build_pool()
        self._check_market_regime()
        regime_profile = self._build_regime_profile()
        self._log_regime_profiles()
        total_loss_limit = regime_profile["daily_total_loss_limit"]
        blocked_windows = ", ".join(
            label for _, _, label in self._entry_block_windows
        ) if self._entry_block_windows else "없음"

        if self._halted and self._halt_date == today:
            logger.info("당일 하드스탑 상태 유지: 신규 거래 중지")
            if self._hard_stop_bypass_for_day:
                logger.info("당일 하드스탑 우회 모드가 적용되어 거래를 계속 진행합니다.")

        logger.info("전략 초기화: 모멘텀 스캘핑")
        logger.info("  시드: %s원, 종목당: %s원",
                     f"{self.cfg.seed_money:,}", f"{self.cfg.per_stock_amount:,}")
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
            "  오늘 적용값 요약: 종목당 %s원(최대 %s원), 개별손절 %s원, 일손실한도 %s원, "
            "보조손실컷 %s원, 진입임계(점수 %.2f/등락률 %.2f%%/최소가 %s원), 차단시간 [%s]",
            f"{self.cfg.per_stock_amount:,}",
            f"{self.cfg.max_per_stock_amount:,}",
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
                         self.cfg.bearish_threshold, self.cfg.inverse_max_positions)
            logger.info(
                "  인버스 진입 강화: 약세점수≥%d, 등락률≥%.2f%%, 모멘텀≥%.2f, "
                "추적손절 발동이익≥%.2f%%, 거래량스파이크 완화=-%.2f",
                max(self.cfg.bearish_threshold, self.cfg.inverse_min_bear_score),
                self.cfg.inverse_min_change_rate,
                self.cfg.inverse_min_momentum,
                self.cfg.inverse_trailing_stop_activation_gain_pct,
                self.cfg.inverse_volume_spike_ratio_offset,
            )
        logger.info("  풀 크기: %d종목", len(self._pool))

    def _log_regime_profiles(self):
        bull_profile = self._build_regime_profile(profile_name="bull")
        neutral_profile = self._build_regime_profile(profile_name="neutral")
        bear_profile = self._build_regime_profile(profile_name="bear")
        logger.info("레짐 프로파일 비교표 (활성=%s)", self._regime_profile_name)
        logger.info(
            "  [강세] long 익절=%s%% 손절=%s원 추적=%s%% 보유=%d분 / inv 익절=%s%% inv손절=%s%% inv추적=%s%% inv보유=%d분",
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
            "  [중립] long 익절=%s%% 손절=%s원 추적=%s%% 보유=%d분 / inv 익절=%s%% inv손절=%s%% inv추적=%s%% inv보유=%d분",
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
            "  [약세] long 익절=%s%% 손절=%s원 추적=%s%% 보유=%d분 / inv 익절=%s%% inv손절=%s%% inv추적=%s%% inv보유=%d분",
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
            if not self._state_path.exists():
                return
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
            raw_date = payload.get("date")
            if not raw_date:
                return

            state_day = datetime.strptime(raw_date, "%Y-%m-%d").date()
            if state_day != datetime.now().date():
                return

            self._current_day = state_day
            self.daily_pnl = DailyPnL(
                realized_gross_pnl=payload.get("realized_gross_pnl", 0),
                realized_net_pnl=payload.get("realized_net_pnl", 0),
                fees_paid=payload.get("fees_paid", 0),
                taxes_paid=payload.get("taxes_paid", 0),
                trade_count=payload.get("trade_count", 0),
            )
            raw_halt_date = payload.get("halt_date")
            if isinstance(raw_halt_date, str):
                try:
                    self._halt_date = datetime.strptime(raw_halt_date, "%Y-%m-%d").date()
                except ValueError:
                    self._halt_date = None
            else:
                self._halt_date = None
            self._halted = bool(payload.get("halted", False))
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
                "halted": self._halted,
                "halt_date": self._halt_date.isoformat() if self._halt_date else None,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
            self._state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("일일 상태 저장 실패(무시): %s", e)

    def get_watchlist(self) -> List[str]:
        now = datetime.now()
        if (self._last_pool_refresh and
                (now - self._last_pool_refresh).total_seconds() >= self.cfg.pool_refresh_interval):
            self._build_pool()
        return self._pool

    def on_tick(self, quote: Quote) -> List[Order]:
        self._quotes_cache[quote.symbol] = quote
        if self._session_start_at is None:
            self._session_start_at = datetime.now()
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
        effective_max_position_count = self._effective_max_position_count()
        now = datetime.now()
        if self._session_start_at is None:
            self._session_start_at = now
        for q in quotes:
            self._quotes_cache[q.symbol] = q
            self._update_tick_volume_state(q)
            self._refresh_entry_signal_if_stale(q.symbol, now)

        self._cleanup_stale_entry_signals(now)
        self._check_market_regime(quotes=quotes)

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
            now = datetime.now()
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

        # 서킷 브레이커: 순실현손익 기준으로 판단
        realized_net = self._effective_realized_net_for_breaker()
        if not self._hard_stop_bypass_for_day:
            daily_loss_limit = int(self._get_regime_value("daily_loss_limit", self.cfg.daily_loss_limit))
            daily_profit_target = int(self._get_regime_value("daily_profit_target", self.cfg.daily_profit_target))

            if realized_net <= daily_loss_limit:
                logger.warning(
                    "일일 손실한도 도달! (순실현: %s원) → 전량 청산 후 거래 중지",
                    f"{realized_net:,}",
                )
                self._alerts.send(
                    event_key="daily_loss_limit_hit",
                    title="일일 손실한도 도달",
                    message=f"순실현손익 {realized_net:,}원으로 손실한도에 도달했습니다. 전량 청산 후 거래를 중지합니다.",
                    level="error",
                    cooldown_seconds=1800,
                )
                self._halted = True
                self._halt_date = datetime.now().date()
                return self._liquidate_all()

            if realized_net >= daily_profit_target:
                logger.info(
                    "일일 목표 달성! (순실현: %s원) → 전량 청산 후 거래 중지",
                    f"{realized_net:,}",
                )
                self._alerts.send(
                    event_key="daily_profit_target_hit",
                    title="일일 목표 달성",
                    message=f"순실현손익 {realized_net:,}원으로 목표를 달성했습니다. 전량 청산 후 거래를 중지합니다.",
                    level="info",
                    cooldown_seconds=1800,
                )
                self._halted = True
                self._halt_date = datetime.now().date()
                return self._liquidate_all()

            # 보조 손실컷: 순손익 추정(순실현 + 미실현 추정) 기준
            if self.cfg.enable_unrealized_loss_guard:
                total_loss_limit = (
                    self._get_regime_value(
                        "daily_total_loss_limit",
                        self.cfg.daily_total_loss_limit
                        if self.cfg.daily_total_loss_limit is not None
                        else self.cfg.daily_loss_limit,
                    )
                )
                unrealized_net = self._estimate_unrealized_net_pnl()
                total_net = realized_net + unrealized_net
                if total_net <= total_loss_limit:
                    logger.warning(
                        "보조 손실컷 도달! (순실현: %s원, 미실현추정: %s원, 합계: %s원) "
                        "→ 전량 청산 후 거래 중지",
                        f"{realized_net:,}",
                        f"{unrealized_net:,}",
                        f"{total_net:,}",
                    )
                    self._alerts.send(
                        event_key="daily_total_loss_limit_hit",
                        title="보조 손실컷 도달",
                        message=(
                            f"순실현 {realized_net:,}원, 미실현추정 {unrealized_net:,}원, "
                            f"합계 {total_net:,}원으로 보조 손실컷에 도달했습니다."
                        ),
                        level="error",
                        cooldown_seconds=1800,
                    )
                    self._halted = True
                    self._halt_date = datetime.now().date()
                    return self._liquidate_all()
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
                if inv_count + pending_inv >= self.cfg.inverse_max_positions:
                    break
                if q.symbol not in self.positions:
                    order = self._evaluate_inverse_buy(q, pending_orders=orders)
                    if order:
                        orders.append(order)

        return orders

    def on_order_filled(self, result: OrderResult):
        if result.side == OrderSide.BUY:
            if not result.success:
                return

            fill_price = result.price
            if fill_price <= 0:
                cached = self._quotes_cache.get(result.symbol)
                fill_price = cached.current_price if cached else 0
            if fill_price <= 0:
                return

            buy_notional = fill_price * result.quantity
            buy_fee = self._calc_commission_cost(buy_notional)
            if buy_fee > 0:
                self.daily_pnl.fees_paid += buy_fee
                # 매수 수수료는 체결 시점에 확정 비용으로 반영
                self.daily_pnl.realized_net_pnl -= buy_fee

            existing = self.positions.get(result.symbol)
            if existing:
                total_qty = existing.quantity + result.quantity
                total_invested = existing.invested_amount + (fill_price * result.quantity)
                existing.quantity = total_qty
                existing.invested_amount = total_invested
                existing.buy_price = int(round(total_invested / total_qty))
                existing.is_restored = False
                if fill_price > existing.high_since_buy:
                    existing.high_since_buy = fill_price
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
                self._save_daily_state()
                return

            self.positions[result.symbol] = PositionState(
                symbol=result.symbol,
                buy_price=fill_price,
                quantity=result.quantity,
                invested_amount=fill_price * result.quantity,
                is_restored=False,
            )
            tag = "[INV] " if result.symbol in self._inverse_symbols else ""
            logger.info("%s매수 체결: %s %d주 @ %s원",
                        tag, result.symbol, result.quantity, f"{fill_price:,}")
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

            pos = self.positions.pop(result.symbol, None)
            if pos:
                sell_price = result.price
                if sell_price <= 0:
                    cached = self._quotes_cache.get(result.symbol)
                    sell_price = cached.current_price if cached else pos.buy_price

                gross_pnl = (sell_price - pos.buy_price) * pos.quantity
                sell_notional = sell_price * pos.quantity
                sell_fee = self._calc_commission_cost(sell_notional)
                sell_tax_slippage = self._calc_sell_tax_slippage_cost(sell_notional)
                net_pnl = gross_pnl - sell_fee - sell_tax_slippage

                self.daily_pnl.realized_gross_pnl += gross_pnl
                self.daily_pnl.realized_net_pnl += net_pnl
                self.daily_pnl.fees_paid += sell_fee
                self.daily_pnl.taxes_paid += sell_tax_slippage
                self.daily_pnl.trade_count += 1

                if net_pnl < 0:
                    self._global_loss_cooldown_until = datetime.now() + timedelta(
                        seconds=self._regime_loss_cooldown_seconds()
                    )

                self._sell_cooldown[result.symbol] = datetime.now()

                tag = "[INV] " if result.symbol in self._inverse_symbols else ""
                logger.info(
                    "%s매도 체결: %s %d주 @ %s원 "
                    "(총손익: %s원, 순손익: %s원, 누적순손익: %s원)",
                    tag, result.symbol, result.quantity, f"{sell_price:,}",
                    f"{gross_pnl:,}", f"{net_pnl:,}", f"{self.daily_pnl.realized_net_pnl:,}",
                )

                self._save_daily_state()

    def should_continue(self) -> bool:
        if self._halted and not self.positions:
            return False
        return True

    def _effective_max_position_count(self) -> int:
        """동시 보유 가능한 최대 종목 수를 계산한다.

        max_position_count <= 0 이면 seed_money/per_stock_amount 기준으로 자동 계산한다.
        """
        if self.cfg.max_position_count > 0:
            return self.cfg.max_position_count

        per_stock = max(1, self.cfg.per_stock_amount)
        auto_count = max(1, self.cfg.seed_money // per_stock)
        return auto_count

    def _resolve_regime_profile_name(self, bear_score: Optional[int] = None) -> str:
        if not self.cfg.enable_regime_adaptive:
            return "static"
        score = self._bear_score if bear_score is None else bear_score
        if self._is_bullish_regime() and score == 0:
            # 상승/평탄 구간에서는 과도한 보수화를 피하기 위해 '강세' 기준으로 운용한다.
            # 과거 score 0이 연속 발생하면 중립 모드가 과도하게 보수적으로 동작해 기회를 놓친 이슈 대응.
            return "bull"
        if self._is_bullish_regime() and score > 0:
            return "bull"
        if score >= 2:
            return "bear"
        return "neutral"

    def _is_early_session_guard_active(self) -> bool:
        if not self.cfg.enable_early_session_guard or self._session_start_at is None:
            return False
        elapsed = (datetime.now() - self._session_start_at).total_seconds()
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
        minute_key = datetime.now().strftime("%Y-%m-%d %H:%M")
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
                "min_change_rate": self.cfg.min_change_rate,
                "bullish_min_change_rate": self.cfg.bullish_min_change_rate,
                "min_momentum_score": self.cfg.min_momentum_score,
                "bullish_min_momentum_score": self.cfg.bullish_min_momentum_score,
                "volume_spike_ratio": self.cfg.volume_spike_ratio,
                "volume_spike_ratio_min": self.cfg.volume_spike_ratio_min,
                "volume_spike_abs_min": self.cfg.volume_spike_abs_min,
                "take_profit_pct": self.cfg.take_profit_pct,
                "per_position_stop_loss": self.cfg.per_position_stop_loss,
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

        if profile_name == "bull":
            return {
                "min_change_rate": max(0.8, self.cfg.min_change_rate * 0.9),
                "bullish_min_change_rate": max(0.8, self.cfg.bullish_min_change_rate * 0.9),
                "min_momentum_score": max(3.0, self.cfg.min_momentum_score * 0.95),
                "bullish_min_momentum_score": max(3.0, self.cfg.bullish_min_momentum_score * 0.95),
                "volume_spike_ratio": max(1.0, self.cfg.volume_spike_ratio * 0.72),
                "volume_spike_ratio_min": max(0.85, self.cfg.volume_spike_ratio_min * 0.85),
                "volume_spike_abs_min": int(max(1_000, self.cfg.volume_spike_abs_min * 0.75)),
                "take_profit_pct": self.cfg.take_profit_pct * 1.1,
                "per_position_stop_loss": int(self.cfg.per_position_stop_loss * 1.2),
                "trailing_stop_pct": self.cfg.trailing_stop_pct * 1.1,
                "trailing_stop_activation_gain_pct": self.cfg.trailing_stop_activation_gain_pct * 1.1,
                "max_position_holding_minutes": int(self.cfg.max_position_holding_minutes * 1.5),
                "cooldown_seconds": max(60, int(self.cfg.cooldown_seconds * 0.6)),
                "loss_trade_cooldown_seconds": max(30, int(self.cfg.loss_trade_cooldown_seconds * 0.7)),
                "expected_move_pct": self.cfg.expected_move_pct * 1.35,
                "min_expected_net_profit": max(120, int(self.cfg.min_expected_net_profit * 0.8)),
                "min_expected_rr_ratio": max(0.78, self.cfg.min_expected_rr_ratio * 0.95),
                "per_stock_alloc_scale": 1.15,
                "max_stock_alloc_scale": 1.10,
                "daily_loss_limit": int(self.cfg.daily_loss_limit * 1.6),
                "daily_profit_target": int(self.cfg.daily_profit_target * 1.2),
                "daily_total_loss_limit": int(
                    (
                        self.cfg.daily_total_loss_limit
                        if self.cfg.daily_total_loss_limit is not None
                        else self.cfg.daily_loss_limit
                    ) * 1.4
                ),
                "min_bear_score_for_new_long": self.cfg.min_bear_score_for_new_long,
                "bear_market_entry_score": self.cfg.bear_market_entry_score,
                "inverse_take_profit_pct": max(0.45, self.cfg.inverse_take_profit_pct * 0.85),
                "inverse_stop_loss_pct": max(-0.35, self.cfg.inverse_stop_loss_pct * 0.8),
                "inverse_trailing_stop_pct": max(-0.25, self.cfg.inverse_trailing_stop_pct * 0.75),
                "inverse_trailing_stop_activation_gain_pct": max(
                    0.2, self.cfg.inverse_trailing_stop_activation_gain_pct * 0.8
                ),
                "inverse_max_hold_minutes": max(20, int(self.cfg.inverse_max_hold_minutes * 0.6)),
            }

        if profile_name == "bear":
            return {
                "min_change_rate": self.cfg.min_change_rate * (1.25 if self._bear_score >= 2 else 1.15),
                "bullish_min_change_rate": self.cfg.bullish_min_change_rate * 1.2,
                "min_momentum_score": self.cfg.min_momentum_score * 1.2,
                "bullish_min_momentum_score": self.cfg.bullish_min_momentum_score * 1.08,
                "volume_spike_ratio": self.cfg.volume_spike_ratio * (1.22 if self._bear_score >= 2 else 1.05),
                "volume_spike_ratio_min": min(2.5, self.cfg.volume_spike_ratio_min * (1.14 if self._bear_score >= 2 else 1.05)),
                "volume_spike_abs_min": int(self.cfg.volume_spike_abs_min * (1.4 if self._bear_score >= 2 else 1.2)),
                "take_profit_pct": self.cfg.take_profit_pct * (0.84 if self._bear_score >= 2 else 0.9),
                "per_position_stop_loss": int(self.cfg.per_position_stop_loss * 0.68),
                "trailing_stop_pct": self.cfg.trailing_stop_pct * 0.86 if self._bear_score >= 2 else self.cfg.trailing_stop_pct * 0.93,
                "trailing_stop_activation_gain_pct": self.cfg.trailing_stop_activation_gain_pct * 0.75,
                "max_position_holding_minutes": int(self.cfg.max_position_holding_minutes * 0.62),
                "cooldown_seconds": int(self.cfg.cooldown_seconds * 1.25),
                "loss_trade_cooldown_seconds": int(self.cfg.loss_trade_cooldown_seconds * 1.35),
                "expected_move_pct": self.cfg.expected_move_pct * (0.92 if self._bear_score >= 2 else 0.97),
                "min_expected_net_profit": max(220, int(self.cfg.min_expected_net_profit * 0.85)),
                "min_expected_rr_ratio": min(0.95, self.cfg.min_expected_rr_ratio * 1.05),
                "per_stock_alloc_scale": 0.80 if self._bear_score >= 2 else 0.9,
                "max_stock_alloc_scale": 0.72 if self._bear_score >= 2 else 0.85,
                "daily_loss_limit": int(self.cfg.daily_loss_limit * 0.72),
                "daily_profit_target": int(self.cfg.daily_profit_target * 0.82),
                "daily_total_loss_limit": int(
                    (
                        self.cfg.daily_total_loss_limit
                        if self.cfg.daily_total_loss_limit is not None
                        else self.cfg.daily_loss_limit
                    ) * 0.82
                ),
                "min_bear_score_for_new_long": max(1, self.cfg.min_bear_score_for_new_long + 1),
                "bear_market_entry_score": self.cfg.bear_market_entry_score * 1.02,
                "inverse_take_profit_pct": min(1.8, self.cfg.inverse_take_profit_pct * 1.3),
                "inverse_stop_loss_pct": min(-0.9, self.cfg.inverse_stop_loss_pct * 1.3),
                "inverse_trailing_stop_pct": min(-0.4, self.cfg.inverse_trailing_stop_pct * 1.3),
                "inverse_trailing_stop_activation_gain_pct": min(
                    1.0, self.cfg.inverse_trailing_stop_activation_gain_pct * 1.4
                ),
                "inverse_max_hold_minutes": int(self.cfg.inverse_max_hold_minutes * 1.6),
            }

        return {
            "min_change_rate": self.cfg.min_change_rate * 1.05,
            "bullish_min_change_rate": self.cfg.bullish_min_change_rate,
            "min_momentum_score": self.cfg.min_momentum_score * 1.05,
            "bullish_min_momentum_score": self.cfg.bullish_min_momentum_score * 1.01,
            "volume_spike_ratio": self.cfg.volume_spike_ratio * 1.05,
            "volume_spike_ratio_min": min(2.2, self.cfg.volume_spike_ratio_min * 1.02),
            "volume_spike_abs_min": int(self.cfg.volume_spike_abs_min * 1.12),
            "take_profit_pct": self.cfg.take_profit_pct * 0.95,
            "per_position_stop_loss": int(self.cfg.per_position_stop_loss * 0.72),
            "trailing_stop_pct": self.cfg.trailing_stop_pct * 0.9,
            "trailing_stop_activation_gain_pct": self.cfg.trailing_stop_activation_gain_pct * 0.8,
            "max_position_holding_minutes": int(self.cfg.max_position_holding_minutes * 0.72),
            "cooldown_seconds": int(self.cfg.cooldown_seconds * 1.05),
            "loss_trade_cooldown_seconds": int(self.cfg.loss_trade_cooldown_seconds * 1.1),
            "expected_move_pct": self.cfg.expected_move_pct * 0.97,
            "min_expected_net_profit": max(200, int(self.cfg.min_expected_net_profit * 0.85)),
            "min_expected_rr_ratio": min(0.92, self.cfg.min_expected_rr_ratio * 1.0),
            "per_stock_alloc_scale": 0.93,
            "max_stock_alloc_scale": 0.82,
            "daily_loss_limit": int(self.cfg.daily_loss_limit * 0.9),
            "daily_profit_target": int(self.cfg.daily_profit_target * 0.9),
            "daily_total_loss_limit": int(
                (
                    self.cfg.daily_total_loss_limit
                    if self.cfg.daily_total_loss_limit is not None
                    else self.cfg.daily_loss_limit
                ) * 0.88
            ),
            "min_bear_score_for_new_long": self.cfg.min_bear_score_for_new_long + 1,
            "bear_market_entry_score": self.cfg.bear_market_entry_score * 1.01,
            "inverse_take_profit_pct": self.cfg.inverse_take_profit_pct,
            "inverse_stop_loss_pct": self.cfg.inverse_stop_loss_pct,
            "inverse_trailing_stop_pct": self.cfg.inverse_trailing_stop_pct,
            "inverse_trailing_stop_activation_gain_pct": self.cfg.inverse_trailing_stop_activation_gain_pct,
            "inverse_max_hold_minutes": self.cfg.inverse_max_hold_minutes,
        }

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

    def _regime_per_stock_alloc_scale(self) -> float:
        return float(self._get_regime_value("per_stock_alloc_scale", 1.0))

    def _regime_max_stock_alloc_scale(self) -> float:
        return float(self._get_regime_value("max_stock_alloc_scale", 1.0))

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
        now = datetime.now()

        for p in account_positions or []:
            qty = int(p.quantity or 0)
            if qty <= 0:
                continue

            avg_price = int(round(float(p.avg_price or 0)))
            if avg_price <= 0:
                continue

            current_price = int(p.current_price or 0)
            synced[p.symbol] = PositionState(
                symbol=p.symbol,
                buy_price=avg_price,
                quantity=qty,
                invested_amount=avg_price * qty,
                buy_time=now,
                high_since_buy=max(avg_price, current_price) if current_price > 0 else avg_price,
                is_restored=True,
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
            self._last_pool_refresh = datetime.now()
            return

        pool = set(self.cfg.static_watchlist)
        appeared: set[str] = set()

        # 인버스 ETF 추가
        if self.cfg.inverse_enabled:
            for sym in self.cfg.inverse_etfs:
                pool.add(sym)

        if self.market_data:
            try:
                rising = self.market_data.get_fluctuation_ranking(
                    count=self.cfg.dynamic_pool_size,
                    min_change_rate=self.cfg.min_change_rate,
                    max_change_rate=self.cfg.max_change_rate,
                    min_price=self.cfg.min_price,
                    min_volume=self.cfg.min_volume,
                )
                for item in rising:
                    appeared.add(item.symbol)
                logger.info("동적 풀 갱신: 등락률 상위 %d개 추가 (총 %d종목)",
                            len(rising), len(pool))
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
            for sym in appeared:
                if sym in persistent:
                    pool.add(sym)
            if appeared:
                logger.info("풀 지속성 반영: %d개 동적 후보 중 %d개 채택",
                            len(appeared), len(persistent))
        else:
            pool.update(appeared)

        self._pool = list(pool)[:55]  # 인버스 포함하여 여유 확보
        self._last_pool_refresh = datetime.now()

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
        return self._bear_score <= 1

    def _check_market_regime(self, quotes: Optional[List[Quote]] = None, force: bool = False):
        """KOSPI + 실시간 후보군 추세를 결합해 약세 점수를 계산한다."""
        now = datetime.now()
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
            self._bear_score = 0
            self._bear_market = False
            self._last_regime_check_at = now
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
            index_score = 0
            index_info = None
            index_error = None
            try:
                end_date = datetime.now().strftime("%Y%m%d")
                start_date = (datetime.now() - timedelta(days=45)).strftime("%Y%m%d")

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

                        index_score = score
                        index_info = (float(current), float(ma20), float(ma5))
                self._cached_index_regime_score = index_score
                self._cached_index_regime_info = index_info
                self._cached_index_regime_error = None
                self._last_index_regime_check_at = now
            except Exception as e:
                index_error = e
                self._cached_index_regime_error = e
                self._cached_index_regime_info = None
                self._last_index_regime_check_at = now

        if index_score is None:
            index_score = 0
        if index_info is None:
            index_info = (0.0, 0.0, 0.0)

        index_weight = 0.7
        quote_weight = 0.3
        quote_score_value = quote_score or 0
        if quote_avg_change is None:
            quote_avg_change = 0.0
        if quote_decline_ratio is None:
            quote_decline_ratio = 0.0

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

        final_score = max(0, min(final_score, 3))
        self._bear_score = final_score
        self._bear_market = final_score >= 2
        self._last_regime_check_at = now

        if index_info is not None and quote_score is not None:
            logger.info(
                "시장 레짐 계산: index=%d(가중 %.2f), quote=%d(평균%.2f%%, 하락비율%.1f%%), 최종=%d(이전=%s, 원시=%.2f), "
                "지수=%s, KOSPI: %.1f, MA20: %.1f, MA5: %.1f",
                final_score,
                raw_final_score,
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
            if is_opening_guard and self._is_bullish_regime():
                min_change_rate += self.cfg.early_session_min_change_rate_boost
            if q.change_rate < min_change_rate:
                continue
            score = self._calc_momentum_score(q)
            min_score = self._regime_min_momentum_score()
            if self._is_bullish_regime():
                min_score = self._regime_bullish_min_momentum_score()
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
            if not self._is_volume_spike(q, score=score):
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
            now = datetime.now()
        if self._is_new_entry_window_blocked(now):
            return False

        if self._global_loss_cooldown_until:
            if now < self._global_loss_cooldown_until:
                logger.info("전역 리스크 쿨다운: %s (종료: %s)",
                            quote.symbol, self._global_loss_cooldown_until)
                return False

        if any(
            o.side == OrderSide.BUY and o.symbol == quote.symbol
            for o in pending_orders or []
        ):
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
            required_ticks = 1
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

        pullback_pct = (quote.current_price - signal.first_price) / signal.first_price * 100
        if pullback_pct < self.cfg.entry_confirmation_max_pullback_pct:
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
                self.cfg.entry_confirmation_max_pullback_pct,
            )
            return False

        if score + self.cfg.entry_confirmation_min_score_tolerance < signal.best_score:
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
        if quote.change_rate < self.cfg.pullback_activation_change_rate:
            return True

        pullback_drop_pct = (quote.high_price - quote.current_price) / quote.high_price * 100
        if pullback_drop_pct < self.cfg.pullback_required_min_drop_pct:
            self._log_entry_filter_once_per_minute(
                quote.symbol,
                "pullback_wait",
                "눌림목 대기: %s 고점대비 조정 %.2f%% < 최소 %.2f%%",
                quote.symbol,
                pullback_drop_pct,
                self.cfg.pullback_required_min_drop_pct,
            )
            return False
        if pullback_drop_pct > self.cfg.pullback_allowed_max_drop_pct:
            self._log_entry_filter_once_per_minute(
                quote.symbol,
                "pullback_broken",
                "눌림목 이탈: %s 고점대비 조정 %.2f%% > 최대 %.2f%%",
                quote.symbol,
                pullback_drop_pct,
                self.cfg.pullback_allowed_max_drop_pct,
            )
            return False

        vs_open = (quote.current_price - quote.open_price) / quote.open_price * 100
        if vs_open < self.cfg.pullback_min_vs_open_pct:
            self._log_entry_filter_once_per_minute(
                quote.symbol,
                "pullback_vs_open",
                "눌림목 취소: %s 시가대비 %.2f%% < 최소 %.2f%%",
                quote.symbol,
                vs_open,
                self.cfg.pullback_min_vs_open_pct,
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

        now = datetime.now()
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
            elapsed = (datetime.now() - last_sold).total_seconds()
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
        if not self._is_volume_spike(quote, score=score):
            return None
        if is_scale_in:
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
            entry_threshold = self._regime_min_momentum_score()
            if self._is_bullish_regime():
                entry_threshold = self._regime_bullish_min_momentum_score()
            if score < entry_threshold:
                return None
            if not self._passes_pullback_entry_filter(quote, is_scale_in=is_scale_in):
                self._entry_signals.pop(quote.symbol, None)
                return None

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

        if not is_scale_in and not self._can_open_new_long(
            quote,
            pending_orders=pending_orders,
            score=score,
            now=now,
        ):
            return None

        if not self._can_confirm_entry(
            quote=quote,
            score=score,
            is_scale_in=is_scale_in,
            now=now,
        ):
            return None

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
            logger.info(
                "매수 신호: %s(%s) 점수=%.1f, %d주 @ %s원 (할당 %s원)",
                quote.name,
                quote.symbol,
                score,
                quantity,
                f"{quote.current_price:,}",
                f"{alloc:,}",
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
        now = datetime.now()
        if self._is_new_entry_window_blocked(now):
            return None
        if quote.current_price <= 0 or quote.open_price <= 0:
            return None
        if quote.change_rate < self.cfg.inverse_min_change_rate:
            return None
        if not self._is_volume_spike(quote):
            return None

        if self._bear_score < max(self.cfg.bearish_threshold, self.cfg.inverse_min_bear_score):
            return None

        if self._global_loss_cooldown_until and now < self._global_loss_cooldown_until:
            logger.info("[INV] 전역 리스크 쿨다운: %s (종료: %s)",
                        quote.symbol, self._global_loss_cooldown_until)
            return None

        # 쿨다운 체크
        last_sold = self._sell_cooldown.get(quote.symbol)
        if last_sold:
            elapsed = (now - last_sold).total_seconds()
            if elapsed < self._regime_cooldown_seconds():
                return None

        # 모멘텀 점수 (인버스도 상승 중이어야 진입)
        score = self._calc_momentum_score(quote)
        if score < self.cfg.inverse_min_momentum:
            return None

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

        logger.info("[INV] 매수 신호: %s 약세점수=%d, 모멘텀=%.1f, %d주 @ %s원",
                     quote.symbol, self._bear_score, score, quantity,
                     f"{quote.current_price:,}")

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
    ) -> int:
        total_exposure = self._get_total_exposure()
        stock_exposure = self._get_stock_exposure(symbol)

        pending_total = 0
        pending_stock = 0
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

        total_room = self.cfg.seed_money - (total_exposure + pending_total)
        stock_room = self.cfg.max_per_stock_amount - (stock_exposure + pending_stock)
        if current_price <= 0:
            return 0

        per_stock_amount = int(self.cfg.per_stock_amount * self._regime_per_stock_alloc_scale())
        max_stock_amount = int(self.cfg.max_per_stock_amount * self._regime_max_stock_alloc_scale())
        stock_room = max_stock_amount - (stock_exposure + pending_stock)
        alloc = min(per_stock_amount, total_room, stock_room)
        if alloc <= 0:
            return 0

        # 비싼 종목(할당액 < 현재가)도 1주라도 살 수 있으면 최소 1주 주문을 허용한다.
        if alloc < current_price:
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

    def _evaluate_sell(self, quote: Quote) -> Optional[Order]:
        """익절/손절/추적손절 판단 (일반 주식)."""
        pos = self.positions.get(quote.symbol)
        if not pos:
            return None

        pnl_pct = (quote.current_price - pos.buy_price) / pos.buy_price * 100
        pnl_amount = (quote.current_price - pos.buy_price) * pos.quantity

        holding_minutes = (datetime.now() - pos.buy_time).total_seconds() / 60
        if holding_minutes >= self._regime_max_holding_minutes() and not pos.is_restored:
            logger.info("보유시간 초과 청산: %s (%.1f분)",
                        quote.symbol, holding_minutes)
            return self._make_sell_order(pos)

        # 익절
        if pnl_pct >= self._regime_take_profit_pct():
            logger.info("익절: %s %.2f%% (%s원)",
                        quote.symbol, pnl_pct, f"{pnl_amount:,}")
            return self._make_sell_order(pos)

        # 개별 포지션 손절 (금액 기준)
        if pnl_amount <= self._regime_per_position_stop_loss():
            logger.info("개별손절: %s %s원 (한도 %s원)",
                        quote.symbol, f"{pnl_amount:,}",
                        f"{self._regime_per_position_stop_loss():,}")
            return self._make_sell_order(pos)

        # 추적손절 (고점 대비)
        if pos.high_since_buy > pos.buy_price:
            drop_from_high = (quote.current_price - pos.high_since_buy) / pos.high_since_buy * 100
            gain_from_entry = (pos.high_since_buy - pos.buy_price) / pos.buy_price * 100
            if (
                gain_from_entry >= self._regime_trailing_stop_activation_gain_pct()
                and drop_from_high <= self._regime_trailing_stop_pct()
            ):
                logger.info("추적손절: %s 고점 %s → 현재 %s (%.2f%%)",
                            quote.symbol, f"{pos.high_since_buy:,}",
                            f"{quote.current_price:,}", drop_from_high)
                return self._make_sell_order(pos)

        return None

    def _effective_realized_net_for_breaker(self) -> int:
        """일일 브레이커 판단에 사용할 순실현손익을 반환한다."""
        return int(self.daily_pnl.realized_net_pnl - self._daily_breaker_pnl_offset)

    def _evaluate_inverse_sell(self, quote: Quote) -> Optional[Order]:
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

        # 1. 익절
        if pnl_pct >= self._regime_inverse_take_profit_pct():
            logger.info("[INV] 익절: %s %.2f%%", quote.symbol, pnl_pct)
            return self._make_sell_order(pos)

        # 2. 손절 (-0.5%, 타이트)
        if pnl_pct <= self._regime_inverse_stop_loss_pct():
            logger.info("[INV] 손절: %s %.2f%%", quote.symbol, pnl_pct)
            return self._make_sell_order(pos)

        # 3. 시간 초과 청산 (음의 복리 방지, 실거래 모드만)
        if self.market_data is not None:
            hold_minutes = (datetime.now() - pos.buy_time).total_seconds() / 60
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
                logger.info("[INV] 추적손절: %s 고점 %s → 현재 %s (%.2f%%)",
                            quote.symbol, f"{pos.high_since_buy:,}",
                            f"{quote.current_price:,}", drop_from_high)
                return self._make_sell_order(pos)

        return None

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
        if price_range > 0:
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

    def _make_sell_order(self, pos: PositionState) -> Order:
        return Order(
            symbol=pos.symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=pos.quantity,
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

        risk_amount = max(1, abs(self._regime_per_position_stop_loss()))
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
