#!/usr/bin/env python3
"""KIS 자동매매 봇 실행 스크립트.

사용법:
    직접 실행:   ./dev py run_bot.py
    launchd:     자동 실행됨
"""

import subprocess
import sys
import os

# 프로젝트 루트로 이동
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from src.config import Config
from src.logger_setup import setup_logger
from src.main import run_scheduled


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
        max_position_count=0,             # 0이면 seed_money/per_stock_amount 기준 자동 계산
        per_stock_amount=100_000,         # 종목당 10만원
        max_per_stock_amount=300_000,     # 종목당 최대 노출 30만원
        daily_profit_target=12_000,       # 일일 목표 +1.2만원
        daily_loss_limit=-2_500,          # 일일 손실한도 -2.5천원
        daily_total_loss_limit=-5_000,    # 보조 손실컷(순실현+미실현 추정): -0.5만원
        enable_unrealized_loss_guard=True,  # 미실현추정 손실컷 활성화
        per_position_stop_loss=-1_200,    # 포지션당 손절 -1,200원
        take_profit_pct=2.5,              # 익절 +2.5%
        trailing_stop_pct=-0.7,           # 추적손절 -0.7%
        min_momentum_score=3.8,           # 진입 점수 강화
        min_change_rate=1.2,              # 등락률 하한 강화
        min_volume=180_000,               # 거래량 하한
        min_price=5_000,                  # 저가주 제외 강화
        enable_expected_net_filter=True,   # 진입 전 기대순익 필터
        expected_move_pct=2.4,            # 기대 상승폭 +2.4%
        min_expected_net_profit=800,      # 최소 기대순익 기본값
        min_expected_rr_ratio=0.65,       # 최소 기대 RR 기본값
        # 거래량 스파이크 강제
        enable_volume_spike_filter=True,
        volume_spike_min_history=2,
        volume_spike_ratio=1.8,
        volume_spike_ratio_min=1.2,
        volume_spike_abs_min=4_000,
        bullish_min_change_rate=0.45,
        bullish_min_momentum_score=2.6,
        bullish_volume_spike_ratio_adjustment=0.30,
        bullish_volume_spike_abs_min_ratio=0.6,
        # 모멘텀 진입 보강
        enable_entry_confirmation=True,       # 연속 재확인 후 진입
        entry_confirmation_ticks=2,
        scale_in_confirmation_ticks=1,
        entry_confirmation_window_seconds=240,
        entry_confirmation_min_score_tolerance=0.4,
        entry_confirmation_max_pullback_pct=-0.6,   # 최초 대비 -0.6% 허용
        # 눌림목(리테스트) 필터: 급등 종목은 고점 추격 대신 조정 확인 후 진입
        enable_pullback_entry_filter=True,
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
        bear_market_mode='A',             # 'A'=약세 필터 보완 적용, 'B'=완전 차단
        min_bear_score_for_new_long=2,    # 약세 점수 2 이상이면 신규 롱 차단
        bear_market_entry_score=3.8,      # 약세장에서도 모멘텀 강할 때 예외 허용
        cooldown_seconds=900,             # 매도 후 재매수 쿨다운(레짐별 보정)
        loss_trade_cooldown_seconds=420,  # 손실 체결 후 전역 진입 정지(레짐별 보정)
        trailing_stop_activation_gain_pct=0.8,
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
        inverse_max_positions=1,          # 인버스 최대 1종목
        bearish_threshold=2,              # 약세 점수 2 이상 시 진입
        inverse_min_bear_score=3,         # 인버스는 약세 점수 3 이상에서만 신규진입
        inverse_min_change_rate=1.4,      # 인버스 최소 등락률 +1.4%
        inverse_min_momentum=3.0,         # 인버스 최소 모멘텀 강화
        inverse_trailing_stop_activation_gain_pct=0.45,  # +0.45% 이익 후에만 인버스 추적손절
    )

    return MomentumScalpStrategy(market_data, strategy_config)


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
