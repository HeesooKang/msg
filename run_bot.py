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
        max_position_count=3,             # 최대 3종목 (롱)
        per_stock_amount=180_000,         # 종목당 18만원
        max_per_stock_amount=350_000,     # 종목당 최대 노출 35만원
        daily_profit_target=10_000,       # 일일 목표 +1만원 (총손익 기준)
        daily_loss_limit=-6_000,          # 일일 손실한도 -6천원
        daily_total_loss_limit=-9_000,    # 보조 손실컷(순실현+미실현 추정): -9천원
        per_position_stop_loss=-6_000,    # 포지션당 손절 -6천원
        take_profit_pct=2.0,              # 익절 +2.0%
        trailing_stop_pct=-0.5,           # 추적손절 -0.5%
        min_momentum_score=2.8,           # 기본 진입 점수 상향
        min_change_rate=1.0,              # 등락률 하한 상향
        min_volume=150_000,               # 거래량 하한 완화
        min_price=2_000,                  # 저가주 제외
        enable_expected_net_filter=True,   # 진입 전 기대순익 필터
        expected_move_pct=2.0,            # 기대 상승폭 +2.0% 가정
        min_expected_net_profit=800,      # 최소 기대순익 완화
        min_expected_rr_ratio=0.25,       # 최소 기대 RR 완화
        # 거래량 스파이크 강제
        enable_volume_spike_filter=True,
        volume_spike_min_history=3,
        volume_spike_ratio=1.6,
        volume_spike_abs_min=3_000,
        # 모멘텀 진입 보강
        enable_entry_confirmation=True,       # 2연속 재확인 후 진입
        entry_confirmation_ticks=2,
        scale_in_confirmation_ticks=1,
        entry_confirmation_window_seconds=240,
        entry_confirmation_min_score_tolerance=0.25,
        entry_confirmation_max_pullback_pct=-0.6,   # 최초 대비 -0.6% 허용
        overheated_jump_change_pct=3.5,
        overheated_retrace_ratio=0.9,
        # 동적 풀 지속성 게이트
        enable_pool_persistence_gate=True,
        momentum_pool_persistence_window=3,
        momentum_pool_min_appearances=2,
        bear_market_mode='A',             # 'A'=공격적(신규 롱 허용), 'B'=보수적
        cooldown_seconds=300,             # 매도 후 5분 재매수 금지
        # 인버스 ETF 설정
        inverse_enabled=True,             # 약세장 인버스 ETF 매매 활성화
        inverse_max_positions=1,          # 인버스 최대 1종목
        bearish_threshold=2,              # 약세 점수 2 이상 시 진입
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
