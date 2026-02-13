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
        max_position_count=5,             # 최대 5종목 (롱)
        per_stock_amount=200_000,         # 종목당 20만원 (고가주는 1주)
        daily_profit_target=10_000,       # 일일 목표 +1만원 (총손익 기준)
        daily_loss_limit=-5_000,          # 일일 손실한도 -5천원
        per_position_stop_loss=-5_000,    # 포지션당 손절 -5천원
        take_profit_pct=1.5,              # 익절 +1.5%
        trailing_stop_pct=-0.7,           # 추적손절 -0.7%
        bear_market_mode='A',             # 'A'=공격적, 'B'=보수적
        cooldown_seconds=600,             # 매도 후 10분 재매수 금지
        # 인버스 ETF 설정
        inverse_enabled=True,             # 약세장 인버스 ETF 매매 활성화
        inverse_max_positions=2,          # 인버스 최대 2종목
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
