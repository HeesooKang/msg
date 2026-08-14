from __future__ import annotations

import signal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.strategies.momentum_scalp import MomentumScalpStrategy


def run_scheduled(
    strategy: MomentumScalpStrategy,
    tick_interval: int = 1,
):
    """스케줄러를 통해 장 시간에 맞춰 자동 실행한다."""
    from src.scheduler import TradingScheduler

    scheduler = TradingScheduler(strategy)

    signal.signal(signal.SIGINT, lambda s, f: scheduler.stop())
    signal.signal(signal.SIGTERM, lambda s, f: scheduler.stop())

    scheduler.run(tick_interval=tick_interval)
