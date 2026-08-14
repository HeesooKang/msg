"""State and configuration for the single expected-value trading path."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


DEFAULT_STATIC_WATCHLIST = [
    "005930",
    "000660",
    "373220",
    "207940",
    "005490",
    "006400",
    "051910",
    "035420",
    "000270",
    "005380",
    "035720",
    "105560",
    "055550",
    "012330",
    "066570",
    "003670",
    "028260",
    "032830",
    "003550",
    "086790",
    "034730",
    "015760",
    "017670",
    "009150",
    "010130",
    "033780",
    "018260",
    "011200",
    "138930",
    "024110",
    "152100",
    "292190",
]

DEFAULT_INVERSE_ETFS = [
    "114800",
    "123310",
    "251340",
    "464930",
]

@dataclass
class MomentumScalpConfig:
    seed_money: int = 1_000_000
    daily_profit_target: int = 10_000
    daily_loss_limit: int = -5_000

    commission_rate: float = 0.00015
    sell_tax_rate: float = 0.002

    static_watchlist: List[str] = field(
        default_factory=lambda: list(DEFAULT_STATIC_WATCHLIST)
    )
    inverse_etfs: List[str] = field(
        default_factory=lambda: list(DEFAULT_INVERSE_ETFS)
    )
    dynamic_pool_size: int = 30
    dynamic_pool_ranking_fetch_count: int = 60
    dynamic_pool_quote_min_change_rate: float = 0.0
    pool_refresh_interval: int = 120
    quote_freshness_seconds: int = 5

    forecast_outcome_root: str = "reports/forecast-outcomes"
    daily_state_path: str = "state/momentum_scalp_daily_state.json"
    state_save_interval_seconds: int = 5
    pending_order_block_seconds: int = 180
    allow_hard_stop_bypass_for_day: bool = False

@dataclass(frozen=True)
class TradePlan:
    symbol: str
    allowed: bool
    reason: str
    quantity: int
    entry_limit_price: int
    expected_exit_bid: int
    lower_exit_bid: int
    expected_net: float
    lower_net: int
    committed_risk_net_abs: int
    signal_at: datetime
    expires_at: datetime


@dataclass
class PositionState:
    symbol: str
    buy_price: int
    quantity: int
    buy_time: datetime = field(default_factory=datetime.now)
    plan: Optional[TradePlan] = None
    trade_key: str = ""
    forecast_id: str = ""

    pending_exit_started_at: Optional[datetime] = None
    pending_exit_quantity: int = 0
    pending_exit_reason: str = ""

    def __post_init__(self) -> None:
        self.quantity = max(0, int(self.quantity or 0))
        self.buy_price = max(0, int(self.buy_price or 0))
        if not self.trade_key:
            timestamp = self.buy_time.isoformat(timespec="seconds")
            self.trade_key = f"{self.symbol}:{timestamp}"


@dataclass
class DailyPnL:
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
