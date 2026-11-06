from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "01"      # 시장가
    LIMIT = "00"       # 지정가


@dataclass
class Quote:
    symbol: str           # 종목코드
    current_price: int    # 현재가
    timestamp: datetime = field(default_factory=datetime.now)
    ask_price: int = 0
    bid_price: int = 0
    trade_volume: int = 0
    cumulative_volume: int = 0
    cumulative_trade_amount: int = 0
    cumulative_sell_volume: int = 0
    cumulative_buy_volume: int = 0
    open_price: int = 0
    high_price: int = 0
    low_price: int = 0
    day_change_pct: float = 0.0
    ask_size: int = 0
    bid_size: int = 0
    total_ask_size: int = 0
    total_bid_size: int = 0
    book_available: bool = False
    flow_available: bool = False
    book_depth_available: bool = False
    day_summary_available: bool = False
    security_group: str = ""
    ask_high_price: int = 0  # Observed valid ask high within this second; not an executable price.
    trade_high_price: int = 0  # Highest actual print within this second.

    @property
    def has_valid_prices(self) -> bool:
        if self.current_price <= 0 or self.ask_price < 0 or self.bid_price < 0:
            return False
        if self.ask_price > 0 and self.bid_price > self.ask_price:
            return False
        if self.high_price > 0 and self.low_price > 0:
            return self.low_price <= self.current_price <= self.high_price
        return True

    def is_price_breakout(self, level: int, *, signal_price: int = 0) -> bool:
        """A new-high signal cannot be a reversal below an already observed print."""
        return (level > 0 and self.has_valid_prices and self.book_available
                and 0 < self.bid_price <= self.ask_price
                and min(self.current_price, self.bid_price) > level
                and self.current_price >= max(self.trade_high_price, signal_price))

    def observe_price_peaks(self, other: "Quote") -> None:
        """Merge observed extrema, never executable prices, sizes or freshness."""
        if (self.symbol != other.symbol or not other.has_valid_prices
                or self.timestamp.replace(microsecond=0) != other.timestamp.replace(microsecond=0)):
            return
        self.trade_high_price = max(self.trade_high_price, self.current_price,
                                    other.trade_high_price, other.current_price)
        self.ask_high_price = max(
            self.ask_high_price, other.ask_high_price,
            self.ask_price if self.book_available and 0 < self.bid_price <= self.ask_price else 0,
            other.ask_price if other.book_available and 0 < other.bid_price <= other.ask_price else 0,
        )


@dataclass
class Order:
    symbol: str
    side: OrderSide
    order_type: OrderType = OrderType.MARKET
    quantity: int = 0
    price: int = 0  # 시장가일 때 0
    reference_price: int = 0  # 시장가 주문 로그/보정용 기준 시세
    requested_reason: str = ""
    submit_before: Optional[float] = None  # Epoch deadline for unsent buys only.
    breakout_price: int = 0  # Original signal level, never recalculated by execution.


@dataclass
class OrderResult:
    success: bool
    order_no: str = ""
    message: str = ""
    error_code: str = ""
    error_category: str = ""
    symbol: str = ""
    side: Optional[OrderSide] = None
    quantity: int = 0
    price: int = 0
    requested_price: int = 0
    reference_price: int = 0
    fill_mode: str = ""
    requested_reason: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    requested_quantity: int = 0
    fill_id: str = ""
    remaining_quantity: int = 0


@dataclass
class Position:
    symbol: str           # 종목코드
    quantity: int         # 보유수량
    avg_price: float      # 평균매입가


@dataclass
class AccountBalance:
    total_eval_amount: int       # 총평가금액
    total_deposit: int           # 예수금
    total_profit_loss: int       # 총평가손익
    positions: List[Position] = field(default_factory=list)
