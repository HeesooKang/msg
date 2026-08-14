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
    ask_size: int = 0
    bid_size: int = 0
    total_ask_size: int = 0
    total_bid_size: int = 0
    book_available: bool = False
    flow_available: bool = False
    book_depth_available: bool = False


@dataclass
class Order:
    symbol: str
    side: OrderSide
    order_type: OrderType = OrderType.MARKET
    quantity: int = 0
    price: int = 0  # 시장가일 때 0
    reference_price: int = 0  # 시장가 주문 로그/보정용 기준 시세
    requested_reason: str = ""


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
