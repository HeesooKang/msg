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
    CONDITIONAL = "02"  # 조건부지정가
    BEST = "03"        # 최유리지정가
    PRIORITY = "04"    # 최우선지정가
    PRE_MARKET = "05"  # 장전 시간외
    POST_MARKET = "06" # 장후 시간외
    TIME_EXT = "07"    # 시간외 단일가


@dataclass
class Quote:
    symbol: str           # 종목코드
    name: str             # 종목명
    current_price: int    # 현재가
    change: int           # 전일 대비
    change_rate: float    # 등락률
    open_price: int       # 시가
    high_price: int       # 고가
    low_price: int        # 저가
    volume: int           # 누적 거래량
    trade_amount: int     # 누적 거래대금
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Order:
    symbol: str
    side: OrderSide
    order_type: OrderType = OrderType.MARKET
    quantity: int = 0
    price: int = 0  # 시장가일 때 0


@dataclass
class OrderResult:
    success: bool
    order_no: str = ""
    message: str = ""
    symbol: str = ""
    side: Optional[OrderSide] = None
    quantity: int = 0
    price: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Position:
    symbol: str           # 종목코드
    name: str             # 종목명
    quantity: int         # 보유수량
    avg_price: float      # 평균매입가
    current_price: int    # 현재가
    eval_amount: int      # 평가금액
    profit_loss: int      # 평가손익
    profit_rate: float    # 수익률(%)


@dataclass
class AccountBalance:
    total_eval_amount: int       # 총평가금액
    total_deposit: int           # 예수금
    total_profit_loss: int       # 총평가손익
    total_profit_rate: float     # 총수익률(%)
    positions: List[Position] = field(default_factory=list)


@dataclass
class RankingItem:
    """순위 조회 결과 항목."""
    symbol: str           # 종목코드
    name: str             # 종목명
    current_price: int    # 현재가
    change_rate: float    # 등락률(%)
    volume: int           # 누적 거래량
    rank: int             # 순위
