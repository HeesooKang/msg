"""이동평균선 크로스 + 목표수익/손절 전략.

매수: 단기MA가 장기MA를 위로 돌파 (골든크로스)
매도: 목표 수익 도달 OR 손절 라인 도달 OR 데드크로스
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from src.api_client import KISClient
from src.market_data import MarketDataAPI
from src.models import Order, OrderResult, OrderSide, OrderType, Quote
from src.strategy import BaseStrategy

logger = logging.getLogger("kis_trader.strategy.ma_cross")


@dataclass
class StrategyConfig:
    """전략 설정. 여기서 값을 바꾸면 전략 동작이 달라진다."""

    # 감시 종목 리스트 (종목코드)
    watchlist: List[str] = field(default_factory=lambda: [
        "005930",  # 삼성전자
        "000660",  # SK하이닉스
        "035720",  # 카카오
    ])

    # 이동평균선 기간
    short_ma_period: int = 5    # 단기 (5일)
    long_ma_period: int = 20    # 장기 (20일)

    # 매수 금액 (종목당)
    buy_amount: int = 2_500_000  # 250만원

    # 목표 수익 / 손절 (원)
    target_profit: int = 100_000   # +10만원이면 익절
    stop_loss: int = -50_000       # -5만원이면 손절

    # 동시 보유 가능 종목 수
    max_holdings: int = 2


@dataclass
class HoldingInfo:
    """보유 중인 종목 정보."""
    symbol: str
    buy_price: int      # 매수 단가
    quantity: int        # 보유 수량
    buy_time: datetime = field(default_factory=datetime.now)


class MACrossStrategy(BaseStrategy):
    """이동평균선 크로스 + 목표수익/손절 전략."""

    def __init__(self, market_data: MarketDataAPI, config: StrategyConfig = None):
        self.market_data = market_data
        self.cfg = config or StrategyConfig()
        self.holdings: Dict[str, HoldingInfo] = {}  # 보유 종목
        self._running = True

    def initialize(self):
        logger.info("전략 초기화: MA크로스 + 목표수익/손절")
        logger.info("  감시종목: %s", self.cfg.watchlist)
        logger.info("  MA기간: 단기=%d일, 장기=%d일", self.cfg.short_ma_period, self.cfg.long_ma_period)
        logger.info("  매수금액: %s원/종목", f"{self.cfg.buy_amount:,}")
        logger.info("  익절: +%s원, 손절: %s원", f"{self.cfg.target_profit:,}", f"{self.cfg.stop_loss:,}")

    def get_watchlist(self) -> List[str]:
        return self.cfg.watchlist

    def on_tick(self, quote: Quote) -> List[Order]:
        symbol = quote.symbol
        orders = []

        # 이미 보유 중인 종목 → 매도 조건 체크
        if symbol in self.holdings:
            order = self._check_sell(quote)
            if order:
                orders.append(order)

        # 미보유 종목 → 매수 조건 체크
        else:
            if len(self.holdings) < self.cfg.max_holdings:
                order = self._check_buy(quote)
                if order:
                    orders.append(order)

        return orders

    def on_order_filled(self, result: OrderResult):
        if not result.success:
            return

        if result.side == OrderSide.BUY:
            self.holdings[result.symbol] = HoldingInfo(
                symbol=result.symbol,
                buy_price=result.price or 0,
                quantity=result.quantity,
            )
            logger.info("보유 추가: %s %d주 @ %s원", result.symbol, result.quantity, f"{result.price:,}")

        elif result.side == OrderSide.SELL:
            if result.symbol in self.holdings:
                holding = self.holdings.pop(result.symbol)
                logger.info("보유 해제: %s", result.symbol)

    def should_continue(self) -> bool:
        return self._running

    # --- 내부 로직 ---

    def _check_buy(self, quote: Quote) -> Optional[Order]:
        """골든크로스 발생 시 매수 주문을 생성한다."""
        is_golden = self._detect_golden_cross(quote.symbol)
        if not is_golden:
            return None

        # 매수 수량 계산 (매수금액 / 현재가)
        if quote.current_price <= 0:
            return None
        quantity = self.cfg.buy_amount // quote.current_price
        if quantity <= 0:
            return None

        logger.info(
            "골든크로스 감지! %s(%s) → %d주 매수 예정 (약 %s원)",
            quote.name, quote.symbol, quantity,
            f"{quantity * quote.current_price:,}",
        )

        return Order(
            symbol=quote.symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=0,
        )

    def _check_sell(self, quote: Quote) -> Optional[Order]:
        """목표 수익 또는 손절 도달 시 매도 주문을 생성한다."""
        holding = self.holdings.get(quote.symbol)
        if not holding:
            return None

        # 평가손익 계산
        profit = (quote.current_price - holding.buy_price) * holding.quantity

        # 익절
        if profit >= self.cfg.target_profit:
            logger.info(
                "익절! %s: 수익 %s원 (목표 %s원 도달)",
                quote.symbol, f"{profit:,}", f"{self.cfg.target_profit:,}",
            )
            return self._make_sell_order(holding)

        # 손절
        if profit <= self.cfg.stop_loss:
            logger.info(
                "손절! %s: 손실 %s원 (한도 %s원 도달)",
                quote.symbol, f"{profit:,}", f"{self.cfg.stop_loss:,}",
            )
            return self._make_sell_order(holding)

        # 데드크로스 발생 시에도 매도
        is_dead = self._detect_dead_cross(quote.symbol)
        if is_dead:
            logger.info("데드크로스 감지! %s → 매도 (현재 손익: %s원)", quote.symbol, f"{profit:,}")
            return self._make_sell_order(holding)

        return None

    def _make_sell_order(self, holding: HoldingInfo) -> Order:
        return Order(
            symbol=holding.symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=holding.quantity,
            price=0,
        )

    def _detect_golden_cross(self, symbol: str) -> bool:
        """최근 데이터에서 골든크로스가 발생했는지 확인한다."""
        short_ma, long_ma, prev_short_ma, prev_long_ma = self._get_ma_values(symbol)
        if short_ma is None:
            return False

        # 골든크로스: 이전에는 단기 < 장기였는데, 지금은 단기 > 장기
        return prev_short_ma <= prev_long_ma and short_ma > long_ma

    def _detect_dead_cross(self, symbol: str) -> bool:
        """최근 데이터에서 데드크로스가 발생했는지 확인한다."""
        short_ma, long_ma, prev_short_ma, prev_long_ma = self._get_ma_values(symbol)
        if short_ma is None:
            return False

        # 데드크로스: 이전에는 단기 > 장기였는데, 지금은 단기 < 장기
        return prev_short_ma >= prev_long_ma and short_ma < long_ma

    def _get_ma_values(self, symbol: str):
        """일봉 데이터에서 이동평균선 값을 계산한다."""
        today = datetime.today().strftime("%Y%m%d")
        # 장기MA 기간 + 여유분의 데이터 조회
        from datetime import timedelta
        start = (datetime.today() - timedelta(days=self.cfg.long_ma_period + 10)).strftime("%Y%m%d")

        df = self.market_data.get_daily_prices(symbol, start, today)
        if df.empty or len(df) < self.cfg.long_ma_period + 1:
            logger.debug("MA 계산 불가: %s 데이터 부족 (%d rows)", symbol, len(df))
            return None, None, None, None

        # 종가 컬럼 (stck_clpr)
        try:
            closes = df["stck_clpr"].astype(int).tolist()
        except (KeyError, ValueError):
            logger.warning("MA 계산 불가: %s 종가 데이터 오류", symbol)
            return None, None, None, None

        # 최신 데이터가 앞에 오므로 뒤집기
        closes = list(reversed(closes))

        n = len(closes)
        sp = self.cfg.short_ma_period
        lp = self.cfg.long_ma_period

        # 현재 MA
        short_ma = sum(closes[n - sp:]) / sp
        long_ma = sum(closes[n - lp:]) / lp

        # 전일 MA (하루 전)
        prev_short_ma = sum(closes[n - sp - 1:n - 1]) / sp
        prev_long_ma = sum(closes[n - lp - 1:n - 1]) / lp

        logger.debug(
            "MA [%s] 단기=%.0f(전일=%.0f) 장기=%.0f(전일=%.0f)",
            symbol, short_ma, prev_short_ma, long_ma, prev_long_ma,
        )

        return short_ma, long_ma, prev_short_ma, prev_long_ma
