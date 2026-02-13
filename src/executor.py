import logging
from typing import List, Tuple

from src.models import Order, OrderResult
from src.trading import TradingAPI

logger = logging.getLogger("kis_trader.executor")


class RiskManager:
    """주문 전 리스크를 체크한다."""

    def __init__(
        self,
        max_order_amount: int = 10_000_000,
        max_position_count: int = 10,
    ):
        self.max_order_amount = max_order_amount
        self.max_position_count = max_position_count
        self._daily_loss = 0

    def check(self, order: Order) -> Tuple[bool, str]:
        """주문을 검증한다. (통과 여부, 사유)"""
        if order.quantity <= 0:
            return False, "수량이 0 이하입니다"

        estimated_amount = order.price * order.quantity
        if order.price > 0 and estimated_amount > self.max_order_amount:
            return False, f"주문금액 {estimated_amount:,}원이 한도 {self.max_order_amount:,}원 초과"

        return True, ""


class OrderExecutor:
    """전략에서 생성한 주문을 실행한다."""

    def __init__(self, trading: TradingAPI, risk_manager: RiskManager = None):
        self.trading = trading
        self.risk = risk_manager or RiskManager()

    def submit_orders(self, orders: List[Order]) -> List[OrderResult]:
        """주문 리스트를 리스크 체크 후 실행한다."""
        results = []
        for order in orders:
            ok, reason = self.risk.check(order)
            if not ok:
                logger.warning("리스크 차단: %s %s - %s", order.side.value, order.symbol, reason)
                results.append(OrderResult(
                    success=False,
                    message=f"리스크 차단: {reason}",
                    symbol=order.symbol,
                    side=order.side,
                ))
                continue

            result = self.trading.place_order(order)
            results.append(result)

        return results
