import logging
import time
from typing import List, Tuple

import requests

from src.api_client import is_kis_rate_limited_message
from src.models import Order, OrderResult
from src.trading import TradingAPI

logger = logging.getLogger("kis_trader.executor")
order_logger = logging.getLogger("kis_trader.orders")


class RiskManager:
    """주문 전 리스크를 체크한다."""

    def __init__(
        self,
        max_order_amount: int = 10_000_000,
    ):
        self.max_order_amount = max_order_amount

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

    def __init__(self, trading: TradingAPI, risk_manager: RiskManager):
        self.trading = trading
        self.risk = risk_manager

    @staticmethod
    def _is_rate_limited_result(result: OrderResult) -> bool:
        return is_kis_rate_limited_message(result.message)

    def _inter_order_delay_seconds(self) -> float:
        return 0.45 if self.trading.client.config.is_paper else 0.15

    def submit_orders(self, orders: List[Order]) -> List[OrderResult]:
        """주문 리스트를 리스크 체크 후 실행한다."""
        results = []
        total_orders = len(orders)
        for index, order in enumerate(orders):
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

            try:
                result = self.trading.place_order(order)
            except requests.exceptions.RequestException as exc:
                reference_price = int(order.reference_price or order.price or 0)
                requested_price = int(order.price or 0)
                requested_label = (
                    f"주문가 {requested_price:,}원"
                    if requested_price > 0
                    else "시장가"
                )
                logger.warning(
                    "주문 결과 미확정: %s %s %d주 @ %s (신호가 %d원, %s)",
                    order.side.value,
                    order.symbol,
                    int(order.quantity or 0),
                    requested_label,
                    reference_price,
                    exc,
                )
                order_logger.warning(
                    "주문 결과 미확정: %s %s %s %d주 @ %s "
                    "(체결가 미확정, 신호가 %s원, fill_mode=order_result_pending, error=%s)",
                    order.side.value,
                    order.symbol,
                    order.order_type.name,
                    int(order.quantity or 0),
                    requested_label,
                    f"{reference_price:,}" if reference_price > 0 else "0",
                    exc,
                )
                result = OrderResult(
                    success=True,
                    message=f"주문 결과 미확정: {exc}",
                    symbol=order.symbol,
                    side=order.side,
                    quantity=0,
                    price=0,
                    requested_price=int(order.price or 0),
                    reference_price=int(order.reference_price or 0),
                    requested_quantity=max(0, int(order.quantity or 0)),
                    fill_mode="order_result_pending",
                    requested_reason=str(order.requested_reason or ""),
                )
            results.append(result)
            remaining_orders = total_orders - index - 1
            if self._is_rate_limited_result(result):
                if remaining_orders > 0:
                    logger.warning(
                        "주문 배치 냉각: 유량 제한으로 %.2fs 대기 후 남은 %d건을 계속 제출합니다.",
                        self._inter_order_delay_seconds() + 1.0,
                        remaining_orders,
                    )
                    time.sleep(self._inter_order_delay_seconds() + 1.0)
                continue
            if remaining_orders > 0:
                time.sleep(self._inter_order_delay_seconds())

        return results
