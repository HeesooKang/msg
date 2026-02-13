import logging
from typing import List, Optional

from src.api_client import KISClient
from src.models import Order, OrderResult, OrderSide, OrderType

logger = logging.getLogger("kis_trader.orders")


class TradingAPI:
    """국내주식 주문 API."""

    def __init__(self, client: KISClient):
        self.client = client

    def place_order(self, order: Order) -> OrderResult:
        """매수/매도 주문을 실행한다."""
        # TR ID: 매수 TTTC0012U, 매도 TTTC0011U (모의투자 시 자동 V 변환)
        if order.side == OrderSide.BUY:
            tr_id = "TTTC0012U"
        else:
            tr_id = "TTTC0011U"

        body = {
            "CANO": "",  # api_client가 자동 주입
            "ACNT_PRDT_CD": "",  # api_client가 자동 주입
            "PDNO": order.symbol,
            "ORD_DVSN": order.order_type.value,
            "ORD_QTY": str(order.quantity),
            "ORD_UNPR": str(order.price),
            "EXCG_ID_DVSN_CD": "KRX",
            "SLL_TYPE": "",
            "CNDT_PRIC": "",
        }

        res = self.client.post(
            api_url="/uapi/domestic-stock/v1/trading/order-cash",
            tr_id=tr_id,
            body=body,
        )

        if res.success:
            output = res.output or {}
            result = OrderResult(
                success=True,
                order_no=output.get("ODNO", ""),
                message=res.error_message,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=order.price,
            )
            logger.info(
                "주문 성공: %s %s %s %d주 @ %s",
                order.side.value,
                order.symbol,
                order.order_type.name,
                order.quantity,
                order.price or "시장가",
            )
        else:
            result = OrderResult(
                success=False,
                message=f"[{res.error_code}] {res.error_message}",
                symbol=order.symbol,
                side=order.side,
            )
            logger.error("주문 실패 [%s]: %s", order.symbol, result.message)

        return result

    def buy(
        self,
        symbol: str,
        quantity: int,
        price: int = 0,
        order_type: OrderType = OrderType.MARKET,
    ) -> OrderResult:
        """매수 주문을 넣는다."""
        order = Order(
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=order_type,
            quantity=quantity,
            price=price,
        )
        return self.place_order(order)

    def sell(
        self,
        symbol: str,
        quantity: int,
        price: int = 0,
        order_type: OrderType = OrderType.MARKET,
    ) -> OrderResult:
        """매도 주문을 넣는다."""
        order = Order(
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=order_type,
            quantity=quantity,
            price=price,
        )
        return self.place_order(order)

    def cancel(
        self,
        order_no: str,
        quantity: int = 0,
        cancel_all: bool = True,
    ) -> OrderResult:
        """주문을 취소한다."""
        body = {
            "CANO": "",
            "ACNT_PRDT_CD": "",
            "KRX_FWDG_ORD_ORGNO": "",
            "ORGN_ODNO": order_no,
            "ORD_DVSN": "00",
            "RVSE_CNCL_DVSN_CD": "02",  # 취소
            "ORD_QTY": str(quantity),
            "ORD_UNPR": "0",
            "QTY_ALL_ORD_YN": "Y" if cancel_all else "N",
            "EXCG_ID_DVSN_CD": "KRX",
        }

        res = self.client.post(
            api_url="/uapi/domestic-stock/v1/trading/order-rvsecncl",
            tr_id="TTTC0013U",
            body=body,
        )

        if res.success:
            logger.info("주문 취소 성공: %s", order_no)
            return OrderResult(success=True, order_no=order_no, message="취소 완료")
        else:
            logger.error("주문 취소 실패 [%s]: %s", order_no, res.error_message)
            return OrderResult(success=False, order_no=order_no, message=res.error_message)

    def modify(
        self,
        order_no: str,
        quantity: int,
        price: int,
        order_type: OrderType = OrderType.LIMIT,
    ) -> OrderResult:
        """주문을 정정한다."""
        body = {
            "CANO": "",
            "ACNT_PRDT_CD": "",
            "KRX_FWDG_ORD_ORGNO": "",
            "ORGN_ODNO": order_no,
            "ORD_DVSN": order_type.value,
            "RVSE_CNCL_DVSN_CD": "01",  # 정정
            "ORD_QTY": str(quantity),
            "ORD_UNPR": str(price),
            "QTY_ALL_ORD_YN": "N",
            "EXCG_ID_DVSN_CD": "KRX",
        }

        res = self.client.post(
            api_url="/uapi/domestic-stock/v1/trading/order-rvsecncl",
            tr_id="TTTC0013U",
            body=body,
        )

        if res.success:
            logger.info("주문 정정 성공: %s → %d주 @ %d", order_no, quantity, price)
            return OrderResult(success=True, order_no=order_no, message="정정 완료")
        else:
            logger.error("주문 정정 실패 [%s]: %s", order_no, res.error_message)
            return OrderResult(success=False, order_no=order_no, message=res.error_message)
