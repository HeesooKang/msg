import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

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
            order_no = output.get("ODNO", "")
            fill_qty, fill_price = self._resolve_fill(order, order_no)
            result = OrderResult(
                success=True,
                order_no=order_no,
                message=res.error_message,
                symbol=order.symbol,
                side=order.side,
                quantity=fill_qty if fill_qty > 0 else order.quantity,
                price=fill_price if fill_price > 0 else order.price,
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

    def _resolve_fill(self, order: Order, order_no: str) -> Tuple[int, int]:
        """주문 직후 체결내역에서 실제 체결수량/평균체결가를 조회한다.

        시장가 체결의 경우 실제 체결가를 전략 손익 계산에 반영하기 위함.
        조회 실패/미체결이면 (0, 0)을 반환하고 상위에서 기존 값을 사용한다.
        """
        if not order_no:
            return 0, 0

        start_date = datetime.now().strftime("%Y%m%d")
        side_code = "02" if order.side == OrderSide.BUY else "01"

        # 시장가는 체결 확정에 약간의 지연이 있을 수 있어 짧게 재조회
        for _ in range(5):
            row = self._fetch_fill_row(
                order_no=order_no,
                symbol=order.symbol,
                side_code=side_code,
                start_date=start_date,
            )
            if row:
                qty = self._to_int(row.get("tot_ccld_qty", 0))
                avg_price = self._to_int(row.get("avg_prvs", 0))
                if qty > 0 and avg_price <= 0:
                    total_amt = self._to_int(row.get("tot_ccld_amt", 0))
                    if total_amt > 0:
                        avg_price = int(round(total_amt / qty))
                if qty > 0 and avg_price > 0:
                    return qty, avg_price
            time.sleep(0.2)

        return 0, 0

    def _fetch_fill_row(
        self,
        order_no: str,
        symbol: str,
        side_code: str,
        start_date: str,
    ) -> Optional[Dict[str, Any]]:
        params = {
            "CANO": "",
            "ACNT_PRDT_CD": "",
            "INQR_STRT_DT": start_date,
            "INQR_END_DT": start_date,
            "SLL_BUY_DVSN_CD": side_code,
            "PDNO": symbol,
            "CCLD_DVSN": "00",
            "INQR_DVSN": "00",
            "INQR_DVSN_3": "00",
            "ORD_GNO_BRNO": "",
            "ODNO": order_no,
            "INQR_DVSN_1": "",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
            "EXCG_ID_DVSN_CD": "KRX",
        }

        res = self.client.get(
            api_url="/uapi/domestic-stock/v1/trading/inquire-daily-ccld",
            tr_id="TTTC0081R",
            params=params,
        )
        if not res.success:
            return None

        rows = res.output1 or []
        for row in rows:
            if row.get("odno", "") == order_no and row.get("pdno", "") == symbol:
                return row
        return None

    @staticmethod
    def _to_int(value: Any) -> int:
        try:
            return int(float(str(value).replace(",", "").strip()))
        except (TypeError, ValueError):
            return 0

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
