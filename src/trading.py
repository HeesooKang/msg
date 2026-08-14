import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.api_client import KISClient, is_kis_rate_limited_message
from src.models import Order, OrderResult, OrderSide, OrderType

logger = logging.getLogger("kis_trader.orders")


class TradingAPI:
    """국내주식 주문 API."""

    def __init__(self, client: KISClient):
        self.client = client
        self._fill_lookup_rate_limited = False
        self._account_order_blocked_message = ""

    @staticmethod
    def _is_rate_limited_message(message: str) -> bool:
        return is_kis_rate_limited_message(message)

    def _rate_limit_backoff(self, attempt: int) -> float:
        try:
            return max(0.8, float(self.client.rate_limit_cooldown_remaining()))
        except Exception:
            pass
        return 0.8 + (0.4 * max(0, attempt))

    @staticmethod
    def _is_tick_error_message(message: str) -> bool:
        normalized = str(message or "")
        return "40030000" in normalized or "호가단위 오류" in normalized

    @classmethod
    def _classify_order_failure(cls, *, error_code: str, message: str) -> str:
        normalized = str(message or "")
        code = str(error_code or "")
        if cls._is_rate_limited_message(normalized):
            return "rate_limit"
        if code == "40100000" or "영업일이 아닙니다" in normalized or "비영업일" in normalized:
            return "market_closed"
        if code == "40910000" or "모의투자 주문이 불가한 계좌" in normalized:
            return "account_order_unavailable"
        if code == "40070000" or "매매불가 종목" in normalized:
            return "symbol_order_unavailable"
        if code == "40240000" or "잔고내역이 없습니다" in normalized:
            return "no_holding"
        if code == "40030000" or cls._is_tick_error_message(normalized):
            return "invalid_price"
        if "정정/취소할 수량이 없습니다" in normalized:
            return "non_cancellable"
        return "other"

    @staticmethod
    def _order_reference_price(order: Order) -> int:
        for value in (
            order.reference_price,
            order.price,
        ):
            price = TradingAPI._to_int(value)
            if price > 0:
                return price
        return 0

    @staticmethod
    def _order_requested_price_label(order: Order) -> str:
        requested_price = TradingAPI._to_int(order.price)
        reference_price = TradingAPI._order_reference_price(order)
        if order.order_type == OrderType.MARKET and requested_price <= 0:
            if reference_price > 0:
                return f"기준가 {reference_price:,}원 (시장가)"
            return "시장가"
        if requested_price > 0:
            return f"{requested_price:,}원"
        return "시장가"

    @staticmethod
    def _order_success_price_suffix(result: OrderResult) -> str:
        fill_mode = str(result.fill_mode or "")
        fill_price = TradingAPI._to_int(result.price)
        reference_price = TradingAPI._to_int(result.reference_price)
        filled_quantity = TradingAPI._to_int(result.quantity)
        requested_quantity = TradingAPI._to_int(result.requested_quantity)
        parts: List[str] = []
        if fill_price > 0 and fill_mode not in {"pending", "market_pending"}:
            parts.append(f"체결가 {fill_price:,}원")
        else:
            parts.append("체결가 미확정")
        if fill_mode == "partial_fill_pending" and requested_quantity > filled_quantity > 0:
            parts.append(f"부분체결 {filled_quantity:,}/{requested_quantity:,}주")
        if reference_price > 0:
            parts.append(f"기준가 {reference_price:,}원")
        if fill_mode:
            parts.append(f"fill_mode={fill_mode}")
        return f" ({', '.join(parts)})" if parts else ""

    def _blocked_account_order_result(self, order: Order) -> OrderResult:
        return OrderResult(
            success=False,
            message=self._account_order_blocked_message or "[40910000] 모의투자 주문이 불가한 계좌입니다.",
            error_code="40910000",
            error_category="account_order_unavailable",
            symbol=order.symbol,
            side=order.side,
            reference_price=self._order_reference_price(order),
            requested_reason=order.requested_reason,
            requested_quantity=max(0, int(order.quantity or 0)),
        )

    def _post_with_rate_limit_retry(
        self,
        *,
        api_url: str,
        tr_id: str,
        body: Dict[str, Any],
        retry_on_rate_limit: bool = True,
    ):
        last_response = None
        for attempt in range(3):
            res = self.client.post(
                api_url=api_url,
                tr_id=tr_id,
                body=body,
            )
            last_response = res
            if res.success:
                return res
            if not self._is_rate_limited_message(res.error_message):
                return res
            if not retry_on_rate_limit:
                logger.warning(
                    "매수 주문 유량 제한: 오래된 신호를 재시도하지 않고 다음 틱에서 다시 평가합니다."
                )
                return res
            if attempt == 2:
                return res
            backoff = self._rate_limit_backoff(attempt)
            logger.warning(
                "주문 API 재시도: 호출 제한 감지 (%d/3, %.1fs 후 재시도)",
                attempt + 1,
                backoff,
            )
            time.sleep(backoff)
        return last_response

    def place_order(self, order: Order) -> OrderResult:
        """매수/매도 주문을 실행한다."""
        if self._account_order_blocked_message:
            logger.warning(
                "계좌 주문불가 상태가 확인되어 추가 주문을 API로 보내지 않습니다: %s %s",
                order.side.value,
                order.symbol,
            )
            return self._blocked_account_order_result(order)
        if (
            order.side == OrderSide.BUY
            and order.order_type == OrderType.LIMIT
            and order.requested_reason == "expected_value"
        ):
            return self._place_expected_value_entry(order)
        return self._place_standard_order(order)

    def _place_standard_order(
        self,
        order: Order,
        *,
        log_failure: bool = True,
        log_success: bool = True,
    ) -> OrderResult:
        """단일 주문을 실행한다."""
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

        res = self._post_with_rate_limit_retry(
            api_url="/uapi/domestic-stock/v1/trading/order-cash",
            tr_id=tr_id,
            body=body,
            retry_on_rate_limit=order.side != OrderSide.BUY,
        )

        if res.success:
            output = res.output or {}
            order_no = output.get("ODNO", "")
            reference_price = self._order_reference_price(order)
            fill_qty, fill_price = self._resolve_fill(order, order_no)
            if fill_qty > 0:
                resolved_qty = fill_qty
                resolved_price = fill_price if fill_price > 0 else order.price
                if int(order.quantity or 0) > 0 and resolved_qty < int(order.quantity or 0):
                    fill_mode = "partial_fill_pending"
                else:
                    fill_mode = "filled" if resolved_price > 0 else "filled_price_pending"
            elif order.order_type == OrderType.MARKET:
                resolved_qty = 0
                resolved_price = 0
                fill_mode = "market_pending"
                logger.debug(
                    "시장가 주문 체결이 아직 조회되지 않았습니다: %s %s %d주 (order_no=%s)",
                    order.side.value,
                    order.symbol,
                    order.quantity,
                    order_no,
                )
            else:
                resolved_qty = 0
                resolved_price = order.price
                fill_mode = "pending"
            result = OrderResult(
                success=True,
                order_no=order_no,
                message=res.error_message,
                error_code=str(res.error_code or ""),
                error_category="",
                symbol=order.symbol,
                side=order.side,
                quantity=resolved_qty,
                price=resolved_price,
                requested_price=order.price,
                reference_price=reference_price,
                fill_mode=fill_mode,
                requested_reason=order.requested_reason,
                requested_quantity=max(0, int(order.quantity or 0)),
            )
            requested_label = self._order_requested_price_label(order)
            if log_success:
                logger.info(
                    "주문 성공: %s %s %s %d주 @ %s%s",
                    order.side.value,
                    order.symbol,
                    order.order_type.name,
                    order.quantity,
                    requested_label,
                    self._order_success_price_suffix(result),
                )
        else:
            error_code = str(res.error_code or "")
            error_message = str(res.error_message or "")
            result = OrderResult(
                success=False,
                message=f"[{error_code}] {error_message}",
                error_code=error_code,
                error_category=self._classify_order_failure(
                    error_code=error_code,
                    message=error_message,
                ),
                symbol=order.symbol,
                side=order.side,
                reference_price=self._order_reference_price(order),
                requested_reason=order.requested_reason,
                requested_quantity=max(0, int(order.quantity or 0)),
            )
            if log_failure:
                if result.error_category == "no_holding" and order.side == OrderSide.SELL:
                    logger.warning("주문 실패 [%s]: %s", order.symbol, result.message)
                else:
                    logger.error("주문 실패 [%s]: %s", order.symbol, result.message)
            if result.error_category == "account_order_unavailable":
                self._account_order_blocked_message = result.message
                cfg = self.client.config
                account_number = str(cfg.account_number or "")
                if len(account_number) >= 2:
                    account_mask = f"{account_number[:2]}******"
                else:
                    account_mask = "***"
                logger.error(
                    "계좌/API 키 조합이 주문 불가 상태입니다. 이후 주문 제출을 중단합니다. mode=%s account=%s product=%s",
                    cfg.trading_mode,
                    account_mask,
                    cfg.account_product_code,
                )

        return result

    def _place_expected_value_entry(self, order: Order) -> OrderResult:
        """EV 가격 상한까지만 체결하고 남은 매수 수량은 즉시 취소한다."""
        limit_result = self._place_standard_order(
            order,
            log_failure=False,
            log_success=False,
        )
        if not limit_result.success:
            logger.warning(
                "EV 제한가 매수 제출 실패: %s %d주 @ %d원 (%s)",
                order.symbol,
                int(order.quantity or 0),
                int(order.price or 0),
                limit_result.message,
            )
            return limit_result

        requested_qty = max(0, int(order.quantity or 0))
        filled_qty = min(requested_qty, max(0, int(limit_result.quantity or 0)))
        if filled_qty >= requested_qty:
            limit_result.fill_mode = "limit"
            logger.info(
                "주문 성공: buy %s LIMIT %d주 @ %d원%s",
                order.symbol,
                requested_qty,
                int(order.price or 0),
                self._order_success_price_suffix(limit_result),
            )
            return limit_result
        if not limit_result.order_no:
            limit_result.fill_mode = "limit_pending"
            return limit_result

        cancel_result = self.cancel(
            limit_result.order_no,
            quantity=max(0, requested_qty - filled_qty),
            cancel_all=True,
            log_result=False,
        )
        if not cancel_result.success and self._cancel_implies_filled(cancel_result.message):
            late_qty, late_price = self._resolve_fill(order, limit_result.order_no)
            resolved_qty = max(filled_qty, late_qty, requested_qty)
            resolved_price = max(1, int(late_price or limit_result.price or order.price or 0))
            logger.info(
                "매수 체결: buy %s LIMIT %d주 @ 체결가 %d원 "
                "(취소 시점 전량체결 확인)",
                order.symbol,
                resolved_qty,
                resolved_price,
            )
            return OrderResult(
                success=True,
                order_no=limit_result.order_no,
                message=cancel_result.message,
                symbol=order.symbol,
                side=order.side,
                quantity=resolved_qty,
                price=resolved_price,
                requested_price=int(order.price or 0),
                reference_price=self._order_reference_price(order),
                fill_mode="limit_assumed_filled",
                requested_reason=order.requested_reason,
                requested_quantity=requested_qty,
            )
        if not cancel_result.success:
            logger.error(
                "EV 제한가 매수 잔량 취소 실패: %s order_no=%s (%s)",
                order.symbol,
                limit_result.order_no,
                cancel_result.message,
            )
            limit_result.fill_mode = (
                "partial_fill_pending" if filled_qty > 0 else "limit_pending_cancel_failed"
            )
            return limit_result

        if filled_qty <= 0:
            logger.info(
                "매수 미체결 취소: buy %s LIMIT %d주 @ 주문가 %d원",
                order.symbol,
                requested_qty,
                int(order.price or 0),
            )
            return OrderResult(
                success=False,
                order_no=limit_result.order_no,
                message="EV 제한가 미체결 취소",
                error_category="not_filled",
                symbol=order.symbol,
                side=order.side,
                requested_price=int(order.price or 0),
                reference_price=self._order_reference_price(order),
                fill_mode="limit_cancelled",
                requested_reason=order.requested_reason,
                requested_quantity=requested_qty,
            )

        limit_result.fill_mode = "limit_partial"
        limit_result.message = "EV 제한가 부분체결 후 잔량 취소"
        logger.info(
            "매수 부분체결: buy %s LIMIT %d/%d주 @ 체결가 %d원 (잔량 취소)",
            order.symbol,
            filled_qty,
            requested_qty,
            int(limit_result.price or order.price or 0),
        )
        return limit_result

    @staticmethod
    def _cancel_implies_filled(message: str) -> bool:
        normalized = str(message or "")
        return "정정/취소할 수량이 없습니다" in normalized

    def _resolve_fill(self, order: Order, order_no: str) -> Tuple[int, int]:
        """주문 직후 체결내역에서 실제 체결수량/평균체결가를 조회한다.

        시장가 체결의 경우 실제 체결가를 전략 손익 계산에 반영하기 위함.
        조회 실패/미체결이면 (0, 0)을 반환하고 상위에서 기존 값을 사용한다.
        """
        if not order_no:
            return 0, 0
        if (
            self.client.config.is_paper
            and order.side == OrderSide.SELL
            and order.order_type == OrderType.MARKET
        ):
            logger.debug(
                "모의투자 시장가 매도 체결조회는 계좌 재동기화로 넘깁니다: %s (order_no=%s)",
                order.symbol,
                order_no,
            )
            return 0, 0

        start_date = datetime.now().strftime("%Y%m%d")
        side_code = "02" if order.side == OrderSide.BUY else "01"

        attempts = 3 if order.order_type == OrderType.MARKET else 1
        self._fill_lookup_rate_limited = False
        # 체결내역 조회는 주문 API 한도와 경쟁하므로, 미반영이면 잔고 재동기화로 넘긴다.
        requested_quantity = max(0, int(order.quantity or 0))
        for attempt in range(attempts):
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
                if qty > 0:
                    resolved_quantity = min(qty, requested_quantity) if requested_quantity > 0 else qty
                    return resolved_quantity, max(0, avg_price)
            if self._fill_lookup_rate_limited:
                logger.debug("체결내역 조회가 거래건수 제한에 걸려 잔고 재동기화로 넘깁니다: %s", order.symbol)
                break
            if attempt + 1 < attempts:
                time.sleep(0.35)

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

        page_params = dict(params)
        tr_cont = ""
        for _ in range(10):
            res = self.client.get(
                api_url="/uapi/domestic-stock/v1/trading/inquire-daily-ccld",
                tr_id="TTTC0081R",
                params=page_params,
                tr_cont=tr_cont,
            )
            if not res.success:
                message = f"{res.error_code} {res.error_message} {res.data}"
                if self._is_rate_limited_message(message):
                    self._fill_lookup_rate_limited = True
                return None

            rows = res.output1 or []
            for row in rows:
                if row.get("odno", "") == order_no and row.get("pdno", "") == symbol:
                    return row

            if not res.has_next:
                return None

            payload = res.data or {}
            next_fk = str(payload.get("ctx_area_fk100", "") or "")
            next_nk = str(payload.get("ctx_area_nk100", "") or "")
            if not next_fk and not next_nk:
                return None

            page_params["CTX_AREA_FK100"] = next_fk
            page_params["CTX_AREA_NK100"] = next_nk
            tr_cont = "N"
            time.sleep(0.1)
        return None

    @staticmethod
    def _to_int(value: Any) -> int:
        try:
            return int(float(str(value).replace(",", "").strip()))
        except (TypeError, ValueError):
            return 0

    def cancel(
        self,
        order_no: str,
        quantity: int = 0,
        cancel_all: bool = True,
        *,
        log_result: bool = True,
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

        res = self._post_with_rate_limit_retry(
            api_url="/uapi/domestic-stock/v1/trading/order-rvsecncl",
            tr_id="TTTC0013U",
            body=body,
        )

        if res.success:
            if log_result:
                logger.info("주문 취소 성공: %s", order_no)
            return OrderResult(success=True, order_no=order_no, message="취소 완료")
        else:
            if log_result:
                logger.error("주문 취소 실패 [%s]: %s", order_no, res.error_message)
            return OrderResult(
                success=False,
                order_no=order_no,
                message=res.error_message,
                error_code=str(res.error_code or ""),
                error_category=self._classify_order_failure(
                    error_code=str(res.error_code or ""),
                    message=str(res.error_message or ""),
                ),
            )
