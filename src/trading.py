import logging
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

from src.api_client import KISClient, is_kis_rate_limited_message
from src.models import Order, OrderResult, OrderSide, OrderType, Quote

logger = logging.getLogger("kis_trader.orders")


class TradingAPI:
    """국내주식 주문 API."""

    def __init__(self, client: KISClient, *, quote_provider: Callable[[str], Quote | None] | None = None):
        self.client = client
        self._quote_provider = quote_provider
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
        if code in {"LOCAL_ORDER_EXPIRED", "LOCAL_ORDER_INVALIDATED"}:
            return "stale_order"
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
        submit_before: float | None = None,
        pre_submit: Callable[[], str | None] | None = None,
    ):
        last_response = None
        for attempt in range(3):
            res = self.client.post(
                api_url=api_url,
                tr_id=tr_id,
                body=body,
                **({"submit_before": submit_before} if submit_before is not None else {}),
                **({"pre_submit": pre_submit} if pre_submit is not None else {}),
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

    @staticmethod
    def entry_quote_rejection(order: Order, quote: Quote | None) -> str | None:
        if (order.breakout_price <= 0 or quote is None or quote.symbol != order.symbol
                or not quote.has_valid_prices or not quote.book_available
                or not 0 < quote.bid_price <= quote.ask_price):
            return "최신 유효 호가/진입 근거가 없어 매수를 제출하지 않았습니다."
        if not quote.is_price_breakout(order.breakout_price, signal_price=order.reference_price):
            return (f"돌파 해소로 매수 미제출: {order.symbol} "
                    f"체결시세={quote.current_price} bid={quote.bid_price} "
                    f"관측고점={max(quote.trade_high_price, order.reference_price)} 돌파선={order.breakout_price}")
        if quote.ask_price > order.price:
            return (f"호가가 계획 지정가를 넘어 매수 미제출: {order.symbol} "
                    f"ask={quote.ask_price} 지정가={order.price}")
        if quote.book_depth_available and min(quote.ask_size, quote.bid_size) <= 0:
            return (f"최우선 호가 잔량이 없어 매수 미제출: {order.symbol} "
                    f"ask잔량={quote.ask_size} bid잔량={quote.bid_size}")
        return None

    def _entry_submission_rejection(self, order: Order) -> str | None:
        quote = self._quote_provider(order.symbol) if self._quote_provider is not None else None
        rejected = self.entry_quote_rejection(order, quote)
        if rejected:
            return rejected
        logger.info("매수 전송 직전: %s %d주 @ %d원 체결시세=%d bid/ask=%d/%d 시세=%s",
                    order.symbol, order.quantity, order.price, quote.current_price,
                    quote.bid_price, quote.ask_price, quote.timestamp.strftime("%H:%M:%S.%f"))
        return None

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
            and order.requested_reason == "price_breakout"
        ):
            return self._place_price_breakout_entry(order)
        return self._place_standard_order(order)

    def _place_standard_order(
        self,
        order: Order,
        *,
        log_failure: bool = True,
        log_success: bool = True,
        resolve_fill: bool = True,
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
            submit_before=order.submit_before if order.side == OrderSide.BUY else None,
            pre_submit=(lambda: self._entry_submission_rejection(order))
            if order.side == OrderSide.BUY and order.requested_reason == "price_breakout" else None,
        )

        if res.success:
            output = res.output or {}
            order_no = output.get("ODNO", "")
            reference_price = self._order_reference_price(order)
            fill_qty, fill_price = self._resolve_fill(order, order_no) if resolve_fill else (0, 0)
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

    def _place_price_breakout_entry(self, order: Order) -> OrderResult:
        """신호 지정가로 제출한 뒤 체결 조회보다 잔량 취소 요청을 먼저 한다."""
        limit_result = self._place_standard_order(
            order,
            log_failure=False,
            log_success=False,
            resolve_fill=False,
        )
        if not limit_result.success:
            logger.warning(
                "돌파 지정가 매수 제출 실패: %s %d주 @ %d원 (%s)",
                order.symbol,
                int(order.quantity or 0),
                int(order.price or 0),
                limit_result.message,
            )
            return limit_result

        requested_qty = max(0, int(order.quantity or 0))
        limit_result.price = 0
        if not limit_result.order_no:
            limit_result.fill_mode = "limit_pending"
            return limit_result

        # Do not keep a stale buy resting while a fill-history GET waits or retries.
        cancel_accepted = False
        try:
            cancel_result = self.cancel(
                limit_result.order_no,
                quantity=requested_qty,
                cancel_all=True,
                log_result=False,
            )
            cancel_accepted = cancel_result.success or self._cancel_has_no_remaining_quantity(cancel_result.message)
            if not cancel_accepted:
                logger.error("돌파 지정가 매수 잔량 취소 실패: %s order_no=%s (%s)",
                             order.symbol, limit_result.order_no, cancel_result.message)
        except Exception as exc:
            logger.warning("매수 잔량 취소 미확정: %s order_no=%s (%s)",
                           order.symbol, limit_result.order_no, exc)

        # An acknowledgement is not a fill. Keep the accepted order's identity
        # even when cancellation or the final lookup cannot be confirmed.
        try:
            row = self._fetch_fill_row(
                order_no=limit_result.order_no,
                symbol=order.symbol,
                side_code="02",
                start_date=datetime.now().strftime("%Y%m%d"),
            )
        except Exception as exc:
            logger.warning("매수 체결 조회 미확정: %s order_no=%s (%s)",
                           order.symbol, limit_result.order_no, exc)
            row = None
        final_qty, final_price = self._fill_quantity_and_price(row or {})
        terminal = self._entry_order_terminal(row, requested_qty)
        if not terminal or (final_qty > 0 and final_price <= 0):
            filled_qty = min(requested_qty, final_qty) if final_price > 0 else 0
            limit_result.quantity = filled_qty
            limit_result.price = final_price if filled_qty > 0 else 0
            limit_result.fill_mode = (
                "partial_fill_pending" if filled_qty > 0
                else "limit_pending" if cancel_accepted else "limit_pending_cancel_failed"
            )
            logger.info("매수 체결 확인 대기: %s order_no=%s 체결=%s 잔여=%s 취소=%s",
                        order.symbol, limit_result.order_no,
                        (row or {}).get("tot_ccld_qty", "-"),
                        (row or {}).get("rmn_qty", "-"),
                        (row or {}).get("cncl_yn", "-"))
            return limit_result

        filled_qty = min(requested_qty, final_qty)
        if filled_qty <= 0:
            return OrderResult(
                success=False,
                order_no=limit_result.order_no,
                message="돌파 지정가 미체결 취소",
                error_category="not_filled",
                symbol=order.symbol,
                side=order.side,
                requested_price=int(order.price or 0),
                reference_price=self._order_reference_price(order),
                fill_mode="limit_cancelled",
                requested_reason=order.requested_reason,
                requested_quantity=requested_qty,
            )

        limit_result.quantity = filled_qty
        limit_result.price = final_price
        limit_result.fill_mode = "limit" if filled_qty == requested_qty else "limit_partial"
        limit_result.message = "돌파 지정가 취소 후 최종 체결 확인"
        return limit_result

    @classmethod
    def _entry_order_terminal(cls, row: Optional[Dict[str, Any]], requested_qty: int) -> bool:
        if row is None or requested_qty <= 0:
            return False
        try:
            filled = int(str(row["tot_ccld_qty"]).strip())
            if filled < 0 or filled > requested_qty:
                return False
            if filled == requested_qty:
                return True
            remaining = int(str(row["rmn_qty"]).strip())
        except (KeyError, TypeError, ValueError):
            return False
        # The original order's cancellation flag also confirms a closed
        # remainder; a cancel POST acknowledgement alone does not.
        return remaining == 0 and (
            str(row.get("cncl_yn", "")).strip().upper() == "Y"
            or filled + cls._to_int(row.get("cnc_cfrm_qty"))
            + cls._to_int(row.get("rjct_qty")) >= requested_qty
        )

    def resolve_pending_entry(self, result: OrderResult) -> Optional[OrderResult]:
        """Return cumulative confirmed fills and the original order's terminal state."""
        row = self._fetch_fill_row(
            order_no=result.order_no, symbol=result.symbol, side_code="02",
            start_date=result.timestamp.strftime("%Y%m%d"),
        )
        if row is None:
            return None
        terminal = self._entry_order_terminal(row, result.requested_quantity)
        if not terminal and self._to_int(row.get("rmn_qty")) > 0:
            # A previous cancel acknowledgement may not have removed the order.
            cancelled = self.cancel(result.order_no, quantity=result.requested_quantity,
                                    cancel_all=True, log_result=False)
            if not cancelled.success and not self._cancel_has_no_remaining_quantity(cancelled.message):
                logger.warning("매수 잔량 취소 재확인 실패: %s order_no=%s (%s)",
                               result.symbol, result.order_no, cancelled.message)
        quantity, price = self._fill_quantity_and_price(row)
        if quantity > result.requested_quantity or (quantity > 0 and price <= 0):
            return None
        if quantity <= 0 and not terminal:
            return None
        return OrderResult(
            success=quantity > 0, symbol=result.symbol, side=OrderSide.BUY,
            order_no=result.order_no, quantity=quantity, price=price,
            requested_quantity=result.requested_quantity, requested_price=result.requested_price,
            requested_reason=result.requested_reason, timestamp=result.timestamp,
            fill_mode=("entry_reconciled" if terminal and quantity > 0
                       else "entry_reconciled_partial" if quantity > 0 else "limit_cancelled"),
            error_category="" if quantity > 0 else "not_filled",
        )

    @staticmethod
    def _cancel_has_no_remaining_quantity(message: str) -> bool:
        normalized = str(message or "")
        return "정정/취소할 수량이 없습니다" in normalized

    @classmethod
    def _fill_quantity_and_price(cls, row: Dict[str, Any]) -> Tuple[int, int]:
        quantity = max(0, cls._to_int(row.get("tot_ccld_qty")))
        price = max(0, cls._to_int(row.get("avg_prvs")))
        if quantity > 0 and price <= 0:
            amount = max(0, cls._to_int(row.get("tot_ccld_amt")))
            price = int(round(amount / quantity)) if amount > 0 else 0
        return quantity, price

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
                qty, avg_price = self._fill_quantity_and_price(row)
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
            # A cancelled buy can remain cncl_yn=N on the original row.
            # Its separate cancellation row is linked by orgn_odno.
            "ODNO": "" if side_code == "02" else order_no,
            "INQR_DVSN_1": "",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
            "EXCG_ID_DVSN_CD": "KRX",
        }

        page_params = dict(params)
        tr_cont = ""
        fill_row = None
        cancellations = {}
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
                return fill_row

            rows = res.output1 or []
            for row in rows:
                if row.get("pdno", "") != symbol:
                    continue
                if row.get("odno", "") == order_no:
                    if side_code != "02":
                        return row
                    if fill_row is None:
                        fill_row = row
                elif (
                    side_code == "02"
                    and row.get("orgn_odno", "") == order_no
                    and row.get("ord_dt", start_date) == start_date
                    and str(row.get("cncl_yn", "")).strip().upper() == "Y"
                    and row.get("odno")
                ):
                    try:
                        quantity = int(str(row["ord_qty"]).strip())
                        filled = int(str(row["tot_ccld_qty"]).strip())
                        remaining = int(str(row["rmn_qty"]).strip())
                        rejected = int(str(row["rjct_qty"]).strip())
                    except (KeyError, TypeError, ValueError):
                        continue
                    if quantity > 0 and filled == remaining == rejected == 0:
                        cancellations[row["odno"]] = quantity

            if fill_row is not None and cancellations:
                # Count each linked cancellation once; keep the original
                # fill and remainder as the source of its terminal status.
                fill_row = dict(fill_row, cnc_cfrm_qty=str(max(
                    self._to_int(fill_row.get("cnc_cfrm_qty")), sum(cancellations.values()),
                )))
            if fill_row is not None and self._entry_order_terminal(
                fill_row, self._to_int(fill_row.get("ord_qty")),
            ):
                return fill_row

            if not res.has_next:
                break

            payload = res.data or {}
            next_fk = str(payload.get("ctx_area_fk100", "") or "")
            next_nk = str(payload.get("ctx_area_nk100", "") or "")
            if not next_fk and not next_nk:
                break

            page_params["CTX_AREA_FK100"] = next_fk
            page_params["CTX_AREA_NK100"] = next_nk
            tr_cont = "N"
            time.sleep(0.1)
        return fill_row

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
