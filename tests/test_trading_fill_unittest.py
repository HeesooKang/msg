import unittest
from dataclasses import replace
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from types import SimpleNamespace

import requests

from src.models import Order, OrderResult, OrderSide, OrderType, Quote
from src.trading import TradingAPI


class DummyResponse:
    def __init__(
        self,
        success,
        output=None,
        output1=None,
        error_code="",
        error_message="",
        *,
        has_next=False,
        data=None,
    ):
        self.success = success
        self.output = output or {}
        self.output1 = output1 or []
        self.error_code = error_code
        self.error_message = error_message
        self.has_next = has_next
        self.data = data or {}


class FakeClient:
    def __init__(self, post_response, get_responses, *, is_paper=False):
        if isinstance(post_response, list):
            self.post_responses = list(post_response)
        else:
            self.post_responses = [post_response]
        self.get_responses = list(get_responses)
        self.get_calls = 0
        self.post_calls = 0
        self.post_requests = []
        self._cooldown_remaining = 0.0
        self.config = SimpleNamespace(
            is_paper=is_paper,
            account_number="12345678",
            account_product_code="01",
            trading_mode="paper" if is_paper else "real",
        )

    def post(self, **kwargs):
        self.post_calls += 1
        self.post_requests.append(kwargs)
        if self.post_responses:
            return self.post_responses.pop(0)
        return DummyResponse(success=False, error_message="no post response")

    def get(self, **kwargs):
        self.get_calls += 1
        if self.get_responses:
            return self.get_responses.pop(0)
        return DummyResponse(success=True, output1=[])

    def rate_limit_cooldown_remaining(self):
        return self._cooldown_remaining


class TradingFillPriceTests(unittest.TestCase):
    def test_submission_reuses_original_breakout_and_limit_without_chasing(self):
        order = Order("033780", OrderSide.BUY, OrderType.LIMIT, 5, 179000,
                      reference_price=178900, requested_reason="price_breakout", breakout_price=178500)
        q = Quote(order.symbol, 178900, datetime.now(), bid_price=178900, ask_price=179000,
                  book_available=True, book_depth_available=True, bid_size=1, ask_size=1)
        provider = Mock(return_value=q)
        trading = TradingAPI(FakeClient([], []), quote_provider=provider)
        for book, allowed in (
            (q, True), (replace(q, bid_price=178600, ask_price=178700), True),
            (replace(q, bid_price=178500), False), (replace(q, bid_price=178400), False),
            (replace(q, current_price=178500), False), (replace(q, current_price=178400), False),
            (replace(q, trade_high_price=179000), False),
            (replace(q, trade_high_price=q.current_price), True),
            (replace(q, timestamp=q.timestamp+timedelta(seconds=1), current_price=178800,
                     bid_price=178700, ask_price=178800, trade_high_price=178800), False),
            (replace(q, timestamp=q.timestamp+timedelta(seconds=1), current_price=178950,
                     trade_high_price=178950), True),
            (replace(q, ask_price=179100), False), (replace(q, book_available=False), False),
            (replace(q, ask_price=0), False), (replace(q, bid_price=179100), False),
            (replace(q, ask_size=0), False), (replace(q, bid_size=0), False),
            (replace(q, symbol="other"), False), (None, False),
        ):
            with self.subTest(book=book):
                provider.return_value = book
                self.assertEqual(trading._entry_submission_rejection(order) is None, allowed)
        self.assertEqual(order.price, 179000)
        self.assertEqual(order.breakout_price, 178500)
        self.assertEqual(order.reference_price, 178900)
        provider.return_value = q
        self.assertIsNotNone(trading._entry_submission_rejection(replace(order, breakout_price=0)))
        self.assertIsNotNone(TradingAPI(trading.client)._entry_submission_rejection(order))

    def test_pending_buy_release_requires_explicit_terminal_zero_fill(self):
        for fields, expected in (
            ({"tot_ccld_qty": "0", "cnc_cfrm_qty": "2", "rmn_qty": "0"}, True),
            ({"tot_ccld_qty": "0", "rjct_qty": "2", "rmn_qty": "0"}, True),
            ({"tot_ccld_qty": "0", "cncl_yn": "Y", "rmn_qty": "0"}, True),
            ({"tot_ccld_qty": "0", "cncl_yn": "N", "rmn_qty": "0"}, False),
            ({"tot_ccld_qty": "0", "cncl_yn": "Y"}, False),
            ({"cncl_yn": "Y", "rmn_qty": "0"}, False),
            ({"tot_ccld_qty": "", "cncl_yn": "Y", "rmn_qty": "0"}, False),
            ({"tot_ccld_qty": "bad", "cncl_yn": "Y", "rmn_qty": "0"}, False),
            ({"tot_ccld_qty": "0", "cncl_yn": "Y", "rmn_qty": "bad"}, False),
            ({"tot_ccld_qty": "-1", "cncl_yn": "Y", "rmn_qty": "0"}, False),
            ({"tot_ccld_qty": "3", "cncl_yn": "Y", "rmn_qty": "0"}, False),
            ({"tot_ccld_qty": "1", "avg_prvs": "70100", "cncl_yn": "Y", "rmn_qty": "0"}, False),
            ({"tot_ccld_qty": "0", "cnc_cfrm_qty": "2"}, False),
            ({"tot_ccld_qty": "0", "rmn_qty": "0"}, False),
            ({"tot_ccld_qty": "1", "avg_prvs": "70100", "cnc_cfrm_qty": "1", "rmn_qty": "0"}, False),
            ({"tot_ccld_qty": "2", "avg_prvs": "70100", "rmn_qty": "0"}, False),
        ):
            with self.subTest(fields=fields):
                client = FakeClient([], [DummyResponse(True, output1=[{
                    "odno": "known", "pdno": "005930", **fields}])])
                result = OrderResult(True, order_no="known", symbol="005930", side=OrderSide.BUY,
                                     requested_quantity=2, requested_price=70100)
                resolved = TradingAPI(client).resolve_pending_entry(result)
                self.assertEqual(resolved is not None and not resolved.success, expected)
                self.assertEqual(client.get_calls, 1)
                self.assertEqual(client.post_calls, 0)

    def test_pending_buy_retries_cancel_but_acknowledgement_does_not_release_order(self):
        client = FakeClient(DummyResponse(True), [
            DummyResponse(True, output1=[{"odno": "known", "pdno": "005930",
                "tot_ccld_qty": "0", "rmn_qty": "2"}]),
            DummyResponse(True, output1=[{"odno": "known", "pdno": "005930",
                "tot_ccld_qty": "0", "cnc_cfrm_qty": "2", "rmn_qty": "0"}])])
        trading = TradingAPI(client)
        result = OrderResult(True, order_no="known", symbol="005930", side=OrderSide.BUY,
                             requested_quantity=2, requested_price=70100)
        self.assertIsNone(trading.resolve_pending_entry(result))
        self.assertEqual(client.post_requests[0]["body"]["ORGN_ODNO"], "known")
        self.assertEqual(client.post_requests[0]["body"]["RVSE_CNCL_DVSN_CD"], "02")
        self.assertFalse(trading.resolve_pending_entry(result).success)
        self.assertEqual(client.post_calls, 1)

    def test_pending_buy_missing_or_mismatched_history_stays_unknown(self):
        for response in (DummyResponse(False, error_message="unavailable"),
                         DummyResponse(True, output1=[]),
                         DummyResponse(True, output1=[{"odno": "other", "pdno": "005930",
                             "tot_ccld_qty": "0", "cnc_cfrm_qty": "2", "rmn_qty": "0"}])):
            with self.subTest(response=response):
                client = FakeClient([], [response])
                result = OrderResult(True, order_no="known", symbol="005930", side=OrderSide.BUY,
                                     requested_quantity=2)
                self.assertIsNone(TradingAPI(client).resolve_pending_entry(result))
                self.assertEqual(client.post_calls, 0)

    def test_pending_buy_joins_original_order_with_its_confirmed_cancellation(self):
        original = dict(odno="0000003997", orgn_odno="0000000000", pdno="152100",
                        ord_dt="20261002", ord_qty="8", tot_ccld_qty="0",
                        cncl_yn="N", rmn_qty="0", rjct_qty="0")
        cancellation = dict(odno="0000004012", orgn_odno="0000003997", pdno="152100",
                            ord_dt="20261002", ord_qty="8", tot_ccld_qty="0",
                            cncl_yn="Y", rmn_qty="0", rjct_qty="0")
        result = OrderResult(True, order_no="0000003997", symbol="152100", side=OrderSide.BUY,
                             requested_quantity=8, requested_price=112725)
        result.timestamp = result.timestamp.replace(year=2026, month=10, day=2)
        cases = [([original, cancellation], True), ([cancellation, original], True),
                 ([cancellation], False), ([original], False)]
        for changes in (
            dict(orgn_odno="other"), dict(pdno="other"), dict(ord_dt="20261001"),
            dict(cncl_yn="N"), dict(rjct_qty="1"), dict(rmn_qty="1"),
            dict(tot_ccld_qty="1"), dict(ord_qty="bad"),
        ):
            cases.append(([original, dict(cancellation, **changes)], False))
        # Repeating one four-share cancellation cannot close an eight-share order.
        half = dict(cancellation, ord_qty="4")
        cases.append(([original, half, half], False))
        for rows, expected in cases:
            with self.subTest(rows=rows):
                client = FakeClient([], [DummyResponse(True, output1=rows)])
                get = client.get

                def lookup(**kwargs):
                    self.assertEqual(kwargs["params"]["ODNO"], "")
                    self.assertEqual(kwargs["params"]["PDNO"], "152100")
                    return get(**kwargs)

                client.get = lookup
                resolved = TradingAPI(client).resolve_pending_entry(result)
                self.assertEqual(resolved is not None and not resolved.success, expected)
                self.assertEqual(client.get_calls, 1)
                self.assertEqual(client.post_calls, 0)
        self.assertNotIn("cnc_cfrm_qty", original)

    def test_linked_cancellation_on_next_page_preserves_partial_buy_fill(self):
        original = dict(odno="known", pdno="005930", ord_qty="8", tot_ccld_qty="2",
                        avg_prvs="70100", cncl_yn="N", rmn_qty="0", rjct_qty="0")
        cancellation = dict(odno="cancel", orgn_odno="known", pdno="005930", ord_qty="6",
                            tot_ccld_qty="0", cncl_yn="Y", rmn_qty="0", rjct_qty="0")
        client = FakeClient([DummyResponse(True, output={"ODNO": "known"}), DummyResponse(True)], [
            DummyResponse(True, output1=[original], has_next=True,
                          data=dict(ctx_area_fk100="next", ctx_area_nk100="page")),
            DummyResponse(True, output1=[cancellation]),
        ])
        with patch("src.trading.time.sleep", return_value=None):
            result = TradingAPI(client).place_order(Order(
                "005930", OrderSide.BUY, OrderType.LIMIT, 8, 70100,
                requested_reason="price_breakout"))
        self.assertEqual((result.quantity, result.price, result.fill_mode), (2, 70100, "limit_partial"))
        self.assertEqual(client.get_calls, 2)
        self.assertEqual(client.post_calls, 2)

    def test_breakout_cancels_before_any_fill_lookup_and_keeps_partial_fill(self):
        client = FakeClient(
            post_response=[DummyResponse(True, output={"ODNO": "immediate-cancel"}), DummyResponse(True)],
            get_responses=[DummyResponse(True, output1=[{
                "odno": "immediate-cancel", "pdno": "005930", "tot_ccld_qty": "1",
                "avg_prvs": "70100", "cnc_cfrm_qty": "1", "rmn_qty": "0",
            }])], is_paper=True)
        get = client.get

        def lookup_after_cancel(**kwargs):
            self.assertEqual(client.post_calls, 2)
            self.assertEqual(client.post_requests[-1]["body"]["QTY_ALL_ORD_YN"], "Y")
            return get(**kwargs)

        client.get = lookup_after_cancel
        result = TradingAPI(client).place_order(Order("005930", OrderSide.BUY, OrderType.LIMIT,
            2, 70100, requested_reason="price_breakout", submit_before=105.0))
        self.assertEqual((result.quantity, result.price, result.fill_mode), (1, 70100, "limit_partial"))
        self.assertEqual(client.get_calls, 1)
        self.assertEqual(client.post_requests[0]["submit_before"], 105.0)
        self.assertNotIn("submit_before", client.post_requests[1])

    def test_breakout_after_acceptance_network_failures_keep_known_order_identity(self):
        for failure in ("cancel", "lookup"):
            with self.subTest(failure=failure):
                client = FakeClient(
                    post_response=[DummyResponse(True, output={"ODNO": "known-order"}), DummyResponse(True)],
                    get_responses=[])
                post = client.post

                def maybe_timeout(**kwargs):
                    if kwargs["api_url"].endswith("order-rvsecncl"):
                        raise requests.exceptions.ReadTimeout("cancel timeout")
                    return post(**kwargs)

                if failure == "cancel":
                    client.post = maybe_timeout
                else:
                    def lookup_timeout(**kwargs):
                        raise requests.exceptions.ReadTimeout("lookup timeout")
                    client.get = lookup_timeout
                result = TradingAPI(client).place_order(Order("005930", OrderSide.BUY, OrderType.LIMIT,
                    2, 70100, requested_reason="price_breakout"))
                self.assertTrue(result.success)
                self.assertEqual(result.order_no, "known-order")
                self.assertEqual((result.quantity, result.price), (0, 0))
                self.assertIn(result.fill_mode, {"limit_pending", "limit_pending_cancel_failed"})
                self.assertEqual(result.requested_quantity, 2)

    def test_expired_buy_skips_fill_lookup_without_affecting_sell(self):
        client = FakeClient(
            post_response=DummyResponse(success=False, error_code="LOCAL_ORDER_EXPIRED",
                                        error_message="signal expired"), get_responses=[])
        trading = TradingAPI(client)
        buy = Order("005930", OrderSide.BUY, OrderType.LIMIT, 2, 10000,
                    requested_reason="price_breakout", submit_before=105.0)
        result = trading.place_order(buy)
        self.assertFalse(result.success)
        self.assertEqual(result.error_category, "stale_order")
        self.assertEqual(client.post_requests[0]["submit_before"], 105.0)
        self.assertEqual(client.get_calls, 0)
        self.assertEqual(client.post_calls, 1)

        client.post_responses = [DummyResponse(success=True, output={"ODNO": "sell"})]
        client.config.is_paper = True
        sell = Order("005930", OrderSide.SELL, quantity=2, submit_before=105.0)
        result = trading.place_order(sell)
        self.assertTrue(result.success)
        self.assertNotIn("submit_before", client.post_requests[-1])

    def test_place_order_uses_actual_fill_price_in_paper_when_fill_history_exists(self):
        client = FakeClient(
            post_response=DummyResponse(
                success=True,
                output={"ODNO": "12345"},
            ),
            get_responses=[
                DummyResponse(
                    success=True,
                    output1=[{
                        "odno": "12345",
                        "pdno": "005930",
                        "tot_ccld_qty": "2",
                        "avg_prvs": "71200",
                    }],
                )
            ],
            is_paper=True,
        )
        trading = TradingAPI(client)
        order = Order(
            symbol="005930",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=2,
            price=0,
        )

        result = trading.place_order(order)

        self.assertTrue(result.success)
        self.assertEqual(result.quantity, 2)
        self.assertEqual(result.price, 71200)

    def test_place_order_classifies_paper_non_business_day(self):
        client = FakeClient(
            post_response=DummyResponse(
                success=False,
                error_code="40100000",
                error_message="모의투자 영업일이 아닙니다.",
            ),
            get_responses=[],
            is_paper=True,
        )
        trading = TradingAPI(client)
        order = Order(
            symbol="005930",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1,
            price=0,
        )

        result = trading.place_order(order)

        self.assertFalse(result.success)
        self.assertEqual(result.error_category, "market_closed")

    def test_place_order_classifies_symbol_order_unavailable(self):
        client = FakeClient(
            post_response=DummyResponse(
                success=False,
                error_code="40070000",
                error_message="모의투자 주문처리가 안되었습니다(매매불가 종목)",
            ),
            get_responses=[],
            is_paper=True,
        )
        trading = TradingAPI(client)
        order = Order(
            symbol="114800",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1,
            price=0,
        )

        result = trading.place_order(order)

        self.assertFalse(result.success)
        self.assertEqual(result.error_category, "symbol_order_unavailable")
        self.assertEqual(result.error_code, "40070000")

    def test_place_order_blocks_subsequent_orders_after_account_unavailable(self):
        client = FakeClient(
            post_response=[
                DummyResponse(
                    success=False,
                    error_code="40910000",
                    error_message="모의투자 주문이 불가한 계좌입니다.",
                ),
                DummyResponse(success=True, output={"ODNO": "should-not-be-used"}),
            ],
            get_responses=[],
            is_paper=True,
        )
        trading = TradingAPI(client)
        first = Order(
            symbol="005930",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1,
            price=0,
        )
        second = Order(
            symbol="000660",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1,
            price=0,
        )

        first_result = trading.place_order(first)
        second_result = trading.place_order(second)

        self.assertFalse(first_result.success)
        self.assertFalse(second_result.success)
        self.assertEqual(first_result.error_category, "account_order_unavailable")
        self.assertEqual(second_result.error_category, "account_order_unavailable")
        self.assertEqual(client.post_calls, 1)

    def test_sell_no_holding_failure_logs_warning_not_error(self):
        client = FakeClient(
            post_response=DummyResponse(
                success=False,
                error_code="40240000",
                error_message="모의투자 잔고내역이 없습니다.",
            ),
            get_responses=[],
        )
        trading = TradingAPI(client)
        order = Order(
            symbol="457370",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=39,
            price=0,
            reference_price=14_530,
        )

        with self.assertLogs("kis_trader.orders", level="WARNING") as logs:
            result = trading.place_order(order)

        self.assertFalse(result.success)
        self.assertEqual(result.error_category, "no_holding")
        self.assertTrue(any(line.startswith("WARNING:") for line in logs.output))
        self.assertFalse(any(line.startswith("ERROR:") for line in logs.output))

    def test_place_order_uses_actual_fill_price_and_qty(self):
        client = FakeClient(
            post_response=DummyResponse(
                success=True,
                output={"ODNO": "12345"},
            ),
            get_responses=[
                DummyResponse(
                    success=True,
                    output1=[{
                        "odno": "12345",
                        "pdno": "005930",
                        "tot_ccld_qty": "3",
                        "avg_prvs": "70100",
                    }],
                )
            ],
        )
        trading = TradingAPI(client)
        order = Order(
            symbol="005930",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=3,
            price=0,
        )

        result = trading.place_order(order)

        self.assertTrue(result.success)
        self.assertEqual(result.order_no, "12345")
        self.assertEqual(result.quantity, 3)
        self.assertEqual(result.price, 70100)

    def test_price_breakout_limit_buy_keeps_partial_fill_and_cancels_remainder(self):
        partial_row = {
            "odno": "303360-order",
            "pdno": "303360",
            "tot_ccld_qty": "2",
            "avg_prvs": "5140",
        }
        client = FakeClient(
            post_response=[
                DummyResponse(success=True, output={"ODNO": "303360-order"}),
                DummyResponse(success=True, output={}),
            ],
            get_responses=[
                DummyResponse(success=True, output1=[{
                    **partial_row, "tot_ccld_qty": "4", "avg_prvs": "5135",
                    "cnc_cfrm_qty": "139", "rmn_qty": "0",
                }]),
            ],
        )
        trading = TradingAPI(client)
        order = Order(
            symbol="303360",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=143,
            price=5_140,
            reference_price=5140,
            requested_reason="price_breakout",
        )

        with patch("src.trading.time.sleep", return_value=None), self.assertNoLogs("kis_trader.orders", level="INFO"):
            result = trading.place_order(order)

        self.assertTrue(result.success)
        self.assertEqual(result.quantity, 4)
        self.assertEqual(result.price, 5135)
        self.assertEqual(result.requested_quantity, 143)
        self.assertEqual(result.fill_mode, "limit_partial")
        self.assertEqual(client.get_calls, 1)
        self.assertEqual(client.post_calls, 2)
        self.assertEqual(client.post_requests[0]["body"]["ORD_DVSN"], "00")
        self.assertEqual(client.post_requests[0]["body"]["ORD_UNPR"], "5140")
        self.assertEqual(client.post_requests[1]["body"]["ORGN_ODNO"], "303360-order")

    def test_place_order_fallback_when_fill_not_found(self):
        client = FakeClient(
            post_response=DummyResponse(
                success=True,
                output={"ODNO": "99999"},
            ),
            get_responses=[
                DummyResponse(success=True, output1=[]),
                DummyResponse(success=True, output1=[]),
                DummyResponse(success=True, output1=[]),
                DummyResponse(success=True, output1=[]),
                DummyResponse(success=True, output1=[]),
            ],
        )
        trading = TradingAPI(client)
        order = Order(
            symbol="005930",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=2,
            price=0,
        )

        with patch("src.trading.time.sleep", return_value=None):
            result = trading.place_order(order)

        self.assertTrue(result.success)
        self.assertEqual(result.order_no, "99999")
        self.assertEqual(result.quantity, 0)
        self.assertEqual(result.price, 0)
        self.assertEqual(result.fill_mode, "market_pending")
        self.assertEqual(client.get_calls, 3)

    def test_market_pending_order_log_includes_reference_price(self):
        client = FakeClient(
            post_response=DummyResponse(
                success=True,
                output={"ODNO": "99997"},
            ),
            get_responses=[
                DummyResponse(success=True, output1=[]),
                DummyResponse(success=True, output1=[]),
                DummyResponse(success=True, output1=[]),
            ],
        )
        trading = TradingAPI(client)
        order = Order(
            symbol="005930",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=2,
            price=0,
            reference_price=70100,
        )

        with patch("src.trading.time.sleep", return_value=None), self.assertLogs("kis_trader.orders", level="INFO") as logs:
            result = trading.place_order(order)

        self.assertTrue(result.success)
        self.assertEqual(result.fill_mode, "market_pending")
        self.assertEqual(result.reference_price, 70100)
        message = "\n".join(logs.output)
        self.assertIn("@ 기준가 70,100원 (시장가)", message)
        self.assertIn("체결가 미확정", message)
        self.assertIn("기준가 70,100원", message)
        self.assertIn("fill_mode=market_pending", message)

    def test_paper_market_sell_defers_fill_lookup_to_account_reconcile(self):
        client = FakeClient(
            post_response=DummyResponse(
                success=True,
                output={"ODNO": "99998"},
            ),
            get_responses=[
                DummyResponse(
                    success=True,
                    output1=[{
                        "odno": "99998",
                        "pdno": "005930",
                        "tot_ccld_qty": "2",
                        "avg_prvs": "70300",
                    }],
                )
            ],
            is_paper=True,
        )
        trading = TradingAPI(client)
        order = Order(
            symbol="005930",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=2,
            price=0,
        )

        result = trading.place_order(order)

        self.assertTrue(result.success)
        self.assertEqual(result.order_no, "99998")
        self.assertEqual(result.quantity, 0)
        self.assertEqual(result.price, 0)
        self.assertEqual(result.fill_mode, "market_pending")
        self.assertEqual(client.get_calls, 0)

    def test_place_order_finds_fill_on_following_order_history_page(self):
        client = FakeClient(
            post_response=DummyResponse(
                success=True,
                output={"ODNO": "77777"},
            ),
            get_responses=[
                DummyResponse(
                    success=True,
                    output1=[],
                    has_next=True,
                    data={
                        "ctx_area_fk100": "NEXTFK",
                        "ctx_area_nk100": "NEXTNK",
                    },
                ),
                DummyResponse(
                    success=True,
                    output1=[{
                        "odno": "77777",
                        "pdno": "005930",
                        "tot_ccld_qty": "2",
                        "avg_prvs": "70300",
                    }],
                ),
            ],
        )
        trading = TradingAPI(client)
        order = Order(
            symbol="005930",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=2,
            price=0,
        )

        with patch("src.trading.time.sleep", return_value=None):
            result = trading.place_order(order)

        self.assertTrue(result.success)
        self.assertEqual(result.quantity, 2)
        self.assertEqual(result.price, 70300)
        self.assertEqual(client.get_calls, 2)

    def test_price_breakout_limit_buy_cancels_without_market_fallback_when_unfilled(self):
        client = FakeClient(
            post_response=[
                DummyResponse(success=True, output={"ODNO": "33333"}),
                DummyResponse(success=True, output={}),
            ],
            get_responses=[
                DummyResponse(success=True, output1=[{
                    "odno": "33333", "pdno": "005930", "tot_ccld_qty": "0",
                    "cnc_cfrm_qty": "2", "rmn_qty": "0",
                }]),
            ],
        )
        trading = TradingAPI(client)
        order = Order(
            symbol="005930",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=2,
            price=70_100,
            reference_price=70_000,
            requested_reason="price_breakout",
        )

        with patch("src.trading.time.sleep", return_value=None):
            result = trading.place_order(order)

        self.assertFalse(result.success)
        self.assertEqual(result.error_category, "not_filled")
        self.assertEqual(result.quantity, 0)
        self.assertEqual(result.fill_mode, "limit_cancelled")
        self.assertEqual(result.requested_price, 70_100)
        self.assertEqual(client.post_calls, 2)
        self.assertEqual(client.post_requests[0]["body"]["ORD_DVSN"], "00")
        self.assertEqual(client.post_requests[1]["api_url"], "/uapi/domestic-stock/v1/trading/order-rvsecncl")

    def test_cancel_acknowledgement_does_not_discard_late_or_unconfirmed_fills(self):
        for final_row, expected in (
            ({"tot_ccld_qty": "2", "avg_prvs": "70100"}, (2, 70100, "limit")),
            ({"tot_ccld_qty": "1", "tot_ccld_amt": "70000", "cnc_cfrm_qty": "1", "rmn_qty": "0"},
             (1, 70000, "limit_partial")),
            ({"tot_ccld_qty": "1", "avg_prvs": "70000", "cncl_yn": "Y", "rmn_qty": "0"},
             (1, 70000, "limit_partial")),
            ({"tot_ccld_qty": "1", "avg_prvs": "0", "cncl_yn": "Y", "rmn_qty": "0"},
             (0, 0, "limit_pending")),
            ({"tot_ccld_qty": "1", "avg_prvs": "70000", "cnc_cfrm_qty": "0", "rmn_qty": "1"},
             (1, 70000, "partial_fill_pending")),
            ({"tot_ccld_qty": "1", "avg_prvs": "0", "cnc_cfrm_qty": "1", "rmn_qty": "0"},
             (0, 0, "limit_pending")),
            (None, (0, 0, "limit_pending")),
        ):
            with self.subTest(final_row=final_row):
                rows = [] if final_row is None else [{"odno": "33333", "pdno": "005930", **final_row}]
                client = FakeClient(
                    post_response=[DummyResponse(True, output={"ODNO": "33333"}), DummyResponse(True)],
                    get_responses=[DummyResponse(True, output1=rows)],
                )
                order = Order("005930", OrderSide.BUY, OrderType.LIMIT, 2, 70100,
                              requested_reason="price_breakout")
                result = TradingAPI(client).place_order(order)
                self.assertTrue(result.success)
                self.assertEqual((result.quantity, result.price, result.fill_mode), expected)
                self.assertEqual(client.post_calls, 2)
                self.assertEqual(client.get_calls, 1)

    def test_failed_cancel_preserves_confirmed_partial_fill(self):
        client = FakeClient(
            post_response=[DummyResponse(True, output={"ODNO": "33333"}),
                           DummyResponse(False, error_message="cancel unavailable")],
            get_responses=[DummyResponse(True, output1=[{
                "odno": "33333", "pdno": "005930", "tot_ccld_qty": "1", "avg_prvs": "70000",
            }])],
        )
        order = Order("005930", OrderSide.BUY, OrderType.LIMIT, 2, 70100,
                      requested_reason="price_breakout")
        result = TradingAPI(client).place_order(order)
        self.assertTrue(result.success)
        self.assertEqual((result.quantity, result.price, result.fill_mode),
                         (1, 70000, "partial_fill_pending"))

    def test_no_cancellable_quantity_still_requires_actual_fill_record(self):
        for rows, expected in (
            ([], (0, 0, "limit_pending")),
            ([{"odno": "33333", "pdno": "005930", "tot_ccld_qty": "2", "avg_prvs": "70000"}],
             (2, 70000, "limit")),
        ):
            with self.subTest(expected=expected):
                client = FakeClient(
                    post_response=[DummyResponse(True, output={"ODNO": "33333"}),
                                   DummyResponse(False, error_message="정정/취소할 수량이 없습니다")],
                    get_responses=[DummyResponse(True, output1=rows)],
                )
                order = Order("005930", OrderSide.BUY, OrderType.LIMIT, 2, 70100,
                              requested_reason="price_breakout")
                result = TradingAPI(client).place_order(order)
                self.assertTrue(result.success)
                self.assertEqual(result.order_no, "33333")
                self.assertEqual((result.quantity, result.price, result.fill_mode), expected)

    def test_buy_order_waits_for_next_tick_when_rate_limited(self):
        client = FakeClient(
            post_response=[
                DummyResponse(success=False, error_message='{"rt_cd":"1","msg1":"초당 거래건수를 초과하였습니다.","message":"EGW00201"}'),
                DummyResponse(success=True, output={"ODNO": "12346"}),
            ],
            get_responses=[],
        )
        client._cooldown_remaining = 0.0
        trading = TradingAPI(client)
        order = Order(
            symbol="005930",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=2,
            price=0,
        )

        with patch("src.trading.time.sleep", return_value=None):
            result = trading.place_order(order)

        self.assertFalse(result.success)
        self.assertEqual(result.error_category, "rate_limit")
        self.assertEqual(client.post_calls, 1)

    def test_sell_order_retries_when_rate_limited(self):
        client = FakeClient(
            post_response=[
                DummyResponse(success=False, error_message='{"rt_cd":"1","msg1":"초당 거래건수를 초과하였습니다.","message":"EGW00201"}'),
                DummyResponse(success=True, output={"ODNO": "12346"}),
            ],
            get_responses=[
                DummyResponse(
                    success=True,
                    output1=[{
                        "odno": "12346",
                        "pdno": "005930",
                        "tot_ccld_qty": "2",
                        "avg_prvs": "70200",
                    }],
                )
            ],
        )
        client._cooldown_remaining = 0.0
        trading = TradingAPI(client)
        order = Order(
            symbol="005930",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=2,
            price=0,
        )

        with patch("src.trading.time.sleep", return_value=None):
            result = trading.place_order(order)

        self.assertTrue(result.success)
        self.assertEqual(result.order_no, "12346")
        self.assertEqual(result.quantity, 2)
        self.assertEqual(client.post_calls, 2)


if __name__ == "__main__":
    unittest.main()
