import unittest
from unittest.mock import patch
from types import SimpleNamespace

from src.models import Order, OrderSide, OrderType
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

    def test_expected_value_limit_buy_keeps_partial_fill_and_cancels_remainder(self):
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
                DummyResponse(success=True, output1=[partial_row]),
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
            requested_reason="expected_value",
        )

        with patch("src.trading.time.sleep", return_value=None), self.assertLogs("kis_trader.orders", level="INFO") as logs:
            result = trading.place_order(order)

        self.assertTrue(result.success)
        self.assertEqual(result.quantity, 2)
        self.assertEqual(result.requested_quantity, 143)
        self.assertEqual(result.fill_mode, "limit_partial")
        self.assertEqual(client.get_calls, 1)
        self.assertEqual(client.post_calls, 2)
        self.assertEqual(client.post_requests[0]["body"]["ORD_DVSN"], "00")
        self.assertEqual(client.post_requests[0]["body"]["ORD_UNPR"], "5140")
        self.assertEqual(client.post_requests[1]["body"]["ORGN_ODNO"], "303360-order")
        self.assertIn("매수 부분체결", "\n".join(logs.output))

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

    def test_expected_value_limit_buy_cancels_without_market_fallback_when_unfilled(self):
        client = FakeClient(
            post_response=[
                DummyResponse(success=True, output={"ODNO": "33333"}),
                DummyResponse(success=True, output={}),
            ],
            get_responses=[DummyResponse(success=True, output1=[])],
        )
        trading = TradingAPI(client)
        order = Order(
            symbol="005930",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=2,
            price=70_100,
            reference_price=70_000,
            requested_reason="expected_value",
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
