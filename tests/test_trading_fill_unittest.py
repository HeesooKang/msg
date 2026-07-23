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
        self._cooldown_remaining = 0.0
        self.config = SimpleNamespace(is_paper=is_paper)

    def post(self, **kwargs):
        self.post_calls += 1
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

    def test_place_order_keeps_partial_buy_fill_pending(self):
        partial_row = {
            "odno": "303360-order",
            "pdno": "303360",
            "tot_ccld_qty": "2",
            "avg_prvs": "5140",
        }
        client = FakeClient(
            post_response=DummyResponse(success=True, output={"ODNO": "303360-order"}),
            get_responses=[
                DummyResponse(success=True, output1=[partial_row]),
                DummyResponse(success=True, output1=[partial_row]),
                DummyResponse(success=True, output1=[partial_row]),
            ],
        )
        trading = TradingAPI(client)
        order = Order(
            symbol="303360",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=143,
            reference_price=5140,
        )

        with patch("src.trading.time.sleep", return_value=None), self.assertLogs("kis_trader.orders", level="INFO") as logs:
            result = trading.place_order(order)

        self.assertTrue(result.success)
        self.assertEqual(result.quantity, 2)
        self.assertEqual(result.requested_quantity, 143)
        self.assertEqual(result.fill_mode, "partial_fill_pending")
        self.assertEqual(client.get_calls, 1)
        self.assertIn("부분체결 2/143주", "\n".join(logs.output))

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

    def test_place_order_uses_limit_then_market_for_protective_exit(self):
        client = FakeClient(
            post_response=[
                DummyResponse(success=True, output={"ODNO": "11111"}),
                DummyResponse(success=True, output={}),
                DummyResponse(success=True, output={"ODNO": "22222"}),
            ],
            get_responses=[
                DummyResponse(success=True, output1=[]),
                DummyResponse(
                    success=True,
                    output1=[{
                        "odno": "22222",
                        "pdno": "005930",
                        "tot_ccld_qty": "5",
                        "avg_prvs": "9910",
                    }],
                ),
            ],
        )
        trading = TradingAPI(client)
        order = Order(
            symbol="005930",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=5,
            price=0,
            protective_exit_mode="limit_then_market",
            protective_limit_price=9930,
            protective_fallback_polls=1,
            stop_reference_amount_krw=1500,
        )

        with patch("src.trading.time.sleep", return_value=None):
            result = trading.place_order(order)

        self.assertTrue(result.success)
        self.assertEqual(result.quantity, 5)
        self.assertEqual(result.fill_mode, "limit_then_market")
        self.assertTrue(result.protective_fallback_used)
        self.assertEqual(result.requested_price, 9930)
        self.assertEqual(result.stop_reference_amount_krw, 1500)
        self.assertEqual(client.post_calls, 3)

    def test_protective_exit_treats_non_cancellable_limit_as_filled(self):
        client = FakeClient(
            post_response=[
                DummyResponse(success=True, output={"ODNO": "11111"}),
                DummyResponse(success=False, error_message="모의투자 정정/취소할 수량이 없습니다."),
            ],
            get_responses=[
                DummyResponse(success=True, output1=[]),
            ],
        )
        trading = TradingAPI(client)
        order = Order(
            symbol="010170",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=25,
            price=0,
            protective_exit_mode="limit_then_market",
            protective_limit_price=13500,
            protective_fallback_polls=1,
            stop_reference_amount_krw=2385,
        )

        with patch("src.trading.time.sleep", return_value=None):
            result = trading.place_order(order)

        self.assertTrue(result.success)
        self.assertEqual(result.quantity, 25)
        self.assertEqual(result.price, 13500)
        self.assertEqual(result.fill_mode, "limit_assumed_filled")
        self.assertEqual(result.protective_exit_mode, "limit_then_market")
        self.assertEqual(result.stop_reference_amount_krw, 2385)

    def test_protective_exit_falls_back_to_market_on_tick_error(self):
        client = FakeClient(
            post_response=[
                DummyResponse(success=False, error_code="40030000", error_message="모의투자 주문처리가 안되었습니다(호가단위 오류)"),
                DummyResponse(success=True, output={"ODNO": "22222"}),
            ],
            get_responses=[
                DummyResponse(
                    success=True,
                    output1=[{
                        "odno": "22222",
                        "pdno": "464930",
                        "tot_ccld_qty": "1",
                        "avg_prvs": "19655",
                    }],
                )
            ],
        )
        trading = TradingAPI(client)
        order = Order(
            symbol="464930",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=1,
            price=0,
            protective_exit_mode="limit_then_market",
            protective_limit_price=19660,
            protective_fallback_polls=1,
            stop_reference_amount_krw=1500,
        )

        with patch("src.trading.time.sleep", return_value=None):
            result = trading.place_order(order)

        self.assertTrue(result.success)
        self.assertEqual(result.quantity, 1)
        self.assertEqual(result.price, 19655)
        self.assertEqual(result.fill_mode, "market_after_limit_reject")
        self.assertTrue(result.protective_fallback_used)
        self.assertEqual(result.requested_price, 19660)

    def test_protective_exit_falls_back_to_market_on_generic_limit_reject(self):
        client = FakeClient(
            post_response=[
                DummyResponse(success=False, error_code="99999999", error_message="지정가 주문 검증 실패"),
                DummyResponse(success=True, output={"ODNO": "33333"}),
            ],
            get_responses=[
                DummyResponse(
                    success=True,
                    output1=[{
                        "odno": "33333",
                        "pdno": "005930",
                        "tot_ccld_qty": "2",
                        "avg_prvs": "70000",
                    }],
                )
            ],
        )
        trading = TradingAPI(client)
        order = Order(
            symbol="005930",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=2,
            price=0,
            protective_exit_mode="limit_then_market",
            protective_limit_price=70100,
            protective_fallback_polls=1,
            stop_reference_amount_krw=1500,
        )

        with patch("src.trading.time.sleep", return_value=None):
            result = trading.place_order(order)

        self.assertTrue(result.success)
        self.assertEqual(result.quantity, 2)
        self.assertEqual(result.price, 70000)
        self.assertEqual(result.fill_mode, "market_after_limit_reject")
        self.assertTrue(result.protective_fallback_used)
        self.assertEqual(result.requested_price, 70100)

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
