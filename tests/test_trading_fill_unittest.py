import unittest
from unittest.mock import patch

from src.models import Order, OrderSide, OrderType
from src.trading import TradingAPI


class DummyResponse:
    def __init__(self, success, output=None, output1=None, error_code="", error_message=""):
        self.success = success
        self.output = output or {}
        self.output1 = output1 or []
        self.error_code = error_code
        self.error_message = error_message


class FakeClient:
    def __init__(self, post_response, get_responses):
        self.post_response = post_response
        self.get_responses = list(get_responses)
        self.get_calls = 0

    def post(self, **kwargs):
        return self.post_response

    def get(self, **kwargs):
        self.get_calls += 1
        if self.get_responses:
            return self.get_responses.pop(0)
        return DummyResponse(success=True, output1=[])


class TradingFillPriceTests(unittest.TestCase):
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
        self.assertEqual(result.quantity, 2)  # 주문 수량 fallback
        self.assertEqual(result.price, 0)     # 시장가 fallback
        self.assertEqual(client.get_calls, 5)


if __name__ == "__main__":
    unittest.main()
