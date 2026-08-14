import unittest
from types import SimpleNamespace
from unittest.mock import patch

import requests

from src.executor import OrderExecutor, RiskManager
from src.models import Order, OrderResult, OrderSide, OrderType


class _FakeTrading:
    def __init__(self, results, *, is_paper=True):
        self._results = list(results)
        self.calls = []
        self.client = SimpleNamespace(config=SimpleNamespace(is_paper=is_paper))

    def place_order(self, order):
        self.calls.append(order.symbol)
        if self._results:
            return self._results.pop(0)
        return OrderResult(
            success=False,
            message="no result",
            symbol=order.symbol,
            side=order.side,
        )


class _RaisingTrading:
    client = SimpleNamespace(config=SimpleNamespace(is_paper=True))

    def place_order(self, order):
        raise requests.exceptions.ConnectionError("Remote end closed connection without response")


class OrderExecutorTests(unittest.TestCase):
    def test_submit_orders_continues_after_rate_limited_failure(self):
        trading = _FakeTrading(
            [
                OrderResult(
                    success=False,
                    message='{"rt_cd":"1","msg1":"초당 거래건수를 초과하였습니다.","message":"EGW00201"}',
                    symbol="005930",
                    side=OrderSide.BUY,
                ),
                OrderResult(
                    success=True,
                    message="should_not_run",
                    symbol="000660",
                    side=OrderSide.BUY,
                    quantity=1,
                    price=1000,
                ),
            ]
        )
        executor = OrderExecutor(trading, RiskManager())
        orders = [
            Order(symbol="005930", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1, price=0),
            Order(symbol="000660", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1, price=0),
        ]

        with patch("src.executor.time.sleep", return_value=None):
            results = executor.submit_orders(orders)

        self.assertEqual(len(results), 2)
        self.assertEqual(trading.calls, ["005930", "000660"])

    def test_submit_orders_treats_ledger_quota_error_as_rate_limit(self):
        trading = _FakeTrading(
            [
                OrderResult(
                    success=False,
                    message=(
                        '{"rt_cd":"1","msg1":"원장에서 허용 가능한 초당 거래건수를 초과하였습니다.",'
                        '"msg_cd":"EGW00215"}'
                    ),
                    symbol="005930",
                    side=OrderSide.BUY,
                ),
                OrderResult(
                    success=True,
                    message="ok",
                    symbol="000660",
                    side=OrderSide.BUY,
                    quantity=1,
                    price=1000,
                ),
            ]
        )
        executor = OrderExecutor(trading, RiskManager())
        orders = [
            Order(symbol="005930", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1, price=0),
            Order(symbol="000660", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1, price=0),
        ]

        with patch("src.executor.time.sleep", return_value=None) as mocked_sleep:
            results = executor.submit_orders(orders)

        self.assertEqual(len(results), 2)
        self.assertEqual(trading.calls, ["005930", "000660"])
        mocked_sleep.assert_called_once()

    def test_submit_orders_waits_between_multiple_orders(self):
        trading = _FakeTrading(
            [
                OrderResult(success=True, message="ok", symbol="005930", side=OrderSide.BUY, quantity=1, price=1000),
                OrderResult(success=True, message="ok", symbol="000660", side=OrderSide.BUY, quantity=1, price=1000),
            ],
            is_paper=True,
        )
        executor = OrderExecutor(trading, RiskManager())
        orders = [
            Order(symbol="005930", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1, price=0),
            Order(symbol="000660", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1, price=0),
        ]

        with patch("src.executor.time.sleep", return_value=None) as mocked_sleep:
            results = executor.submit_orders(orders)

        self.assertEqual(len(results), 2)
        mocked_sleep.assert_called_once_with(0.45)

    def test_submit_orders_returns_pending_result_on_network_disconnect(self):
        executor = OrderExecutor(_RaisingTrading(), RiskManager())
        order = Order(
            symbol="457370",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=39,
            price=0,
            reference_price=14360,
            requested_reason="expected_value",
        )

        with self.assertLogs("kis_trader.orders", level="WARNING") as captured:
            results = executor.submit_orders([order])

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertTrue(result.success)
        self.assertEqual(result.symbol, "457370")
        self.assertEqual(result.side, OrderSide.BUY)
        self.assertEqual(result.quantity, 0)
        self.assertEqual(result.reference_price, 14360)
        self.assertEqual(result.fill_mode, "order_result_pending")
        self.assertEqual(result.requested_reason, "expected_value")
        message = "\n".join(captured.output)
        self.assertIn(
            "주문 결과 미확정: buy 457370 MARKET 39주 @ 시장가 "
            "(체결가 미확정, 신호가 14,360원",
            message,
        )
        self.assertIn("fill_mode=order_result_pending", message)


if __name__ == "__main__":
    unittest.main()
