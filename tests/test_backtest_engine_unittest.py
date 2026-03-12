import unittest

import pandas as pd

from src.backtest.engine import BacktestEngine
from src.models import Order, OrderResult, OrderSide, Quote
from src.strategy import BaseStrategy


class DummyBacktestStrategy(BaseStrategy):
    def __init__(self):
        self.has_position = False
        self.completed_round_trip = False
        self.simulated_times = []

    def set_simulated_now(self, now):
        if now is not None:
            self.simulated_times.append(now)

    def initialize(self):
        self.has_position = False
        self.completed_round_trip = False

    def get_watchlist(self):
        return ["AAA"]

    def on_tick(self, quote):
        return []

    def on_batch_tick(self, quotes):
        quote = quotes[0]
        if self.completed_round_trip:
            return []
        if not self.has_position:
            return [Order(symbol=quote.symbol, side=OrderSide.BUY, quantity=1)]
        return [Order(symbol=quote.symbol, side=OrderSide.SELL, quantity=1)]

    def on_order_filled(self, result: OrderResult):
        if not result.success:
            return
        if result.side == OrderSide.BUY:
            self.has_position = True
        elif result.side == OrderSide.SELL:
            self.has_position = False
            self.completed_round_trip = True

    def should_continue(self):
        return True


class BacktestEngineTests(unittest.TestCase):
    def test_engine_tracks_daily_trade_count_and_trade_dates(self):
        strategy = DummyBacktestStrategy()
        data = {
            "AAA": pd.DataFrame(
                [
                    {
                        "stck_bsop_date": "20260102",
                        "stck_oprc": 10_000,
                        "stck_hgpr": 10_500,
                        "stck_lwpr": 9_900,
                        "stck_clpr": 10_300,
                        "stck_prdy_clpr": 9_800,
                        "acml_vol": 1_200_000,
                    }
                ]
            )
        }
        engine = BacktestEngine(strategy=strategy, data=data, initial_capital=100_000)

        result = engine.run("20260102", "20260102")

        self.assertEqual(result.total_trades, 2)
        self.assertEqual(len(result.daily_records), 1)
        self.assertEqual(result.daily_records[0].trade_count, 2)
        self.assertEqual([record.date for record in result.trade_records], ["20260102", "20260102"])
        self.assertTrue(strategy.simulated_times)
        self.assertEqual(strategy.simulated_times[0].strftime("%Y%m%d"), "20260102")
        self.assertEqual(strategy.simulated_times[-1].hour, 15)


if __name__ == "__main__":
    unittest.main()
