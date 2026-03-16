import unittest

import pandas as pd
from types import SimpleNamespace

from src.backtest.engine import BacktestEngine
from src.backtest.intraday_engine import IntradayBacktestEngine
from src.models import Order, OrderResult, OrderSide, Quote
from src.strategy import BaseStrategy


class DummyBacktestStrategy(BaseStrategy):
    def __init__(self):
        self.has_position = False
        self.completed_round_trip = False
        self.simulated_times = []
        self.positions = {}
        self._inverse_symbols = set()

    def set_simulated_now(self, now):
        if now is not None:
            self.simulated_times.append(now)

    def initialize(self):
        self.has_position = False
        self.completed_round_trip = False
        self.positions = {}

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
            self.positions[result.symbol] = SimpleNamespace(
                entry_strategy_name="bull_breakout_strategy",
                entry_setup_name="bull_breakout",
                entry_reason="local_high_breakout",
                regime_label="bull",
                bear_score=0,
                planned_risk_stage="normal",
            )
        elif result.side == OrderSide.SELL:
            self.has_position = False
            self.completed_round_trip = True
            self.positions.pop(result.symbol, None)

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
        self.assertEqual(result.trade_records[0].strategy_name, "bull_breakout_strategy")
        self.assertEqual(result.trade_records[0].setup_name, "bull_breakout")
        self.assertEqual(result.trade_records[1].regime_label, "bull")
        self.assertTrue(strategy.simulated_times)
        self.assertEqual(strategy.simulated_times[0].strftime("%Y%m%d"), "20260102")
        self.assertEqual(strategy.simulated_times[-1].hour, 15)

    def test_intraday_engine_replays_minute_bars(self):
        strategy = DummyBacktestStrategy()
        data = {
            "AAA": pd.DataFrame(
                [
                    {
                        "symbol": "AAA",
                        "trade_date": "20260102",
                        "trade_time": "090000",
                        "timestamp": "2026-01-02 09:00:00",
                        "open": 10_000,
                        "high": 10_020,
                        "low": 9_990,
                        "close": 10_010,
                        "volume": 1_000,
                        "cumulative_volume": 1_000,
                    },
                    {
                        "symbol": "AAA",
                        "trade_date": "20260102",
                        "trade_time": "090100",
                        "timestamp": "2026-01-02 09:01:00",
                        "open": 10_010,
                        "high": 10_040,
                        "low": 10_000,
                        "close": 10_030,
                        "volume": 1_200,
                        "cumulative_volume": 2_200,
                    },
                    {
                        "symbol": "AAA",
                        "trade_date": "20260102",
                        "trade_time": "090200",
                        "timestamp": "2026-01-02 09:02:00",
                        "open": 10_030,
                        "high": 10_060,
                        "low": 10_020,
                        "close": 10_050,
                        "volume": 1_400,
                        "cumulative_volume": 3_600,
                    },
                ]
            )
        }
        engine = IntradayBacktestEngine(strategy=strategy, data=data, initial_capital=100_000)

        result = engine.run("20260102", "20260102")

        self.assertEqual(result.total_trades, 2)
        self.assertEqual(len(result.daily_records), 1)
        self.assertEqual(result.daily_records[0].trade_count, 2)
        self.assertEqual([record.date for record in result.trade_records], ["20260102", "20260102"])
        self.assertTrue(strategy.simulated_times)
        self.assertEqual(strategy.simulated_times[0].strftime("%Y%m%d"), "20260102")


if __name__ == "__main__":
    unittest.main()
