import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from src.analytics.price_prediction import PREDICTION_FEATURES, Prediction
from src.models import OrderResult, OrderSide, OrderType, Quote
from src.strategies.momentum_scalp import MomentumScalpStrategy
from src.strategies.momentum_scalp_types import (
    MomentumScalpConfig,
    PositionState,
    TradePlan,
)


def make_quote(
    symbol: str,
    price: int,
    timestamp: datetime,
    *,
    ask: int | None = None,
    bid: int | None = None,
) -> Quote:
    return Quote(
        symbol=symbol,
        current_price=price,
        timestamp=timestamp,
        ask_price=price + 10 if ask is None else ask,
        bid_price=price - 10 if bid is None else bid,
    )


def prediction(
    expected_net_return_pct: float,
    *,
    lower_net_return_pct: float = -0.5,
    ready: bool = True,
    horizon_seconds: int = 180,
) -> Prediction:
    return Prediction(
        ready=ready,
        reason="ok" if ready else "not_ready",
        horizon_seconds=horizon_seconds,
        sample_count=20 if ready else 0,
        expected_net_return_pct=expected_net_return_pct,
        lower_net_return_pct=lower_net_return_pct,
        compact_features={name: 0.0 for name in PREDICTION_FEATURES},
    )


def make_plan(
    symbol: str,
    now: datetime,
    *,
    quantity: int = 10,
    entry: int = 10_010,
    expected_exit: int = 10_110,
    lower_exit: int = 9_950,
    expected_net: float = 700.0,
    lower_net: int = -800,
    risk: int = 800,
    expires_delta: int = 180,
) -> TradePlan:
    return TradePlan(
        symbol=symbol,
        allowed=True,
        reason="positive_expected_value",
        quantity=quantity,
        entry_limit_price=entry,
        expected_exit_bid=expected_exit,
        lower_exit_bid=lower_exit,
        expected_net=expected_net,
        lower_net=lower_net,
        committed_risk_net_abs=risk,
        signal_at=now,
        expires_at=now + timedelta(seconds=expires_delta),
    )


class ExpectedValueStrategyContractTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.now = datetime(2026, 8, 4, 10, 0, 0)

    def config(self, **overrides):
        values = {
            "static_watchlist": [],
            "inverse_etfs": [],
            "daily_state_path": str(Path(self.directory.name) / "state.json"),
            "forecast_outcome_root": str(Path(self.directory.name) / "forecasts"),
        }
        values.update(overrides)
        return MomentumScalpConfig(**values)

    def strategy(self, **overrides) -> MomentumScalpStrategy:
        strategy = MomentumScalpStrategy(None, self.config(**overrides))
        strategy.set_simulated_now(self.now)
        strategy.initialize()
        return strategy

    def test_small_positive_expected_net_is_allowed_below_remaining_daily_target(self):
        strategy = self.strategy()
        quote = make_quote("005930", 10_000, self.now)

        plan = strategy._build_trade_plan(quote, prediction(0.35), now=self.now)

        self.assertTrue(plan.allowed)
        self.assertGreater(plan.expected_net, 0.0)
        self.assertLess(plan.expected_net, strategy.config.daily_profit_target)

    def test_unaffordable_one_share_is_rejected(self):
        strategy = self.strategy(seed_money=1_000_000)
        quote = make_quote("005930", 2_300_000, self.now)

        plan = strategy._build_trade_plan(quote, prediction(2.0), now=self.now)

        self.assertFalse(plan.allowed)
        self.assertEqual(plan.reason, "unaffordable_one_share")

    def test_negative_execution_ev_is_rejected(self):
        strategy = self.strategy()
        quote = make_quote("005930", 10_000, self.now)

        plan = strategy._build_trade_plan(quote, prediction(-0.1), now=self.now)

        self.assertFalse(plan.allowed)
        self.assertEqual(plan.reason, "non_positive_expected_net")

    def test_only_remaining_three_hundred_won_loss_room_sizes_quantity(self):
        strategy = self.strategy()
        strategy.daily_pnl.realized_net_pnl = -4_700
        quote = make_quote("005930", 10_000, self.now)

        plan = strategy._build_trade_plan(
            quote,
            prediction(2.0, lower_net_return_pct=-1.0),
            now=self.now,
        )

        self.assertTrue(plan.allowed)
        self.assertLessEqual(plan.committed_risk_net_abs, 300)
        self.assertLess(plan.quantity, strategy.config.seed_money // plan.entry_limit_price)

    def test_inverse_and_regular_symbols_use_identical_plan_formula(self):
        strategy = self.strategy()
        regular = make_quote("005930", 10_000, self.now)
        inverse = make_quote("114800", 10_000, self.now)
        forecast = prediction(1.0, lower_net_return_pct=-0.4)

        regular_plan = strategy._build_trade_plan(regular, forecast, now=self.now)
        inverse_plan = strategy._build_trade_plan(inverse, forecast, now=self.now)

        self.assertEqual(regular_plan.quantity, inverse_plan.quantity)
        self.assertEqual(regular_plan.expected_net, inverse_plan.expected_net)
        self.assertEqual(regular_plan.committed_risk_net_abs, inverse_plan.committed_risk_net_abs)

    def test_single_batch_selects_highest_expected_net_across_symbols_and_horizons(self):
        strategy = self.strategy()
        low = make_quote("005930", 10_000, self.now)
        high = make_quote("000660", 10_000, self.now)
        predictions = {
            horizon: {
                low.symbol: prediction(0.5, horizon_seconds=horizon),
                high.symbol: prediction(
                    1.0 if horizon == 60 else 0.4,
                    horizon_seconds=horizon,
                ),
            }
            for horizon in (30, 60, 120, 180)
        }

        with patch.object(
            strategy,
            "_build_predictions",
            return_value=predictions,
        ) as predictor:
            orders = strategy.on_batch_tick([low, high])

        predictor.assert_called_once()
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].symbol, high.symbol)
        self.assertEqual(orders[0].requested_reason, "expected_value")
        self.assertEqual(
            strategy._pending_entry_meta[high.symbol]["plan"].expires_at,
            self.now + timedelta(seconds=60),
        )

    def test_buy_fill_keeps_original_plan_without_second_prediction(self):
        strategy = self.strategy()
        quote = make_quote("005930", 10_000, self.now)
        initial = strategy._build_trade_plan(quote, prediction(1.0), now=self.now)
        strategy._quotes_cache[quote.symbol] = quote
        strategy._pending_entry_meta[quote.symbol] = {
            "plan": initial,
            "forecast_id": "F1",
            "signal_price": quote.current_price,
            "created_at": self.now,
            "order_no": "",
        }

        position = strategy._apply_buy_fill(
            OrderResult(success=True, symbol=quote.symbol, side=OrderSide.BUY),
            quantity=initial.quantity,
            price=initial.entry_limit_price,
        )

        self.assertFalse(hasattr(position, "needs_fill_revalidation"))
        self.assertEqual(position.plan.signal_at, self.now)
        self.assertEqual(position.plan.expires_at, self.now + timedelta(seconds=180))

    def test_live_quote_without_sell_offer_is_not_purchasable(self):
        strategy = self.strategy()
        quote = make_quote("005930", 10_000, self.now, ask=0, bid=10_000)
        quote.book_available = True

        plan = strategy._build_trade_plan(quote, prediction(1.0), now=self.now)

        self.assertFalse(plan.allowed)
        self.assertEqual(plan.reason, "no_executable_ask")

    def test_previous_trading_day_quote_cannot_enter_prediction_history(self):
        strategy = self.strategy()
        previous_day = make_quote(
            "005930",
            9_800,
            self.now - timedelta(days=1),
        )
        current = make_quote("005930", 10_000, self.now)

        strategy._ingest_quotes([previous_day, current])

        self.assertEqual(list(strategy._recent_quotes["005930"]), [current])

    def test_position_has_only_horizon_expiry_exit_before_daily_hardstop(self):
        strategy = self.strategy()
        quote = make_quote("005930", 12_000, self.now)
        position = PositionState(
            symbol=quote.symbol,
            buy_price=10_000,
            quantity=10,
            buy_time=self.now - timedelta(seconds=100),
            plan=make_plan(quote.symbol, self.now, expires_delta=1),
        )

        self.assertIsNone(strategy._position_exit_order(position, quote, now=self.now))
        order = strategy._position_exit_order(
            position,
            quote,
            now=self.now + timedelta(seconds=1),
        )
        self.assertEqual(order.order_type, OrderType.MARKET)
        self.assertEqual(order.requested_reason, "horizon_expiry")

    def test_daily_loss_hardstop_liquidates_and_profit_target_stops_entries(self):
        strategy = self.strategy()
        quote = make_quote("005930", 10_000, self.now)
        strategy.positions[quote.symbol] = PositionState(
            symbol=quote.symbol,
            buy_price=10_000,
            quantity=10,
            plan=make_plan(quote.symbol, self.now),
        )
        strategy.daily_pnl.realized_net_pnl = -5_000

        orders = strategy.on_batch_tick([quote])

        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].side, OrderSide.SELL)
        self.assertFalse(strategy.should_continue())

        profit_strategy = self.strategy()
        profit_strategy.daily_pnl.realized_net_pnl = 10_000
        self.assertFalse(profit_strategy.should_continue())
        self.assertEqual(profit_strategy._halt_reason, "daily_profit_target")

    def test_hardstop_bypass_uses_restart_realized_baseline(self):
        strategy = self.strategy(allow_hard_stop_bypass_for_day=True)
        strategy.daily_pnl.realized_net_pnl = -5_000
        strategy._loss_guard_realized_baseline = -5_000
        strategy._hard_stop_bypass_active = True

        self.assertTrue(strategy.should_continue())
        self.assertEqual(strategy._realized_net_pnl_for_loss_guard(), 0)

if __name__ == "__main__":
    unittest.main()
