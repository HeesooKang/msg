import unittest

from src.strategies.momentum_scalp_exit import (
    LongExitSnapshot,
    decide_long_exit,
)
from src.strategies.momentum_scalp_pnl import (
    calculate_trade_pnl_from_prices,
    estimate_trade_net_pnl_from_prices,
)
from src.strategies.momentum_scalp_state import rebuild_daily_pnl_snapshot_from_ledgers


class MomentumScalpRefactorTests(unittest.TestCase):
    def test_trade_pnl_helper_uses_one_net_cost_model(self):
        pnl = calculate_trade_pnl_from_prices(
            entry_price=100_000,
            exit_price=101_500,
            quantity=9,
            commission_rate=0.00015,
            tax_slippage_rate=0.002,
        )
        net = estimate_trade_net_pnl_from_prices(
            entry_price=100_000,
            exit_price=101_500,
            quantity=9,
            commission_rate=0.00015,
            tax_slippage_rate=0.002,
        )

        self.assertEqual(pnl.gross_pnl, 13_500)
        self.assertEqual(pnl.net_pnl, 11_401)
        self.assertEqual(pnl.fees, 272)
        self.assertEqual(pnl.taxes, 1_827)
        self.assertEqual(net, 11_401)







    def test_decide_long_exit_preserves_stop_and_profit_priority(self):
        stop_decision = decide_long_exit(
            LongExitSnapshot(
                quantity=3,
                held_minutes=10.0,
                held_seconds=600.0,
                gain_pct=2.0,
                trail_drawdown_pct=0.0,
                unrealized_pnl=6_000,
                estimated_net_pnl=-2_100,
                net_stop_amount=2_000,
                take_profit_pct=1.5,
                adaptive_target=False,
                max_hold_minutes=45.0,
                min_trailing_seconds=0.0,
                trailing_activation_pct=0.7,
                trailing_stop_pct=-0.3,
                adaptive_stop_loss_pct=0.0,
            ),
            partial_exit_done=False,
            partial_exit_ratio=0.5,
        )
        partial_decision = decide_long_exit(
            LongExitSnapshot(
                quantity=3,
                held_minutes=10.0,
                held_seconds=600.0,
                gain_pct=1.0,
                trail_drawdown_pct=0.0,
                unrealized_pnl=3_000,
                estimated_net_pnl=2_300,
                net_stop_amount=2_000,
                take_profit_pct=1.5,
                adaptive_target=False,
                max_hold_minutes=45.0,
                min_trailing_seconds=0.0,
                trailing_activation_pct=1.2,
                trailing_stop_pct=-0.3,
                adaptive_stop_loss_pct=0.0,
            ),
            partial_exit_done=False,
            partial_exit_ratio=0.5,
        )
        cost_negative_time_decision = decide_long_exit(
            LongExitSnapshot(
                quantity=2,
                held_minutes=18.0,
                held_seconds=1080.0,
                gain_pct=0.20,
                trail_drawdown_pct=0.0,
                unrealized_pnl=1_500,
                estimated_net_pnl=-100,
                net_stop_amount=2_000,
                take_profit_pct=0.70,
                adaptive_target=True,
                max_hold_minutes=17.0,
                min_trailing_seconds=0.0,
                trailing_activation_pct=0.4,
                trailing_stop_pct=-0.3,
                adaptive_stop_loss_pct=0.0,
            ),
            partial_exit_done=False,
            partial_exit_ratio=0.5,
        )

        self.assertEqual(stop_decision.reason, "protective_stop_net")
        self.assertEqual(stop_decision.quantity, 3)
        self.assertEqual(partial_decision.reason, "partial_take_profit")
        self.assertEqual(partial_decision.quantity, 2)
        self.assertFalse(cost_negative_time_decision.should_exit)

    def test_decide_long_exit_defers_trailing_until_min_hold_seconds(self):
        decision = decide_long_exit(
            LongExitSnapshot(
                quantity=1,
                held_minutes=1.0,
                held_seconds=60.0,
                gain_pct=1.4,
                trail_drawdown_pct=-0.5,
                unrealized_pnl=1_200,
                estimated_net_pnl=900,
                net_stop_amount=2_000,
                take_profit_pct=2.0,
                adaptive_target=True,
                max_hold_minutes=45.0,
                min_trailing_seconds=180.0,
                trailing_activation_pct=0.7,
                trailing_stop_pct=-0.3,
                adaptive_stop_loss_pct=0.0,
            ),
            partial_exit_done=False,
            partial_exit_ratio=0.5,
        )

        self.assertFalse(decision.should_exit)

    def test_rebuild_daily_pnl_snapshot_from_ledgers_aggregates_single_source(self):
        snapshot = rebuild_daily_pnl_snapshot_from_ledgers(
            seed_snapshot={"realized_net_pnl": 100, "breaker_excluded_realized_net_pnl": 10},
            sell_fill_ledger=[
                {"gross_pnl": 2_000, "net_pnl": 1_500, "fees": 100, "taxes": 400},
                {
                    "gross_pnl": -700,
                    "net_pnl": -900,
                    "fees": 50,
                    "taxes": 150,
                    "counts_for_daily_breaker": False,
                },
            ],
            closed_trade_ledger={
                "win": {"net_pnl": 1_500},
                "loss": {"net_pnl": -900},
                "flat": {"net_pnl": 0},
            },
        )

        self.assertEqual(snapshot["realized_gross_pnl"], 1_300)
        self.assertEqual(snapshot["realized_net_pnl"], 700)
        self.assertEqual(snapshot["fees_paid"], 150)
        self.assertEqual(snapshot["taxes_paid"], 550)
        self.assertEqual(snapshot["breaker_excluded_realized_net_pnl"], -890)
        self.assertEqual(snapshot["trade_count"], 3)
        self.assertEqual(snapshot["win_count"], 1)
        self.assertEqual(snapshot["loss_count"], 1)
        self.assertEqual(snapshot["breakeven_count"], 1)
        self.assertEqual(snapshot["winning_net_pnl_sum"], 1_500)
        self.assertEqual(snapshot["losing_net_pnl_sum"], -900)



if __name__ == "__main__":
    unittest.main()
