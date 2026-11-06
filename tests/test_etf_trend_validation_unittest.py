from contextlib import redirect_stdout
from io import StringIO
import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from scripts.validate_etf_trend import BidChannels, ETFTrendReplay, SYMBOLS, comparison_protocol, main
from src.models import OrderResult, OrderSide, Quote
from src.strategies.momentum_scalp_types import MomentumScalpConfig


class ETFTrendValidationTests(unittest.TestCase):
    def setUp(self):
        self.start = datetime(2026, 9, 21, 9)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.strategy = ETFTrendReplay(None, MomentumScalpConfig(
            commission_rate=.000142, daily_state_path=str(Path(self.tmp.name) / "state.json")))
        self.strategy.fixed_pool = ["152100", "114800"]
        self.strategy.instrument_groups.update({s:"EF" for s in self.strategy.fixed_pool})

    def quote(self, second, bid=10000, symbol="152100"):
        return Quote(symbol, bid, self.start + timedelta(seconds=second),
                     ask_price=bid+5, bid_price=bid, ask_size=10000, bid_size=10000,
                     book_available=True, book_depth_available=True, security_group="EF")

    def tick(self, second, bid=10000, symbol="152100"):
        q = self.quote(second, bid, symbol)
        self.strategy.set_simulated_now(q.timestamp)
        return self.strategy.on_batch_tick([q])

    def warm(self):
        for minute in range(21):
            for second in range(60):
                self.assertEqual(self.tick(minute*60+second,
                    10010 if minute == 20 and second == 59 else 10000), [])

    def fill(self, order):
        self.strategy.on_order_filled(OrderResult(success=True, symbol=order.symbol,
            side=OrderSide.BUY, order_no="test-buy", quantity=order.quantity,
            price=order.price, requested_quantity=order.quantity, fill_mode="limit"))

    def test_signal_uses_only_completed_bars_and_partial_fill_keeps_stop(self):
        self.warm()
        orders = self.tick(21*60, 10010)
        self.assertEqual(len(orders), 1)
        o = orders[0]
        plan = self.strategy._pending_entry_meta[o.symbol]["plan"]
        self.assertEqual((o.price, plan.breakout_price), (10015, 10000))
        self.assertEqual(plan.expires_at, self.start.replace(hour=15, minute=20))
        self.fill(replace(o, quantity=1))
        p = self.strategy.positions[o.symbol]
        self.assertEqual(p.quantity, 1)
        self.assertEqual(p.plan.breakout_price, 10000)
        self.assertEqual(p.plan.expires_at, plan.expires_at)
        self.assertEqual(self.strategy.diagnostics["decisions"][0]["channel_high"], 10000)

    def test_missing_minutes_future_prices_and_closed_bar_revisions_are_not_filled(self):
        c = BidChannels()
        q = self.quote(0)
        c.update([q, self.quote(59, 50000)], q.timestamp)
        self.assertEqual(c.current[q.symbol].high, 10000)
        for second in range(1, 60):
            c.update([self.quote(second)], self.quote(second).timestamp)
        c.update([], self.quote(60).timestamp)
        c.update([self.quote(59, 50000)], self.quote(61).timestamp)
        self.assertEqual(c.closed[q.symbol][-1].high, 10000)
        for second in range(120, 180):
            c.update([self.quote(second)], self.quote(second).timestamp)
        c.update([], self.quote(180).timestamp)
        self.assertEqual(len(c.closed[q.symbol]), 1)
        self.assertEqual(c.reasons["missing_minute"], 1)

    def test_minute_endpoints_do_not_hide_missing_intraminute_prices(self):
        channels = BidChannels()
        for second in (0, 59):
            q = self.quote(second)
            channels.update([q], q.timestamp)
        channels.update([], self.quote(60).timestamp)
        self.assertEqual(list(channels.closed["152100"]), [])
        self.assertEqual(channels.reasons["incomplete_minute_path"], 1)

    def test_full_watchlist_includes_ordinary_shares_without_future_membership(self):
        self.strategy = ETFTrendReplay(None, MomentumScalpConfig(
            commission_rate=.000142, sell_tax_rate=.0015,
            daily_state_path=str(Path(self.tmp.name)/"full-state.json")), symbols=None)
        self.strategy.fixed_pool = ["005930"]
        for second in range(1260):
            price = 10010 if second == 1259 else 10000
            q = replace(self.quote(second, price, "005930"), security_group="ST")
            # A different observed symbol must not enter without watchlist membership.
            future = replace(q, symbol="000660")
            self.strategy.set_simulated_now(q.timestamp)
            self.assertEqual(self.strategy.on_batch_tick([q, future]), [])
        orders = self.tick(1260, 10010, "005930")
        self.assertEqual([order.symbol for order in orders], ["005930"])
        self.assertEqual(self.strategy._sell_tax_rate_for_symbol("005930"), .0015)
        self.strategy.reset_observations()
        self.assertIsNone(self.strategy.channels.symbols)

    def test_trailing_stop_only_rises_no_180_second_expiry_and_timer_end_exit(self):
        self.warm()
        self.fill(self.tick(1260, 10010)[0])
        for second in range(1261, 1621):
            self.assertEqual(self.tick(second, 10030 + ((second - 1260) // 60) * 10), [])
        p = self.strategy.positions["152100"]
        self.assertEqual(p.plan.breakout_price, 10040)
        # New highs don't lower an already raised stop; equality triggers an exit.
        self.assertEqual(self.tick(1621, 10090), [])
        self.assertEqual(p.plan.breakout_price, 10040)
        self.assertEqual(self.tick(1622, 10040)[0].requested_reason, "trailing_low")
        self.strategy.set_simulated_now(self.start.replace(hour=15, minute=20))
        self.assertEqual(self.strategy.on_batch_tick([])[0].requested_reason, "session_end")

    def test_old_stop_touch_is_not_erased_by_rebound_in_same_batch(self):
        self.warm()
        self.fill(self.tick(1260, 10010)[0])
        self.strategy.set_simulated_now(self.quote(1262).timestamp)
        orders = self.strategy.on_batch_tick([self.quote(1261, 10000), self.quote(1262, 10020)])
        self.assertEqual(orders[0].requested_reason, "trailing_low")

    def test_executable_daily_target_submits_exit_but_waits_for_confirmed_fill(self):
        self.warm()
        self.fill(self.tick(1260, 10010)[0])
        orders = self.tick(1261, 10300)
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].requested_reason, "daily_profit_target")
        self.assertFalse(self.strategy._halted)
        self.assertEqual(self.strategy.daily_pnl.realized_net_pnl, 0)
        self.strategy.on_order_filled(OrderResult(success=True, symbol="152100",
            side=OrderSide.SELL, order_no="target-sell", quantity=orders[0].quantity,
            price=10300, requested_quantity=orders[0].quantity, fill_mode="market"))
        self.tick(1262, 10300)
        self.assertEqual(self.strategy._halt_reason, "daily_profit_target")

    def test_inverse_same_sizing_costs_and_loss_room_as_regular_etf(self):
        q = self.quote(1260, 10010)
        self.warm()
        self.tick(1260, 10010)
        signal = self.strategy.channels.closed[q.symbol][-1]
        self.strategy._pending_entry_meta.clear()
        self.strategy.daily_pnl.realized_net_pnl = -4700
        a = self.strategy._plan(q, signal, 10000, q.timestamp)
        b = self.strategy._plan(replace(q, symbol="114800"), signal, 10000, q.timestamp)
        self.assertEqual((a.quantity, a.committed_risk_net_abs), (b.quantity, b.committed_risk_net_abs))
        self.assertTrue(a.allowed)
        self.assertLessEqual(a.committed_risk_net_abs, 300)

    def test_unaffordable_and_daily_breakers_preserved(self):
        self.warm()
        self.tick(1260, 10010)
        signal = self.strategy.channels.closed["152100"][-1]
        self.strategy._pending_entry_meta.clear()
        q = self.quote(1261, 2300000)
        self.assertFalse(self.strategy._plan(q, signal, 2290000, q.timestamp).allowed)
        self.strategy.daily_pnl.realized_net_pnl = 10000
        self.assertEqual(self.tick(1261, 10020), [])
        self.assertEqual(self.strategy._halt_reason, "daily_profit_target")
        self.strategy._halted = False
        self.strategy.daily_pnl.realized_net_pnl = -5000
        self.assertEqual(self.tick(1262, 10020), [])
        self.assertEqual(self.strategy._halt_reason, "daily_total_loss_limit")

    def test_random_control_is_reproducible_without_future_prices(self):
        self.warm()
        self.tick(1260, 10010)
        bar = self.strategy.channels.closed["152100"][-1]
        self.strategy.control_rate = .5
        self.strategy.control_seed = 7
        first = self.strategy._entry_signal("152100", bar, 10000)
        self.assertEqual(first, self.strategy._entry_signal("152100", bar, 999999))
        self.strategy.control_rate = 0
        self.assertFalse(self.strategy._entry_signal("152100", bar, 10000))
        self.strategy.control_rate = 1
        self.assertTrue(self.strategy._entry_signal("152100", bar, 10000))

    def test_later_comparison_freezes_control_frequency_without_evaluation_leakage(self):
        historical = [dict(date="2026-09-18", symbols={
            "114800": dict(ready_minutes=100, breakout_minutes=3)}),
            dict(date="2026-09-21", symbols={
                "114800": dict(ready_minutes=100, breakout_minutes=99)})]
        protocol = comparison_protocol(historical)
        self.assertEqual(protocol["control_rate"], .03)
        reference = dict(coverage=historical, control_rate=.03,
                         frozen_rules=dict(entry_bars=20, stop_bars=5, symbols=sorted(SYMBOLS)))
        later = [dict(date="2026-09-22", symbols={
            "114800": dict(ready_minutes=1000, breakout_minutes=999)})]
        result = comparison_protocol(later, reference)
        self.assertEqual(result, dict(control_rate=.03, comparison_dates=["2026-09-22"],
                                      diagnostic_dates=["2026-09-18"]))
        with self.assertRaises(ValueError):
            comparison_protocol(later)
        with self.assertRaises(ValueError):
            comparison_protocol(historical, reference)
        with self.assertRaises(ValueError):
            comparison_protocol(later, dict(reference, control_rate=.99))
        with self.assertRaises(ValueError):
            comparison_protocol(later, dict(reference, frozen_rules={}))
        with self.assertRaises(ValueError):
            comparison_protocol(later * 2, reference)

    def test_later_date_cli_runs_all_comparisons_without_claiming_profitability(self):
        root = Path(self.tmp.name)
        reference = root / "reference.json"
        output = root / "result.json"
        reference.write_text(json.dumps(dict(
            frozen_rules=dict(entry_bars=20, stop_bars=5, symbols=sorted(SYMBOLS)),
            coverage=[dict(date="2026-09-18", symbols={
                "114800": dict(ready_minutes=100, breakout_minutes=3)})], control_rate=.03)))
        args = ["validate_etf_trend", "--root", str(root), "--date", "2026-09-22",
                "--reference", str(reference), "--output", str(output)]
        with patch("sys.argv", args), redirect_stdout(StringIO()), \
                patch("scripts.validate_etf_trend.audit_observations",
                      return_value=dict(date="2026-09-22", symbols={})), \
                patch("scripts.validate_etf_trend.replay_day",
                      side_effect=lambda root, day, **kwargs: dict(date=day, complete=False)) as replay:
            main()
        result = json.loads(output.read_text())
        self.assertEqual(replay.call_count, 23)
        self.assertEqual({r["date"] for r in result["reports"]}, {"2026-09-22"})
        controls = [r for r in result["reports"] if r["policy"] == "random_entry_control"]
        self.assertEqual([r["seed"] for r in controls], list(range(20)))
        self.assertEqual(result["control_rate"], .03)
        self.assertEqual(result["reference_report"]["path"], str(reference))
        self.assertFalse(result["independent_holdout"])
        self.assertFalse(result["runtime_adopted"])


if __name__ == "__main__":
    unittest.main()
