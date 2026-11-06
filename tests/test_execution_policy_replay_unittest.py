import gzip
import json
import tempfile
import unittest
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

from scripts.replay_execution_policy import replay_day
from src.analytics.execution_policy import limit_fill, market_fill
from src.models import Quote


class BreakoutReplayTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.start = datetime(2026, 9, 21, 10)

    def quote(self, second, price, depth=1000):
        return Quote(symbol="005930", timestamp=self.start + timedelta(seconds=second),
            current_price=price, ask_price=price+10, bid_price=price-10,
            book_available=True, book_depth_available=depth is not None,
            ask_size=depth or 0, bid_size=depth or 0)

    def replay(self, points, *, universe=True, observed_delay=0, **options):
        path = self.root / "2026/09/quote-batches.2026-09-21.jsonl.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt") as out:
            for i, q in enumerate(points):
                observed = q.timestamp + timedelta(seconds=observed_delay)
                item = dict(observed_timestamp=observed.isoformat(), reset_history=i == 0,
                            quotes=[dict(asdict(q), timestamp=q.timestamp.isoformat())])
                if universe:
                    item["universe"] = [q.symbol]
                out.write(json.dumps(item) + "\n")
        original = path.read_bytes()
        result = replay_day(self.root, "2026-09-21", **options)
        self.assertEqual(path.read_bytes(), original)
        return result

    def history(self):
        return [self.quote(i, 10000 + (i // 4 - 14) * 20) for i in range(60)]

    def test_limit_and_market_use_observed_depth_not_current_price(self):
        q = self.quote(61, 10030, depth=2)
        self.assertEqual(limit_fill(q, 10040, 10), (2, 10040))
        self.assertEqual(limit_fill(q, 10000, 10), (0, 0))
        self.assertEqual(market_fill(q, 10), (2, 10020))
        q.ask_price = 0
        self.assertIsNone(market_fill(q, 10))
        q.ask_price = 10040
        q.book_depth_available = False
        self.assertIsNone(limit_fill(q, 10040, 10))
        self.assertIsNone(market_fill(q, 10))

    def test_partial_buy_stop_and_confirmed_net_fill_ledger(self):
        result = self.replay(self.history() + [self.quote(60, 10030, 3), self.quote(61, 10030, 1),
                            self.quote(62, 9710), self.quote(63, 9710)])
        self.assertTrue(result["complete"])
        self.assertTrue(result["universe_verified"])
        self.assertEqual(len(result["orders"]), 2)
        self.assertGreater(result["orders"][0]["requested"], 3)
        self.assertEqual(result["orders"][0]["filled"], 1)
        self.assertEqual(result["orders"][1]["requested"], 1)
        self.assertEqual(result["orders"][1]["reason"], "breakout_failed")
        self.assertEqual(result["sell_fills"][0]["buy_price"], 10040)
        self.assertEqual(result["sell_fills"][0]["sell_price"], 9700)
        self.assertEqual(result["sell_fills"][0]["stop_price"], 10010)
        self.assertEqual(result["net_total"], sum(r["net_pnl"] for r in result["sell_fills"]))
        self.assertLess(result["net_total"], 0)

    def test_expiry_waits_for_next_executable_quote_and_sells_remainder(self):
        points = self.history() + [self.quote(i, 10030 if i < 230 else 10100, 3) for i in range(60, 241)]
        points += [self.quote(241, 10100, 1), self.quote(242, 10100, 2)]
        result = self.replay(points)
        self.assertTrue(result["complete"])
        self.assertEqual(result["orders"][1]["reason"], "holding_expiry")
        self.assertEqual(result["orders"][1]["filled"], 3)
        self.assertEqual([r["quantity"] for r in result["sell_fills"]], [1, 2])
        self.assertGreater(result["net_total"], 0)

    def test_first_unfilled_attempt_cannot_fill_a_later_dip(self):
        result = self.replay(self.history() + [self.quote(60, 10030), self.quote(61, 10050),
                            self.quote(62, 10000), self.quote(63, 10000)])
        self.assertTrue(result["complete"])
        self.assertEqual(len(result["orders"]), 1)
        self.assertEqual(result["orders"][0]["status"], "rejected")
        self.assertIn("지정가", result["orders"][0]["rejection_reason"])
        self.assertEqual(result["net_total"], 0)

    def test_execution_reuses_live_entry_check_before_assuming_a_limit_fill(self):
        reversal = self.quote(61, 10020)
        hidden_peak = self.quote(61, 10030)
        hidden_peak.trade_high_price = 10040
        empty_bid = self.quote(61, 10030)
        empty_bid.bid_size = 0
        for quote in (reversal, hidden_peak, empty_bid):
            with self.subTest(quote=quote):
                result = self.replay(self.history() + [self.quote(60, 10030), quote,
                                    self.quote(62, 10000), self.quote(63, 10000)])
                self.assertTrue(result["complete"])
                self.assertEqual(len(result["orders"]), 1)
                order = result["orders"][0]
                self.assertEqual(order["status"], "rejected")
                self.assertTrue(order["rejection_reason"])
                self.assertEqual(order["filled"], 0)
                self.assertEqual(result["sell_fills"], [])
                self.assertEqual(result["net_total"], 0)

    def test_entry_deadline_uses_received_time_not_exchange_timestamp(self):
        result = self.replay(self.history() + [self.quote(60, 10030), self.quote(64, 10030),
                            self.quote(65, 10000)], observed_delay=1)
        self.assertTrue(result["complete"])
        self.assertEqual(len(result["orders"]), 1)
        self.assertEqual(result["orders"][0]["status"], "expired")
        self.assertEqual(result["orders"][0]["filled"], 0)
        self.assertEqual(result["net_total"], 0)

    def test_paper_profit_exit_uses_subsequent_fills_not_trigger_quote(self):
        result = self.replay(self.history() + [self.quote(60, 10030, 3), self.quote(61, 10030, 3),
                            self.quote(62, 10160, 3), self.quote(63, 10120, 1),
                            self.quote(64, 10120, 2)], require_fresh_hold=True)
        self.assertTrue(result["complete"])
        self.assertEqual(len(result["orders"]), 2)
        sell = result["orders"][1]
        self.assertEqual(sell["reason"], "planned_profit")
        self.assertEqual(sell["timestamp"], (self.start + timedelta(seconds=62)).isoformat())
        self.assertEqual(sell["filled"], 3)
        self.assertEqual([(f["quantity"], f["sell_price"]) for f in result["sell_fills"]],
                         [(1, 10110), (2, 10110)])
        self.assertGreater(result["net_total"], 0)

    def test_unknown_execution_stops_portfolio_instead_of_inventing_no_fill(self):
        for next_quote in (self.quote(61, 10030, None), self.quote(67, 10030)):
            with self.subTest(timestamp=next_quote.timestamp):
                result = self.replay(self.history() + [self.quote(60, 10030), next_quote, self.quote(68, 11000)])
                self.assertFalse(result["complete"])
                self.assertIsNone(result["net_total"])
                self.assertIsNotNone(result["unknown_execution"])
                self.assertEqual(result["unresolved_orders"], 1)
                self.assertEqual(len(result["orders"]), 1)

    def test_loss_limit_still_liquidates_after_a_price_gap(self):
        result = self.replay(self.history() + [self.quote(60, 10030), self.quote(61, 10030),
                            self.quote(62, 9000), self.quote(63, 9000)], universe=False)
        self.assertTrue(result["complete"])
        self.assertFalse(result["universe_verified"])
        self.assertEqual(result["halt_reason"], "daily_total_loss_limit")
        self.assertEqual(result["orders"][1]["reason"], "daily_total_loss_limit")
        self.assertLess(result["net_total"], -5000)

    def test_strict_hold_audit_does_not_invent_an_exit_through_missing_book_data(self):
        missing = self.quote(64, 10030)
        missing.book_available = False
        result = self.replay(self.history() + [self.quote(60, 10030), self.quote(61, 10030),
                            missing, self.quote(67, 11000)], require_fresh_hold=True)
        self.assertFalse(result["complete"])
        self.assertIsNone(result["net_total"])
        self.assertIn(result["unknown_execution"]["reason"], {"held_quote_gap", "held_quote_stale"})
        self.assertEqual(result["sell_fills"], [])


if __name__ == "__main__":
    unittest.main()
