import gzip
import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta
from math import ceil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.models import OrderResult, OrderSide, OrderType, Position, Quote
from src.strategies.momentum_scalp import MAX_HOLD_SECONDS, MomentumScalpStrategy
from src.strategies.momentum_scalp_types import MomentumScalpConfig, PositionState, TradePlan


def make_quote(symbol, price, timestamp, *, ask=None, bid=None):
    return Quote(symbol=symbol, current_price=price, timestamp=timestamp,
                 ask_price=price + 10 if ask is None else ask,
                 bid_price=price - 10 if bid is None else bid,
                 book_available=True, book_depth_available=True, ask_size=1000, bid_size=1000)


def make_plan(symbol, now, *, quantity=10, entry=10010, breakout=9950, risk=800, expires_delta=180,
              target=None):
    if target is None:
        target = ceil((entry * 1.000142 + risk / max(1, quantity)) / (1 - .000142 - .0015))
    return TradePlan(symbol, True, "price_breakout", quantity, entry, breakout, risk,
                     now, now + timedelta(seconds=expires_delta), target)


class BreakoutStrategyContractTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.now = datetime(2026, 9, 21, 10)

    def strategy(self, **overrides):
        options = dict(daily_state_path=str(Path(self.directory.name) / "state.json"),
                       quote_archive_root=str(Path(self.directory.name) / "quotes"),
                       commission_rate=.000142, sell_tax_rate=.0015)
        options.update(overrides)
        strategy = MomentumScalpStrategy(None, MomentumScalpConfig(**options))
        strategy.set_simulated_now(self.now)
        strategy._ensure_active_day()
        return strategy

    def warm(self, strategy, symbol="005930", price=10000, *, movement=None):
        movement = max(200, price // 1000 * 10) if movement is None else movement
        quotes = [make_quote(symbol, price + (s // 6 + 1) * movement // 10,
                             self.now + timedelta(seconds=s)) for s in range(-60, 0)]
        strategy._ingest_quotes(quotes)
        return make_quote(symbol, price + 30, self.now)

    def test_alphanumeric_discovery_survives_watchlist_restore_and_same_entry_rules(self):
        options = dict(static_watchlist=["005930"], inverse_etfs=["114800"], dynamic_pool_size=2)
        strategy = self.strategy(**options)
        strategy.update_runtime_pool(["005930", "18064K", "18064K", "Q500001", "020400", "199430"])
        self.assertEqual(strategy._pool, ["18064K", "020400"])
        self.assertEqual(strategy.get_watchlist(), ["18064K", "020400", "005930", "114800"])
        restored = self.strategy(**options)
        restored._load_daily_state()
        self.assertEqual(restored._pool, strategy._pool)
        ordinary = restored._build_trade_plan(self.warm(restored, "020400"), now=self.now)
        preferred = restored._build_trade_plan(self.warm(restored, "18064K"), now=self.now)
        self.assertTrue(preferred.allowed, preferred.reason)
        self.assertEqual(replace(preferred, symbol="020400"), ordinary)
        locked = replace(self.warm(restored, "18064K"), ask_price=0, ask_size=0)
        self.assertFalse(restored._build_trade_plan(locked, now=self.now).allowed)

    def test_breakout_uses_past_window_not_current_or_future_quote(self):
        s = self.strategy()
        q = self.warm(s)
        future = make_quote(q.symbol, 20000, self.now + timedelta(seconds=1))
        s._ingest_quotes([q, future])
        plan = s._build_trade_plan(q, now=self.now)
        self.assertTrue(plan.allowed)
        self.assertEqual(plan.breakout_price, 10010)
        self.assertEqual(plan.expires_at, self.now + timedelta(seconds=MAX_HOLD_SECONDS))
        flat = replace(q, bid_price=10010)
        self.assertEqual(s._build_trade_plan(flat, now=self.now).reason, "no_breakout")
        self.assertEqual(s._deserialize_plan(s._serialize_plan(plan)), plan)

    def test_delayed_fresh_quotes_keep_the_lookback_anchor_without_filling_gaps(self):
        for delay in (0, 1, 3, 5):
            for missing_interval in (False, True):
                with self.subTest(delay=delay, missing_interval=missing_interval):
                    s = self.strategy()
                    history = [make_quote("005930", 10000 + (second // 6 + 1) * 40,
                                          self.now + timedelta(seconds=second))
                               for second in range(-64, 0, 5)
                               if not (missing_interval and second == -34)]
                    signal = make_quote("005930", 10030, self.now)
                    for quote in [*history, signal]:
                        s.set_simulated_now(quote.timestamp + timedelta(seconds=delay))
                        s._ingest_quotes([quote])
                    observed_at = self.now + timedelta(seconds=delay)
                    plan = s._build_trade_plan(signal, now=observed_at)
                    if missing_interval:
                        self.assertEqual(plan.reason, "price_window_incomplete")
                    else:
                        self.assertTrue(plan.allowed, plan.reason)
                        self.assertEqual(s._breakout_window(signal)[0].timestamp,
                                         self.now - timedelta(seconds=64))
                        self.assertEqual(plan.expires_at, self.now + timedelta(seconds=180))
                    self.assertEqual(s._build_trade_plan(signal,
                        now=self.now + timedelta(seconds=5.1)).reason, "stale_quote")

    def test_quote_repricing_or_return_to_an_old_trade_high_is_not_a_new_breakout(self):
        for peak, last in ((10000, 10000), (10050, 10050), (10080, 10060)):
            with self.subTest(peak=peak, last=last):
                s = self.strategy()
                q = self.warm(s, movement=400)
                s._ingest_quotes([make_quote(q.symbol, peak, self.now-timedelta(seconds=1),
                                            bid=9990, ask=10010)])
                signal = replace(q, current_price=last, bid_price=10060, ask_price=10070)
                self.assertFalse(s._build_trade_plan(signal, now=self.now).allowed)
                real_breakout = replace(signal, current_price=peak+40, bid_price=peak+30,
                                        ask_price=peak+50)
                self.assertTrue(s._build_trade_plan(real_breakout, now=self.now).allowed)

    def test_same_second_trade_peak_survives_a_lower_print_and_book(self):
        for delivery in ("split", "batch", "compacted"):
            with self.subTest(delivery=delivery):
                s = self.strategy()
                q = self.warm(s)
                peak = make_quote(q.symbol, 10050, self.now-timedelta(seconds=1),
                                  bid=9990, ask=10010)
                last = replace(peak, current_price=10000)
                if delivery == "split":
                    s._ingest_quotes([peak])
                    s._ingest_quotes([last])
                elif delivery == "batch":
                    s._ingest_quotes([peak, last])
                else:
                    s._ingest_quotes([replace(last, trade_high_price=10050)])
                signal = replace(q, current_price=10040, bid_price=10030, ask_price=10040)
                self.assertEqual(s._build_trade_plan(signal, now=self.now).breakout_price, 10050)
                self.assertFalse(s._build_trade_plan(signal, now=self.now).allowed)
                self.assertEqual(s._quotes_cache[q.symbol].current_price, 10000)

    def test_current_second_reversal_is_not_selected_as_a_new_high(self):
        for delivery in ("split", "batch", "compacted"):
            with self.subTest(delivery=delivery):
                s = self.strategy(static_watchlist=["001210"], inverse_etfs=[])
                self.warm(s, "001210", price=11790)
                peak = make_quote("001210", 11840, self.now, bid=11830, ask=11840)
                last = replace(peak, current_price=11830)
                if delivery == "split":
                    s._ingest_quotes([peak])
                    quotes = [last]
                elif delivery == "batch":
                    quotes = [peak, last]
                else:
                    quotes = [replace(last, trade_high_price=11840)]
                with self.assertLogs("kis_trader.strategy.momentum_scalp", level="INFO") as captured:
                    self.assertEqual(s.on_batch_tick(quotes), [])
                self.assertIn("돌파=0 계획=0", next(line for line in captured.output if "돌파 배치:" in line))
                self.assertEqual(s._pending_entry_meta, {})
                self.assertEqual(s._quotes_cache[last.symbol].current_price, 11830)
                self.assertEqual(s._quotes_cache[last.symbol].trade_high_price, 11840)
                self.assertEqual(s._build_trade_plan(s._quotes_cache[last.symbol], now=self.now).reason,
                                 "no_breakout")

                s.set_simulated_now(self.now + timedelta(seconds=1))
                higher = replace(last, timestamp=s._now(), current_price=11850,
                                 bid_price=11850, ask_price=11860, trade_high_price=11850)
                self.assertEqual(len(s.on_batch_tick([higher])), 1)

        s = self.strategy(static_watchlist=["373220"], inverse_etfs=[])
        self.warm(s, "373220", price=402990, movement=6000)
        advancing = make_quote("373220", 404000, self.now, bid=403500, ask=409500)
        advancing.trade_high_price = 404000
        plan = s._build_trade_plan(advancing, now=self.now)
        self.assertTrue(advancing.is_price_breakout(plan.breakout_price))
        # A valid high is not permission to buy one share with more than 5,000 won of risk.
        self.assertEqual(plan.reason, "insufficient_capital_or_loss_room")
        self.assertEqual(s.on_batch_tick([advancing]), [])

    def test_fractional_rest_timestamp_keeps_same_second_peak_without_repricing_book(self):
        for earlier_delivered_first in (False, True):
            with self.subTest(earlier_delivered_first=earlier_delivered_first):
                s = self.strategy(static_watchlist=["001210"], inverse_etfs=[])
                self.warm(s, "001210", price=11790)
                peak = make_quote("001210", 11840, self.now, bid=11830, ask=11840)
                latest = make_quote("001210", 11830, self.now + timedelta(microseconds=500000),
                                    bid=11820, ask=11830)
                s.set_simulated_now(latest.timestamp)
                for q in ([peak, latest] if earlier_delivered_first else [latest, peak]):
                    s._ingest_quotes([q])
                cached = s._quotes_cache[latest.symbol]
                self.assertEqual((cached.timestamp, cached.current_price, cached.bid_price,
                                  cached.ask_price), (latest.timestamp, 11830, 11820, 11830))
                self.assertEqual(cached.trade_high_price, 11840)
                self.assertEqual(s._build_trade_plan(cached, now=s._now()).reason, "no_breakout")
                self.assertEqual(s.on_batch_tick([]), [])

                s.set_simulated_now(self.now + timedelta(seconds=1))
                next_quote = make_quote("001210", 11850, s._now(), bid=11850, ask=11860)
                self.assertEqual(len(s.on_batch_tick([next_quote])), 1)

    def test_missing_window_book_stale_and_unaffordable_inputs_cannot_enter(self):
        s = self.strategy()
        q = make_quote("005930", 10030, self.now)
        self.assertEqual(s._build_trade_plan(q, now=self.now).reason, "price_window_incomplete")
        q = self.warm(s)
        self.assertEqual(s._build_trade_plan(replace(q, book_available=False), now=self.now).reason,
                         "no_executable_book")
        self.assertEqual(s._build_trade_plan(replace(q, ask_price=0), now=self.now).reason,
                         "no_executable_book")
        for empty in (replace(q, ask_size=0), replace(q, bid_size=0)):
            self.assertEqual(s._build_trade_plan(empty, now=self.now).reason, "no_executable_book")
        self.assertEqual(s._build_trade_plan(q, now=self.now + timedelta(seconds=6)).reason, "stale_quote")
        high = self.warm(s, "000660", 2300000)
        self.assertEqual(s._build_trade_plan(high, now=self.now).reason, "unaffordable_one_share")

    def test_projected_target_excludes_current_spike_and_requires_net_room_after_costs(self):
        s = self.strategy()
        symbol = "005930"
        history = [make_quote(symbol, 10000 + (second // 10 + 1) * 10,
                              self.now + timedelta(seconds=second)) for second in range(-60, 0)]
        s._ingest_quotes(history)
        signal = make_quote(symbol, 10030, self.now)
        plan = s._build_trade_plan(signal, now=self.now)
        # Average bid 9,965 projects 10,015, rounded to 10,010: no net room.
        self.assertEqual((plan.target_price, plan.reason), (10010, "target_not_profitable"))
        s._ingest_quotes([replace(signal, current_price=10500, bid_price=10490, ask_price=10510)])
        s._recent_window(symbol).append(replace(signal, timestamp=self.now+timedelta(seconds=1),
                                               bid_price=9000, ask_price=11000))
        chased = s._build_trade_plan(s._quotes_cache[symbol], now=self.now)
        self.assertEqual((chased.target_price, chased.reason), (10010, "target_not_profitable"))

    def test_same_projected_bid_applies_symbol_costs_without_inverse_routing(self):
        s = self.strategy()
        for symbol, group in (("005930", "ST"), ("114800", "EF")):
            s.instrument_groups[symbol] = group
            s._ingest_quotes([make_quote(symbol, 10000 + (second // 10 + 1) * 32,
                                        self.now + timedelta(seconds=second)) for second in range(-60, 0)])
        stock, fund = [s._build_trade_plan(make_quote(symbol, 10030, self.now), now=self.now)
                       for symbol in ("005930", "114800")]
        self.assertEqual(stock.target_price, fund.target_price)
        self.assertEqual(stock.reason, "target_not_profitable")
        self.assertTrue(fund.allowed)
        self.assertEqual(stock.exit_price_allowance, fund.exit_price_allowance)

    def test_target_proceeds_include_the_same_observed_execution_allowance_as_risk(self):
        plans = []
        for adverse_step in (False, True):
            s = self.strategy()
            history = [make_quote("005930", 9800 if second < -30 else 10000,
                                  self.now + timedelta(seconds=second)) for second in range(-60, 0)]
            if adverse_step:
                history[20] = replace(history[20], bid_price=9730)
            s._ingest_quotes(history)
            q = make_quote("005930", 10030, self.now)
            with patch.object(s, "_exit_price_allowance", wraps=s._exit_price_allowance) as allowance:
                plan = s._build_trade_plan(q, now=self.now)
                allowance.assert_called_once()
            self.assertGreater(s._trade_pnl(q.symbol, plan.entry_limit_price, plan.target_price,
                                           plan.quantity).net_pnl, 0)
            self.assertEqual(plan.exit_price_allowance, 80 if adverse_step else 20)
            self.assertEqual(plan.allowed, not adverse_step)
            self.assertEqual(s._target_net_pnl(plan) > 0, plan.allowed)
            risk = s._risk_per_share(q, plan.entry_limit_price, plan.stop_price)
            self.assertEqual(plan.committed_risk_net_abs, ceil(risk * plan.quantity))
            self.assertLessEqual(plan.committed_risk_net_abs, 5000)
            restored = s._deserialize_plan(s._serialize_plan(plan))
            self.assertEqual(restored, plan)
            filled = s._reprice_plan_for_fill(plan=plan, fill_price=q.ask_price-10, quantity=1)
            self.assertEqual((filled.target_price, filled.stop_price, filled.exit_price_allowance),
                             (plan.target_price, plan.stop_price, plan.exit_price_allowance))
            plans.append(plan)
        # The cost check changes, not the price forecast or a fixed won-profit threshold.
        self.assertEqual(plans[0].target_price, plans[1].target_price)

    def test_target_uses_elapsed_time_and_does_not_project_an_isolated_low_as_upside(self):
        targets = []
        for interval in (1, 5):
            s = self.strategy()
            history = [make_quote("005930", 9800 if second < -30 else 10000,
                                  self.now + timedelta(seconds=second),
                                  bid=9800 if second < -30 else 10000,
                                  ask=9810 if second < -30 else 10010)
                       for second in range(-60, 0, interval)]
            s._ingest_quotes(history)
            signal = make_quote("005930", 10030, self.now)
            plan = s._build_trade_plan(signal, now=self.now)
            self.assertTrue(plan.allowed)
            targets.append(plan.target_price)
            if interval == 1:
                s._recent_quotes.clear()
                s._quotes_cache.clear()
                history[15] = replace(history[15], current_price=9400, bid_price=9400, ask_price=9410)
                s._ingest_quotes(history)
                self.assertEqual(s._build_trade_plan(signal, now=self.now).target_price, 10100)
        self.assertEqual(targets, [10100, 10100])

        s = self.strategy()
        s._ingest_quotes([make_quote("005930", 10000 if second < -30 else 9800,
                                     self.now+timedelta(seconds=second),
                                     bid=10000 if second < -30 else 9800,
                                     ask=10010 if second < -30 else 9810)
                          for second in range(-60, 0)])
        self.assertEqual(s._build_trade_plan(signal, now=self.now).reason, "target_not_profitable")

    def test_recent_wide_spread_is_not_erased_by_a_tight_signal_book(self):
        plans = []
        for wide in (False, True):
            s = self.strategy()
            history = [make_quote("005930", 9800 if second < -30 else 10000,
                                  self.now + timedelta(seconds=second),
                                  bid=9800 if second < -30 else 10000,
                                  ask=9810 if second < -30 else 10010)
                       for second in range(-60, 0)]
            if wide:
                history[20] = replace(history[20], ask_price=9900)
            s._ingest_quotes(history)
            q = make_quote("005930", 10030, self.now)
            p = s._build_trade_plan(q, now=self.now)
            plans.append(p)
            self.assertEqual(p.exit_price_allowance, 100 if wide else 20)
            self.assertEqual(p.allowed, not wide)
            self.assertEqual(s._target_net_pnl(p) > 0, p.allowed)
            self.assertEqual(p.committed_risk_net_abs, ceil(p.quantity * s._risk_per_share(
                q, p.entry_limit_price, p.stop_price)))
            self.assertLessEqual(p.committed_risk_net_abs, 5000)
            self.assertEqual(s._deserialize_plan(s._serialize_plan(p)), p)
            # An old anchor, a future quote or a missing book is not a recent spread.
            s._recent_window(q.symbol).extend([
                replace(q, timestamp=self.now-timedelta(seconds=61), ask_price=20000),
                replace(q, timestamp=self.now+timedelta(seconds=1), ask_price=20000),
                replace(q, timestamp=self.now-timedelta(seconds=2), ask_price=20000,
                        book_available=False),
            ])
            self.assertEqual(s._exit_price_allowance(q, p.stop_price), p.exit_price_allowance)
        self.assertEqual(plans[0].target_price, plans[1].target_price)
        self.assertEqual(plans[0].breakout_price, plans[1].breakout_price)
        self.assertEqual(plans[0].stop_price, plans[1].stop_price)
        self.assertEqual(plans[1].reason, "target_not_profitable")

    def test_same_second_update_keeps_observed_high_without_changing_latest_book(self):
        for delivery in ("split", "batch", "compacted_stream"):
            with self.subTest(delivery=delivery):
                s = self.strategy()
                history = [make_quote("092870", 48200, self.now + timedelta(seconds=t),
                                      bid=48200, ask=48250) for t in range(-60, 0)]
                peak_at = self.now - timedelta(seconds=40)
                peak = make_quote("092870", 48200, peak_at, bid=48200, ask=48300)
                later = replace(peak, bid_price=48150, ask_price=48200)
                missing = replace(later, ask_price=0)
                before = [q for q in history if q.timestamp < peak_at]
                after = [q for q in history if q.timestamp > peak_at]
                observations = [*before, peak, later, missing, *after]
                if delivery == "compacted_stream":
                    s._ingest_quotes([*before, replace(missing, ask_high_price=48300), *after])
                elif delivery == "batch":
                    s._ingest_quotes(observations)
                else:
                    for q in observations:
                        s._ingest_quotes([q])
                        cached = s._quotes_cache[q.symbol]
                        self.assertEqual((cached.timestamp, cached.current_price, cached.bid_price, cached.ask_price),
                                         (q.timestamp, q.current_price, q.bid_price, q.ask_price))
                signal = make_quote("092870", 48300, self.now, bid=48300, ask=48350)
                s._ingest_quotes([signal])
                plan = s._build_trade_plan(signal, now=self.now)
                self.assertEqual(plan.breakout_price, 48300)
                self.assertEqual(plan.reason, "no_breakout")
                self.assertFalse(plan.allowed)
                self.assertEqual(s._quotes_cache[signal.symbol], signal)
                self.assertEqual(s._current_exit_price(signal), 48300)
                self.assertEqual(len(s._recent_window(signal.symbol)), 61)

                genuinely_higher = replace(signal, current_price=48350, bid_price=48350,
                                           ask_price=48400)
                self.assertEqual(s._build_trade_plan(genuinely_higher, now=self.now).reason,
                                 "target_not_profitable")
                s.set_simulated_now(self.now + timedelta(seconds=61))
                s._ingest_quotes([make_quote(signal.symbol, 48200, self.now + timedelta(seconds=t),
                                             bid=48200, ask=48250) for t in range(1, 61)])
                expired_peak = s._build_trade_plan(replace(signal, timestamp=s._now()), now=s._now())
                self.assertEqual(expired_peak.reason, "target_not_profitable")
                self.assertEqual(expired_peak.breakout_price, 48250)

    def test_breakout_window_requires_valid_books_across_the_whole_interval(self):
        for missing_book in (False, True):
            with self.subTest(missing_book=missing_book):
                s = self.strategy()
                history = []
                for second in range(-60, 0):
                    q = make_quote("005930", 10000, self.now + timedelta(seconds=second))
                    if -45 <= second <= -39:
                        if not missing_book:
                            continue
                        q.book_available = False
                    history.append(q)
                s._ingest_quotes(history)
                signal = make_quote("005930", 10030, self.now)
                self.assertEqual(s._build_trade_plan(signal, now=self.now).reason,
                                 "price_window_incomplete")
                s.positions[signal.symbol] = PositionState(signal.symbol, 10030, 1,
                    self.now, make_plan(signal.symbol, self.now, breakout=10010))
                self.assertEqual(s.on_batch_tick([replace(signal, bid_price=10010)])[0].requested_reason,
                                 "breakout_failed")

                s.set_simulated_now(self.now + timedelta(seconds=61))
                s._ingest_quotes([make_quote(signal.symbol, 10000, self.now + timedelta(seconds=t))
                                  for t in range(1, 61)])
                refreshed = replace(signal, timestamp=s._now())
                self.assertEqual(s._build_trade_plan(refreshed, now=s._now()).reason,
                                 "target_not_profitable")

        s = self.strategy()
        s._ingest_quotes([make_quote("005930", 10000, self.now + timedelta(seconds=t))
                          for t in range(-60, 0, 5)])
        signal = make_quote("005930", 10030, self.now)
        self.assertEqual(s._build_trade_plan(signal, now=self.now).reason, "target_not_profitable")

    def test_batch_reports_observed_breakout_separately_from_executable_quantity(self):
        s = self.strategy()
        signal = self.warm(s)
        s.daily_pnl.realized_net_pnl = -4987
        with self.assertLogs("kis_trader.strategy.momentum_scalp", level="INFO") as captured:
            self.assertEqual(s.on_batch_tick([signal]), [])
        summary = next(line for line in captured.output if "돌파 배치:" in line)
        self.assertIn("돌파=1 계획=0", summary)
        self.assertIn("손실여유=13원", summary)

    def test_quantity_uses_capital_and_loss_room_not_instantaneous_top_depth(self):
        for capital, realized in ((1000000, 0), (1000000, -4700), (30000, 0)):
            with self.subTest(capital=capital, realized=realized):
                s = self.strategy(seed_money=capital)
                q = self.warm(s)
                s.daily_pnl.realized_net_pnl = realized
                plan = s._build_trade_plan(q, now=self.now)
                risk = s._risk_per_share(q, q.ask_price, plan.stop_price)
                unit_cost = plan.entry_limit_price * (1 + s.config.commission_rate)
                self.assertTrue(plan.allowed)
                self.assertGreater(plan.quantity, 0)
                self.assertEqual(plan.quantity, min(int(capital // unit_cost), int((5000 + realized) // risk)))
                self.assertLessEqual(plan.committed_risk_net_abs, 5000 + realized)
                self.assertLessEqual(plan.quantity * unit_cost, capital)
                self.assertGreater(risk, q.ask_price - plan.stop_price)
                for thin in (replace(q, ask_size=1), replace(q, bid_size=1),
                             replace(q, ask_size=2, bid_size=2)):
                    self.assertEqual(s._build_trade_plan(thin, now=self.now), plan)
                s.daily_pnl.realized_net_pnl = -5000
                self.assertFalse(s._build_trade_plan(q, now=self.now).allowed)

    def test_risk_uses_observed_bid_declines_without_future_or_missing_interval_prices(self):
        s = self.strategy()
        q = make_quote("001210", 11830, self.now, ask=11840, bid=11830)
        history = [make_quote(q.symbol, 11780, self.now + timedelta(seconds=t),
                              ask=11780, bid=11770) for t in range(-60, 0)]
        history[:20] = [replace(q, current_price=11380, ask_price=11380, bid_price=11370)
                        for q in history[:20]]
        history[20] = replace(history[20], current_price=11800, ask_price=11800, bid_price=11800)
        history[21] = replace(history[21], current_price=11740, ask_price=11740, bid_price=11730)
        s._ingest_quotes(history)
        s.daily_pnl.realized_net_pnl = -2655
        plan = s._build_trade_plan(q, now=self.now)
        self.assertFalse(plan.allowed)
        self.assertEqual(plan.reason, "target_not_profitable")
        self.assertGreater(s._trade_pnl(q.symbol, plan.entry_limit_price, plan.target_price,
                                       plan.quantity).net_pnl, 0)
        self.assertLessEqual(s._target_net_pnl(plan), 0)
        self.assertEqual(plan.breakout_price, 11800)
        self.assertEqual(plan.stop_price, 11830 - (11800 - 11730))
        self.assertEqual(plan.quantity, 13)
        self.assertEqual(plan.committed_risk_net_abs, 2222)
        original_risk = s._risk_per_share(q, q.ask_price, plan.stop_price)
        for timestamp in (self.now - timedelta(seconds=66), self.now + timedelta(seconds=1)):
            s._recent_window(q.symbol).append(replace(q, timestamp=timestamp, bid_price=1000))
        self.assertEqual(s._risk_per_share(q, q.ask_price, plan.stop_price), original_risk)

        s._recent_quotes.clear()
        s._quotes_cache.clear()
        s._ingest_quotes([replace(q, timestamp=self.now - timedelta(seconds=10), bid_price=13000,
                                 ask_price=13010, current_price=13010),
                          replace(q, timestamp=self.now - timedelta(seconds=1))])
        self.assertLess(s._risk_per_share(q, q.ask_price, plan.stop_price), original_risk)

    def test_risk_reserves_open_pending_and_unconfirmed_sells(self):
        s = self.strategy()
        held = make_plan("005930", self.now, risk=500)
        waiting = make_plan("000660", self.now, risk=700)
        selling = make_plan("035420", self.now, risk=300)
        s.positions[held.symbol] = PositionState(held.symbol, 10010, 10, plan=held)
        s._pending_entry_meta[waiting.symbol] = {"plan": waiting}
        s._pending_sell_fills[selling.symbol] = {"position": PositionState(selling.symbol, 10010, 10, plan=selling)}
        self.assertEqual(s._remaining_daily_loss_room(), 3500)
        s.positions[held.symbol].plan = replace(held, breakout_price=0)
        self.assertEqual(s._remaining_daily_loss_room(), 0)

    def test_inverse_has_identical_signal_and_risk_formula_given_same_costs(self):
        s = self.strategy(sell_tax_rate=0)
        regular = self.warm(s, "005930")
        inverse = self.warm(s, "114800")
        a, b = (s._build_trade_plan(q, now=self.now) for q in (regular, inverse))
        self.assertEqual((a.allowed, a.quantity, a.breakout_price, a.committed_risk_net_abs),
                         (b.allowed, b.quantity, b.breakout_price, b.committed_risk_net_abs))

    def test_batch_evaluates_once_selects_net_target_and_deduplicates_snapshot(self):
        s = self.strategy()
        a = self.warm(s, "005930")
        b = self.warm(s, "000660")
        a = replace(a, bid_price=10050, ask_price=10060, current_price=10060)
        with patch.object(s, "_build_trade_plan", wraps=s._build_trade_plan) as evaluate:
            orders = s.on_batch_tick([a, b])
        self.assertEqual(evaluate.call_count, 2)
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].symbol, b.symbol)
        self.assertEqual(orders[0].requested_reason, "price_breakout")
        self.assertEqual(orders[0].breakout_price, s._pending_entry_meta[b.symbol]["plan"].breakout_price)
        self.assertEqual(orders[0].submit_before,
                         b.timestamp.timestamp() + s.config.quote_freshness_seconds)
        s.on_order_filled(OrderResult(success=False, symbol=b.symbol, side=OrderSide.BUY,
                                     error_code="LOCAL_ORDER_EXPIRED", error_category="stale_order"))
        self.assertNotIn(b.symbol, s._pending_entry_meta)
        self.assertEqual(s._pending_entry_risk(), 0)
        s.fixed_pool = [b.symbol]
        s.config.static_watchlist = [b.symbol]
        self.assertEqual(s.on_batch_tick([b]), [])
        self.assertFalse(hasattr(s, "_prediction_model"))

    def test_api_execution_cooldown_does_not_skip_signal_evaluation(self):
        s = self.strategy()
        q = self.warm(s)
        with patch.object(s, "_api_cooldown_remaining", return_value=10), patch.object(s, "_build_trade_plan", wraps=s._build_trade_plan) as evaluate:
            self.assertEqual(s.on_batch_tick([q]), [])
            evaluate.assert_called_once()
        self.assertEqual(s._last_evaluation_count, 1)

    def test_candidate_ranking_uses_execution_adjusted_not_ideal_target_proceeds(self):
        s = self.strategy()
        a, b = [make_quote(symbol, 10030, self.now) for symbol in ("005930", "000660")]
        plans = {
            a.symbol: replace(make_plan(a.symbol, self.now, target=10200), exit_price_allowance=150),
            b.symbol: replace(make_plan(b.symbol, self.now, target=10150), exit_price_allowance=10),
        }
        raw = lambda p: s._trade_pnl(p.symbol, p.entry_limit_price, p.target_price, p.quantity).net_pnl
        self.assertGreater(raw(plans[a.symbol]), raw(plans[b.symbol]))
        self.assertLess(s._target_net_pnl(plans[a.symbol]), s._target_net_pnl(plans[b.symbol]))
        with patch.object(s, "_build_trade_plan", side_effect=lambda q, now: plans[q.symbol]):
            self.assertEqual(s.on_batch_tick([a, b])[0].symbol, b.symbol)

        payload = s._serialize_plan(plans[a.symbol])
        payload.pop("exit_price_allowance")
        legacy = s._deserialize_plan(payload)
        self.assertEqual(legacy.exit_price_allowance, 0)
        self.assertEqual(legacy.target_price, plans[a.symbol].target_price)

    def test_stop_precedes_expiry_and_does_not_move_after_fill(self):
        s = self.strategy()
        q = self.warm(s)
        plan = s._build_trade_plan(q, now=self.now)
        repriced = s._reprice_plan_for_fill(plan=plan, fill_price=q.ask_price-10, quantity=2)
        self.assertEqual(repriced.breakout_price, plan.breakout_price)
        self.assertEqual(repriced.stop_price, plan.stop_price)
        self.assertEqual(repriced.target_price, plan.target_price)
        self.assertEqual(repriced.expires_at, plan.expires_at)
        self.assertEqual(repriced.quantity, 2)
        p = PositionState(q.symbol, q.ask_price-10, 2, plan=repriced)
        self.assertIsNone(s._position_exit_order(p, replace(q, bid_price=plan.stop_price+10), now=self.now))
        failed = replace(q, bid_price=plan.stop_price)
        order = s._position_exit_order(p, failed, now=plan.expires_at)
        self.assertEqual(order.requested_reason, "breakout_failed")
        self.assertEqual(order.order_type, OrderType.MARKET)
        self.assertEqual(s._position_exit_order(p, None, now=plan.expires_at).requested_reason, "breakout_failed")
        p = replace(p, stop_triggered=False)
        self.assertEqual(s._position_exit_order(p, None, now=plan.expires_at).requested_reason, "holding_expiry")
        s.positions[p.symbol] = p
        s.set_simulated_now(plan.expires_at)
        self.assertEqual(s.on_batch_tick([])[0].requested_reason, "holding_expiry")

    def test_batch_stop_survives_rebound_failure_and_restart_without_past_price_fill(self):
        s = self.strategy()
        symbol = "18064K"
        plan = replace(make_plan(symbol, self.now, quantity=6, entry=73700,
                                 breakout=73500, target=74700), stop_price=72900)
        p = PositionState(symbol, 73683, 6, self.now - timedelta(seconds=20), plan=plan)
        s.positions[symbol] = p
        crossed = make_quote(symbol, 73000, self.now - timedelta(seconds=1), ask=73000, bid=72900)
        rebound = make_quote(symbol, 73800, self.now, ask=73800, bid=73700)
        order, = s.on_batch_tick([crossed, rebound])
        self.assertEqual((order.requested_reason, order.order_type, order.price, order.reference_price),
                         ("breakout_failed", OrderType.MARKET, 0, 73700))
        self.assertTrue(p.stop_triggered)
        self.assertIsNone(p.pending_exit_started_at)
        self.assertEqual(s.daily_pnl.realized_net_pnl, 0)
        self.assertEqual(p.quantity, 6)

        s.on_order_filled(OrderResult(False, symbol=symbol, side=OrderSide.SELL,
                                     requested_reason=order.requested_reason, error_category="rate_limit"))
        restored = self.strategy()
        restored._load_daily_state()
        self.assertTrue(restored.positions[symbol].stop_triggered)
        retry, = restored.on_batch_tick([rebound])
        self.assertEqual((retry.requested_reason, retry.reference_price), ("breakout_failed", 73700))
        restored.on_order_filled(OrderResult(True, symbol=symbol, side=OrderSide.SELL,
            order_no="stop-sell", requested_quantity=6, fill_mode="market_pending",
            requested_reason=retry.requested_reason))
        self.assertEqual(restored.on_batch_tick([crossed, rebound]), [])

    def test_batch_stop_ignores_pre_fill_stale_future_and_invalid_observations(self):
        symbol = "005930"
        crossed = make_quote(symbol, 9950, self.now - timedelta(seconds=1), bid=9940, ask=9950)
        variants = {
            "pre_fill": replace(crossed, timestamp=self.now - timedelta(seconds=4)),
            "stale": replace(crossed, timestamp=self.now - timedelta(seconds=6)),
            "future": replace(crossed, timestamp=self.now + timedelta(seconds=1)),
            "other_symbol": replace(crossed, symbol="000660"),
            "no_book": replace(crossed, book_available=False),
            "crossed_book": replace(crossed, ask_price=9930),
        }
        for name, observed in variants.items():
            with self.subTest(name=name):
                s = self.strategy()
                p = PositionState(symbol, 10010, 10, self.now - timedelta(seconds=3),
                                  plan=make_plan(symbol, self.now))
                latest = make_quote(symbol, 10020, self.now)
                self.assertIsNone(s._position_exit_order(p, latest, now=self.now,
                                                         observed_quotes=[observed]))
                self.assertFalse(p.stop_triggered)
                payload = s._serialize_position(p)
                payload.pop("stop_triggered")
                self.assertFalse(s._deserialize_position(payload).stop_triggered)

    def test_frozen_drawdown_stop_sizes_the_same_risk_and_preserves_old_plans(self):
        s = self.strategy()
        q = self.warm(s)
        s.daily_pnl.realized_net_pnl = -4700
        plan = s._build_trade_plan(q, now=self.now)
        self.assertTrue(plan.allowed)
        # A monotonic observed path still reserves one legal tick, spread and costs.
        self.assertEqual(plan.stop_price, q.bid_price - 10)
        self.assertLessEqual(plan.committed_risk_net_abs, 300)
        per_share = s._risk_per_share(q, plan.entry_limit_price, plan.stop_price)
        self.assertEqual(plan.committed_risk_net_abs, ceil(plan.quantity * per_share))
        self.assertGreater((plan.quantity + 1) * per_share, 300)
        self.assertEqual(s._deserialize_plan(s._serialize_plan(plan)), plan)

        previous = replace(plan, stop_price=min(p.bid_price for p in s._breakout_window(q)))
        self.assertEqual(s._deserialize_plan(s._serialize_plan(previous)), previous)

        # Restoring an old holding must not widen a stop whose larger risk was never reserved.
        payload = s._serialize_plan(plan)
        payload.pop("stop_price")
        old = s._deserialize_plan(payload)
        self.assertEqual(old.stop_price, 0)
        position = PositionState(q.symbol, plan.entry_limit_price, plan.quantity, plan=old)
        order = s._position_exit_order(position, replace(q, bid_price=plan.breakout_price), now=self.now)
        self.assertEqual(order.requested_reason, "breakout_failed")

        # Subsequent lower quotes, partial fills and restored state cannot move a new stop.
        repriced = s._reprice_plan_for_fill(plan=plan, fill_price=plan.entry_limit_price-10, quantity=1)
        self.assertEqual(repriced.stop_price, plan.stop_price)
        self.assertEqual(s._deserialize_plan(s._serialize_plan(repriced)).stop_price, plan.stop_price)

    def test_stop_uses_peak_to_trough_drawdown_not_the_completed_rise(self):
        plans = []
        for start in (9500, 9800):
            s = self.strategy()
            history = []
            for second in range(-60, 0):
                bid = (start if second < -30 else 9950 if second < -20
                       else 9940 if second < -15 else 9930 if second < -10 else 9990)
                history.append(make_quote("005930", bid, self.now + timedelta(seconds=second),
                                          bid=bid, ask=bid+10))
            s._ingest_quotes(history)
            signal = make_quote("005930", 10020, self.now, bid=10010, ask=10020)
            # Neither the current observation nor a future low belongs to the prior path.
            s._ingest_quotes([signal, replace(signal, timestamp=self.now+timedelta(seconds=1),
                                              current_price=9000, bid_price=8990, ask_price=9000)])
            plan = s._build_trade_plan(signal, now=self.now)
            self.assertTrue(plan.allowed)
            self.assertEqual(plan.stop_price, 9990)  # Two successive 10-won declines, not just one step.
            self.assertEqual(plan.quantity, 86)
            self.assertGreater(plan.quantity * plan.entry_limit_price, 850000)
            per_share = s._risk_per_share(signal, plan.entry_limit_price, plan.stop_price)
            self.assertEqual(plan.committed_risk_net_abs, ceil(per_share * plan.quantity))
            self.assertLessEqual(plan.committed_risk_net_abs, 5000)
            self.assertGreater(per_share * (plan.quantity + 1), 5000)
            plans.append(plan)
        self.assertEqual(plans[0].stop_price, plans[1].stop_price)
        self.assertEqual(plans[0].quantity, plans[1].quantity)
        self.assertEqual(plans[0].committed_risk_net_abs, plans[1].committed_risk_net_abs)

        s = self.strategy()
        s._ingest_quotes([make_quote("005930", bid, self.now+timedelta(seconds=second),
                                    bid=bid, ask=bid+10)
                          for second in range(-60, 0)
                          for bid in [9800 if second < -30 else 9990]])
        plan = s._build_trade_plan(signal, now=self.now)
        self.assertTrue(plan.allowed)
        self.assertEqual((plan.stop_price, plan.quantity), (10000, 99))
        self.assertLessEqual(plan.committed_risk_net_abs, 5000)
        unit_cost = plan.entry_limit_price * (1 + s.config.commission_rate)
        self.assertLessEqual(plan.quantity * unit_cost, 1000000)
        self.assertGreater((plan.quantity + 1) * unit_cost, 1000000)

    def test_buy_fill_uses_actual_quantity_without_changing_stop_or_expiry(self):
        s = self.strategy()
        q = self.warm(s)
        order = s.on_batch_tick([q])[0]
        original = s._pending_entry_meta[q.symbol]["plan"]
        s.on_order_filled(OrderResult(success=True, symbol=q.symbol, side=OrderSide.BUY,
            order_no="buy-1", quantity=2, price=order.price-10, requested_quantity=order.quantity, fill_mode="limit"))
        p = s.positions[q.symbol]
        self.assertEqual((p.quantity, p.buy_price), (2, order.price-10))
        self.assertEqual((p.plan.breakout_price, p.plan.expires_at), (original.breakout_price, original.expires_at))
        self.assertEqual(p.plan.stop_price, original.stop_price)
        self.assertNotIn(q.symbol, s._pending_entry_meta)

    def test_partial_fill_keeps_signal_risk_when_the_book_changes(self):
        s = self.strategy()
        q = make_quote("003670", 196200, self.now, ask=196200, bid=196100)
        original = make_plan(q.symbol, self.now, quantity=4, entry=196200,
                             breakout=196000, risk=2599)
        s._pending_entry_meta[q.symbol] = dict(plan=original, created_at=self.now, order_no="buy")
        s._quotes_cache[q.symbol] = replace(q, bid_price=195900)
        s.on_order_filled(OrderResult(True, symbol=q.symbol, side=OrderSide.BUY,
            order_no="buy", quantity=2, price=196200, requested_quantity=4, fill_mode="limit_partial"))
        p = s.positions[q.symbol]
        self.assertEqual(p.plan.committed_risk_net_abs, 1300)
        self.assertEqual((p.plan.breakout_price, p.plan.expires_at),
                         (original.breakout_price, original.expires_at))
        s.sync_positions_from_account([Position(q.symbol, 3, 196200)])
        self.assertEqual(p.plan.committed_risk_net_abs, 1950)
        s.sync_positions_from_account([Position(q.symbol, 3, 196200)])
        self.assertEqual(p.plan.committed_risk_net_abs, 1950)

        cheaper = s._reprice_plan_for_fill(plan=original, fill_price=196100, quantity=2)
        self.assertEqual(cheaper.committed_risk_net_abs, 1100)

    def test_old_buy_result_cannot_change_a_new_order_for_the_same_symbol(self):
        s = self.strategy()
        q = self.warm(s)
        s.on_batch_tick([q])
        s._pending_entry_meta[q.symbol]["order_no"] = "new-buy"
        plan = s._pending_entry_meta[q.symbol]["plan"]
        for success in (True, False):
            with self.subTest(success=success):
                accepted = s.on_order_filled(OrderResult(success, symbol=q.symbol, side=OrderSide.BUY,
                    order_no="old-buy", quantity=2, price=q.ask_price,
                    requested_quantity=plan.quantity, fill_mode="entry_reconciled"))
                self.assertFalse(accepted)
                self.assertEqual(s._pending_entry_meta[q.symbol]["order_no"], "new-buy")
                self.assertEqual(s._pending_entry_meta[q.symbol]["plan"], plan)
                self.assertEqual(s.positions, {})

    def test_no_holding_failure_keeps_sell_snapshot_until_history_confirms_the_fill(self):
        s = self.strategy()
        q = make_quote("005930", 10000, self.now)
        s.positions[q.symbol] = PositionState(q.symbol, 10010, 5, self.now,
                                             make_plan(q.symbol, self.now, quantity=5))
        s.on_order_filled(OrderResult(True, symbol=q.symbol, side=OrderSide.SELL,
            order_no="sell", requested_quantity=5, fill_mode="market_pending"))
        pending = s._pending_sell_fills[q.symbol]
        s.on_order_filled(OrderResult(False, symbol=q.symbol, side=OrderSide.SELL,
            requested_quantity=5, error_category="no_holding"))
        self.assertIs(s._pending_sell_fills[q.symbol], pending)
        self.assertIsNotNone(s.positions[q.symbol].pending_exit_started_at)
        s.reconcile_no_holding_sell_failures_from_account([
            OrderResult(False, symbol=q.symbol, side=OrderSide.SELL, error_category="no_holding")], [])
        s.sync_positions_from_account([])
        account = Mock()
        account._coerce_int.side_effect = lambda value: int(value or 0)
        history = Mock(empty=False)
        history.to_dict.return_value = [dict(odno="sell", pdno=q.symbol,
            tot_ccld_qty=5, tot_ccld_amt=49900, avg_prvs=9980)]
        self.assertEqual(len(s.confirm_reconciled_sell_fills(account, history=history)), 1)
        self.assertEqual(s.daily_pnl.realized_net_pnl, s._trade_pnl(q.symbol, 10010, 9980, 5).net_pnl)
        self.assertFalse(s._pending_sell_fills)

    def test_paper_profit_exit_uses_fixed_target_and_positive_net_not_risk(self):
        for paper in (True, False):
            with self.subTest(paper=paper):
                s = self.strategy()
                s.market_data = SimpleNamespace(client=SimpleNamespace(config=SimpleNamespace(is_paper=paper)))
                q = make_quote("005930", 10200, self.now, bid=10110, ask=10210)
                net = s._trade_pnl(q.symbol, 10010, q.bid_price, 10).net_pnl
                self.assertLess(net, (q.bid_price - 10010) * 10)
                p = PositionState(q.symbol, 10010, 10,
                                  plan=make_plan(q.symbol, self.now, risk=1, target=q.bid_price + 10))
                self.assertIsNone(s._position_exit_order(p, q, now=self.now))
                p.plan = replace(p.plan, target_price=q.bid_price, committed_risk_net_abs=net + 1000)
                order = s._position_exit_order(p, q, now=self.now)
                if paper:
                    self.assertEqual((order.requested_reason, order.order_type, order.quantity),
                                     ("planned_profit", OrderType.MARKET, 10))
                    self.assertEqual(order.reference_price, q.bid_price)
                else:
                    self.assertIsNone(order)

    def test_profit_quote_validity_does_not_block_stop_or_expiry(self):
        s = self.strategy()
        s.market_data = SimpleNamespace(client=SimpleNamespace(config=SimpleNamespace(is_paper=True)))
        q = make_quote("005930", 10200, self.now)
        p = PositionState(q.symbol, 10010, 10, plan=make_plan(q.symbol, self.now))
        for invalid in (replace(q, book_available=False), replace(q, book_depth_available=False),
                        replace(q, bid_size=9), replace(q, ask_price=q.bid_price - 1),
                        replace(q, timestamp=self.now - timedelta(seconds=s.config.quote_freshness_seconds + 1)),
                        replace(q, timestamp=self.now + timedelta(seconds=1))):
            with self.subTest(quote=invalid):
                self.assertIsNone(s._position_exit_order(p, invalid, now=self.now))
                self.assertEqual(s._position_exit_order(p, invalid, now=p.plan.expires_at).requested_reason,
                                 "holding_expiry")
        expired = replace(q, timestamp=p.plan.expires_at)
        self.assertEqual(s._position_exit_order(p, expired, now=p.plan.expires_at).requested_reason,
                         "planned_profit")
        # A low cost basis can make the stop quote profitable: the stop still takes precedence.
        p.buy_price = 9000
        failed = replace(q, bid_price=p.plan.breakout_price, bid_size=0)
        self.assertEqual(s._position_exit_order(p, failed, now=p.plan.expires_at).requested_reason,
                         "breakout_failed")

    def test_partial_buy_profit_keeps_fixed_target_and_waits_for_confirmed_sell(self):
        s = self.strategy()
        s.market_data = SimpleNamespace(client=SimpleNamespace(
            config=SimpleNamespace(is_paper=True), rate_limit_cooldown_remaining=lambda: 0.0))
        q = self.warm(s)
        buy = s.on_batch_tick([q])[0]
        original = s._pending_entry_meta[q.symbol]["plan"]
        s.on_order_filled(OrderResult(True, symbol=q.symbol, side=OrderSide.BUY, order_no="buy-profit",
            quantity=2, price=buy.price - 10, requested_quantity=buy.quantity, fill_mode="limit_partial"))
        p = s.positions[q.symbol]
        self.assertLess(p.plan.committed_risk_net_abs, original.committed_risk_net_abs)
        self.assertEqual(p.plan.target_price, original.target_price)
        q = replace(q, timestamp=self.now + timedelta(seconds=1), current_price=original.target_price,
                    bid_price=original.target_price, ask_price=original.target_price + 10)
        net = s._trade_pnl(q.symbol, p.buy_price, q.bid_price, p.quantity).net_pnl
        self.assertGreater(net, 0)
        s.set_simulated_now(q.timestamp)
        sell = s.on_batch_tick([q])[0]
        self.assertEqual((sell.requested_reason, sell.quantity), ("planned_profit", 2))
        self.assertEqual(s.daily_pnl.realized_net_pnl, 0)
        s.on_order_filled(OrderResult(True, symbol=q.symbol, side=OrderSide.SELL, order_no="sell-profit",
            requested_quantity=2, fill_mode="market_pending", requested_reason=sell.requested_reason))
        self.assertEqual(s.on_batch_tick([q]), [])
        self.assertEqual(s.daily_pnl.realized_net_pnl, 0)
        s.on_order_filled(OrderResult(True, symbol=q.symbol, side=OrderSide.SELL, order_no="sell-profit",
            quantity=2, price=q.bid_price, requested_quantity=2, fill_mode="market",
            requested_reason=sell.requested_reason))
        self.assertEqual(s.daily_pnl.realized_net_pnl, net)
        self.assertNotIn(q.symbol, s.positions)
        self.assertFalse(s._halted)

    def test_partial_fill_does_not_replace_original_limit_with_cheaper_fill(self):
        s = self.strategy()
        q = self.warm(s)
        order = s.on_batch_tick([q])[0]
        original = s._pending_entry_meta[q.symbol]["plan"]
        s.on_order_filled(OrderResult(success=True, symbol=q.symbol, side=OrderSide.BUY,
            order_no="buy-1", quantity=2, price=order.price-10,
            requested_quantity=order.quantity, fill_mode="limit_partial"))
        s.sync_positions_from_account([Position(q.symbol, 4, order.price-5)])
        p = s.positions[q.symbol]
        self.assertEqual((p.quantity, p.buy_price), (4, order.price-5))
        self.assertEqual(p.plan.entry_limit_price, order.price)
        self.assertTrue(p.plan.allowed)
        self.assertEqual((p.plan.breakout_price, p.plan.expires_at),
                         (original.breakout_price, original.expires_at))
        self.assertIsNone(s._position_exit_order(p, q, now=self.now))
        s.sync_positions_from_account([Position(q.symbol, 4, order.price+5)])
        self.assertEqual(p.buy_price, order.price-5)
        self.assertTrue(p.plan.allowed)
        # Only newly acquired shares can change the account-derived cost basis.
        s.sync_positions_from_account([Position(q.symbol, 5, order.price+5)])
        self.assertFalse(p.plan.allowed)
        self.assertEqual(s._position_exit_order(p, q, now=self.now).requested_reason,
                         "invalid_entry_plan")

    def test_account_average_cannot_rewrite_fill_cost_during_sell_retry(self):
        s = self.strategy()
        q = make_quote("001820", 156000, self.now, bid=155900, ask=156000)
        plan = make_plan(q.symbol, self.now, quantity=6, entry=156000, breakout=155800)
        s._quotes_cache[q.symbol] = q
        s._pending_entry_meta[q.symbol] = {"plan": plan, "created_at": self.now}
        s.on_order_filled(OrderResult(True, symbol=q.symbol, side=OrderSide.BUY,
            order_no="buy-known", quantity=6, price=156000, requested_quantity=6, fill_mode="limit"))
        position = s.positions[q.symbol]
        original_risk = position.plan.committed_risk_net_abs
        s.on_order_filled(OrderResult(True, symbol=q.symbol, side=OrderSide.SELL,
            requested_quantity=6, fill_mode="order_result_pending", requested_reason="breakout_failed"))

        s.sync_positions_from_account([Position(q.symbol, 6, 155650)])
        self.assertEqual(position.buy_price, 156000)
        self.assertEqual(position.plan.committed_risk_net_abs, original_risk)
        s.set_simulated_now(self.now + timedelta(seconds=181))
        s.sync_positions_from_account([Position(q.symbol, 6, 155650)])
        self.assertIsNotNone(position.pending_exit_started_at)
        self.assertEqual(position.buy_price, 156000)

        s.on_order_filled(OrderResult(True, symbol=q.symbol, side=OrderSide.SELL,
            order_no="sell-retry", quantity=6, price=156000, requested_quantity=6, fill_mode="filled"))
        self.assertEqual(s.daily_pnl.realized_net_pnl, -1669)
        self.assertEqual(s.daily_pnl.win_count, 0)
        self.assertEqual(s._remaining_daily_loss_room(), 3331)
        self.assertEqual(s._sell_fill_ledger[-1]["buy_price"], 156000)

    def test_account_quantity_reduction_keeps_remaining_shares_at_original_cost(self):
        s = self.strategy()
        q = make_quote("001820", 156000, self.now, bid=155900, ask=156000)
        plan = make_plan(q.symbol, self.now, quantity=6, entry=156000, breakout=155800)
        s._quotes_cache[q.symbol] = q
        s.positions[q.symbol] = PositionState(q.symbol, 156000, 6, self.now, plan=plan)
        s.on_order_filled(OrderResult(True, symbol=q.symbol, side=OrderSide.SELL,
            order_no="sell-partial", requested_quantity=6, fill_mode="market_pending"))
        s.sync_positions_from_account([Position(q.symbol, 3, 155650)])
        self.assertEqual((s.positions[q.symbol].quantity, s.positions[q.symbol].buy_price), (3, 156000))
        self.assertEqual(s.positions[q.symbol].plan.quantity, 3)
        self.assertEqual(s._pending_sell_fills[q.symbol]["position"].buy_price, 156000)

    def test_unconfirmed_partial_buy_keeps_plan_until_account_reconciliation(self):
        s = self.strategy()
        q = self.warm(s)
        order = s.on_batch_tick([q])[0]
        result = OrderResult(success=True, symbol=q.symbol, side=OrderSide.BUY,
            order_no="buy-1", quantity=2, price=order.price-10,
            requested_quantity=order.quantity, fill_mode="partial_fill_pending")
        s.on_order_filled(result)
        self.assertEqual(s.positions[q.symbol].quantity, 2)
        self.assertEqual(s._pending_entry_meta[q.symbol]["plan"].quantity, order.quantity)
        self.assertEqual(s._pending_entry_meta[q.symbol]["order_no"], "buy-1")
        account = [Position(q.symbol, 4, order.price-5)]
        with self.assertLogs("kis_trader.orders", level="INFO") as logs:
            recovered = s.reconcile_pending_fills_from_account([result], account)
        self.assertEqual((recovered[0].quantity, recovered[0].price), (2, order.price))
        self.assertEqual(len(logs.output), 1)
        self.assertIn(f"2주 @ 체결가 {order.price}원", logs.output[0])
        with self.assertNoLogs("kis_trader.orders", level="INFO"):
            self.assertEqual(s.reconcile_pending_fills_from_account([result], account), [])
        s.sync_positions_from_account(account)
        self.assertIn(q.symbol, s._pending_entry_meta)
        p = s.positions[q.symbol]
        self.assertEqual((p.quantity, p.buy_price), (4, order.price-5))
        self.assertTrue(p.plan.allowed)
        self.assertEqual(p.plan.entry_limit_price, order.price)
        with self.assertLogs("kis_trader.orders", level="INFO") as logs:
            s.on_order_filled(OrderResult(True, order_no="buy-1", symbol=q.symbol,
                side=OrderSide.BUY, quantity=4, price=order.price-5,
                requested_quantity=order.quantity, fill_mode="entry_reconciled"))
        self.assertEqual(len(logs.output), 1)
        self.assertIn("매수 잔량 취소 확정", logs.output[0])
        self.assertIn(f"취소={order.quantity-4}주 누적체결=4주", logs.output[0])
        self.assertNotIn(q.symbol, s._pending_entry_meta)
        self.assertEqual(s.positions[q.symbol].quantity, 4)

    def test_empty_balance_cannot_expire_a_known_buy_order_or_release_its_risk(self):
        s = self.strategy()
        q = self.warm(s)
        order = s.on_batch_tick([q])[0]
        s.on_order_filled(OrderResult(True, order_no="known-buy", symbol=q.symbol,
            side=OrderSide.BUY, requested_quantity=order.quantity, fill_mode="limit_pending"))
        risk = s._pending_entry_risk()
        s.set_simulated_now(self.now + timedelta(seconds=181))
        s.sync_positions_from_account([])
        self.assertEqual(s._pending_entry_meta[q.symbol]["order_no"], "known-buy")
        self.assertEqual(s._pending_entry_risk(), risk)
        self.assertFalse(s._entry_submission_available())
        cancelled = OrderResult(False, order_no="known-buy", symbol=q.symbol,
            side=OrderSide.BUY, fill_mode="limit_cancelled", error_category="not_filled",
            requested_quantity=order.quantity, requested_price=order.price)
        with self.assertLogs("kis_trader.orders", level="INFO") as logs:
            self.assertTrue(s.on_order_filled(cancelled))
        self.assertEqual(len(logs.output), 1)
        self.assertIn(f"매수 미체결 취소: buy {q.symbol} {order.quantity}주", logs.output[0])
        self.assertIn(f"주문가 {order.price}원 order_no=known-buy", logs.output[0])
        with self.assertNoLogs("kis_trader.orders", level="INFO"):
            self.assertFalse(s.on_order_filled(cancelled))
        self.assertNotIn(q.symbol, s._pending_entry_meta)
        self.assertEqual(s._pending_entry_risk(), 0)
        self.assertEqual(s.daily_pnl.realized_net_pnl, 0)

    def test_unknown_buy_survives_empty_balance_timeout_and_restart(self):
        s = self.strategy()
        q = self.warm(s)
        s.on_batch_tick([q])
        original_risk = s._pending_entry_risk()
        s.set_simulated_now(self.now+timedelta(seconds=600))
        s.sync_positions_from_account([])
        restored = self.strategy()
        restored.initialize()
        self.assertEqual(restored._pending_entry_risk(), original_risk)
        self.assertIn(q.symbol, restored._pending_entry_meta)
        self.assertFalse(restored._entry_submission_available())

    def test_partial_buy_keeps_only_unfilled_reservation_and_recovers_cumulative_fill_once(self):
        s = self.strategy()
        q = self.warm(s)
        order = s.on_batch_tick([q])[0]
        original = s._pending_entry_meta[q.symbol]["plan"]
        price = order.price-10
        partial = OrderResult(True, symbol=q.symbol, side=OrderSide.BUY,
            order_no="partial", quantity=2, price=price,
            requested_quantity=order.quantity, fill_mode="partial_fill_pending")
        with self.assertLogs("kis_trader.orders", level="INFO") as logs:
            s.on_order_filled(replace(partial))
        self.assertEqual(len(logs.output), 1)
        self.assertIn(f"매수 체결: buy {q.symbol} 2주 @ 체결가 {price}원", logs.output[0])
        with self.assertNoLogs("kis_trader.orders", level="INFO"):
            self.assertFalse(s.on_order_filled(replace(partial)))
        self.assertEqual(s._pending_entry_capital(), (order.quantity-2)*order.price)
        self.assertEqual(s._pending_entry_risk(), int(ceil(
            original.committed_risk_net_abs*(order.quantity-2)/order.quantity)))
        restored = self.strategy()
        restored.initialize()
        self.assertEqual(restored._pending_entry_meta[q.symbol]["confirmed_quantity"], 2)
        self.assertEqual(restored._quotes_cache, {})
        result = OrderResult(True, symbol=q.symbol, side=OrderSide.BUY,
            order_no="partial", quantity=5, price=price,
            requested_quantity=order.quantity, fill_mode="entry_reconciled")
        with self.assertLogs("kis_trader.orders", level="INFO") as logs:
            restored.on_order_filled(result)
        self.assertEqual(len(logs.output), 2)
        self.assertIn(f"매수 체결: buy {q.symbol} 3주 @ 체결가 {price}원", logs.output[0])
        self.assertIn("order_no=partial", logs.output[0])
        self.assertIn(f"취소={order.quantity-5}주 누적체결=5주", logs.output[1])
        self.assertEqual((result.quantity, result.price), (3, price))
        self.assertEqual(restored.positions[q.symbol].quantity, 5)
        self.assertEqual(restored.positions[q.symbol].plan.expires_at, original.expires_at)
        self.assertNotIn(q.symbol, restored._pending_entry_meta)
        with self.assertNoLogs("kis_trader.orders", level="INFO"):
            self.assertFalse(restored.on_order_filled(result))
        self.assertEqual(restored.positions[q.symbol].quantity, 5)
        self.assertEqual(restored.daily_pnl.realized_net_pnl, 0)

    def test_legacy_plan_preserves_order_identity_but_is_never_a_new_signal(self):
        s = self.strategy()
        payload = s._serialize_plan(make_plan("005930", self.now))
        payload.pop("breakout_price")
        payload.update(expected_net=1000, expected_exit_bid=10110, lower_exit_bid=9950, lower_net=-800)
        old = s._deserialize_pending_entry("005930", dict(plan=payload, order_no="existing-order", created_at=self.now.isoformat()))
        self.assertEqual(old["order_no"], "existing-order")
        self.assertEqual(old["plan"].breakout_price, 0)
        p = PositionState("005930", 10010, 10, plan=old["plan"])
        self.assertEqual(s._position_exit_order(p, None, now=self.now).requested_reason, "invalid_entry_plan")

    def test_old_profit_policy_is_converted_once_without_new_forecast_or_expiry(self):
        s = self.strategy()
        original = make_plan("005930", self.now, entry=10040, quantity=3, risk=240)
        payload = s._serialize_plan(original)
        payload.pop("target_price")
        migrated = s._deserialize_plan(payload)
        self.assertEqual((migrated.signal_at, migrated.expires_at, migrated.breakout_price),
                         (original.signal_at, original.expires_at, original.breakout_price))
        self.assertGreaterEqual(s._trade_pnl(migrated.symbol, migrated.entry_limit_price,
                                           migrated.target_price, migrated.quantity).net_pnl, 240)
        restored = s._deserialize_plan(s._serialize_plan(migrated))
        self.assertEqual(restored, migrated)
        cheaper = s._reprice_plan_for_fill(plan=migrated, fill_price=10030, quantity=1)
        self.assertEqual(cheaper.target_price, migrated.target_price)
        position = s._deserialize_position(dict(symbol="005930", quantity=3,
                                                buy_price=10000, plan=payload))
        self.assertLess(position.plan.target_price, migrated.target_price)
        self.assertGreaterEqual(s._trade_pnl(position.symbol, position.buy_price,
                                           position.plan.target_price, position.quantity).net_pnl, 240)

    def test_state_roundtrip_and_journal_keep_prices_without_forecast_fields(self):
        s = self.strategy()
        q = self.warm(s)
        s.on_batch_tick([q])
        s._save_daily_state()
        restored = self.strategy()
        restored._load_daily_state()
        self.assertEqual(restored._pending_entry_meta[q.symbol]["plan"], s._pending_entry_meta[q.symbol]["plan"])
        path = Path(s.config.quote_archive_root) / "2026/09/quote-batches.2026-09-21.jsonl.gz"
        batch = json.loads(gzip.decompress(path.read_bytes()).decode().splitlines()[0])
        self.assertEqual(batch["quotes"][0]["ask_price"], q.ask_price)
        self.assertTrue(batch["reset_history"])
        self.assertNotIn("forecast_id", json.loads(Path(s.config.daily_state_path).read_text())["pending_entry_meta"][q.symbol])

    def test_entry_cutoff_does_not_delay_existing_exit(self):
        s = self.strategy()
        s.set_simulated_now(self.now.replace(hour=15, minute=17))
        q = replace(self.warm(s), timestamp=s._now())
        self.assertEqual(s.on_batch_tick([q]), [])
        s.positions[q.symbol] = PositionState(q.symbol, q.ask_price, 1,
            plan=make_plan(q.symbol, s._now(), expires_delta=0))
        self.assertEqual(s.on_batch_tick([q])[0].requested_reason, "holding_expiry")

    def test_previous_trading_day_quote_cannot_enter_price_history(self):
        strategy = self.strategy()
        previous_day = make_quote(
            "005930",
            9_800,
            self.now - timedelta(days=1),
        )
        current = make_quote("005930", 10_000, self.now)

        strategy._ingest_quotes([previous_day, current])

        self.assertEqual(list(strategy._recent_quotes["005930"]), [current])

    def test_delayed_quotes_cannot_replace_newer_price_input(self):
        strategy = self.strategy()
        latest = make_quote("005930", 10_100, self.now)
        earlier = make_quote("005930", 10_000, self.now - timedelta(seconds=1))
        oldest = make_quote("005930", 9_900, self.now - timedelta(seconds=2))

        strategy._ingest_quotes([latest, earlier])
        self.assertEqual(list(strategy._recent_window("005930")), [earlier, latest])
        self.assertEqual(strategy._fresh_quotes(), [latest])
        strategy._ingest_quotes([oldest])
        self.assertEqual(strategy._fresh_quotes(), [latest])
        self.assertEqual(list(strategy._recent_window("005930")), [oldest, earlier, latest])

    def test_late_observation_repairs_only_observed_history_and_can_disprove_breakout(self):
        strategy = self.strategy()
        history = [make_quote("005930", 10000 + (second // 6 + 1) * 20,
                              self.now + timedelta(seconds=second))
                   for second in range(-60, 0, 5)]
        missing = history[6]
        signal = make_quote("005930", 10030, self.now)
        strategy._ingest_quotes([q for q in history if q is not missing] + [signal])
        self.assertEqual(strategy._build_trade_plan(signal, now=self.now).reason,
                         "price_window_incomplete")
        strategy._ingest_quotes([missing])
        self.assertTrue(strategy._build_trade_plan(signal, now=self.now).allowed)
        self.assertEqual(strategy._fresh_quotes(), [signal])
        self.assertEqual([q.timestamp for q in strategy._recent_window(signal.symbol)],
                         [q.timestamp for q in history] + [signal.timestamp])

        # A previously missing high can invalidate a signal; arrival is not a new tick.
        peak = replace(missing, current_price=11000, bid_price=10990, ask_price=11010)
        strategy._ingest_quotes([peak])
        self.assertEqual(strategy._build_trade_plan(signal, now=self.now).reason, "no_breakout")
        self.assertEqual(strategy._fresh_quotes(), [signal])
        self.assertEqual(len(strategy._recent_window(signal.symbol)), len(history) + 1)
        strategy.set_simulated_now(self.now + timedelta(seconds=6))
        strategy._ingest_quotes([peak])
        self.assertEqual(strategy._fresh_quotes(), [])

        # A later fractional quote must not write its high into an earlier observation.
        strategy = self.strategy()
        newer = make_quote("005930", 10200, self.now + timedelta(microseconds=500000))
        older = make_quote("005930", 10100, self.now)
        strategy.set_simulated_now(newer.timestamp)
        strategy._ingest_quotes([newer])
        strategy._ingest_quotes([older])
        first, last = strategy._recent_window(newer.symbol)
        self.assertLessEqual(first.trade_high_price, older.current_price)
        self.assertLessEqual(first.ask_high_price, older.ask_price)
        self.assertEqual(last.current_price, newer.current_price)

    def test_exit_becomes_pending_only_after_order_submission_succeeds(self):
        strategy = self.strategy()
        quote = make_quote("005930", 12_000, self.now)
        position = PositionState(
            symbol=quote.symbol,
            buy_price=10_000,
            quantity=10,
            buy_time=self.now - timedelta(seconds=100),
            plan=make_plan(quote.symbol, self.now, expires_delta=1),
        )
        strategy.positions[quote.symbol] = position

        order = strategy._position_exit_order(
            position,
            quote,
            now=self.now + timedelta(seconds=1),
        )

        self.assertIsNotNone(order)
        self.assertIsNone(position.pending_exit_started_at)
        self.assertNotIn(quote.symbol, strategy._pending_sell_fills)

        strategy.on_order_filled(
            OrderResult(
                success=True,
                order_no="0000012345",
                symbol=quote.symbol,
                side=OrderSide.SELL,
                quantity=0,
                price=0,
                reference_price=order.reference_price,
                requested_quantity=order.quantity,
                fill_mode="market_pending",
                requested_reason=order.requested_reason,
            )
        )

        self.assertEqual(position.pending_exit_started_at, self.now)
        self.assertEqual(position.pending_exit_quantity, 10)
        self.assertEqual(
            strategy._pending_sell_fills[quote.symbol]["order_no"],
            "0000012345",
        )

    def test_confirmed_sell_updates_inventory_before_daily_loss_check(self):
        for prior_loss, should_halt in ((-4027, False), (-4700, True)):
            with self.subTest(prior_loss=prior_loss):
                s = self.strategy()
                q = make_quote("138930", 15650, self.now, bid=15640, ask=15650)
                plan = make_plan(q.symbol, self.now, quantity=16, entry=15660, breakout=15640)
                s.positions[q.symbol] = PositionState(q.symbol, 15658, 16, self.now, plan=plan)
                s._quotes_cache[q.symbol] = q
                s._sell_fill_ledger = [dict(fill_id="prior", trade_key="prior", quantity=1,
                    entry_quantity=1, net_pnl=prior_loss, timestamp=self.now.isoformat())]
                s._rebuild_daily_pnl()
                s.on_order_filled(OrderResult(True, order_no="sell-1", symbol=q.symbol,
                    side=OrderSide.SELL, requested_quantity=16, fill_mode="market_pending"))
                account = Mock()
                account._coerce_int.side_effect = lambda value: int(value or 0)
                account.get_order_history.return_value.empty = False
                account.get_order_history.return_value.to_dict.return_value = [dict(
                    odno="sell-1", pdno=q.symbol, tot_ccld_qty=16, avg_prvs=15660,
                    tot_ccld_amt=16*15660)]

                s.confirm_reconciled_sell_fills(account)

                self.assertNotIn(q.symbol, s.positions)
                self.assertNotIn(q.symbol, s._pending_sell_fills)
                self.assertEqual(s._unrealized_net_pnl(), 0)
                self.assertEqual(s.daily_pnl.realized_net_pnl, prior_loss-414)
                self.assertEqual(s._halted, should_halt)
                self.assertEqual(s.confirm_reconciled_sell_fills(account), [])
                self.assertEqual(s.daily_pnl.realized_net_pnl, prior_loss-414)

    def test_partial_sell_reconciliation_is_cumulative_and_waits_for_final_price(self):
        for account_quantity in (5, 3, 0):
            with self.subTest(account_quantity=account_quantity):
                s = self.strategy()
                q = make_quote("005930", 10000, self.now)
                plan = make_plan(q.symbol, self.now, quantity=5)
                s.positions[q.symbol] = PositionState(q.symbol, 10000, 5, self.now, plan=plan)
                s._quotes_cache[q.symbol] = q
                s.on_order_filled(OrderResult(True, order_no="sell-1", symbol=q.symbol,
                    side=OrderSide.SELL, requested_quantity=5, fill_mode="market_pending"))
                account_positions = [Position(q.symbol, account_quantity, 10000)] if account_quantity else []
                s.sync_positions_from_account(account_positions)
                account = Mock()
                account._coerce_int.side_effect = lambda value: int(value or 0)
                account.get_order_history.return_value.empty = False
                account.get_order_history.return_value.to_dict.return_value = [dict(
                    odno="sell-1", pdno=q.symbol, tot_ccld_qty=2, avg_prvs=10000,
                    tot_ccld_amt=20000)]

                corrections = s.confirm_reconciled_sell_fills(account)

                self.assertEqual(corrections[0]["remaining_quantity"], 3)
                self.assertTrue(corrections[0]["partial"])
                self.assertEqual(s._pending_sell_fills[q.symbol]["confirmed_quantity"], 2)
                self.assertFalse(s._entry_submission_available())
                self.assertEqual(s.positions[q.symbol].quantity if q.symbol in s.positions else 0,
                                 min(3, account_quantity))
                if account_quantity:
                    s.sync_positions_from_account(account_positions)
                    self.assertEqual(s.positions[q.symbol].quantity, min(3, account_quantity))
                else:
                    s.sync_positions_from_account([Position(q.symbol, 5, 10000)])
                    restored = s.positions[q.symbol]
                    snapshot = s._pending_sell_fills[q.symbol]["position"]
                    self.assertEqual(restored.quantity, 3)
                    self.assertEqual(restored.trade_key, snapshot.trade_key)
                    self.assertIsNotNone(restored.plan)
                    self.assertEqual(restored.pending_exit_quantity, 3)
                    self.assertIsNone(s._position_exit_order(restored, q, now=self.now))
                pnl = s.daily_pnl.realized_net_pnl
                self.assertEqual(s.confirm_reconciled_sell_fills(account), [])
                self.assertEqual(s.daily_pnl.realized_net_pnl, pnl)

                account.get_order_history.return_value.to_dict.return_value = [dict(
                    odno="sell-1", pdno=q.symbol, tot_ccld_qty=5, avg_prvs=10006,
                    tot_ccld_amt=50030)]
                corrections = s.confirm_reconciled_sell_fills(account)
                self.assertEqual(corrections[0]["quantity"], 3)
                self.assertEqual(corrections[0]["corrected_price"], 10010)
                self.assertFalse(corrections[0]["partial"])
                self.assertNotIn(q.symbol, s.positions)
                self.assertNotIn(q.symbol, s._pending_sell_fills)
                self.assertEqual(sum(r["quantity"] for r in s._sell_fill_ledger), 5)
                self.assertTrue(s._entry_submission_available())

    def test_old_sell_confirmation_does_not_reduce_a_new_position(self):
        s = self.strategy()
        q = make_quote("005930", 10000, self.now)
        plan = make_plan(q.symbol, self.now, quantity=2)
        old = PositionState(q.symbol, 10000, 2, self.now, plan=plan)
        s.positions[q.symbol] = old
        s.on_order_filled(OrderResult(True, order_no="sell-old", symbol=q.symbol,
            side=OrderSide.SELL, requested_quantity=2, fill_mode="market_pending"))
        new = PositionState(q.symbol, 10100, 4, self.now+timedelta(seconds=60),
                            plan=replace(plan, quantity=4))
        s.positions[q.symbol] = new
        account = Mock()
        account._coerce_int.side_effect = lambda value: int(value or 0)
        account.get_order_history.return_value.empty = False
        account.get_order_history.return_value.to_dict.return_value = [dict(
            odno="sell-old", pdno=q.symbol, tot_ccld_qty=2, avg_prvs=10000)]

        s.confirm_reconciled_sell_fills(account)

        self.assertIs(s.positions[q.symbol], new)
        self.assertEqual(new.quantity, 4)

    def test_stale_unfilled_exit_keeps_unknown_order_until_terminal_history(self):
        strategy = self.strategy(pending_order_block_seconds=30)
        quote = make_quote("114800", 1_039, self.now)
        position = PositionState(
            symbol=quote.symbol,
            buy_price=1_039,
            quantity=2,
            buy_time=self.now - timedelta(minutes=5),
            plan=make_plan(
                quote.symbol,
                self.now - timedelta(minutes=5),
                quantity=2,
                entry=1_039,
                breakout=1_033,
                expires_delta=30,
            ),
            pending_exit_started_at=self.now - timedelta(seconds=31),
            pending_exit_quantity=2,
            pending_exit_reason="holding_expiry",
        )
        strategy.positions[quote.symbol] = position
        strategy._pending_sell_fills[quote.symbol] = {
            "position": strategy._copy_position(position),
            "order_no": "",
            "requested_quantity": 2,
            "requested_at": self.now - timedelta(seconds=31),
            "reference_price": 1_038,
            "reason": "holding_expiry",
            "account_remaining_quantity": 2,
            "confirmed_quantity": 0,
            "confirmed_amount": 0,
        }

        strategy.sync_positions_from_account(
            [Position(symbol=quote.symbol, quantity=2, avg_price=1_039)]
        )

        self.assertIsNotNone(position.pending_exit_started_at)
        self.assertIn(quote.symbol, strategy._pending_sell_fills)
        retry = strategy._position_exit_order(position, quote, now=self.now)
        self.assertIsNone(retry)

    def test_sell_retry_requires_closed_original_order_not_elapsed_time(self):
        import pandas as pd
        from src.account import AccountAPI

        for filled in (0, 1):
            with self.subTest(filled=filled):
                s = self.strategy()
                q = make_quote("005930", 10000, self.now)
                s.positions[q.symbol] = PositionState(q.symbol, 10000, 2, self.now,
                    plan=make_plan(q.symbol, self.now, quantity=2, expires_delta=-1))
                s.on_order_filled(OrderResult(True, symbol=q.symbol, side=OrderSide.SELL,
                    order_no="sell-known", requested_quantity=2, fill_mode="market_pending"))
                row = dict(pdno=q.symbol, odno="sell-known", tot_ccld_qty=str(filled),
                           tot_ccld_amt=str(filled*10000), avg_prvs="10000", rmn_qty="0",
                           cncl_yn="N", cnc_cfrm_qty=str(2-filled))
                history = pd.DataFrame([row])
                account = AccountAPI.__new__(AccountAPI)
                account.get_order_history = Mock(side_effect=AssertionError("network on strategy thread"))
                s.confirm_reconciled_sell_fills(account, history=history)
                self.assertNotIn(q.symbol, s._pending_sell_fills)
                self.assertEqual(s.positions[q.symbol].quantity, 2-filled)
                self.assertIsNone(s.positions[q.symbol].pending_exit_started_at)
                self.assertIsNotNone(s._position_exit_order(s.positions[q.symbol], q, now=self.now))
                self.assertEqual(len(s._sell_fill_ledger), filled)

    def test_previous_day_absent_position_does_not_pollute_today_pending_sells(self):
        strategy = self.strategy()
        previous_day = self.now - timedelta(days=1)
        carried = PositionState(
            symbol="114800",
            buy_price=1_039,
            quantity=2,
            buy_time=previous_day,
            plan=None,
        )
        orphan = PositionState(
            symbol="123310",
            buy_price=1_152,
            quantity=2,
            buy_time=previous_day,
            plan=None,
        )
        strategy.positions[carried.symbol] = carried
        strategy._pending_sell_fills[orphan.symbol] = {
            "position": orphan,
            "order_no": "",
            "requested_quantity": 2,
            "requested_at": self.now,
            "reference_price": 1_152,
            "reason": "external_or_reconciled_sell",
            "account_remaining_quantity": 0,
            "confirmed_quantity": 0,
            "confirmed_amount": 0,
        }

        strategy.sync_positions_from_account([])

        self.assertEqual(strategy.positions, {})
        self.assertEqual(strategy._pending_sell_fills, {})

    def test_same_day_absent_position_still_waits_for_sell_confirmation(self):
        strategy = self.strategy()
        position = PositionState(
            symbol="005930",
            buy_price=80_000,
            quantity=3,
            buy_time=self.now,
            plan=None,
        )
        strategy.positions[position.symbol] = position

        strategy.sync_positions_from_account([])

        self.assertNotIn(position.symbol, strategy.positions)
        self.assertEqual(
            strategy._pending_sell_fills[position.symbol]["reason"],
            "external_or_reconciled_sell",
        )

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

        with patch(
            "src.analytics.quote_archive.gzip.compress",
            side_effect=OSError("quote archive unavailable"),
        ):
            orders = strategy.on_batch_tick([quote])

        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].side, OrderSide.SELL)
        self.assertEqual(orders[0].order_type, OrderType.MARKET)
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

    def test_daily_profit_liquidates_before_expiry_but_waits_for_actual_fills_to_halt(self):
        for fill_price in (3490, 3475):
            with self.subTest(fill_price=fill_price):
                s = self.strategy(allow_hard_stop_bypass_for_day=True)
                s._sell_fill_ledger = [dict(fill_id="morning", trade_key="morning",
                    quantity=1, entry_quantity=1, net_pnl=-5574, timestamp=self.now.isoformat())]
                s._rebuild_daily_pnl()
                s._loss_guard_realized_baseline = -5574
                p = PositionState("131100", 3409, 237, self.now - timedelta(seconds=36),
                    make_plan("131100", self.now, quantity=237, entry=3410, breakout=3400))
                s.positions[p.symbol] = p
                quote = replace(make_quote(p.symbol, 3490, self.now, bid=3490, ask=3495),
                                bid_size=3191)
                orders = s.on_batch_tick([quote])
                self.assertEqual(len(orders), 1)
                order = orders[0]
                self.assertEqual((order.side, order.order_type, order.quantity),
                                 (OrderSide.SELL, OrderType.MARKET, 237))
                self.assertEqual(order.requested_reason, "daily_profit_target")
                self.assertIsNone(order.submit_before)
                self.assertTrue(s.should_continue())

                s.on_order_filled(OrderResult(success=True, symbol=p.symbol, side=OrderSide.SELL,
                    order_no="take-profit", requested_quantity=237, fill_mode="market_pending",
                    requested_reason=order.requested_reason))
                self.assertEqual(s.on_batch_tick([quote]), [])
                s.on_order_filled(OrderResult(success=True, symbol=p.symbol, side=OrderSide.SELL,
                    order_no="take-profit", quantity=100, price=fill_price, requested_quantity=237,
                    requested_reason=order.requested_reason))
                self.assertEqual(s.positions[p.symbol].quantity, 137)
                self.assertTrue(s.should_continue())
                self.assertEqual(s.on_batch_tick([quote]), [])
                s.on_order_filled(OrderResult(success=True, symbol=p.symbol, side=OrderSide.SELL,
                    order_no="take-profit", quantity=137, price=fill_price, requested_quantity=137,
                    requested_reason=order.requested_reason))
                self.assertNotIn(p.symbol, s.positions)
                self.assertFalse(s._pending_sell_fills)
                self.assertEqual(s.should_continue(), fill_price == 3475)
                self.assertEqual(s._halt_reason, "daily_profit_target" if fill_price == 3490 else "")

    def test_profit_exit_uses_net_bid_value_with_fresh_full_depth_and_no_unknown_orders(self):
        for case in ("costs", "stale", "missing_book", "unknown_depth", "thin_depth",
                     "missing_position_quote", "unconfirmed_sell", "unconfirmed_buy", "valid"):
            with self.subTest(case=case):
                s = self.strategy()
                s.daily_pnl.realized_net_pnl = 9000
                s.positions["005930"] = PositionState("005930", 10000, 10, self.now,
                    make_plan("005930", self.now, breakout=9950))
                q = make_quote("005930", 10200, self.now, bid=10200, ask=10210)
                if case == "costs":
                    q = replace(q, current_price=11000, ask_price=11000, bid_price=10110)
                elif case == "stale":
                    q = replace(q, timestamp=self.now - timedelta(seconds=6))
                elif case == "missing_book":
                    q = replace(q, book_available=False)
                elif case == "unknown_depth":
                    q = replace(q, book_depth_available=False)
                elif case == "thin_depth":
                    q = replace(q, bid_size=9)
                elif case == "missing_position_quote":
                    s.positions["000660"] = PositionState("000660", 10000, 1, self.now,
                        make_plan("000660", self.now))
                elif case == "unconfirmed_sell":
                    s._pending_sell_fills["000660"] = {"position": PositionState("000660", 10000, 1)}
                elif case == "unconfirmed_buy":
                    s._pending_entry_meta["000660"] = {"plan": make_plan("000660", self.now)}
                orders = s.on_batch_tick([q])
                if case == "valid":
                    self.assertEqual([o.requested_reason for o in orders], ["daily_profit_target"])
                else:
                    self.assertEqual(orders, [])
                self.assertFalse(s._halted)


if __name__ == "__main__":
    unittest.main()
