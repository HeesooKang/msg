import gzip
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path
from importlib.util import find_spec
from types import SimpleNamespace

import numpy as np

HAS_RESEARCH_DEPENDENCIES = find_spec("sklearn") is not None
if HAS_RESEARCH_DEPENDENCIES:
    from scripts.evaluate_return_distribution import collect, evaluate, observe, summarize_execution_predictions
    from scripts.return_distribution import ExecutionReturnDistribution, ReturnDistribution
    from scripts.return_distribution import ReturnEstimate
    from scripts.replay_return_distribution import candidate_strategy, limit_entry_rejection
    from scripts.replay_execution_policy import MemoryArchive
    from scripts.evaluate_horizons import label_endpoints, label_limit_entries, select_signals
from src.models import Quote
from src.strategies.momentum_scalp_types import MomentumScalpConfig, PositionState
from src.strategies.momentum_scalp_pnl import estimate_trade_net_pnl_unrounded


@unittest.skipUnless(HAS_RESEARCH_DEPENDENCIES, "install requirements-research.txt for offline model tests")
class ReturnDistributionTests(unittest.TestCase):
    def row(self, index, value, features=None):
        return dict(day="2026-10-19", symbol=f"{index:06d}",
                    signal_at="2026-10-19T09:00:00", exit_at="2026-10-19T09:03:01",
                    status="complete", net_return=value, features=features or [0., 0.])

    def quote(self, second, bid=10000):
        return Quote(symbol="005930", timestamp=datetime(2026, 10, 20, 10) + timedelta(seconds=second),
                     current_price=bid+5, bid_price=bid, ask_price=bid+10,
                     bid_size=100, ask_size=100, book_available=True, book_depth_available=True)

    def test_distribution_is_over_outcomes_not_tree_mean_predictions(self):
        rows = [self.row(i, -.01 if i % 2 else .03) for i in range(200)]
        model = ReturnDistribution().fit(rows, before="2026-10-20")
        p = model.predict([[0., 0.]])[0]
        self.assertAlmostEqual(p.loss_probability, .5, delta=.03)
        self.assertAlmostEqual(p.expected_return, .01, delta=.001)
        self.assertEqual(p.lower_return, -.01)
        # Tree means are all positive; treating them as scenarios would report no losses.
        self.assertTrue(all(tree.predict([[0., 0.]])[0] > 0 for tree in model.forest.estimators_))

    def test_future_and_unknown_outcomes_cannot_change_fit_or_weights(self):
        rows = [self.row(i, -.01 if i % 2 else .02, [float(i % 2), 0.]) for i in range(100)]
        extra = [dict(self.row(999, 100), exit_at="2026-10-21T09:00:00"),
                 dict(self.row(999, 100), status="unknown")]
        a = ReturnDistribution().fit(rows, before="2026-10-20")
        b = ReturnDistribution().fit(rows + extra, before="2026-10-20")
        self.assertEqual(a.predict([[0., 0.], [1., 0.]]), b.predict([[0., 0.], [1., 0.]]))
        self.assertEqual(b.training_count, 100)
        self.assertLess(b.training_through, "2026-10-20")
        with self.assertRaisesRegex(ValueError, "no_completed"):
            ReturnDistribution().fit(extra, before="2026-10-20")

    def test_same_day_symbol_repetitions_do_not_dominate_distribution(self):
        rows = [self.row(1, -.01)] * 1000 + [self.row(2, .01)] * 100
        model = ReturnDistribution().fit(rows, before="2026-10-20")
        p = model.predict([[0., 0.]])[0]
        self.assertAlmostEqual(p.loss_probability, .5, delta=.03)
        self.assertAlmostEqual(p.expected_return, 0, delta=.0006)

    def test_fixed_evaluation_uses_only_prior_training_not_query_outcomes(self):
        training = [self.row(i, -.01 if i % 2 else .03) for i in range(100)]
        query = dict(self.row(999, 100.), day="2026-10-20",
                     signal_at="2026-10-20T09:00:00", exit_at="2026-10-20T09:03:01")
        future = dict(query, day="2026-10-21", exit_at="2026-10-21T09:03:01")
        a = evaluate([query], training_rows=training + [query, future])
        b = evaluate([dict(query, net_return=-100.)], training_rows=training)
        self.assertEqual(len(a), 1)
        self.assertEqual(a[0]["training_count"], 100)
        self.assertEqual(a[0]["prediction"], b[0]["prediction"])
        self.assertEqual(evaluate([query], training_rows=[]), [])

    def test_features_use_observed_queue_depth_and_no_future_prices(self):
        history = [self.quote(i, 9700 + i * 5) for i in range(60)]
        q = self.quote(60)
        a = observe(history, q, commission=.000142, tax=.0015)
        b = observe(history + [self.quote(61, 9000)], q, commission=.000142, tax=.0015)
        self.assertEqual(a, b)
        self.assertEqual(a["features"][7:10], [0., 0., 0.])
        c = observe(history, replace(q, bid_size=900), commission=.000142, tax=.0015)
        self.assertAlmostEqual(c["features"][7], .8)
        self.assertIsNone(observe(history[:20] + history[30:], q, commission=.000142, tax=.0015))
        self.assertIsNone(observe(history, replace(q, book_depth_available=False), commission=.000142, tax=.0015))

    def dataset(self, quotes, universe=True):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "2026/10/quote-batches.2026-10-20.jsonl.gz"
            path.parent.mkdir(parents=True)
            with gzip.open(path, "wt") as out:
                for q in quotes:
                    out.write(json.dumps(dict(observed_timestamp=q.timestamp.isoformat(),
                        universe=[q.symbol] if universe else [],
                        quotes=[dict(asdict(q), timestamp=q.timestamp.isoformat())])) + "\n")
            return collect(root, "2026-10-20", {})

    def test_label_uses_next_bid_after_stop_and_exact_costs(self):
        quotes = [self.quote(i, 9700 + i * 5) for i in range(60)]
        quotes += [self.quote(60), self.quote(61, 9980), self.quote(62, 9950)]
        rows, summary = self.dataset(quotes)
        self.assertEqual(summary["complete"], 1)
        row = rows[0]
        self.assertEqual((row["entry"], row["exit_price"], row["exit_reason"]), (10010, 9950, "stop"))
        expected = estimate_trade_net_pnl_unrounded(entry_price=10010, exit_price=9950,
            quantity=1, commission_rate=.000142, sell_tax_rate=.0015) / 10010
        self.assertEqual(row["net_return"], expected)
        self.assertEqual(self.dataset(quotes, universe=False)[0], [])

    def test_missing_execution_or_one_sided_book_is_unknown_not_loss_or_win(self):
        history = [self.quote(i, 9700 + i * 5) for i in range(60)] + [self.quote(60)]
        for tail in ([self.quote(66, 10200)], [replace(self.quote(61, 10200), ask_price=0)]):
            rows, summary = self.dataset(history + tail)
            self.assertEqual(summary["complete"], 0)
            self.assertEqual(rows[0]["status"], "unknown")
            self.assertNotIn("net_return", rows[0])

    def test_sampling_is_non_overlapping_and_does_not_use_one_share_rounding(self):
        rows, _ = self.dataset([self.quote(i, 9700 + i * 5) for i in range(450)])
        starts = [datetime.fromisoformat(r["signal_at"]) for r in rows]
        self.assertTrue(all((b-a).total_seconds() >= 185 for a, b in zip(starts, starts[1:])))
        self.assertTrue(all(np.isfinite(r["features"]).all() for r in rows))

    def test_terminal_comparison_keeps_the_same_entry_and_late_gaps_unknown(self):
        history = [self.quote(i, 9700 + i * 5) for i in range(60)] + [self.quote(60)]
        path = history + [self.quote(i, 9980 if i < 65 else 10100) for i in range(61, 242)]
        rows, _ = self.dataset(path)
        first = rows[0]
        self.assertEqual(first["exit_reason"], "stop")
        self.assertLess(first["net_return"], 0)
        self.assertEqual(first["terminal_status"], "complete")
        self.assertGreater(first["terminal_net_return"], 0)
        self.assertEqual(first["terminal_at"], self.quote(241).timestamp.isoformat())
        rows, _ = self.dataset(path[:70] + path[80:])
        self.assertEqual(rows[0]["status"], "complete")
        self.assertEqual(rows[0]["terminal_status"], "unknown")
        self.assertNotIn("terminal_net_return", rows[0])

    def test_price_prediction_applies_query_costs_once_not_training_costs(self):
        features = [0.] * 19
        features[4], features[6], features[14] = .01, .001, .0015
        bid_return = .02
        make_net = lambda tax: (1-.001) * (1+bid_return) * (1-.000142-tax) - 1.000142
        rows = [dict(self.row(i, make_net(.0015), features),
                     commission=.000142, tax=.0015, price_scale=.01) for i in range(100)]
        model = ReturnDistribution(price_move=True).fit(rows, before="2026-10-20")
        query = dict(rows[0], tax=0., features=list(features))
        query["features"][14] = 0.
        result = model.predict([query["features"]], contexts=[query])[0]
        self.assertAlmostEqual(result.expected_return, make_net(0.), places=12)
        self.assertEqual(result.loss_probability, 0.)
        self.assertAlmostEqual(result.lower_return, make_net(0.), places=12)
        with self.assertRaisesRegex(ValueError, "price_context_required"):
            model.predict([features])

    def test_real_trade_flow_uses_cumulative_changes_and_rejects_counter_reset(self):
        history = [replace(self.quote(i, 9700 + i * 5), flow_available=True,
                           cumulative_buy_volume=i*9, cumulative_sell_volume=i,
                           cumulative_trade_amount=i*10000) for i in range(60)]
        q = replace(self.quote(60), flow_available=True, cumulative_buy_volume=540,
                    cumulative_sell_volume=60, cumulative_trade_amount=600000)
        row = observe(history, q, commission=.000142, tax=.0015)
        self.assertEqual(row["features"][15:], [.8, .8, 0., 1.])
        bad = history.copy()
        bad[40] = replace(bad[40], cumulative_buy_volume=0)
        self.assertEqual(observe(bad, q, commission=.000142, tax=.0015)["features"][15:], [0.]*4)
        self.assertEqual(observe(history, replace(q, flow_available=False),
            commission=.000142, tax=.0015)["features"][15:], [0.]*4)

    def test_price_forecast_does_not_multiply_observed_returns_by_query_volatility(self):
        features = [0.] * 19
        features[4], features[6] = .01, .001
        for bid_return in (.02, -.02):
            with self.subTest(bid_return=bid_return):
                net = .999 * (1+bid_return) * (1-.000142-.0015) - 1.000142
                rows = [dict(self.row(i, net, features), commission=.000142,
                             tax=.0015, price_scale=.01) for i in range(100)]
                model = ReturnDistribution(price_move=True).fit(rows, before="2026-10-20")
                query = dict(rows[0], price_scale=.05, features=list(features))
                query["features"][4] = .05
                prediction = model.predict([query["features"]], contexts=[query])[0]
                self.assertAlmostEqual(prediction.expected_return, net)
                self.assertAlmostEqual(prediction.lower_return, net)
                self.assertAlmostEqual(prediction.loss_probability, float(net < 0))

    def test_entry_probes_are_recorded_between_regular_samples(self):
        history = [self.quote(i, 9700 + i * 5) for i in range(60)]
        points = history + [self.quote(60), self.quote(61, 10020), self.quote(62, 10030)]
        rows, _ = self.dataset(points)
        samples = [r for r in rows if r["cohort"] == "sample"]
        entries = [r for r in rows if r["cohort"] == "entry"]
        self.assertEqual(len(samples), 1)
        self.assertEqual(len(entries), 1)
        self.assertNotEqual(samples[0]["signal_at"], entries[0]["signal_at"])
        self.assertTrue(entries[0]["breakout"])
        self.assertGreater(entries[0]["target_net_return"], 0)

    def test_replay_candidate_uses_ev_with_live_sizing_and_daily_controls(self):
        class ConstantEstimate:
            training_through = "2026-10-19T15:00:00"
            def predict(self, features, *, contexts):
                return [ReturnEstimate(.02, .2, -.01) for _ in features]

        market = SimpleNamespace(client=SimpleNamespace(config=SimpleNamespace(is_paper=True)))
        with tempfile.TemporaryDirectory() as directory:
            config = MomentumScalpConfig(daily_state_path=f"{directory}/state.json",
                quote_archive_root=f"{directory}/quotes", commission_rate=.000142, sell_tax_rate=.0015)
            strategy = candidate_strategy(ConstantEstimate())(market, config, fixed_pool=["005930"])
            strategy._quote_archive = MemoryArchive()
            signal = self.quote(60)
            strategy.set_simulated_now(signal.timestamp)
            strategy._ingest_quotes([self.quote(i) for i in range(60)])
            orders = strategy.on_batch_tick([signal])
            self.assertEqual(len(orders), 1)
            order = orders[0]
            self.assertEqual(order.requested_reason, "conditional_return")
            self.assertIsNone(limit_entry_rejection(order, self.quote(61)))
            self.assertIsNotNone(limit_entry_rejection(order, replace(self.quote(61), ask_price=10020)))
            plan = strategy._pending_entry_meta[signal.symbol]["plan"]
            self.assertLessEqual(plan.committed_risk_net_abs, 5000)
            self.assertLessEqual(plan.quantity*plan.entry_limit_price*1.000142, 1_000_000)
            self.assertEqual(strategy.on_batch_tick([signal]), [])
            strategy._pending_entry_meta.clear()
            position = PositionState(order.symbol, order.price, order.quantity, signal.timestamp, plan)
            crossed = self.quote(61, plan.stop_price)
            rebound = self.quote(62)
            strategy.positions[order.symbol] = position
            strategy.set_simulated_now(rebound.timestamp)
            stop, = strategy.on_batch_tick([crossed, rebound])
            self.assertEqual(stop.requested_reason, "breakout_failed")
            self.assertTrue(position.stop_triggered)
            strategy.positions.clear()
            strategy.daily_pnl.realized_net_pnl = -5000
            self.assertEqual(strategy.on_batch_tick([self.quote(61)]), [])
            self.assertEqual(strategy._halt_reason, "daily_total_loss_limit")

    def execution_row(self, index, *, filled=True, buy=10000, sell=10100):
        features = [0.] * 20
        features[4], features[6] = .01, .001
        row = dict(self.row(index, 0., features), cohort="sample", entry=10000,
                   entry_limit=10000, price_scale=.01, commission=0., tax=0., filled=filled)
        if filled:
            row.update(fill_price=buy, exit_price=sell)
        return row

    def test_execution_ev_keeps_entry_exit_paired_and_nonfills_zero(self):
        rows = [self.execution_row(i, filled=i % 3 != 2,
                    buy=9000 if i % 3 == 0 else 10000,
                    sell=9100 if i % 3 == 0 else 10100) for i in range(90)]
        model = ExecutionReturnDistribution().fit(rows, before="2026-10-20")
        result = model.predict([rows[0]])[0]
        self.assertAlmostEqual(result.fill_probability, 2/3)
        self.assertAlmostEqual(result.expected_return_if_filled, .01)
        self.assertAlmostEqual(result.expected_return, .01 * 2/3)
        self.assertEqual(result.loss_probability_if_filled, 0.)
        self.assertAlmostEqual(result.lower_return_if_filled, .01)
        # Independent entry/exit sampling would invent losing 10000 -> 9100 pairs.
        self.assertEqual(model.predict([]), [])

    def test_execution_forecast_uses_query_costs_once_and_preserves_adverse_move(self):
        rows = [self.execution_row(i, buy=9000, sell=8900) for i in range(100)]
        model = ExecutionReturnDistribution().fit(rows, before="2026-10-20")
        query = dict(rows[0], commission=.000142, tax=.0015)
        result = model.predict([query])[0]
        expected = estimate_trade_net_pnl_unrounded(entry_price=9000, exit_price=8900,
            quantity=1, commission_rate=query["commission"], sell_tax_rate=query["tax"]) / 10000
        self.assertAlmostEqual(result.expected_return, expected)
        self.assertAlmostEqual(result.lower_return_if_filled, expected)
        self.assertAlmostEqual(result.loss_probability_if_filled, 1.)

    def test_execution_duplicates_future_and_unknown_do_not_change_distribution(self):
        rows = [self.execution_row(i, filled=i % 2 == 0) for i in range(100)]
        extra = [dict(rows[0], status="unknown"),
                 dict(rows[0], exit_at="2026-10-20T09:00:00"),
                 dict(rows[0], cohort="entry")]
        a = ExecutionReturnDistribution().fit(rows, before="2026-10-20")
        b = ExecutionReturnDistribution().fit(rows*3 + extra, before="2026-10-20")
        self.assertEqual(a.predict([rows[0]]), b.predict([rows[0]]))
        self.assertEqual(b.training_count, 100)
        with self.assertRaisesRegex(ValueError, "conflicting_execution"):
            ExecutionReturnDistribution().fit(rows + [dict(rows[0], exit_price=11000)], before="2026-10-20")

    def test_execution_all_nonfills_has_no_fabricated_conditional_win_rate(self):
        rows = [self.execution_row(i, filled=False) for i in range(20)]
        result = ExecutionReturnDistribution().fit(rows, before="2026-10-20").predict([rows[0]])[0]
        self.assertEqual((result.expected_return, result.fill_probability), (0., 0.))
        self.assertIsNone(result.loss_probability_if_filled)
        self.assertIsNone(result.lower_return_if_filled)

    def test_selection_metrics_do_not_mistake_win_rate_for_profitability(self):
        rows = [dict(status="complete", filled=True, quantity=10, entry_limit=10000,
                     net_return=value, expected_net=100,
                     prediction=dict(loss_probability_if_filled=.3))
                for value in (.001, .001, .001, -.01)]
        rows += [dict(rows[0], status="unknown", net_return=100.),
                 dict(rows[0], filled=False, net_return=0.)]
        result = summarize_execution_predictions(rows)
        self.assertEqual((result["observations"], result["complete"], result["filled"]), (6, 5, 4))
        self.assertEqual((result["wins"], result["losses"]), (3, 1))
        self.assertEqual(result["independent_trade_net_sum"], -700.)
        self.assertEqual(result["average_win_net"], 100.)
        self.assertEqual(result["average_loss_net"], -1000.)
        self.assertEqual(result["mean_net_overprediction"], 240.)

    def test_endpoint_labels_can_observe_a_terminal_price_without_claiming_a_path(self):
        q = self.quote(60)
        row = observe([self.quote(i) for i in range(60)], q, commission=.000142, tax=.0015)
        points = [q, self.quote(70, 10100), self.quote(90, 10200)]
        stream = [({}, p.timestamp, [p]) for p in points]
        rows = label_endpoints(stream, [row], (10, 30))
        self.assertEqual([r["exit_price"] for r in rows], [10100, 10200])
        self.assertTrue(all(r["status"] == "complete" and not r["continuous_path"] for r in rows))
        for item in rows:
            expected = estimate_trade_net_pnl_unrounded(entry_price=q.ask_price, exit_price=item["exit_price"],
                quantity=1, commission_rate=.000142, sell_tax_rate=.0015) / q.ask_price
            self.assertAlmostEqual(item["net_return"], expected)

    def test_endpoint_does_not_use_late_or_one_sided_prices_as_an_exact_horizon(self):
        q = self.quote(60)
        row = observe([self.quote(i) for i in range(60)], q, commission=0., tax=0.)
        for point in (self.quote(76, 20000), replace(self.quote(70, 20000), ask_price=0)):
            stream = [({}, q.timestamp, [q]), ({}, point.timestamp, [point]),
                      ({}, self.quote(80).timestamp, [])]
            result = label_endpoints(stream, [row], (10,))[0]
            self.assertEqual(result["status"], "unknown")
            self.assertNotIn("net_return", result)

    def test_horizon_sampling_is_past_only_and_nonoverlapping_for_all_comparisons(self):
        rows = [dict(self.row(1, value), cohort="sample", signal_at=self.quote(second).timestamp.isoformat())
                for second, value in ((60, -.99), (100, 10.), (1865, .01))]
        selected = select_signals(rows, (180, 900, 1800))
        self.assertEqual([r["signal_at"] for r in selected], [rows[0]["signal_at"], rows[2]["signal_at"]])
        self.assertTrue(all("net_return" not in r and "exit_at" not in r for r in selected))

    def test_lower_entry_limit_changes_fills_not_the_future_exit(self):
        signal = self.quote(60)
        row = dict(observe([self.quote(i) for i in range(60)], signal, commission=.000142, tax=.0015),
                   status="complete", exit_at=self.quote(240).timestamp.isoformat(), exit_price=9800)
        def run(next_quote, endpoint=row):
            stream = [({}, q.timestamp, [q]) for q in (signal, next_quote)]
            return {r["action"]: r for r in label_limit_entries(stream, [endpoint])}
        rows = run(self.quote(61, 9995))
        self.assertEqual(rows["ask"]["fill_price"], 10005)
        self.assertFalse(rows["bid"]["filled"])
        self.assertEqual(rows["bid"]["net_return"], 0.)
        rising = run(self.quote(61, 10020))
        self.assertFalse(rising["ask"]["filled"])
        self.assertFalse(rising["bid"]["filled"])
        self.assertEqual((rising["next_ask"]["fill_price"], rising["next_ask"]["exit_price"]), (10030, 9800))
        for end in (9800, 11000):
            rows = run(self.quote(61, 9980), dict(row, exit_price=end))
            for action in ("ask", "bid"):
                item = rows[action]
                self.assertEqual((item["fill_price"], item["exit_price"]), (9990, end))
                expected = estimate_trade_net_pnl_unrounded(entry_price=9990, exit_price=end,
                    quantity=1, commission_rate=.000142, sell_tax_rate=.0015) / item["entry_limit"]
                self.assertAlmostEqual(item["net_return"], expected)

    def test_limit_entry_missing_quotes_are_unknown_not_unfilled(self):
        signal = self.quote(60)
        row = dict(observe([self.quote(i) for i in range(60)], signal, commission=0., tax=0.),
                   status="unknown", reason="endpoint_missing")
        for quote in (self.quote(66, 9980), replace(self.quote(61, 9980), bid_size=0)):
            rows = label_limit_entries([({}, q.timestamp, [q]) for q in (signal, quote)], [row])
            self.assertTrue(all(r["status"] == "unknown" and "net_return" not in r for r in rows))
        rows = label_limit_entries([({}, q.timestamp, [q]) for q in (signal, self.quote(61, 9995))], [row])
        self.assertEqual(rows[0]["reason"], "endpoint_missing")
        self.assertTrue(rows[0]["filled"])
        self.assertEqual((rows[1]["status"], rows[1]["net_return"]), ("complete", 0.))

    def test_endpoint_replay_preserves_risk_and_daily_stop_but_does_not_use_old_target(self):
        class Model:
            training_through = "2026-10-19T15:00:00"
            def predict(self, features, *, contexts):
                return [ReturnEstimate(.02, .3, -.05) for _ in features]

        market = SimpleNamespace(client=SimpleNamespace(config=SimpleNamespace(is_paper=True)))
        with tempfile.TemporaryDirectory() as directory:
            config = MomentumScalpConfig(daily_state_path=f"{directory}/state.json",
                                         quote_archive_root=f"{directory}/quotes")
            bot = candidate_strategy(Model(), expiry_only=True)(market, config, fixed_pool=["005930"])
            bot._quote_archive = MemoryArchive()
            signal = self.quote(60)
            bot.set_simulated_now(signal.timestamp)
            bot._ingest_quotes([self.quote(i) for i in range(60)])
            order = bot.on_batch_tick([signal])[0]
            plan = bot._pending_entry_meta[order.symbol]["plan"]
            self.assertLessEqual(plan.quantity*plan.entry_limit_price*.05, 5000)
            position = PositionState(order.symbol, order.price, order.quantity, signal.timestamp, plan)
            self.assertIsNone(bot._position_exit_order(position, self.quote(61, 9900), now=self.quote(61).timestamp))
            self.assertIsNone(bot._position_exit_order(position, self.quote(61, 11000), now=self.quote(61).timestamp))
            self.assertEqual(bot._position_exit_order(position, self.quote(240), now=self.quote(240).timestamp)
                             .requested_reason, "holding_expiry")
            bot._pending_entry_meta.clear()
            bot.positions[order.symbol] = position
            bot.set_simulated_now(self.quote(61).timestamp)
            self.assertEqual(bot.on_batch_tick([self.quote(61, 9900)]), [])
            bot.daily_pnl.realized_net_pnl = -5000
            orders = bot.on_batch_tick([self.quote(61, 9900)])
            self.assertEqual(orders[0].requested_reason, "daily_total_loss_limit")


if __name__ == "__main__":
    unittest.main()
