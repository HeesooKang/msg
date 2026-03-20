import json
import tempfile
import unittest
from collections import deque
from datetime import datetime
from pathlib import Path

from src.analytics.math_signals import (
    build_entry_ev_table,
    build_leader_signals,
    compute_regime_probabilities,
)
from src.analytics.quote_tape import QuoteTapeRecorder
from src.models import Quote
from src.strategies.momentum_scalp import MomentumScalpConfig, MomentumScalpStrategy


class MathShadowTests(unittest.TestCase):
    def test_leader_score_ranks_stronger_candidate_higher(self):
        now = datetime(2026, 3, 19, 10, 5, 0)
        strong = Quote(
            symbol="AAA",
            name="AAA",
            current_price=12000,
            change=1000,
            change_rate=9.1,
            open_price=11000,
            high_price=12100,
            low_price=10800,
            volume=500000,
            trade_amount=6_000_000_000,
            timestamp=now,
        )
        weak = Quote(
            symbol="BBB",
            name="BBB",
            current_price=10100,
            change=100,
            change_rate=1.0,
            open_price=10000,
            high_price=10200,
            low_price=9900,
            volume=50000,
            trade_amount=500_000_000,
            timestamp=now,
        )
        recent_quotes = {
            "AAA": deque(
                [
                    Quote("AAA", "AAA", 11000, 0, 0.0, 10800, 11000, 10700, 10000, 100000000, now),
                    Quote("AAA", "AAA", 11800, 0, 0.0, 10800, 11850, 10700, 20000, 200000000, now),
                    Quote("AAA", "AAA", 11550, 0, 0.0, 10800, 11850, 10700, 30000, 300000000, now),
                    strong,
                ],
                maxlen=8,
            ),
            "BBB": deque(
                [
                    Quote("BBB", "BBB", 10000, 0, 0.0, 9950, 10050, 9900, 5000, 50000000, now),
                    weak,
                ],
                maxlen=8,
            ),
        }

        signals = build_leader_signals(
            [strong, weak],
            avg_volumes={"AAA": 150000, "BBB": 80000},
            recent_quotes_by_symbol=recent_quotes,
        )

        self.assertGreater(signals["AAA"].leader_score, signals["BBB"].leader_score)
        self.assertEqual(signals["AAA"].entry_grade, "A")

    def test_regime_probabilities_sum_to_one_and_favor_bull(self):
        probs = compute_regime_probabilities(
            index_gap_ma20_pct=2.3,
            index_gap_ma5_pct=1.4,
            avg_change=1.8,
            decliner_ratio=0.25,
            strong_leader_count=3,
            strong_leader_avg_score=1.7,
        )
        total = probs.bull_prob + probs.neutral_prob + probs.soft_bear_prob + probs.bear_prob
        self.assertAlmostEqual(total, 1.0, places=5)
        self.assertGreater(probs.bull_prob, probs.soft_bear_prob)
        self.assertEqual(probs.dominant_profile(), "bull")

    def test_entry_ev_table_applies_smoothing(self):
        scorecards = [
            {
                "date": "2026-03-17",
                "log_analysis": {
                    "trade_records": [
                        {
                            "strategy_name": "bull_breakout_strategy",
                            "regime_label": "bull",
                            "hour_bucket": "10",
                            "entry_grade_math": "A",
                            "net_pnl": 2000,
                        },
                        {
                            "strategy_name": "bull_breakout_strategy",
                            "regime_label": "bull",
                            "hour_bucket": "11",
                            "entry_grade_math": "A",
                            "net_pnl": -500,
                        },
                    ]
                },
            }
        ]

        table = build_entry_ev_table(scorecards, window_days=5, min_samples=4)
        estimate = table[("bull_breakout_strategy", "bull", "10", "A")]
        self.assertEqual(estimate.confidence, "low")
        self.assertGreater(estimate.p_win, 0.0)

    def test_quote_tape_recorder_writes_quotes_and_leaders(self):
        now = datetime(2026, 3, 19, 10, 10, 0)
        quote = Quote(
            symbol="AAA",
            name="AAA",
            current_price=12000,
            change=1000,
            change_rate=9.1,
            open_price=11000,
            high_price=12100,
            low_price=10800,
            volume=500000,
            trade_amount=6_000_000_000,
            timestamp=now,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            recorder = QuoteTapeRecorder(tmp_dir, enabled=True)
            recorder.record_quotes(now, [quote], regime_label="bull", bear_score=0, market_data_ready=True)
            recorder.record_leaders(
                now,
                event="pool_refresh",
                rows=[{"symbol": "AAA", "leader_score": 1.2, "entry_grade_math": "A"}],
            )

            quote_path = Path(tmp_dir) / "2026" / "03" / "19" / "quotes.ndjson"
            leader_path = Path(tmp_dir) / "2026" / "03" / "19" / "leaders.ndjson"
            self.assertTrue(quote_path.exists())
            self.assertTrue(leader_path.exists())
            quote_row = json.loads(quote_path.read_text(encoding="utf-8").splitlines()[0])
            leader_row = json.loads(leader_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(quote_row["symbol"], "AAA")
            self.assertEqual(leader_row["event"], "pool_refresh")

    def test_strategy_build_entry_metadata_attaches_math_shadow_fields(self):
        cfg = MomentumScalpConfig(enable_math_shadow_layer=True, quote_tape_enabled=False)
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy._latest_regime_probabilities = compute_regime_probabilities(
            index_gap_ma20_pct=1.5,
            index_gap_ma5_pct=0.8,
            avg_change=0.9,
            decliner_ratio=0.35,
            strong_leader_count=1,
            strong_leader_avg_score=1.0,
        )
        now = datetime(2026, 3, 19, 10, 20, 0)
        strategy.set_simulated_now(now)
        quote = Quote(
            symbol="AAA",
            name="AAA",
            current_price=12000,
            change=1000,
            change_rate=9.1,
            open_price=11000,
            high_price=12100,
            low_price=10800,
            volume=500000,
            trade_amount=6_000_000_000,
            timestamp=now,
        )
        strategy._quotes_cache[quote.symbol] = quote
        strategy._recent_quotes[quote.symbol] = deque([quote], maxlen=8)
        strategy._latest_math_leader_signals = build_leader_signals([quote], recent_quotes_by_symbol={quote.symbol: [quote]})
        strategy._entry_ev_table = {
            ("bull_breakout_strategy", "bull", "10", "A"): type(
                "Estimate",
                (),
                {
                    "strategy_name": "bull_breakout_strategy",
                    "regime_label": "bull",
                    "hour_bucket": "10",
                    "entry_grade": "A",
                    "entry_ev": 321.0,
                    "p_win": 0.66,
                    "confidence": "medium",
                    "closed_trades": 5,
                },
            )()
        }
        strategy._bear_score = 0

        metadata = strategy._build_entry_metadata(
            quote.symbol,
            "bull_breakout",
            "setup_name=bull_breakout entry_reason=local_high_breakout",
            strategy_name="bull_breakout_strategy",
            quote=quote,
        )

        self.assertIn("leader_score", metadata)
        self.assertIn("entry_ev", metadata)
        self.assertIn("bull_prob", metadata)

    def test_math_live_gate_blocks_negative_ev_with_enough_trades(self):
        cfg = MomentumScalpConfig(
            enable_math_shadow_layer=True,
            enable_math_live_layer=True,
            quote_tape_enabled=False,
            math_live_ev_min_trades=4,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        quote = Quote(
            symbol="AAA",
            name="AAA",
            current_price=12000,
            change=1000,
            change_rate=9.1,
            open_price=11000,
            high_price=12100,
            low_price=10800,
            volume=500000,
            trade_amount=6_000_000_000,
            timestamp=datetime(2026, 3, 19, 10, 30, 0),
        )
        ok, reason = strategy._passes_math_live_entry_gate(
            quote,
            strategy_name="bull_breakout_strategy",
            entry_meta={
                "leader_percentile": 0.95,
                "entry_ev": -120.0,
                "entry_ev_closed_trades": 5,
                "bull_prob": 0.7,
                "neutral_prob": 0.2,
                "soft_bear_prob": 0.07,
                "bear_prob": 0.03,
            },
            is_inverse=False,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "math_negative_ev")

    def test_math_live_regime_profile_resolves_soft_bear_to_neutral_when_bull_prob_dominates(self):
        cfg = MomentumScalpConfig(
            enable_math_shadow_layer=True,
            enable_math_live_layer=True,
            quote_tape_enabled=False,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy._bear_score = 2
        strategy._latest_regime_probabilities = compute_regime_probabilities(
            index_gap_ma20_pct=2.2,
            index_gap_ma5_pct=1.1,
            avg_change=1.2,
            decliner_ratio=0.35,
            strong_leader_count=3,
            strong_leader_avg_score=1.8,
        )
        self.assertEqual(strategy._resolve_regime_profile_name(), "bull")

    def test_math_live_can_override_disabled_bull_strategy_gate(self):
        cfg = MomentumScalpConfig(
            enable_math_shadow_layer=True,
            enable_math_live_layer=True,
            quote_tape_enabled=False,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        quote = Quote(
            symbol="AAA",
            name="AAA",
            current_price=12000,
            change=1000,
            change_rate=9.1,
            open_price=11000,
            high_price=12100,
            low_price=10800,
            volume=500000,
            trade_amount=6_000_000_000,
            timestamp=datetime(2026, 3, 19, 10, 30, 0),
        )
        strategy._quotes_cache[quote.symbol] = quote
        strategy._recent_quotes[quote.symbol] = deque([quote], maxlen=8)
        strategy._latest_math_leader_signals = build_leader_signals([quote], recent_quotes_by_symbol={quote.symbol: [quote]})
        strategy._latest_regime_probabilities = compute_regime_probabilities(
            index_gap_ma20_pct=2.5,
            index_gap_ma5_pct=1.3,
            avg_change=1.6,
            decliner_ratio=0.30,
            strong_leader_count=3,
            strong_leader_avg_score=1.9,
        )
        allowed, meta = strategy._can_math_live_override_strategy_gate(
            quote,
            strategy_name="bull_breakout_strategy",
            regime_label="neutral",
            is_inverse=False,
        )
        self.assertTrue(allowed)
        self.assertGreater(float(meta.get("bull_prob", 0.0)), 0.0)

    def test_active_pool_quotes_prioritize_math_queue_before_legacy_pool_order(self):
        cfg = MomentumScalpConfig(enable_math_shadow_layer=True, enable_math_live_layer=True, quote_tape_enabled=False)
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        now = datetime(2026, 3, 19, 10, 40, 0)
        for symbol in ["AAA", "BBB", "CCC"]:
            strategy._quotes_cache[symbol] = Quote(
                symbol=symbol,
                name=symbol,
                current_price=10000,
                change=100,
                change_rate=1.0,
                open_price=9900,
                high_price=10100,
                low_price=9800,
                volume=100000,
                trade_amount=1_000_000_000,
                timestamp=now,
            )
        strategy._pool = ["CCC", "BBB", "AAA"]
        strategy._latest_math_queue_symbols = ["AAA"]
        strategy._latest_math_backfill_symbols = ["BBB"]
        strategy._latest_legacy_backfill_symbols = ["CCC"]

        ordered = [quote.symbol for quote in strategy._active_pool_quotes()]
        self.assertEqual(ordered[:3], ["AAA", "BBB", "CCC"])

    def test_math_size_multiplier_scales_with_leader_and_ev_and_caps_by_strategy(self):
        cfg = MomentumScalpConfig(
            enable_math_shadow_layer=True,
            enable_math_live_layer=True,
            quote_tape_enabled=False,
            math_size_min_multiplier=0.70,
            math_size_max_multiplier=1.50,
            math_size_bull_a_max_multiplier=1.65,
            math_ev_scale_krw=2500,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)

        bull_multiplier = strategy._math_size_multiplier(
            strategy_name="bull_breakout_strategy",
            entry_meta={
                "entry_grade": "A",
                "leader_percentile": 0.98,
                "entry_ev": 2500.0,
                "entry_ev_closed_trades": 10,
            },
            is_inverse=False,
        )
        neutral_multiplier = strategy._math_size_multiplier(
            strategy_name="neutral_pullback_strategy",
            entry_meta={
                "entry_grade": "B",
                "leader_percentile": 0.85,
                "entry_ev": 2000.0,
                "entry_ev_closed_trades": 10,
            },
            is_inverse=False,
        )
        low_conf_multiplier = strategy._math_size_multiplier(
            strategy_name="neutral_pullback_strategy",
            entry_meta={
                "entry_grade": "C",
                "leader_percentile": 0.55,
                "entry_ev": -1000.0,
                "entry_ev_closed_trades": 2,
            },
            is_inverse=False,
        )

        self.assertGreater(bull_multiplier, neutral_multiplier)
        self.assertLessEqual(bull_multiplier, 1.65)
        self.assertLessEqual(neutral_multiplier, 1.10)
        self.assertLessEqual(low_conf_multiplier, 1.0)
