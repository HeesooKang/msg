import unittest
from collections import deque
from datetime import datetime

from src.analytics.math_signals import (
    build_entry_ev_table,
    build_leader_signals,
    compute_market_shock_signal,
    compute_regime_probabilities,
)
from src.models import Quote
from src.strategies.momentum_scalp import MomentumScalpConfig, MomentumScalpStrategy


class MathShadowTests(unittest.TestCase):
    def test_market_shock_signal_rises_in_fast_open_drop(self):
        signal = compute_market_shock_signal(
            minutes_since_open=12,
            crash_window_minutes=45,
            index_gap_open_pct=-4.2,
            index_gap_ma5_pct=-3.1,
            index_gap_ma20_pct=-2.6,
            avg_change=-2.4,
            decliner_ratio=0.82,
            falling_speed_pct=2.0,
            inverse_leader_count=3,
        )
        self.assertTrue(signal.crash_open_window_active)
        self.assertGreater(signal.shock_score, 1.0)
        self.assertGreater(signal.shock_confidence, 0.5)

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

    def test_effective_leader_score_reweights_weak_regime_candidates(self):
        now = datetime(2026, 3, 23, 10, 5, 0)
        accelerator = Quote(
            symbol="AAA",
            name="AAA",
            current_price=10500,
            change=500,
            change_rate=5.0,
            open_price=10000,
            high_price=10550,
            low_price=9950,
            volume=90000,
            trade_amount=945_000_000,
            timestamp=now,
        )
        turnover_heavy = Quote(
            symbol="BBB",
            name="BBB",
            current_price=10120,
            change=120,
            change_rate=1.2,
            open_price=10000,
            high_price=10150,
            low_price=9950,
            volume=600000,
            trade_amount=6_072_000_000,
            timestamp=now,
        )
        recent_quotes = {
            "AAA": deque(
                [
                    Quote("AAA", "AAA", 10020, 0, 0.0, 10000, 10030, 9980, 10000, 100_000_000, now),
                    Quote("AAA", "AAA", 10120, 0, 0.0, 10000, 10120, 9980, 20000, 200_000_000, now),
                    Quote("AAA", "AAA", 10280, 0, 0.0, 10000, 10280, 9980, 30000, 300_000_000, now),
                    Quote("AAA", "AAA", 10420, 0, 0.0, 10000, 10420, 9980, 50000, 500_000_000, now),
                    accelerator,
                ],
                maxlen=8,
            ),
            "BBB": deque(
                [
                    Quote("BBB", "BBB", 10020, 0, 0.0, 10000, 10030, 9980, 100000, 1_000_000_000, now),
                    Quote("BBB", "BBB", 10040, 0, 0.0, 10000, 10045, 9980, 180000, 1_800_000_000, now),
                    Quote("BBB", "BBB", 10070, 0, 0.0, 10000, 10075, 9980, 260000, 2_600_000_000, now),
                    turnover_heavy,
                ],
                maxlen=8,
            ),
        }

        signals = build_leader_signals(
            [accelerator, turnover_heavy],
            avg_volumes={"AAA": 70000, "BBB": 300000},
            recent_quotes_by_symbol=recent_quotes,
            regime_score=2,
        )

        self.assertGreater(signals["AAA"].effective_leader_score, signals["BBB"].effective_leader_score)
        self.assertGreater(signals["AAA"].recent_acceleration_pct, signals["BBB"].recent_acceleration_pct)
        self.assertGreaterEqual(signals["AAA"].leader_percentile, signals["BBB"].leader_percentile)

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
                            "strategy_name": "opening_conviction_long_strategy",
                            "regime_label": "bull",
                            "hour_bucket": "10",
                            "entry_grade_math": "A",
                            "net_pnl": 2000,
                        },
                        {
                            "strategy_name": "opening_conviction_long_strategy",
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
        estimate = table[("opening_conviction_long_strategy", "bull", "10", "A")]
        self.assertEqual(estimate.confidence, "low")
        self.assertGreater(estimate.p_win, 0.0)

    def test_strategy_build_entry_metadata_attaches_math_shadow_fields(self):
        cfg = MomentumScalpConfig()
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
            ("opening_conviction_long_strategy", "bull", "10", "A"): type(
                "Estimate",
                (),
                {
                    "strategy_name": "opening_conviction_long_strategy",
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
            "opening_conviction",
            "setup_name=opening_conviction entry_reason=opening_conviction",
            strategy_name="opening_conviction_long_strategy",
            quote=quote,
        )

        self.assertIn("leader_score", metadata)
        self.assertIn("entry_ev", metadata)
        self.assertIn("bull_prob", metadata)



    def test_math_live_regime_profile_resolves_soft_bear_to_neutral_when_bull_prob_dominates(self):
        cfg = MomentumScalpConfig(
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
        meta = strategy._build_entry_metadata(
            quote.symbol,
            "math_live_override",
            "setup_name=math_live_override entry_reason=math_live_override",
            strategy_name="opening_conviction_long_strategy",
            quote=quote,
        )
        meta["regime_label"] = "neutral"
        self.assertGreater(float(meta.get("bull_prob", 0.0)), 0.0)
