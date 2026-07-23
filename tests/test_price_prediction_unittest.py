import unittest
from datetime import datetime, timedelta
from pathlib import Path
import tempfile

from src.analytics.forecast_outcomes import ForecastOutcomeLedger, calibrate_walk_forward
from src.analytics.math_signals import LeaderSignal
from src.analytics.price_prediction import predict_short_horizon_return
from src.models import Quote


class PricePredictionTests(unittest.TestCase):
    @staticmethod
    def _quotes(
        prices,
        *,
        symbol="TEST",
        open_price=None,
        interval_seconds=11,
        volumes=None,
    ):
        opening = int(open_price or prices[0])
        start = datetime(2026, 7, 14, 10, 0, 0)
        cumulative_volumes = volumes or [100_000 + index * 10_000 for index in range(len(prices))]
        quotes = []
        for index, (price, volume) in enumerate(zip(prices, cumulative_volumes)):
            quotes.append(
                Quote(
                    symbol=symbol,
                    name=symbol,
                    current_price=price,
                    change=price - opening,
                    change_rate=((price - opening) / opening) * 100.0,
                    open_price=opening,
                    high_price=max(prices[: index + 1]),
                    low_price=min(opening, min(prices[: index + 1])),
                    volume=volume,
                    trade_amount=price * volume,
                    timestamp=start + timedelta(seconds=index * interval_seconds),
                )
            )
        return quotes

    @staticmethod
    def _leader(quote, *, score=0.70, percentile=0.90, acceleration=0.40):
        return LeaderSignal(
            symbol=quote.symbol,
            leader_score=score,
            leader_percentile=percentile,
            entry_grade="A",
            change_rate=quote.change_rate,
            trade_amount=quote.trade_amount,
            vs_open_pct=quote.change_rate,
            high_proximity=quote.current_price / max(1, quote.high_price),
            volume_vs_avg=1.5,
            reclaim_speed_ticks=1,
            recent_acceleration_pct=acceleration,
            effective_leader_score=score,
        )

    def _predict(self, quotes, *, leader=None):
        return predict_short_horizon_return(
            quotes[-1],
            recent_quotes=quotes,
            leader=leader or self._leader(quotes[-1]),
            market_state={"tape_heat": 0.50, "tape_caution": 0.10, "vs_open_p90": 8.0},
            min_samples=5,
        )

    def test_predictor_requires_real_price_history(self):
        quotes = self._quotes([10_000])

        prediction = self._predict(quotes)

        self.assertFalse(prediction.ready)
        self.assertEqual(prediction.reason, "insufficient_samples")

    def test_repeated_price_updates_beat_one_jump_then_flat(self):
        one_jump = self._quotes([330_000, 330_500, 330_500, 330_500, 330_500, 332_000, 332_000])
        confirmed = self._quotes([330_000, 330_300, 330_600, 330_900, 331_200, 331_600, 332_000])
        erratic_tail_burst = self._quotes([10_000, 10_050, 9_980, 10_040, 9_990, 10_000, 10_020, 10_040])
        opening_chase = self._quotes(
            [6_840, 6_840, 6_770, 6_770, 7_050, 7_050, 7_070, 7_070, 7_040, 7_120, 7_120, 7_350],
            open_price=6_220,
        )
        late_chase = self._quotes(
            [4_925, 4_925, 4_855, 4_855, 4_875, 4_875, 4_970, 4_970, 5_000, 5_035, 5_035, 5_080, 5_080, 5_250],
            open_price=4_480,
        )
        for quote in opening_chase:
            quote.high_price = 7_665
        for quote in late_chase:
            quote.high_price = 5_415

        jump_prediction = self._predict(one_jump)
        confirmed_prediction = self._predict(confirmed)
        erratic_prediction = self._predict(erratic_tail_burst)
        opening_chase_prediction = self._predict(
            opening_chase,
            leader=self._leader(opening_chase[-1], score=-0.3946, percentile=0.52, acceleration=-0.20),
        )
        late_chase_prediction = self._predict(
            late_chase,
            leader=self._leader(late_chase[-1], score=0.5520, percentile=1.0, acceleration=0.80),
        )

        self.assertTrue(jump_prediction.ready)
        self.assertLess(jump_prediction.predicted_return_pct, confirmed_prediction.predicted_return_pct)
        self.assertLess(jump_prediction.lower_bound_return_pct, 0.0)
        self.assertGreater(jump_prediction.features["single_tick_impulse_risk"], 0.60)
        self.assertGreater(confirmed_prediction.predicted_return_pct, 0.70)
        self.assertGreater(confirmed_prediction.lower_bound_return_pct, 0.0)
        self.assertGreater(confirmed_prediction.confidence, jump_prediction.confidence)
        self.assertLess(erratic_prediction.confidence, 0.70)
        self.assertGreater(erratic_prediction.features["terminal_trend_fit"], 0.95)
        for prediction in (opening_chase_prediction, late_chase_prediction):
            self.assertGreater(prediction.features["single_tick_impulse_risk"], 0.90)
            self.assertLess(prediction.lower_bound_return_pct, 0.0)
            self.assertLess(prediction.confidence, 0.40)

    def test_predictor_recognizes_confirmed_rebound_before_the_last_jump(self):
        prices = [
            12_320,
            12_230,
            12_290,
            12_280,
            12_260,
            12_210,
            12_160,
            12_110,
            12_100,
            12_210,
            12_200,
            12_190,
            12_220,
            12_270,
            12_220,
            12_230,
            12_290,
            12_380,
            12_590,
            12_610,
        ]
        early = self._quotes(prices[:17], open_price=10_780)
        confirmed = self._quotes(prices[:18], open_price=10_780)

        early_prediction = self._predict(early)
        confirmed_prediction = self._predict(confirmed)

        self.assertLess(early_prediction.predicted_return_pct, 0.60)
        self.assertGreater(confirmed_prediction.predicted_return_pct, 0.90)
        self.assertGreater(confirmed_prediction.lower_bound_return_pct, 0.0)
        self.assertGreater(confirmed_prediction.features["positive_move_support"], 0.90)
        self.assertLess(confirmed_prediction.features["trap_risk_score"], 0.20)

    def test_sparse_observations_keep_lower_bound_negative(self):
        prices = [10_000, 10_020, 10_050, 10_090, 10_140, 10_200]
        dense = self._quotes(prices, interval_seconds=12)
        sparse = self._quotes(prices, interval_seconds=75)

        dense_prediction = self._predict(dense)
        sparse_prediction = self._predict(sparse)

        self.assertGreater(sparse_prediction.features["quote_gap_risk"], 0.60)
        self.assertLess(sparse_prediction.lower_bound_return_pct, dense_prediction.lower_bound_return_pct)
        self.assertLess(sparse_prediction.predicted_return_pct, dense_prediction.predicted_return_pct)
        self.assertLess(sparse_prediction.confidence, dense_prediction.confidence)

    def test_latest_two_step_reversal_overrides_sparse_older_jump(self):
        quotes = self._quotes(
            [38_300, 38_300, 38_350, 38_350, 40_350, 40_450, 40_300, 40_150],
            open_price=38_300,
        )
        start = quotes[0].timestamp
        for quote, offset in zip(quotes, [0, 22, 43, 54, 281, 292, 329, 358]):
            quote.timestamp = start + timedelta(seconds=offset)

        prediction = self._predict(quotes)

        self.assertGreater(prediction.features["fast_trend_pct_per_second"], 0.0)
        self.assertLess(prediction.features["terminal_trend_pct_per_second"], 0.0)
        self.assertLess(prediction.predicted_return_pct, 0.0)
        self.assertLess(prediction.lower_bound_return_pct, prediction.predicted_return_pct)

    def test_sustained_high_hold_remains_positive(self):
        prices = [10_000, 10_200, 10_350, 10_420, 10_420, 10_450, 10_450, 10_470]
        volumes = [100_000, 120_000, 150_000, 190_000, 240_000, 300_000, 370_000, 450_000]
        quotes = self._quotes(prices, volumes=volumes)

        prediction = self._predict(quotes)

        self.assertGreater(prediction.predicted_return_pct, 0.70)
        self.assertGreater(prediction.lower_bound_return_pct, 0.0)
        self.assertGreater(prediction.features["confirmed_high_hold_continuation_score"], 0.30)
        self.assertGreater(prediction.confidence, 0.60)

    def test_opening_rally_that_stalls_is_not_extrapolated_as_continuation(self):
        quotes = self._quotes(
            [1_150, 1_195, 1_219, 1_219, 1_219],
            open_price=1_109,
            interval_seconds=22,
        )

        prediction = self._predict(
            quotes,
            leader=self._leader(quotes[-1], score=0.302476, percentile=0.92, acceleration=0.40),
        )

        self.assertLess(prediction.predicted_return_pct, 0.20)
        self.assertLess(prediction.lower_bound_return_pct, -1.0)
        self.assertGreater(prediction.features["effective_deceleration_score"], 0.60)
        self.assertLess(prediction.confidence, 0.45)

    def test_late_acceleration_after_large_prior_extension_inflates_downside(self):
        prices = [4_875, 4_875, 4_875, 4_880, 4_890, 4_900, 5_010, 5_010, 5_010, 5_050, 5_100, 5_140]
        quotes = self._quotes(prices, open_price=4_000)
        unextended_quotes = self._quotes(prices, open_price=4_875)

        prediction = self._predict(
            quotes,
            leader=self._leader(quotes[-1], score=0.2212, percentile=0.92, acceleration=0.40),
        )
        unextended_prediction = self._predict(
            unextended_quotes,
            leader=self._leader(unextended_quotes[-1], score=0.2212, percentile=0.92, acceleration=0.40),
        )

        self.assertGreater(prediction.features["late_extension_risk"], 0.70)
        self.assertLess(prediction.lower_bound_return_pct, unextended_prediction.lower_bound_return_pct)
        self.assertLess(prediction.confidence, unextended_prediction.confidence)

    def test_rebound_far_below_intraday_high_is_not_scored_as_fresh_breakout(self):
        quotes = self._quotes(
            [2_380, 2_390, 2_410, 2_430, 2_460, 2_490, 2_525],
            open_price=1_926,
        )
        for quote in quotes:
            quote.high_price = 3_105
            quote.low_price = 1_900
        fresh_breakout_quotes = self._quotes(
            [2_380, 2_390, 2_410, 2_430, 2_460, 2_490, 2_525],
            open_price=1_926,
        )
        for quote in fresh_breakout_quotes:
            quote.high_price = 2_525
            quote.low_price = 1_900

        prediction = self._predict(
            quotes,
            leader=self._leader(quotes[-1], score=0.266, percentile=0.88, acceleration=0.40),
        )
        fresh_breakout_prediction = self._predict(
            fresh_breakout_quotes,
            leader=self._leader(fresh_breakout_quotes[-1], score=0.266, percentile=0.88, acceleration=0.40),
        )

        self.assertLess(prediction.predicted_return_pct, fresh_breakout_prediction.predicted_return_pct)
        self.assertLess(prediction.lower_bound_return_pct, 0.0)
        self.assertGreater(prediction.features["rejection_risk_score"], 0.35)

    def test_downtrend_is_negative_even_with_strong_leader_metadata(self):
        quotes = self._quotes([10_000, 9_980, 9_950, 9_910, 9_880, 9_840, 9_800])
        leader = self._leader(quotes[-1], score=1.10, percentile=1.0, acceleration=0.80)

        prediction = self._predict(quotes, leader=leader)

        self.assertLess(prediction.predicted_return_pct, 0.0)
        self.assertLess(prediction.lower_bound_return_pct, prediction.predicted_return_pct)
        self.assertLess(prediction.direction_score, 0.20)

    def test_leader_quality_adjusts_but_does_not_replace_price_evidence(self):
        quotes = self._quotes([10_000, 10_040, 10_090, 10_150, 10_220, 10_300, 10_390])
        strong = self._predict(
            quotes,
            leader=self._leader(quotes[-1], score=1.0, percentile=0.98, acceleration=0.70),
        )
        weak = self._predict(
            quotes,
            leader=self._leader(quotes[-1], score=-0.05, percentile=0.80, acceleration=-0.20),
        )

        self.assertGreaterEqual(strong.predicted_return_pct, weak.predicted_return_pct)
        self.assertGreater(strong.confidence, weak.confidence)
        self.assertGreater(weak.predicted_return_pct, 0.0)
        self.assertGreater(weak.features["positive_move_support"], 0.90)

    def test_flow_spike_inflates_uncertainty_instead_of_direction(self):
        prices = [69_700, 69_900, 69_900, 70_800, 71_700]
        normal = self._quotes(prices, volumes=[100_000, 110_000, 120_000, 130_000, 140_000])
        spike = self._quotes(prices, volumes=[100_000, 110_000, 120_000, 130_000, 900_000])

        normal_prediction = self._predict(normal)
        spike_prediction = self._predict(spike)

        self.assertGreater(spike_prediction.features["tail_flow_spike_risk"], 0.30)
        self.assertLess(spike_prediction.lower_bound_return_pct, normal_prediction.lower_bound_return_pct)
        self.assertLess(spike_prediction.confidence, normal_prediction.confidence)

    def test_walk_forward_uses_gross_forecast_error_and_current_execution_cost(self):
        outcomes = [
            {
                "horizon_seconds": 180,
                "strategy_name": "intraday_conviction_long_strategy",
                "signal_timestamp": "2026-07-20T10:00:00",
                "symbol": f"10000{index}",
                "raw_win_probability": 0.6,
                "raw_predicted_return_pct": 0.5,
                "actual_return_pct": 0.6,
                "actual_net_return_pct": -5.0,
                "round_trip_cost_pct": 5.6,
                "profitable": False,
            }
            for index in range(4)
        ]
        outcomes.extend(
            {
                "horizon_seconds": 180,
                "strategy_name": "opening_conviction_long_strategy",
                "signal_timestamp": "2026-07-20T09:00:00",
                "symbol": f"20000{index}",
                "raw_win_probability": 0.6,
                "raw_predicted_return_pct": 0.5,
                "actual_return_pct": -3.0,
            }
            for index in range(4)
        )

        calibration = calibrate_walk_forward(
            outcomes,
            raw_win_probability=0.6,
            raw_return_pct=0.5,
            horizon_seconds=180,
            round_trip_cost_pct=0.43,
            strategy_name="intraday_conviction_long_strategy",
        )

        self.assertEqual(calibration.sample_count, 4)
        self.assertGreater(calibration.calibrated_win_probability, 0.6)
        self.assertGreater(calibration.calibrated_return_pct, 0.5)

        unsupported = calibrate_walk_forward(
            outcomes,
            raw_win_probability=0.9,
            raw_return_pct=3.0,
            horizon_seconds=180,
            round_trip_cost_pct=0.43,
            strategy_name="intraday_conviction_long_strategy",
        )
        self.assertAlmostEqual(unsupported.calibrated_win_probability, 0.9, delta=0.02)
        self.assertAlmostEqual(unsupported.calibrated_return_pct, 3.0, delta=0.02)

        corrected = calibrate_walk_forward(
            [
                {
                    "horizon_seconds": 180,
                    "strategy_name": "intraday_conviction_long_strategy",
                    "raw_win_probability": 0.6,
                    "raw_predicted_return_pct": 0.5,
                    "actual_return_pct": -1.0,
                    "symbol": f"LOSS{index}",
                }
                for index in range(4)
            ],
            raw_win_probability=0.6,
            raw_return_pct=0.5,
            horizon_seconds=180,
            round_trip_cost_pct=0.43,
            strategy_name="intraday_conviction_long_strategy",
        )
        self.assertLess(corrected.calibrated_return_pct, 0.0)

    def test_forecast_ledger_calibrates_from_similar_prior_day_outcomes_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger = ForecastOutcomeLedger(Path(tmpdir))
            first_signal_at = datetime(2026, 7, 20, 10, 0, 0)
            symbols = ["100001", "100002", "100003", "100004"]
            for symbol in symbols:
                payload = {
                    "symbol": symbol,
                    "signal_timestamp": first_signal_at.isoformat(timespec="seconds"),
                    "signal_price": 10_000,
                    "horizon_seconds": 180,
                    "round_trip_cost_pct": 0.4,
                    "raw_win_probability": 0.8,
                    "raw_predicted_return_pct": 1.5,
                }
                self.assertTrue(ledger.record(payload, now=first_signal_at, selected=False))

            outcome_at = first_signal_at + timedelta(seconds=180)
            outcome_quotes = []
            for symbol in symbols:
                outcome_quote = self._quotes([9_800], symbol=symbol)[0]
                outcome_quote.timestamp = outcome_at
                outcome_quotes.append(outcome_quote)
            settled = ledger.settle(outcome_quotes, now=outcome_at)
            self.assertEqual(len(settled), 4)

            next_day = datetime(2026, 7, 21, 10, 0, 0)
            calibration = ledger.calibrate(
                as_of=next_day,
                raw_win_probability=0.8,
                raw_return_pct=1.5,
                horizon_seconds=180,
                round_trip_cost_pct=0.4,
            )
            self.assertEqual(calibration.sample_count, 4)
            self.assertLess(calibration.calibrated_win_probability, 0.8)
            self.assertLess(calibration.calibrated_return_pct, 1.5)

            current_payload = {
                "symbol": "100005",
                "signal_timestamp": next_day.isoformat(timespec="seconds"),
                "signal_price": 10_000,
                "horizon_seconds": 180,
                "round_trip_cost_pct": 0.4,
                "raw_win_probability": 0.8,
                "raw_predicted_return_pct": 1.5,
            }
            ledger.record(current_payload, now=next_day, selected=False)
            current_outcome = self._quotes([10_500], symbol="100005")[0]
            current_outcome.timestamp = next_day + timedelta(seconds=180)
            ledger.settle([current_outcome], now=current_outcome.timestamp)
            same_day_calibration = ledger.calibrate(
                as_of=current_outcome.timestamp,
                raw_win_probability=0.8,
                raw_return_pct=1.5,
                horizon_seconds=180,
                round_trip_cost_pct=0.4,
            )
            self.assertEqual(same_day_calibration, calibration)



if __name__ == "__main__":
    unittest.main()
