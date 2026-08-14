import json
import tempfile
import unittest
from datetime import datetime, timedelta
from math import log, sqrt
from pathlib import Path

from src.analytics.forecast_outcomes import FORECAST_SCHEMA_VERSION, ForecastOutcomeLedger
from src.analytics.price_prediction import (
    MODEL_FEATURES,
    PREDICTION_FEATURES,
    FeatureRow,
    RidgeModel,
    build_feature_rows,
    fit_model,
    predict_batch,
)
from src.models import Quote
from src.strategies.momentum_scalp_pnl import estimate_trade_net_pnl_unrounded


def make_quote(
    symbol: str,
    timestamp: datetime,
    price: int,
    *,
    cumulative_volume: int = 0,
    cumulative_sell_volume: int = 0,
    cumulative_buy_volume: int = 0,
    total_ask_size: int = 0,
    total_bid_size: int = 0,
    flow_available: bool = True,
    book_depth_available: bool = True,
) -> Quote:
    return Quote(
        symbol=symbol,
        current_price=price,
        timestamp=timestamp,
        ask_price=price + 5,
        bid_price=price - 5,
        cumulative_volume=cumulative_volume,
        cumulative_sell_volume=cumulative_sell_volume,
        cumulative_buy_volume=cumulative_buy_volume,
        total_ask_size=total_ask_size,
        total_bid_size=total_bid_size,
        book_available=True,
        flow_available=flow_available,
        book_depth_available=book_depth_available,
    )


def feature_values(value: float = 0.1) -> dict[str, float]:
    values = {name: 0.0 for name in PREDICTION_FEATURES}
    values.update(
        {
            "return_15s_pct": value,
            "return_60s_pct": value * 2.0,
            "return_180s_pct": value * 3.0,
            "net_buy_volume_ratio": value,
            "book_imbalance": value * 0.5,
            "relative_60s_pct": value,
            "spread_pct": 0.05,
        }
    )
    return values


def outcome(
    symbol: str,
    signal_at: datetime,
    actual: float,
    value: float = 0.1,
    *,
    horizon_seconds: int = 180,
):
    return {
        "symbol": symbol,
        "signal_timestamp": signal_at.isoformat(),
        "outcome_timestamp": (
            signal_at + timedelta(seconds=horizon_seconds)
        ).isoformat(),
        "horizon_seconds": horizon_seconds,
        "compact_features": feature_values(value),
        "actual_net_return_pct": actual,
        "training_eligible": True,
    }


class SimplePredictionTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 8, 5, 10, 0, 0)

    def test_features_are_time_based_price_flow_and_relative_strength(self):
        start = self.now - timedelta(seconds=180)
        history = [
            make_quote(
                "005930",
                start + timedelta(seconds=second),
                10_000 + second,
                cumulative_volume=second * 100,
                cumulative_sell_volume=second * 100,
                cumulative_buy_volume=second * 200,
                total_ask_size=1_000,
                total_bid_size=2_000,
            )
            for second in range(181)
        ]
        future = make_quote("005930", self.now + timedelta(seconds=30), 20_000)

        row = build_feature_rows(
            [history[-1]],
            recent_quotes_by_symbol={"005930": [*history, future]},
        )["005930"]

        self.assertTrue(row.ready)
        self.assertEqual(tuple(row.compact_features), PREDICTION_FEATURES)
        self.assertEqual(row.unavailable_features, ())
        self.assertAlmostEqual(
            row.compact_features["return_15s_pct"],
            ((10_180 / 10_165) - 1.0) * 100.0,
            places=8,
        )
        self.assertAlmostEqual(row.compact_features["relative_60s_pct"], 0.0)
        self.assertAlmostEqual(row.compact_features["net_buy_volume_ratio"], 1.0 / 3.0)
        self.assertAlmostEqual(row.compact_features["book_imbalance"], 1.0 / 3.0)

    def test_rest_microstructure_placeholders_are_marked_unavailable(self):
        past = make_quote(
            "005930",
            self.now - timedelta(seconds=60),
            10_000,
            flow_available=False,
            book_depth_available=False,
        )
        current = make_quote(
            "005930",
            self.now,
            10_010,
            flow_available=False,
            book_depth_available=False,
        )

        row = build_feature_rows(
            [current],
            recent_quotes_by_symbol={"005930": [past, current]},
        )["005930"]

        self.assertIn("net_buy_volume_ratio", row.unavailable_features)
        self.assertIn("book_imbalance", row.unavailable_features)
        self.assertIn("return_15s_pct", row.unavailable_features)

    def test_short_history_is_ready_without_a_static_history_gate(self):
        history = [
            make_quote("005930", self.now - timedelta(seconds=second), 10_000)
            for second in range(61)
        ]

        row = build_feature_rows(
            [history[0]],
            recent_quotes_by_symbol={"005930": history},
        )["005930"]

        self.assertTrue(row.ready)
        self.assertEqual(tuple(row.compact_features), PREDICTION_FEATURES)
        self.assertIn("return_180s_pct", row.unavailable_features)

    def test_directional_flow_uses_actual_buy_and_sell_volume_deltas(self):
        past = make_quote(
            "005930",
            self.now - timedelta(seconds=60),
            10_000,
            cumulative_sell_volume=1_000,
            cumulative_buy_volume=2_000,
        )
        current = make_quote(
            "005930",
            self.now,
            10_000,
            cumulative_sell_volume=1_500,
            cumulative_buy_volume=3_500,
            total_ask_size=300,
            total_bid_size=700,
        )

        row = build_feature_rows(
            [current],
            recent_quotes_by_symbol={"005930": [past, current]},
        )["005930"]

        self.assertAlmostEqual(row.compact_features["net_buy_volume_ratio"], 0.5)
        self.assertAlmostEqual(row.compact_features["book_imbalance"], 0.4)

    def test_training_uses_settled_selected_rows_but_ignores_future_and_incomplete_rows(self):
        valid = outcome("005930", self.now - timedelta(minutes=10), 0.4)
        future = outcome("000660", self.now, 5.0)
        selected = outcome("003670", self.now - timedelta(minutes=10), 5.0)
        selected["selected"] = True
        incomplete = outcome("035420", self.now - timedelta(minutes=10), 5.0)
        incomplete["compact_features"].pop("spread_pct")
        incompatible = outcome("051910", self.now - timedelta(minutes=10), 5.0)
        incompatible["compact_features"]["legacy_score"] = 99.0

        model = fit_model(
            [valid, future, selected, incomplete, incompatible],
            as_of=self.now,
        )

        self.assertIsNotNone(model)
        self.assertEqual(model.sample_count, 2)

    def test_unfinished_current_day_groups_receive_only_observed_day_fraction(self):
        records = []
        for symbol_index in range(3):
            for observation_index in range(8):
                records.append(
                    outcome(
                        f"{symbol_index + 1:06d}",
                        self.now
                        - timedelta(days=1, minutes=observation_index + 10),
                        -0.2,
                    )
                )
        for symbol_index in range(4):
            records.append(
                outcome(
                    f"{symbol_index + 100:06d}",
                    self.now - timedelta(minutes=10),
                    5.0,
                )
            )

        model = fit_model(records, as_of=self.now)

        self.assertIsNotNone(model)
        self.assertEqual(model.sample_count, 28)
        self.assertAlmostEqual(model.effective_sample_size, 3.5)

    def test_all_negative_history_cannot_be_extrapolated_positive(self):
        records = [
            outcome(
                f"{index + 1:06d}",
                self.now - timedelta(minutes=20 - index),
                actual,
                value=float(index + 1),
            )
            for index, actual in enumerate((-0.38, -0.41, -0.20, -1.36))
        ]
        model = fit_model(records, as_of=self.now)
        symbol = "005930"
        row = FeatureRow(
            ready=True,
            reason="ok",
            compact_features=feature_values(-100.0),
        )

        prediction = predict_batch(
            {symbol: row},
            model=model,
            evaluated_candidate_count=40,
        )[symbol]

        self.assertLess(prediction.expected_net_return_pct, 0.0)

    def test_one_extreme_outcome_does_not_dominate_the_ridge_target(self):
        records = [
            outcome(
                f"{index + 1:06d}",
                self.now - timedelta(minutes=10),
                -0.2,
            )
            for index in range(99)
        ]
        records.append(
            outcome("999999", self.now - timedelta(minutes=10), 30.0)
        )
        model = fit_model(records, as_of=self.now)
        prediction = predict_batch(
            {"005930": FeatureRow(True, "ok", feature_values(0.0))},
            model=model,
            evaluated_candidate_count=40,
        )["005930"]

        self.assertTrue(prediction.ready)
        self.assertLess(model.target_max_pct, 30.0)
        self.assertGreater(model.residual_rms_pct, 1.0)
        self.assertLess(prediction.expected_net_return_pct, 0.0)

    def test_all_horizons_share_one_model_instead_of_tiny_separate_models(self):
        records = [
            outcome(
                f"{index + 1:06d}",
                self.now - timedelta(minutes=30 - index),
                -0.4,
            )
            for index in range(12)
        ]
        records.append(
            outcome(
                "005930",
                self.now - timedelta(minutes=5),
                1.0,
                horizon_seconds=60,
            )
        )

        model = fit_model(records, as_of=self.now)
        prediction = predict_batch(
            {"005930": FeatureRow(True, "ok", feature_values(0.1))},
            model=model,
            evaluated_candidate_count=40,
            horizon_seconds=60,
        )["005930"]

        self.assertEqual(model.sample_count, len(records))
        self.assertEqual(prediction.sample_count, len(records))
        self.assertLess(prediction.expected_net_return_pct, 0.0)

    def test_candidate_count_reduces_the_same_model_estimate(self):
        model = RidgeModel(
            coefficients=(0.5,) + (0.0,) * len(MODEL_FEATURES),
            medians=(0.0,) * len(MODEL_FEATURES),
            scales=(1.0,) * len(MODEL_FEATURES),
            feature_mins=(-1.0,) * len(MODEL_FEATURES),
            feature_maxs=(1.0,) * len(MODEL_FEATURES),
            sample_count=20,
            effective_sample_size=20,
            target_min_pct=-1.0,
            target_max_pct=1.0,
            residual_rms_pct=0.2,
            residual_p10_pct=-0.3,
        )
        symbol = "005930"
        row = FeatureRow(
            ready=True,
            reason="ok",
            compact_features={name: 0.0 for name in PREDICTION_FEATURES},
        )

        single = predict_batch({symbol: row}, model=model, evaluated_candidate_count=1)[symbol]
        wide = predict_batch({symbol: row}, model=model, evaluated_candidate_count=40)[symbol]

        self.assertEqual(single.expected_net_return_pct, 0.5)
        self.assertLess(wide.expected_net_return_pct, single.expected_net_return_pct)
        self.assertAlmostEqual(
            wide.expected_net_return_pct,
            0.5
            - (0.2 / sqrt(20.0))
            * sqrt(20.0 / 18.0)
            * sqrt(2.0 * log(40.0)),
            places=8,
        )

    def test_model_learns_actual_net_buy_pressure_without_price_momentum(self):
        records = []
        for index in range(20):
            positive = index % 2 == 0
            record = outcome(
                f"{index + 1:06d}",
                self.now - timedelta(minutes=30 - index),
                0.6 if positive else -0.6,
                value=0.0,
            )
            record["compact_features"]["net_buy_volume_ratio"] = 0.8 if positive else -0.8
            record["compact_features"]["book_imbalance"] = 0.5 if positive else -0.5
            records.append(record)
        model = fit_model(records, as_of=self.now)
        buy_pressure = feature_values(0.0)
        buy_pressure["net_buy_volume_ratio"] = 0.8
        buy_pressure["book_imbalance"] = 0.5
        sell_pressure = feature_values(0.0)
        sell_pressure["net_buy_volume_ratio"] = -0.8
        sell_pressure["book_imbalance"] = -0.5

        predictions = predict_batch(
            {
                "005930": FeatureRow(True, "ok", buy_pressure),
                "000660": FeatureRow(True, "ok", sell_pressure),
            },
            model=model,
            evaluated_candidate_count=1,
        )

        self.assertGreater(predictions["005930"].expected_net_return_pct, 0.0)
        self.assertLess(predictions["000660"].expected_net_return_pct, 0.0)

    def test_prediction_clamps_features_to_observed_training_range(self):
        size = len(MODEL_FEATURES)
        model = RidgeModel(
            coefficients=(0.0, 1.0) + (0.0,) * (size - 1),
            medians=(0.0,) * size,
            scales=(1.0,) * size,
            feature_mins=(-1.0,) * size,
            feature_maxs=(1.0,) * size,
            sample_count=20,
            effective_sample_size=20,
            target_min_pct=-200.0,
            target_max_pct=200.0,
            residual_rms_pct=0.0,
            residual_p10_pct=-0.2,
        )
        values = feature_values()
        values["return_15s_pct"] = 100.0
        row = FeatureRow(True, "ok", values)

        result = predict_batch(
            {"005930": row},
            model=model,
            evaluated_candidate_count=1,
        )["005930"]

        self.assertEqual(result.expected_net_return_pct, 1.0)

    def test_zero_mad_feature_uses_observed_dispersion_scale(self):
        records = [
            outcome(
                f"{index + 1:06d}",
                self.now - timedelta(minutes=20 - index),
                actual=float(index) / 10.0,
                value=0.0 if index < 4 else 0.5,
            )
            for index in range(5)
        ]

        model = fit_model(records, as_of=self.now)

        self.assertGreater(model.scales[0], 0.01)

    def test_no_outcomes_is_not_ready(self):
        symbol = "005930"
        row = FeatureRow(
            ready=True,
            reason="ok",
            compact_features=feature_values(),
        )

        prediction = predict_batch(
            {symbol: row},
            model=None,
            evaluated_candidate_count=1,
        )[symbol]

        self.assertFalse(prediction.ready)
        self.assertEqual(prediction.reason, "no_settled_outcomes")


class ForecastLedgerTests(unittest.TestCase):
    def test_multi_horizon_observations_share_one_non_overlapping_signal(self):
        signal_at = datetime(2026, 8, 5, 10, 0, 0)
        with tempfile.TemporaryDirectory() as directory:
            ledger = ForecastOutcomeLedger(directory)
            payloads = [
                {
                    "symbol": "005930",
                    "signal_timestamp": signal_at.isoformat(),
                    "signal_entry_ask": 10_010,
                    "horizon_seconds": horizon,
                    "compact_features": feature_values(),
                    "commission_rate": 0.00015,
                    "sell_tax_rate": 0.002,
                }
                for horizon in (30, 60, 120, 180)
            ]

            first = ledger.record_observation_sets(payloads, now=signal_at)
            duplicate = ledger.record_observation_sets(
                payloads,
                now=signal_at + timedelta(seconds=31),
            )
            settled = ledger.settle(
                [
                    make_quote("005930", signal_at + timedelta(seconds=horizon), 10_100)
                    for horizon in (30, 60, 120, 180)
                ],
                now=signal_at + timedelta(seconds=180),
            )

            self.assertEqual(set(horizon for _, horizon in first), {30, 60, 120, 180})
            self.assertEqual(duplicate, {})
            self.assertEqual(
                [item["horizon_seconds"] for item in settled],
                [30, 60, 120, 180],
            )

    def test_incompatible_old_outcome_is_kept_but_not_trained(self):
        now = datetime(2026, 8, 5, 10, 0, 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "2026" / "08" / "forecast-outcomes.2026-08-05.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            legacy = {
                "symbol": "005930",
                "signal_timestamp": now.isoformat(),
                "outcome_timestamp": (now + timedelta(seconds=180)).isoformat(),
                "raw_expected_return_pct": 3.0,
            }
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "date": "2026-08-05",
                        "pending": [],
                        "outcomes": [legacy],
                    }
                ),
                encoding="utf-8",
            )
            ledger = ForecastOutcomeLedger(directory)
            ledger.record(
                {
                    "symbol": "000660",
                    "signal_timestamp": now.isoformat(),
                    "signal_entry_ask": 10_010,
                    "signal_exit_bid": 9_990,
                    "horizon_seconds": 180,
                    "compact_features": feature_values(),
                },
                now=now,
                selected=False,
            )

            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["schema_version"], FORECAST_SCHEMA_VERSION)
            self.assertEqual(saved["outcomes"][0], legacy)
            self.assertIsNone(
                fit_model(
                    ledger.training_outcomes(as_of=now + timedelta(minutes=10)),
                    as_of=now + timedelta(minutes=10),
                )
            )

    def test_signal_ask_settles_at_180_second_bid_with_costs(self):
        signal_at = datetime(2026, 8, 5, 10, 0, 0)
        with tempfile.TemporaryDirectory() as directory:
            ledger = ForecastOutcomeLedger(directory)
            ledger.record(
                {
                    "symbol": "005930",
                    "signal_timestamp": signal_at.isoformat(),
                    "signal_entry_ask": 10_010,
                    "signal_exit_bid": 9_990,
                    "horizon_seconds": 180,
                    "compact_features": feature_values(),
                    "commission_rate": 0.00015,
                    "sell_tax_rate": 0.002,
                },
                now=signal_at,
                selected=True,
            )
            settled = ledger.settle(
                [make_quote("005930", signal_at + timedelta(seconds=180), 10_100)],
                now=signal_at + timedelta(seconds=180),
            )

            self.assertEqual(len(settled), 1)
            expected_net = estimate_trade_net_pnl_unrounded(
                entry_price=10_010,
                exit_price=10_095,
                quantity=1,
                commission_rate=0.00015,
                sell_tax_rate=0.002,
            )
            self.assertAlmostEqual(
                settled[0]["actual_net_return_pct"],
                expected_net / 10_010 * 100.0,
                places=8,
            )


if __name__ == "__main__":
    unittest.main()
