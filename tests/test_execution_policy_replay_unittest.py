import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from src.analytics.forecast_outcomes import FORECAST_SCHEMA_VERSION
from src.analytics.price_prediction import (
    PREDICTION_HORIZONS_SECONDS,
    PREDICTION_FEATURES,
    build_feature_rows,
)
from src.models import Quote
from src.strategies.momentum_scalp import MomentumScalpStrategy
from src.strategies.momentum_scalp_types import MomentumScalpConfig


def quote_at(symbol: str, timestamp: datetime, second: int) -> Quote:
    price = 10_000 + second
    return Quote(
        symbol=symbol,
        current_price=price,
        timestamp=timestamp,
        ask_price=price + 5,
        bid_price=price - 5,
    )


class PaperExecutionPolicyReplayTests(unittest.TestCase):
    def test_selected_forecast_has_exactly_one_on_horizon_executable_outcome(self):
        start = datetime(2026, 8, 4, 10, 0, 0)
        symbol = "005930"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            forecast_root = root / "forecasts"
            config = MomentumScalpConfig(
                static_watchlist=[],
                inverse_etfs=[],
                daily_state_path=str(root / "state.json"),
                forecast_outcome_root=str(forecast_root),
            )
            strategy = MomentumScalpStrategy(None, config)
            path = [quote_at(symbol, start + timedelta(seconds=second), second) for second in range(181)]
            feature_row = build_feature_rows(
                [path[-1]],
                recent_quotes_by_symbol={symbol: path},
            )[symbol]
            self.assertTrue(feature_row.ready)
            self.assertTrue(
                all(name in feature_row.compact_features for name in PREDICTION_FEATURES)
            )

            prior_date = "2026-08-03"
            prior_path = forecast_root / "2026" / "08" / f"forecast-outcomes.{prior_date}.json"
            prior_path.parent.mkdir(parents=True, exist_ok=True)
            prior_outcomes = []
            for index, prior_symbol in enumerate(("000001", "000002", "000003", "000004", "000005")):
                signal_at = datetime(2026, 8, 3, 10, index, 0)
                prior_outcomes.append(
                    {
                        "forecast_id": f"prior-{index}",
                        "symbol": prior_symbol,
                        "signal_timestamp": signal_at.isoformat(),
                        "outcome_timestamp": (signal_at + timedelta(seconds=180)).isoformat(),
                        "horizon_seconds": 180,
                        "compact_features": dict(feature_row.compact_features),
                        "actual_net_return_pct": 1.5,
                        "training_eligible": True,
                    }
                )
            prior_path.write_text(
                json.dumps(
                    {
                        "schema_version": FORECAST_SCHEMA_VERSION,
                        "date": prior_date,
                        "pending": [],
                        "outcomes": prior_outcomes,
                    }
                ),
                encoding="utf-8",
            )

            strategy.set_simulated_now(start)
            strategy.initialize()
            selected_order = None
            for second in range(361):
                now = start + timedelta(seconds=second)
                strategy.set_simulated_now(now)
                orders = strategy.on_batch_tick([quote_at(symbol, now, second)])
                if orders and selected_order is None:
                    selected_order = orders[0]

            self.assertIsNotNone(selected_order)
            self.assertEqual(selected_order.symbol, symbol)
            current_path = forecast_root / "2026" / "08" / "forecast-outcomes.2026-08-04.json"
            payload = json.loads(current_path.read_text(encoding="utf-8"))
            selected = [row for row in payload["outcomes"] if row.get("selected")]
            self.assertEqual(len(selected), 1)
            row = selected[0]
            signal_at = datetime.fromisoformat(row["signal_timestamp"])
            outcome_at = datetime.fromisoformat(row["outcome_timestamp"])
            selected_horizon = int(row["horizon_seconds"])
            self.assertIn(selected_horizon, PREDICTION_HORIZONS_SECONDS)
            self.assertEqual(
                (outcome_at - signal_at).total_seconds(),
                selected_horizon,
            )
            signal_second = int((signal_at - start).total_seconds())
            self.assertLess(signal_second, 180)
            self.assertEqual(row["signal_entry_ask"], 10_000 + signal_second + 5)
            self.assertEqual(
                row["outcome_bid"],
                10_000 + signal_second + selected_horizon - 5,
            )
            self.assertTrue(row["training_eligible"])


if __name__ == "__main__":
    unittest.main()
