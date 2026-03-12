import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from src.performance_reporting import (
    build_daily_scorecard,
    evaluate_real_trading_readiness,
    update_performance_reports,
)


class PerformanceReportingTests(unittest.TestCase):
    def test_build_daily_scorecard_computes_trade_metrics(self):
        strategy = SimpleNamespace(
            daily_pnl=SimpleNamespace(
                realized_gross_pnl=4_500,
                realized_net_pnl=3_200,
                fees_paid=200,
                taxes_paid=1_100,
                trade_count=4,
                win_count=3,
                loss_count=1,
                breakeven_count=0,
                winning_net_pnl_sum=4_400,
                losing_net_pnl_sum=-1_200,
                largest_win_net=2_100,
                largest_loss_net=-1_200,
            ),
            positions={"005930": object()},
            _halted=False,
        )
        balance = SimpleNamespace(
            total_eval_amount=9_999_000,
            total_deposit=1_250_000,
            total_profit_loss=12_345,
            total_profit_rate=1.234,
        )

        scorecard = build_daily_scorecard(
            strategy=strategy,
            balance=balance,
            session_pnl=3_200,
            trading_mode="paper",
            generated_at=datetime(2026, 3, 11, 15, 20, 0),
        )

        self.assertEqual(scorecard["date"], "2026-03-11")
        self.assertEqual(scorecard["pnl"]["realized_net_pnl"], 3_200)
        self.assertEqual(scorecard["trades"]["closed_trades"], 4)
        self.assertEqual(scorecard["trades"]["wins"], 3)
        self.assertAlmostEqual(scorecard["trades"]["win_rate"], 0.75)
        self.assertAlmostEqual(scorecard["trades"]["profit_factor"], 3.6667, places=4)
        self.assertEqual(scorecard["strategy"]["open_positions_count"], 1)

    def test_update_performance_reports_writes_scorecard_and_gate_files(self):
        strategy = SimpleNamespace(
            daily_pnl=SimpleNamespace(
                realized_gross_pnl=1_200,
                realized_net_pnl=900,
                fees_paid=50,
                taxes_paid=250,
                trade_count=2,
                win_count=1,
                loss_count=1,
                breakeven_count=0,
                winning_net_pnl_sum=1_500,
                losing_net_pnl_sum=-600,
                largest_win_net=1_500,
                largest_loss_net=-600,
            ),
            positions={},
            _halted=False,
        )
        balance = SimpleNamespace(
            total_eval_amount=9_800_000,
            total_deposit=900_000,
            total_profit_loss=900,
            total_profit_rate=0.09,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = update_performance_reports(
                strategy=strategy,
                balance=balance,
                session_pnl=900,
                trading_mode="paper",
                report_root=Path(tmp_dir),
            )

            self.assertTrue(paths["scorecard"]["json"].exists())
            self.assertTrue(paths["scorecard"]["md"].exists())
            self.assertTrue(paths["readiness"]["json"].exists())
            self.assertTrue(paths["readiness"]["md"].exists())

            payload = json.loads(paths["scorecard"]["json"].read_text(encoding="utf-8"))
            self.assertEqual(payload["pnl"]["session_pnl"], 900)

    def test_readiness_passes_when_sample_profit_factor_and_expectancy_are_good(self):
        scorecards = []
        start = datetime(2026, 2, 10, 15, 20, 0)
        for idx in range(20):
            current = start + timedelta(days=idx)
            scorecards.append(
                {
                    "date": current.date().isoformat(),
                    "generated_at": current.isoformat(timespec="seconds"),
                    "pnl": {
                        "session_pnl": 1_500,
                        "realized_net_pnl": 1_500,
                        "winning_net_pnl_sum": 2_400,
                        "losing_net_pnl_sum": -900,
                    },
                    "trades": {
                        "closed_trades": 5,
                    },
                }
            )

        readiness = evaluate_real_trading_readiness(scorecards)

        self.assertTrue(readiness["criteria"]["sample_size"]["ok"])
        self.assertTrue(readiness["criteria"]["profit_factor"]["ok"])
        self.assertTrue(readiness["criteria"]["expectancy_positive"]["ok"])
        self.assertTrue(readiness["ready_for_real_trading"])

    def test_readiness_fails_when_net_and_expectancy_are_negative(self):
        scorecards = [
            {
                "date": "2026-03-10",
                "generated_at": "2026-03-10T15:20:00",
                "pnl": {
                    "session_pnl": -1_000,
                    "realized_net_pnl": -1_000,
                    "winning_net_pnl_sum": 400,
                    "losing_net_pnl_sum": -1_400,
                },
                "trades": {
                    "closed_trades": 4,
                },
            }
        ]

        readiness = evaluate_real_trading_readiness(scorecards, min_days=1, min_closed_trades=1)

        self.assertFalse(readiness["criteria"]["net_pnl_positive"]["ok"])
        self.assertFalse(readiness["criteria"]["expectancy_positive"]["ok"])
        self.assertFalse(readiness["ready_for_real_trading"])


if __name__ == "__main__":
    unittest.main()
