import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from src.performance_reporting import (
    build_daily_scorecard,
    evaluate_paper_trading_gate,
    update_performance_reports,
)
from src.strategies.momentum_scalp_types import DailyPnL


class PerformanceReportingTests(unittest.TestCase):
    def strategy(self, *, net_pnl=3_200, halt_reason=""):
        return SimpleNamespace(
            daily_pnl=DailyPnL(
                realized_gross_pnl=3_600,
                realized_net_pnl=net_pnl,
                fees_paid=200,
                taxes_paid=200,
                trade_count=2,
                win_count=1,
                loss_count=1,
                winning_net_pnl_sum=4_000,
                losing_net_pnl_sum=net_pnl - 4_000,
                largest_win_net=4_000,
                largest_loss_net=net_pnl - 4_000,
            ),
            positions={},
            _halted=bool(halt_reason),
            _halt_reason=halt_reason,
            _real_money_stage=0,
            _capital_scale=1.0,
        )

    @staticmethod
    def balance():
        return SimpleNamespace(
            total_eval_amount=1_003_200,
            total_deposit=1_003_200,
            total_profit_loss=3_200,
        )

    def test_scorecard_uses_confirmed_strategy_ledger_without_log_parsing(self):
        card = build_daily_scorecard(
            self.strategy(halt_reason="daily_profit_target"),
            self.balance(),
            session_pnl=99_999,
            trading_mode="paper",
            generated_at=datetime(2026, 7, 31, 15, 30),
        )
        self.assertEqual(card["pnl"]["realized_net_pnl"], 3_200)
        self.assertEqual(card["pnl"]["source"], "confirmed_sell_fill_ledger")
        self.assertEqual(card["strategy"]["name"], "expected_value_long")
        self.assertEqual(card["strategy"]["halt_reason"], "daily_profit_target")
        self.assertNotIn("log_analysis", card)

    def test_update_writes_only_scorecard_and_readiness_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = update_performance_reports(
                self.strategy(),
                self.balance(),
                session_pnl=3_200,
                trading_mode="paper",
                report_root=Path(directory),
            )
            self.assertEqual(set(paths), {"scorecard", "readiness"})
            self.assertTrue(paths["scorecard"]["json"].exists())
            self.assertTrue(paths["readiness"]["json"].exists())
            payload = json.loads(
                paths["scorecard"]["json"].read_text(encoding="utf-8")
            )
            self.assertEqual(payload["trades"]["closed_trades"], 2)

    def test_paper_gate_reads_explicit_halt_reason(self):
        cards = []
        for day, net in enumerate((10_000, 2_000, 2_000, 2_000, 2_000), start=1):
            cards.append(
                {
                    "date": f"2026-07-{day:02d}",
                    "trading_mode": "paper",
                    "pnl": {"realized_net_pnl": net},
                    "strategy": {
                        "halt_reason": "daily_profit_target" if day == 1 else ""
                    },
                }
            )
        gate = evaluate_paper_trading_gate(cards)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["target_hit_days"], 1)


if __name__ == "__main__":
    unittest.main()
