import unittest
from types import SimpleNamespace

from src.scheduler import TradingScheduler


class DummyAccount:
    def __init__(self, realized_pnl):
        self.realized_pnl = realized_pnl

    def get_realized_profit_loss(self):
        return self.realized_pnl


class SchedulerPnLResolutionTests(unittest.TestCase):
    def _make_scheduler(self, realized_pnl, daily_pnl=None):
        scheduler = TradingScheduler.__new__(TradingScheduler)
        scheduler.account = DummyAccount(realized_pnl)
        if daily_pnl is None:
            scheduler.strategy = SimpleNamespace()
        else:
            scheduler.strategy = SimpleNamespace(daily_pnl=daily_pnl)
        return scheduler

    def test_resolve_session_pnl_prefers_realized_api(self):
        scheduler = self._make_scheduler(
            realized_pnl=-5861,
            daily_pnl=SimpleNamespace(realized_net_pnl=1234),
        )
        balance = SimpleNamespace(total_profit_loss=0)

        resolved = scheduler._resolve_session_profit_loss(balance)

        self.assertEqual(resolved, -5861)

    def test_resolve_session_pnl_fallbacks_to_strategy(self):
        scheduler = self._make_scheduler(
            realized_pnl=None,
            daily_pnl=SimpleNamespace(realized_net_pnl=1234),
        )
        balance = SimpleNamespace(total_profit_loss=0)

        resolved = scheduler._resolve_session_profit_loss(balance)

        self.assertEqual(resolved, 1234)

    def test_resolve_session_pnl_uses_strategy_session_delta_when_baseline_exists(self):
        scheduler = self._make_scheduler(
            realized_pnl=None,
            daily_pnl=SimpleNamespace(realized_net_pnl=19727),
        )
        balance = SimpleNamespace(total_profit_loss=0)

        resolved = scheduler._resolve_session_profit_loss(
            balance,
            strategy_pnl_baseline=19727,
        )

        self.assertEqual(resolved, 0)

    def test_resolve_session_pnl_fallbacks_to_balance(self):
        scheduler = self._make_scheduler(realized_pnl=None, daily_pnl=None)
        balance = SimpleNamespace(total_profit_loss=-200)

        resolved = scheduler._resolve_session_profit_loss(balance)

        self.assertEqual(resolved, -200)


if __name__ == "__main__":
    unittest.main()
