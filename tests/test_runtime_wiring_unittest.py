import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.main import run_scheduled
from src.scheduler import TradingScheduler


class RuntimeWiringTests(unittest.TestCase):
    @patch("src.scheduler.AlertManager")
    @patch("src.scheduler.OrderExecutor")
    @patch("src.scheduler.AccountAPI")
    @patch("src.scheduler.TradingAPI")
    def test_scheduler_reuses_strategy_market_data_client(
        self,
        trading_cls,
        account_cls,
        executor_cls,
        alerts_cls,
    ):
        shared_config = SimpleNamespace(off_hours_check_interval=1800, is_paper=True, trading_mode="paper")
        shared_client = SimpleNamespace(config=shared_config, token_manager=object())
        shared_market_data = SimpleNamespace(client=shared_client)
        strategy = SimpleNamespace(
            market_data=shared_market_data,
            config=SimpleNamespace(quote_freshness_seconds=5),
        )

        scheduler = TradingScheduler(strategy)

        self.assertIs(scheduler.config, shared_config)
        self.assertIs(scheduler.client, shared_client)
        self.assertIs(scheduler.market_data, shared_market_data)
        trading_cls.assert_called_once_with(shared_client)
        account_cls.assert_called_once_with(shared_client)
        executor_cls.assert_called_once()
        alerts_cls.assert_called_once()

    @patch("src.scheduler.TradingScheduler")
    def test_run_scheduled_uses_strategy_as_the_only_runtime_root(self, scheduler_cls):
        strategy = SimpleNamespace()
        scheduler = Mock()
        scheduler_cls.return_value = scheduler

        run_scheduled(strategy, tick_interval=7)

        scheduler_cls.assert_called_once_with(strategy)
        scheduler.run.assert_called_once_with(tick_interval=7)


if __name__ == "__main__":
    unittest.main()
