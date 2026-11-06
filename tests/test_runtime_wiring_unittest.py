import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.main import run_scheduled
from src.scheduler import TradingScheduler


class RuntimeWiringTests(unittest.TestCase):
    def test_paper_launch_keeps_user_capital_daily_limits_and_costs(self):
        from run_bot import get_strategy
        with (patch("src.api_client.KISClient"), patch("src.auth.TokenManager"),
              patch("src.market_data.MarketDataAPI") as market_data):
            strategy = get_strategy(SimpleNamespace(is_paper=True))
        config = strategy.config
        self.assertEqual((config.seed_money, config.daily_profit_target, config.daily_loss_limit),
                         (1_000_000, 10_000, -5_000))
        self.assertEqual((config.commission_rate, config.sell_tax_rate), (.000142, .0015))
        self.assertEqual(str(market_data.call_args.kwargs["discovery_archive"].root), config.quote_archive_root)

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
        trading_cls.assert_called_once_with(shared_client, quote_provider=scheduler._quote_stream.latest_quote)
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
