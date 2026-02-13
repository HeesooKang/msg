import unittest
from unittest.mock import patch

from src.api_client import KISClient
from src.config import Config
from src.market_data import MarketDataAPI
from src.models import OrderResult, OrderSide, Quote
from src.strategies.momentum_scalp import (
    MomentumScalpConfig,
    MomentumScalpStrategy,
    PositionState,
)


class DummyTokenManager:
    def get_token(self):
        return "dummy-token"


class DummyResponse:
    def __init__(self, success, error_code="", error_message="", output=None):
        self.success = success
        self.error_code = error_code
        self.error_message = error_message
        self.output = output


class DummyClient:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def get(self, **kwargs):
        self.calls += 1
        return self.response


class RiskControlTests(unittest.TestCase):
    def test_market_open_opsq_warn_once_and_cache(self):
        client = DummyClient(
            DummyResponse(
                success=False,
                error_code="OPSQ0002",
                error_message="Service code does not exist",
            )
        )
        market = MarketDataAPI(client)
        date = "20260213"  # Friday

        with patch("src.market_data.logger.warning") as mock_warn:
            first = market.is_market_open(date)
            second = market.is_market_open(date)

        self.assertTrue(first)
        self.assertTrue(second)
        self.assertEqual(client.calls, 1)
        self.assertEqual(mock_warn.call_count, 1)

    def test_momentum_allocation_respects_total_and_stock_caps(self):
        cfg = MomentumScalpConfig(
            seed_money=1_000_000,
            per_stock_amount=200_000,
            max_per_stock_amount=400_000,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy.positions["AAA"] = PositionState(symbol="AAA", buy_price=10_000, quantity=30)  # 300,000
        strategy.positions["BBB"] = PositionState(symbol="BBB", buy_price=13_000, quantity=50)  # 650,000

        alloc = strategy._compute_buy_allocation("AAA", current_price=10_000)
        # total_room=50,000, stock_room=100,000 -> alloc=50,000
        self.assertEqual(alloc, 50_000)

    def test_api_client_keeps_ctca_tr_id_in_paper(self):
        config = Config(
            trading_mode="paper",
            is_paper=True,
            api_key="k",
            api_secret="s",
            account_number="12345678",
            account_product_code="01",
            hts_id="id",
            base_url="https://example.com",
            ws_url="ws://example.com",
            rate_limit_interval=0.5,
            log_level="INFO",
        )
        client = KISClient(config, DummyTokenManager())

        holiday_headers = client._build_headers("CTCA0903R")
        order_headers = client._build_headers("TTTC0012U")

        self.assertEqual(holiday_headers["tr_id"], "CTCA0903R")
        self.assertEqual(order_headers["tr_id"], "VTTC0012U")

    def test_momentum_daily_hard_stop_uses_net_realized_pnl(self):
        cfg = MomentumScalpConfig(
            daily_profit_target=20_000,
            daily_loss_limit=-5_000,
            commission_rate=0.00015,
            tax_slippage_rate=0.002,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy.initialize()

        strategy.on_order_filled(
            OrderResult(
                success=True,
                symbol="005930",
                side=OrderSide.BUY,
                quantity=10,
                price=10_000,
            )
        )
        strategy.on_order_filled(
            OrderResult(
                success=True,
                symbol="005930",
                side=OrderSide.SELL,
                quantity=10,
                price=12_000,
            )
        )

        # gross: +20,000 / net: +19,727 (매수수수료15 + 매도수수료18 + 세금/슬리피지240 차감)
        self.assertEqual(strategy.daily_pnl.realized_gross_pnl, 20_000)
        self.assertEqual(strategy.daily_pnl.realized_net_pnl, 19_727)

        # 순손익 기준이므로 목표(+20,000) 미달 상태
        strategy.on_batch_tick([])
        self.assertFalse(strategy._halted)

    def test_momentum_sell_failure_keeps_position(self):
        strategy = MomentumScalpStrategy(market_data=None, config=MomentumScalpConfig())
        strategy.positions["005930"] = PositionState(
            symbol="005930",
            buy_price=10_000,
            quantity=3,
            invested_amount=30_000,
        )

        strategy.on_order_filled(
            OrderResult(
                success=False,
                symbol="005930",
                side=OrderSide.SELL,
                quantity=3,
                price=9_500,
            )
        )

        self.assertIn("005930", strategy.positions)

    def test_momentum_unrealized_loss_guard_liquidates_all(self):
        cfg = MomentumScalpConfig(
            daily_loss_limit=-5_000,
            enable_unrealized_loss_guard=True,
            per_position_stop_loss=-100_000,  # 개별 손절보다 보조컷이 먼저 작동하도록 완화
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy.positions["005930"] = PositionState(
            symbol="005930",
            buy_price=10_000,
            quantity=1,
            invested_amount=10_000,
        )
        quote = Quote(
            symbol="005930",
            name="삼성전자",
            current_price=4_000,
            change=-6_000,
            change_rate=-60.0,
            open_price=10_000,
            high_price=10_100,
            low_price=3_900,
            volume=1_000_000,
            trade_amount=4_000_000_000,
        )

        orders = strategy.on_batch_tick([quote])

        self.assertTrue(strategy._halted)
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].symbol, "005930")
        self.assertEqual(orders[0].side, OrderSide.SELL)


if __name__ == "__main__":
    unittest.main()
