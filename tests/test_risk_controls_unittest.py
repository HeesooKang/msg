import unittest
from collections import deque
from datetime import datetime, timedelta
from unittest.mock import patch

from src.api_client import KISClient
from src.config import Config
from src.market_data import MarketDataAPI
from src.models import OrderResult, OrderSide, Position, Quote
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
            daily_total_loss_limit=-5_000,
            enable_unrealized_loss_guard=True,
            enable_regime_adaptive=False,
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

    def test_momentum_daily_breaker_uses_effective_pnl_after_restore_offset(self):
        cfg = MomentumScalpConfig(
            daily_profit_target=500,
            daily_loss_limit=-500,
            use_restored_pnl_for_daily_breaker=False,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)

        # 재시작 복구값(누적 +1,500원)이 있어도 브레이커 시작값은 0원으로 본다.
        strategy.daily_pnl.realized_net_pnl = 1_500
        strategy._daily_breaker_pnl_offset = 1_500
        strategy.on_batch_tick([])
        self.assertFalse(strategy._halted)

        # 세션 중 신규 순실현이 +650원으로 늘면(강세 프로파일 목표 600원 초과) 목표 달성 처리.
        strategy.daily_pnl.realized_net_pnl = 2_150
        strategy.on_batch_tick([])
        self.assertTrue(strategy._halted)

    def test_momentum_sync_positions_from_account(self):
        strategy = MomentumScalpStrategy(market_data=None, config=MomentumScalpConfig())
        account_positions = [
            Position(
                symbol="005930",
                name="삼성전자",
                quantity=3,
                avg_price=71234.0,
                current_price=71900,
                eval_amount=0,
                profit_loss=0,
                profit_rate=0.0,
            )
        ]

        strategy.sync_positions_from_account(account_positions)

        self.assertIn("005930", strategy.positions)
        pos = strategy.positions["005930"]
        self.assertEqual(pos.quantity, 3)
        self.assertEqual(pos.buy_price, 71234)
        self.assertEqual(pos.invested_amount, 213702)

    def test_inverse_buy_blocked_by_global_loss_cooldown(self):
        cfg = MomentumScalpConfig(
            inverse_enabled=True,
            enable_volume_spike_filter=False,
            enable_expected_net_filter=False,
            inverse_min_change_rate=1.0,
            inverse_min_momentum=0.0,
            inverse_min_bear_score=2,
            bearish_threshold=2,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy._bear_score = 3
        strategy._global_loss_cooldown_until = datetime.now() + timedelta(minutes=5)

        quote = Quote(
            symbol="252670",
            name="KODEX 200선물인버스2X",
            current_price=2_000,
            change=40,
            change_rate=2.0,
            open_price=1_980,
            high_price=2_010,
            low_price=1_970,
            volume=500_000,
            trade_amount=1_000_000_000,
        )

        order = strategy._evaluate_inverse_buy(quote)
        self.assertIsNone(order)

    def test_inverse_volume_spike_gate_relaxes_only_for_inverse(self):
        cfg = MomentumScalpConfig(
            inverse_enabled=True,
            inverse_etfs=["252670"],
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy._bear_score = 3

        inverse_quote = Quote(
            symbol="252670",
            name="KODEX 200선물인버스2X",
            current_price=2_000,
            change=40,
            change_rate=2.0,
            open_price=1_980,
            high_price=2_010,
            low_price=1_970,
            volume=500_000,
            trade_amount=1_000_000_000,
        )
        regular_quote = Quote(
            symbol="005930",
            name="삼성전자",
            current_price=60_000,
            change=1_200,
            change_rate=2.0,
            open_price=59_400,
            high_price=60_300,
            low_price=59_100,
            volume=500_000,
            trade_amount=30_000_000_000,
        )

        for symbol in ("252670", "005930"):
            strategy._latest_tick_volumes[symbol] = 15_500
            strategy._recent_tick_volumes[symbol] = deque([10_000, 10_000, 15_500], maxlen=12)

        self.assertTrue(strategy._is_volume_spike(inverse_quote, score=2.5))
        self.assertFalse(strategy._is_volume_spike(regular_quote, score=2.5))

    def test_inverse_buy_can_pass_relaxed_volume_spike_gate(self):
        cfg = MomentumScalpConfig(
            inverse_enabled=True,
            inverse_etfs=["252670"],
            enable_expected_net_filter=False,
            inverse_min_change_rate=1.0,
            inverse_min_momentum=0.0,
            inverse_min_bear_score=2,
            bearish_threshold=2,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy._bear_score = 3

        quote = Quote(
            symbol="252670",
            name="KODEX 200선물인버스2X",
            current_price=2_000,
            change=40,
            change_rate=2.0,
            open_price=1_980,
            high_price=2_010,
            low_price=1_970,
            volume=500_000,
            trade_amount=1_000_000_000,
        )
        strategy._latest_tick_volumes["252670"] = 15_500
        strategy._recent_tick_volumes["252670"] = deque([10_000, 10_000, 15_500], maxlen=12)

        order = strategy._evaluate_inverse_buy(quote)

        self.assertIsNotNone(order)
        self.assertEqual(order.side, OrderSide.BUY)

    def test_inverse_trailing_stop_requires_activation_gain(self):
        cfg = MomentumScalpConfig(
            inverse_enabled=True,
            enable_regime_adaptive=False,
            inverse_take_profit_pct=5.0,
            inverse_stop_loss_pct=-2.0,
            inverse_trailing_stop_pct=-0.3,
            inverse_trailing_stop_activation_gain_pct=0.45,
            bearish_threshold=2,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy._bear_score = 2  # 시장반등 청산 우회
        strategy.positions["252670"] = PositionState(
            symbol="252670",
            buy_price=10_000,
            quantity=1,
            invested_amount=10_000,
        )

        pos = strategy.positions["252670"]

        # 고점 이익이 0.45% 미만이면 추적손절 조건 미발동
        pos.high_since_buy = 10_020  # +0.2%
        no_trigger_quote = Quote(
            symbol="252670",
            name="KODEX 200선물인버스2X",
            current_price=9_985,
            change=0,
            change_rate=0.0,
            open_price=10_000,
            high_price=10_030,
            low_price=9_960,
            volume=100_000,
            trade_amount=200_000_000,
        )
        self.assertIsNone(strategy._evaluate_inverse_sell(no_trigger_quote))

        # 고점 이익이 충분하고 고점 대비 하락률이 임계 이하면 추적손절 발동
        pos.high_since_buy = 10_080  # +0.8%
        trigger_quote = Quote(
            symbol="252670",
            name="KODEX 200선물인버스2X",
            current_price=10_040,  # 고점 대비 약 -0.40%
            change=0,
            change_rate=0.0,
            open_price=10_000,
            high_price=10_080,
            low_price=10_000,
            volume=120_000,
            trade_amount=250_000_000,
        )
        order = strategy._evaluate_inverse_sell(trigger_quote)
        self.assertIsNotNone(order)
        self.assertEqual(order.side, OrderSide.SELL)

    def test_entry_window_blocked_when_dynamic_mode_disabled(self):
        cfg = MomentumScalpConfig(
            block_new_entry_windows=["11:00-12:00"],
            enable_dynamic_entry_block_windows=False,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy._bear_score = 3

        blocked = strategy._is_new_entry_window_blocked(datetime(2026, 3, 9, 11, 30, 0))
        self.assertTrue(blocked)

    def test_entry_window_dynamic_unblock_in_bear_market(self):
        cfg = MomentumScalpConfig(
            block_new_entry_windows=["11:00-12:00"],
            enable_dynamic_entry_block_windows=True,
            dynamic_entry_block_disable_bear_score=2,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy._bear_score = 3

        blocked = strategy._is_new_entry_window_blocked(datetime(2026, 3, 9, 11, 30, 0))
        self.assertFalse(blocked)


if __name__ == "__main__":
    unittest.main()
