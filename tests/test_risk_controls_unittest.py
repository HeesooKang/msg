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

    def test_momentum_allocation_uses_regime_budget_ratios(self):
        cfg = MomentumScalpConfig(
            seed_money=1_000_000,
            max_position_count=2,
            bull_max_position_count=2,
            neutral_max_position_count=1,
            per_stock_amount=220_000,
            max_per_stock_amount=500_000,
            capital_utilization_pct=0.60,
            bull_capital_utilization_pct=0.84,
            neutral_capital_utilization_pct=0.68,
            max_single_position_pct=0.50,
            bull_max_single_position_pct=0.42,
            neutral_max_single_position_pct=0.36,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)

        strategy._bear_score = 0
        bull_alloc = strategy._compute_buy_allocation("AAA", current_price=10_000)
        self.assertEqual(bull_alloc, 420_000)

        strategy._bear_score = 1
        neutral_alloc = strategy._compute_buy_allocation("BBB", current_price=10_000)
        self.assertEqual(neutral_alloc, 360_000)

    def test_momentum_allocation_compounds_realized_pnl_intraday(self):
        cfg = MomentumScalpConfig(
            seed_money=1_000_000,
            max_position_count=2,
            per_stock_amount=200_000,
            max_per_stock_amount=500_000,
            capital_utilization_pct=0.60,
            max_single_position_pct=0.50,
            profit_protect_threshold=999_999,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy.daily_pnl.realized_net_pnl = 100_000

        alloc = strategy._compute_buy_allocation("AAA", current_price=10_000)
        self.assertEqual(alloc, 330_000)

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

    def test_momentum_daily_loss_limit_waits_for_open_profit_offset(self):
        cfg = MomentumScalpConfig(
            daily_loss_limit=-500,
            daily_total_loss_limit=-1_500,
            enable_regime_adaptive=False,
            take_profit_pct=20.0,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy.initialize()
        strategy.daily_pnl.realized_net_pnl = -600
        strategy.positions["005930"] = PositionState(
            symbol="005930",
            buy_price=10_000,
            quantity=1,
            invested_amount=10_000,
        )
        quote = Quote(
            symbol="005930",
            name="삼성전자",
            current_price=10_600,
            change=600,
            change_rate=6.0,
            open_price=10_050,
            high_price=10_650,
            low_price=9_980,
            volume=500_000,
            trade_amount=5_300_000_000,
        )

        orders = strategy.on_batch_tick([quote])

        self.assertFalse(strategy._halted)

    def test_momentum_backtest_regime_uses_quotes_without_market_data(self):
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(inverse_enabled=False),
        )
        quotes = [
            Quote(
                symbol="AAA",
                name="AAA",
                current_price=9_000,
                change=-1_000,
                change_rate=-10.0,
                open_price=9_800,
                high_price=9_900,
                low_price=8_900,
                volume=500_000,
                trade_amount=4_500_000_000,
            ),
            Quote(
                symbol="BBB",
                name="BBB",
                current_price=9_200,
                change=-800,
                change_rate=-8.0,
                open_price=9_700,
                high_price=9_800,
                low_price=9_100,
                volume=420_000,
                trade_amount=3_864_000_000,
            ),
            Quote(
                symbol="CCC",
                name="CCC",
                current_price=9_400,
                change=-600,
                change_rate=-6.0,
                open_price=9_900,
                high_price=10_000,
                low_price=9_300,
                volume=410_000,
                trade_amount=3_854_000_000,
            ),
        ]

        strategy.on_batch_tick(quotes)

        self.assertGreaterEqual(strategy._bear_score, 2)
        self.assertTrue(strategy._bear_market)

    def test_momentum_regime_profile_uses_neutral_for_bear_score_one(self):
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(inverse_enabled=False),
        )
        strategy._bear_score = 1

        self.assertEqual(strategy._resolve_regime_profile_name(), "neutral")
        self.assertFalse(strategy._is_bullish_regime())

    def test_momentum_regime_profile_uses_soft_bear_for_bear_score_two(self):
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(inverse_enabled=False),
        )
        strategy._bear_score = 2

        self.assertEqual(strategy._resolve_regime_profile_name(), "soft_bear")
        self.assertFalse(strategy._is_bullish_regime())

    def test_momentum_regime_specific_max_position_count(self):
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(
                max_position_count=3,
                bull_max_position_count=4,
                neutral_max_position_count=2,
                soft_bear_max_position_count=1,
                bear_max_position_count=1,
                inverse_enabled=False,
            ),
        )

        strategy._bear_score = 0
        self.assertEqual(strategy._effective_max_position_count(), 4)

        strategy._bear_score = 1
        self.assertEqual(strategy._effective_max_position_count(), 2)

        strategy._bear_score = 2
        self.assertEqual(strategy._effective_max_position_count(), 1)

        strategy._bear_score = 3
        self.assertEqual(strategy._effective_max_position_count(), 1)

    def test_momentum_soft_bear_inverse_profile_relaxes_thresholds(self):
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(
                inverse_enabled=True,
                inverse_max_positions=2,
                soft_bear_inverse_max_positions=1,
                inverse_min_bear_score=3,
                inverse_min_change_rate=1.4,
                inverse_min_momentum=3.0,
            ),
        )

        strategy._bear_score = 2
        self.assertEqual(strategy._resolve_regime_profile_name(), "soft_bear")
        self.assertEqual(strategy._regime_inverse_max_positions(), 1)
        self.assertEqual(strategy._regime_inverse_min_bear_score(), 2)
        self.assertLess(strategy._regime_inverse_min_change_rate(), 1.4)
        self.assertLess(strategy._regime_inverse_min_momentum(), 3.0)

        strategy._bear_score = 3
        self.assertEqual(strategy._resolve_regime_profile_name(), "bear")
        self.assertEqual(strategy._regime_inverse_max_positions(), 2)
        self.assertEqual(strategy._regime_inverse_min_bear_score(), 3)

    def test_neutral_profile_relaxes_long_entry_thresholds(self):
        cfg = MomentumScalpConfig()
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy._bear_score = 1

        self.assertEqual(strategy._resolve_regime_profile_name(), "neutral")
        self.assertLess(strategy._regime_min_change_rate(), cfg.min_change_rate)
        self.assertLess(strategy._regime_min_momentum_score(), cfg.min_momentum_score)
        self.assertLess(strategy._regime_volume_spike_ratio(), cfg.volume_spike_ratio)

    def test_momentum_daily_loss_limit_still_halts_when_total_net_below_limit(self):
        cfg = MomentumScalpConfig(
            daily_loss_limit=-500,
            daily_total_loss_limit=-500,
            enable_regime_adaptive=False,
            take_profit_pct=20.0,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy.initialize()
        strategy.daily_pnl.realized_net_pnl = -600
        strategy.positions["005930"] = PositionState(
            symbol="005930",
            buy_price=10_000,
            quantity=1,
            invested_amount=10_000,
        )
        quote = Quote(
            symbol="005930",
            name="삼성전자",
            current_price=10_050,
            change=50,
            change_rate=0.5,
            open_price=10_000,
            high_price=10_080,
            low_price=9_980,
            volume=500_000,
            trade_amount=5_025_000_000,
        )

        orders = strategy.on_batch_tick([quote])

        self.assertTrue(strategy._halted)
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].symbol, "005930")
        self.assertEqual(orders[0].side, OrderSide.SELL)

    def test_neutral_chase_block_rejects_day_high_without_pullback(self):
        cfg = MomentumScalpConfig(
            enable_volume_spike_filter=False,
            enable_expected_net_filter=False,
            enable_pool_persistence_gate=False,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy._bear_score = 1

        history = [
            Quote("011700", "미원", 10_000, 0, 0.0, 10_000, 10_000, 9_950, 10_000, 100_000),
            Quote("011700", "미원", 10_120, 120, 1.2, 10_000, 10_120, 9_950, 20_000, 200_000),
            Quote("011700", "미원", 10_220, 220, 2.2, 10_000, 10_220, 9_950, 30_000, 300_000),
            Quote("011700", "미원", 10_240, 240, 2.4, 10_000, 10_240, 9_950, 40_000, 400_000),
        ]
        for quote in history:
            strategy._record_recent_quote(quote)

        ok, _, reject = strategy._passes_neutral_pullback_reclaim_setup(history[-1], score=2.4)
        self.assertFalse(ok)
        self.assertEqual(reject, "neutral_chase_block")

    def test_neutral_pullback_reclaim_requires_pullback_and_reclaim(self):
        cfg = MomentumScalpConfig(
            enable_volume_spike_filter=False,
            enable_expected_net_filter=False,
            enable_pool_persistence_gate=False,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy._bear_score = 1

        history = [
            Quote("011700", "미원", 10_000, 0, 0.0, 10_000, 10_000, 9_950, 10_000, 100_000),
            Quote("011700", "미원", 10_200, 200, 2.0, 10_000, 10_200, 9_950, 20_000, 200_000),
            Quote("011700", "미원", 10_120, 120, 1.2, 10_000, 10_220, 9_950, 30_000, 300_000),
            Quote("011700", "미원", 10_205, 205, 2.05, 10_000, 10_220, 9_950, 40_000, 400_000),
        ]
        for quote in history:
            strategy._record_recent_quote(quote)

        ok, setup_name, reason = strategy._passes_neutral_pullback_reclaim_setup(history[-1], score=2.1)
        self.assertTrue(ok)
        self.assertEqual(setup_name, "neutral_pullback_reclaim")
        self.assertIn("pullback_reclaim", reason)

    def test_soft_bear_inverse_setup_allows_reclaim_after_pullback(self):
        cfg = MomentumScalpConfig(
            inverse_enabled=True,
            inverse_etfs=["252670"],
            enable_volume_spike_filter=False,
            enable_expected_net_filter=False,
            enable_pool_persistence_gate=False,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy._bear_score = 2

        history = [
            Quote("252670", "KODEX 인버스", 2_000, 0, 0.0, 2_000, 2_000, 1_995, 10_000, 20_000_000),
            Quote("252670", "KODEX 인버스", 2_030, 30, 1.5, 2_000, 2_030, 1_995, 20_000, 40_000_000),
            Quote("252670", "KODEX 인버스", 2_020, 20, 1.0, 2_000, 2_030, 1_995, 30_000, 60_000_000),
            Quote("252670", "KODEX 인버스", 2_032, 32, 1.6, 2_000, 2_032, 1_995, 40_000, 80_000_000),
        ]
        for quote in history:
            strategy._record_recent_quote(quote)

        ok, setup_name, reason = strategy._passes_soft_bear_inverse_setup(history[-1], score=2.4)
        self.assertTrue(ok)
        self.assertEqual(setup_name, "soft_bear_inverse_breakdown")
        self.assertIn("weak_rebound_failure", reason)

    def test_order_fill_preserves_entry_metadata(self):
        cfg = MomentumScalpConfig(
            enable_volume_spike_filter=False,
            enable_expected_net_filter=False,
            enable_pool_persistence_gate=False,
            enable_entry_confirmation=False,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy._bear_score = 1

        history = [
            Quote("011700", "미원", 10_000, 0, 0.0, 10_000, 10_000, 9_950, 10_000, 100_000),
            Quote("011700", "미원", 10_200, 200, 2.0, 10_000, 10_200, 9_950, 20_000, 200_000),
            Quote("011700", "미원", 10_120, 120, 1.2, 10_000, 10_220, 9_950, 30_000, 300_000),
            Quote("011700", "미원", 10_205, 205, 2.05, 10_000, 10_220, 9_950, 40_000, 400_000),
        ]
        for quote in history:
            strategy._quotes_cache[quote.symbol] = quote
            strategy._record_recent_quote(quote)

        order = strategy._evaluate_buy(history[-1])
        self.assertIsNotNone(order)

        strategy.on_order_filled(
            OrderResult(
                success=True,
                symbol="011700",
                side=OrderSide.BUY,
                quantity=order.quantity,
                price=history[-1].current_price,
            )
        )

        pos = strategy.positions["011700"]
        self.assertEqual(pos.entry_setup_name, "neutral_pullback_reclaim")
        self.assertEqual(pos.entry_reason, "pullback_reclaim")
        self.assertEqual(pos.regime_label, "neutral")
        self.assertEqual(pos.bear_score, 1)

    def test_soft_bear_profile_can_disable_inverse_by_config(self):
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(
                inverse_enabled=True,
                inverse_max_positions=1,
                soft_bear_inverse_max_positions=0,
            ),
        )
        strategy._bear_score = 2

        self.assertEqual(strategy._resolve_regime_profile_name(), "soft_bear")
        self.assertEqual(strategy._regime_inverse_max_positions(), 0)

    def test_stage1_blocks_new_neutral_long_entries(self):
        cfg = MomentumScalpConfig(
            enable_volume_spike_filter=False,
            enable_expected_net_filter=False,
            enable_pool_persistence_gate=False,
            stage1_loss_threshold=-3_000,
            profit_protect_threshold=999_999,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy._bear_score = 1
        strategy.daily_pnl.realized_net_pnl = -3_100

        history = [
            Quote("011700", "미원", 10_000, 0, 0.0, 10_000, 10_000, 9_950, 10_000, 100_000),
            Quote("011700", "미원", 10_200, 200, 2.0, 10_000, 10_200, 9_950, 20_000, 200_000),
            Quote("011700", "미원", 10_120, 120, 1.2, 10_000, 10_220, 9_950, 30_000, 300_000),
            Quote("011700", "미원", 10_205, 205, 2.05, 10_000, 10_220, 9_950, 40_000, 400_000),
        ]
        for quote in history:
            strategy._record_recent_quote(quote)

        order = strategy._evaluate_buy(history[-1])
        self.assertIsNone(order)

    def test_dynamic_stop_amount_uses_tighter_notional_cap(self):
        strategy = MomentumScalpStrategy(market_data=None, config=MomentumScalpConfig())
        long_pos = PositionState(symbol="AAA", buy_price=10_000, quantity=20, invested_amount=200_000)
        inverse_pos = PositionState(symbol="252670", buy_price=2_000, quantity=500, invested_amount=1_000_000)

        self.assertEqual(strategy._long_stop_loss_amount(long_pos), -1_400)
        self.assertEqual(strategy._inverse_stop_loss_amount(inverse_pos), -1_800)

    def test_bullish_marginal_signal_requires_full_confirmation(self):
        cfg = MomentumScalpConfig(
            enable_regime_adaptive=True,
            enable_early_session_guard=True,
            entry_confirmation_ticks=2,
            bullish_min_momentum_score=2.6,
            bullish_min_momentum_score_floor=3.4,
            bullish_fast_entry_score_bonus=0.9,
            bullish_fast_entry_change_rate_bonus=0.6,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy._bear_score = 0

        now = datetime.now()
        strategy._session_start_at = now - timedelta(minutes=30)
        quote = Quote(
            symbol="005930",
            name="삼성전자",
            current_price=71_000,
            change=800,
            change_rate=1.14,
            open_price=70_200,
            high_price=71_100,
            low_price=70_100,
            volume=300_000,
            trade_amount=21_300_000_000,
        )

        first = strategy._can_confirm_entry(
            quote=quote,
            score=3.4,
            is_scale_in=False,
            now=now,
        )
        second = strategy._can_confirm_entry(
            quote=quote,
            score=3.4,
            is_scale_in=False,
            now=now + timedelta(seconds=10),
        )

        self.assertFalse(first)
        self.assertTrue(second)

    def test_bullish_exceptional_signal_can_still_fast_enter(self):
        cfg = MomentumScalpConfig(
            enable_regime_adaptive=True,
            enable_early_session_guard=True,
            entry_confirmation_ticks=2,
            bullish_min_momentum_score=2.6,
            bullish_min_momentum_score_floor=3.4,
            bullish_fast_entry_score_bonus=0.9,
            bullish_fast_entry_change_rate_bonus=0.6,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy._bear_score = 0

        now = datetime.now()
        strategy._session_start_at = now - timedelta(minutes=30)
        quote = Quote(
            symbol="000660",
            name="SK하이닉스",
            current_price=210_000,
            change=4_500,
            change_rate=2.19,
            open_price=205_500,
            high_price=210_500,
            low_price=205_000,
            volume=350_000,
            trade_amount=73_500_000_000,
        )

        fast_entry = strategy._can_confirm_entry(
            quote=quote,
            score=4.5,
            is_scale_in=False,
            now=now,
        )

        self.assertTrue(fast_entry)

    def test_neutral_strong_signal_can_fast_enter(self):
        cfg = MomentumScalpConfig(
            enable_regime_adaptive=True,
            entry_confirmation_ticks=2,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy._bear_score = 1

        now = datetime.now()
        quote = Quote(
            symbol="005930",
            name="삼성전자",
            current_price=71_000,
            change=900,
            change_rate=strategy._regime_min_change_rate() + 0.2,
            open_price=70_100,
            high_price=71_050,
            low_price=70_000,
            volume=300_000,
            trade_amount=21_300_000_000,
        )

        fast_entry = strategy._can_confirm_entry(
            quote=quote,
            score=strategy._regime_min_momentum_score() + 0.4,
            is_scale_in=False,
            now=now,
        )

        self.assertTrue(fast_entry)

    def test_soft_bear_pullback_filter_relaxes_for_mild_breakout(self):
        cfg = MomentumScalpConfig(
            enable_regime_adaptive=True,
            enable_pullback_entry_filter=True,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        quote = Quote(
            symbol="005930",
            name="삼성전자",
            current_price=10_220,
            change=220,
            change_rate=2.2,
            open_price=10_000,
            high_price=10_225,
            low_price=9_980,
            volume=500_000,
            trade_amount=5_110_000_000,
        )

        strategy._bear_score = 2
        self.assertTrue(strategy._passes_pullback_entry_filter(quote, is_scale_in=False))

        strategy._bear_score = 3
        self.assertFalse(strategy._passes_pullback_entry_filter(quote, is_scale_in=False))

    def test_bullish_trailing_stop_waits_for_meaningful_gain(self):
        cfg = MomentumScalpConfig(
            enable_regime_adaptive=True,
            take_profit_pct=5.0,
            trailing_stop_pct=-0.7,
            trailing_stop_activation_gain_pct=0.8,
            bullish_trailing_stop_activation_gain_pct_floor=1.1,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy._bear_score = 0
        strategy.positions["005930"] = PositionState(
            symbol="005930",
            buy_price=10_000,
            quantity=1,
            invested_amount=10_000,
        )

        pos = strategy.positions["005930"]
        pos.high_since_buy = 10_100  # +1.0%
        quote = Quote(
            symbol="005930",
            name="삼성전자",
            current_price=10_000,
            change=0,
            change_rate=0.0,
            open_price=10_000,
            high_price=10_100,
            low_price=9_980,
            volume=200_000,
            trade_amount=2_000_000_000,
        )

        self.assertIsNone(strategy._evaluate_sell(quote))

    def test_take_profit_defers_when_round_trip_net_is_negative(self):
        cfg = MomentumScalpConfig(
            enable_regime_adaptive=False,
            take_profit_pct=0.1,
            enable_cost_aware_profit_exit=True,
            min_profit_exit_net_pnl=1,
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
            current_price=10_010,
            change=10,
            change_rate=0.1,
            open_price=10_000,
            high_price=10_020,
            low_price=9_990,
            volume=100_000,
            trade_amount=100_100_000,
        )

        self.assertIsNone(strategy._evaluate_sell(quote))

    def test_trailing_stop_defers_when_round_trip_net_is_negative(self):
        cfg = MomentumScalpConfig(
            enable_regime_adaptive=False,
            take_profit_pct=5.0,
            trailing_stop_pct=-0.7,
            trailing_stop_activation_gain_pct=0.5,
            enable_cost_aware_profit_exit=True,
            min_profit_exit_net_pnl=1,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy.positions["005930"] = PositionState(
            symbol="005930",
            buy_price=10_000,
            quantity=10,
            invested_amount=100_000,
        )
        strategy.positions["005930"].high_since_buy = 10_100

        quote = Quote(
            symbol="005930",
            name="삼성전자",
            current_price=10_020,
            change=20,
            change_rate=0.2,
            open_price=10_000,
            high_price=10_100,
            low_price=9_990,
            volume=100_000,
            trade_amount=1_002_000_000,
        )

        self.assertIsNone(strategy._evaluate_sell(quote))

    def test_inverse_take_profit_defers_when_round_trip_net_is_negative(self):
        cfg = MomentumScalpConfig(
            enable_regime_adaptive=False,
            inverse_enabled=True,
            inverse_take_profit_pct=0.1,
            enable_cost_aware_profit_exit=True,
            min_profit_exit_net_pnl=1,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy.positions["252670"] = PositionState(
            symbol="252670",
            buy_price=10_000,
            quantity=1,
            invested_amount=10_000,
        )

        quote = Quote(
            symbol="252670",
            name="KODEX 200선물인버스2X",
            current_price=10_010,
            change=10,
            change_rate=0.1,
            open_price=10_000,
            high_price=10_020,
            low_price=9_990,
            volume=100_000,
            trade_amount=100_100_000,
        )

        self.assertIsNone(strategy._evaluate_inverse_sell(quote))

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

    def test_restored_position_uses_saved_buy_time_for_time_exit(self):
        cfg = MomentumScalpConfig(
            enable_regime_adaptive=False,
            restored_position_grace_seconds=0,
            max_position_holding_minutes=10,
            take_profit_pct=5.0,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        restored_buy_time = datetime.now() - timedelta(minutes=35)
        strategy._loaded_position_meta = {
            "005930": {
                "buy_time": restored_buy_time.isoformat(timespec="seconds"),
                "invested_amount": 10_000,
                "high_since_buy": 10_000,
            }
        }

        strategy.sync_positions_from_account(
            [
                Position(
                    symbol="005930",
                    name="삼성전자",
                    quantity=1,
                    avg_price=10_000.0,
                    current_price=10_050,
                    eval_amount=0,
                    profit_loss=0,
                    profit_rate=0.0,
                )
            ]
        )

        quote = Quote(
            symbol="005930",
            name="삼성전자",
            current_price=10_050,
            change=50,
            change_rate=0.5,
            open_price=10_000,
            high_price=10_100,
            low_price=9_980,
            volume=500_000,
            trade_amount=5_025_000_000,
        )

        orders = strategy.on_batch_tick([quote])

        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].symbol, "005930")
        self.assertEqual(orders[0].side, OrderSide.SELL)

    def test_restored_position_keeps_grace_before_time_exit(self):
        cfg = MomentumScalpConfig(
            enable_regime_adaptive=False,
            restored_position_grace_seconds=300,
            max_position_holding_minutes=10,
            take_profit_pct=5.0,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        restored_buy_time = datetime.now() - timedelta(minutes=35)
        strategy._loaded_position_meta = {
            "005930": {
                "buy_time": restored_buy_time.isoformat(timespec="seconds"),
                "invested_amount": 10_000,
                "high_since_buy": 10_000,
            }
        }

        strategy.sync_positions_from_account(
            [
                Position(
                    symbol="005930",
                    name="삼성전자",
                    quantity=1,
                    avg_price=10_000.0,
                    current_price=10_050,
                    eval_amount=0,
                    profit_loss=0,
                    profit_rate=0.0,
                )
            ]
        )

        quote = Quote(
            symbol="005930",
            name="삼성전자",
            current_price=10_050,
            change=50,
            change_rate=0.5,
            open_price=10_000,
            high_price=10_100,
            low_price=9_980,
            volume=500_000,
            trade_amount=5_025_000_000,
        )

        orders = strategy.on_batch_tick([quote])

        self.assertEqual(orders, [])
        self.assertTrue(strategy.positions["005930"].is_restored)

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
        strategy._bear_score = 3

        for symbol in ("252670", "005930"):
            strategy._latest_tick_volumes[symbol] = 10_500
            strategy._recent_tick_volumes[symbol] = deque([10_000, 10_000, 10_500], maxlen=12)

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
