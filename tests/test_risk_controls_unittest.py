import unittest
from collections import deque
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pandas as pd
from src.api_client import KISClient
from src.config import Config
from src.market_data import MarketDataAPI
from src.models import OrderResult, OrderSide, Position, Quote, RankingItem
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


class DummyRankingMarketData:
    def __init__(self, items):
        self.items = items

    def get_fluctuation_ranking(self, **kwargs):
        return list(self.items)


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

    def test_momentum_allocation_allows_expensive_single_share_for_bull_a_grade(self):
        cfg = MomentumScalpConfig(
            seed_money=1_000_000,
            per_stock_amount=200_000,
            max_per_stock_amount=350_000,
            bull_max_single_position_pct=0.25,
            allow_expensive_single_share_override=True,
            expensive_single_share_min_price=120_000,
            expensive_single_share_cap_multiplier=1.5,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy._bear_score = 0

        blocked = strategy._compute_buy_allocation("AAA", current_price=320_000)
        allowed = strategy._compute_buy_allocation(
            "AAA",
            current_price=320_000,
            allow_expensive_single_share_override=True,
        )

        self.assertEqual(blocked, 0)
        self.assertEqual(allowed, 320_000)

    def test_momentum_allocation_supports_priority_bull_concentration(self):
        cfg = MomentumScalpConfig(
            seed_money=1_000_000,
            max_position_count=2,
            bull_max_position_count=2,
            per_stock_amount=200_000,
            max_per_stock_amount=350_000,
            capital_utilization_pct=0.70,
            bull_max_single_position_pct=0.35,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy._bear_score = 0

        base_alloc = strategy._compute_buy_allocation("AAA", current_price=60_000)
        priority_alloc = strategy._compute_buy_allocation(
            "AAA",
            current_price=60_000,
            per_stock_amount_multiplier=3.0,
            max_per_stock_amount_multiplier=3.0,
            max_single_position_pct_override=0.65,
            side_slot_override=1,
        )

        self.assertEqual(base_alloc, 350_000)
        self.assertEqual(priority_alloc, 650_000)
        self.assertGreater(priority_alloc, base_alloc)

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

    def test_sell_fill_triggers_daily_hard_stop_alert_immediately(self):
        cfg = MomentumScalpConfig(
            daily_loss_limit=-500,
            daily_total_loss_limit=-500,
            enable_regime_adaptive=False,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy.initialize()
        strategy.positions["005930"] = PositionState(
            symbol="005930",
            buy_price=10_000,
            quantity=1,
            invested_amount=10_000,
        )

        with patch.object(strategy._alerts, "send") as mock_send:
            strategy.on_order_filled(
                OrderResult(
                    success=True,
                    symbol="005930",
                    side=OrderSide.SELL,
                    quantity=1,
                    price=9_000,
                )
            )

        self.assertTrue(strategy._halted)
        event_keys = [call.kwargs.get("event_key") for call in mock_send.call_args_list]
        self.assertIn("daily_total_loss_limit_hit", event_keys)

    def test_neutral_chase_block_rejects_day_high_without_pullback(self):
        cfg = MomentumScalpConfig(
            enable_volume_spike_filter=False,
            enable_expected_net_filter=False,
            enable_pool_persistence_gate=False,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy._bear_score = 1
        strategy.set_simulated_now(datetime(2026, 3, 16, 10, 0))

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
        strategy.set_simulated_now(datetime(2026, 3, 16, 10, 0))

        history = [
            Quote("011700", "미원", 10_000, 0, 0.0, 10_000, 10_000, 9_950, 10_000, 100_000),
            Quote("011700", "미원", 10_200, 200, 2.0, 10_000, 10_200, 9_950, 20_000, 200_000),
            Quote("011700", "미원", 10_120, 120, 1.2, 10_000, 10_220, 9_950, 30_000, 300_000),
            Quote("011700", "미원", 10_150, 150, 1.5, 10_000, 10_220, 9_950, 35_000, 350_000),
            Quote("011700", "미원", 10_220, 220, 2.2, 10_000, 10_220, 9_950, 40_000, 400_000),
        ]
        for quote in history:
            strategy._record_recent_quote(quote)

        strategy.daily_pnl.trade_count = 1
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
            soft_bear_inverse_min_runup_pct=0.6,
            soft_bear_inverse_min_drop_pct=0.15,
            soft_bear_inverse_max_drop_pct=0.8,
            soft_bear_inverse_reclaim_buffer_pct=0.03,
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

    def test_soft_bear_profile_rejects_long_entries(self):
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(
                enable_volume_spike_filter=False,
                enable_expected_net_filter=False,
                enable_pool_persistence_gate=False,
            ),
        )
        strategy._bear_score = 2
        quote = Quote(
            "005930",
            "삼성전자",
            60_000,
            600,
            1.0,
            59_500,
            60_100,
            59_400,
            500_000,
            30_000_000_000,
        )

        self.assertFalse(strategy._can_open_new_long(quote, score=3.0))

    def test_soft_bear_strong_leader_lane_ranks_candidate(self):
        cfg = MomentumScalpConfig(
            enable_volume_spike_filter=False,
            enable_expected_net_filter=False,
            enable_pool_persistence_gate=False,
            bull_leader_top_n=3,
            bull_leader_relative_strength_pp=0.0,
            bull_breakout_hold_ticks=2,
            bull_breakout_buffer_pct=0.03,
            soft_bear_strong_leader_min_change_rate=3.0,
            soft_bear_strong_leader_min_momentum=2.5,
            soft_bear_strong_leader_min_trade_amount=1_500_000_000,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy._bear_score = 2
        leader = [
            Quote("263750", "펄어비스", 60_000, 0, 0.0, 60_000, 60_000, 59_500, 20_000, 1_200_000_000),
            Quote("263750", "펄어비스", 61_500, 1_500, 2.5, 60_000, 61_500, 59_500, 30_000, 1_845_000_000),
            Quote("263750", "펄어비스", 62_800, 2_800, 4.7, 60_000, 62_800, 59_500, 45_000, 2_826_000_000),
            Quote("263750", "펄어비스", 63_200, 3_200, 5.3, 60_000, 63_200, 59_500, 55_000, 3_476_000_000),
        ]
        others = [
            Quote("005930", "삼성전자", 60_500, 500, 0.8, 60_000, 60_550, 59_900, 120_000, 7_260_000_000),
            Quote("000660", "SK하이닉스", 81_200, 600, 0.7, 80_600, 81_250, 80_400, 90_000, 7_308_000_000),
        ]
        for quote in leader + others:
            strategy._quotes_cache[quote.symbol] = quote
            strategy._record_recent_quote(quote)
        strategy._pool = [quote.symbol for quote in leader[-1:] + others]
        strategy._latest_strong_leader_symbols = {"263750"}
        strategy._latest_strong_leader_snapshot = {
            "263750": {
                "change_rate": leader[-1].change_rate,
                "trade_amount": leader[-1].trade_amount,
                "rank": 1,
            }
        }

        ranked = strategy._rank_long_entry_candidates([leader[-1], *others])

        self.assertEqual([quote.symbol for _, quote in ranked], ["263750"])

    def test_direct_dynamic_symbol_bypasses_persistence_gate(self):
        cfg = MomentumScalpConfig(
            enable_volume_spike_filter=False,
            enable_expected_net_filter=False,
            enable_pool_persistence_gate=True,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy._bear_score = 0
        strategy._latest_direct_dynamic_symbols = {"047040"}
        quote = Quote(
            "047040",
            "대우건설",
            4_500,
            220,
            5.1,
            4_280,
            4_520,
            4_240,
            1_200_000,
            5_400_000_000,
        )

        self.assertTrue(strategy._can_open_new_long(quote, score=3.5))

    def test_neutral_leader_filter_blocks_low_rank_candidate(self):
        cfg = MomentumScalpConfig(
            enable_volume_spike_filter=False,
            enable_expected_net_filter=False,
            enable_pool_persistence_gate=False,
            neutral_leader_top_n=2,
            neutral_leader_relative_strength_pp=0.0,
            neutral_pullback_min_ticks=1,
            enable_math_shadow_layer=False,
            enable_math_live_layer=False,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy._bear_score = 1
        strategy.set_simulated_now(datetime(2026, 3, 16, 10, 0))

        pool_quotes = [
            Quote("AAA", "A", 10_300, 300, 3.0, 10_000, 10_300, 9_950, 200_000, 2_060_000_000),
            Quote("BBB", "B", 10_250, 250, 2.5, 10_000, 10_250, 9_950, 180_000, 1_845_000_000),
            Quote("CCC", "C", 10_220, 220, 2.2, 10_000, 10_220, 9_950, 50_000, 511_000_000),
        ]
        for quote in pool_quotes:
            strategy._quotes_cache[quote.symbol] = quote
            strategy._pool.append(quote.symbol)
        history = [
            Quote("CCC", "C", 10_000, 0, 0.0, 10_000, 10_000, 9_950, 10_000, 100_000_000),
            Quote("CCC", "C", 10_260, 260, 2.6, 10_000, 10_260, 9_950, 20_000, 200_000_000),
            Quote("CCC", "C", 10_170, 170, 1.7, 10_000, 10_260, 9_950, 25_000, 250_000_000),
            Quote("CCC", "C", 10_280, 280, 2.8, 10_000, 10_280, 9_950, 30_000, 300_000_000),
        ]
        for quote in history:
            strategy._record_recent_quote(quote)

        strategy.daily_pnl.trade_count = 1
        ok, _, reject = strategy._passes_neutral_pullback_reclaim_setup(history[-1], score=2.8)
        self.assertFalse(ok)
        self.assertEqual(reject, "neutral_low_turnover_rank")

    def test_order_fill_preserves_entry_metadata(self):
        cfg = MomentumScalpConfig(
            enable_volume_spike_filter=False,
            enable_expected_net_filter=False,
            enable_pool_persistence_gate=False,
            enable_entry_confirmation=False,
            enable_math_shadow_layer=False,
            enable_math_live_layer=False,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy._bear_score = 1
        strategy.set_simulated_now(datetime(2026, 3, 16, 10, 0))

        history = [
            Quote("011700", "미원", 10_000, 0, 0.0, 10_000, 10_000, 9_950, 10_000, 100_000),
            Quote("011700", "미원", 10_200, 200, 2.0, 10_000, 10_200, 9_950, 20_000, 200_000),
            Quote("011700", "미원", 10_120, 120, 1.2, 10_000, 10_220, 9_950, 30_000, 300_000),
            Quote("011700", "미원", 10_150, 150, 1.5, 10_000, 10_220, 9_950, 35_000, 350_000),
            Quote("011700", "미원", 10_220, 220, 2.2, 10_000, 10_220, 9_950, 40_000, 400_000),
        ]
        for quote in history:
            strategy._quotes_cache[quote.symbol] = quote
            strategy._record_recent_quote(quote)

        strategy.daily_pnl.trade_count = 1
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

    def test_neutral_pullback_reclaim_rejects_too_early_in_session(self):
        cfg = MomentumScalpConfig(
            enable_volume_spike_filter=False,
            enable_expected_net_filter=False,
            enable_pool_persistence_gate=False,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy._bear_score = 1
        strategy.set_simulated_now(datetime(2026, 3, 16, 9, 20))

        history = [
            Quote("011700", "미원", 10_000, 0, 0.0, 10_000, 10_000, 9_950, 10_000, 100_000),
            Quote("011700", "미원", 10_200, 200, 2.0, 10_000, 10_200, 9_950, 20_000, 200_000),
            Quote("011700", "미원", 10_120, 120, 1.2, 10_000, 10_220, 9_950, 30_000, 300_000),
            Quote("011700", "미원", 10_220, 220, 2.2, 10_000, 10_220, 9_950, 40_000, 400_000),
        ]
        for quote in history:
            strategy._record_recent_quote(quote)

        ok, _, reject = strategy._passes_neutral_pullback_reclaim_setup(history[-1], score=2.1)
        self.assertFalse(ok)
        self.assertEqual(reject, "neutral_too_early")

    def test_bull_bias_override_routes_neutral_market_to_bull_breakout(self):
        cfg = MomentumScalpConfig(
            enable_volume_spike_filter=False,
            enable_expected_net_filter=False,
            enable_pool_persistence_gate=False,
            bull_breakout_hold_ticks=2,
            bull_breakout_buffer_pct=0.03,
            bull_leader_top_n=4,
            bull_leader_relative_strength_pp=0.40,
            bull_bias_avg_change_rate_threshold=0.7,
            bull_bias_max_decliner_ratio=0.45,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy._bear_score = 1
        strategy._cached_index_regime_info = (5_700.0, 5_620.0, 5_650.0)
        strategy.set_simulated_now(datetime(2026, 3, 18, 10, 30))

        leader_history = [
            Quote("090710", "휴림로봇", 10_000, 0, 0.0, 10_000, 10_000, 9_950, 80_000, 800_000_000),
            Quote("090710", "휴림로봇", 10_100, 100, 1.0, 10_000, 10_100, 9_950, 120_000, 1_212_000_000),
            Quote("090710", "휴림로봇", 10_200, 200, 2.0, 10_000, 10_200, 9_950, 180_000, 1_836_000_000),
            Quote("090710", "휴림로봇", 10_220, 220, 2.2, 10_000, 10_220, 9_950, 240_000, 2_452_800_000),
            Quote("090710", "휴림로봇", 10_240, 240, 2.4, 10_000, 10_240, 9_950, 300_000, 3_072_000_000),
        ]
        for quote in leader_history:
            strategy._record_recent_quote(quote)
            strategy._quotes_cache[quote.symbol] = quote

        for symbol, change_rate, volume in (
            ("005930", 0.8, 150_000),
            ("000660", 0.9, 140_000),
            ("035420", 1.1, 130_000),
        ):
            quote = Quote(
                symbol,
                symbol,
                10_000,
                100,
                change_rate,
                9_900,
                10_050,
                9_850,
                volume,
                1_000_000_000,
            )
            strategy._quotes_cache[symbol] = quote

        self.assertEqual(strategy._current_profile_entry_strategy_name(is_inverse=False), "bull_breakout_strategy")
        score = strategy._calc_momentum_score(leader_history[-1])
        decision = strategy._regime_router.evaluate_long_entry(strategy, leader_history[-1], score)

        self.assertTrue(decision.allowed)
        self.assertEqual(decision.strategy_name, "bull_breakout_strategy")
        self.assertIn("entry_grade=A", decision.payload)

    def test_bull_breakout_requires_higher_score_in_late_session(self):
        cfg = MomentumScalpConfig(
            enable_volume_spike_filter=False,
            enable_expected_net_filter=False,
            enable_pool_persistence_gate=False,
            bullish_min_momentum_score=3.0,
            bullish_min_momentum_score_floor=3.0,
            bull_breakout_hold_ticks=2,
            bull_breakout_buffer_pct=0.03,
            bull_breakout_late_entry_start_minutes_after_open=255,
            bull_breakout_late_entry_score_bonus=0.35,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy._bear_score = 0
        strategy.set_simulated_now(datetime(2026, 3, 18, 13, 45))

        history = [
            Quote("090710", "휴림로봇", 10_000, 0, 0.0, 10_000, 10_000, 9_950, 80_000, 800_000_000),
            Quote("090710", "휴림로봇", 10_100, 100, 1.0, 10_000, 10_100, 9_950, 120_000, 1_212_000_000),
            Quote("090710", "휴림로봇", 10_200, 200, 2.0, 10_000, 10_200, 9_950, 180_000, 1_836_000_000),
            Quote("090710", "휴림로봇", 10_220, 220, 2.2, 10_000, 10_220, 9_950, 240_000, 2_452_800_000),
            Quote("090710", "휴림로봇", 10_240, 240, 2.4, 10_000, 10_240, 9_950, 300_000, 3_072_000_000),
        ]
        for quote in history:
            strategy._record_recent_quote(quote)
            strategy._quotes_cache[quote.symbol] = quote

        for symbol, change_rate, volume in (
            ("005930", 0.8, 150_000),
            ("000660", 0.9, 140_000),
            ("035420", 1.1, 130_000),
        ):
            quote = Quote(
                symbol,
                symbol,
                10_000,
                100,
                change_rate,
                9_900,
                10_050,
                9_850,
                volume,
                1_000_000_000,
            )
            strategy._quotes_cache[symbol] = quote

        ok, _, reject = strategy._passes_bull_breakout_setup(history[-1], score=3.0)
        self.assertFalse(ok)
        self.assertEqual(reject, "bull_late_day_score")

    def test_strong_bull_override_caps_soft_bear_to_neutral(self):
        cfg = MomentumScalpConfig(
            enable_volume_spike_filter=False,
            enable_expected_net_filter=False,
            enable_pool_persistence_gate=False,
            strong_bull_override_index_gap_pct=1.5,
            strong_bull_override_avg_change_rate_threshold=2.0,
            strong_bull_override_max_decliner_ratio=0.25,
            strong_bull_override_min_quote_count=8,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy.market_data = object()
        now = datetime(2026, 3, 18, 12, 30)
        strategy.set_simulated_now(now)
        strategy._bear_score = 2
        strategy._last_regime_check_at = now - timedelta(seconds=30)
        strategy._last_index_regime_check_at = now
        strategy._cached_index_regime_score = 2
        strategy._cached_index_regime_info = (5850.0, 5700.0, 5620.0)

        quotes = []
        for idx in range(8):
            quotes.append(
                Quote(
                    symbol=f"A{idx:03d}",
                    name=f"A{idx:03d}",
                    current_price=10_000 + idx,
                    change=300,
                    change_rate=3.0,
                    open_price=9_700,
                    high_price=10_050 + idx,
                    low_price=9_650,
                    volume=200_000 + idx * 1_000,
                    trade_amount=2_500_000_000 + idx * 10_000_000,
                )
            )

        strategy._check_market_regime(quotes=quotes)

        self.assertEqual(strategy._bear_score, 1)
        self.assertTrue(strategy._strong_bull_override_active)

    def test_index_support_bull_bias_override_reclassifies_soft_bear(self):
        cfg = MomentumScalpConfig(
            enable_volume_spike_filter=False,
            enable_expected_net_filter=False,
            enable_pool_persistence_gate=False,
            index_support_bull_bias_index_gap_pct=1.0,
            index_support_bull_bias_avg_change_rate_threshold=1.0,
            index_support_bull_bias_max_decliner_ratio=0.55,
            index_support_bull_bias_min_quote_count=8,
            strong_bull_override_index_gap_pct=3.0,
            strong_bull_override_avg_change_rate_threshold=3.0,
            strong_bull_override_max_decliner_ratio=0.20,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy.market_data = object()
        now = datetime(2026, 3, 19, 12, 15)
        strategy.set_simulated_now(now)
        strategy._bear_score = 2
        strategy._last_regime_check_at = now - timedelta(seconds=30)
        strategy._last_index_regime_check_at = now
        strategy._cached_index_regime_score = 2
        strategy._cached_index_regime_info = (5858.5, 5721.5, 5692.2)

        quotes = []
        for idx in range(10):
            change_rate = 1.8 if idx < 5 else (1.2 if idx < 8 else -0.2)
            change = int(10_000 * (change_rate / 100))
            quotes.append(
                Quote(
                    symbol=f"B{idx:03d}",
                    name=f"B{idx:03d}",
                    current_price=10_000 + change,
                    change=change,
                    change_rate=change_rate,
                    open_price=10_000,
                    high_price=10_200,
                    low_price=9_900,
                    volume=150_000 + idx * 1_000,
                    trade_amount=1_600_000_000 + idx * 50_000_000,
                )
            )

        strategy._check_market_regime(quotes=quotes)

        self.assertEqual(strategy._bear_score, 1)
        self.assertTrue(strategy._index_support_bull_bias_active)
        self.assertTrue(strategy._is_bull_bias_market())
        self.assertEqual(strategy._resolve_regime_profile_name(), "bull")

    def test_leader_support_bull_bias_override_reclassifies_soft_bear(self):
        cfg = MomentumScalpConfig(
            enable_volume_spike_filter=False,
            enable_expected_net_filter=False,
            enable_pool_persistence_gate=False,
            leader_support_bull_bias_min_count=1,
            leader_support_bull_bias_min_change_rate=4.0,
            leader_support_bull_bias_min_trade_amount=2_000_000_000,
            leader_support_bull_bias_max_decliner_ratio=0.70,
            strong_bull_override_index_gap_pct=9.0,
            strong_bull_override_avg_change_rate_threshold=9.0,
            index_support_bull_bias_index_gap_pct=9.0,
            index_support_bull_bias_avg_change_rate_threshold=9.0,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy.market_data = object()
        now = datetime(2026, 3, 19, 10, 30)
        strategy.set_simulated_now(now)
        strategy._bear_score = 2
        strategy._last_regime_check_at = now - timedelta(seconds=30)
        strategy._last_index_regime_check_at = now
        strategy._cached_index_regime_score = 2
        strategy._cached_index_regime_info = (5850.0, 5830.0, 5825.0)
        strategy._latest_strong_leader_snapshot = {
            "263750": {
                "change_rate": 6.2,
                "trade_amount": 3_400_000_000,
                "rank": 1,
            },
        }
        strategy._latest_strong_leader_symbols = {"263750"}

        quotes = [
            Quote(
                symbol=f"L{idx:03d}",
                name=f"L{idx:03d}",
                current_price=10_000,
                change=100 if idx < 7 else -50,
                change_rate=1.0 if idx < 7 else -0.5,
                open_price=9_900,
                high_price=10_050,
                low_price=9_850,
                volume=150_000 + idx * 1_000,
                trade_amount=1_200_000_000 + idx * 50_000_000,
            )
            for idx in range(10)
        ]

        strategy._check_market_regime(quotes=quotes)

        self.assertEqual(strategy._bear_score, 1)
        self.assertTrue(strategy._leader_support_bull_bias_active)

    def test_invalid_zero_index_response_keeps_previous_index_cache(self):
        cfg = MomentumScalpConfig(
            enable_volume_spike_filter=False,
            enable_expected_net_filter=False,
            enable_pool_persistence_gate=False,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy.market_data = Mock()
        strategy.market_data.get_index_daily_prices.return_value = pd.DataFrame(
            {
                "stck_bsop_date": [20260319 - idx for idx in range(20)],
                "bstp_nmix_prpr": [0 for _ in range(20)],
            }
        )
        now = datetime(2026, 3, 19, 12, 30)
        strategy.set_simulated_now(now)
        strategy._last_index_regime_check_at = now - timedelta(minutes=10)
        strategy._cached_index_regime_score = 2
        strategy._cached_index_regime_info = (5858.5, 5721.5, 5692.2)

        quotes = [
            Quote(
                symbol=f"C{idx:03d}",
                name=f"C{idx:03d}",
                current_price=10_100,
                change=100,
                change_rate=1.0,
                open_price=10_000,
                high_price=10_150,
                low_price=9_980,
                volume=120_000 + idx * 1_000,
                trade_amount=1_200_000_000 + idx * 10_000_000,
            )
            for idx in range(8)
        ]

        strategy._check_market_regime(quotes=quotes, force=True)

        self.assertEqual(strategy._cached_index_regime_score, 2)
        self.assertEqual(strategy._cached_index_regime_info, (5858.5, 5721.5, 5692.2))

    def test_market_data_warmup_blocks_new_entries_until_ready(self):
        cfg = MomentumScalpConfig(
            inverse_enabled=False,
            enable_volume_spike_filter=False,
            enable_expected_net_filter=False,
            enable_pool_persistence_gate=False,
            startup_market_data_ready_ticks=2,
            startup_market_data_min_valid_quote_count=2,
        )
        strategy = MomentumScalpStrategy(market_data=Mock(), config=cfg)
        now = datetime(2026, 3, 19, 9, 5)
        strategy.set_simulated_now(now)
        strategy._last_index_regime_check_at = now

        quote = Quote(
            symbol="090710",
            name="휴림로봇",
            current_price=10_200,
            change=200,
            change_rate=2.0,
            open_price=10_000,
            high_price=10_220,
            low_price=9_950,
            volume=180_000,
            trade_amount=1_836_000_000,
        )
        strategy._rank_long_entry_candidates = Mock(return_value=[(5.0, quote)])
        strategy._evaluate_buy = Mock()

        orders = strategy.on_batch_tick([quote])

        self.assertEqual(orders, [])
        strategy._rank_long_entry_candidates.assert_not_called()
        strategy._evaluate_buy.assert_not_called()
        self.assertFalse(strategy._market_data_ready_for_entries)

    def test_market_data_warmup_allows_sell_while_blocking_new_entries(self):
        cfg = MomentumScalpConfig(
            inverse_enabled=False,
            enable_volume_spike_filter=False,
            enable_expected_net_filter=False,
            enable_pool_persistence_gate=False,
        )
        strategy = MomentumScalpStrategy(market_data=Mock(), config=cfg)
        now = datetime(2026, 3, 19, 9, 6)
        strategy.set_simulated_now(now)
        strategy._last_index_regime_check_at = now
        strategy.positions["005930"] = PositionState(
            symbol="005930",
            buy_price=10_000,
            quantity=1,
            invested_amount=10_000,
        )

        quote = Quote(
            symbol="005930",
            name="삼성전자",
            current_price=7_000,
            change=-3_000,
            change_rate=-30.0,
            open_price=10_000,
            high_price=10_050,
            low_price=6_950,
            volume=500_000,
            trade_amount=3_500_000_000,
        )

        orders = strategy.on_batch_tick([quote])

        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].side, OrderSide.SELL)
        self.assertEqual(orders[0].symbol, "005930")
        self.assertFalse(strategy._market_data_ready_for_entries)

    def test_market_data_warmup_requires_consecutive_valid_ticks(self):
        cfg = MomentumScalpConfig(
            inverse_enabled=False,
            startup_market_data_ready_ticks=2,
            startup_market_data_min_valid_quote_count=2,
        )
        strategy = MomentumScalpStrategy(market_data=Mock(), config=cfg)
        now = datetime(2026, 3, 19, 9, 7)
        strategy.set_simulated_now(now)
        strategy._cached_index_regime_info = (5858.5, 5721.5, 5692.2)

        quotes = [
            Quote("AAA", "A", 10_100, 100, 1.0, 10_000, 10_150, 9_980, 100_000, 1_010_000_000),
            Quote("BBB", "B", 10_200, 200, 2.0, 10_000, 10_220, 9_970, 120_000, 1_224_000_000),
        ]

        first = strategy._update_market_data_readiness(quotes, now)
        second = strategy._update_market_data_readiness(quotes, now + timedelta(seconds=1))

        self.assertFalse(first)
        self.assertTrue(second)
        self.assertTrue(strategy._market_data_ready_for_entries)

    def test_bull_post_loss_requires_stronger_score(self):
        cfg = MomentumScalpConfig(
            enable_volume_spike_filter=False,
            enable_expected_net_filter=False,
            enable_pool_persistence_gate=False,
            enable_backtest_score_entry_fallback=False,
            enable_math_shadow_layer=False,
            enable_math_live_layer=False,
            bullish_min_momentum_score=3.0,
            bullish_min_momentum_score_floor=3.0,
            bull_breakout_hold_ticks=2,
            bull_breakout_buffer_pct=0.03,
            bull_post_loss_score_bonus=0.30,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy._bear_score = 0
        strategy.set_simulated_now(datetime(2026, 3, 18, 11, 0))

        history = [
            Quote("090710", "휴림로봇", 10_000, 0, 0.0, 10_000, 10_000, 9_950, 80_000, 800_000_000),
            Quote("090710", "휴림로봇", 10_100, 100, 1.0, 10_000, 10_100, 9_950, 120_000, 1_212_000_000),
            Quote("090710", "휴림로봇", 10_200, 200, 2.0, 10_000, 10_200, 9_950, 180_000, 1_836_000_000),
            Quote("090710", "휴림로봇", 10_220, 220, 2.2, 10_000, 10_220, 9_950, 240_000, 2_452_800_000),
            Quote("090710", "휴림로봇", 10_240, 240, 2.4, 10_000, 10_240, 9_950, 300_000, 3_072_000_000),
        ]
        for quote in history:
            strategy._record_recent_quote(quote)
            strategy._quotes_cache[quote.symbol] = quote

        for symbol, change_rate, volume in (
            ("005930", 0.8, 150_000),
            ("000660", 0.9, 140_000),
            ("035420", 1.1, 130_000),
        ):
            quote = Quote(
                symbol,
                symbol,
                10_000,
                100,
                change_rate,
                9_900,
                10_050,
                9_850,
                volume,
                1_000_000_000,
            )
            strategy._quotes_cache[symbol] = quote

        allowed_order = strategy._evaluate_buy(history[-1], score_hint=3.0)
        self.assertIsNotNone(allowed_order)

        strategy._bull_loss_count_today = 1
        blocked_order = strategy._evaluate_buy(history[-1], score_hint=3.0)
        self.assertIsNone(blocked_order)

    def test_build_pool_mixes_in_high_turnover_cached_leader(self):
        cfg = MomentumScalpConfig(
            dynamic_pool_size=2,
            dynamic_pool_ranking_fetch_count=4,
            dynamic_pool_turnover_slots=1,
            dynamic_pool_quote_trade_amount_slots=1,
            dynamic_pool_quote_min_change_rate=0.8,
            enable_pool_persistence_gate=False,
        )
        ranking_items = [
            RankingItem("AAA", "AAA", 10_000, 5.0, 10_000, 1),
            RankingItem("BBB", "BBB", 9_000, 4.5, 12_000, 2),
            RankingItem("CCC", "CCC", 8_000, 4.0, 150_000, 3),
            RankingItem("DDD", "DDD", 7_000, 3.5, 11_000, 4),
        ]
        strategy = MomentumScalpStrategy(
            market_data=DummyRankingMarketData(ranking_items),
            config=cfg,
        )
        strategy._quotes_cache["ZZZ"] = Quote(
            "ZZZ",
            "ZZZ",
            12_000,
            360,
            3.1,
            11_640,
            12_120,
            11_600,
            500_000,
            6_000_000_000,
        )

        strategy._build_pool()

        self.assertIn("AAA", strategy._pool)
        self.assertIn("CCC", strategy._pool)
        self.assertIn("ZZZ", strategy._pool)

    def test_build_pool_directly_includes_turnover_leader_with_persistence_gate(self):
        cfg = MomentumScalpConfig(
            dynamic_pool_size=2,
            dynamic_pool_ranking_fetch_count=4,
            dynamic_pool_turnover_slots=2,
            dynamic_pool_quote_trade_amount_slots=0,
            dynamic_pool_direct_turnover_slots=1,
            dynamic_pool_direct_quote_leader_slots=0,
            enable_pool_persistence_gate=True,
            momentum_pool_persistence_window=3,
            momentum_pool_min_appearances=2,
        )
        ranking_items = [
            RankingItem("AAA", "AAA", 10_000, 5.0, 10_000, 1),
            RankingItem("BBB", "BBB", 9_000, 4.5, 12_000, 2),
            RankingItem("CCC", "CCC", 40_000, 4.0, 150_000, 3),
            RankingItem("DDD", "DDD", 7_000, 3.5, 11_000, 4),
        ]
        strategy = MomentumScalpStrategy(
            market_data=DummyRankingMarketData(ranking_items),
            config=cfg,
        )

        strategy._build_pool()

        self.assertIn("CCC", strategy._pool)

    def test_build_pool_directly_includes_top_rank_leader_with_persistence_gate(self):
        cfg = MomentumScalpConfig(
            dynamic_pool_size=2,
            dynamic_pool_ranking_fetch_count=4,
            dynamic_pool_turnover_slots=0,
            dynamic_pool_quote_trade_amount_slots=0,
            dynamic_pool_direct_rank_slots=1,
            dynamic_pool_direct_turnover_slots=0,
            dynamic_pool_direct_quote_leader_slots=0,
            enable_pool_persistence_gate=True,
            momentum_pool_persistence_window=3,
            momentum_pool_min_appearances=2,
        )
        ranking_items = [
            RankingItem("AAA", "AAA", 10_000, 5.0, 10_000, 1),
            RankingItem("BBB", "BBB", 9_000, 4.5, 12_000, 2),
        ]
        strategy = MomentumScalpStrategy(
            market_data=DummyRankingMarketData(ranking_items),
            config=cfg,
        )

        strategy._build_pool()

        self.assertIn("AAA", strategy._pool)
        self.assertIn("AAA", strategy._latest_direct_dynamic_symbols)

    def test_build_pool_tracks_strong_leader_snapshot(self):
        cfg = MomentumScalpConfig(
            dynamic_pool_size=2,
            dynamic_pool_ranking_fetch_count=4,
            dynamic_pool_turnover_slots=1,
            dynamic_pool_quote_trade_amount_slots=0,
            enable_pool_persistence_gate=False,
            strong_leader_min_change_rate=2.0,
            strong_leader_min_trade_amount=1_000_000_000,
            strong_leader_top_rank=5,
        )
        ranking_items = [
            RankingItem("AAA", "AAA", 20_000, 4.5, 80_000, 4),
            RankingItem("BBB", "BBB", 9_000, 1.5, 12_000, 2),
        ]
        strategy = MomentumScalpStrategy(
            market_data=DummyRankingMarketData(ranking_items),
            config=cfg,
        )

        strategy._build_pool()

        self.assertIn("AAA", strategy._latest_strong_leader_symbols)

    def test_bull_a_grade_initial_entry_uses_scaled_allocation(self):
        base_kwargs = dict(
            seed_money=1_000_000,
            max_position_count=2,
            per_stock_amount=220_000,
            max_per_stock_amount=500_000,
            capital_utilization_pct=0.60,
            bull_max_single_position_pct=0.42,
            enable_volume_spike_filter=False,
            enable_expected_net_filter=False,
            enable_pool_persistence_gate=False,
            enable_backtest_score_entry_fallback=False,
            enable_math_shadow_layer=False,
            enable_math_live_layer=False,
            bullish_min_momentum_score=3.0,
            bullish_min_momentum_score_floor=3.0,
            bull_breakout_hold_ticks=2,
            bull_breakout_buffer_pct=0.03,
            bull_priority_turnover_rank_max=1,
        )
        scaled_strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(**base_kwargs, bull_breakout_initial_entry_scale=0.5),
        )
        full_strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(**base_kwargs, bull_breakout_initial_entry_scale=1.0),
        )
        scaled_strategy._bear_score = 0
        full_strategy._bear_score = 0
        now = datetime(2026, 3, 18, 11, 5)
        scaled_strategy.set_simulated_now(now)
        full_strategy.set_simulated_now(now)

        history = [
            Quote("090710", "휴림로봇", 10_000, 0, 0.0, 10_000, 10_000, 9_950, 80_000, 800_000_000),
            Quote("090710", "휴림로봇", 10_100, 100, 1.0, 10_000, 10_100, 9_950, 120_000, 1_212_000_000),
            Quote("090710", "휴림로봇", 10_200, 200, 2.0, 10_000, 10_200, 9_950, 180_000, 1_836_000_000),
            Quote("090710", "휴림로봇", 10_220, 220, 2.2, 10_000, 10_220, 9_950, 240_000, 2_452_800_000),
            Quote("090710", "휴림로봇", 10_240, 240, 2.4, 10_000, 10_240, 9_950, 300_000, 3_072_000_000),
        ]
        for quote in history:
            scaled_strategy._record_recent_quote(quote)
            scaled_strategy._quotes_cache[quote.symbol] = quote
            full_strategy._record_recent_quote(quote)
            full_strategy._quotes_cache[quote.symbol] = quote

        for symbol, price, change_rate, volume in (
            ("005930", 70_000, 0.8, 50_000),
            ("000660", 10_000, 0.9, 140_000),
            ("035420", 10_000, 1.1, 130_000),
        ):
            quote = Quote(
                symbol,
                symbol,
                price,
                100,
                change_rate,
                price - 100,
                price + 50,
                price - 150,
                volume,
                price * volume,
            )
            scaled_strategy._quotes_cache[symbol] = quote
            full_strategy._quotes_cache[symbol] = quote

        scaled_order = scaled_strategy._evaluate_buy(history[-1])
        full_order = full_strategy._evaluate_buy(history[-1])
        self.assertIsNotNone(scaled_order)
        self.assertIsNotNone(full_order)
        self.assertLess(scaled_order.quantity, full_order.quantity)

    def test_bull_priority_entry_sizes_top_leader_more_aggressively(self):
        base_kwargs = dict(
            seed_money=1_000_000,
            max_position_count=2,
            bull_max_position_count=2,
            per_stock_amount=200_000,
            max_per_stock_amount=350_000,
            capital_utilization_pct=0.70,
            bull_max_single_position_pct=0.35,
            enable_volume_spike_filter=False,
            enable_expected_net_filter=False,
            enable_pool_persistence_gate=False,
            enable_backtest_score_entry_fallback=False,
            enable_math_shadow_layer=False,
            enable_math_live_layer=False,
            bullish_min_momentum_score=3.0,
            bullish_min_momentum_score_floor=3.0,
            bull_breakout_hold_ticks=2,
            bull_breakout_buffer_pct=0.03,
        )
        priority_strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(
                **base_kwargs,
                bull_priority_turnover_rank_max=2,
                bull_priority_per_stock_amount_multiplier=3.0,
                bull_priority_max_per_stock_amount_multiplier=3.0,
                bull_priority_max_single_position_pct=0.65,
                bull_priority_effective_slots=1,
                bull_priority_initial_entry_scale=0.85,
            ),
        )
        plain_strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(
                **base_kwargs,
                bull_priority_turnover_rank_max=1,
                bull_priority_per_stock_amount_multiplier=1.0,
                bull_priority_max_per_stock_amount_multiplier=1.0,
                bull_priority_max_single_position_pct=0.35,
                bull_priority_effective_slots=2,
                bull_priority_initial_entry_scale=0.65,
            ),
        )
        priority_strategy._bear_score = 0
        plain_strategy._bear_score = 0
        now = datetime(2026, 3, 18, 10, 30)
        priority_strategy.set_simulated_now(now)
        plain_strategy.set_simulated_now(now)

        leader_history = [
            Quote("263750", "펄어비스", 60_000, 0, 0.0, 60_000, 60_000, 59_500, 30_000, 1_800_000_000),
            Quote("263750", "펄어비스", 60_800, 800, 1.33, 60_000, 60_800, 59_500, 45_000, 2_736_000_000),
            Quote("263750", "펄어비스", 61_400, 1_400, 2.33, 60_000, 61_400, 59_500, 60_000, 3_684_000_000),
            Quote("263750", "펄어비스", 61_800, 1_800, 3.00, 60_000, 61_800, 59_500, 72_000, 4_449_600_000),
            Quote("263750", "펄어비스", 62_000, 2_000, 3.33, 60_000, 62_000, 59_500, 78_000, 4_836_000_000),
        ]
        for quote in leader_history:
            priority_strategy._record_recent_quote(quote)
            priority_strategy._quotes_cache[quote.symbol] = quote
            plain_strategy._record_recent_quote(quote)
            plain_strategy._quotes_cache[quote.symbol] = quote

        higher_turnover = Quote(
            "005930",
            "삼성전자",
            70_000,
            700,
            1.0,
            69_300,
            70_100,
            69_000,
            80_000,
            5_600_000_000,
        )
        for strategy in (priority_strategy, plain_strategy):
            strategy._quotes_cache[higher_turnover.symbol] = higher_turnover
            for symbol, change_rate, price, volume in (
                ("000660", 0.9, 40_000, 20_000),
                ("035420", 1.1, 25_000, 18_000),
            ):
                quote = Quote(
                    symbol,
                    symbol,
                    price,
                    100,
                    change_rate,
                    price - 200,
                    price + 100,
                    price - 300,
                    volume,
                    price * volume,
                )
                strategy._quotes_cache[symbol] = quote

        priority_order = priority_strategy._evaluate_buy(leader_history[-1])
        plain_order = plain_strategy._evaluate_buy(leader_history[-1])

        self.assertIsNotNone(priority_order)
        self.assertIsNotNone(plain_order)
        self.assertGreater(priority_order.quantity, plain_order.quantity)

    def test_neutral_first_entry_requires_stronger_quality_than_later_entries(self):
        cfg = MomentumScalpConfig(
            enable_volume_spike_filter=False,
            enable_expected_net_filter=False,
            enable_pool_persistence_gate=False,
            neutral_first_entry_score_bonus=0.4,
            neutral_first_entry_change_rate_bonus=0.2,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy._bear_score = 1
        strategy.set_simulated_now(datetime(2026, 3, 16, 10, 0))

        history = [
            Quote("011700", "미원", 10_000, 0, 0.0, 10_000, 10_000, 9_950, 10_000, 100_000),
            Quote("011700", "미원", 10_200, 200, 2.0, 10_000, 10_200, 9_950, 20_000, 200_000),
            Quote("011700", "미원", 10_120, 120, 1.2, 10_000, 10_220, 9_950, 30_000, 300_000),
            Quote("011700", "미원", 10_150, 150, 1.5, 10_000, 10_220, 9_950, 35_000, 350_000),
            Quote("011700", "미원", 10_220, 220, 2.2, 10_000, 10_220, 9_950, 40_000, 400_000),
        ]
        for quote in history:
            strategy._record_recent_quote(quote)

        ok, _, reject = strategy._passes_neutral_pullback_reclaim_setup(history[-1], score=2.1)
        self.assertFalse(ok)
        self.assertEqual(reject, "neutral_score")

        strategy.daily_pnl.trade_count = 1
        ok, setup_name, _ = strategy._passes_neutral_pullback_reclaim_setup(history[-1], score=2.1)
        self.assertTrue(ok)
        self.assertEqual(setup_name, "neutral_pullback_reclaim")

    def test_bull_a_grade_uses_partial_take_profit_once(self):
        cfg = MomentumScalpConfig(
            enable_regime_adaptive=False,
            take_profit_pct=1.0,
            bull_partial_exit_ratio=0.5,
            enable_cost_aware_profit_exit=False,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy.positions["005930"] = PositionState(
            symbol="005930",
            buy_price=10_000,
            quantity=4,
            invested_amount=40_000,
            entry_strategy_name="bull_breakout_strategy",
            entry_setup_name="bull_breakout",
            entry_grade="A",
        )

        quote = Quote(
            symbol="005930",
            name="삼성전자",
            current_price=10_120,
            change=120,
            change_rate=1.2,
            open_price=10_000,
            high_price=10_120,
            low_price=9_980,
            volume=200_000,
            trade_amount=2_024_000_000,
        )

        order = strategy._default_long_exit(quote)
        self.assertIsNotNone(order)
        self.assertEqual(order.quantity, 2)

        strategy.on_order_filled(
            OrderResult(
                success=True,
                symbol="005930",
                side=OrderSide.SELL,
                quantity=2,
                price=10_120,
            )
        )

        self.assertIn("005930", strategy.positions)
        self.assertEqual(strategy.positions["005930"].quantity, 2)
        self.assertTrue(strategy.positions["005930"].partial_exit_done)
        self.assertEqual(strategy.daily_pnl.trade_count, 1)
        self.assertIsNone(strategy._default_long_exit(quote))

    def test_neutral_long_is_blocked_after_one_neutral_loss(self):
        cfg = MomentumScalpConfig(
            enable_volume_spike_filter=False,
            enable_expected_net_filter=False,
            enable_pool_persistence_gate=False,
            neutral_max_losses_per_day=1,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy._bear_score = 1
        strategy.set_simulated_now(datetime(2026, 3, 16, 10, 0))
        strategy._neutral_loss_count_today = 1

        quote = Quote(
            "011700",
            "미원",
            10_220,
            220,
            2.2,
            10_000,
            10_220,
            9_950,
            40_000,
            400_000,
        )

        can_open = strategy._can_open_new_long(quote)
        self.assertFalse(can_open)

    def test_neutral_long_blocks_during_post_loss_cooldown(self):
        cfg = MomentumScalpConfig(
            enable_volume_spike_filter=False,
            enable_expected_net_filter=False,
            enable_pool_persistence_gate=False,
            neutral_max_losses_per_day=1,
            neutral_post_loss_cooldown_minutes=30,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy._bear_score = 1
        strategy._neutral_loss_count_today = 1
        strategy._neutral_last_loss_at = datetime(2026, 3, 16, 9, 40)
        strategy.set_simulated_now(datetime(2026, 3, 16, 10, 0))

        quote = Quote(
            "011700",
            "미원",
            10_220,
            220,
            2.2,
            10_000,
            10_220,
            9_950,
            40_000,
            400_000,
        )

        with patch("src.strategies.momentum_scalp.logger.info") as mock_info:
            can_open = strategy._can_open_new_long(quote, score=2.6)

        self.assertFalse(can_open)
        log_args = " ".join(str(call.args) for call in mock_info.call_args_list)
        self.assertIn("neutral_loss_cooldown", log_args)

    def test_loss_stage1_allows_only_high_quality_neutral_entries(self):
        cfg = MomentumScalpConfig(
            enable_volume_spike_filter=False,
            enable_expected_net_filter=False,
            enable_pool_persistence_gate=False,
            stage1_loss_threshold=-2_000,
            stage1_neutral_score_bonus=0.5,
            stage1_neutral_change_rate_bonus=0.2,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy._bear_score = 1
        strategy.set_simulated_now(datetime(2026, 3, 16, 10, 0))
        strategy.daily_pnl.realized_net_pnl = -2_500
        strategy.daily_pnl.trade_count = 1

        weak_quote = Quote(
            "011700",
            "미원",
            10_220,
            220,
            2.2,
            10_000,
            10_220,
            9_950,
            40_000,
            400_000,
        )
        strong_quote = Quote(
            "011700",
            "미원",
            10_220,
            220,
            2.6,
            10_000,
            10_220,
            9_950,
            40_000,
            400_000,
        )

        self.assertFalse(strategy._can_open_new_long(weak_quote, score=2.5))
        self.assertTrue(strategy._can_open_new_long(strong_quote, score=3.2))

    def test_neutral_post_loss_retry_requires_stronger_setup(self):
        cfg = MomentumScalpConfig(
            enable_volume_spike_filter=False,
            enable_expected_net_filter=False,
            enable_pool_persistence_gate=False,
            neutral_max_losses_per_day=1,
            neutral_post_loss_cooldown_minutes=20,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy._bear_score = 1
        strategy._neutral_loss_count_today = 1
        strategy._neutral_last_loss_at = datetime(2026, 3, 16, 9, 30)
        strategy.set_simulated_now(datetime(2026, 3, 16, 10, 0))

        borderline_history = [
            Quote("011700", "미원", 10_000, 0, 0.0, 10_000, 10_000, 9_950, 10_000, 100_000),
            Quote("011700", "미원", 10_200, 200, 2.0, 10_000, 10_200, 9_950, 20_000, 200_000),
            Quote("011700", "미원", 10_120, 120, 1.2, 10_000, 10_220, 9_950, 30_000, 300_000),
            Quote("011700", "미원", 10_150, 150, 1.5, 10_000, 10_220, 9_950, 35_000, 350_000),
            Quote("011700", "미원", 10_220, 220, 2.2, 10_000, 10_220, 9_950, 40_000, 400_000),
        ]
        for quote in borderline_history:
            strategy._record_recent_quote(quote)

        ok, _, reject = strategy._passes_neutral_pullback_reclaim_setup(borderline_history[-1], score=2.1)
        self.assertFalse(ok)
        self.assertEqual(reject, "neutral_chase_block")

        strategy._recent_quotes = {}
        strong_history = [
            Quote("011700", "미원", 10_000, 0, 0.0, 10_000, 10_000, 9_950, 10_000, 100_000),
            Quote("011700", "미원", 10_260, 260, 2.6, 10_000, 10_260, 9_950, 20_000, 200_000),
            Quote("011700", "미원", 10_150, 150, 1.5, 10_000, 10_260, 9_950, 30_000, 300_000),
            Quote("011700", "미원", 10_165, 165, 1.65, 10_000, 10_260, 9_950, 35_000, 350_000),
            Quote("011700", "미원", 10_170, 170, 1.7, 10_000, 10_260, 9_950, 38_000, 380_000),
            Quote("011700", "미원", 10_280, 280, 2.8, 10_000, 10_280, 9_950, 42_000, 420_000),
        ]
        for quote in strong_history:
            strategy._record_recent_quote(quote)

        can_open = strategy._can_open_new_long(strong_history[-1], score=2.6)
        ok, setup_name, _ = strategy._passes_neutral_pullback_reclaim_setup(strong_history[-1], score=2.6)
        self.assertTrue(can_open)
        self.assertTrue(ok)
        self.assertEqual(setup_name, "neutral_pullback_reclaim")

    def test_neutral_post_loss_retry_is_limited_to_one_fill(self):
        cfg = MomentumScalpConfig(
            enable_volume_spike_filter=False,
            enable_expected_net_filter=False,
            enable_pool_persistence_gate=False,
            neutral_max_losses_per_day=1,
            neutral_post_loss_cooldown_minutes=20,
            neutral_post_loss_reentry_limit=1,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy._bear_score = 1
        strategy._neutral_loss_count_today = 1
        strategy._neutral_last_loss_at = datetime(2026, 3, 16, 9, 30)
        strategy.set_simulated_now(datetime(2026, 3, 16, 10, 5))

        strategy._pending_entry_meta["011700"] = {"neutral_post_loss_retry": True}
        strategy.on_order_filled(
            OrderResult(
                success=True,
                symbol="011700",
                side=OrderSide.BUY,
                quantity=10,
                price=10_280,
            )
        )

        self.assertEqual(strategy._neutral_post_loss_reentries_today, 1)

        quote = Quote(
            "011700",
            "미원",
            10_300,
            300,
            3.0,
            10_000,
            10_300,
            9_950,
            50_000,
            500_000,
        )
        can_open = strategy._can_open_new_long(quote, score=2.8)
        self.assertFalse(can_open)

    def test_shadow_tracking_records_blocked_neutral_candidate_outcome(self):
        cfg = MomentumScalpConfig(
            enable_volume_spike_filter=False,
            enable_expected_net_filter=False,
            enable_pool_persistence_gate=False,
            neutral_max_losses_per_day=1,
            shadow_blocked_candidate_window_minutes=1,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy._bear_score = 1
        strategy._neutral_loss_count_today = 1
        strategy.set_simulated_now(datetime(2026, 3, 16, 13, 2, 0))

        blocked_quote = Quote(
            "011700",
            "미원",
            10_220,
            220,
            2.2,
            10_000,
            10_220,
            9_950,
            40_000,
            400_000,
        )

        with patch("src.strategies.momentum_scalp.logger.info") as mock_info:
            can_open = strategy._can_open_new_long(blocked_quote)
            self.assertFalse(can_open)
            self.assertIn("011700", strategy._shadow_blocked_candidates)

            candidate = strategy._shadow_blocked_candidates["011700"]
            self.assertEqual(candidate.reject_reason, "neutral_loss_limit_block")
            self.assertEqual(candidate.entry_price, 10_220)

            target_price = int(candidate.entry_price * (1 + (candidate.target_pct / 100) + 0.01))
            strategy.set_simulated_now(datetime(2026, 3, 16, 13, 3, 1))
            strategy._update_shadow_blocked_candidates(
                [
                    Quote(
                        "011700",
                        "미원",
                        target_price,
                        target_price - 10_000,
                        3.4,
                        10_000,
                        target_price,
                        9_950,
                        50_000,
                        500_000,
                    )
                ]
            )

        self.assertEqual(candidate.first_hit_outcome, "take_profit_first")
        self.assertNotIn("011700", strategy._shadow_blocked_candidates)
        log_args = " ".join(str(call.args) for call in mock_info.call_args_list)
        self.assertIn("그림자 후보 추적 시작", log_args)
        self.assertIn("그림자 후보 종료", log_args)
        self.assertIn("take_profit_first", log_args)

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

    def test_neutral_entry_requires_multiple_confirmation_ticks(self):
        cfg = MomentumScalpConfig(
            enable_regime_adaptive=True,
            entry_confirmation_ticks=2,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy._bear_score = 1

        now = datetime(2026, 3, 16, 10, 0)
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

        first_entry = strategy._can_confirm_entry(
            quote=quote,
            score=strategy._regime_min_momentum_score() + 0.4,
            is_scale_in=False,
            now=now,
        )
        second_entry = strategy._can_confirm_entry(
            quote=quote,
            score=strategy._regime_min_momentum_score() + 0.4,
            is_scale_in=False,
            now=now + timedelta(seconds=10),
        )
        third_entry = strategy._can_confirm_entry(
            quote=quote,
            score=strategy._regime_min_momentum_score() + 0.4,
            is_scale_in=False,
            now=now + timedelta(seconds=20),
        )

        self.assertFalse(first_entry)
        self.assertFalse(second_entry)
        self.assertTrue(third_entry)

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

    def test_inverse_buy_blocked_by_strategy_cooldown(self):
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
        strategy._bear_score = 2
        strategy._strategy_cooldown_until["soft_bear_inverse_strategy"] = datetime.now() + timedelta(minutes=5)

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

    def test_symbol_cooldown_blocks_only_same_symbol(self):
        cfg = MomentumScalpConfig(
            enable_volume_spike_filter=False,
            enable_expected_net_filter=False,
            enable_pool_persistence_gate=False,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy._bear_score = 1
        now = datetime(2026, 3, 16, 10, 0)
        strategy.set_simulated_now(now)
        strategy._symbol_cooldown_until["AAA"] = now + timedelta(minutes=5)

        aaa_quote = Quote("AAA", "A", 10_200, 200, 2.0, 10_000, 10_220, 9_950, 30_000, 300_000_000)
        bbb_quote = Quote("BBB", "B", 10_200, 200, 2.0, 10_000, 10_220, 9_950, 30_000, 300_000_000)

        self.assertFalse(strategy._can_open_new_long(aaa_quote, score=3.0))
        self.assertTrue(strategy._can_open_new_long(bbb_quote, score=3.0))

    def test_neutral_strategy_cooldown_does_not_block_soft_bear_inverse(self):
        cfg = MomentumScalpConfig(
            inverse_enabled=True,
            inverse_etfs=["252670"],
            enable_volume_spike_filter=False,
            enable_expected_net_filter=False,
            enable_pool_persistence_gate=False,
            soft_bear_inverse_min_change_rate=0.0,
            soft_bear_inverse_min_momentum=0.0,
            enable_math_shadow_layer=False,
            enable_math_live_layer=False,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        now = datetime(2026, 3, 16, 10, 0)
        strategy.set_simulated_now(now)
        strategy._strategy_cooldown_until["neutral_pullback_strategy"] = now + timedelta(minutes=10)
        strategy._bear_score = 2

        history = [
            Quote("252670", "KODEX 인버스", 2_000, 0, 0.0, 2_000, 2_000, 1_995, 10_000, 20_000_000),
            Quote("252670", "KODEX 인버스", 2_030, 30, 1.5, 2_000, 2_030, 1_995, 20_000, 40_000_000),
            Quote("252670", "KODEX 인버스", 2_020, 20, 1.0, 2_000, 2_030, 1_995, 30_000, 60_000_000),
            Quote("252670", "KODEX 인버스", 2_032, 32, 1.6, 2_000, 2_032, 1_995, 40_000, 80_000_000),
        ]
        for quote in history:
            strategy._record_recent_quote(quote)

        order = strategy._evaluate_inverse_buy(history[-1])
        self.assertIsNotNone(order)

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
            enable_math_shadow_layer=False,
            enable_math_live_layer=False,
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
