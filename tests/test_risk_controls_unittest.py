import unittest
from collections import deque
from datetime import datetime, timedelta
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.analytics.price_prediction import ShortHorizonPrediction
from src.analytics.math_signals import (
    ExpectedValueEstimate,
    LeaderSignal,
)
from src.models import Order, OrderResult, OrderSide, Quote
from src.strategies.momentum_scalp import (
    ExpectedValueCandidate,
    ExpectedValueTradePlan,
    INTRADAY_STRATEGY,
    OPENING_STRATEGY,
    MomentumScalpConfig,
    MomentumScalpStrategy,
)
from src.strategies.momentum_scalp_micro import symbol_micro_edge_metrics
from src.strategies.momentum_scalp_pnl import calculate_trade_pnl_from_prices
from src.strategies.momentum_scalp_types import PositionState


class RiskControlTests(unittest.TestCase):
    def setUp(self):
        self._state_tmpdir = tempfile.TemporaryDirectory()
        self._config_init_defaults = MomentumScalpConfig.__init__.__defaults__
        isolated_defaults = list(self._config_init_defaults)
        isolated_defaults[-2] = str(Path(self._state_tmpdir.name) / "forecast-outcomes")
        isolated_defaults[-1] = str(Path(self._state_tmpdir.name) / "momentum-state.json")
        MomentumScalpConfig.__init__.__defaults__ = tuple(isolated_defaults)

    def tearDown(self):
        MomentumScalpConfig.__init__.__defaults__ = self._config_init_defaults
        self._state_tmpdir.cleanup()

    def _install_strong_ev_prediction(
        self,
        strategy,
        *,
        predicted_return_pct: float = 1.60,
        lower_bound_return_pct: float = 0.20,
        upper_bound_return_pct: float = 2.10,
        confidence: float = 0.78,
        direction_score: float = 0.74,
    ):
        strategy._price_prediction_for_entry = lambda *_args, **_kwargs: ShortHorizonPrediction(
            ready=True,
            reason="unit_test_ev_ready",
            horizon_seconds=180,
            sample_count=8,
            predicted_return_pct=predicted_return_pct,
            lower_bound_return_pct=lower_bound_return_pct,
            upper_bound_return_pct=upper_bound_return_pct,
            confidence=confidence,
            direction_score=direction_score,
            volatility_pct=0.24,
            features={
                "continuation_quality": 0.72,
                "follow_through_score": 0.70,
                "rejection_risk_score": 0.08,
                "chase_risk_score": 0.10,
                "trap_risk_score": 0.07,
            },
        )

    def test_expected_value_plan_allows_small_positive_net_below_daily_target(self):
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(
                seed_money=300_000,
                capital_utilization_pct=1.0,
                daily_profit_target=10_000,
                daily_loss_limit=-5_000,
                daily_total_loss_limit=-5_000,
                long_stop_loss_cap_amount=2_500,
                daily_state_path="/tmp/nonexistent-momentum-state.json",
            ),
        )
        now = datetime(2026, 5, 27, 10, 30, 0)
        strategy.set_simulated_now(now)
        quote = Quote("EV001", "EV001", 10_000, 200, 2.0, 9_800, 10_050, 9_700, 200_000, 2_000_000_000, now)
        self._install_strong_ev_prediction(strategy, predicted_return_pct=1.35, lower_bound_return_pct=-0.25)
        meta = {"strategy_name": INTRADAY_STRATEGY, "live_route": INTRADAY_STRATEGY}

        plan = strategy._build_expected_value_trade_plan(quote, meta, pending_orders=[])

        self.assertTrue(plan.allowed, plan.reject_reason)
        self.assertGreater(plan.expected_net, 0.0)
        self.assertGreater(plan.predicted_net, 0)
        self.assertLess(plan.planned_target_net, 10_000)


    def test_expected_value_plan_rejects_one_share_unrounded_net_loss(self):
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(
                seed_money=1_000_000,
                capital_utilization_pct=1.0,
                daily_profit_target=10_000,
                daily_loss_limit=-5_000,
                daily_total_loss_limit=-5_000,
                long_stop_loss_cap_amount=3_500,
                daily_state_path="/tmp/nonexistent-momentum-state.json",
            ),
        )
        now = datetime(2026, 7, 14, 15, 7, 40)
        strategy.set_simulated_now(now)
        quote = Quote(
            "025870",
            "신라에스지",
            2_235,
            0,
            0.0,
            1_695,
            2_235,
            1_695,
            1_000_000,
            2_235_000_000,
            now,
        )
        strategy._price_prediction_for_entry = lambda *_args, **_kwargs: ShortHorizonPrediction(
            ready=True,
            reason="unit_test_rounding_artifact",
            horizon_seconds=180,
            sample_count=15,
            predicted_return_pct=0.40,
            lower_bound_return_pct=-0.798,
            upper_bound_return_pct=1.50,
            confidence=0.946,
            direction_score=0.662,
            volatility_pct=0.30,
            features={},
        )

        plan = strategy._build_expected_value_trade_plan(
            quote,
            {"strategy_name": INTRADAY_STRATEGY, "live_route": INTRADAY_STRATEGY},
            pending_orders=[],
        )

        self.assertFalse(plan.allowed)
        self.assertEqual(plan.reject_reason, "ev_no_positive_quantity")
        self.assertEqual(plan.quantity, 1)
        self.assertEqual(plan.reject_detail, "predicted_net_non_positive")





    def test_expected_value_plan_keeps_robust_breakout_despite_prior_soft_reject(self):
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(
                seed_money=1_000_000,
                capital_utilization_pct=1.0,
                daily_profit_target=10_000,
                daily_loss_limit=-5_000,
                daily_total_loss_limit=-5_000,
                daily_state_path="/tmp/nonexistent-momentum-state.json",
            ),
        )
        first_seen = datetime(2026, 6, 19, 11, 10, 50)
        strategy.set_simulated_now(first_seen)
        strategy._session_start_at = first_seen.replace(hour=9, minute=0, second=0, microsecond=0)
        quote = Quote("085620", "085620", 35_200, 1_200, 3.53, 34_000, 35_250, 33_900, 750_000, 26_400_000_000, first_seen)
        meta = {
            "strategy_name": INTRADAY_STRATEGY,
            "live_route": INTRADAY_STRATEGY,
            "leader_percentile": 0.95,
            "effective_leader_score": 1.053,
        }
        strategy._price_prediction_for_entry = lambda *_args, **_kwargs: ShortHorizonPrediction(
            ready=True,
            reason="unit_test_prior_soft_lower",
            horizon_seconds=180,
            sample_count=8,
            predicted_return_pct=0.637,
            lower_bound_return_pct=-0.228,
            upper_bound_return_pct=1.10,
            confidence=0.653,
            direction_score=0.62,
            volatility_pct=0.30,
            features={
                "continuation_quality": 0.62,
                "follow_through_score": 0.61,
                "rejection_risk_score": 0.06,
                "chase_risk_score": 0.05,
                "trap_risk_score": 0.05,
            },
        )
        strategy._build_expected_value_trade_plan(quote, dict(meta), pending_orders=[])

        entry_time = first_seen + timedelta(seconds=42)
        strategy.set_simulated_now(entry_time)
        quote = Quote("085620", "085620", 35_700, 1_700, 5.00, 34_000, 35_750, 33_900, 900_000, 32_130_000_000, entry_time)
        strategy._price_prediction_for_entry = lambda *_args, **_kwargs: ShortHorizonPrediction(
            ready=True,
            reason="unit_test_robust_breakout",
            horizon_seconds=180,
            sample_count=8,
            predicted_return_pct=2.243,
            lower_bound_return_pct=1.604,
            upper_bound_return_pct=2.80,
            confidence=0.684,
            direction_score=0.740,
            volatility_pct=0.32,
            features={
                "continuation_quality": 0.76,
                "follow_through_score": 0.74,
                "rejection_risk_score": 0.05,
                "chase_risk_score": 0.05,
                "trap_risk_score": 0.04,
            },
        )
        live_meta = dict(meta)
        plan = strategy._build_expected_value_trade_plan(quote, live_meta, pending_orders=[])

        self.assertTrue(plan.allowed, plan.reject_reason)
        self.assertGreater(plan.expected_net, 0.0)
        self.assertLess(live_meta.get("ev_prediction_stability_penalty", 0.0), 0.35)



    def test_expected_value_plan_allows_rebound_when_ev_is_positive(self):
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(
                seed_money=1_000_000,
                capital_utilization_pct=1.0,
                daily_profit_target=10_000,
                daily_loss_limit=-5_000,
                daily_total_loss_limit=-5_000,
                long_stop_loss_cap_amount=5_000,
                daily_state_path="/tmp/nonexistent-momentum-state.json",
            ),
        )
        entry_time = datetime(2026, 6, 25, 13, 37, 9)
        strategy.set_simulated_now(entry_time)
        strategy._session_start_at = datetime(2026, 6, 25, 9, 0, 0)
        quote = Quote("002990", "002990", 5_440, 380, 7.60, 5_056, 9_800, 5_000, 2_000_000, 10_880_000_000, entry_time)
        strategy._price_prediction_for_entry = lambda *_args, **_kwargs: ShortHorizonPrediction(
            ready=True,
            reason="unit_test_today_002990_rebound_after_noise_floor",
            horizon_seconds=180,
            sample_count=8,
            predicted_return_pct=1.966,
            lower_bound_return_pct=0.325,
            upper_bound_return_pct=2.350,
            confidence=0.603,
            direction_score=0.705,
            volatility_pct=0.35,
            features={
                "continuation_quality": 0.64,
                "follow_through_score": 0.62,
                "rejection_risk_score": 0.10,
                "chase_risk_score": 0.12,
                "trap_risk_score": 0.10,
            },
        )
        meta = {
            "strategy_name": INTRADAY_STRATEGY,
            "live_route": INTRADAY_STRATEGY,
            "queue_source": "math_queue",
            "conviction_rank": 1,
            "conviction_score": 1.3108,
            "leader_percentile": 0.96,
            "effective_leader_score": 0.8413,
            "vs_open_pct": 7.60,
            "high_proximity": 0.555,
            "entry_grade": "A",
        }

        plan = strategy._build_expected_value_trade_plan(quote, meta, pending_orders=[])

        self.assertTrue(plan.allowed, plan.reject_detail)
        self.assertGreater(plan.expected_net, 0)
        self.assertGreater(plan.predicted_net, 0)
        self.assertLessEqual(plan.planned_risk_net_loss_abs, 5_000)


    def test_expected_value_plan_metadata_uses_live_plan_ev_for_position(self):
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(daily_state_path="/tmp/nonexistent-momentum-state.json"),
        )
        meta = {
            "entry_ev": -2_436.89,
            "entry_ev_confidence": "medium",
            "entry_ev_closed_trades": 6,
        }
        plan = ExpectedValueTradePlan(
            allowed=True,
            quantity=5,
            budget=591_500,
            expected_net=637.91,
            predicted_net=3_883,
            lower_net=-1_024,
            upper_net=5_200,
            win_probability=0.605,
            break_even_probability=0.527,
            planned_target_net=3_883,
            planned_stop_net_loss_abs=3_500,
            planned_risk_net_loss_abs=4_323,
        )

        strategy._apply_expected_value_trade_plan_metadata(meta, plan)

        self.assertEqual(meta["historical_entry_ev"], -2_436.89)
        self.assertEqual(meta["historical_entry_ev_confidence"], "medium")
        self.assertEqual(meta["historical_entry_ev_closed_trades"], 6)
        self.assertEqual(meta["entry_ev"], 637.91)
        self.assertEqual(meta["entry_ev_confidence"], "live_plan")
        self.assertEqual(meta["entry_expected_net_pnl"], 637.91)

    def test_expected_value_plan_rejects_weak_context_prediction_when_tick_history_is_not_ready(self):
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(
                seed_money=1_000_000,
                capital_utilization_pct=1.0,
                daily_profit_target=10_000,
                daily_loss_limit=-5_000,
                daily_total_loss_limit=-5_000,
                daily_state_path="/tmp/nonexistent-momentum-state.json",
            ),
        )
        now = datetime(2026, 5, 27, 12, 58, 0)
        strategy.set_simulated_now(now)
        quote = Quote("EV005", "EV005", 10_000, 260, 2.67, 9_740, 10_050, 9_650, 300_000, 3_000_000_000, now)
        strategy._price_prediction_for_entry = lambda *_args, **_kwargs: ShortHorizonPrediction(
            ready=False,
            reason="insufficient_samples",
            horizon_seconds=180,
            sample_count=1,
            predicted_return_pct=0.0,
            lower_bound_return_pct=0.0,
            upper_bound_return_pct=0.0,
            confidence=0.0,
            direction_score=0.0,
            volatility_pct=0.0,
        )
        meta = {
            "strategy_name": INTRADAY_STRATEGY,
            "live_route": INTRADAY_STRATEGY,
            "leader_percentile": 1.0,
            "effective_leader_score": 1.25,
            "recent_accel": 0.28,
            "volume_vs_avg": 1.20,
            "vs_open_pct": 2.67,
            "high_proximity": 0.995,
        }

        plan = strategy._build_expected_value_trade_plan(quote, meta, pending_orders=[])

        self.assertFalse(plan.allowed)
        self.assertEqual(plan.reject_reason, "ev_prediction_not_ready")
        self.assertIsNotNone(plan.prediction)
        self.assertEqual(plan.prediction.reason, "insufficient_samples")

    def test_ev_reject_log_is_compact_without_legacy_gate_fields(self):
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(daily_state_path="/tmp/nonexistent-momentum-state.json"),
        )
        now = datetime(2026, 5, 27, 12, 59, 51)
        quote = Quote("308080", "테스트", 10_000, 1_000, 11.27, 8_987, 10_560, 8_860, 300_000, 3_000_000_000, now)
        prediction = ShortHorizonPrediction(
            ready=True,
            reason="context_quote_projection",
            horizon_seconds=180,
            sample_count=1,
            predicted_return_pct=-0.12,
            lower_bound_return_pct=-0.74,
            upper_bound_return_pct=0.31,
            confidence=0.42,
            direction_score=0.37,
            volatility_pct=0.30,
        )
        plan = ExpectedValueTradePlan(
            allowed=False,
            reject_reason="ev_prediction_non_positive",
            prediction=prediction,
        )
        meta = {
            "queue_source": "math_queue",
            "conviction_rank": 6,
            "conviction_score": 0.133,
            "vs_open_pct": 11.27,
            "high_proximity": 0.560,
        }

        with self.assertLogs("kis_trader.strategy.momentum_scalp", level="INFO") as captured:
            strategy._log_ev_reject(quote, INTRADAY_STRATEGY, plan.reject_reason, meta, plan)

        line = captured.output[0]
        self.assertIn("EV 진입 거부[ev_prediction_non_positive]", line)
        self.assertIn("pred=-0.120", line)
        self.assertLess(len(line), 420)
        self.assertNotIn("target_edge", line)
        self.assertNotIn("scout_neg_ev", line)
        self.assertNotIn("risk_sizing", line)

    def test_ev_reject_log_includes_best_rejected_quantity_detail(self):
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(daily_state_path="/tmp/nonexistent-momentum-state.json"),
        )
        now = datetime(2026, 6, 22, 9, 15, 39)
        quote = Quote("080220", "080220", 123_000, 11_000, 9.82, 112_000, 123_500, 112_000, 500_000, 61_500_000_000, now)
        prediction = ShortHorizonPrediction(
            ready=True,
            reason="unit_test_reject_detail",
            horizon_seconds=180,
            sample_count=8,
            predicted_return_pct=1.332,
            lower_bound_return_pct=0.580,
            upper_bound_return_pct=1.75,
            confidence=0.716,
            direction_score=0.70,
            volatility_pct=0.34,
        )
        plan = ExpectedValueTradePlan(
            allowed=False,
            reject_reason="ev_no_positive_quantity",
            quantity=4,
            budget=492_000,
            expected_net=312.4,
            predicted_net=5_210,
            lower_net=2_080,
            win_probability=0.442,
            break_even_probability=0.487,
            planned_target_net=5_210,
            planned_stop_net_loss_abs=2_200,
            planned_risk_net_loss_abs=2_650,
            prediction=prediction,
            reject_detail="risk_room_exceeded:need=5800 room=5000",
        )
        meta = {
            "queue_source": "math_queue",
            "conviction_rank": 3,
            "conviction_score": 0.83,
            "vs_open_pct": 9.82,
            "high_proximity": 0.996,
        }

        with self.assertLogs("kis_trader.strategy.momentum_scalp", level="INFO") as captured:
            strategy._log_ev_reject(quote, INTRADAY_STRATEGY, plan.reject_reason, meta, plan)

        line = captured.output[0]
        self.assertIn("EV 진입 거부[ev_no_positive_quantity]", line)
        self.assertIn("qty=4", line)
        self.assertIn("pnet=5210", line)
        self.assertIn("detail=risk_room_exceeded:need=5800 room=5000", line)

    def test_expected_value_plan_allows_small_recovery_trade_when_loss_room_supports_risk(self):
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(
                seed_money=1_000_000,
                capital_utilization_pct=1.0,
                daily_profit_target=10_000,
                daily_loss_limit=-5_000,
                daily_total_loss_limit=-5_000,
                long_stop_loss_cap_amount=2_500,
                daily_state_path="/tmp/nonexistent-momentum-state.json",
            ),
        )
        strategy.daily_pnl.realized_net_pnl = -4_519
        now = datetime(2026, 6, 1, 13, 32, 57)
        strategy.set_simulated_now(now)
        quote = Quote("006660", "006660", 17_640, 1_250, 7.63, 16_390, 17_700, 16_390, 500_000, 8_820_000_000, now)
        strategy._price_prediction_for_entry = lambda *_args, **_kwargs: ShortHorizonPrediction(
            ready=True,
            reason="unit_test_recovery_continuation",
            horizon_seconds=180,
            sample_count=8,
            predicted_return_pct=1.50,
            lower_bound_return_pct=0.20,
            upper_bound_return_pct=2.20,
            confidence=0.860,
            direction_score=0.860,
            volatility_pct=0.30,
            features={
                "continuation_quality": 0.88,
                "follow_through_score": 0.88,
                "rejection_risk_score": 0.02,
                "chase_risk_score": 0.03,
                "trap_risk_score": 0.02,
            },
        )
        meta = {"strategy_name": INTRADAY_STRATEGY, "live_route": INTRADAY_STRATEGY}

        plan = strategy._build_expected_value_trade_plan(quote, meta, pending_orders=[])

        self.assertTrue(plan.allowed, plan.reject_reason)
        self.assertGreater(plan.expected_net, 0.0)
        self.assertLessEqual(plan.planned_risk_net_loss_abs, 481)
        self.assertLess(plan.planned_target_net, 10_000)




    def test_expected_value_plan_allows_late_session_high_extension_continuation(self):
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(
                seed_money=1_000_000,
                capital_utilization_pct=1.0,
                daily_profit_target=10_000,
                daily_loss_limit=-5_000,
                daily_total_loss_limit=-5_000,
                long_stop_loss_cap_amount=2_500,
                daily_state_path="/tmp/nonexistent-momentum-state.json",
            ),
        )
        now = datetime(2026, 6, 11, 12, 6, 40)
        strategy.set_simulated_now(now)
        strategy._session_start_at = datetime(2026, 6, 11, 9, 0, 0)
        quote = Quote("001820", "001820", 121_900, 15_620, 14.69, 106_280, 121_900, 106_000, 1_000_000, 121_900_000_000, now)
        strategy._price_prediction_for_entry = lambda *_args, **_kwargs: ShortHorizonPrediction(
            ready=True,
            reason="unit_test_late_high_extension_continuation",
            horizon_seconds=180,
            sample_count=8,
            predicted_return_pct=1.155,
            lower_bound_return_pct=0.175,
            upper_bound_return_pct=1.700,
            confidence=0.776,
            direction_score=0.776,
            volatility_pct=0.30,
            features={
                "continuation_quality": 0.72,
                "follow_through_score": 0.70,
                "rejection_risk_score": 0.05,
                "chase_risk_score": 0.08,
                "trap_risk_score": 0.06,
            },
        )
        meta = {
            "strategy_name": INTRADAY_STRATEGY,
            "live_route": INTRADAY_STRATEGY,
            "queue_source": "math_queue",
            "conviction_rank": 1,
            "conviction_score": 0.6423,
            "vs_open_pct": 14.69,
            "high_proximity": 1.0,
        }

        plan = strategy._build_expected_value_trade_plan(quote, meta, pending_orders=[])

        self.assertTrue(plan.allowed, plan.reject_reason)
        self.assertGreater(plan.expected_net, 0.0)
        self.assertLessEqual(plan.planned_risk_net_loss_abs, 4_750)



    def test_expected_value_plan_sizes_down_opening_budget_consuming_b_trade(self):
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(
                seed_money=1_000_000,
                capital_utilization_pct=1.0,
                daily_profit_target=10_000,
                daily_loss_limit=-5_000,
                daily_total_loss_limit=-5_000,
                long_stop_loss_cap_amount=2_500,
                daily_state_path="/tmp/nonexistent-momentum-state.json",
            ),
        )
        now = datetime(2026, 6, 12, 9, 4, 20)
        strategy.set_simulated_now(now)
        strategy._session_start_at = datetime(2026, 6, 12, 9, 0, 0)
        quote = Quote("089030", "089030", 66_000, 1_900, 2.96, 64_100, 66_100, 63_900, 600_000, 39_600_000_000, now)
        strategy._price_prediction_for_entry = lambda *_args, **_kwargs: ShortHorizonPrediction(
            ready=True,
            reason="unit_test_opening_budget_consuming_b_trade",
            horizon_seconds=180,
            sample_count=8,
            predicted_return_pct=1.960,
            lower_bound_return_pct=0.203,
            upper_bound_return_pct=2.300,
            confidence=0.721,
            direction_score=0.721,
            volatility_pct=0.30,
            features={
                "continuation_quality": 0.68,
                "follow_through_score": 0.66,
                "rejection_risk_score": 0.05,
                "chase_risk_score": 0.08,
                "trap_risk_score": 0.06,
            },
        )
        meta = {
            "strategy_name": OPENING_STRATEGY,
            "live_route": OPENING_STRATEGY,
            "queue_source": "opening_hot_queue",
            "entry_grade": "B",
            "leader_percentile": 0.7895,
            "conviction_rank": 1,
            "conviction_score": 0.4211,
            "vs_open_pct": 2.96,
            "high_proximity": 0.998,
        }

        plan = strategy._build_expected_value_trade_plan(quote, meta, pending_orders=[])

        self.assertTrue(plan.allowed, plan.reject_reason)
        self.assertLess(plan.quantity, 15)
        self.assertLessEqual(plan.planned_risk_net_loss_abs, 5_000)

    def test_expected_value_plan_allows_opening_ev_within_loss_room_today_089030(self):
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(
                seed_money=1_000_000,
                capital_utilization_pct=1.0,
                daily_profit_target=10_000,
                daily_loss_limit=-5_000,
                daily_total_loss_limit=-5_000,
                long_stop_loss_cap_amount=5_000,
                daily_state_path="/tmp/nonexistent-momentum-state.json",
            ),
        )
        now = datetime(2026, 6, 25, 9, 2, 50)
        strategy.set_simulated_now(now)
        strategy._session_start_at = datetime(2026, 6, 25, 9, 0, 0)
        quote = Quote("089030", "089030", 55_400, 2_900, 5.52, 52_500, 60_950, 52_400, 900_000, 49_860_000_000, now)
        strategy._price_prediction_for_entry = lambda *_args, **_kwargs: ShortHorizonPrediction(
            ready=True,
            reason="unit_test_today_089030_opening_ev",
            horizon_seconds=180,
            sample_count=8,
            predicted_return_pct=2.021,
            lower_bound_return_pct=1.392,
            upper_bound_return_pct=2.450,
            confidence=0.568,
            direction_score=0.836,
            volatility_pct=0.30,
            features={
                "continuation_quality": 0.72,
                "follow_through_score": 0.70,
                "rejection_risk_score": 0.05,
                "chase_risk_score": 0.08,
                "trap_risk_score": 0.06,
            },
        )
        meta = {
            "strategy_name": OPENING_STRATEGY,
            "live_route": OPENING_STRATEGY,
            "queue_source": "opening_fast_queue",
            "entry_grade": "A",
            "leader_percentile": 0.90,
            "conviction_rank": 1,
            "conviction_score": 0.7728,
            "effective_leader_score": 0.95,
            "vs_open_pct": 5.52,
            "high_proximity": 0.909,
        }

        plan = strategy._build_expected_value_trade_plan(quote, meta, pending_orders=[])

        self.assertTrue(plan.allowed, plan.reject_detail)
        self.assertGreater(plan.expected_net, 0)
        self.assertGreater(plan.predicted_net, 0)
        self.assertLessEqual(plan.planned_risk_net_loss_abs, 5_000)
        self.assertNotIn("ev_opening_risk_share_ceiling", meta)

    def test_prediction_uncertainty_does_not_masquerade_as_execution_cost(self):
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(
                commission_rate=0.00015,
                tax_slippage_rate=0.002,
                entry_market_slippage_rate=0.001,
                exit_market_slippage_rate=0.001,
            ),
        )
        prediction = ShortHorizonPrediction(
            ready=True,
            reason="unit_test_high_uncertainty",
            horizon_seconds=180,
            sample_count=5,
            predicted_return_pct=1.0,
            lower_bound_return_pct=-2.0,
            upper_bound_return_pct=4.0,
            confidence=0.2,
            direction_score=0.6,
            volatility_pct=1.8,
            features={
                "opening_instability_risk": 1.0,
                "intraday_impulse_instability_risk": 1.0,
                "quote_gap_risk": 1.0,
                "trap_risk_score": 1.0,
            },
        )

        self.assertEqual(strategy._entry_execution_slippage_rate(prediction), 0.001)
        self.assertAlmostEqual(strategy._round_trip_execution_cost_pct(), 0.43)

    def test_long_shortlist_excludes_symbols_that_cannot_be_bought_even_one_share(self):
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(seed_money=1_000_000),
        )
        now = datetime(2026, 7, 23, 10, 30, 0)
        expensive = Quote("000660", "000660", 1_860_000, 0, 0.0, 1_860_000, 1_860_000, 1_860_000, 1, 1, now)
        affordable = Quote("005930", "005930", 70_000, 0, 0.0, 70_000, 70_000, 70_000, 1, 1, now)
        strategy._fresh_market_state_quotes = lambda _quotes: [expensive, affordable]
        strategy._long_ev_strategy_name_for_quote = lambda _quote: INTRADAY_STRATEGY
        strategy._remaining_long_exposure_budget = lambda _orders: 1_000_000
        strategy._remaining_long_seed_exposure_budget = lambda _orders: 1_000_000

        shortlist = strategy._long_entry_shortlist([expensive, affordable])

        self.assertEqual([quote.symbol for quote in shortlist], ["005930"])





    def test_expected_value_plan_sizes_126730_shape_inside_loss_room(self):
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(
                seed_money=1_000_000,
                capital_utilization_pct=1.0,
                daily_profit_target=10_000,
                daily_loss_limit=-5_000,
                daily_total_loss_limit=-5_000,
                long_stop_loss_cap_amount=5_000,
                daily_state_path="/tmp/nonexistent-momentum-state.json",
            ),
        )
        now = datetime(2026, 5, 27, 13, 46, 58)
        strategy.set_simulated_now(now)
        quote = Quote("126730", "126730", 26_400, 1_400, 5.60, 25_000, 26_500, 24_900, 400_000, 10_000_000_000, now)
        strategy._price_prediction_for_entry = lambda *_args, **_kwargs: ShortHorizonPrediction(
            ready=True,
            reason="unit_test_126730_shape",
            horizon_seconds=180,
            sample_count=8,
            predicted_return_pct=1.018,
            lower_bound_return_pct=-0.814,
            upper_bound_return_pct=1.55,
            confidence=0.534,
            direction_score=0.648,
            volatility_pct=0.35,
        )
        meta = {"strategy_name": INTRADAY_STRATEGY, "live_route": INTRADAY_STRATEGY}

        plan = strategy._build_expected_value_trade_plan(quote, meta, pending_orders=[])

        self.assertTrue(plan.allowed, plan.reject_detail)
        self.assertGreater(plan.expected_net, 0.0)
        self.assertLessEqual(plan.planned_risk_net_loss_abs, 5_000)


    def test_expected_value_plan_caps_quantity_by_lower_bound_loss_risk(self):
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(
                seed_money=1_000_000,
                capital_utilization_pct=1.0,
                daily_profit_target=10_000,
                daily_loss_limit=-5_000,
                daily_total_loss_limit=-5_000,
                long_stop_loss_cap_amount=5_000,
                daily_state_path="/tmp/nonexistent-momentum-state.json",
            ),
        )
        now = datetime(2026, 5, 27, 13, 50, 0)
        strategy.set_simulated_now(now)
        quote = Quote("EV006", "EV006", 26_400, 1_400, 5.60, 25_000, 26_500, 24_900, 400_000, 10_000_000_000, now)
        strategy._price_prediction_for_entry = lambda *_args, **_kwargs: ShortHorizonPrediction(
            ready=True,
            reason="unit_test_lower_risk_cap",
            horizon_seconds=180,
            sample_count=8,
            predicted_return_pct=3.20,
            lower_bound_return_pct=-0.814,
            upper_bound_return_pct=3.80,
            confidence=0.82,
            direction_score=0.82,
            volatility_pct=0.35,
        )
        meta = {"strategy_name": INTRADAY_STRATEGY, "live_route": INTRADAY_STRATEGY}

        plan = strategy._build_expected_value_trade_plan(quote, meta, pending_orders=[])

        self.assertTrue(plan.allowed, plan.reject_reason)
        self.assertLess(plan.quantity, 37)
        self.assertLessEqual(plan.planned_risk_net_loss_abs, 5_000)
        self.assertLessEqual(abs(min(0, int(plan.lower_net))), 5_000)
        self.assertLessEqual(plan.planned_stop_net_loss_abs, plan.planned_risk_net_loss_abs)

    def test_expected_value_plan_reserves_execution_buffer_inside_daily_loss_room(self):
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(
                seed_money=1_000_000,
                capital_utilization_pct=1.0,
                daily_profit_target=10_000,
                daily_loss_limit=-5_000,
                daily_total_loss_limit=-5_000,
                long_stop_loss_cap_amount=5_000,
                daily_state_path="/tmp/nonexistent-momentum-state.json",
            ),
        )
        now = datetime(2026, 6, 5, 9, 2, 53)
        strategy.set_simulated_now(now)
        quote = Quote("403870", "403870", 49_800, 2_500, 5.29, 47_300, 49_900, 47_300, 500_000, 24_900_000_000, now)
        strategy._price_prediction_for_entry = lambda *_args, **_kwargs: ShortHorizonPrediction(
            ready=True,
            reason="unit_test_403870_loss_room_buffer",
            horizon_seconds=180,
            sample_count=8,
            predicted_return_pct=1.869,
            lower_bound_return_pct=-0.100,
            upper_bound_return_pct=2.20,
            confidence=0.597,
            direction_score=0.603,
            volatility_pct=0.35,
            features={
                "continuation_quality": 0.70,
                "follow_through_score": 0.68,
                "rejection_risk_score": 0.05,
                "chase_risk_score": 0.08,
                "trap_risk_score": 0.06,
            },
        )
        meta = {"strategy_name": OPENING_STRATEGY, "live_route": OPENING_STRATEGY}

        plan = strategy._build_expected_value_trade_plan(quote, meta, pending_orders=[])

        self.assertTrue(plan.allowed, plan.reject_reason)
        self.assertLess(plan.planned_stop_net_loss_abs, plan.planned_risk_net_loss_abs)
        self.assertLessEqual(plan.planned_risk_net_loss_abs, 5_000)
        self.assertLessEqual(plan.planned_stop_net_loss_abs, 4_100)

    def test_expected_value_plan_allows_opening_positive_lower_without_double_counted_noise(self):
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(
                seed_money=1_000_000,
                capital_utilization_pct=1.0,
                daily_profit_target=10_000,
                daily_loss_limit=-5_000,
                daily_total_loss_limit=-5_000,
                daily_state_path="/tmp/nonexistent-momentum-state.json",
            ),
        )
        now = datetime(2026, 7, 6, 9, 0, 16)
        strategy.set_simulated_now(now)
        strategy._session_start_at = now.replace(hour=9, minute=0, second=0, microsecond=0)
        quote = Quote(
            "000210",
            "000210",
            47_150,
            0,
            2.06,
            46_198,
            47_150,
            46_100,
            100_000,
            4_715_000_000,
            now,
        )
        strategy._price_prediction_for_entry = lambda *_args, **_kwargs: ShortHorizonPrediction(
            ready=True,
            reason="unit_test_opening_positive_lower_no_double_counted_noise",
            horizon_seconds=180,
            sample_count=5,
            predicted_return_pct=0.725,
            lower_bound_return_pct=0.277,
            upper_bound_return_pct=0.90,
            confidence=0.699,
            direction_score=0.715,
            volatility_pct=0.20,
        )
        meta = {
            "strategy_name": OPENING_STRATEGY,
            "live_route": OPENING_STRATEGY,
            "queue_source": "opening_hot_queue",
            "vs_open_pct": 2.06,
            "high_proximity": 1.0,
        }

        plan = strategy._build_expected_value_trade_plan(quote, meta, pending_orders=[])

        self.assertTrue(plan.allowed, plan.reject_reason)
        self.assertGreater(plan.predicted_net, 0)
        self.assertGreater(plan.expected_net, 0.0)
        self.assertLessEqual(plan.planned_risk_net_loss_abs, 4_750)


    def test_expected_value_plan_counts_open_long_risk_against_daily_loss_room(self):
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(
                seed_money=1_000_000,
                capital_utilization_pct=1.0,
                daily_profit_target=10_000,
                daily_loss_limit=-5_000,
                daily_total_loss_limit=-5_000,
                long_stop_loss_cap_amount=3_500,
                daily_state_path="/tmp/nonexistent-momentum-state.json",
            ),
        )
        now = datetime(2026, 6, 8, 10, 30, 0)
        strategy.set_simulated_now(now)
        strategy.positions["006220"] = PositionState(
            symbol="006220",
            buy_price=11_930,
            quantity=16,
            invested_amount=190_880,
            buy_time=now - timedelta(minutes=85),
            planned_stop_net_loss_abs=1_339,
            planned_risk_net_loss_abs=1_741,
        )
        quote = Quote("036170", "036170", 5_490, 120, 2.23, 5_370, 5_510, 5_360, 500_000, 2_745_000_000, now)
        self._install_strong_ev_prediction(
            strategy,
            predicted_return_pct=1.80,
            lower_bound_return_pct=-0.12,
            upper_bound_return_pct=2.20,
            confidence=0.76,
            direction_score=0.74,
        )
        meta = {"strategy_name": INTRADAY_STRATEGY, "live_route": INTRADAY_STRATEGY}

        plan = strategy._build_expected_value_trade_plan(quote, meta, pending_orders=[])

        self.assertTrue(plan.allowed, plan.reject_reason)
        self.assertEqual(meta["open_long_planned_loss_risk"], 1_741)
        self.assertEqual(meta["committed_long_planned_loss_risk"], 1_741)
        self.assertLessEqual(plan.planned_risk_net_loss_abs, 3_259)


    def test_expected_value_plan_does_not_let_historical_ev_block_live_positive_ev(self):
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(
                seed_money=1_000_000,
                capital_utilization_pct=1.0,
                daily_profit_target=10_000,
                daily_loss_limit=-5_000,
                daily_total_loss_limit=-5_000,
                long_stop_loss_cap_amount=5_000,
                daily_state_path="/tmp/nonexistent-momentum-state.json",
            ),
        )
        now = datetime(2026, 5, 27, 13, 57, 42)
        strategy.set_simulated_now(now)
        quote = Quote("001540", "001540", 11_650, 650, 5.91, 11_000, 11_700, 11_000, 500_000, 5_825_000_000, now)
        strategy._price_prediction_for_entry = lambda *_args, **_kwargs: ShortHorizonPrediction(
            ready=True,
            reason="unit_test_001540_shape",
            horizon_seconds=180,
            sample_count=8,
            predicted_return_pct=2.554,
            lower_bound_return_pct=0.304,
            upper_bound_return_pct=3.10,
            confidence=0.779,
            direction_score=0.775,
            volatility_pct=0.40,
        )
        meta = {
            "strategy_name": INTRADAY_STRATEGY,
            "live_route": INTRADAY_STRATEGY,
            "entry_ev": -1_896.56,
            "entry_ev_confidence": "high",
            "entry_ev_closed_trades": 8,
        }

        plan = strategy._build_expected_value_trade_plan(quote, meta, pending_orders=[])

        self.assertTrue(plan.allowed, plan.reject_reason)
        self.assertGreater(plan.quantity, 0)
        self.assertLess(plan.quantity, 85)
        self.assertLessEqual(plan.planned_risk_net_loss_abs, 5_000)






    def test_ev_precheck_honors_recent_loss_symbol_cooldown(self):
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(
                enable_intraday_conviction_lane=True,
                loss_symbol_cooldown_seconds=1800,
                daily_state_path="/tmp/nonexistent-momentum-state.json",
            ),
        )
        now = datetime(2026, 6, 5, 13, 52, 0)
        strategy.set_simulated_now(now)
        quote = Quote("090360", "090360", 151_300, 3_300, 2.23, 148_000, 152_000, 144_000, 900_000, 90_000_000_000, now)
        strategy._mark_symbol_entry_cooldown("090360", seconds=1800)

        reason = strategy._long_ev_precheck_reject_reason(
            quote,
            pending_orders=[],
            strategy_name_override=INTRADAY_STRATEGY,
        )

        self.assertEqual(reason, "symbol_recent_loss_cooldown")

    def test_ev_precheck_rejects_unsupported_long_symbol_shape(self):
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(
                enable_intraday_conviction_lane=True,
                daily_state_path="/tmp/nonexistent-momentum-state.json",
            ),
        )
        now = datetime(2026, 6, 11, 10, 13, 43)
        strategy.set_simulated_now(now)
        quote = Quote("0195S0", "0195S0", 20_515, 3_000, 16.74, 17_570, 20_520, 17_500, 500_000, 10_257_500_000, now)

        reason = strategy._long_ev_precheck_reject_reason(
            quote,
            pending_orders=[],
            strategy_name_override=INTRADAY_STRATEGY,
        )

        self.assertEqual(reason, "unsupported_long_symbol")


    def test_ev_precheck_blocks_unresolved_market_pending_long_entry(self):
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(
                enable_intraday_conviction_lane=True,
                daily_state_path="/tmp/nonexistent-momentum-state.json",
            ),
        )
        now = datetime(2026, 6, 5, 15, 13, 0)
        strategy.set_simulated_now(now)
        strategy._pending_entry_meta["001820"] = {
            "strategy_name": INTRADAY_STRATEGY,
            "live_route": INTRADAY_STRATEGY,
            "pending_order_quantity": 7,
            "pending_order_reference_price": 114_800,
            "planned_risk_net_loss_abs": 4_537,
        }
        quote = Quote("242040", "242040", 8_630, 650, 8.14, 7_980, 8_700, 7_950, 700_000, 6_041_000_000, now)

        reason = strategy._long_ev_precheck_reject_reason(
            quote,
            pending_orders=[],
            strategy_name_override=INTRADAY_STRATEGY,
        )

        self.assertEqual(reason, "pending_long_entry_unresolved")

    def test_planned_stop_exit_runs_before_rebound_guards(self):
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(
                daily_state_path="/tmp/nonexistent-momentum-state.json",
            ),
        )
        now = datetime(2026, 5, 27, 13, 30, 0)
        strategy.set_simulated_now(now)
        strategy.positions["EV004"] = PositionState(
            symbol="EV004",
            buy_price=10_000,
            quantity=10,
            invested_amount=100_000,
            buy_time=now - timedelta(minutes=5),
            planned_stop_net_loss_abs=200,
            planned_target_net_pnl=500,
        )
        quote = Quote("EV004", "EV004", 9_950, -50, -0.5, 10_000, 10_010, 9_940, 100_000, 995_000_000, now)

        order = strategy._default_long_exit(quote)

        self.assertIsNotNone(order)
        self.assertEqual(order.side, OrderSide.SELL)
        self.assertEqual(order.requested_reason, "ev_planned_stop_net")

    def test_planned_stop_runs_while_market_buy_reconcile_is_pending(self):
        strategy = MomentumScalpStrategy(
            market_data=SimpleNamespace(
                client=SimpleNamespace(config=SimpleNamespace(is_paper=True)),
            ),
            config=MomentumScalpConfig(
                paper_position_exit_grace_seconds=20,
                daily_state_path="/tmp/nonexistent-momentum-state.json",
            ),
        )
        now = datetime(2026, 7, 16, 9, 2, 10)
        strategy.set_simulated_now(now)
        strategy.positions["051980"] = PositionState(
            symbol="051980",
            buy_price=1_219,
            quantity=301,
            invested_amount=366_919,
            buy_time=now - timedelta(seconds=5),
            planned_stop_net_loss_abs=2_575,
            planned_target_net_pnl=3_660,
            pending_entry_started_at=now - timedelta(seconds=5),
            pending_entry_reference_price=1_219,
            pending_entry_fill_mode="market_pending",
        )
        quote = Quote(
            "051980", "051980", 1_172, 63, 5.68, 1_109, 1_219, 1_109,
            1_000_000, 1_172_000_000, now,
        )

        order = strategy._default_long_exit(quote)

        self.assertIsNotNone(order)
        self.assertEqual(order.quantity, 301)
        self.assertEqual(order.requested_reason, "ev_planned_stop_net")

    def test_entry_ev_loader_ignores_stale_scorecards_by_calendar_age(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            previous_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                report_dir = Path("reports/2026/05")
                report_dir.mkdir(parents=True)
                today = datetime.now().date()
                stale_date = today - timedelta(days=30)
                recent_date = today - timedelta(days=1)
                stale_card = {
                    "date": stale_date.isoformat(),
                    "log_analysis": {
                        "trade_records": [
                            {
                                "strategy_name": INTRADAY_STRATEGY,
                                "regime_label": "neutral",
                                "hour_bucket": "10",
                                "entry_grade_math": "A",
                                "net_pnl": -5000,
                            }
                        ]
                    },
                }
                recent_card = {
                    "date": recent_date.isoformat(),
                    "log_analysis": {
                        "trade_records": [
                            {
                                "strategy_name": INTRADAY_STRATEGY,
                                "regime_label": "neutral",
                                "hour_bucket": "10",
                                "entry_grade_math": "A",
                                "net_pnl": 1500,
                            }
                        ]
                    },
                }
                (report_dir / f"daily-scorecard.{stale_date.isoformat()}.json").write_text(
                    json.dumps(stale_card),
                    encoding="utf-8",
                )
                (report_dir / f"daily-scorecard.{recent_date.isoformat()}.json").write_text(
                    json.dumps(recent_card),
                    encoding="utf-8",
                )
                cfg = MomentumScalpConfig(
                    ev_window_days=5,
                    conviction_ev_window_days=20,
                    ev_scorecard_max_age_days=8,
                    conviction_ev_scorecard_max_age_days=14,
                    ev_min_samples=1,
                    daily_state_path=str(Path(tmpdir) / "state.json"),
                    strategy_gate_path=str(Path(tmpdir) / "strategy-gates.json"),
                )

                strategy = MomentumScalpStrategy(market_data=None, config=cfg)
                strategy._load_entry_ev_data()

                self.assertEqual(len(strategy._entry_ev_history_records), 1)
                self.assertEqual(strategy._entry_ev_history_records[0]["net_pnl"], 1500)
            finally:
                os.chdir(previous_cwd)

    def test_intraday_conviction_remains_active_until_late_cutoff(self):
        cfg = MomentumScalpConfig(
            enable_intraday_conviction_lane=True,
            opening_conviction_window_minutes=5,
            intraday_conviction_end_minutes_after_open=381,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy._session_start_at = datetime(2026, 5, 13, 9, 0, 0)

        strategy.set_simulated_now(datetime(2026, 5, 13, 15, 10, 0))
        self.assertTrue(strategy._intraday_conviction_window_active())

        strategy.set_simulated_now(datetime(2026, 5, 13, 15, 22, 0))
        self.assertFalse(strategy._intraday_conviction_window_active())

    def test_sell_fill_uses_recent_quote_when_broker_price_is_zero(self):
        cfg = MomentumScalpConfig(
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        now = datetime(2026, 4, 9, 13, 19, 14)
        strategy.set_simulated_now(now)
        strategy.positions["005930"] = PositionState(
            symbol="005930",
            buy_price=10_000,
            quantity=10,
            entry_strategy_name="intraday_conviction_long_strategy",
        )
        strategy._recent_quotes["005930"] = deque(
            [
                Quote(
                    symbol="005930",
                    name="삼성전자",
                    current_price=10_100,
                    change=100,
                    change_rate=1.0,
                    open_price=10_000,
                    high_price=10_120,
                    low_price=9_980,
                    volume=100_000,
                    trade_amount=1_010_000_000,
                    timestamp=now,
                )
            ],
            maxlen=8,
        )

        strategy.on_order_filled(
            OrderResult(
                success=True,
                symbol="005930",
                side=OrderSide.SELL,
                quantity=10,
                price=0,
                requested_price=0,
                timestamp=now,
            )
        )

        self.assertNotIn("005930", strategy.positions)
        self.assertGreater(strategy.daily_pnl.realized_net_pnl, 0)

    def test_pending_fill_does_not_mutate_strategy_positions(self):
        cfg = MomentumScalpConfig(
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        now = datetime(2026, 4, 9, 9, 5, 0)
        strategy.set_simulated_now(now)

        strategy.on_order_filled(
            OrderResult(
                success=True,
                symbol="005930",
                side=OrderSide.BUY,
                quantity=0,
                price=0,
                fill_mode="market_pending",
                timestamp=now,
            )
        )

        self.assertNotIn("005930", strategy.positions)

    def test_pending_market_sell_reference_waits_for_reconcile_before_loss_halt(self):
        cfg = MomentumScalpConfig(
            enable_unrealized_loss_guard=True,
            daily_loss_limit=-5_000,
            daily_total_loss_limit=-5_000,
            daily_state_path="/tmp/nonexistent-momentum-state.json",
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        now = datetime(2026, 6, 9, 12, 36, 19)
        strategy.set_simulated_now(now)
        strategy.daily_pnl.realized_net_pnl = -135
        strategy.positions["128940"] = PositionState(
            symbol="128940",
            buy_price=434_000,
            quantity=2,
            buy_time=now - timedelta(minutes=1),
            entry_strategy_name=INTRADAY_STRATEGY,
            entry_setup_name="expected_value",
        )
        strategy._quotes_cache["128940"] = Quote(
            symbol="128940",
            name="128940",
            current_price=431_500,
            change=-2500,
            change_rate=-0.58,
            open_price=415_000,
            high_price=435_000,
            low_price=431_000,
            volume=100_000,
            trade_amount=43_150_000_000,
            timestamp=now,
        )

        strategy.on_order_filled(
            OrderResult(
                success=True,
                symbol="128940",
                side=OrderSide.SELL,
                quantity=0,
                price=0,
                reference_price=433_000,
                fill_mode="market_pending",
                requested_reason="ev_planned_stop_net",
                timestamp=now,
            )
        )
        strategy._update_daily_breakers()

        pending_pnl = calculate_trade_pnl_from_prices(
            entry_price=434_000,
            exit_price=433_000,
            quantity=2,
            commission_rate=cfg.commission_rate,
            tax_slippage_rate=cfg.tax_slippage_rate,
        ).net_pnl
        self.assertEqual(strategy.positions["128940"].pending_exit_reference_price, 433_000)
        self.assertLess(pending_pnl, 0)
        self.assertEqual(strategy._unrealized_net_pnl_for_daily_breaker(), 0)
        self.assertGreater(int(strategy.daily_pnl.realized_net_pnl) + pending_pnl, -5_000)
        self.assertFalse(strategy._halted)

    def test_partial_reconciled_pending_market_sell_keeps_remaining_exit_pending_for_breaker(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = MomentumScalpConfig(
                enable_unrealized_loss_guard=True,
                daily_loss_limit=-500,
                daily_total_loss_limit=-500,
                daily_state_path=str(Path(tmpdir) / "state.json"),
            )
            strategy = MomentumScalpStrategy(market_data=None, config=cfg)
            now = datetime(2026, 6, 19, 12, 44, 1)
            strategy.set_simulated_now(now)
            strategy.positions["046970"] = PositionState(
                symbol="046970",
                buy_price=8_650,
                quantity=101,
                buy_time=now - timedelta(minutes=1),
                entry_strategy_name=INTRADAY_STRATEGY,
                entry_setup_name="expected_value",
            )
            strategy._quotes_cache["046970"] = Quote(
                symbol="046970",
                name="046970",
                current_price=8_625,
                change=-25,
                change_rate=-0.29,
                open_price=8_990,
                high_price=9_050,
                low_price=8_600,
                volume=1_000_000,
                trade_amount=8_625_000_000,
                timestamp=now,
            )

            strategy.on_order_filled(
                OrderResult(
                    success=True,
                    symbol="046970",
                    side=OrderSide.SELL,
                    quantity=0,
                    price=0,
                    reference_price=8_625,
                    fill_mode="market_pending",
                    requested_reason="protective_stop_net",
                    timestamp=now,
                )
            )
            strategy.on_order_filled(
                OrderResult(
                    success=True,
                    symbol="046970",
                    side=OrderSide.SELL,
                    quantity=11,
                    price=8_625,
                    reference_price=8_625,
                    fill_mode="account_reconciled_estimated",
                    requested_reason="protective_stop_net",
                    timestamp=now + timedelta(seconds=3),
                )
            )

            position = strategy.positions["046970"]
            self.assertEqual(position.quantity, 90)
            self.assertEqual(position.pending_exit_quantity, 90)
            self.assertEqual(position.pending_exit_reference_price, 8_625)
            self.assertEqual(strategy._unrealized_net_pnl_for_daily_breaker(), 0)
            self.assertGreater(strategy.daily_pnl.realized_net_pnl, -500)
            self.assertFalse(strategy._halted)

            strategy.on_order_filled(
                OrderResult(
                    success=True,
                    symbol="046970",
                    side=OrderSide.SELL,
                    quantity=90,
                    price=8_625,
                    reference_price=8_625,
                    fill_mode="account_reconciled_estimated",
                    requested_reason="protective_stop_net",
                    timestamp=now + timedelta(seconds=19),
                )
            )

            self.assertNotIn("046970", strategy.positions)
            self.assertLessEqual(strategy.daily_pnl.realized_net_pnl, -500)
            self.assertTrue(strategy._halted)
            self.assertIn(strategy._halt_reason, {"daily_loss_limit", "daily_total_loss_limit"})

    def test_daily_total_loss_limit_ignores_restored_position_when_config_disabled(self):
        cfg = MomentumScalpConfig(
            enable_unrealized_loss_guard=True,
            daily_total_loss_limit=-500,
            use_restored_pnl_for_daily_breaker=False,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        now = datetime(2026, 4, 9, 10, 0, 0)
        strategy.set_simulated_now(now)
        strategy.positions["464930"] = PositionState(
            symbol="464930",
            buy_price=20_000,
            quantity=1,
            invested_amount=20_000,
            buy_time=now,
            is_restored=True,
            restored_at=now,
            entry_strategy_name=INTRADAY_STRATEGY,
            entry_setup_name="restored_position",
            queue_source="account_restore",
        )
        strategy._quotes_cache["464930"] = Quote(
            symbol="464930",
            name="TIGER 2차전지TOP10 인버스",
            current_price=19_000,
            change=-1000,
            change_rate=-5.0,
            open_price=20_000,
            high_price=20_100,
            low_price=18_900,
            volume=100_000,
            trade_amount=1_900_000_000,
            timestamp=now,
        )
        strategy._update_daily_breakers()

        self.assertFalse(strategy._halted)

    def test_realized_restored_sell_counts_for_daily_loss_breaker(self):
        cfg = MomentumScalpConfig(
            commission_rate=0.0,
            tax_slippage_rate=0.0,
            daily_loss_limit=-5_000,
            daily_total_loss_limit=-5_000,
            use_restored_pnl_for_daily_breaker=False,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        now = datetime(2026, 6, 10, 9, 30, 0)
        strategy.set_simulated_now(now)
        strategy.positions["396300"] = PositionState(
            symbol="396300",
            buy_price=5_600,
            quantity=88,
            invested_amount=492_800,
            buy_time=now - timedelta(minutes=25),
            is_restored=True,
            restored_at=now - timedelta(minutes=1),
            entry_strategy_name=INTRADAY_STRATEGY,
            entry_setup_name="restored_position",
            queue_source="account_restore",
        )

        strategy.on_order_filled(
            OrderResult(
                success=True,
                symbol="396300",
                side=OrderSide.SELL,
                quantity=88,
                price=4_855,
                fill_mode="account_reconciled_confirmed",
                requested_reason="protective_stop_net",
                timestamp=now,
            )
        )

        self.assertEqual(strategy.daily_pnl.realized_net_pnl, -65_560)
        self.assertEqual(strategy._realized_net_pnl_for_daily_breaker(), -65_560)
        self.assertTrue(strategy._halted)
        self.assertIn(strategy._halt_reason, {"daily_loss_limit", "daily_total_loss_limit"})
        self.assertTrue(strategy._sell_fill_ledger[0]["counts_for_daily_breaker"])
        self.assertEqual(strategy._bull_loss_count_today, 0)

    def test_loaded_restored_sell_fill_is_migrated_into_daily_breaker(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = str(Path(tmpdir) / "state.json")
            now = datetime(2026, 6, 10, 11, 0, 0)
            payload = {
                "date": now.strftime("%Y%m%d"),
                "halted": False,
                "halt_reason": "",
                "daily_pnl": {
                    "realized_gross_pnl": -21_560,
                    "realized_net_pnl": -22_545,
                    "fees_paid": 131,
                    "taxes_paid": 854,
                    "trade_count": 1,
                    "win_count": 0,
                    "loss_count": 1,
                    "breakeven_count": 0,
                    "winning_net_pnl_sum": 0,
                    "losing_net_pnl_sum": -22_545,
                    "largest_win_net": 0,
                    "largest_loss_net": -22_545,
                },
                "breaker_excluded_realized_net_pnl": -22_545,
                "ledger_seed_snapshot": {"breaker_excluded_realized_net_pnl": 0},
                "session_start_at": now.replace(hour=9, minute=0, second=0, microsecond=0).isoformat(),
                "sell_fill_ledger": [
                    {
                        "fill_id": "0000007315",
                        "order_no": "0000007315",
                        "symbol": "396300",
                        "trade_key": "396300:2026-06-10T09:29:20",
                        "quantity": 88,
                        "buy_price": 5_100,
                        "sell_price": 4_855,
                        "gross_pnl": -21_560,
                        "net_pnl": -22_545,
                        "fees": 131,
                        "taxes": 854,
                        "counts_for_daily_breaker": False,
                        "count_as_closed_trade": False,
                        "price_estimated": False,
                        "fill_mode": "account_reconciled_confirmed",
                        "timestamp": now.replace(hour=9, minute=30, second=5, microsecond=0).isoformat(),
                        "requested_reason": "protective_stop_net",
                        "entry_strategy_name": INTRADAY_STRATEGY,
                        "entry_setup_name": "restored_position",
                    }
                ],
                "closed_trade_ledger": {
                    "396300:2026-06-10T09:29:20": {
                        "trade_key": "396300:2026-06-10T09:29:20",
                        "symbol": "396300",
                        "strategy_name": INTRADAY_STRATEGY,
                        "setup_name": "restored_position",
                        "net_pnl": -22_545,
                    }
                },
                "positions": [],
            }
            Path(state_path).write_text(json.dumps(payload), encoding="utf-8")

            strategy = MomentumScalpStrategy(
                market_data=None,
                config=MomentumScalpConfig(
                    daily_loss_limit=-5_000,
                    daily_total_loss_limit=-5_000,
                    daily_state_path=state_path,
                    use_restored_pnl_for_daily_breaker=False,
                ),
            )
            strategy.set_simulated_now(now)
            strategy.initialize()

            self.assertEqual(strategy._breaker_excluded_realized_net_pnl, 0)
            self.assertEqual(strategy._realized_net_pnl_for_daily_breaker(), -22_545)
            self.assertTrue(strategy._sell_fill_ledger[0]["counts_for_daily_breaker"])
            self.assertEqual(strategy._sell_fill_ledger[0]["daily_breaker_flag_migrated"], "realized_sell_fill")
            self.assertTrue(strategy._halted)
            self.assertIn(strategy._halt_reason, {"daily_loss_limit", "daily_total_loss_limit"})

    def test_restored_stale_daily_total_loss_halt_releases_when_actual_pnl_above_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = str(Path(tmpdir) / "state.json")
            now = datetime(2026, 6, 9, 12, 40, 0)
            payload = {
                "date": now.strftime("%Y%m%d"),
                "halted": True,
                "halt_reason": "daily_total_loss_limit",
                "daily_pnl": {
                    "realized_gross_pnl": 996,
                    "realized_net_pnl": -2_131,
                    "fees_paid": 407,
                    "taxes_paid": 2720,
                    "trade_count": 2,
                    "win_count": 0,
                    "loss_count": 2,
                    "breakeven_count": 0,
                    "winning_net_pnl_sum": 0,
                    "losing_net_pnl_sum": -2_131,
                    "largest_win_net": 0,
                    "largest_loss_net": -1_996,
                },
                "sell_fill_ledger": [],
                "closed_trade_ledger": {},
                "positions": [],
            }
            Path(state_path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            strategy = MomentumScalpStrategy(
                market_data=None,
                config=MomentumScalpConfig(
                    enable_unrealized_loss_guard=True,
                    daily_loss_limit=-5_000,
                    daily_total_loss_limit=-5_000,
                    daily_state_path=state_path,
                ),
            )
            strategy.set_simulated_now(now)

            strategy._load_daily_state()

            self.assertFalse(strategy._halted)
            self.assertEqual(strategy._halt_reason, "")
            self.assertEqual(strategy.daily_pnl.realized_net_pnl, -2_131)

    def test_daily_loss_near_stop_halts_without_position(self):
        cfg = MomentumScalpConfig(
            daily_loss_limit=-5_000,
            daily_total_loss_limit=-5_000,
            daily_loss_near_stop_buffer=250,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy.daily_pnl.realized_net_pnl = -4_965

        strategy._update_daily_breakers()

        self.assertTrue(strategy._halted)

    def test_daily_loss_near_stop_keeps_position_management_open(self):
        cfg = MomentumScalpConfig(
            daily_loss_limit=-5_000,
            daily_total_loss_limit=-5_000,
            daily_loss_near_stop_buffer=250,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        strategy.daily_pnl.realized_net_pnl = -4_965
        strategy.positions["005930"] = PositionState(
            symbol="005930",
            buy_price=10_000,
            quantity=1,
        )

        strategy._update_daily_breakers()

        self.assertFalse(strategy._halted)

    def test_no_holding_sell_failure_keeps_position_until_account_reconcile(self):
        cfg = MomentumScalpConfig(
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        now = datetime(2026, 4, 9, 13, 27, 1)
        strategy.set_simulated_now(now)
        strategy.positions["464930"] = PositionState(
            symbol="464930",
            buy_price=19_600,
            quantity=21,
            entry_strategy_name=INTRADAY_STRATEGY,
        )

        strategy.on_order_filled(
            OrderResult(
                success=False,
                symbol="464930",
                side=OrderSide.SELL,
                error_category="no_holding",
                timestamp=now,
            )
        )

        self.assertIn("464930", strategy.positions)

        strategy.sync_positions_from_account(
            [
                SimpleNamespace(
                    symbol="464930",
                    quantity=21,
                    avg_price=19600,
                    eval_amount=411600,
                )
            ]
        )

        self.assertIn("464930", strategy.positions)









    def test_daily_profit_lock_halts_when_realized_pnl_is_close_to_target(self):
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(
                daily_profit_target=10_000,
                profit_protect_threshold=8_000,
                daily_profit_lock_buffer=1_500,
                daily_state_path="/tmp/nonexistent-momentum-state.json",
            ),
        )
        strategy.daily_pnl.realized_net_pnl = 8_950

        strategy._update_daily_breakers()

        self.assertFalse(strategy.should_continue())



















    def test_long_entry_shortlist_skips_stale_cached_quotes(self):
        cfg = MomentumScalpConfig(
            enable_intraday_conviction_lane=True,
            intraday_conviction_live_top_n=1,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        now = datetime(2026, 4, 10, 12, 18, 0)
        strategy.set_simulated_now(now)
        strategy._session_start_at = datetime(2026, 4, 10, 9, 0, 0)
        fresh_quote = Quote(
            symbol="100002",
            name="100002",
            current_price=10_460,
            change=460,
            change_rate=4.6,
            open_price=10_000,
            high_price=10_560,
            low_price=9_980,
            volume=280_000,
            trade_amount=2_928_800_000,
            timestamp=now,
        )
        stale_time = now - timedelta(seconds=90)
        stale_quote = Quote(
            symbol="100001",
            name="100001",
            current_price=10_460,
            change=460,
            change_rate=4.6,
            open_price=10_000,
            high_price=10_560,
            low_price=9_980,
            volume=280_000,
            trade_amount=2_928_800_000,
            timestamp=stale_time,
        )
        strategy._latest_math_queue_symbols = ["100001", "100002"]
        strategy._latest_math_queue_source["100001"] = "math_queue"
        strategy._latest_math_queue_source["100002"] = "math_queue"
        strategy._quotes_cache["100001"] = stale_quote
        strategy._quotes_cache["100002"] = fresh_quote
        strategy._recent_quotes["100002"] = deque(
            [
                Quote("100002", "100002", 10_040, 40, 0.4, 10_000, 10_040, 9_980, 80_000, 803_200_000, now),
                Quote("100002", "100002", 10_560, 560, 5.6, 10_000, 10_560, 9_980, 140_000, 1_478_400_000, now),
                Quote("100002", "100002", 10_300, 300, 3.0, 10_000, 10_560, 9_980, 180_000, 1_854_000_000, now),
                Quote("100002", "100002", 10_360, 360, 3.6, 10_000, 10_560, 9_980, 220_000, 2_279_200_000, now),
                Quote("100002", "100002", 10_420, 420, 4.2, 10_000, 10_560, 9_980, 250_000, 2_605_000_000, now),
                fresh_quote,
            ],
            maxlen=8,
        )
        strategy._latest_math_leader_signals["100001"] = LeaderSignal(
            symbol="100001",
            leader_score=1.40,
            leader_percentile=0.98,
            entry_grade="A",
            change_rate=4.6,
            trade_amount=2_928_800_000,
            vs_open_pct=4.6,
            high_proximity=0.82,
            volume_vs_avg=1.4,
            reclaim_speed_ticks=2,
            recent_acceleration_pct=0.18,
            effective_leader_score=1.25,
        )
        strategy._latest_math_leader_signals["100002"] = LeaderSignal(
            symbol="100002",
            leader_score=1.40,
            leader_percentile=0.98,
            entry_grade="A",
            change_rate=4.6,
            trade_amount=2_928_800_000,
            vs_open_pct=4.6,
            high_proximity=0.82,
            volume_vs_avg=1.4,
            reclaim_speed_ticks=2,
            recent_acceleration_pct=0.18,
            effective_leader_score=1.25,
        )
        strategy._entry_ev_for_context = lambda **kwargs: ExpectedValueEstimate(
            strategy_name="intraday_conviction_long_strategy",
            regime_label="bull",
            hour_bucket="12",
            entry_grade="A",
            entry_ev=950.0,
            p_win=0.61,
            confidence="high",
            closed_trades=10,
        )

        shortlist = strategy._long_entry_shortlist([fresh_quote])

        self.assertEqual([quote.symbol for quote in shortlist], ["100002"])

    def test_on_batch_tick_can_emit_inverse_etf_through_regular_ev_route(self):
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(
                inverse_etfs=["114800"],
                market_shock_window_minutes_after_open=45,
                seed_money=1_000_000,
                capital_utilization_pct=1.0,
                max_position_count=2,
                daily_loss_limit=-5_000,
                daily_total_loss_limit=-5_000,
                long_stop_loss_cap_amount=5_000,
                daily_state_path="/tmp/nonexistent-momentum-state.json",
            ),
        )
        now = datetime(2026, 6, 10, 13, 40, 0)
        strategy.set_simulated_now(now)
        strategy._session_start_at = now.replace(hour=9, minute=0, second=0, microsecond=0)
        strategy._entry_ev_for_context = lambda **_kwargs: ExpectedValueEstimate(
            strategy_name=INTRADAY_STRATEGY,
            regime_label="bear",
            hour_bucket="13",
            entry_grade="A",
            entry_ev=1_500.0,
            p_win=0.62,
            confidence="high",
            closed_trades=10,
        )
        weak_market = Quote("005930", "삼성전자", 70_000, -3_000, -4.1, 73_000, 73_100, 69_900, 1_000_000, 70_000_000_000, now)
        inverse = Quote("114800", "KODEX 인버스", 2_080, 70, 3.48, 2_010, 2_090, 2_000, 800_000, 1_664_000_000, now)
        strategy._recent_quotes["005930"] = deque([weak_market], maxlen=8)
        strategy._recent_quotes["114800"] = deque([inverse], maxlen=8)
        self._install_strong_ev_prediction(
            strategy,
            predicted_return_pct=2.20,
            lower_bound_return_pct=0.45,
            upper_bound_return_pct=3.10,
            confidence=0.82,
            direction_score=0.80,
        )

        orders = strategy.on_batch_tick([weak_market, inverse])

        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].symbol, "114800")
        self.assertEqual(orders[0].requested_reason, "expected_value")
        meta = strategy._pending_entry_meta["114800"]
        self.assertEqual(meta["strategy_name"], INTRADAY_STRATEGY)
        self.assertEqual(meta["live_route"], INTRADAY_STRATEGY)

    def test_on_batch_tick_evaluates_all_candidates_and_selects_highest_ev(self):
        cfg = MomentumScalpConfig(
            enable_intraday_conviction_lane=True,
            enable_expected_net_filter=False,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        now = datetime(2026, 4, 13, 13, 10, 0)
        strategy.set_simulated_now(now)
        strategy._active_day = strategy._today()
        strategy._session_start_at = datetime(2026, 4, 13, 9, 0, 0)

        candidate = Quote(
            symbol="100001",
            name="100001",
            current_price=10_420,
            change=420,
            change_rate=4.2,
            open_price=10_000,
            high_price=10_800,
            low_price=9_900,
            volume=260_000,
            trade_amount=2_709_200_000,
            timestamp=now,
        )
        background = Quote(
            symbol="100002",
            name="100002",
            current_price=9_980,
            change=-20,
            change_rate=-0.2,
            open_price=10_000,
            high_price=10_020,
            low_price=9_950,
            volume=110_000,
            trade_amount=1_097_800_000,
            timestamp=now,
        )
        strategy._quotes_cache["100001"] = candidate
        strategy._quotes_cache["100002"] = background
        strategy._update_market_state = lambda quotes: None
        strategy._long_entry_shortlist = lambda _candidates: [candidate, background]
        evaluated = []

        def build_candidate(quote, *, pending_orders):
            evaluated.append(quote.symbol)
            expected_net = 300.0 if quote.symbol == "100001" else 900.0
            return ExpectedValueCandidate(
                quote=quote,
                strategy_name=INTRADAY_STRATEGY,
                metadata={},
                plan=ExpectedValueTradePlan(
                    allowed=True,
                    quantity=1,
                    expected_net=expected_net,
                    predicted_net=int(expected_net + 100),
                    planned_risk_net_loss_abs=200,
                ),
            )

        strategy._build_expected_value_candidate = build_candidate
        strategy._record_expected_value_forecast = lambda *_args, **_kwargs: None
        strategy._order_from_expected_value_candidate = lambda selected: Order(
            symbol=selected.quote.symbol,
            side=OrderSide.BUY,
            quantity=1,
        )

        orders = strategy.on_batch_tick([background])

        self.assertEqual(evaluated, ["100001", "100002"])
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].symbol, "100002")


    def test_losing_trade_sets_bull_loss_count_and_symbol_cooldown(self):
        cfg = MomentumScalpConfig(
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        now = datetime(2026, 4, 7, 11, 5, 0)
        strategy.set_simulated_now(now)
        strategy.positions["000660"] = PositionState(
            symbol="000660",
            buy_price=10_000,
            quantity=10,
            invested_amount=100_000,
            buy_time=now,
            entry_strategy_name="intraday_conviction_long_strategy",
            entry_setup_name="intraday_conviction",
            regime_label="bull",
            post_loss_admission_class="general",
        )

        strategy.on_order_filled(
            OrderResult(
                success=True,
                symbol="000660",
                side=OrderSide.SELL,
                quantity=10,
                price=9_700,
                timestamp=now,
            )
        )

        self.assertEqual(strategy._bull_loss_count_today, 1)
        self.assertGreater(strategy._symbol_entry_cooldown_remaining("000660"), 0.0)

    def test_intraday_conviction_win_recovers_one_bull_loss_count(self):
        cfg = MomentumScalpConfig(
            commission_rate=0.0,
            tax_slippage_rate=0.0,
            bull_risk_mode_profit_recovery_min_net=1_200,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        now = datetime(2026, 5, 6, 12, 58, 0)
        strategy.set_simulated_now(now)
        strategy._bull_loss_count_today = 2
        strategy.positions["192250"] = PositionState(
            symbol="192250",
            buy_price=14_350,
            quantity=11,
            invested_amount=157_850,
            buy_time=now - timedelta(minutes=2),
            entry_strategy_name="intraday_conviction_long_strategy",
            entry_setup_name="intraday_conviction",
            regime_label="bull",
            post_loss_admission_class="general",
        )

        strategy.on_order_filled(
            OrderResult(
                success=True,
                symbol="192250",
                side=OrderSide.SELL,
                quantity=11,
                price=14_590,
                timestamp=now,
            )
        )

        self.assertEqual(strategy._bull_loss_count_today, 1)

    def test_restored_position_win_does_not_recover_bull_loss_count(self):
        cfg = MomentumScalpConfig(
            commission_rate=0.0,
            tax_slippage_rate=0.0,
            use_restored_pnl_for_daily_breaker=False,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        now = datetime(2026, 5, 6, 12, 24, 0)
        strategy.set_simulated_now(now)
        strategy._bull_loss_count_today = 2
        strategy.positions["005930"] = PositionState(
            symbol="005930",
            buy_price=261_750,
            quantity=1,
            invested_amount=261_750,
            buy_time=now - timedelta(minutes=40),
            is_restored=True,
            entry_strategy_name="intraday_conviction_long_strategy",
            entry_setup_name="restored_position",
            queue_source="account_restore",
        )

        strategy.on_order_filled(
            OrderResult(
                success=True,
                symbol="005930",
                side=OrderSide.SELL,
                quantity=1,
                price=266_500,
                timestamp=now,
            )
        )

        self.assertEqual(strategy._bull_loss_count_today, 2)


    def test_partial_sell_does_not_increment_closed_trade_count_or_bull_loss_count(self):
        cfg = MomentumScalpConfig(
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        now = datetime(2026, 4, 9, 11, 10, 0)
        strategy.set_simulated_now(now)
        strategy.positions["000660"] = PositionState(
            symbol="000660",
            buy_price=10_000,
            quantity=10,
            invested_amount=100_000,
            buy_time=now,
            entry_strategy_name="intraday_conviction_long_strategy",
            entry_setup_name="intraday_conviction",
            regime_label="bull",
            post_loss_admission_class="general",
        )

        strategy.on_order_filled(
            OrderResult(
                success=True,
                symbol="000660",
                side=OrderSide.SELL,
                quantity=5,
                price=9_700,
                timestamp=now,
            )
        )

        self.assertEqual(strategy.daily_pnl.trade_count, 0)
        self.assertEqual(strategy._bull_loss_count_today, 0)
        self.assertIn("000660", strategy.positions)
        self.assertEqual(strategy.positions["000660"].quantity, 5)

    def test_sell_failure_with_no_position_response_waits_for_account_snapshot(self):
        cfg = MomentumScalpConfig(
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        now = datetime(2026, 4, 8, 12, 0, 0)
        strategy.set_simulated_now(now)
        strategy.positions["015760"] = PositionState(
            symbol="015760",
            buy_price=42_000,
            quantity=9,
            invested_amount=378_000,
            buy_time=now,
            entry_strategy_name="intraday_conviction_long_strategy",
            entry_setup_name="intraday_conviction",
            regime_label="bull",
        )

        strategy.on_order_filled(
            OrderResult(
                success=False,
                symbol="015760",
                side=OrderSide.SELL,
                message="[40240000] 모의투자 잔고내역이 없습니다.",
                error_code="40240000",
                error_category="no_holding",
                timestamp=now,
            )
        )

        self.assertIn("015760", strategy.positions)

        strategy.sync_positions_from_account([])

        self.assertNotIn("015760", strategy.positions)

    def test_no_holding_sell_failure_account_absent_records_estimated_exit(self):
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(
                commission_rate=0.0,
                tax_slippage_rate=0.0,
                daily_state_path="/tmp/nonexistent-momentum-state.json",
            ),
        )
        now = datetime(2026, 6, 9, 12, 8, 7)
        strategy.set_simulated_now(now)
        strategy.positions["457370"] = PositionState(
            symbol="457370",
            buy_price=14_360,
            quantity=39,
            invested_amount=560_040,
            buy_time=now - timedelta(minutes=5),
            entry_strategy_name=INTRADAY_STRATEGY,
            entry_setup_name="expected_value",
            queue_source="opening_hot_queue",
        )

        inferred = strategy.reconcile_no_holding_sell_failures_from_account(
            [
                OrderResult(
                    success=False,
                    symbol="457370",
                    side=OrderSide.SELL,
                    message="[40240000] 모의투자 잔고내역이 없습니다.",
                    error_code="40240000",
                    error_category="no_holding",
                    reference_price=14_530,
                    timestamp=now,
                )
            ],
            [],
        )

        self.assertEqual(len(inferred), 1)
        self.assertEqual(inferred[0].quantity, 39)
        self.assertEqual(inferred[0].price, 14_530)
        self.assertNotIn("457370", strategy.positions)
        self.assertEqual(strategy.daily_pnl.realized_net_pnl, 6_630)
        self.assertEqual(strategy.daily_pnl.trade_count, 1)

    def test_no_holding_sell_failure_clears_stale_unconfirmed_market_buy_without_pnl(self):
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(
                commission_rate=0.0,
                tax_slippage_rate=0.0,
                daily_state_path="/tmp/nonexistent-momentum-state.json",
            ),
        )
        now = datetime(2026, 6, 30, 10, 14, 31)
        strategy.set_simulated_now(now)
        strategy.positions["066980"] = PositionState(
            symbol="066980",
            buy_price=2_485,
            quantity=302,
            invested_amount=750_470,
            buy_time=now - timedelta(minutes=21),
            entry_strategy_name=INTRADAY_STRATEGY,
            entry_setup_name="expected_value",
            queue_source="math_queue",
            trade_key="066980:2026-06-30T09:53:22",
            pending_entry_started_at=now - timedelta(minutes=21),
            pending_entry_reference_price=2_485,
            pending_entry_fill_mode="market_pending",
        )

        inferred = strategy.reconcile_no_holding_sell_failures_from_account(
            [
                OrderResult(
                    success=False,
                    symbol="066980",
                    side=OrderSide.SELL,
                    message="[40240000] 모의투자 잔고내역이 없습니다.",
                    error_code="40240000",
                    error_category="no_holding",
                    reference_price=2_485,
                    timestamp=now,
                )
            ],
            [],
        )
        strategy.sync_positions_from_account([])

        self.assertEqual(inferred, [])
        self.assertNotIn("066980", strategy.positions)
        self.assertEqual(strategy.daily_pnl.realized_net_pnl, 0)
        self.assertEqual(strategy._sell_fill_ledger, [])

    def test_account_restore_reopens_estimated_no_holding_exit_without_double_counting(self):
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(
                commission_rate=0.0,
                tax_slippage_rate=0.0,
                daily_state_path="/tmp/nonexistent-momentum-state.json",
            ),
        )
        now = datetime(2026, 6, 30, 10, 14, 31)
        strategy.set_simulated_now(now)
        strategy.positions["066980"] = PositionState(
            symbol="066980",
            buy_price=2_485,
            quantity=302,
            invested_amount=750_470,
            buy_time=now - timedelta(minutes=21),
            entry_strategy_name=INTRADAY_STRATEGY,
            entry_setup_name="expected_value",
            queue_source="math_queue",
            entry_ev=8_852.19,
            entry_ev_confidence="live_plan",
            planned_target_net_pnl=10_000,
            planned_stop_net_loss_abs=3_839,
            planned_risk_net_loss_abs=4_991,
            trade_key="066980:2026-06-30T09:53:22",
        )
        inferred = strategy.reconcile_no_holding_sell_failures_from_account(
            [
                OrderResult(
                    success=False,
                    symbol="066980",
                    side=OrderSide.SELL,
                    message="[40240000] 모의투자 잔고내역이 없습니다.",
                    error_code="40240000",
                    error_category="no_holding",
                    reference_price=2_470,
                    timestamp=now,
                )
            ],
            [],
        )
        self.assertEqual(len(inferred), 1)
        self.assertLess(strategy.daily_pnl.realized_net_pnl, 0)
        self.assertNotIn("066980", strategy.positions)

        strategy.sync_positions_from_account(
            [
                SimpleNamespace(
                    symbol="066980",
                    quantity=302,
                    avg_price=2485,
                    eval_amount=750470,
                )
            ]
        )

        self.assertEqual(strategy.daily_pnl.realized_net_pnl, 0)
        self.assertEqual(strategy._sell_fill_ledger, [])
        self.assertNotIn("066980:2026-06-30T09:53:22", strategy._closed_trade_ledger)
        self.assertIn("066980", strategy.positions)
        restored = strategy.positions["066980"]
        self.assertFalse(restored.is_restored)
        self.assertEqual(restored.entry_setup_name, "expected_value")
        self.assertEqual(restored.queue_source, "math_queue")
        self.assertEqual(restored.planned_risk_net_loss_abs, 4_991)





    def test_daily_state_round_trip_restores_positions_pool_and_symbol_cooldown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "momentum_scalp_daily_state.json"
            cfg = MomentumScalpConfig(
                daily_state_path=str(state_path),
            )
            now = datetime(2026, 4, 10, 11, 15, 0)
            strategy = MomentumScalpStrategy(market_data=None, config=cfg)
            strategy.set_simulated_now(now)
            strategy._active_day = strategy._today()
            strategy._session_start_at = datetime(2026, 4, 10, 9, 0, 0)
            strategy._pool = ["129920", "043260", "046970"]
            strategy._latest_math_queue_symbols = ["129920", "043260"]
            strategy._latest_math_backfill_symbols = ["046970"]
            strategy._latest_opening_fast_symbols = {"129920"}
            strategy._latest_opening_hot_symbols = {"043260"}
            strategy._latest_math_queue_source = {
                "129920": "opening_fast_queue",
                "043260": "opening_hot_queue",
                "046970": "math_backfill",
            }
            strategy._mark_symbol_entry_cooldown("129920", seconds=900)
            strategy.positions["129920"] = PositionState(
                symbol="129920",
                buy_price=10_400,
                quantity=3,
                invested_amount=31_200,
                buy_time=datetime(2026, 4, 10, 10, 45, 0),
                entry_strategy_name="intraday_conviction_long_strategy",
                entry_setup_name="intraday_conviction",
                queue_source="math_queue",
                conviction_score=1.24,
                conviction_rank=1,
            )
            strategy.daily_pnl.realized_net_pnl = 1200

            strategy._save_daily_state()

            restored = MomentumScalpStrategy(market_data=None, config=cfg)
            restored.set_simulated_now(now)
            restored.initialize()

            self.assertTrue(restored.has_runtime_state_snapshot())
            self.assertEqual(restored._pool[:3], ["129920", "043260", "046970"])
            self.assertEqual(restored._latest_math_queue_symbols, ["129920", "043260"])
            self.assertIn("129920", restored._latest_opening_fast_symbols)
            self.assertGreater(restored._symbol_entry_cooldown_remaining("129920"), 0.0)
            self.assertIn("129920", restored.positions)
            self.assertEqual(restored.positions["129920"].entry_setup_name, "intraday_conviction")
            self.assertEqual(restored.positions["129920"].conviction_rank, 1)
            self.assertEqual(restored.daily_pnl.realized_net_pnl, 1200)

    def test_symbol_order_unavailable_failure_blocks_symbol_for_day(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "momentum_scalp_daily_state.json"
            cfg = MomentumScalpConfig(
                daily_state_path=str(state_path),
            )
            now = datetime(2026, 7, 7, 9, 13, 0)
            strategy = MomentumScalpStrategy(market_data=None, config=cfg)
            strategy.set_simulated_now(now)
            strategy.initialize()

            strategy.on_order_filled(
                OrderResult(
                    success=False,
                    symbol="114800",
                    side=OrderSide.BUY,
                    error_code="40070000",
                    error_category="symbol_order_unavailable",
                    message="[40070000] 모의투자 주문처리가 안되었습니다(매매불가 종목)",
                    timestamp=now,
                )
            )

            quote = Quote(
                symbol="114800",
                name="KODEX 인버스",
                current_price=988,
                change=18,
                change_rate=1.86,
                open_price=970,
                high_price=988,
                low_price=970,
                volume=1_000_000,
                trade_amount=988_000_000,
                timestamp=now + timedelta(seconds=5),
            )
            self.assertTrue(strategy._is_symbol_order_unavailable("114800"))
            strategy._quotes_cache["114800"] = quote
            self.assertEqual(strategy._long_entry_shortlist([quote]), [])
            self.assertEqual(
                strategy._long_ev_precheck_reject_reason(
                    quote,
                    pending_orders=[],
                    strategy_name_override=INTRADAY_STRATEGY,
                    skip_capacity=True,
                ),
                "symbol_order_unavailable",
            )

            restored = MomentumScalpStrategy(market_data=None, config=cfg)
            restored.set_simulated_now(now + timedelta(minutes=1))
            restored.initialize()

            self.assertTrue(restored._is_symbol_order_unavailable("114800"))
            self.assertEqual(
                restored._long_ev_precheck_reject_reason(
                    quote,
                    pending_orders=[],
                    strategy_name_override=INTRADAY_STRATEGY,
                    skip_capacity=True,
                ),
                "symbol_order_unavailable",
            )

    def test_update_runtime_pool_persists_daily_state_without_manual_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "momentum_scalp_daily_state.json"
            cfg = MomentumScalpConfig(
                daily_state_path=str(state_path),
            )
            now = datetime(2026, 4, 10, 10, 0, 0)
            strategy = MomentumScalpStrategy(market_data=None, config=cfg)
            strategy.set_simulated_now(now)
            strategy.initialize()

            strategy.update_runtime_pool(["129920", "043260", "046970"])

            restored = MomentumScalpStrategy(market_data=None, config=cfg)
            restored.set_simulated_now(now)
            restored.initialize()

            self.assertTrue(restored.has_runtime_state_snapshot())
            self.assertEqual(restored._pool[:3], ["129920", "043260", "046970"])

    def test_symbol_entry_cooldown_persists_without_manual_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "momentum_scalp_daily_state.json"
            cfg = MomentumScalpConfig(
                daily_state_path=str(state_path),
            )
            now = datetime(2026, 4, 10, 11, 0, 0)
            strategy = MomentumScalpStrategy(market_data=None, config=cfg)
            strategy.set_simulated_now(now)
            strategy.initialize()

            strategy._mark_symbol_entry_cooldown("005930", seconds=900)

            restored = MomentumScalpStrategy(market_data=None, config=cfg)
            restored.set_simulated_now(now)
            restored.initialize()

            self.assertTrue(restored.has_runtime_state_snapshot())
            self.assertGreater(restored._symbol_entry_cooldown_remaining("005930"), 0.0)

    def test_halt_state_persists_without_manual_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "momentum_scalp_daily_state.json"
            cfg = MomentumScalpConfig(
                daily_state_path=str(state_path),
                daily_profit_target=10_000,
            )
            now = datetime(2026, 4, 10, 13, 15, 0)
            strategy = MomentumScalpStrategy(market_data=None, config=cfg)
            strategy.set_simulated_now(now)
            strategy.initialize()
            strategy.daily_pnl.realized_net_pnl = 10_000

            strategy._update_daily_breakers()

            restored = MomentumScalpStrategy(market_data=None, config=cfg)
            restored.set_simulated_now(now)
            restored.initialize()

            self.assertTrue(restored.has_runtime_state_snapshot())
            self.assertFalse(restored.should_continue())
            self.assertEqual(restored.daily_pnl.realized_net_pnl, 10_000)

    def test_default_long_exit_uses_net_stop_before_gross_stop(self):
        cfg = MomentumScalpConfig(
            long_stop_loss_notional_pct=0.007,
            long_stop_loss_cap_amount=2_500,
            commission_rate=0.00015,
            tax_slippage_rate=0.002,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        now = datetime(2026, 4, 10, 9, 3, 0)
        strategy.set_simulated_now(now)
        strategy.positions["009150"] = PositionState(
            symbol="009150",
            buy_price=500_000,
            quantity=1,
            invested_amount=500_000,
            buy_time=datetime(2026, 4, 10, 9, 0, 0),
            high_since_buy=500_000,
            entry_strategy_name="opening_conviction_long_strategy",
        )
        quote = Quote(
            symbol="009150",
            name="삼성전기",
            current_price=497_600,
            change=-2_400,
            change_rate=-0.48,
            open_price=500_000,
            high_price=500_500,
            low_price=497_500,
            volume=80_000,
            trade_amount=39_808_000_000,
            timestamp=now,
        )

        order = strategy._default_long_exit(quote)

        self.assertIsNotNone(order)
        self.assertEqual(order.requested_reason, "protective_stop_net")

    def test_default_long_exit_uses_adaptive_max_hold_minutes(self):
        cfg = MomentumScalpConfig(
            commission_rate=0.0,
            tax_slippage_rate=0.0,
            max_position_holding_minutes=45,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        now = datetime(2026, 5, 20, 14, 10, 0)
        strategy.set_simulated_now(now)
        strategy._session_start_at = datetime(2026, 5, 20, 9, 0, 0)
        strategy.positions["009150"] = PositionState(
            symbol="009150",
            buy_price=100_000,
            quantity=5,
            invested_amount=500_000,
            buy_time=now - timedelta(minutes=13),
            high_since_buy=100_400,
            entry_strategy_name=INTRADAY_STRATEGY,
            adaptive_max_hold_minutes=12,
        )
        quote = Quote(
            symbol="009150",
            name="009150",
            current_price=100_000,
            change=0,
            change_rate=0.0,
            open_price=99_500,
            high_price=100_500,
            low_price=99_000,
            volume=150_000,
            trade_amount=15_000_000_000,
            timestamp=now,
        )

        order = strategy._default_long_exit(quote)

        self.assertIsNotNone(order)
        self.assertEqual(order.requested_reason, "time_exit")

    def test_default_long_exit_defers_too_early_trailing_but_allows_later(self):
        cfg = MomentumScalpConfig(
            commission_rate=0.0,
            tax_slippage_rate=0.0,
            take_profit_pct=1.8,
            trailing_stop_activation_gain_pct=0.45,
            trailing_stop_pct=-0.35,
            max_position_holding_minutes=45,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        now = datetime(2026, 5, 20, 13, 20, 0)
        strategy.set_simulated_now(now)
        strategy._session_start_at = datetime(2026, 5, 20, 9, 0, 0)
        strategy.positions["072950"] = PositionState(
            symbol="072950",
            buy_price=10_000,
            quantity=10,
            invested_amount=100_000,
            buy_time=now - timedelta(seconds=60),
            high_since_buy=10_120,
            entry_strategy_name=INTRADAY_STRATEGY,
            adaptive_max_hold_minutes=20,
        )
        quote = Quote(
            symbol="072950",
            name="072950",
            current_price=10_070,
            change=70,
            change_rate=0.70,
            open_price=10_000,
            high_price=10_120,
            low_price=9_980,
            volume=200_000,
            trade_amount=2_014_000_000,
            timestamp=now,
        )

        early_order = strategy._default_long_exit(quote)

        self.assertIsNone(early_order)

        strategy.positions["072950"].buy_time = now - timedelta(minutes=3)
        later_order = strategy._default_long_exit(quote)

        self.assertIsNotNone(later_order)
        self.assertEqual(later_order.requested_reason, "trailing_stop")

    def test_make_sell_order_uses_direct_market_exit_in_paper_mode(self):
        cfg = MomentumScalpConfig(
        )
        strategy = MomentumScalpStrategy(
            market_data=SimpleNamespace(client=SimpleNamespace(config=SimpleNamespace(is_paper=True))),
            config=cfg,
        )
        pos = PositionState(
            symbol="005930",
            buy_price=71_000,
            quantity=3,
            invested_amount=213_000,
            buy_time=datetime(2026, 4, 9, 10, 0, 0),
            high_since_buy=71_800,
        )
        strategy._quotes_cache["005930"] = Quote(
            "005930",
            "005930",
            70_500,
            -500,
            -0.70,
            71_000,
            71_800,
            70_400,
            100_000,
            7_050_000_000,
            datetime(2026, 4, 9, 10, 1, 0),
        )

        order = strategy._make_sell_order(pos, 3, reason="protective_stop")

        self.assertEqual(order.order_type.value, "01")
        self.assertEqual(order.protective_exit_mode, "")
        self.assertEqual(order.protective_limit_price, 0)
        self.assertEqual(order.protective_fallback_polls, 0)
        self.assertEqual(order.reference_price, 70_500)

    def test_conviction_entry_ev_uses_best_partial_match_when_exact_bucket_missing(self):
        strategy = MomentumScalpStrategy(market_data=None, config=MomentumScalpConfig())
        strategy.set_simulated_now(datetime(2026, 4, 6, 13, 15, 0))
        strategy._entry_ev_table = {
            (
                "bull_breakout_strategy",
                "bull",
                "09",
                "A",
            ): ExpectedValueEstimate(
                strategy_name="bull_breakout_strategy",
                regime_label="bull",
                hour_bucket="09",
                entry_grade="A",
                entry_ev=210.0,
                p_win=0.54,
                confidence="medium",
                closed_trades=5,
            ),
            (
                "bull_breakout_strategy",
                "bull",
                "13",
                "B",
            ): ExpectedValueEstimate(
                strategy_name="bull_breakout_strategy",
                regime_label="bull",
                hour_bucket="13",
                entry_grade="B",
                entry_ev=120.0,
                p_win=0.52,
                confidence="medium",
                closed_trades=4,
            ),
        }

        estimate = strategy._entry_ev_for_context(
            strategy_name="intraday_conviction_long_strategy",
            regime_label="bull",
            entry_grade_math="A",
        )

        self.assertEqual(estimate.strategy_name, "intraday_conviction_long_strategy")
        self.assertEqual(estimate.closed_trades, 5)
        self.assertEqual(estimate.confidence, "medium")
        self.assertEqual(estimate.entry_ev, 210.0)

    def test_reconcile_pending_buy_from_account_keeps_live_entry_context(self):
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(daily_state_path="/tmp/nonexistent-momentum-state.json"),
        )
        now = datetime(2026, 4, 22, 9, 7, 3)
        strategy.set_simulated_now(now)
        strategy._pending_entry_meta["100790"] = {
            "strategy_name": "intraday_conviction_long_strategy",
            "setup_name": "intraday_conviction",
            "entry_reason": "intraday_conviction",
            "regime_label": "bull",
            "queue_source": "opening_hot_queue",
            "execution_mode": "live",
            "live_route": "intraday_conviction_long_strategy",
            "size_multiplier": 0.825,
        }
        pending = OrderResult(
            success=True,
            symbol="100790",
            side=OrderSide.BUY,
            quantity=0,
            price=0,
            fill_mode="market_pending",
            timestamp=now,
        )
        account_pos = SimpleNamespace(symbol="100790", quantity=10, avg_price=54100)

        inferred = strategy.reconcile_pending_fills_from_account([pending], [account_pos])

        self.assertEqual(len(inferred), 1)
        self.assertEqual(inferred[0].quantity, 10)
        self.assertEqual(inferred[0].price, 54100)
        self.assertNotIn("100790", strategy._pending_entry_meta)
        self.assertIn("100790", strategy.positions)
        self.assertFalse(strategy.positions["100790"].is_restored)
        self.assertEqual(strategy.positions["100790"].entry_strategy_name, "intraday_conviction_long_strategy")
        self.assertEqual(strategy.positions["100790"].queue_source, "opening_hot_queue")

    def test_account_sync_promotes_pending_buy_to_live_position_for_daily_target(self):
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(
                commission_rate=0.0,
                tax_slippage_rate=0.0,
                daily_profit_target=10_000,
                daily_state_path="/tmp/nonexistent-momentum-state.json",
            ),
        )
        now = datetime(2026, 5, 4, 10, 22, 57)
        strategy.set_simulated_now(now)
        strategy._pending_entry_meta["006910"] = {
            "strategy_name": "intraday_conviction_long_strategy",
            "setup_name": "intraday_conviction",
            "entry_reason": "intraday_conviction",
            "regime_label": "bull",
            "queue_source": "math_queue",
            "execution_mode": "live",
            "live_route": "intraday_conviction_long_strategy",
        }
        account_pos = SimpleNamespace(symbol="006910", quantity=20, avg_price=13150, eval_amount=263000)

        strategy.sync_positions_from_account([account_pos])

        self.assertNotIn("006910", strategy._pending_entry_meta)
        self.assertIn("006910", strategy.positions)
        self.assertFalse(strategy.positions["006910"].is_restored)
        self.assertEqual(strategy.positions["006910"].entry_setup_name, "intraday_conviction")
        self.assertEqual(strategy.positions["006910"].queue_source, "math_queue")

        strategy.on_order_filled(
            OrderResult(
                success=True,
                symbol="006910",
                side=OrderSide.SELL,
                quantity=10,
                price=14650,
                requested_reason="partial_take_profit",
                timestamp=now + timedelta(minutes=1),
            )
        )

        self.assertEqual(strategy.daily_pnl.realized_net_pnl, 15000)
        self.assertEqual(strategy._realized_net_pnl_for_daily_breaker(), 15000)
        self.assertFalse(strategy.should_continue())

    def test_account_restore_logs_existing_holding_to_order_log(self):
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(
                commission_rate=0.0,
                tax_slippage_rate=0.0,
                daily_state_path="/tmp/nonexistent-momentum-state.json",
            ),
        )
        strategy.set_simulated_now(datetime(2026, 7, 8, 9, 0, 10))

        with self.assertLogs("kis_trader.orders", level="INFO") as captured:
            strategy.sync_positions_from_account(
                [SimpleNamespace(symbol="365660", quantity=3, avg_price=10_950, eval_amount=32_850)]
            )

        message = "\n".join(captured.output)
        self.assertIn("기존보유 복원: 365660 3주 @ 평균단가 10,950원", message)
        self.assertIn("실시간 매수주문 아님", message)
        self.assertTrue(strategy.positions["365660"].is_restored)
        self.assertEqual(strategy.positions["365660"].queue_source, "account_restore")

    def test_account_sync_clears_stale_pending_buy_missing_from_account(self):
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(daily_state_path="/tmp/nonexistent-momentum-state.json"),
        )
        created_at = datetime(2026, 6, 9, 9, 11, 27)
        strategy.set_simulated_now(created_at + timedelta(minutes=5))
        strategy._pending_entry_meta["457370"] = {
            "strategy_name": INTRADAY_STRATEGY,
            "setup_name": "expected_value",
            "queue_source": "opening_hot_queue",
            "pending_order_quantity": 39,
            "pending_order_reference_price": 14_360,
            "pending_order_created_at": created_at.isoformat(timespec="seconds"),
        }

        strategy.sync_positions_from_account([])

        self.assertNotIn("457370", strategy._pending_entry_meta)

    def test_pending_buy_meta_survives_market_pending_restart_window(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = str(Path(tmpdir) / "state.json")
            now = datetime(2026, 5, 4, 9, 38, 27)
            strategy = MomentumScalpStrategy(
                market_data=None,
                config=MomentumScalpConfig(daily_state_path=state_path),
            )
            strategy.set_simulated_now(now)
            strategy._pending_entry_meta["006910"] = {
                "strategy_name": "intraday_conviction_long_strategy",
                "setup_name": "intraday_conviction",
                "queue_source": "math_queue",
            }

            strategy.on_order_filled(
                OrderResult(
                    success=True,
                    symbol="006910",
                    side=OrderSide.BUY,
                    quantity=0,
                    price=0,
                    fill_mode="market_pending",
                    timestamp=now,
                )
            )

            restored = MomentumScalpStrategy(
                market_data=None,
                config=MomentumScalpConfig(daily_state_path=state_path),
            )
            restored.set_simulated_now(now + timedelta(minutes=1))
            restored.initialize()

            self.assertIn("006910", restored._pending_entry_meta)
            self.assertEqual(restored._pending_entry_meta["006910"]["queue_source"], "math_queue")

    def test_market_pending_buy_creates_provisional_live_position(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = str(Path(tmpdir) / "state.json")
            now = datetime(2026, 6, 5, 15, 0, 16)
            strategy = MomentumScalpStrategy(
                market_data=None,
                config=MomentumScalpConfig(daily_state_path=state_path),
            )
            strategy.set_simulated_now(now)
            strategy._pending_entry_meta["001820"] = {
                "strategy_name": INTRADAY_STRATEGY,
                "setup_name": "expected_value",
                "entry_reason": "expected_value",
                "queue_source": "math_queue",
                "execution_mode": "live",
                "live_route": INTRADAY_STRATEGY,
                "pending_order_quantity": 7,
                "pending_order_reference_price": 114_800,
                "planned_target_net_pnl": 5_408,
                "planned_stop_net_loss_abs": 3_500,
                "planned_risk_net_loss_abs": 4_537,
                "entry_expected_net_pnl": 4_053.3,
            }

            strategy.on_order_filled(
                OrderResult(
                    success=True,
                    symbol="001820",
                    side=OrderSide.BUY,
                    quantity=0,
                    price=0,
                    reference_price=114_800,
                    fill_mode="market_pending",
                    timestamp=now,
                )
            )

            self.assertNotIn("001820", strategy._pending_entry_meta)
            self.assertIn("001820", strategy.positions)
            position = strategy.positions["001820"]
            self.assertEqual(position.quantity, 7)
            self.assertEqual(position.buy_price, 114_800)
            self.assertEqual(position.planned_target_net_pnl, 5_408)
            self.assertEqual(position.planned_stop_net_loss_abs, 3_500)

            restored = MomentumScalpStrategy(
                market_data=None,
                config=MomentumScalpConfig(daily_state_path=state_path),
            )
            restored.set_simulated_now(now + timedelta(minutes=1))
            restored.initialize()

            self.assertIn("001820", restored.positions)
            self.assertNotIn("001820", restored._pending_entry_meta)
            self.assertEqual(restored.positions["001820"].quantity, 7)
            self.assertIsNotNone(restored.positions["001820"].pending_entry_started_at)
            self.assertEqual(restored.positions["001820"].pending_entry_reference_price, 114_800)

    def test_account_sync_drops_unconfirmed_market_pending_position_after_grace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = str(Path(tmpdir) / "state.json")
            now = datetime(2026, 6, 10, 9, 4, 31)
            strategy = MomentumScalpStrategy(
                market_data=None,
                config=MomentumScalpConfig(daily_state_path=state_path),
            )
            strategy.set_simulated_now(now)
            strategy._pending_entry_meta["396300"] = {
                "strategy_name": OPENING_STRATEGY,
                "setup_name": "expected_value",
                "entry_reason": "expected_value",
                "queue_source": "math_queue",
                "execution_mode": "live",
                "live_route": OPENING_STRATEGY,
                "pending_order_quantity": 88,
                "pending_order_reference_price": 5_600,
                "planned_target_net_pnl": 10_000,
                "planned_stop_net_loss_abs": 3_458,
                "planned_risk_net_loss_abs": 4_989,
                "entry_expected_net_pnl": 3_967.9,
            }

            strategy.on_order_filled(
                OrderResult(
                    success=True,
                    symbol="396300",
                    side=OrderSide.BUY,
                    quantity=0,
                    price=0,
                    reference_price=5_600,
                    fill_mode="market_pending",
                    timestamp=now,
                )
            )
            strategy.set_simulated_now(now + timedelta(minutes=2))

            strategy.sync_positions_from_account([])

            self.assertIn("396300", strategy.positions)
            position = strategy.positions["396300"]
            self.assertEqual(position.quantity, 88)
            self.assertEqual(position.buy_price, 5_600)
            self.assertEqual(position.pending_entry_fill_mode, "market_pending")
            self.assertEqual(position.planned_risk_net_loss_abs, 4_989)

            strategy.set_simulated_now(now + timedelta(minutes=4))
            strategy.sync_positions_from_account([])

            self.assertNotIn("396300", strategy.positions)
            self.assertEqual(strategy.daily_pnl.realized_net_pnl, 0)
            self.assertEqual(strategy.daily_pnl.trade_count, 0)

    def test_account_sync_reprices_market_pending_ev_when_fill_slips_past_prediction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = str(Path(tmpdir) / "state.json")
            now = datetime(2026, 7, 7, 12, 47, 21)
            strategy = MomentumScalpStrategy(
                market_data=None,
                config=MomentumScalpConfig(daily_state_path=state_path),
            )
            strategy.set_simulated_now(now)
            strategy._pending_entry_meta["365660"] = {
                "strategy_name": INTRADAY_STRATEGY,
                "setup_name": "expected_value",
                "entry_reason": "expected_value",
                "queue_source": "math_queue",
                "execution_mode": "live",
                "live_route": INTRADAY_STRATEGY,
                "pending_order_quantity": 62,
                "pending_order_reference_price": 11_980,
                "entry_signal_price": 11_980,
                "price_prediction_return_pct": 2.691,
                "price_prediction_lower_pct": 0.043,
                "price_prediction_upper_pct": 3.20,
                "entry_prediction_win_probability": 0.464,
                "planned_target_net_pnl": 10_000,
                "planned_stop_net_loss_abs": 3_815,
                "planned_risk_net_loss_abs": 4_959,
                "entry_expected_net_pnl": 2_507.4,
                "price_prediction_net_pnl": 22_542,
                "price_prediction_lower_net_pnl": -3_504,
            }

            strategy.on_order_filled(
                OrderResult(
                    success=True,
                    symbol="365660",
                    side=OrderSide.BUY,
                    quantity=0,
                    price=0,
                    reference_price=11_980,
                    fill_mode="market_pending",
                    timestamp=now,
                )
            )

            strategy.set_simulated_now(now + timedelta(minutes=2))
            strategy.sync_positions_from_account(
                [SimpleNamespace(symbol="365660", quantity=8, avg_price=12_220, eval_amount=97_760)]
            )

            position = strategy.positions["365660"]
            self.assertEqual(position.quantity, 8)
            self.assertEqual(position.buy_price, 12_220)
            self.assertEqual(position.entry_signal_price, 11_980)
            self.assertEqual(position.entry_ev_confidence, "live_plan_repriced")
            self.assertLess(position.entry_expected_net_pnl, 0)
            self.assertEqual(position.planned_stop_net_loss_abs, 1)
            self.assertEqual(position.planned_risk_net_loss_abs, 4_959)

            exit_order = strategy._default_long_exit(
                Quote("365660", "365660", 12_220, 0, 0.0, 11_980, 12_220, 11_980, 1_000_000, 12_220_000_000, now)
            )

            self.assertIsNotNone(exit_order)
            self.assertEqual(exit_order.requested_reason, "ev_planned_stop_net")

    def test_reprice_keeps_accepted_stop_when_favorable_fill_lower_bound_only_covers_cost(self):
        now = datetime(2026, 7, 20, 9, 7, 41)
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(daily_state_path="/tmp/nonexistent-momentum-state.json"),
        )
        strategy.set_simulated_now(now)
        position = PositionState(
            symbol="005930",
            buy_price=252_250,
            quantity=3,
            buy_time=now - timedelta(seconds=13),
            entry_setup_name="expected_value",
            entry_signal_price=252_500,
            entry_prediction_return_pct=0.760,
            entry_prediction_lower_pct=0.227,
            entry_prediction_upper_pct=1.293,
            entry_prediction_win_probability=0.782,
            planned_target_net_pnl=2_492,
            planned_stop_net_loss_abs=3_253,
            planned_risk_net_loss_abs=3_768,
        )

        strategy._reprice_position_ev_after_confirmed_entry(position)
        strategy.positions[position.symbol] = position
        exit_order = strategy._default_long_exit(
            Quote(
                "005930",
                "005930",
                252_000,
                0,
                0.0,
                252_500,
                252_500,
                252_000,
                1_000_000,
                252_000_000_000,
                now,
            )
        )

        self.assertEqual(position.planned_stop_net_loss_abs, 3_253)
        self.assertEqual(position.planned_risk_net_loss_abs, 3_768)
        self.assertIsNone(exit_order)

    def test_reprice_preserves_full_downside_after_entry_slippage(self):
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(daily_state_path="/tmp/nonexistent-momentum-state.json"),
        )
        position = PositionState(
            symbol="049080",
            buy_price=13_550,
            quantity=7,
            entry_setup_name="expected_value",
            entry_signal_price=13_470,
            entry_prediction_return_pct=2.302,
            entry_prediction_lower_pct=-1.764,
            entry_prediction_upper_pct=3.0,
            entry_prediction_win_probability=0.632,
            planned_target_net_pnl=1_768,
            planned_stop_net_loss_abs=661,
            planned_risk_net_loss_abs=2_125,
        )

        strategy._reprice_position_ev_after_confirmed_entry(position)

        self.assertEqual(position.entry_prediction_lower_net_pnl, -2_530)
        self.assertEqual(position.planned_stop_net_loss_abs, 1)
        self.assertGreater(position.planned_risk_net_loss_abs, 2_530)
        self.assertLess(position.entry_expected_net_pnl, 0.0)

    def test_partial_buy_reconciles_to_account_total_before_ev_exit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            now = datetime(2026, 7, 15, 13, 39, 22)
            strategy = MomentumScalpStrategy(
                market_data=None,
                config=MomentumScalpConfig(daily_state_path=str(Path(tmpdir) / "state.json")),
            )
            strategy.set_simulated_now(now)
            strategy._pending_entry_meta["303360"] = {
                "strategy_name": INTRADAY_STRATEGY,
                "setup_name": "expected_value",
                "entry_reason": "expected_value",
                "queue_source": "math_queue",
                "pending_order_quantity": 143,
                "pending_order_reference_price": 5_140,
                "entry_signal_price": 5_140,
                "price_prediction_return_pct": 0.979,
                "price_prediction_lower_pct": 0.147,
                "price_prediction_upper_pct": 1.811,
                "entry_prediction_win_probability": 0.725308,
                "planned_target_net_pnl": 3_016,
                "planned_stop_net_loss_abs": 4_121,
                "planned_risk_net_loss_abs": 4_718,
                "entry_expected_net_pnl": 890.8,
                "price_prediction_net_pnl": 3_016,
                "price_prediction_lower_net_pnl": -3_120,
            }
            partial = OrderResult(
                success=True,
                symbol="303360",
                side=OrderSide.BUY,
                quantity=2,
                price=5_140,
                reference_price=5_140,
                fill_mode="partial_fill_pending",
                requested_quantity=143,
                timestamp=now,
            )

            strategy.on_order_filled(partial)

            self.assertEqual(strategy.positions["303360"].quantity, 2)
            self.assertIsNotNone(strategy.positions["303360"].pending_entry_started_at)
            self.assertIn("303360", strategy._pending_entry_meta)
            self.assertIsNone(strategy._default_long_exit(Quote(
                "303360", "303360", 5_130, 0, 0.0, 4_000, 5_170, 4_000, 1_000_000, 5_130_000_000, now
            )))

            inferred = strategy.reconcile_pending_fills_from_account(
                [partial],
                [SimpleNamespace(symbol="303360", quantity=143, avg_price=5_140, eval_amount=735_020)],
            )

            position = strategy.positions["303360"]
            self.assertEqual(len(inferred), 1)
            self.assertEqual(position.quantity, 143)
            self.assertIsNone(position.pending_entry_started_at)
            self.assertNotIn("303360", strategy._pending_entry_meta)
            self.assertGreater(position.planned_stop_net_loss_abs, 24)

    def test_pending_market_buy_meta_counts_as_exposure_and_loss_room(self):
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(
                seed_money=1_000_000,
                capital_utilization_pct=1.0,
                daily_profit_target=10_000,
                daily_loss_limit=-5_000,
                daily_total_loss_limit=-5_000,
                daily_state_path="/tmp/nonexistent-momentum-state.json",
            ),
        )
        strategy._pending_entry_meta["425040"] = {
            "strategy_name": INTRADAY_STRATEGY,
            "live_route": INTRADAY_STRATEGY,
            "pending_order_quantity": 43,
            "pending_order_reference_price": 20_250,
            "planned_risk_net_loss_abs": 4_290,
        }
        pending_order = Order(
            symbol="425040",
            side=OrderSide.BUY,
            quantity=43,
            price=0,
            reference_price=20_250,
        )

        self.assertEqual(strategy._pending_long_exposure_amount([]), 870_750)
        self.assertEqual(strategy._pending_long_planned_loss_risk([]), 4_290)
        self.assertEqual(strategy._remaining_long_seed_exposure_budget([]), 129_250)
        self.assertEqual(strategy._daily_loss_room() - strategy._pending_long_planned_loss_risk([]), 710)
        self.assertEqual(strategy._pending_long_exposure_amount([pending_order]), 870_750)

    def test_network_pending_buy_without_account_position_keeps_entry_meta_during_grace(self):
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(daily_state_path="/tmp/nonexistent-momentum-state.json"),
        )
        now = datetime(2026, 6, 9, 9, 11, 29)
        strategy.set_simulated_now(now)
        strategy._pending_entry_meta["457370"] = {
            "strategy_name": INTRADAY_STRATEGY,
            "setup_name": "expected_value",
            "queue_source": "opening_hot_queue",
            "pending_order_quantity": 39,
            "pending_order_reference_price": 14_360,
        }
        pending = OrderResult(
            success=True,
            symbol="457370",
            side=OrderSide.BUY,
            quantity=0,
            price=0,
            reference_price=14_360,
            fill_mode="order_result_pending",
            timestamp=now,
        )

        inferred = strategy.reconcile_pending_fills_from_account([pending], [])

        self.assertEqual(inferred, [])
        self.assertIn("457370", strategy._pending_entry_meta)

    def test_stale_network_pending_buy_without_account_position_clears_entry_meta(self):
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(
                daily_state_path="/tmp/nonexistent-momentum-state.json",
            ),
        )
        now = datetime(2026, 6, 9, 9, 11, 29)
        strategy.set_simulated_now(now)
        stale_created_at = now - timedelta(seconds=181)
        strategy._pending_entry_meta["457370"] = {
            "strategy_name": INTRADAY_STRATEGY,
            "setup_name": "expected_value",
            "queue_source": "opening_hot_queue",
            "pending_order_quantity": 39,
            "pending_order_reference_price": 14_360,
            "pending_order_created_at": stale_created_at.isoformat(timespec="seconds"),
        }
        pending = OrderResult(
            success=True,
            symbol="457370",
            side=OrderSide.BUY,
            quantity=0,
            price=0,
            reference_price=14_360,
            fill_mode="order_result_pending",
            timestamp=now,
        )

        inferred = strategy.reconcile_pending_fills_from_account([pending], [])

        self.assertEqual(inferred, [])
        self.assertNotIn("457370", strategy._pending_entry_meta)

    def test_account_sync_records_delayed_pending_sell_quantity_drop(self):
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(
                commission_rate=0.0,
                tax_slippage_rate=0.0,
                daily_state_path="/tmp/nonexistent-momentum-state.json",
            ),
        )
        now = datetime(2026, 4, 22, 9, 31, 21)
        strategy.set_simulated_now(now)
        strategy.positions["007660"] = PositionState(
            symbol="007660",
            buy_price=144900,
            quantity=3,
            entry_strategy_name="intraday_conviction_long_strategy",
            entry_setup_name="intraday_conviction",
            queue_source="math_queue",
        )
        quote = Quote(
            symbol="007660",
            name="007660",
            current_price=146600,
            change=0,
            change_rate=0.0,
            open_price=140000,
            high_price=146800,
            low_price=140000,
            volume=1_000_000,
            trade_amount=146_600_000_000,
            timestamp=now,
        )
        strategy._quotes_cache["007660"] = quote
        strategy.on_order_filled(
            OrderResult(
                success=True,
                order_no="pending-sell-1",
                symbol="007660",
                side=OrderSide.SELL,
                quantity=0,
                price=0,
                reference_price=146600,
                fill_mode="market_pending",
                requested_reason="partial_take_profit",
                timestamp=now,
            )
        )
        account_pos = SimpleNamespace(symbol="007660", quantity=1, avg_price=144900, eval_amount=146600)

        strategy.sync_positions_from_account([account_pos])

        self.assertEqual(strategy.daily_pnl.realized_net_pnl, 3400)
        self.assertEqual(strategy.daily_pnl.trade_count, 0)
        self.assertEqual(strategy.positions["007660"].quantity, 1)
        self.assertTrue(strategy.positions["007660"].partial_exit_done)
        self.assertEqual(strategy._sell_fill_ledger[-1]["order_no"], "pending-sell-1")

        strategy.sync_positions_from_account([])

        self.assertNotIn("007660", strategy.positions)
        self.assertEqual(strategy.daily_pnl.realized_net_pnl, 5100)
        self.assertEqual(strategy.daily_pnl.trade_count, 1)

    def test_filled_small_pending_sell_releases_unreserved_remainder(self):
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(
                commission_rate=0.0,
                tax_slippage_rate=0.0,
                daily_state_path="/tmp/nonexistent-momentum-state.json",
            ),
        )
        now = datetime(2026, 7, 15, 13, 40, 42)
        strategy.set_simulated_now(now)
        strategy.positions["303360"] = PositionState(
            symbol="303360",
            buy_price=5_140,
            quantity=143,
            buy_time=now,
        )
        strategy.on_order_filled(OrderResult(
            success=True,
            order_no="small-sell",
            symbol="303360",
            side=OrderSide.SELL,
            quantity=0,
            reference_price=5_130,
            fill_mode="market_pending",
            requested_quantity=2,
            requested_reason="ev_planned_stop_net",
            timestamp=now,
        ))

        reserved = strategy._make_sell_order(strategy.positions["303360"], 143, reason="liquidate_all")
        self.assertIsNotNone(reserved)
        self.assertEqual(reserved.quantity, 141)

        strategy.on_order_filled(OrderResult(
            success=True,
            order_no="small-sell",
            symbol="303360",
            side=OrderSide.SELL,
            quantity=2,
            price=5_130,
            reference_price=5_130,
            fill_mode="account_reconciled_estimated",
            requested_reason="ev_planned_stop_net",
            timestamp=now + timedelta(seconds=2),
        ))

        position = strategy.positions["303360"]
        self.assertEqual(position.quantity, 141)
        self.assertEqual(position.pending_exit_quantity, 0)
        self.assertIsNone(position.pending_exit_started_at)

    def test_partial_then_full_exit_counts_cumulative_trade_pnl_once(self):
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(
                commission_rate=0.0,
                tax_slippage_rate=0.0,
                daily_state_path="/tmp/nonexistent-momentum-state.json",
            ),
        )
        now = datetime(2026, 4, 27, 9, 1, 0)
        strategy.set_simulated_now(now)
        strategy.positions["006340"] = PositionState(
            symbol="006340",
            buy_price=9_760,
            quantity=71,
            buy_time=now,
            entry_strategy_name=INTRADAY_STRATEGY,
            entry_setup_name="queue_value_scout",
            trade_key="006340:test-trade",
        )

        strategy.on_order_filled(
            OrderResult(
                success=True,
                order_no="10001",
                symbol="006340",
                side=OrderSide.SELL,
                quantity=36,
                price=10_110,
                requested_reason="partial_take_profit",
                timestamp=now,
            )
        )
        strategy.on_order_filled(
            OrderResult(
                success=True,
                order_no="10002",
                symbol="006340",
                side=OrderSide.SELL,
                quantity=35,
                price=10_220,
                requested_reason="take_profit",
                timestamp=now + timedelta(seconds=20),
            )
        )

        self.assertEqual(strategy.daily_pnl.realized_net_pnl, 28_700)
        self.assertEqual(strategy.daily_pnl.realized_gross_pnl, 28_700)
        self.assertEqual(strategy.daily_pnl.trade_count, 1)
        self.assertEqual(strategy.daily_pnl.win_count, 1)
        self.assertEqual(strategy.daily_pnl.winning_net_pnl_sum, 28_700)
        self.assertEqual(strategy.daily_pnl.largest_win_net, 28_700)
        self.assertNotIn("006340", strategy.positions)

    def test_confirm_reconciled_sell_fill_updates_state_and_trade_stats(self):
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(
                commission_rate=0.0,
                tax_slippage_rate=0.0,
                daily_state_path="/tmp/nonexistent-momentum-state.json",
            ),
        )
        now = datetime(2026, 4, 27, 9, 1, 0)
        strategy.set_simulated_now(now)
        strategy.positions["006340"] = PositionState(
            symbol="006340",
            buy_price=9_760,
            quantity=35,
            buy_time=now,
            entry_strategy_name=INTRADAY_STRATEGY,
            entry_setup_name="queue_value_scout",
            trade_key="006340:test-trade",
        )

        strategy.on_order_filled(
            OrderResult(
                success=True,
                order_no="20001",
                symbol="006340",
                side=OrderSide.SELL,
                quantity=35,
                price=10_110,
                fill_mode="account_reconciled_estimated",
                requested_reason="take_profit",
                timestamp=now,
            )
        )
        self.assertEqual(strategy.daily_pnl.realized_net_pnl, 12_250)
        self.assertEqual(strategy.daily_pnl.trade_count, 1)
        self.assertEqual(strategy.daily_pnl.winning_net_pnl_sum, 12_250)

        class DummyAccount:
            @staticmethod
            def _coerce_int(value):
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return 0

            def get_order_history(self, *args, **kwargs):
                return pd.DataFrame(
                    [
                        {
                            "odno": "20001",
                            "pdno": "006340",
                            "tot_ccld_qty": "35",
                            "avg_prvs": "10220",
                            "tot_ccld_amt": "357700",
                        }
                    ]
                )

        corrected_results = [
            OrderResult(
                success=True,
                order_no="20001",
                symbol="006340",
                side=OrderSide.SELL,
                quantity=35,
                price=10_110,
                fill_mode="account_reconciled_estimated",
                timestamp=now,
            )
        ]

        corrections = strategy.confirm_reconciled_sell_fills(DummyAccount(), results=corrected_results)

        self.assertEqual(len(corrections), 1)
        self.assertEqual(corrections[0]["corrected_price"], 10_220)
        self.assertEqual(corrected_results[0].price, 10_220)
        self.assertEqual(corrected_results[0].fill_mode, "account_reconciled_confirmed")
        self.assertEqual(strategy.daily_pnl.realized_net_pnl, 16_100)
        self.assertEqual(strategy.daily_pnl.winning_net_pnl_sum, 16_100)

    def test_estimated_sell_profit_waits_for_confirmation_before_target_halt(self):
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(
                commission_rate=0.0,
                tax_slippage_rate=0.0,
                daily_profit_target=10_000,
                daily_profit_lock_buffer=0,
                daily_state_path="/tmp/nonexistent-momentum-state.json",
            ),
        )
        now = datetime(2026, 5, 27, 13, 40, 0)
        strategy.set_simulated_now(now)
        strategy.positions["274090"] = PositionState(
            symbol="274090",
            buy_price=38_350,
            quantity=26,
            buy_time=now - timedelta(minutes=1),
            entry_strategy_name=INTRADAY_STRATEGY,
            entry_setup_name="expected_value",
            trade_key="274090:test-trade",
        )

        strategy.on_order_filled(
            OrderResult(
                success=True,
                order_no="20002",
                symbol="274090",
                side=OrderSide.SELL,
                quantity=26,
                price=38_750,
                fill_mode="account_reconciled_estimated",
                requested_reason="ev_planned_target_net",
                timestamp=now,
            )
        )

        self.assertEqual(strategy.daily_pnl.realized_net_pnl, 10_400)
        self.assertTrue(strategy.should_continue())

        class DummyAccount:
            @staticmethod
            def _coerce_int(value):
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return 0

            def get_order_history(self, *args, **kwargs):
                return pd.DataFrame(
                    [
                        {
                            "odno": "20002",
                            "pdno": "274090",
                            "tot_ccld_qty": "26",
                            "avg_prvs": "38750",
                            "tot_ccld_amt": str(38_750 * 26),
                        }
                    ]
                )

        strategy.confirm_reconciled_sell_fills(DummyAccount())

        self.assertFalse(strategy.should_continue())

    def test_estimated_target_halt_is_not_kept_after_correction_below_target(self):
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(
                commission_rate=0.0,
                tax_slippage_rate=0.0,
                daily_profit_target=10_000,
                daily_profit_lock_buffer=0,
                daily_state_path="/tmp/nonexistent-momentum-state.json",
            ),
        )
        now = datetime(2026, 5, 27, 13, 40, 0)
        strategy.set_simulated_now(now)
        strategy.positions["274090"] = PositionState(
            symbol="274090",
            buy_price=38_350,
            quantity=26,
            buy_time=now - timedelta(minutes=1),
            entry_strategy_name=INTRADAY_STRATEGY,
            entry_setup_name="expected_value",
            trade_key="274090:test-trade",
        )
        strategy.on_order_filled(
            OrderResult(
                success=True,
                order_no="20003",
                symbol="274090",
                side=OrderSide.SELL,
                quantity=26,
                price=38_750,
                fill_mode="account_reconciled_estimated",
                requested_reason="ev_planned_target_net",
                timestamp=now,
            )
        )

        class DummyAccount:
            @staticmethod
            def _coerce_int(value):
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return 0

            def get_order_history(self, *args, **kwargs):
                return pd.DataFrame(
                    [
                        {
                            "odno": "20003",
                            "pdno": "274090",
                            "tot_ccld_qty": "26",
                            "avg_prvs": "38700",
                            "tot_ccld_amt": str(38_700 * 26),
                        }
                    ]
                )

        strategy.confirm_reconciled_sell_fills(DummyAccount())

        self.assertEqual(strategy.daily_pnl.realized_net_pnl, 9_100)
        self.assertTrue(strategy.should_continue())

    def test_estimated_loss_halt_is_not_kept_after_correction_above_loss_limit(self):
        strategy = MomentumScalpStrategy(
            market_data=None,
            config=MomentumScalpConfig(
                commission_rate=0.0,
                tax_slippage_rate=0.0,
                daily_loss_limit=-5_000,
                daily_total_loss_limit=-5_000,
                daily_loss_near_stop_buffer=0,
                daily_state_path="/tmp/nonexistent-momentum-state.json",
            ),
        )
        now = datetime(2026, 7, 20, 9, 8, 36)
        strategy.set_simulated_now(now)
        strategy.positions["005930"] = PositionState(
            symbol="005930",
            buy_price=10_000,
            quantity=1,
            buy_time=now - timedelta(minutes=2),
            entry_strategy_name=INTRADAY_STRATEGY,
            entry_setup_name="expected_value",
            trade_key="005930:test-trade",
        )
        strategy.on_order_filled(
            OrderResult(
                success=True,
                order_no="72000",
                symbol="005930",
                side=OrderSide.SELL,
                quantity=1,
                price=7_000,
                fill_mode="account_reconciled_confirmed",
                requested_reason="ev_planned_stop_net",
                timestamp=now - timedelta(minutes=1),
            )
        )
        strategy.positions["067310"] = PositionState(
            symbol="067310",
            buy_price=40_100,
            quantity=13,
            buy_time=now - timedelta(seconds=13),
            entry_strategy_name=INTRADAY_STRATEGY,
            entry_setup_name="expected_value",
            trade_key="067310:test-trade",
        )
        strategy.on_order_filled(
            OrderResult(
                success=True,
                order_no="72001",
                symbol="067310",
                side=OrderSide.SELL,
                quantity=13,
                price=39_900,
                fill_mode="account_reconciled_estimated",
                requested_reason="ev_planned_stop_net",
                timestamp=now,
            )
        )

        self.assertLessEqual(strategy.daily_pnl.realized_net_pnl, -5_000)
        self.assertFalse(strategy.should_continue())

        class DummyAccount:
            @staticmethod
            def _coerce_int(value):
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return 0

            def get_order_history(self, *args, **kwargs):
                return pd.DataFrame(
                    [
                        {
                            "odno": "72001",
                            "pdno": "067310",
                            "tot_ccld_qty": "13",
                            "avg_prvs": "40100",
                            "tot_ccld_amt": str(40_100 * 13),
                        }
                    ]
                )

        strategy.confirm_reconciled_sell_fills(DummyAccount())

        self.assertEqual(strategy.daily_pnl.realized_net_pnl, -3_000)
        self.assertTrue(strategy.should_continue())

    def test_leader_signal_cache_miss_uses_conservative_non_leader_fallback(self):
        strategy = MomentumScalpStrategy(market_data=None, config=MomentumScalpConfig())
        now = datetime(2026, 4, 3, 10, 0, 0)
        quote = Quote(
            symbol="AAA",
            name="AAA",
            current_price=10_200,
            change=200,
            change_rate=2.0,
            open_price=10_000,
            high_price=10_250,
            low_price=9_980,
            volume=80_000,
            trade_amount=816_000_000,
            timestamp=now,
        )
        strategy._quotes_cache["AAA"] = quote
        strategy._recent_quotes["AAA"] = deque([quote], maxlen=8)

        signal = strategy._leader_signal_for_quote(quote)

        self.assertEqual(signal.leader_score, 0.0)
        self.assertEqual(signal.effective_leader_score, 0.0)
        self.assertEqual(signal.leader_percentile, 0.0)
        self.assertEqual(signal.entry_grade, "C")
        self.assertEqual(signal.change_rate, quote.change_rate)

    def test_refresh_runtime_math_candidate_queue_prunes_stale_intraday_sources(self):
        cfg = MomentumScalpConfig()
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        now = datetime(2026, 4, 3, 10, 5, 0)

        stale = "100001"
        hot = "100002"
        fresh = Quote(
            symbol="100003",
            name="100003",
            current_price=10_400,
            change=400,
            change_rate=4.0,
            open_price=10_000,
            high_price=10_450,
            low_price=9_980,
            volume=150_000,
            trade_amount=1_560_000_000,
            timestamp=now,
        )
        strategy._latest_math_queue_symbols = [hot, stale]
        strategy._latest_math_backfill_symbols = [stale]
        strategy._latest_math_queue_source = {
            hot: "opening_hot_queue",
            stale: "math_backfill",
        }
        strategy._recent_quotes[fresh.symbol] = deque([fresh], maxlen=8)

        strategy._refresh_runtime_math_candidate_queue([fresh])

        self.assertNotIn(stale, strategy._latest_math_queue_source)
        self.assertNotIn(hot, strategy._latest_math_queue_source)
        self.assertIn(fresh.symbol, strategy._latest_math_queue_source)

    def test_refresh_runtime_math_candidate_queue_excludes_unsupported_long_symbols(self):
        cfg = MomentumScalpConfig()
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        now = datetime(2026, 6, 11, 10, 10, 28)
        supported = Quote(
            symbol="100004",
            name="100004",
            current_price=10_400,
            change=400,
            change_rate=4.0,
            open_price=10_000,
            high_price=10_450,
            low_price=9_980,
            volume=150_000,
            trade_amount=1_560_000_000,
            timestamp=now,
        )
        derivative_like = Quote(
            symbol="0195S0",
            name="0195S0",
            current_price=20_130,
            change=2_750,
            change_rate=15.82,
            open_price=17_380,
            high_price=20_350,
            low_price=17_300,
            volume=300_000,
            trade_amount=6_039_000_000,
            timestamp=now,
        )
        vendor_like = Quote(
            symbol="Q520101",
            name="Q520101",
            current_price=9_800,
            change=900,
            change_rate=10.11,
            open_price=8_900,
            high_price=9_820,
            low_price=8_880,
            volume=250_000,
            trade_amount=2_450_000_000,
            timestamp=now,
        )

        strategy._refresh_runtime_math_candidate_queue([derivative_like, vendor_like, supported])

        queued_symbols = set(strategy._latest_math_queue_symbols) | set(strategy._latest_math_backfill_symbols)
        self.assertIn("100004", queued_symbols)
        self.assertNotIn("0195S0", queued_symbols)
        self.assertNotIn("Q520101", queued_symbols)
        self.assertNotIn("0195S0", strategy._latest_math_queue_source)
        self.assertNotIn("Q520101", strategy._latest_math_queue_source)

    def test_adaptive_market_thresholds_relax_hot_tape_and_tighten_caution_tape(self):
        strategy = MomentumScalpStrategy(market_data=None, config=MomentumScalpConfig())
        strategy._adaptive_market_state = {
            "quote_count": 26.0,
            "tape_heat": 0.86,
            "tape_caution": 0.10,
            "overheat": 0.35,
            "vs_open_p90": 17.8,
        }

        hot = strategy._adaptive_market_entry_thresholds()
        hot_queue_floor = strategy._adaptive_math_queue_percentile_floor(0.80)

        self.assertLess(hot["leader_percentile_delta"], 0.0)
        self.assertLess(hot["effective_score_delta"], 0.0)
        self.assertGreater(hot["vs_open_ceiling_delta"], 0.0)
        self.assertGreater(hot["negative_ev_floor_scale"], 1.0)
        self.assertLess(hot_queue_floor, 0.80)

        strategy._adaptive_market_state = {
            "quote_count": 26.0,
            "tape_heat": 0.12,
            "tape_caution": 0.82,
            "overheat": 0.78,
            "vs_open_p90": 6.0,
        }

        caution = strategy._adaptive_market_entry_thresholds()
        caution_queue_floor = strategy._adaptive_math_queue_percentile_floor(0.80)

        self.assertGreater(caution["leader_percentile_delta"], 0.0)
        self.assertGreater(caution["effective_score_delta"], 0.0)
        self.assertLess(caution["vs_open_ceiling_delta"], 0.0)
        self.assertLess(caution["negative_ev_floor_scale"], 1.0)
        self.assertGreater(caution_queue_floor, 0.80)

    def test_symbol_micro_edge_scores_fast_launch_higher_than_reversal(self):
        strategy = MomentumScalpStrategy(market_data=None, config=MomentumScalpConfig())
        now = datetime(2026, 5, 13, 10, 30, 0)
        strategy.set_simulated_now(now)
        launch_quotes = [
            Quote("AAA", "AAA", 10_000, 0, 0.0, 10_000, 10_520, 9_980, 100_000, 1_000_000_000, now),
            Quote("AAA", "AAA", 10_060, 60, 0.6, 10_000, 10_520, 9_980, 130_000, 1_301_800_000, now),
            Quote("AAA", "AAA", 10_150, 150, 1.5, 10_000, 10_520, 9_980, 180_000, 1_809_500_000, now),
            Quote("AAA", "AAA", 10_300, 300, 3.0, 10_000, 10_520, 9_980, 260_000, 2_633_500_000, now),
            Quote("AAA", "AAA", 10_480, 480, 4.8, 10_000, 10_520, 9_980, 380_000, 3_891_100_000, now),
        ]
        strategy._recent_quotes["AAA"] = deque(launch_quotes, maxlen=8)
        launch_leader = LeaderSignal(
            symbol="AAA",
            leader_score=1.0,
            leader_percentile=0.98,
            entry_grade="A",
            change_rate=4.8,
            trade_amount=3_891_100_000,
            vs_open_pct=4.8,
            high_proximity=0.93,
            volume_vs_avg=1.4,
            reclaim_speed_ticks=1,
            recent_acceleration_pct=0.5,
            effective_leader_score=1.1,
        )

        launch = symbol_micro_edge_metrics(strategy, launch_quotes[-1], leader=launch_leader)

        reversal_quotes = [
            Quote("BBB", "BBB", 10_000, 0, 0.0, 10_000, 10_900, 9_980, 100_000, 1_000_000_000, now),
            Quote("BBB", "BBB", 10_850, 850, 8.5, 10_000, 10_900, 9_980, 170_000, 1_759_500_000, now),
            Quote("BBB", "BBB", 10_780, 780, 7.8, 10_000, 10_900, 9_980, 220_000, 2_298_500_000, now),
            Quote("BBB", "BBB", 10_560, 560, 5.6, 10_000, 10_900, 9_980, 245_000, 2_562_500_000, now),
            Quote("BBB", "BBB", 10_360, 360, 3.6, 10_000, 10_900, 9_980, 255_000, 2_666_100_000, now),
        ]
        strategy._recent_quotes["BBB"] = deque(reversal_quotes, maxlen=8)
        reversal_leader = LeaderSignal(
            symbol="BBB",
            leader_score=1.0,
            leader_percentile=0.98,
            entry_grade="A",
            change_rate=3.6,
            trade_amount=2_666_100_000,
            vs_open_pct=11.0,
            high_proximity=0.99,
            volume_vs_avg=1.1,
            reclaim_speed_ticks=99,
            recent_acceleration_pct=-0.4,
            effective_leader_score=1.0,
        )

        reversal = symbol_micro_edge_metrics(strategy, reversal_quotes[-1], leader=reversal_leader)

        self.assertTrue(launch["micro_ready"])
        self.assertGreater(launch["micro_launch_score"], launch["micro_downside_score"])
        self.assertGreater(launch["micro_net_score"], 0.0)
        self.assertTrue(bool(launch["micro_fast_launch"]))
        self.assertGreater(reversal["micro_downside_score"], reversal["micro_launch_score"])
        self.assertLess(reversal["micro_net_score"], 0.0)




    def test_protective_stop_exits_immediately(self):
        cfg = MomentumScalpConfig(
            commission_rate=0.0,
            tax_slippage_rate=0.0,
            long_stop_loss_cap_amount=2_200,
            long_stop_loss_notional_pct=0.007,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        now = datetime(2026, 5, 13, 10, 35, 0)
        strategy.set_simulated_now(now)
        strategy._session_start_at = datetime(2026, 5, 13, 9, 0, 0)
        strategy.positions["AAA"] = PositionState(
            symbol="AAA",
            buy_price=10_000,
            quantity=30,
            invested_amount=300_000,
            buy_time=now - timedelta(minutes=3),
            high_since_buy=10_080,
        )
        rebound_quotes = [
            Quote("AAA", "AAA", 10_000, 0, 0.0, 10_000, 10_080, 9_850, 100_000, 1_000_000_000, now),
            Quote("AAA", "AAA", 9_900, -100, -1.0, 10_000, 10_080, 9_850, 150_000, 1_495_000_000, now),
            Quote("AAA", "AAA", 9_860, -140, -1.4, 10_000, 10_080, 9_850, 210_000, 2_086_600_000, now),
            Quote("AAA", "AAA", 9_895, -105, -1.05, 10_000, 10_080, 9_850, 300_000, 2_977_150_000, now),
            Quote("AAA", "AAA", 9_930, -70, -0.7, 10_000, 10_080, 9_850, 430_000, 4_268_050_000, now),
        ]
        strategy._recent_quotes["AAA"] = deque(rebound_quotes, maxlen=8)
        strategy._latest_math_leader_signals["AAA"] = LeaderSignal(
            symbol="AAA",
            leader_score=0.8,
            leader_percentile=0.90,
            entry_grade="A",
            change_rate=-0.7,
            trade_amount=4_268_050_000,
            vs_open_pct=-0.7,
            high_proximity=0.65,
            volume_vs_avg=1.4,
            reclaim_speed_ticks=2,
            recent_acceleration_pct=0.20,
            effective_leader_score=0.8,
        )

        order = strategy._default_long_exit(rebound_quotes[-1])

        self.assertIsNotNone(order)
        self.assertEqual(order.requested_reason, "protective_stop_net")




    def test_update_market_state_populates_adaptive_market_snapshot(self):
        cfg = MomentumScalpConfig(
            enable_expected_net_filter=False,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)
        now = datetime(2026, 5, 13, 10, 5, 0)
        strategy.set_simulated_now(now)
        strategy._session_start_at = datetime(2026, 5, 13, 9, 0, 0)
        quotes = []
        for idx in range(10):
            symbol = f"100{idx:03d}"
            open_price = 10_000 + idx * 100
            current_price = int(open_price * (1.02 + idx * 0.002))
            quote = Quote(
                symbol=symbol,
                name=symbol,
                current_price=current_price,
                change=current_price - open_price,
                change_rate=((current_price - open_price) / open_price) * 100,
                open_price=open_price,
                high_price=current_price + 40,
                low_price=open_price - 40,
                volume=120_000 + idx * 10_000,
                trade_amount=current_price * (120_000 + idx * 10_000),
                timestamp=now,
            )
            strategy._quotes_cache[symbol] = quote
            strategy._recent_quotes[symbol] = deque([quote], maxlen=8)
            quotes.append(quote)

        strategy._update_market_state(quotes)

        self.assertEqual(strategy._adaptive_market_state["quote_count"], 10.0)
        self.assertGreater(strategy._adaptive_market_state["avg_change"], 2.0)
        self.assertLess(strategy._adaptive_market_state["decliner_ratio"], 0.01)
        self.assertGreater(strategy._adaptive_market_state["tape_heat"], strategy._adaptive_market_state["tape_caution"])

    def test_update_market_state_uses_fresh_cached_quotes_beyond_current_batch(self):
        cfg = MomentumScalpConfig(
            enable_expected_net_filter=False,
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg, pool_override=["100001", "100002"])
        now = datetime(2026, 4, 3, 10, 5, 0)
        strategy.set_simulated_now(now)
        strategy._session_start_at = datetime(2026, 4, 3, 9, 0, 0)
        quote_a = Quote(
            symbol="100001",
            name="100001",
            current_price=10_100,
            change=100,
            change_rate=1.0,
            open_price=10_000,
            high_price=10_120,
            low_price=9_980,
            volume=120_000,
            trade_amount=1_212_000_000,
            timestamp=now,
        )
        quote_b = Quote(
            symbol="100002",
            name="100002",
            current_price=10_400,
            change=400,
            change_rate=4.0,
            open_price=10_000,
            high_price=10_450,
            low_price=9_980,
            volume=150_000,
            trade_amount=1_560_000_000,
            timestamp=now,
        )
        strategy._quotes_cache["100001"] = quote_a
        strategy._quotes_cache["100002"] = quote_b
        strategy._recent_quotes["100001"] = deque([quote_a], maxlen=8)
        strategy._recent_quotes["100002"] = deque([quote_b], maxlen=8)

        strategy._update_market_state([quote_a])

        self.assertIn("100001", strategy._latest_math_leader_signals)
        self.assertIn("100002", strategy._latest_math_leader_signals)

    def test_update_runtime_pool_changes_watchlist_front(self):
        cfg = MomentumScalpConfig(
            dynamic_pool_size=4,
            inverse_etfs=["252670"],
        )
        strategy = MomentumScalpStrategy(market_data=None, config=cfg)

        strategy.update_runtime_pool(["009150", "000660", "005930", "035420"])
        watchlist = strategy.get_watchlist()

        self.assertEqual(watchlist[:4], ["009150", "000660", "005930", "035420"])
        self.assertIn("252670", watchlist)



if __name__ == "__main__":
    unittest.main()
