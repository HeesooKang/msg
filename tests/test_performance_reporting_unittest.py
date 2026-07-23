import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from src.performance_reporting import (
    build_daily_scorecard,
    evaluate_math_shadow_report,
    evaluate_strategy_gates,
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

    def test_build_daily_scorecard_includes_log_analysis(self):
        strategy = SimpleNamespace(
            daily_pnl=SimpleNamespace(
                realized_gross_pnl=1_500,
                realized_net_pnl=500,
                fees_paid=50,
                taxes_paid=250,
                trade_count=1,
                win_count=1,
                loss_count=0,
                breakeven_count=0,
                winning_net_pnl_sum=500,
                losing_net_pnl_sum=0,
                largest_win_net=500,
                largest_loss_net=0,
            ),
            positions={},
            _halted=False,
            _real_money_stage=0,
            _capital_scale=1.0,
        )
        balance = SimpleNamespace(
            total_eval_amount=9_800_000,
            total_deposit=900_000,
            total_profit_loss=1_200,
            total_profit_rate=0.12,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            log_root = Path(tmp_dir)
            (log_root / "trading.log").write_text(
                "\n".join(
                    [
                        "2026-03-11 10:01:00 [INFO] kis_trader.strategy.momentum: "
                        "후보 평가 요약: stage=queue_entered window_start=10:00:00 window_end=10:01:00 "
                        "total=1 math_queue=1",
                        "2026-03-11 10:01:01 [INFO] kis_trader.strategy.momentum: "
                        "후보 평가 요약: stage=precheck_reached window_start=10:00:00 window_end=10:01:01 "
                        "total=1 math_queue=1",
                        "2026-03-11 10:01:02 [INFO] kis_trader.strategy.momentum: "
                        "후보 평가 요약: stage=router_reached window_start=10:00:00 window_end=10:01:02 "
                        "total=1 math_queue=1",
                        "2026-03-11 10:01:03 [INFO] kis_trader.strategy.momentum: "
                        "후보 평가 요약: stage=admission_blocked window_start=10:00:00 window_end=10:01:03 total=1",
                        "2026-03-11 10:01:05 [INFO] kis_trader.strategy.momentum: "
                        "EV 매수 신호: 011700 10주 @ 10000원 budget=100,000 "
                        "route=opening_conviction_long_strategy source=math_queue rank=1 score=0.9000 "
                        "pred=1.200 lower=0.200 conf=0.700 win=0.650 exp=1300.0 "
                        "pnet=1500 lnet=100 target=1200 stop=900 risk=1000 model=ok",
                        "2026-03-11 10:02:00 [INFO] kis_trader.strategy.momentum: "
                        "진입 거부[conviction_rank_overflow_reject] reject_reason=conviction_rank_overflow_reject: 011700 (conviction_rank_overflow_reject)",
                        "2026-03-11 10:30:00 [INFO] kis_trader.strategy.momentum: "
                        "매도 체결: 011700 10주 @ 10,150원 (총손익: 1,500원, 순손익: 1,200원, 누적순손익: 1,200원, "
                        "strategy_name=opening_conviction_long_strategy, setup_name=expected_value, regime_label=bull)",
                    ]
                ),
                encoding="utf-8",
            )

            scorecard = build_daily_scorecard(
                strategy=strategy,
                balance=balance,
                session_pnl=500,
                trading_mode="paper",
                generated_at=datetime(2026, 3, 11, 15, 20, 0),
                log_root=log_root,
            )

        self.assertEqual(scorecard["log_analysis"]["entries"]["by_setup"]["expected_value"], 1)
        self.assertEqual(scorecard["log_analysis"]["entries"]["by_strategy"]["opening_conviction_long_strategy"], 1)
        self.assertEqual(scorecard["log_analysis"]["rejections"]["by_reason"]["conviction_rank_overflow_reject"], 1)
        self.assertEqual(scorecard["log_analysis"]["strategy_pnl"]["opening_conviction_long_strategy"]["net_pnl"], 1_200)
        self.assertEqual(scorecard["log_analysis"]["setup_pnl"]["expected_value"]["net_pnl"], 1_200)
        self.assertEqual(scorecard["log_analysis"]["trade_records"][0]["entry_ev"], 1300.0)
        self.assertEqual(
            scorecard["log_analysis"]["trade_records"][0]["price_prediction_return_pct"],
            1.2,
        )
        self.assertEqual(scorecard["log_analysis"]["symbols"]["top_winners"][0]["symbol"], "011700")
        self.assertEqual(scorecard["log_analysis"]["math_shadow"]["queue_counts"]["math_queue"], 1)
        self.assertEqual(scorecard["log_analysis"]["math_shadow"]["router_reached_counts"]["math_queue"], 1)
        self.assertEqual(scorecard["log_analysis"]["math_shadow"]["admission_counts"]["blocked"], 1)
        self.assertEqual(scorecard["pnl"]["realized_net_pnl"], 1_200)
        self.assertEqual(scorecard["pnl"]["realized_source"], "sell_fill_log")
        self.assertEqual(
            scorecard["pnl"]["winning_net_pnl_sum"]
            + scorecard["pnl"]["losing_net_pnl_sum"],
            scorecard["pnl"]["realized_net_pnl"],
        )
        self.assertEqual(scorecard["reconciliation"]["net_pnl_difference"], 700)

    def test_build_daily_scorecard_preserves_entry_metadata_across_partial_sell(self):
        strategy = SimpleNamespace(
            daily_pnl=SimpleNamespace(
                realized_gross_pnl=2_400,
                realized_net_pnl=2_100,
                fees_paid=100,
                taxes_paid=200,
                trade_count=2,
                win_count=2,
                loss_count=0,
                breakeven_count=0,
                winning_net_pnl_sum=2_100,
                losing_net_pnl_sum=0,
                largest_win_net=1_400,
                largest_loss_net=0,
            ),
            positions={},
            _halted=False,
            _real_money_stage=0,
            _capital_scale=1.0,
        )
        balance = SimpleNamespace(
            total_eval_amount=9_900_000,
            total_deposit=1_000_000,
            total_profit_loss=2_100,
            total_profit_rate=0.21,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            log_root = Path(tmp_dir)
            (log_root / "trading.log").write_text(
                "\n".join(
                    [
                        "2026-03-24 12:19:00 [INFO] kis_trader.strategy.momentum: "
                        "매수 신호: 테스트(072950) 점수=3.1, 20주 @ 10,000원 "
                        "(할당 200,000원, strategy_name=opening_conviction_long_strategy setup_name=opening_conviction "
                        "entry_reason=opening_conviction regime_label=bull entry_grade_signal=A "
                        "leader_score=2.4000 effective_leader_score=2.7000 leader_pct=0.9800 "
                        "entry_ev=850.00 entry_ev_conf=medium entry_ev_trades=5 candidate_class=priority queue_source=math_queue)",
                        "2026-03-24 12:25:00 [INFO] kis_trader.strategy.momentum: "
                        "부분매도 체결: 072950 10주 @ 10,120원 (총손익: 1,400원, 순손익: 1,100원, 누적순손익: 1,100원, "
                        "strategy_name=opening_conviction_long_strategy, setup_name=opening_conviction, regime_label=bull)",
                        "2026-03-24 12:38:00 [INFO] kis_trader.strategy.momentum: "
                        "매도 체결: 072950 10주 @ 10,140원 (총손익: 2,700원, 순손익: 2,100원, 누적순손익: 2,100원, "
                        "strategy_name=opening_conviction_long_strategy, setup_name=opening_conviction, regime_label=bull)",
                    ]
                ),
                encoding="utf-8",
            )

            scorecard = build_daily_scorecard(
                strategy=strategy,
                balance=balance,
                session_pnl=2_100,
                trading_mode="paper",
                generated_at=datetime(2026, 3, 24, 15, 20, 0),
                log_root=log_root,
            )

        trade_records = scorecard["log_analysis"]["trade_records"]
        self.assertEqual(len(trade_records), 2)
        self.assertEqual(trade_records[0]["symbol"], "072950")
        self.assertEqual(trade_records[1]["symbol"], "072950")
        self.assertAlmostEqual(trade_records[0]["leader_score"], 2.4, places=4)
        self.assertAlmostEqual(trade_records[1]["leader_score"], 2.4, places=4)
        self.assertAlmostEqual(trade_records[1]["effective_leader_score"], 2.7, places=4)
        self.assertEqual(trade_records[1]["candidate_class"], "priority")
        self.assertEqual(trade_records[1]["queue_source"], "math_queue")
        self.assertEqual(
            scorecard["log_analysis"]["strategy_pnl"]["opening_conviction_long_strategy"]["closed_trades"],
            1,
        )
        self.assertEqual(
            scorecard["log_analysis"]["setup_pnl"]["opening_conviction"]["closed_trades"],
            1,
        )
        self.assertEqual(
            scorecard["log_analysis"]["strategy_pnl"]["opening_conviction_long_strategy"]["net_pnl"],
            2_100,
        )

    def test_build_daily_scorecard_treats_legacy_partial_sell_log_as_open_trade(self):
        strategy = SimpleNamespace(
            daily_pnl=SimpleNamespace(
                realized_gross_pnl=2_700,
                realized_net_pnl=2_700,
                fees_paid=0,
                taxes_paid=0,
                trade_count=1,
                win_count=1,
                loss_count=0,
                breakeven_count=0,
                winning_net_pnl_sum=2_700,
                losing_net_pnl_sum=0,
                largest_win_net=2_700,
                largest_loss_net=0,
            ),
            positions={},
            _halted=False,
            _real_money_stage=0,
            _capital_scale=1.0,
        )
        balance = SimpleNamespace(
            total_eval_amount=9_900_000,
            total_deposit=1_000_000,
            total_profit_loss=2_700,
            total_profit_rate=0.27,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            log_root = Path(tmp_dir)
            (log_root / "trading.log").write_text(
                "\n".join(
                    [
                        "2026-03-24 12:19:00 [INFO] kis_trader.strategy.momentum: "
                        "매수 신호: 테스트(072950) 점수=3.1, 20주 @ 10,000원 "
                        "(할당 200,000원, strategy_name=opening_conviction_long_strategy setup_name=opening_conviction "
                        "entry_reason=opening_conviction regime_label=bull candidate_class=priority queue_source=math_queue)",
                        "2026-03-24 12:25:00 [INFO] kis_trader.strategy.momentum: "
                        "매도 체결: 072950 10주 @ 10,120원 (총손익: 1,200원, 순손익: 1,200원, 누적순손익: 1,200원, "
                        "order_no=30001, fill_mode=filled, strategy_name=opening_conviction_long_strategy, "
                        "setup_name=opening_conviction, regime_label=bull, exit_reason=partial_take_profit)",
                        "2026-03-24 12:38:00 [INFO] kis_trader.strategy.momentum: "
                        "매도 체결: 072950 10주 @ 10,150원 (총손익: 1,500원, 순손익: 1,500원, 누적순손익: 2,700원, "
                        "order_no=30002, fill_mode=filled, strategy_name=opening_conviction_long_strategy, "
                        "setup_name=opening_conviction, regime_label=bull, exit_reason=take_profit)",
                    ]
                ),
                encoding="utf-8",
            )

            scorecard = build_daily_scorecard(
                strategy=strategy,
                balance=balance,
                session_pnl=2_700,
                trading_mode="paper",
                generated_at=datetime(2026, 3, 24, 15, 20, 0),
                log_root=log_root,
            )

        self.assertEqual(len(scorecard["log_analysis"]["trade_records"]), 2)
        self.assertEqual(
            scorecard["log_analysis"]["strategy_pnl"]["opening_conviction_long_strategy"]["closed_trades"],
            1,
        )
        self.assertEqual(
            scorecard["log_analysis"]["strategy_pnl"]["opening_conviction_long_strategy"]["net_pnl"],
            2_700,
        )

    def test_build_daily_scorecard_applies_sell_correction_log_delta(self):
        strategy = SimpleNamespace(
            daily_pnl=SimpleNamespace(
                realized_gross_pnl=16_100,
                realized_net_pnl=16_100,
                fees_paid=0,
                taxes_paid=0,
                trade_count=1,
                win_count=1,
                loss_count=0,
                breakeven_count=0,
                winning_net_pnl_sum=16_100,
                losing_net_pnl_sum=0,
                largest_win_net=16_100,
                largest_loss_net=0,
            ),
            positions={},
            _halted=False,
            _real_money_stage=0,
            _capital_scale=1.0,
        )
        balance = SimpleNamespace(
            total_eval_amount=9_900_000,
            total_deposit=1_000_000,
            total_profit_loss=16_100,
            total_profit_rate=1.61,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            log_root = Path(tmp_dir)
            (log_root / "trading.log").write_text(
                "\n".join(
                    [
                        "2026-04-27 09:00:43 [INFO] kis_trader.strategy.momentum: "
                        "매수 신호: 테스트(006340) 점수=3.1, 35주 @ 9,760원 "
                        "(할당 341,600원, strategy_name=intraday_conviction_long_strategy setup_name=queue_value_scout "
                        "entry_reason=intraday_conviction regime_label=bull candidate_class=priority queue_source=math_queue)",
                        "2026-04-27 09:01:21 [INFO] kis_trader.strategy.momentum: "
                        "매도 체결: 006340 35주 @ 10,110원 (총손익: 12,250원, 순손익: 12,250원, 누적순손익: 12,250원, "
                        "order_no=20001, fill_mode=account_reconciled_estimated, strategy_name=intraday_conviction_long_strategy, "
                        "setup_name=queue_value_scout, regime_label=bull, exit_reason=take_profit)",
                        "2026-04-27 09:01:24 [INFO] kis_trader.scheduler: "
                        "계좌 재동기화 매도 체결가 확정: 006340 order_no=20001 35주 @ 10220원 (기존 10110원, 순손익 +3850원 보정)",
                        "2026-04-27 09:01:24 [INFO] kis_trader.strategy.momentum: "
                        "매도 체결 정정: 006340 order_no=20001 previous_price=10110 corrected_price=10220 "
                        "previous_net_pnl=12250 corrected_net_pnl=16100 delta_net_pnl=3850",
                    ]
                ),
                encoding="utf-8",
            )

            scorecard = build_daily_scorecard(
                strategy=strategy,
                balance=balance,
                session_pnl=16_100,
                trading_mode="paper",
                generated_at=datetime(2026, 4, 27, 15, 20, 0),
                log_root=log_root,
            )

        self.assertEqual(scorecard["log_analysis"]["trade_records"][0]["net_pnl"], 16_100)
        self.assertEqual(scorecard["log_analysis"]["trade_records"][0]["fill_mode"], "account_reconciled_confirmed")
        self.assertEqual(
            scorecard["log_analysis"]["strategy_pnl"]["intraday_conviction_long_strategy"]["net_pnl"],
            16_100,
        )

    def test_build_daily_scorecard_tracks_conviction_tier_opening_queue_and_stop_stats(self):
        strategy = SimpleNamespace(
            daily_pnl=SimpleNamespace(
                realized_gross_pnl=-6_000,
                realized_net_pnl=-6_470,
                fees_paid=120,
                taxes_paid=350,
                trade_count=1,
                win_count=0,
                loss_count=1,
                breakeven_count=0,
                winning_net_pnl_sum=0,
                losing_net_pnl_sum=-6_470,
                largest_win_net=0,
                largest_loss_net=-6_470,
            ),
            positions={},
            _halted=False,
            _real_money_stage=0,
            _capital_scale=1.0,
        )
        balance = SimpleNamespace(
            total_eval_amount=9_900_000,
            total_deposit=1_000_000,
            total_profit_loss=-6_470,
            total_profit_rate=-0.64,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            log_root = Path(tmp_dir)
            (log_root / "trading.log").write_text(
                "\n".join(
                    [
                        "2026-03-25 09:00:15 [INFO] kis_trader.strategy.momentum: 장초 초고속 후보 큐: 2개 [263750,010170]",
                        "2026-03-25 09:00:30 [INFO] kis_trader.strategy.momentum: "
                        "후보 평가 요약: stage=queue_entered window_start=09:00:00 window_end=09:00:30 "
                        "total=5 opening_fast_queue=2 opening_hot_queue=3",
                        "2026-03-25 09:00:31 [INFO] kis_trader.strategy.momentum: "
                        "후보 평가 요약: stage=precheck_reached window_start=09:00:00 window_end=09:00:31 "
                        "total=5 opening_fast_queue=2 opening_hot_queue=3",
                        "2026-03-25 09:05:44 [INFO] kis_trader.strategy.momentum: "
                        "매수 신호: 테스트(010170) 점수=2.5, 100주 @ 8,840원 "
                        "(할당 884,000원, strategy_name=opening_conviction_long_strategy setup_name=opening_conviction "
                        "entry_reason=opening_conviction regime_label=bull conviction_tier=low_confidence "
                        "bull_risk_mode=guarded post_loss_admission_class=general candidate_class=opening_hot "
                        "entry_grade_signal=A leader_score=1.5000 effective_leader_score=1.8000 leader_pct=1.0000 "
                        "entry_ev=0.00 entry_ev_conf=none entry_ev_trades=0 queue_source=opening_hot_queue "
                        "volume_gate_threshold_used=0.72 size_multiplier=1.0000)",
                        "2026-03-25 09:06:19 [INFO] kis_trader.strategy.momentum: "
                        "매도 체결: 010170 100주 @ 8,780원 (총손익: -6,000원, 순손익: -6,470원, 누적순손익: -6,470원, "
                        "strategy_name=opening_conviction_long_strategy, setup_name=opening_conviction, regime_label=bull, entry_grade=A, "
                        "entry_grade_math=A, leader_score=1.5000, leader_pct=1.0000, entry_ev=0.00, entry_ev_conf=none, "
                        "conviction_tier=low_confidence, bull_risk_mode=guarded, post_loss_admission_class=general, candidate_class=opening_hot, queue_source=opening_hot_queue, size_multiplier=1.0000, "
                        "volume_gate_threshold_used=0.72, "
                        "protective_exit_mode=limit_then_market, fill_mode=limit_then_market, stop_overshoot_krw=4898)",
                        "2026-03-25 09:08:00 [INFO] kis_trader.strategy.momentum: "
                        "그림자 후보 종료: 014970 shadow_reason=general_bull_shadow regime_label=bull strategy_name=opening_conviction_long_strategy "
                        "entry=8,840원 last=8,960원 max=9,020원 min=8,780원 MFE=2.04% MAE=-0.68% close=1.36% outcome=take_profit_first",
                    ]
                ),
                encoding="utf-8",
            )

            scorecard = build_daily_scorecard(
                strategy=strategy,
                balance=balance,
                session_pnl=-6_470,
                trading_mode="paper",
                generated_at=datetime(2026, 3, 25, 15, 20, 0),
                log_root=log_root,
            )

        math_shadow = scorecard["log_analysis"]["math_shadow"]
        self.assertEqual(math_shadow["queue_counts"]["opening_fast_queue"], 2)
        self.assertEqual(math_shadow["queue_counts"]["opening_hot_queue"], 3)
        self.assertEqual(math_shadow["conviction_tier_pnl"]["low_confidence"]["closed_trades"], 1)
        self.assertEqual(math_shadow["bull_risk_mode_pnl"]["guarded"]["closed_trades"], 1)
        self.assertEqual(math_shadow["post_loss_admission_class_pnl"]["general"]["closed_trades"], 1)
        self.assertEqual(math_shadow["candidate_class_pnl"]["opening_hot"]["closed_trades"], 1)
        self.assertEqual(math_shadow["queue_source_pnl"]["opening_hot_queue"]["net_pnl"], -6470)
        self.assertEqual(math_shadow["candidate_reject_reason_outcomes"]["general_bull_shadow"]["take_profit_first"], 1)
        self.assertEqual(math_shadow["stop_stats"]["stop_triggered_count"], 1)
        self.assertEqual(math_shadow["stop_stats"]["stop_limit_fallback_count"], 1)
        self.assertEqual(math_shadow["stop_stats"]["stop_overshoot_max_krw"], 4898)

    def test_build_daily_scorecard_parses_candidate_funnel_summary_logs(self):
        strategy = SimpleNamespace(
            daily_pnl=SimpleNamespace(
                realized_gross_pnl=0,
                realized_net_pnl=0,
                fees_paid=0,
                taxes_paid=0,
                trade_count=0,
                win_count=0,
                loss_count=0,
                breakeven_count=0,
                winning_net_pnl_sum=0,
                losing_net_pnl_sum=0,
                largest_win_net=0,
                largest_loss_net=0,
            ),
            positions={},
            _halted=False,
            _real_money_stage=0,
            _capital_scale=1.0,
        )
        balance = SimpleNamespace(
            total_eval_amount=9_800_000,
            total_deposit=900_000,
            total_profit_loss=0,
            total_profit_rate=0.0,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            log_root = Path(tmp_dir)
            (log_root / "trading.log").write_text(
                "\n".join(
                    [
                        "2026-03-24 09:05:00 [INFO] kis_trader.strategy.momentum: "
                        "후보 평가 요약: stage=queue_entered window_start=09:00:00 window_end=09:05:00 "
                        "total=20 math_queue=14 math_backfill=6",
                        "2026-03-24 09:05:00 [INFO] kis_trader.strategy.momentum: "
                        "후보 평가 요약: stage=precheck_reached window_start=09:00:00 window_end=09:05:00 "
                        "total=18 math_queue=12 math_backfill=6",
                        "2026-03-24 09:05:00 [INFO] kis_trader.strategy.momentum: "
                        "후보 평가 요약: stage=policy_blocked window_start=09:00:00 window_end=09:05:00 "
                        "total=5 legacy_backfill=5",
                        "2026-03-24 09:05:00 [INFO] kis_trader.strategy.momentum: "
                        "후보 평가 요약: stage=router_reached window_start=09:00:00 window_end=09:05:00 "
                        "total=10 math_queue=8 math_backfill=2",
                        "2026-03-24 09:05:00 [INFO] kis_trader.strategy.momentum: "
                        "후보 평가 요약: stage=router_blocked window_start=09:00:00 window_end=09:05:00 "
                        "total=10 math_queue=8 math_backfill=2 reason_bull_only=10",
                        "2026-03-24 09:05:00 [INFO] kis_trader.strategy.momentum: "
                        "후보 평가 요약: stage=admission_blocked window_start=09:00:00 window_end=09:05:00 total=3",
                    ]
                ),
                encoding="utf-8",
            )

            scorecard = build_daily_scorecard(
                strategy=strategy,
                balance=balance,
                session_pnl=0,
                trading_mode="paper",
                generated_at=datetime(2026, 3, 24, 15, 20, 0),
                log_root=log_root,
            )

        self.assertEqual(scorecard["log_analysis"]["math_shadow"]["queue_counts"]["math_queue"], 14)
        self.assertEqual(scorecard["log_analysis"]["math_shadow"]["queue_counts"]["math_backfill"], 6)
        self.assertEqual(scorecard["log_analysis"]["math_shadow"]["policy_blocked_counts"]["legacy_backfill"], 5)
        self.assertEqual(scorecard["log_analysis"]["math_shadow"]["router_reached_counts"]["math_queue"], 8)
        self.assertEqual(scorecard["log_analysis"]["math_shadow"]["admission_counts"]["blocked"], 3)

    def test_build_daily_scorecard_parses_setup_reject_summary_logs(self):
        strategy = SimpleNamespace(
            daily_pnl=SimpleNamespace(
                realized_gross_pnl=0,
                realized_net_pnl=0,
                fees_paid=0,
                taxes_paid=0,
                trade_count=0,
                win_count=0,
                loss_count=0,
                breakeven_count=0,
                winning_net_pnl_sum=0,
                losing_net_pnl_sum=0,
                largest_win_net=0,
                largest_loss_net=0,
            ),
            positions={},
            _halted=False,
            _real_money_stage=0,
            _capital_scale=1.0,
        )
        balance = SimpleNamespace(
            total_eval_amount=9_800_000,
            total_deposit=900_000,
            total_profit_loss=0,
            total_profit_rate=0.0,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            log_root = Path(tmp_dir)
            (log_root / "trading.log").write_text(
                "\n".join(
                    [
                        "2026-03-24 10:05:00 [INFO] kis_trader.strategy.momentum: "
                        "진입 거부 요약: reject_reason=bull_score window_start=10:00:00 window_end=10:05:00 "
                        "total=42 opening_conviction_long_strategy=40 intraday_conviction_long_strategy=2",
                    ]
                ),
                encoding="utf-8",
            )

            scorecard = build_daily_scorecard(
                strategy=strategy,
                balance=balance,
                session_pnl=0,
                trading_mode="paper",
                generated_at=datetime(2026, 3, 24, 15, 20, 0),
                log_root=log_root,
            )

        self.assertEqual(scorecard["log_analysis"]["rejections"]["total"], 42)
        self.assertEqual(scorecard["log_analysis"]["rejections"]["by_reason"]["bull_score"], 42)

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
            cfg=SimpleNamespace(
                strategy_gate_window_days=5,
                strategy_gate_min_closed_trades=4,
            ),
        )
        balance = SimpleNamespace(
            total_eval_amount=9_800_000,
            total_deposit=900_000,
            total_profit_loss=900,
            total_profit_rate=0.09,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            log_root = Path(tmp_dir) / "logs"
            log_root.mkdir()
            (log_root / "trading.log").write_text("", encoding="utf-8")
            paths = update_performance_reports(
                strategy=strategy,
                balance=balance,
                session_pnl=900,
                trading_mode="paper",
                report_root=Path(tmp_dir),
                log_root=log_root,
            )

            self.assertTrue(paths["scorecard"]["json"].exists())
            self.assertTrue(paths["scorecard"]["md"].exists())
            self.assertTrue(paths["strategy_gates"]["json"].exists())
            self.assertTrue(paths["strategy_gates"]["md"].exists())
            self.assertTrue(paths["math_shadow"]["json"].exists())
            self.assertTrue(paths["math_shadow"]["md"].exists())
            self.assertTrue(paths["readiness"]["json"].exists())
            self.assertTrue(paths["readiness"]["md"].exists())

            payload = json.loads(paths["scorecard"]["json"].read_text(encoding="utf-8"))
            self.assertEqual(payload["pnl"]["session_pnl"], 900)
            self.assertIn("paper_gate", payload)

    def test_update_performance_reports_reads_strategy_config_attribute(self):
        strategy = SimpleNamespace(
            daily_pnl=SimpleNamespace(
                realized_gross_pnl=0,
                realized_net_pnl=0,
                fees_paid=0,
                taxes_paid=0,
                trade_count=0,
                win_count=0,
                loss_count=0,
                breakeven_count=0,
                winning_net_pnl_sum=0,
                losing_net_pnl_sum=0,
                largest_win_net=0,
                largest_loss_net=0,
            ),
            positions={},
            _halted=False,
            config=SimpleNamespace(
                strategy_gate_window_days=7,
                strategy_gate_min_closed_trades=9,
            ),
        )
        balance = SimpleNamespace(
            total_eval_amount=10_000_000,
            total_deposit=1_000_000,
            total_profit_loss=0,
            total_profit_rate=0.0,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            log_root = Path(tmp_dir) / "logs"
            log_root.mkdir()
            (log_root / "trading.log").write_text("", encoding="utf-8")
            paths = update_performance_reports(
                strategy=strategy,
                balance=balance,
                session_pnl=0,
                trading_mode="paper",
                report_root=Path(tmp_dir),
                log_root=log_root,
            )

            gate_payload = json.loads(paths["strategy_gates"]["json"].read_text(encoding="utf-8"))
            self.assertEqual(gate_payload["window_days"], 7)
            self.assertEqual(gate_payload["min_closed_trades"], 9)

    def test_evaluate_strategy_gates_disables_negative_expectancy_strategy(self):
        scorecards = [
            {
                "date": "2026-03-10",
                "log_analysis": {
                    "strategy_pnl": {
                        "opening_conviction_long_strategy": {
                            "closed_trades": 2,
                            "net_pnl": -800,
                            "wins": 0,
                            "losses": 2,
                        }
                    },
                    "strategy_hourly_pnl": {
                        "opening_conviction_long_strategy": {
                            "10": {"closed_trades": 2, "net_pnl": -800, "expectancy": -400.0}
                        }
                    },
                    "shadow_blocked": {
                        "opening_conviction_long_strategy": {
                            "total": 1,
                            "outcomes": {"take_profit_first": 1},
                            "by_reason": {"neutral_loss_limit_block": 1},
                        }
                    },
                },
            },
            {
                "date": "2026-03-11",
                "log_analysis": {
                    "strategy_pnl": {
                        "opening_conviction_long_strategy": {
                            "closed_trades": 2,
                            "net_pnl": -600,
                            "wins": 1,
                            "losses": 1,
                        }
                    },
                    "strategy_hourly_pnl": {
                        "opening_conviction_long_strategy": {
                            "11": {"closed_trades": 2, "net_pnl": -600, "expectancy": -300.0}
                        }
                    },
                    "shadow_blocked": {},
                },
            },
        ]

        payload = evaluate_strategy_gates(scorecards, window_days=5, min_closed_trades=4)

        strategy_gate = payload["strategies"]["opening_conviction_long_strategy"]
        self.assertFalse(strategy_gate["enabled"])
        self.assertEqual(strategy_gate["reason"], "negative_expectancy")
        self.assertEqual(strategy_gate["closed_trades"], 4)
        self.assertAlmostEqual(strategy_gate["expectancy"], -350.0)
        self.assertIn("10", strategy_gate["hour_bucket_expectancy"])

    def test_evaluate_math_shadow_report_summarizes_queue_and_admission_metrics(self):
        payload = evaluate_math_shadow_report(
            [
                {
                    "date": "2026-03-19",
                    "log_analysis": {
                        "trade_records": [
                            {
                                "symbol": "263750",
                                "strategy_name": "opening_conviction_long_strategy",
                                "entry_grade_math": "A",
                                "leader_pct": 0.96,
                                "entry_ev": 850.0,
                                "net_pnl": 2100,
                                "candidate_class": "opening_hot",
                                "bull_prob": 0.72,
                                "neutral_prob": 0.18,
                                "soft_bear_prob": 0.07,
                                "bear_prob": 0.03,
                                "math_dominant_profile": "bull",
                            }
                        ],
                        "math_shadow": {
                            "entries_by_grade": {"A": 1},
                            "grade_pnl": {"A": {"closed_trades": 1, "net_pnl": 2100}},
                            "ev_buckets": {"positive": {"closed_trades": 1, "net_pnl": 2100}},
                            "queue_counts": {"math_queue": 12},
                            "precheck_reached_counts": {"math_queue": 5},
                            "precheck_blocked_counts": {"math_queue": 1},
                            "router_reached_counts": {"math_queue": 4},
                            "router_blocked_counts": {"math_queue": 1},
                            "hard_guard_blocked_counts": {"math_queue": 1},
                            "queue_eval_reached_counts": {"math_queue": 4},
                            "admission_counts": {"passed": 2, "blocked": 1},
                            "candidate_reject_reason_outcomes": {"general_bull_shadow": {"take_profit_first": 2}},
                        },
                    },
                }
            ],
            window_days=5,
            min_closed_trades=1,
        )

        self.assertEqual(payload["queue_counts"]["math_queue"], 12)
        self.assertEqual(payload["precheck_reached_counts"]["math_queue"], 5)
        self.assertEqual(payload["router_reached_counts"]["math_queue"], 4)
        self.assertEqual(payload["queue_eval_reached_counts"]["math_queue"], 4)
        self.assertAlmostEqual(payload["queue_eval_reach_rate"], 4 / 12, places=4)
        self.assertAlmostEqual(payload["admission_pass_rate"], 2 / 3, places=4)
        self.assertIn("0.90-1.00", payload["leader_percentile_buckets"])
        self.assertIn("bull", payload["dominant_regime_profiles"])
        self.assertEqual(payload["candidate_classes"]["opening_hot"]["closed_trades"], 1)
        self.assertEqual(payload["candidate_reject_reason_outcomes"]["general_bull_shadow"]["take_profit_first"], 2)

    def test_evaluate_math_shadow_report_clamps_queue_reach_rate_to_same_funnel_counts(self):
        payload = evaluate_math_shadow_report(
            [
                {
                    "date": "2026-03-24",
                    "log_analysis": {
                        "trade_records": [],
                        "math_shadow": {
                            "queue_counts": {"math_queue": 10},
                            "precheck_reached_counts": {"math_queue": 11},
                            "router_reached_counts": {"math_queue": 12},
                            "queue_eval_reached_counts": {"math_queue": 12},
                            "admission_counts": {"passed": 1, "blocked": 1},
                        },
                    },
                }
            ],
            window_days=5,
            min_closed_trades=1,
        )

        self.assertEqual(payload["queue_eval_reached_counts"]["math_queue"], 10)
        self.assertAlmostEqual(payload["queue_eval_reach_rate"], 1.0, places=4)

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
        self.assertFalse(readiness["ready_for_real_trading"])

    def test_readiness_promotes_stage_one_after_paper_gate(self):
        scorecards = []
        start = datetime(2026, 3, 3, 15, 20, 0)
        paper_net = [10_500, 500, 500, 500, 500]
        for idx, realized_net in enumerate(paper_net):
            current = start + timedelta(days=idx)
            scorecards.append(
                {
                    "date": current.date().isoformat(),
                    "generated_at": current.isoformat(timespec="seconds"),
                    "trading_mode": "paper",
                    "pnl": {
                        "session_pnl": realized_net,
                        "realized_net_pnl": realized_net,
                        "winning_net_pnl_sum": max(realized_net, 0),
                        "losing_net_pnl_sum": min(realized_net, 0),
                    },
                    "trades": {
                        "closed_trades": 5,
                    },
                    "strategy": {
                        "real_money_stage": 0,
                    },
                    "log_analysis": {
                        "risk_events": {
                            "daily_hard_stop_triggered": False,
                            "daily_profit_target_triggered": realized_net >= 10_000,
                        }
                    },
                }
            )

        readiness = evaluate_real_trading_readiness(scorecards)

        self.assertTrue(readiness["paper_gate"]["passed"])
        self.assertEqual(readiness["promotion"]["current_stage_allowed"], 1)
        self.assertTrue(readiness["promotion"]["eligible_for_real_money_stage1"])

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

    def test_build_daily_scorecard_tracks_opening_conviction_and_ev_reject_shadow(self):
        strategy = SimpleNamespace(
            daily_pnl=SimpleNamespace(
                realized_gross_pnl=9_000,
                realized_net_pnl=8_400,
                fees_paid=200,
                taxes_paid=400,
                trade_count=1,
                win_count=1,
                loss_count=0,
                breakeven_count=0,
                winning_net_pnl_sum=8_400,
                losing_net_pnl_sum=0,
                largest_win_net=8_400,
                largest_loss_net=0,
            ),
            positions={},
            _halted=False,
            _real_money_stage=0,
            _capital_scale=1.0,
        )
        balance = SimpleNamespace(
            total_eval_amount=10_080_000,
            total_deposit=1_000_000,
            total_profit_loss=8_400,
            total_profit_rate=0.84,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            log_root = Path(tmp_dir)
            (log_root / "trading.log").write_text(
                "\n".join(
                    [
                        "2026-03-31 09:01:00 [INFO] kis_trader.strategy.momentum: "
                        "매수 신호: 263750 50주 @ 10,000원 "
                        "(할당 500,000원, strategy_name=opening_conviction_long_strategy setup_name=opening_conviction "
                        "entry_reason=opening_conviction regime_label=bull candidate_class=opening_conviction "
                        "execution_mode=live live_route=opening_conviction_long_strategy "
                        "queue_source=opening_fast_queue conviction_score=0.9720 conviction_rank=1 "
                        "leader_score=2.1000 effective_leader_score=1.6500 leader_pct=0.9900 recent_accel=0.5500 "
                        "entry_ev=650.00 entry_ev_conf=medium entry_ev_trades=6 size_multiplier=1.0000)",
                        "2026-03-31 09:04:00 [INFO] kis_trader.strategy.momentum: "
                        "진입 거부[ev_prediction_non_positive] reject_reason=ev_prediction_non_positive: 123310 (ev_prediction_non_positive execution_mode=rejected live_route=intraday_conviction_long_strategy)",
                        "2026-03-31 09:05:00 [INFO] kis_trader.strategy.momentum: "
                        "그림자 후보 종료: 123310 strategy_name=intraday_conviction_long_strategy shadow_reason=ev_prediction_non_positive outcome=stop_loss_first",
                        "2026-03-31 09:12:00 [INFO] kis_trader.strategy.momentum: "
                        "매도 체결: 263750 50주 @ 10,180원 (총손익: 9,000원, 순손익: 8,400원, 누적순손익: 8,400원, "
                        "strategy_name=opening_conviction_long_strategy, setup_name=opening_conviction, regime_label=bull, "
                        "entry_grade=A, entry_grade_math=A, leader_score=2.1000, leader_pct=0.9900, entry_ev=650.00, entry_ev_conf=medium, "
                        "conviction_tier=confirmed, bull_risk_mode=normal, post_loss_admission_class=general, candidate_class=opening_conviction, execution_mode=live, live_route=opening_conviction_long_strategy, queue_source=opening_fast_queue, size_multiplier=1.0000, conviction_score=0.9720, conviction_rank=1)",
                    ]
                ),
                encoding="utf-8",
            )

            scorecard = build_daily_scorecard(
                strategy=strategy,
                balance=balance,
                session_pnl=8_400,
                trading_mode="paper",
                generated_at=datetime(2026, 3, 31, 15, 20, 0),
                log_root=log_root,
            )

        record = scorecard["log_analysis"]["trade_records"][0]
        self.assertEqual(record["strategy_name"], "opening_conviction_long_strategy")
        self.assertEqual(record["queue_source"], "opening_fast_queue")
        self.assertEqual(record["execution_mode"], "live")
        self.assertEqual(record["live_route"], "opening_conviction_long_strategy")
        self.assertAlmostEqual(record["conviction_score"], 0.9720, places=4)
        self.assertEqual(record["conviction_rank"], 1)
        self.assertEqual(
            scorecard["log_analysis"]["math_shadow"]["candidate_reject_reason_outcomes"]["ev_prediction_non_positive"]["stop_loss_first"],
            1,
        )
        self.assertEqual(
            scorecard["log_analysis"]["math_shadow"]["live_route_pnl"]["opening_conviction_long_strategy"]["closed_trades"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
