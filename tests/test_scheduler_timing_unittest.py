import gzip
import json
import unittest
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from threading import Event, get_ident
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import requests

from src.api_client import KISClient
from src.market_stream import MarketQuoteStream
from src.models import Order, OrderResult, OrderSide, Quote
from src.scheduler import TradingScheduler
from src.strategies.momentum_scalp import MomentumScalpStrategy
from src.strategies.momentum_scalp_types import MomentumScalpConfig, PositionState


class _SequenceDateTime(datetime):
    timeline = []
    last = None

    @classmethod
    def now(cls, tz=None):
        if cls.timeline:
            cls.last = cls.timeline.pop(0)
            return cls.last
        if cls.last is None:
            raise AssertionError("datetime.now() 호출 시각이 준비되지 않았습니다.")
        return cls.last


def strategy_stub(**values):
    defaults = {
        "positions": {},
        "config": SimpleNamespace(
            quote_freshness_seconds=5,
            static_watchlist=[],
            inverse_etfs=[],
            pending_order_block_seconds=180,
            dynamic_pool_quote_min_change_rate=0.0,
            dynamic_pool_ranking_fetch_count=60,
            pool_refresh_interval=120,
        ),
        "fixed_pool": [],
        "_pool": [],
        "_last_evaluation_count": 0,
        "_pending_entry_meta": {},
        "_pending_sell_fills": {},
        "_sell_fill_ledger": [],
        "on_order_filled": Mock(),
        "daily_pnl": SimpleNamespace(
            realized_net_pnl=0,
            trade_count=0,
            win_count=0,
            loss_count=0,
        ),
        "set_simulated_now": Mock(),
        "should_continue": Mock(return_value=True),
        "sync_positions_from_account": Mock(),
        "reconcile_pending_fills_from_account": Mock(return_value=[]),
        "reconcile_no_holding_sell_failures_from_account": Mock(return_value=[]),
        "confirm_reconciled_sell_fills": Mock(return_value=[]),
    }
    defaults.update(values)
    return SimpleNamespace(**defaults)


class SchedulerTimingTests(unittest.TestCase):
    def setUp(self):
        self.scheduler = TradingScheduler.__new__(TradingScheduler)
        self.scheduler.config = SimpleNamespace(off_hours_check_interval=1800, is_paper=True, trading_mode="paper")
        self.scheduler._shutdown = False
        self.scheduler._last_rate_limit_watchlist_log_at = None
        self.scheduler.client = SimpleNamespace(rate_limit_cooldown_remaining=lambda: 0.0)
        self.scheduler.account = Mock()
        self.scheduler.market_data = SimpleNamespace(
            refresh_instruments=Mock(),
            multi_price_backoff_remaining=lambda: 0.0,
            get_fluctuation_symbols=lambda **kwargs: [],
            get_multi_price=lambda symbols: [],
        )
        self.scheduler.strategy = strategy_stub()
        self.scheduler._quote_stream = Mock()
        self.scheduler._quote_stream.stale_symbols.return_value = []
        self.scheduler._quote_stream.drain_quotes.return_value = []
        self.scheduler._alerts = Mock()
        self.scheduler._send_alert = self.scheduler._alerts.send
        self.scheduler._last_halt_alert_day = None
        self.scheduler._last_loop_error_alert_at = None
        self.scheduler._last_runtime_pool_refresh_at = None
        self.scheduler._session_started_at = None
        self.scheduler._market_closed_session_date = None
        self.scheduler._account_order_blocked_session_date = None
        self.scheduler._last_stream_stale_log_at = None
        self.scheduler._last_emergency_position_refresh_at = None
        self.scheduler._last_tick_activity_log_at = None
        self.scheduler._last_pending_entry_reconcile_at = None
        self.scheduler._last_pending_exit_reconcile_at = None
        self.scheduler._pending_order_blocks = {}
        self.scheduler._market_reader = None
        self.scheduler._market_read = None
        self.scheduler._last_supplemental_read_at = {}
        self.scheduler._pool_read = None
        self.scheduler._order_execution = None
        self.scheduler._executing_orders = []
        self.scheduler._account_read = None
        self.scheduler._account_reconcile_results = []
        self.scheduler._account_snapshot_results = []
        self.scheduler._order_sequence = 0
        self.scheduler._account_order_sequence = 0
        self.scheduler._alert_sender = None
        self.scheduler.trading = SimpleNamespace(resolve_pending_entry=Mock(return_value=None))
        self.addCleanup(self.scheduler._stop_alerts)
        self.addCleanup(self.scheduler._stop_market_reads)

    def test_session_start_balance_delay_uses_paper_buffer(self):
        delay = self.scheduler._session_start_balance_delay_seconds()

        self.assertEqual(delay, 2.0)

    def test_pending_market_result_blocks_duplicate_same_side_order(self):
        now = datetime(2026, 5, 27, 13, 40, 23)
        self.scheduler._remember_pending_order_results(
            [
                OrderResult(
                    success=True,
                    symbol="274090",
                    side=OrderSide.SELL,
                    quantity=0,
                    price=0,
                    fill_mode="market_pending",
                    timestamp=now,
                )
            ],
            now,
        )

        filtered = self.scheduler._filter_pending_duplicate_orders(
            [
                Order(symbol="274090", side=OrderSide.SELL, quantity=26),
                Order(symbol="005930", side=OrderSide.SELL, quantity=1),
            ],
            now + timedelta(seconds=20),
        )

        self.assertEqual([order.symbol for order in filtered], ["005930"])

    def test_pending_exit_is_reconciled_again_on_following_tick(self):
        now = datetime(2026, 7, 28, 12, 54, 50)
        position = SimpleNamespace(
            symbol="463020",
            pending_exit_started_at=now - timedelta(seconds=15),
            pending_exit_quantity=91,
        )
        sync_positions = Mock()
        self.scheduler.strategy = strategy_stub(
            positions={"463020": position},
            sync_positions_from_account=sync_positions,
        )
        balance = SimpleNamespace(positions=[])
        self.scheduler._fetch_balance_with_retry = Mock(return_value=balance)
        self.scheduler._confirm_reconciled_sell_prices = Mock()

        reconciled = self.scheduler._reconcile_pending_orders(wait=True)

        self.assertTrue(reconciled)
        self.scheduler._fetch_balance_with_retry.assert_called_once_with(
            "주문 상태 재동기화",
            max_attempts=1,
            base_delay_seconds=1,
        )
        sync_positions.assert_called_once_with([])
        self.scheduler._confirm_reconciled_sell_prices.assert_called_once_with(history=self.scheduler.account.get_order_history.return_value)

    def test_orphan_pending_sell_is_reconciled_after_position_disappears(self):
        sync_positions = Mock()
        self.scheduler.strategy = strategy_stub(
            positions={},
            _pending_sell_fills={"463020": {"requested_quantity": 91}},
            sync_positions_from_account=sync_positions,
        )
        self.scheduler._fetch_balance_with_retry = Mock(
            return_value=SimpleNamespace(positions=[])
        )
        self.scheduler._confirm_reconciled_sell_prices = Mock()

        reconciled = self.scheduler._reconcile_pending_orders(wait=True)

        self.assertTrue(reconciled)
        sync_positions.assert_called_once_with([])
        self.scheduler._confirm_reconciled_sell_prices.assert_called_once_with(history=self.scheduler.account.get_order_history.return_value)

    def test_confirmed_reconcile_clears_pending_order_block(self):
        now = datetime(2026, 5, 27, 13, 40, 23)
        pending = OrderResult(
            success=True,
            symbol="274090",
            side=OrderSide.SELL,
            quantity=0,
            price=0,
            fill_mode="market_pending",
            timestamp=now,
        )
        self.scheduler._remember_pending_order_results([pending], now)
        self.scheduler.strategy.confirm_reconciled_sell_fills = Mock(
            return_value=[
                {
                    "order_no": "0000012345",
                    "symbol": "274090",
                    "quantity": 26,
                    "requested_quantity": 26,
                    "corrected_price": 38_650,
                    "alert_quantity": 26,
                    "alert_price": 38_650,
                    "remaining_quantity": 0,
                    "partial": False,
                    "notify": False,
                }
            ]
        )

        self.scheduler._confirm_reconciled_sell_prices()
        filtered = self.scheduler._filter_pending_duplicate_orders(
            [Order(symbol="274090", side=OrderSide.SELL, quantity=26)],
            now + timedelta(seconds=20),
        )

        self.assertEqual(len(filtered), 1)

    def test_partial_reconcile_keeps_pending_order_block(self):
        now = datetime(2026, 7, 16, 9, 6, 48)
        pending = OrderResult(
            success=True,
            symbol="051980",
            side=OrderSide.SELL,
            quantity=0,
            requested_quantity=301,
            fill_mode="market_pending",
            timestamp=now,
        )
        self.scheduler._remember_pending_order_results([pending], now)
        partial = OrderResult(
            success=True,
            symbol="051980",
            side=OrderSide.SELL,
            quantity=124,
            requested_quantity=301,
            price=1_167,
            fill_mode="account_reconciled_estimated",
            timestamp=now + timedelta(seconds=4),
        )

        self.scheduler._clear_pending_order_blocks_for_results([partial])
        filtered = self.scheduler._filter_pending_duplicate_orders(
            [Order(symbol="051980", side=OrderSide.SELL, quantity=177)],
            now + timedelta(seconds=10),
        )

        self.assertEqual(filtered, [])

    def test_sell_no_holding_failure_does_not_create_three_minute_order_block(self):
        now = datetime(2026, 6, 9, 12, 8, 7)
        no_holding = OrderResult(
            success=False,
            symbol="457370",
            side=OrderSide.SELL,
            message="[40240000] 모의투자 잔고내역이 없습니다.",
            error_category="no_holding",
            timestamp=now,
        )
        self.scheduler._remember_pending_order_results([no_holding], now)

        filtered = self.scheduler._filter_pending_duplicate_orders(
            [Order(symbol="457370", side=OrderSide.SELL, quantity=39)],
            now + timedelta(seconds=20),
        )

        self.assertEqual(len(filtered), 1)

    def test_pending_sell_allows_only_unreserved_remainder(self):
        now = datetime(2026, 7, 15, 13, 40, 42)
        self.scheduler.strategy.positions = {
            "303360": SimpleNamespace(quantity=143, pending_exit_quantity=2),
        }
        self.scheduler._remember_pending_order_results(
            [
                OrderResult(
                    success=True,
                    symbol="303360",
                    side=OrderSide.SELL,
                    quantity=0,
                    requested_quantity=2,
                    fill_mode="market_pending",
                    timestamp=now,
                )
            ],
            now,
        )

        filtered = self.scheduler._filter_pending_duplicate_orders(
            [Order(symbol="303360", side=OrderSide.SELL, quantity=141)],
            now + timedelta(seconds=10),
        )

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].quantity, 141)

    def test_session_start_balance_delay_respects_client_cooldown(self):
        self.scheduler.client = SimpleNamespace(rate_limit_cooldown_remaining=lambda: 2.2)

        delay = self.scheduler._session_start_balance_delay_seconds()

        self.assertEqual(delay, 2.2)

    def test_fetch_balance_with_retry_waits_for_rate_limit_cooldown(self):
        sleep_calls = []
        responses = [None, SimpleNamespace(total_deposit=1, positions=[])]
        self.scheduler.account = SimpleNamespace(get_balance=Mock(side_effect=responses))
        self.scheduler.client = SimpleNamespace(rate_limit_cooldown_remaining=lambda: 3.2)
        self.scheduler._interruptible_sleep = lambda seconds: sleep_calls.append(seconds)

        balance = self.scheduler._fetch_balance_with_retry("세션 시작 잔고 조회", max_attempts=2, base_delay_seconds=2)

        self.assertIsNotNone(balance)
        self.assertEqual(sleep_calls, [4])
        self.assertEqual(self.scheduler.account.get_balance.call_args_list, [call(timeout=None)] * 2)

    def test_fetch_balance_with_retry_handles_transport_timeout_inside_retry_loop(self):
        sleep_calls = []
        responses = [
            requests.exceptions.Timeout("balance timeout"),
            SimpleNamespace(total_deposit=1, positions=[]),
        ]

        self.scheduler.account = SimpleNamespace(get_balance=Mock(side_effect=responses))
        self.scheduler._interruptible_sleep = lambda seconds: sleep_calls.append(seconds)

        balance = self.scheduler._fetch_balance_with_retry(
            "세션 시작 잔고 조회", max_attempts=2, base_delay_seconds=2, timeout=(10.0, 60.0)
        )

        self.assertIsNotNone(balance)
        self.assertEqual(sleep_calls, [2])
        self.assertEqual(
            self.scheduler.account.get_balance.call_args_list, [call(timeout=(10.0, 60.0))] * 2
        )

    def test_session_start_balance_timeout_still_blocks_trading_after_retries(self):
        self.scheduler.market_data.is_market_open = lambda: True
        self.scheduler.strategy.initialize = Mock()
        self.scheduler._session_start_balance_delay_seconds = lambda: 0.0
        self.scheduler._balance_retry_attempts = 3
        self.scheduler._balance_retry_delay_seconds = 2
        self.scheduler._interruptible_sleep = Mock()
        self.scheduler.account.get_balance.side_effect = requests.exceptions.ReadTimeout("balance timeout")

        halted = self.scheduler._run_trading_session(1)

        self.assertFalse(halted)
        self.assertEqual(
            self.scheduler.account.get_balance.call_args_list, [call(timeout=(10.0, 60.0))] * 3
        )
        self.assertEqual(self.scheduler._interruptible_sleep.call_args_list, [call(2), call(4), call(60)])
        self.scheduler.strategy.sync_positions_from_account.assert_not_called()
        self.scheduler._quote_stream.start.assert_not_called()

    def test_run_does_not_send_kakao_alert_for_scheduler_network_error(self):
        self.scheduler.config.log_level = "INFO"
        self.scheduler._is_trading_time = Mock(side_effect=requests.exceptions.Timeout("balance timeout"))
        self.scheduler._send_loop_error_alert = Mock()

        def stop_after_sleep(seconds):
            self.scheduler._shutdown = True

        self.scheduler._interruptible_sleep = stop_after_sleep

        with patch("src.scheduler.setup_logger", return_value=None):
            self.scheduler.run(tick_interval=10)

        self.scheduler._send_loop_error_alert.assert_not_called()

    def test_daily_halt_finishes_summary_before_quote_collection_and_wait(self):
        now = datetime(2026, 10, 15, 12, 30)
        for reason in ("daily_hard_stop", "daily_profit_target"):
            with self.subTest(reason=reason):
                events = []
                self.scheduler._shutdown = False
                self.scheduler.config.log_level = "INFO"
                self.scheduler.market_data.is_market_open = Mock(return_value=True)
                self.scheduler.strategy = strategy_stub(
                    initialize=Mock(), get_watchlist=Mock(return_value=["005930"]),
                    should_continue=Mock(return_value=False), on_batch_tick=Mock(),
                )
                self.scheduler.executor = Mock()
                self.scheduler._session_start_balance_delay_seconds = Mock(return_value=0)
                self.scheduler._refresh_runtime_pool_if_needed = Mock(return_value=False)
                self.scheduler._fetch_balance_with_retry = Mock(return_value=SimpleNamespace(
                    total_deposit=1000000, total_eval_amount=1000000, positions=[]))
                self.scheduler._resolve_session_profit_loss = Mock(return_value=0)
                def finish_report(*args):
                    self.scheduler._last_session_end_reason = reason
                    events.append("report")
                self.scheduler._write_performance_reports = finish_report
                self.scheduler._send_session_summary_alert = Mock(side_effect=lambda *args: events.append("summary"))
                self.scheduler._collect_quotes_until_close = Mock(side_effect=lambda day: events.append("quotes"))
                def wait_until_tomorrow():
                    events.append("wait")
                    self.scheduler._shutdown = True
                self.scheduler._sleep_until_preopen = Mock(side_effect=wait_until_tomorrow)
                with patch("src.scheduler.datetime") as clock, patch("src.scheduler.setup_logger"):
                    clock.now.return_value = now
                    self.scheduler.run()
                self.assertEqual(events, ["report", "summary", "quotes", "wait"])
                self.scheduler._collect_quotes_until_close.assert_called_once_with(now.date())
                self.assertEqual(self.scheduler._fetch_balance_with_retry.call_count, 2)
                self.scheduler.strategy.on_batch_tick.assert_not_called()
                self.scheduler.executor.assert_not_called()
                self.assertEqual(self.scheduler.executor.mock_calls, [])
                self.assertEqual(self.scheduler._last_session_end_reason, reason)

    def test_quote_collection_archives_full_watchlist_and_refreshes_pool_without_trading(self):
        clock = [datetime(2026, 10, 15, 12, 30)]
        with tempfile.TemporaryDirectory() as directory:
            config = MomentumScalpConfig(
                daily_state_path=f"{directory}/state.json", quote_archive_root=f"{directory}/quotes",
                allow_hard_stop_bypass_for_day=True)
            strategy = MomentumScalpStrategy(self.scheduler.market_data, config)
            strategy.set_simulated_now(clock[0])
            strategy._active_day = "20261015"
            strategy._pool = [f"{900000 + i:06d}" for i in range(30)]
            strategy._halted, strategy._halt_reason = True, "daily_total_loss_limit"
            strategy.daily_pnl.realized_net_pnl = -8664
            strategy._loss_guard_realized_baseline = -4720
            strategy._hard_stop_bypass_active = True
            strategy.on_batch_tick = Mock()
            strategy.should_continue = Mock()
            strategy.initialize = Mock()
            self.scheduler.strategy = strategy
            self.scheduler.executor = Mock()
            self.scheduler._submit_strategy_orders = Mock()
            self.scheduler._reconcile_pending_orders = Mock()
            self.scheduler._collect_order_execution = Mock()
            initial_universe = strategy.get_watchlist()
            strategy._quote_archive.record([], now=clock[0] - timedelta(seconds=1), universe=initial_universe)
            refreshed_pool = ["900100"] + strategy._pool[1:]
            self.scheduler.market_data.get_fluctuation_symbols = Mock(return_value=refreshed_pool)
            desired = []
            def update_symbols(symbols):
                desired[:] = symbols
            def quote(symbol):
                return Quote(symbol=symbol, timestamp=clock[0], current_price=10000,
                             bid_price=9990, ask_price=10010, bid_size=100, ask_size=200,
                             book_available=True, book_depth_available=True,
                             trade_high_price=10005, security_group="ST")
            def drain():
                quotes = [quote(symbol) for symbol in desired[:40]]
                return quotes + [replace(quotes[0], current_price=0),
                                 replace(quotes[0], timestamp=clock[0] + timedelta(seconds=60))]
            self.scheduler.market_data.get_multi_price = Mock(side_effect=lambda symbols: [quote(s) for s in symbols])
            self.scheduler._quote_stream.start.side_effect = update_symbols
            self.scheduler._quote_stream.update_symbols.side_effect = update_symbols
            self.scheduler._quote_stream.subscribed_symbols.side_effect = lambda: desired[:40]
            self.scheduler._quote_stream.drain_quotes.side_effect = drain
            ticks = []
            def advance(seconds):
                # Finish mocked REST I/O deterministically, without any broker connection.
                for pending in (self.scheduler._pool_read, self.scheduler._market_read):
                    if pending is not None:
                        pending.result(timeout=2)
                ticks.append(seconds)
                clock[0] = (clock[0] + timedelta(seconds=120 if len(ticks) == 1 else 1)
                            if len(ticks) < 3 else datetime(2026, 10, 15, 15, 30))
            self.scheduler._interruptible_sleep = advance
            with patch("src.scheduler.datetime") as dt:
                dt.now.side_effect = lambda: clock[0]
                dt.combine.side_effect = datetime.combine
                self.scheduler._collect_quotes_until_close(clock[0].date())
            path = Path(directory) / "quotes/2026/10/quote-batches.2026-10-15.jsonl.gz"
            with gzip.open(path, "rt") as handle:
                rows = [json.loads(line) for line in handle]
            self.assertEqual(len(rows), 4)
            self.assertEqual([row["reset_history"] for row in rows], [True, False, False, False])
            self.assertEqual(len(rows[-1]["universe"]), 66)
            self.assertEqual(set(rows[-1]["universe"]), set(strategy.get_watchlist()))
            self.assertEqual({q["symbol"] for q in rows[-1]["quotes"]}, set(strategy.get_watchlist()))
            self.assertIn("900100", rows[-1]["universe"])
            self.assertNotIn("900000", rows[-1]["universe"])
            for q in rows[-1]["quotes"]:
                self.assertEqual((q["bid_price"], q["ask_price"], q["bid_size"], q["ask_size"]),
                                 (9990, 10010, 100, 200))
                age = (datetime.fromisoformat(rows[-1]["observed_timestamp"])
                       - datetime.fromisoformat(q["timestamp"])).total_seconds()
                self.assertIn(age, (0, 1))
                self.assertEqual(q["trade_high_price"], 10005)
            self.assertEqual({q["timestamp"] for q in rows[-1]["quotes"]},
                             {"2026-10-15T12:32:00", "2026-10-15T12:32:01"})
            self.assertEqual(self.scheduler.market_data.get_fluctuation_symbols.call_count, 2)
            self.assertEqual(len(config.static_watchlist), 32)
            self.assertTrue(set(config.static_watchlist + config.inverse_etfs) <= set(rows[-1]["universe"]))
            self.assertTrue(strategy._halted)
            self.assertEqual(strategy._halt_reason, "daily_total_loss_limit")
            self.assertEqual(strategy.daily_pnl.realized_net_pnl, -8664)
            self.assertEqual(strategy._loss_guard_realized_baseline, -4720)
            self.assertTrue(strategy._hard_stop_bypass_active)
            saved = json.loads(Path(config.daily_state_path).read_text())
            self.assertTrue(saved["halted"])
            self.assertEqual(saved["halt_reason"], "daily_total_loss_limit")
            self.assertEqual(saved["sell_fill_ledger"], [])
            self.assertEqual(saved["positions"], [])
            strategy.on_batch_tick.assert_not_called()
            strategy.should_continue.assert_not_called()
            strategy.initialize.assert_not_called()
            self.scheduler._submit_strategy_orders.assert_not_called()
            self.scheduler._reconcile_pending_orders.assert_not_called()
            self.scheduler._collect_order_execution.assert_not_called()
            self.assertEqual(self.scheduler.account.mock_calls, [])
            self.assertEqual(self.scheduler.executor.mock_calls, [])
            self.assertEqual(self.scheduler._alerts.mock_calls, [])
            self.scheduler._quote_stream.stop.assert_called_once()
            self.assertIsNone(self.scheduler._market_reader)
            self.assertIsNone(self.scheduler._market_read)
            self.assertIsNone(self.scheduler._pool_read)

    def test_quote_collection_close_stop_errors_and_overnight_do_not_trade_or_alert(self):
        start = datetime(2026, 10, 15, 15, 29, 58)
        for mode in ("close", "stop", "overnight", "timeout", "error", "already_closed", "next_day"):
            with self.subTest(mode=mode):
                self.scheduler._shutdown = False
                clock = [start]
                if mode == "already_closed":
                    clock[0] = datetime(2026, 10, 15, 15, 30)
                elif mode == "next_day":
                    clock[0] = start + timedelta(days=1)
                self.scheduler.strategy = strategy_stub(
                    fixed_pool=["005930"], get_watchlist=Mock(return_value=["005930"]),
                    _quote_archive=Mock(), on_batch_tick=Mock())
                self.scheduler._quote_stream = Mock()
                self.scheduler._quote_stream.subscribed_symbols.return_value = ["005930"]
                self.scheduler._quote_stream.stale_symbols.return_value = []
                self.scheduler._quote_stream.drain_quotes.return_value = []
                if mode in {"timeout", "error"}:
                    self.scheduler._quote_stream.drain_quotes.side_effect = (
                        requests.exceptions.Timeout("mock timeout") if mode == "timeout" else RuntimeError("mock error"))
                def advance(seconds):
                    if mode == "stop":
                        self.scheduler.stop()
                    elif mode == "overnight":
                        clock[0] = datetime(2026, 10, 16, 9)
                    else:
                        clock[0] += timedelta(seconds=seconds)
                self.scheduler._interruptible_sleep = advance
                with patch("src.scheduler.datetime") as dt:
                    dt.now.side_effect = lambda: clock[0]
                    dt.combine.side_effect = datetime.combine
                    self.scheduler._collect_quotes_until_close(start.date())
                if mode in {"already_closed", "next_day"}:
                    self.scheduler._quote_stream.start.assert_not_called()
                    self.scheduler.strategy._quote_archive.record.assert_not_called()
                else:
                    self.scheduler._quote_stream.start.assert_called_once()
                    self.assertGreaterEqual(self.scheduler._quote_stream.stop.call_count, 1)
                if mode in {"timeout", "error"}:
                    self.scheduler.strategy._quote_archive.record.assert_not_called()
                self.scheduler.strategy.on_batch_tick.assert_not_called()
                self.scheduler.strategy.should_continue.assert_not_called()
                self.assertEqual(self.scheduler.account.mock_calls, [])
                self.assertEqual(self.scheduler._alerts.mock_calls, [])
                self.assertIsNone(self.scheduler._market_reader)

    def test_overnight_resume_after_collection_can_start_next_trading_day(self):
        clock = [datetime(2026, 10, 15, 12, 30)]
        self.scheduler.config.log_level = "INFO"
        def session(tick_interval):
            if clock[0].date() == datetime(2026, 10, 15).date():
                return True
            self.scheduler._shutdown = True
            return False
        def collect(day):
            clock[0] = datetime(2026, 10, 16, 9, 10)
        self.scheduler._run_trading_session = Mock(side_effect=session)
        self.scheduler._collect_quotes_until_close = Mock(side_effect=collect)
        self.scheduler._sleep_until_preopen = Mock()
        with patch("src.scheduler.datetime") as dt, patch("src.scheduler.setup_logger"):
            dt.now.side_effect = lambda: clock[0]
            self.scheduler.run()
        self.assertEqual(self.scheduler._run_trading_session.call_count, 2)
        self.scheduler._collect_quotes_until_close.assert_called_once_with(datetime(2026, 10, 15).date())
        self.scheduler._sleep_until_preopen.assert_not_called()

    def test_run_trading_session_does_not_send_kakao_alert_for_intraday_network_error(self):
        self.scheduler.market_data = SimpleNamespace(is_market_open=lambda: True, refresh_instruments=Mock())
        self.scheduler.strategy = strategy_stub(
            initialize=lambda: None,
            get_watchlist=lambda: ["005930"],
            sync_positions_from_account=lambda positions: None,
            should_continue=Mock(side_effect=requests.exceptions.ConnectionError("temporary network")),
            positions={},
        )
        self.scheduler._alerts = Mock()
        self.scheduler._send_loop_error_alert = Mock()
        self.scheduler._session_start_balance_delay_seconds = lambda: 0.0
        self.scheduler._extract_strategy_realized_pnl = lambda: 0
        self.scheduler._refresh_runtime_pool_if_needed = lambda now, force=False: False
        self.scheduler._fetch_balance_with_retry = lambda *args, **kwargs: SimpleNamespace(
            total_deposit=1,
            total_eval_amount=1,
            total_profit_loss=0,
            positions=[],
        )
        self.scheduler._is_trading_time = lambda now: not self.scheduler._shutdown
        self.scheduler._write_performance_reports = lambda balance, session_pnl: None
        self.scheduler._resolve_session_profit_loss = lambda balance, strategy_pnl_baseline=None: 0
        self.scheduler._send_session_summary_alert = lambda balance, session_pnl: None
        self.scheduler._reset_strategy_simulated_clock = lambda: None
        self.scheduler._confirm_reconciled_sell_prices = lambda results=None: None
        self.scheduler._interruptible_sleep = lambda seconds: setattr(self.scheduler, "_shutdown", True)

        halted = self.scheduler._run_trading_session(10)

        self.assertFalse(halted)
        self.scheduler._send_loop_error_alert.assert_not_called()

    def test_halted_position_uses_batch_route_until_sell_confirmation(self):
        for already_submitted in (False, True):
            with self.subTest(already_submitted=already_submitted), tempfile.TemporaryDirectory() as directory:
                now = datetime(2026, 10, 21, 14, 44, 40)
                clock = [now]
                self.scheduler._shutdown = False
                s = MomentumScalpStrategy(None, MomentumScalpConfig(
                    daily_state_path=f"{directory}/state.json", quote_archive_root=f"{directory}/quotes"),
                    fixed_pool=["011370"])
                s.set_simulated_now(now)
                s._ensure_active_day()
                position = PositionState("011370", 4670, 3, now-timedelta(seconds=20))
                if already_submitted:
                    position.pending_exit_started_at = now
                    position.pending_exit_quantity = 3
                s.positions[position.symbol] = position
                s._halted, s._halt_reason = True, "daily_total_loss_limit"
                s.daily_pnl.realized_net_pnl = -4140
                s.initialize = Mock()
                s.sync_positions_from_account = Mock()
                self.scheduler.strategy = s
                self.scheduler.market_data.is_market_open = lambda: True
                self.scheduler._session_start_balance_delay_seconds = lambda: 0
                self.scheduler._reset_strategy_simulated_clock = Mock()
                self.scheduler._fetch_balance_with_retry = Mock(return_value=SimpleNamespace(
                    total_deposit=1, total_eval_amount=1, positions=[]))
                self.scheduler._refresh_runtime_pool_if_needed = Mock(return_value=False)
                self.scheduler._reconcile_pending_orders = Mock(return_value=False)
                self.scheduler._schedule_market_reads = Mock()
                self.scheduler._write_performance_reports = Mock()
                self.scheduler._resolve_session_profit_loss = Mock(return_value=-4140)
                self.scheduler._send_session_summary_alert = Mock()
                self.scheduler._is_trading_time = lambda _: not self.scheduler._shutdown
                self.scheduler._quote_stream.drain_quotes.side_effect = lambda: [
                    Quote("011370", 4325, timestamp=clock[0], bid_price=4320, ask_price=4330,
                          book_available=True)]
                def submit(orders):
                    self.assertEqual([(o.symbol, o.side, o.quantity) for o in orders],
                                     [("011370", OrderSide.SELL, 3)])
                    position.pending_exit_started_at = clock[0]
                    position.pending_exit_quantity = 3
                    return True
                self.scheduler._submit_strategy_orders = Mock(side_effect=submit)
                def advance(seconds):
                    clock[0] += timedelta(seconds=1)
                    s.set_simulated_now(clock[0])
                    self.scheduler._shutdown = clock[0] >= now+timedelta(seconds=3)
                self.scheduler._interruptible_sleep = advance
                with patch("src.scheduler.datetime") as dt:
                    dt.now.side_effect = lambda: clock[0]
                    self.assertTrue(self.scheduler._run_trading_session(1))
                path = Path(directory)/"quotes/2026/10/quote-batches.2026-10-21.jsonl.gz"
                self.assertTrue(path.exists(), "halted holdings must still drain and archive quotes")
                with gzip.open(path, "rt") as stream:
                    rows = [json.loads(line) for line in stream]
                self.assertEqual(len(rows), 3)
                self.assertEqual([row["quotes"][0]["timestamp"] for row in rows],
                                 [(now+timedelta(seconds=i)).isoformat() for i in range(3)])
                self.assertEqual(s._quotes_cache["011370"].timestamp, now+timedelta(seconds=2))
                self.assertEqual(self.scheduler._submit_strategy_orders.call_count, 0 if already_submitted else 1)
                self.assertEqual(self.scheduler._schedule_market_reads.call_count, 3)
                self.assertTrue(s._halted)
                self.assertEqual(s.daily_pnl.realized_net_pnl, -4140)
                self.assertEqual(s._sell_fill_ledger, [])
                self.assertEqual(s.positions["011370"].quantity, 3)

    def test_halted_session_waits_for_pending_sell_confirmation_before_resuming(self):
        runtime = {"continue": False}
        self.scheduler.market_data = SimpleNamespace(is_market_open=lambda: True, refresh_instruments=Mock())
        self.scheduler.strategy = strategy_stub(
            initialize=lambda: None,
            get_watchlist=lambda: ["067310"],
            sync_positions_from_account=lambda positions: None,
            should_continue=lambda: runtime["continue"],
            positions={},
            _pending_sell_fills={"067310": {"requested_quantity": 1}},
            on_batch_tick=lambda quotes: setattr(self.scheduler, "_shutdown", True) or [],
        )
        self.scheduler._alerts = Mock()
        self.scheduler._session_start_balance_delay_seconds = lambda: 0.0
        self.scheduler._extract_strategy_realized_pnl = lambda: 0
        self.scheduler._refresh_runtime_pool_if_needed = lambda now, force=False: False
        self.scheduler._fetch_balance_with_retry = lambda *args, **kwargs: SimpleNamespace(
            total_deposit=1,
            total_eval_amount=1,
            total_profit_loss=0,
            positions=[],
        )
        self.scheduler._is_trading_time = lambda now: True
        self.scheduler._write_performance_reports = lambda balance, session_pnl: None
        self.scheduler._resolve_session_profit_loss = lambda balance, strategy_pnl_baseline=None: 0
        self.scheduler._send_session_summary_alert = lambda balance, session_pnl: None
        self.scheduler._reset_strategy_simulated_clock = lambda: None
        def confirm(results=None, **kwargs):
            runtime["continue"] = True
            self.scheduler.strategy._pending_sell_fills.clear()
        self.scheduler._confirm_reconciled_sell_prices = confirm
        self.scheduler._interruptible_sleep = lambda seconds: None

        halted = self.scheduler._run_trading_session(10)

        self.assertFalse(halted)
        self.assertTrue(runtime["continue"])

    def test_run_trading_session_fetches_balance_before_runtime_pool_refresh(self):
        order = []
        self.scheduler.market_data = SimpleNamespace(is_market_open=lambda: True, refresh_instruments=Mock())
        self.scheduler.strategy = strategy_stub(
            initialize=lambda: order.append("initialize"),
            get_watchlist=lambda: order.append("watchlist") or ["005930"],
            sync_positions_from_account=lambda positions: order.append("sync"),
            should_continue=lambda: False,
            positions={},
        )
        self.scheduler._alerts = Mock()
        self.scheduler._interruptible_sleep = lambda seconds: None
        self.scheduler._session_start_balance_delay_seconds = lambda: 0.0
        self.scheduler._extract_strategy_realized_pnl = lambda: 0
        self.scheduler._refresh_runtime_pool_if_needed = lambda now, force=False: order.append("refresh")
        self.scheduler._quote_stream.start.side_effect = lambda symbols: order.append("stream")
        self.scheduler._fetch_balance_with_retry = Mock(
            side_effect=lambda *args, **kwargs: order.append("balance") or SimpleNamespace(
                total_deposit=1,
                total_eval_amount=1,
                total_profit_loss=0,
                positions=[],
            )
        )
        self.scheduler._is_trading_time = lambda now: False
        self.scheduler._write_performance_reports = lambda balance, session_pnl: None
        self.scheduler._resolve_session_profit_loss = lambda balance, strategy_pnl_baseline=None: 0
        self.scheduler._log_tick_activity = lambda now, watchlist, refresh_batch, quotes: None

        halted = self.scheduler._run_trading_session(10)

        self.assertFalse(halted)
        self.assertEqual(order[:6], ["initialize", "balance", "sync", "watchlist", "stream", "refresh"])
        self.assertEqual(
            self.scheduler._fetch_balance_with_retry.call_args_list,
            [
                call("세션 시작 잔고 조회", timeout=(10.0, 60.0)),
                call("세션 종료 잔고 조회", max_attempts=2, base_delay_seconds=1),
            ],
        )

    def test_run_trading_session_clears_stale_strategy_simulated_time_before_initialize(self):
        order = []
        self.scheduler.market_data = SimpleNamespace(is_market_open=lambda: True, refresh_instruments=Mock())
        self.scheduler.strategy = strategy_stub(
            set_simulated_now=lambda now: order.append(("set_simulated_now", now)),
            initialize=lambda: order.append("initialize"),
            get_watchlist=lambda: ["005930"],
            sync_positions_from_account=lambda positions: None,
            should_continue=lambda: False,
            positions={},
        )
        self.scheduler._alerts = Mock()
        self.scheduler._interruptible_sleep = lambda seconds: None
        self.scheduler._session_start_balance_delay_seconds = lambda: 0.0
        self.scheduler._extract_strategy_realized_pnl = lambda: 0
        self.scheduler._refresh_runtime_pool_if_needed = lambda now, force=False: False
        self.scheduler._fetch_balance_with_retry = lambda *args, **kwargs: SimpleNamespace(
            total_deposit=1,
            total_eval_amount=1,
            total_profit_loss=0,
            positions=[],
        )
        self.scheduler._is_trading_time = lambda now: False
        self.scheduler._write_performance_reports = lambda balance, session_pnl: None
        self.scheduler._resolve_session_profit_loss = lambda balance, strategy_pnl_baseline=None: 0
        self.scheduler._log_tick_activity = lambda now, watchlist, refresh_batch, quotes: None

        halted = self.scheduler._run_trading_session(10)

        self.assertFalse(halted)
        self.assertGreaterEqual(len(order), 2)
        self.assertEqual(order[0], ("set_simulated_now", None))
        self.assertEqual(order[1], "initialize")

    def test_run_trading_session_forces_runtime_pool_refresh_when_state_snapshot_exists(self):
        self.scheduler.market_data = SimpleNamespace(is_market_open=lambda: True, refresh_instruments=Mock())
        refresh_runtime_pool = Mock(return_value=False)
        self.scheduler.strategy = strategy_stub(
            initialize=lambda: None,
            get_watchlist=lambda: ["005930"],
            sync_positions_from_account=lambda positions: None,
            should_continue=lambda: False,
            has_runtime_state_snapshot=lambda: True,
            positions={},
        )
        self.scheduler._alerts = Mock()
        self.scheduler._interruptible_sleep = lambda seconds: None
        self.scheduler._session_start_balance_delay_seconds = lambda: 0.0
        self.scheduler._extract_strategy_realized_pnl = lambda: 0
        self.scheduler._refresh_runtime_pool_if_needed = refresh_runtime_pool
        self.scheduler._fetch_balance_with_retry = lambda *args, **kwargs: SimpleNamespace(
            total_deposit=1,
            total_eval_amount=1,
            total_profit_loss=0,
            positions=[],
        )
        self.scheduler._is_trading_time = lambda now: False
        self.scheduler._write_performance_reports = lambda balance, session_pnl: None
        self.scheduler._resolve_session_profit_loss = lambda balance, strategy_pnl_baseline=None: 0
        self.scheduler._log_tick_activity = lambda now, watchlist, refresh_batch, quotes: None

        halted = self.scheduler._run_trading_session(10)

        self.assertFalse(halted)
        self.assertTrue(refresh_runtime_pool.call_args.kwargs["force"])

    def test_log_tick_activity_reports_single_route_evaluation_count(self):
        now = datetime(2026, 4, 15, 9, 9, 0)
        self.scheduler.strategy = strategy_stub(
            positions={},
            _last_evaluation_count=1,
        )
        self.scheduler._last_tick_activity_log_at = None

        with patch("src.scheduler.logger.info") as info_mock:
            self.scheduler._log_tick_activity(
                now,
                ["046970", "001440"],
                ["046970"],
                [SimpleNamespace(symbol="046970")],
            )

        self.assertTrue(info_mock.called)
        args = info_mock.call_args[0]
        self.assertEqual(args[0], "틱 상태: 감시=%d 배치=%d 수신=%d 평가=%d 보유=%d")
        self.assertEqual(args[4], 1)

    def test_seconds_until_preopen_before_preopen_time(self):
        now = datetime(2026, 2, 27, 8, 49, 30)

        wait = self.scheduler._seconds_until_preopen(now)

        self.assertEqual(wait, 30)

    def test_seconds_until_preopen_during_preopen_waits_until_market_open(self):
        now = datetime(2026, 2, 27, 8, 50, 1)

        wait = self.scheduler._seconds_until_preopen(now)

        self.assertEqual(wait, 599)
        self.assertLess(wait, 3600)  # 다음날로 밀리지 않아야 한다.

    def test_is_trading_time_at_market_open_boundary(self):
        at_open = datetime(2026, 3, 5, 9, 0, 0)   # Thursday
        at_close = datetime(2026, 3, 5, 15, 30, 0)

        self.assertTrue(self.scheduler._is_trading_time(at_open))
        self.assertFalse(self.scheduler._is_trading_time(at_close))

    def test_is_trading_time_weekend_is_false(self):
        weekend_open = datetime(2026, 3, 7, 9, 0, 0)  # Saturday
        self.assertFalse(self.scheduler._is_trading_time(weekend_open))

    def test_is_trading_time_known_holiday_is_false(self):
        labor_day_open = datetime(2026, 5, 1, 9, 30, 0)  # Friday, KRX closed

        self.assertFalse(self.scheduler._is_trading_time(labor_day_open))

    def test_seconds_until_preopen_skips_known_holiday(self):
        labor_day = datetime(2026, 5, 1, 10, 0, 0)

        wait = self.scheduler._seconds_until_preopen(labor_day)

        self.assertEqual(labor_day + timedelta(seconds=wait), datetime(2026, 5, 4, 8, 50, 0))

    def test_next_preopen_after_skips_known_holiday(self):
        labor_day_morning = datetime(2026, 5, 1, 8, 0, 0)

        next_preopen = self.scheduler._next_preopen_after(labor_day_morning)

        self.assertEqual(next_preopen, datetime(2026, 5, 4, 8, 50, 0))

    def test_sleep_until_preopen_returns_at_next_day_preopen_after_hardstop(self):
        sleep_calls = []

        def fake_sleep(seconds):
            sleep_calls.append(seconds)

        self.scheduler._interruptible_sleep = fake_sleep
        _SequenceDateTime.timeline = [
            datetime(2026, 3, 9, 15, 15, 13),
            datetime(2026, 3, 9, 15, 15, 13),
            datetime(2026, 3, 10, 8, 45, 0),
            datetime(2026, 3, 10, 8, 50, 1),
        ]
        _SequenceDateTime.last = None

        with patch("src.scheduler.datetime", _SequenceDateTime):
            self.scheduler._sleep_until_preopen()

        self.assertEqual(sleep_calls, [1800, 300])

    def test_refresh_runtime_pool_if_needed_uses_fluctuation_symbols_for_discovery(self):
        now = datetime(2026, 4, 7, 9, 2, 0)
        update_runtime_pool = Mock()
        self.scheduler.strategy = strategy_stub(
            positions={},
            fixed_pool=[],
            config=SimpleNamespace(
                dynamic_pool_ranking_fetch_count=30,
                dynamic_pool_quote_min_change_rate=0.8,
                pool_refresh_interval=300,
            ),
            update_runtime_pool=update_runtime_pool,
        )
        self.scheduler.market_data = Mock()
        self.scheduler.market_data.get_fluctuation_symbols.return_value = [
            "009150",
            "000660",
        ]

        self.scheduler._refresh_runtime_pool_if_needed(now)
        self.scheduler._pool_read.result(timeout=2)
        self.scheduler._collect_market_reads()
        self.scheduler.market_data.get_fluctuation_symbols.assert_called_once()
        update_runtime_pool.assert_called_once_with(["009150", "000660"])

    def test_runtime_pool_lookup_does_not_skip_supplemental_quotes(self):
        now = datetime(2026, 4, 7, 9, 2, 0)
        self.scheduler.strategy = strategy_stub(
            positions={},
            fixed_pool=[],
            config=SimpleNamespace(
                quote_freshness_seconds=5,
                dynamic_pool_ranking_fetch_count=30,
                dynamic_pool_quote_min_change_rate=0.0,
                pool_refresh_interval=300,
            ),
            update_runtime_pool=Mock(),
        )
        self.scheduler.market_data = Mock()
        self.scheduler.market_data.get_fluctuation_symbols.return_value = ["009150"]
        self.scheduler.market_data.get_multi_price.return_value = []

        self.scheduler._refresh_runtime_pool_if_needed(now)
        self.scheduler._schedule_market_reads(
            ["009150", "005930"],
            ["009150"],
            now,
        )
        self.scheduler._market_read.result(timeout=2)
        self.scheduler._collect_market_reads()
        self.scheduler.market_data.get_multi_price.assert_called_once_with(["005930"])

    def test_orders_precede_new_background_reads_and_reads_resume_after_completion(self):
        for side in (OrderSide.BUY, OrderSide.SELL):
            with self.subTest(side=side):
                entered, release = Event(), Event()
                events, evaluations = [], []
                quote = Quote(symbol="005930", current_price=10000)
                self.scheduler._shutdown = False
                self.scheduler._pending_order_blocks.clear()

                def evaluate(quotes):
                    events.append("evaluate")
                    evaluations.append(list(quotes))
                    return [Order(symbol="005930", side=side, quantity=1)] if len(evaluations) == 1 else []

                def submit(orders):
                    events.append("submit")
                    entered.set()
                    if not release.wait(3):
                        raise requests.exceptions.Timeout("test order was not released")
                    return []

                def refresh(now, *, force=False):
                    if not force:
                        events.append("discovery")
                    return False

                def read(symbols):
                    events.append("quotes")
                    return []

                def end_tick(seconds):
                    if len(evaluations) <= 2:
                        self.assertTrue(entered.wait(2))
                        self.scheduler.market_data.get_multi_price.assert_not_called()
                        self.assertNotIn("discovery", events)
                    if len(evaluations) == 2:
                        release.set()
                        self.scheduler._order_execution.result(timeout=2)
                    if len(evaluations) >= 3 and self.scheduler._market_read is not None:
                        self.scheduler._market_read.result(timeout=2)
                    if len(evaluations) >= 3 or self.scheduler._send_loop_error_alert.called:
                        self.scheduler._shutdown = True
                        release.set()

                self.scheduler.market_data.is_market_open = lambda: True
                self.scheduler.market_data.get_multi_price = Mock(side_effect=read)
                self.scheduler.strategy = strategy_stub(
                    initialize=Mock(), get_watchlist=lambda: ["005930", "000660"],
                    on_batch_tick=Mock(side_effect=evaluate))
                self.scheduler._quote_stream.subscribed_symbols.return_value = ["005930"]
                self.scheduler._quote_stream.drain_quotes.return_value = [quote]
                self.scheduler._refresh_runtime_pool_if_needed = Mock(side_effect=refresh)
                self.scheduler.executor = Mock(submit_orders=Mock(side_effect=submit))
                self.scheduler._handle_order_results = Mock()
                self.scheduler._send_loop_error_alert = Mock()
                self.scheduler._session_start_balance_delay_seconds = lambda: 0
                self.scheduler._fetch_balance_with_retry = lambda *a, **kw: SimpleNamespace(
                    total_deposit=1000000, total_eval_amount=1000000, total_profit_loss=0, positions=[])
                self.scheduler._confirm_reconciled_sell_prices = Mock()
                self.scheduler._resolve_session_profit_loss = Mock(return_value=0)
                self.scheduler._write_performance_reports = Mock()
                self.scheduler._send_session_summary_alert = Mock()
                self.scheduler._is_trading_time = lambda now: not self.scheduler._shutdown
                self.scheduler._interruptible_sleep = end_tick
                try:
                    self.scheduler._run_trading_session(1)
                finally:
                    release.set()

                self.assertEqual(evaluations, [[quote], [quote], [quote]])
                self.assertEqual(events, ["evaluate", "submit", "evaluate", "evaluate", "discovery", "quotes"])
                self.scheduler.market_data.get_multi_price.assert_called_once_with(["000660"])
                self.scheduler.executor.submit_orders.assert_called_once()
                self.scheduler._send_loop_error_alert.assert_not_called()

                self.scheduler._shutdown = False
                self.scheduler._account_read = Future()
                self.scheduler.market_data.get_multi_price.reset_mock()
                self.scheduler._refresh_runtime_pool_if_needed.reset_mock()
                self.scheduler._schedule_market_reads(["005930", "000660"], ["005930"], datetime.now())
                self.scheduler.market_data.get_multi_price.assert_not_called()
                self.scheduler._refresh_runtime_pool_if_needed.assert_not_called()
                self.scheduler._account_read = None
                self.scheduler._schedule_market_reads(["005930", "000660"], ["005930"], datetime.now())
                self.scheduler._market_read.result(timeout=2)
                self.scheduler._collect_market_reads()
                self.scheduler.market_data.get_multi_price.assert_called_once_with(["000660"])
                self.scheduler._stop_quote_reads()

    def test_blocked_rest_and_order_io_do_not_block_batch_ticks(self):
        entered, order_entered, release = Event(), Event(), Event()
        main_thread = get_ident()
        evaluations, submissions = [], []
        quote = Quote(symbol="005930", current_price=10000, ask_price=10005, bid_price=9995)

        def blocked_read(symbols):
            entered.set()
            if not release.wait(3):
                raise requests.exceptions.Timeout("test read was not released")
            return []

        def evaluate(quotes):
            evaluations.append((get_ident(), entered.is_set(), release.is_set(), list(quotes)))
            return [Order(symbol="005930", side=OrderSide.SELL, quantity=1)]

        def submit(orders):
            submissions.append((get_ident(), release.is_set()))
            order_entered.set()
            if not release.wait(3):
                raise requests.exceptions.Timeout("test order was not released")
            return []

        def end_tick(seconds):
            if evaluations:
                self.assertTrue(order_entered.wait(2))
            if len(evaluations) >= 3 or self.scheduler._send_loop_error_alert.called:
                self.scheduler._shutdown = True
                release.set()

        self.scheduler.market_data.is_market_open = lambda: True
        self.scheduler.market_data.get_multi_price = Mock(side_effect=blocked_read)
        self.scheduler.strategy = strategy_stub(
            fixed_pool=["005930", "000660"],
            initialize=Mock(), get_watchlist=lambda: ["005930", "000660"],
            on_batch_tick=Mock(side_effect=evaluate),
            should_continue=lambda: not order_entered.is_set(),
        )
        self.scheduler._quote_stream.subscribed_symbols.return_value = ["005930"]
        def drain():
            self.assertTrue(entered.wait(2))
            return [quote] if not evaluations else []

        self.scheduler._quote_stream.drain_quotes.side_effect = drain
        self.scheduler.executor = Mock(submit_orders=Mock(side_effect=submit))
        self.scheduler._handle_order_results = Mock()
        self.scheduler._send_loop_error_alert = Mock()
        self.scheduler._session_start_balance_delay_seconds = lambda: 0
        self.scheduler._fetch_balance_with_retry = lambda *a, **kw: SimpleNamespace(
            total_deposit=1000000, total_eval_amount=1000000, total_profit_loss=0, positions=[])
        self.scheduler._confirm_reconciled_sell_prices = Mock()
        self.scheduler._resolve_session_profit_loss = Mock(return_value=0)
        self.scheduler._write_performance_reports = Mock()
        self.scheduler._send_session_summary_alert = Mock()
        self.scheduler._is_trading_time = lambda now: not self.scheduler._shutdown
        self.scheduler._interruptible_sleep = end_tick
        # Simulate a GET already in flight from the preceding tick.
        self.scheduler._market_reader = ThreadPoolExecutor(max_workers=4)
        self.scheduler._market_read = self.scheduler._market_reader.submit(
            self.scheduler._read_supplemental_quotes, ["000660"])
        try:
            self.scheduler._run_trading_session(1)
        finally:
            release.set()

        self.assertEqual(evaluations, [
            (main_thread, True, False, [quote]),
            (main_thread, True, False, []),
            (main_thread, True, False, []),
        ])
        self.assertEqual(len(submissions), 1)
        self.assertTrue(all(thread != main_thread for thread, _ in submissions))
        self.assertFalse(submissions[0][1])
        self.scheduler.market_data.get_multi_price.assert_called_once_with(["000660"])
        self.scheduler._send_loop_error_alert.assert_not_called()
        self.assertIsNone(self.scheduler._market_reader)
        self.assertIsNone(self.scheduler._market_read)
        self.assertIsNone(self.scheduler._pool_read)

    def test_inflight_order_preserves_real_strategy_risk_and_main_thread_fill_contract(self):
        main_thread = get_ident()
        entered, release = Event(), Event()
        self.addCleanup(release.set)
        with tempfile.TemporaryDirectory() as directory:
            s = MomentumScalpStrategy(None, MomentumScalpConfig(
                daily_state_path=f"{directory}/state.json", quote_archive_root=f"{directory}/quotes",
                commission_rate=.000142, sell_tax_rate=.0015))
            now = datetime(2026, 10, 6, 10)
            s.set_simulated_now(now)
            s._ensure_active_day()
            s._ingest_quotes([Quote(symbol="005930", current_price=10000+(i//4+1)*20,
                timestamp=now + timedelta(seconds=i), bid_price=9990+(i//4+1)*20,
                ask_price=10010+(i//4+1)*20,
                book_available=True, book_depth_available=True, ask_size=100, bid_size=100)
                for i in range(-60, 0)])
            q = Quote(symbol="005930", current_price=10030, timestamp=now,
                      bid_price=10020, ask_price=10040, book_available=True,
                      book_depth_available=True, ask_size=1, bid_size=1)
            buy = s.on_batch_tick([q])
            self.assertEqual(len(buy), 1)
            self.assertGreater(buy[0].quantity, q.ask_size)
            self.scheduler.strategy = s
            callback_threads = []
            apply_fill = s.on_order_filled

            def on_fill(result):
                callback_threads.append((get_ident(), result.quantity))
                apply_fill(result)

            s.on_order_filled = Mock(side_effect=on_fill)
            workers = []

            def submit(orders):
                workers.append(get_ident())
                entered.set()
                if not release.wait(3):
                    raise requests.exceptions.Timeout("test order was not released")
                o = orders[0]
                return [OrderResult(True, order_no=f"{o.side.value}-confirmed",
                    symbol=o.symbol, side=o.side, quantity=o.quantity,
                    price=o.price or o.reference_price, requested_quantity=o.quantity,
                    requested_reason=o.requested_reason, fill_mode="filled")]

            self.scheduler.executor = SimpleNamespace(submit_orders=submit)
            try:
                self.assertTrue(self.scheduler._start_order_execution(buy))
                self.assertTrue(entered.wait(2))
                self.assertFalse(self.scheduler._start_order_execution(buy))
                self.assertEqual(self.scheduler._collect_order_execution(), [])
                self.assertFalse(self.scheduler._reconcile_pending_orders(wait=True))
                self.assertFalse(self.scheduler._reconcile_pending_orders(wait=True))
                self.assertEqual(s.positions, {})
                self.assertGreater(s._pending_entry_risk(), 0)
                self.assertEqual(s.daily_pnl.realized_net_pnl, 0)
                self.scheduler._alerts.send.assert_not_called()
                stop_price = s._pending_entry_meta[q.symbol]["plan"].stop_price
                for i in range(1, 4):
                    s.set_simulated_now(now + timedelta(seconds=i))
                    q = replace(q, timestamp=s._now(), bid_price=stop_price)
                    self.assertEqual(s.on_batch_tick([q]), [])
                release.set()
                self.scheduler._order_execution.result(timeout=2)
                self.scheduler._collect_order_execution()
                self.assertEqual(s.positions[q.symbol].quantity, buy[0].quantity)
                self.assertEqual(s._pending_entry_risk(), 0)
                self.assertEqual(self.scheduler._alerts.send.call_count, 1)
                self.assertEqual(self.scheduler._collect_order_execution(), [])
                sell = s.on_batch_tick([q])
                self.assertEqual(sell[0].requested_reason, "breakout_failed")
                entered.clear()
                release.clear()
                self.assertTrue(self.scheduler._start_order_execution(sell))
                self.assertTrue(entered.wait(2))
                self.assertIsNotNone(s.positions[q.symbol].pending_exit_started_at)
                self.assertGreater(s._open_committed_risk(), 0)
                self.assertEqual(s.on_batch_tick([q]), [])
                self.assertEqual(self.scheduler._filter_pending_duplicate_orders(sell), [])
                self.assertEqual(self.scheduler._alerts.send.call_count, 1)
                release.set()
                # Shutdown must apply the late result once, not discard or cancel it.
                self.scheduler._stop_market_reads()
                self.assertEqual(s.positions, {})
                self.assertEqual(s.daily_pnl.trade_count, 1)
                self.assertEqual(len(s._sell_fill_ledger), 1)
                self.assertEqual(self.scheduler._alerts.send.call_count, 2)
                self.assertEqual([qty for _, qty in callback_threads], [0, buy[0].quantity, 0, buy[0].quantity])
                self.assertTrue(all(t == main_thread for t, _ in callback_threads))
                self.assertTrue(all(t != main_thread for t in workers))
            finally:
                release.set()
                self.scheduler._stop_market_reads()

    def test_order_worker_failure_remains_unknown_and_dispatch_failure_is_not_accepted(self):
        order = Order(symbol="005930", side=OrderSide.BUY, quantity=1, price=10000)
        self.scheduler._handle_order_results = Mock()
        pending = Future()
        pending.set_exception(RuntimeError("unknown worker outcome"))
        self.scheduler._order_execution = pending
        self.scheduler._executing_orders = [order]
        with self.assertLogs("kis_trader.scheduler", level="ERROR"):
            result = self.scheduler._collect_order_execution()[0]
        self.assertTrue(result.success)
        self.assertEqual(result.quantity, 0)
        self.assertEqual(result.price, 0)
        self.assertEqual(result.fill_mode, "order_result_pending")
        self.assertIsNone(self.scheduler._order_execution)
        self.assertEqual(self.scheduler._collect_order_execution(), [])

        self.scheduler._market_reader = Mock()
        self.scheduler._market_reader.submit.side_effect = RuntimeError("worker shut down")
        self.scheduler.executor = Mock()
        with self.assertRaisesRegex(RuntimeError, "worker shut down"):
            self.scheduler._start_order_execution([order])
        failed = self.scheduler._handle_order_results.call_args.args[0][0]
        self.assertFalse(failed.success)
        self.assertEqual(failed.quantity, 0)
        self.assertEqual(self.scheduler._executing_orders, [])

    def test_account_and_alert_waits_do_not_stop_the_real_strategy_batch_loop(self):
        account_entered, alert_entered, release = Event(), Event(), Event()
        self.addCleanup(release.set)
        main_thread = get_ident()
        with tempfile.TemporaryDirectory() as directory:
            now = datetime.now().replace(hour=10, minute=0, second=0, microsecond=0)
            s = MomentumScalpStrategy(None, MomentumScalpConfig(
                daily_state_path=f"{directory}/state.json", quote_archive_root=f"{directory}/quotes"))
            s.set_simulated_now(now)
            s._ensure_active_day()
            q = Quote("005930", 10030, timestamp=now, bid_price=10020, ask_price=10040,
                      book_available=True, book_depth_available=True, bid_size=10, ask_size=10)
            s._ingest_quotes([replace(q, timestamp=now+timedelta(seconds=i),
                current_price=10000+(i//4+1)*20, bid_price=9990+(i//4+1)*20,
                ask_price=10010+(i//4+1)*20) for i in range(-60, 0)])
            buy = s.on_batch_tick([q])[0]
            s._pending_entry_meta[q.symbol]["created_at"] = now-timedelta(seconds=5)
            s._pending_entry_meta[q.symbol]["order_no"] = "unknown-fill"
            risk = s._pending_entry_risk()
            s.initialize = Mock()
            evaluate = s.on_batch_tick
            evaluations = []

            def tick(quotes):
                evaluations.append(get_ident())
                return evaluate(quotes)

            s.on_batch_tick = tick
            self.scheduler.strategy = s
            del self.scheduler._send_alert
            self.scheduler.market_data.is_market_open = lambda: True
            self.scheduler._refresh_runtime_pool_if_needed = lambda *a, **kw: False
            self.scheduler._reset_strategy_simulated_clock = lambda: None
            self.scheduler._session_start_balance_delay_seconds = lambda: 0
            balance = SimpleNamespace(total_deposit=1000000, total_eval_amount=1000000,
                                      total_profit_loss=0, positions=[])

            def fetch(context, **kwargs):
                if context == "주문 상태 재동기화":
                    account_entered.set()
                    self.assertNotEqual(get_ident(), main_thread)
                    if not release.wait(3):
                        raise requests.exceptions.Timeout("blocked account fixture")
                return balance

            def alert(**kwargs):
                alert_entered.set()
                self.assertNotEqual(get_ident(), main_thread)
                if not release.wait(3):
                    raise requests.exceptions.Timeout("blocked alert fixture")

            def end_tick(seconds):
                self.assertTrue(account_entered.wait(2))
                self.assertTrue(alert_entered.wait(2))
                if len(evaluations) >= 3:
                    self.scheduler._shutdown = True
                    release.set()

            self.scheduler._fetch_balance_with_retry = fetch
            self.scheduler._alerts.send.side_effect = alert
            self.scheduler._quote_stream.subscribed_symbols.return_value = [q.symbol]
            self.scheduler._quote_stream.drain_quotes.return_value = [q]
            self.scheduler.executor = Mock()
            self.scheduler._is_trading_time = lambda _: not self.scheduler._shutdown
            self.scheduler._interruptible_sleep = end_tick
            self.scheduler._write_performance_reports = Mock()
            self.scheduler._resolve_session_profit_loss = Mock(return_value=0)
            self.scheduler._send_loop_error_alert = Mock()
            try:
                self.scheduler._run_trading_session(1)
                self.assertEqual(evaluations, [main_thread]*3)
                self.scheduler.executor.submit_orders.assert_not_called()
                self.scheduler._send_loop_error_alert.assert_not_called()
                self.assertEqual(s.positions, {})
                self.assertEqual(s._pending_entry_risk(), risk)
                self.assertEqual(s._pending_entry_meta[q.symbol]["plan"].quantity, buy.quantity)
                self.assertEqual(s.daily_pnl.trade_count, 0)
            finally:
                release.set()
                self.scheduler._stop_market_reads()
                self.scheduler._stop_alerts()

    def test_unsent_buy_proposal_cannot_leave_a_phantom_pending_order(self):
        pending = {"005930": {"order_no": ""}}
        applied = Mock(side_effect=lambda r: pending.pop(r.symbol, None))
        self.scheduler.strategy = strategy_stub(_pending_entry_meta=pending, on_order_filled=applied)
        self.scheduler._pending_order_blocks[("005930", "buy")] = {
            "expires_at": datetime.now()+timedelta(seconds=180)}
        self.scheduler.executor = Mock()
        order = Order("005930", OrderSide.BUY, quantity=1, price=10000)
        self.assertFalse(self.scheduler._submit_strategy_orders([order]))
        self.assertEqual(pending, {})
        self.scheduler.executor.submit_orders.assert_not_called()
        self.assertFalse(applied.call_args.args[0].success)

    def test_real_order_path_rechecks_book_after_wait_and_releases_invalid_entry(self):
        for change in ("fallen", "reversal", "later_second_reversal", "empty_depth", "thin_depth", "valid"):
            rejected = change not in {"thin_depth", "valid"}
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                entered, release = Event(), Event()
                config = SimpleNamespace(
                    is_paper=True, trading_mode="paper", base_url="https://example.invalid",
                    api_key="test-key", api_secret="test-secret", account_number="12345678",
                    account_product_code="01", rate_limit_interval=0.0,
                    request_connect_timeout=1.0, request_read_timeout=1.0)
                client = KISClient(config, SimpleNamespace(get_token=lambda: "test", last_token_issue_at=0))
                s = MomentumScalpStrategy(SimpleNamespace(client=client), MomentumScalpConfig(
                    daily_state_path=f"{directory}/state.json", quote_archive_root=f"{directory}/quotes"))
                now = datetime(2026, 10, 7, 10)
                s.set_simulated_now(now)
                s._ensure_active_day()
                q = Quote("005930", 10030, now, bid_price=10020, ask_price=10040,
                          book_available=True, book_depth_available=True, bid_size=3, ask_size=3)
                s._ingest_quotes([replace(q, timestamp=now+timedelta(seconds=i),
                    current_price=10000+(i//4+1)*20, bid_price=9990+(i//4+1)*20,
                    ask_price=10010+(i//4+1)*20) for i in range(-60, 0)])

                def response(payload):
                    return Mock(status_code=200, headers={}, text="fixture", json=Mock(return_value=payload))

                def confirmed_history(*args, **kwargs):
                    filled = 2 if change == "thin_depth" else order.quantity
                    return response({"rt_cd": "0", "output1": [{
                        "odno": "known", "pdno": q.symbol, "tot_ccld_qty": str(filled),
                        "avg_prvs": "10035", "rmn_qty": "0",
                        "cnc_cfrm_qty": str(order.quantity - filled)}]})

                session = Mock(
                    post=Mock(return_value=response({"rt_cd": "0", "output": {"ODNO": "known"}})),
                    get=Mock(side_effect=confirmed_history))
                client._session = session
                prepare = client._prepare_request

                def wait_for_new_book(tr_id, tr_cont, **kwargs):
                    if tr_id == "TTTC0012U":
                        entered.set()
                        if not release.wait(3):
                            raise AssertionError("quote update fixture timed out")
                    return prepare(tr_id, tr_cont, **kwargs)

                client._prepare_request = wait_for_new_book
                with (patch("src.scheduler.AlertManager", return_value=Mock()),
                      patch("src.api_client.requests.Session", return_value=session),
                      patch("src.api_client.time.time", side_effect=lambda: s._now().timestamp()),
                      patch("src.api_client.time.sleep", side_effect=lambda seconds:
                            s.set_simulated_now(s._now() + timedelta(seconds=seconds))),
                      patch("src.market_stream.datetime", wraps=datetime) as clock):
                    clock.now.side_effect = s._now
                    scheduler = TradingScheduler(s)
                    scheduler._send_alert = scheduler._alerts.send
                    try:
                        scheduler._quote_stream.observe_quotes([q])
                        order = s.on_batch_tick([q])[0]
                        self.assertGreater(order.quantity, q.ask_size)
                        plan = s._pending_entry_meta[q.symbol]["plan"]
                        self.assertLessEqual(plan.committed_risk_net_abs, 5000)
                        self.assertTrue(scheduler._submit_strategy_orders([order]))
                        self.assertTrue(entered.wait(2))
                        s.set_simulated_now(now + timedelta(seconds=1))
                        latest = replace(q, timestamp=s._now(),
                            current_price=10020 if change == "later_second_reversal" else 10030,
                            bid_price=10000 if change == "fallen" else 10020,
                            ask_price=10010 if change == "fallen" else 10035,
                            ask_size=0 if change == "empty_depth" else 2 if change == "thin_depth" else 100,
                            bid_size=1 if change == "thin_depth" else 100,
                            trade_high_price=(10040 if change == "reversal"
                                              else 10020 if change == "later_second_reversal" else 10030))
                        scheduler._quote_stream.observe_quotes([latest])
                        self.assertEqual(s.on_batch_tick([latest]), [])
                        release.set()
                        scheduler._order_execution.result(timeout=2)
                        result = scheduler._collect_order_execution()[0]
                        self.assertEqual(result.success, not rejected)
                        self.assertEqual(s._pending_entry_risk(), 0)
                        self.assertEqual(scheduler._pending_order_blocks, {})
                        if rejected:
                            self.assertEqual(result.error_code, "LOCAL_ORDER_INVALIDATED")
                            self.assertEqual(result.error_category, "stale_order")
                            self.assertEqual(s.positions, {})
                            session.post.assert_not_called()
                            session.get.assert_not_called()
                            scheduler._alerts.send.assert_not_called()
                            s.set_simulated_now(now + timedelta(seconds=2))
                            next_q = replace(q, timestamp=s._now(), current_price=10060,
                                             bid_price=10050, ask_price=10060)
                            self.assertEqual(len(s.on_batch_tick([next_q])), 1)
                        else:
                            self.assertEqual(session.post.call_count, 2)  # Buy and cancel only.
                            submitted = json.loads(session.post.call_args_list[0].kwargs["data"])
                            self.assertEqual((submitted["ORD_QTY"], submitted["ORD_UNPR"]),
                                             (str(order.quantity), str(order.price)))
                            self.assertEqual(s.positions[q.symbol].quantity,
                                             2 if change == "thin_depth" else order.quantity)
                            self.assertEqual(s.positions[q.symbol].buy_price, 10035)
                            self.assertLessEqual(s._open_committed_risk(), plan.committed_risk_net_abs)
                            self.assertEqual(s.positions[q.symbol].plan.expires_at, plan.expires_at)
                            self.assertEqual(scheduler._alerts.send.call_count, 1)
                            self.assertIn(f"{s.positions[q.symbol].quantity}주 @ 10,035원",
                                          scheduler._alerts.send.call_args.kwargs["message"])

                        before = session.post.call_count
                        scheduler.trading._quote_provider = lambda _: None
                        sell = scheduler.trading._place_standard_order(
                            Order(q.symbol, OrderSide.SELL, quantity=3), resolve_fill=False)
                        self.assertTrue(sell.success)
                        self.assertEqual(session.post.call_count, before + 1)
                    finally:
                        release.set()
                        scheduler._stop_market_reads()
                        scheduler._stop_alerts()

    def test_repeated_fills_do_not_duplicate_inventory_alerts_or_revive_a_closed_sell(self):
        from src.strategies.momentum_scalp_types import TradePlan

        with tempfile.TemporaryDirectory() as directory:
            now = datetime.now()
            s = MomentumScalpStrategy(None, MomentumScalpConfig(
                daily_state_path=f"{directory}/state.json", quote_archive_root=f"{directory}/quotes"))
            s.set_simulated_now(now)
            s._ensure_active_day()
            symbol = "005930"
            plan = TradePlan(symbol, True, "price_breakout", 5, 10010, 9950, 800,
                             now, now + timedelta(seconds=180))
            s._pending_entry_meta[symbol] = dict(plan=plan, created_at=now, order_no="buy")
            self.scheduler.strategy = s
            self.scheduler._reconcile_pending_orders = Mock()
            partial = OrderResult(True, symbol=symbol, side=OrderSide.BUY, order_no="buy",
                quantity=2, price=10010, requested_quantity=5, fill_mode="partial_fill_pending")
            self.scheduler._handle_order_results([replace(partial)])
            self.scheduler._handle_order_results([replace(partial)])
            self.assertEqual(s.positions[symbol].quantity, 2)
            self.assertEqual(self.scheduler._alerts.send.call_count, 1)

            # Persisted cumulative quantity, not an in-memory seen-ID set, survives restart.
            s._save_daily_state()
            s._load_daily_state()
            self.scheduler._handle_order_results([replace(partial)])
            self.assertEqual(s.positions[symbol].quantity, 2)
            self.assertEqual(self.scheduler._alerts.send.call_count, 1)
            final = replace(partial, quantity=5, fill_mode="limit")
            self.scheduler._handle_order_results([replace(final)])
            self.scheduler._handle_order_results([replace(final)])
            self.assertEqual(s.positions[symbol].quantity, 5)
            self.assertFalse(s._pending_entry_meta)
            self.assertNotIn((symbol, "buy"), self.scheduler._pending_order_blocks)
            self.assertEqual(self.scheduler._alerts.send.call_count, 2)
            self.assertIn("3주", self.scheduler._alerts.send.call_args.kwargs["message"])

            sell = OrderResult(True, symbol=symbol, side=OrderSide.SELL, order_no="sell",
                quantity=2, price=10020, requested_quantity=2, fill_mode="market")
            self.scheduler._handle_order_results([replace(sell)])
            self.scheduler._handle_order_results([replace(sell, quantity=0, price=0,
                                                          fill_mode="market_pending")])
            net = s.daily_pnl.realized_net_pnl
            self.scheduler._handle_order_results([replace(sell)])
            self.assertEqual(s.positions[symbol].quantity, 3)
            self.assertEqual(s.daily_pnl.realized_net_pnl, net)
            self.assertFalse(s._pending_sell_fills)
            self.assertIsNone(s.positions[symbol].pending_exit_started_at)
            self.assertEqual(self.scheduler._alerts.send.call_count, 3)

            partial_sell = replace(sell, order_no="sell-2", quantity=1, requested_quantity=3)
            self.scheduler._handle_order_results([replace(partial_sell)])
            net = s.daily_pnl.realized_net_pnl
            for reply in (replace(partial_sell), replace(sell), replace(sell, success=False)):
                self.scheduler._handle_order_results([reply])
            self.assertEqual(s.positions[symbol].quantity, 2)
            self.assertEqual(s.daily_pnl.realized_net_pnl, net)
            self.assertEqual(s._pending_sell_fills[symbol]["order_no"], "sell-2")
            self.assertEqual(self.scheduler._alerts.send.call_count, 4)
            s._save_daily_state()
            s._load_daily_state()
            self.scheduler._handle_order_results([replace(partial_sell)])
            self.scheduler._handle_order_results([replace(sell, quantity=0, price=0,
                                                          fill_mode="market_pending")])
            self.assertEqual(self.scheduler._alerts.send.call_count, 4)
            self.assertEqual(s.positions[symbol].quantity, 2)

            self.scheduler.account._coerce_int.side_effect = lambda value: int(value or 0)
            history = Mock(empty=False)
            history.to_dict.return_value = [dict(odno="sell-2", pdno=symbol,
                tot_ccld_qty=3, tot_ccld_amt=30060, avg_prvs=10020)]
            self.scheduler._confirm_reconciled_sell_prices(history=history)
            self.assertFalse(s.positions)
            self.assertFalse(s._pending_sell_fills)
            self.assertEqual(sum(r["quantity"] for r in s._sell_fill_ledger), 5)
            self.assertEqual(self.scheduler._alerts.send.call_count, 5)
            self.scheduler._handle_order_results([replace(partial_sell)])
            self.assertFalse(s._pending_sell_fills)
            self.assertEqual(self.scheduler._alerts.send.call_count, 5)

    def test_urgent_sell_overtakes_account_read_without_resurrecting_sold_position(self):
        from src.models import Position
        import pandas as pd

        entered, release = Event(), Event()
        self.addCleanup(release.set)
        with tempfile.TemporaryDirectory() as directory:
            now = datetime.now().replace(hour=10, minute=0, second=0, microsecond=0)
            s = MomentumScalpStrategy(None, MomentumScalpConfig(
                daily_state_path=f"{directory}/state.json", quote_archive_root=f"{directory}/quotes"))
            s.set_simulated_now(now)
            s._ensure_active_day()
            q = Quote("005930", 10030, timestamp=now, bid_price=10020, ask_price=10040,
                      book_available=True, book_depth_available=True, bid_size=1, ask_size=1)
            s._ingest_quotes([replace(q, timestamp=now+timedelta(seconds=i),
                current_price=10000+(i//4+1)*20, bid_price=9990+(i//4+1)*20,
                ask_price=10010+(i//4+1)*20) for i in range(-60, 0)])
            buy = s.on_batch_tick([q])[0]
            s.on_order_filled(OrderResult(True, symbol=q.symbol, side=OrderSide.BUY,
                order_no="buy", quantity=1, price=buy.price, requested_quantity=1, fill_mode="filled"))
            self.scheduler.strategy = s
            calls = []

            def balance(*args, **kwargs):
                calls.append(get_ident())
                if len(calls) == 1:
                    entered.set()
                    if not release.wait(3):
                        raise requests.exceptions.Timeout("account fixture not released")
                    return SimpleNamespace(positions=[Position(q.symbol, 1, buy.price)])
                return SimpleNamespace(positions=[])

            self.scheduler._fetch_balance_with_retry = balance
            self.scheduler.account.get_order_history.return_value = pd.DataFrame()
            self.scheduler.executor = SimpleNamespace(submit_orders=lambda orders: [
                OrderResult(True, symbol=o.symbol, side=o.side, order_no="sell",
                    quantity=o.quantity, price=o.reference_price, requested_quantity=o.quantity,
                    requested_reason=o.requested_reason, fill_mode="filled") for o in orders])
            sync = s.sync_positions_from_account = Mock(wraps=s.sync_positions_from_account)
            try:
                self.scheduler._reconcile_pending_orders([OrderResult(False, symbol="000660",
                    side=OrderSide.SELL, error_category="no_holding")])
                self.assertTrue(entered.wait(2))
                stop = s.on_batch_tick([replace(q, bid_price=s.positions[q.symbol].plan.stop_price)])
                self.assertEqual(stop[0].requested_reason, "breakout_failed")
                self.assertTrue(self.scheduler._submit_strategy_orders(stop))
                self.scheduler._order_execution.result(timeout=2)
                self.scheduler._collect_order_execution()
                self.assertEqual(s.positions, {})
                release.set()
                self.scheduler._reconcile_pending_orders(wait=True)
                self.assertEqual(s.positions, {})
                sync.assert_not_called()
                self.assertEqual(len(s._sell_fill_ledger), 1)
                self.scheduler._stop_market_reads()
                self.assertEqual(s.positions, {})
            finally:
                release.set()
                self.scheduler._stop_market_reads()

    def test_market_read_results_apply_once_on_main_thread_without_refreshing_quote_time(self):
        now = datetime(2026, 9, 18, 13, 20)
        quote = Quote(symbol="005930", current_price=10000, timestamp=now)
        threads = []
        self.scheduler.strategy.update_runtime_pool = Mock(side_effect=lambda symbols: threads.append(get_ident()))
        self.scheduler.market_data.get_fluctuation_symbols = Mock(return_value=["009150"])
        self.scheduler.market_data.get_multi_price = Mock(return_value=[quote])

        self.scheduler._schedule_market_reads(["005930"], [], now)
        pending = self.scheduler._market_read
        pending.result(timeout=2)
        self.scheduler._pool_read.result(timeout=2)
        self.scheduler.strategy.update_runtime_pool.assert_not_called()
        self.scheduler._schedule_market_reads(["000660"], [], now + timedelta(seconds=1))
        self.assertIs(self.scheduler._market_read, pending)
        self.assertEqual(self.scheduler._collect_market_reads(), [quote])
        self.assertEqual(quote.timestamp, now)
        self.assertEqual(threads, [get_ident()])
        self.scheduler.strategy.update_runtime_pool.assert_called_once_with(["009150"])
        self.assertEqual(self.scheduler._collect_market_reads(), [])
        self.scheduler.market_data.get_multi_price.assert_called_once_with(["005930"])

        self.scheduler._schedule_market_reads(["005930"], [], now + timedelta(minutes=5))
        self.scheduler._market_read.result(timeout=2)
        self.scheduler._pool_read.result(timeout=2)
        self.scheduler._stop_market_reads()
        self.assertEqual(self.scheduler._collect_market_reads(), [])
        self.assertIsNone(self.scheduler._pool_read)
        self.scheduler.strategy.update_runtime_pool.assert_called_once()

    def test_market_read_network_error_is_collected_without_alert_or_loop_backoff(self):
        self.scheduler.strategy.fixed_pool = ["005930"]
        for error in (requests.exceptions.Timeout("test timeout"), ValueError("bad REST payload")):
            with self.subTest(error=type(error).__name__):
                self.scheduler.market_data.get_multi_price = Mock(side_effect=error)
                self.scheduler._schedule_market_reads(["005930"], [], datetime.now())
                with self.assertRaises(type(error)):
                    self.scheduler._market_read.result(timeout=2)
                with self.assertLogs("kis_trader.scheduler", level="WARNING"):
                    self.assertEqual(self.scheduler._collect_market_reads(), [])
                self.assertIsNone(self.scheduler._market_read)
                self.scheduler._alerts.send.assert_not_called()

    def test_slow_pool_read_does_not_block_successive_quote_reads(self):
        now = datetime(2026, 9, 23, 13)
        entered, release = Event(), Event()
        quote = Quote(symbol="005930", current_price=10000, timestamp=now)
        applied_threads = []

        def slow_pool(**kwargs):
            entered.set()
            if not release.wait(5):
                raise AssertionError("ranking test was not released")
            return ["009150"]

        self.scheduler.strategy.update_runtime_pool = Mock(
            side_effect=lambda symbols: applied_threads.append(get_ident()))
        self.scheduler.market_data.get_fluctuation_symbols = Mock(side_effect=slow_pool)
        self.scheduler.market_data.get_multi_price = Mock(return_value=[quote])
        try:
            self.scheduler._schedule_market_reads([quote.symbol], [], now)
            self.assertTrue(entered.wait(2))
            self.scheduler._market_read.result(timeout=2)
            self.assertEqual(self.scheduler._collect_market_reads(), [quote])
            self.scheduler.strategy.update_runtime_pool.assert_not_called()

            self.scheduler._schedule_market_reads([quote.symbol], [], now + timedelta(seconds=2))
            self.scheduler._market_read.result(timeout=2)
            self.assertEqual(self.scheduler._collect_market_reads(), [quote])
            self.scheduler.market_data.get_fluctuation_symbols.assert_called_once()
            self.assertEqual(self.scheduler.market_data.get_multi_price.call_count, 2)
        finally:
            release.set()

        self.scheduler._pool_read.result(timeout=2)
        self.assertEqual(self.scheduler._collect_market_reads(), [])
        self.assertEqual(applied_threads, [get_ident()])
        self.scheduler.strategy.update_runtime_pool.assert_called_once_with(["009150"])
        self.assertEqual(quote.timestamp, now)
        self.assertEqual(self.scheduler._collect_market_reads(), [])
        self.scheduler.strategy.update_runtime_pool.assert_called_once()

    def test_pool_timeout_keeps_quotes_and_existing_watchlist(self):
        now = datetime(2026, 9, 23, 13)
        quote = Quote(symbol="005930", current_price=10000, timestamp=now)
        self.scheduler.strategy.update_runtime_pool = Mock()
        self.scheduler.market_data.get_fluctuation_symbols = Mock(
            side_effect=requests.exceptions.Timeout("ranking timeout"))
        self.scheduler.market_data.get_multi_price = Mock(return_value=[quote])
        self.scheduler._schedule_market_reads([quote.symbol], [], now)
        self.scheduler._market_read.result(timeout=2)
        with self.assertRaises(requests.exceptions.Timeout):
            self.scheduler._pool_read.result(timeout=2)
        with self.assertLogs("kis_trader.scheduler", level="WARNING"):
            self.assertEqual(self.scheduler._collect_market_reads(), [quote])
        self.scheduler.strategy.update_runtime_pool.assert_not_called()
        self.assertIsNone(self.scheduler._pool_read)
        self.scheduler._alerts.send.assert_not_called()

    def test_refresh_runtime_pool_uses_same_discovery_policy_at_market_open(self):
        now = datetime(2026, 4, 7, 9, 0, 3)
        update_runtime_pool = Mock()
        self.scheduler._session_started_at = datetime(2026, 4, 7, 9, 0, 0)
        self.scheduler.strategy = strategy_stub(
            positions={},
            fixed_pool=[],
            config=SimpleNamespace(
                dynamic_pool_ranking_fetch_count=30,
                dynamic_pool_quote_min_change_rate=0.8,
                pool_refresh_interval=300,
            ),
            update_runtime_pool=update_runtime_pool,
        )
        self.scheduler.market_data = Mock()
        self.scheduler.market_data.get_fluctuation_symbols.return_value = [
            "009150",
            "000660",
        ]

        self.scheduler._refresh_runtime_pool_if_needed(now)
        self.scheduler._pool_read.result(timeout=2)
        self.scheduler._collect_market_reads()
        self.scheduler.market_data.get_fluctuation_symbols.assert_called_once_with(
            count=30,
            min_change_rate=0.8,
        )
        update_runtime_pool.assert_called_once_with(["009150", "000660"])

    def test_refresh_runtime_pool_if_needed_keeps_existing_pool_when_fluctuation_empty(self):
        now = datetime(2026, 4, 7, 10, 0, 0)
        update_runtime_pool = Mock()
        self.scheduler.strategy = strategy_stub(
            positions={},
            fixed_pool=[],
            config=SimpleNamespace(
                dynamic_pool_ranking_fetch_count=30,
                dynamic_pool_quote_min_change_rate=0.8,
                pool_refresh_interval=300,
            ),
            update_runtime_pool=update_runtime_pool,
        )
        self.scheduler.market_data = Mock()
        self.scheduler.market_data.get_fluctuation_symbols.return_value = []

        refreshed = self.scheduler._refresh_runtime_pool_if_needed(now)
        self.scheduler._pool_read.result(timeout=2)
        self.scheduler._collect_market_reads()
        self.assertTrue(refreshed)
        update_runtime_pool.assert_not_called()
        self.assertEqual(self.scheduler._last_runtime_pool_refresh_at, now)

    def test_refresh_runtime_pool_if_needed_force_refreshes_during_warmup(self):
        now = datetime(2026, 4, 7, 10, 0, 5)
        update_runtime_pool = Mock()
        self.scheduler._session_started_at = datetime(2026, 4, 7, 10, 0, 0)
        self.scheduler.strategy = strategy_stub(
            positions={},
            fixed_pool=[],
            config=SimpleNamespace(
                dynamic_pool_ranking_fetch_count=30,
                dynamic_pool_quote_min_change_rate=0.8,
                pool_refresh_interval=300,
            ),
            update_runtime_pool=update_runtime_pool,
        )
        self.scheduler.market_data = Mock()
        self.scheduler.market_data.get_fluctuation_symbols.return_value = [
            "009150",
            "000660",
        ]

        self.scheduler._refresh_runtime_pool_if_needed(now, force=True)
        self.scheduler._pool_read.result(timeout=2)
        self.scheduler._collect_market_reads()
        update_runtime_pool.assert_called_once_with(["009150", "000660"])

    def test_refresh_runtime_pool_if_needed_skips_during_market_data_pressure(self):
        now = datetime(2026, 4, 7, 10, 0, 0)
        update_runtime_pool = Mock()
        self.scheduler.client = SimpleNamespace(rate_limit_cooldown_remaining=lambda: 1.0)
        self.scheduler.strategy = strategy_stub(
            positions={},
            fixed_pool=[],
            config=SimpleNamespace(
                dynamic_pool_ranking_fetch_count=30,
                dynamic_pool_quote_min_change_rate=0.8,
                pool_refresh_interval=300,
            ),
            update_runtime_pool=update_runtime_pool,
        )
        self.scheduler.market_data = Mock()

        self.scheduler._refresh_runtime_pool_if_needed(now)

        self.scheduler.market_data.get_fluctuation_symbols.assert_not_called()
        update_runtime_pool.assert_not_called()

    def test_refresh_runtime_pool_if_needed_skips_discovery_while_rate_limited(self):
        now = datetime(2026, 4, 7, 10, 0, 0)
        update_runtime_pool = Mock()
        self.scheduler.strategy = strategy_stub(
            positions={},
            fixed_pool=[],
            config=SimpleNamespace(
                dynamic_pool_ranking_fetch_count=30,
                dynamic_pool_quote_min_change_rate=0.8,
                pool_refresh_interval=300,
            ),
            update_runtime_pool=update_runtime_pool,
        )
        self.scheduler.market_data = SimpleNamespace(
            get_fluctuation_symbols=Mock(),
            fluctuation_backoff_remaining=lambda: 25.0,
            multi_price_backoff_remaining=lambda: 0.0,
        )

        refreshed = self.scheduler._refresh_runtime_pool_if_needed(now)

        self.assertFalse(refreshed)
        self.scheduler.market_data.get_fluctuation_symbols.assert_not_called()
        update_runtime_pool.assert_not_called()

    def test_emergency_rest_quotes_are_positions_only_when_stream_is_stale(self):
        now = datetime(2026, 4, 7, 10, 0, 1)
        held_quote = SimpleNamespace(symbol="005930")
        self.scheduler.strategy.positions = {"005930": object()}
        self.scheduler._quote_stream.stale_symbols.return_value = ["005930"]
        self.scheduler.market_data.get_multi_price = Mock(return_value=[held_quote])
        self.scheduler.strategy.fixed_pool = ["005930"]

        self.scheduler._schedule_market_reads(["005930"], ["005930"], now)
        self.scheduler._market_read.result(timeout=2)
        quotes = self.scheduler._collect_market_reads()

        self.assertEqual(quotes, [held_quote])
        self.scheduler.market_data.get_multi_price.assert_called_once_with(
            ["005930"],
        )
        self.scheduler._quote_stream.stale_symbols.assert_called_once_with(["005930"], now=now)
        self.scheduler._schedule_market_reads(["005930"], ["005930"], now + timedelta(seconds=1))
        self.assertIsNone(self.scheduler._market_read)

    def test_stream_priority_keeps_positions_pending_buys_and_dynamic_movers_first(self):
        self.scheduler.strategy = strategy_stub(
            positions={"005930": object()},
            _pending_entry_meta={"035420": {}},
            _pool=["001210", "004410"],
            config=SimpleNamespace(
                inverse_etfs=["114800"],
                static_watchlist=["005930", "000660"],
            ),
        )

        symbols = self.scheduler._prioritized_stream_symbols(
            ["000660", "035420"]
        )

        self.assertEqual(
            symbols,
            [
                "005930",
                "035420",
                "001210",
                "004410",
                "114800",
                "000660",
            ],
        )

    def test_supplemental_quotes_include_stale_stream_but_not_fresh_stream_or_held_symbols(self):
        now = datetime(2026, 10, 19, 13)
        quote = SimpleNamespace(symbol="005930")
        self.scheduler.strategy.positions = {"004410": object()}
        self.scheduler._quote_stream.stale_symbols.return_value = ["001210"]
        self.scheduler.market_data.get_multi_price = Mock(return_value=[quote])

        first = self.scheduler._supplemental_watchlist_symbols(
            ["001210", "004410", "005930"],
            ["001210"],
            now,
        )
        self.scheduler._quote_stream.stale_symbols.return_value = []
        second = self.scheduler._supplemental_watchlist_symbols(
            ["001210", "004410", "005930"],
            ["001210"],
            now + timedelta(seconds=1),
        )

        self.assertEqual(first, ["001210", "005930"])
        self.assertEqual(second, ["005930"])
        self.scheduler.market_data.get_multi_price.assert_not_called()
        self.assertEqual(self.scheduler._quote_stream.stale_symbols.call_args_list,
                         [call(["001210"], now=now, max_age_seconds=3.0),
                          call(["001210"], now=now+timedelta(seconds=1), max_age_seconds=3.0)])

    def test_supplemental_rotation_covers_both_pools_and_preserves_emergency_priority(self):
        now = datetime(2026, 10, 19, 13)
        watchlist = [f"{i:06d}" for i in range(1, 67)]
        subscribed = watchlist[:40]
        self.scheduler._quote_stream.stale_symbols.side_effect = lambda symbols, **kw: list(symbols)
        self.scheduler.market_data.get_multi_price = Mock(return_value=[])
        # Empty responses must not make the same symbols monopolize the next request.
        for second in (0, 2, 4):
            self.scheduler._schedule_market_reads(watchlist, subscribed, now+timedelta(seconds=second))
            self.scheduler._market_read.result(timeout=2)
            self.scheduler._collect_market_reads()
        calls = [c.args[0] for c in self.scheduler.market_data.get_multi_price.call_args_list]
        self.assertEqual([len(c) for c in calls], [30, 30, 30])
        self.assertEqual(calls[:2], [watchlist[:30], watchlist[30:60]])
        self.assertEqual(calls[2], watchlist[60:] + watchlist[:24])
        self.assertEqual(set().union(*map(set, calls)), set(watchlist))

        self.scheduler.strategy.positions = {watchlist[-1]: object()}
        self.scheduler._schedule_market_reads(watchlist, subscribed, now+timedelta(seconds=6))
        self.scheduler._market_read.result(timeout=2)
        self.scheduler._collect_market_reads()
        latest = self.scheduler.market_data.get_multi_price.call_args.args[0]
        self.assertEqual((latest[0], len(latest), len(set(latest))), (watchlist[-1], 30, 30))
        self.assertEqual(self.scheduler.market_data.get_multi_price.call_count, 4)

        remaining = watchlist[30:]
        self.scheduler._supplemental_watchlist_symbols(remaining, remaining[:10], now+timedelta(seconds=8))
        self.assertTrue(set(self.scheduler._last_supplemental_read_at) <= set(remaining))

        # Emergency holdings must not be dropped even when they require multiple API batches.
        self.scheduler.strategy.positions = dict.fromkeys(watchlist, object())
        self.scheduler._schedule_market_reads(watchlist, subscribed, now+timedelta(seconds=12))
        self.scheduler._market_read.result(timeout=2)
        self.scheduler._collect_market_reads()
        self.assertEqual([c.args[0] for c in self.scheduler.market_data.get_multi_price.call_args_list[-3:]],
                         [watchlist[:30], watchlist[30:60], watchlist[60:]])

    def test_quiet_subscriptions_can_build_real_strategy_history_without_faking_stream_ticks(self):
        start = datetime(2026, 10, 19, 13)
        clock = [start]
        watchlist = [f"{i:06d}" for i in range(1, 61)]
        with tempfile.TemporaryDirectory() as directory:
            strategy = MomentumScalpStrategy(None, MomentumScalpConfig(
                static_watchlist=watchlist, inverse_etfs=watchlist[-4:],
                daily_state_path=f"{directory}/state.json", quote_archive_root=f"{directory}/quotes"),
                fixed_pool=watchlist)
            self.scheduler.strategy = strategy
            self.scheduler._quote_stream = MarketQuoteStream(self.scheduler.config, stale_seconds=5)
            self.scheduler.market_data.get_multi_price = Mock(side_effect=lambda symbols: [
                Quote(s, 10000, timestamp=clock[0], bid_price=9990, ask_price=10010,
                      book_available=True, book_depth_available=True, bid_size=100, ask_size=100)
                for s in symbols])
            for second in range(0, 68, 2):
                clock[0] = start+timedelta(seconds=second)
                strategy.set_simulated_now(clock[0])
                universe, subscribed, quotes = self.scheduler._read_quote_batch()
                strategy._ingest_quotes(quotes)
                self.scheduler._schedule_market_reads(universe, subscribed, clock[0])
                self.scheduler._market_read.result(timeout=2)
            clock[0] = start+timedelta(seconds=68)
            strategy.set_simulated_now(clock[0])
            _, subscribed, quotes = self.scheduler._read_quote_batch()
            strategy._ingest_quotes(quotes)

            self.assertEqual(len(subscribed), 40)
            self.assertEqual(self.scheduler._quote_stream.drain_quotes(), [])
            self.assertEqual(self.scheduler._quote_stream.stale_symbols(subscribed, now=clock[0]), subscribed)
            fresh = strategy._fresh_quotes()
            self.assertEqual({q.symbol for q in fresh}, set(watchlist))
            self.assertTrue(all(strategy._breakout_window(q) for q in fresh))
            self.assertEqual({strategy._build_trade_plan(q, now=clock[0]).reason for q in fresh}, {"no_breakout"})
            self.assertEqual(self.scheduler.market_data.get_multi_price.call_count, 34)
            self.assertTrue(all(len(c.args[0]) == 30 for c in self.scheduler.market_data.get_multi_price.call_args_list))
            strategy.set_simulated_now(clock[0]+timedelta(seconds=6))
            self.assertEqual(strategy._fresh_quotes(), [])

    def test_intermittent_stream_is_refreshed_before_it_breaks_the_price_window(self):
        start = datetime(2026, 10, 20, 13)
        clock = [start]
        watchlist = [f"{i:06d}" for i in range(1, 67)]
        with tempfile.TemporaryDirectory() as directory:
            strategy = MomentumScalpStrategy(None, MomentumScalpConfig(
                static_watchlist=watchlist, inverse_etfs=watchlist[-4:],
                daily_state_path=f"{directory}/state.json", quote_archive_root=f"{directory}/quotes"),
                fixed_pool=watchlist)
            self.scheduler.strategy = strategy
            stream = MarketQuoteStream(self.scheduler.config, stale_seconds=5)
            self.scheduler._quote_stream = stream
            self.scheduler.market_data.get_multi_price = Mock(side_effect=lambda symbols: [
                Quote(s, 10000, timestamp=clock[0]+timedelta(seconds=.25),
                      bid_price=9990, ask_price=10010, book_available=True,
                      book_depth_available=True, bid_size=100, ask_size=100)
                for s in symbols])
            stream.update_symbols(watchlist)
            checks = 0
            for step in range(110):
                # One-second loop plus processing time; REST completes between ticks.
                clock[0] = start+timedelta(seconds=step*1.1)
                strategy.set_simulated_now(clock[0])
                for index, symbol in enumerate(watchlist[:40]):
                    if index >= 10 and step % 12 != 0:
                        continue
                    row = ["0"] * 47
                    row[0], row[1], row[2] = symbol, clock[0].strftime("%H%M%S"), "10000"
                    row[10], row[11], row[33] = "10010", "9990", start.strftime("%Y%m%d")
                    row[36:40] = ["100"] * 4
                    stream.feed_realtime_message("0|H0STCNT0|001|"+"^".join(row), received_at=clock[0])
                universe, subscribed, quotes = self.scheduler._read_quote_batch()
                strategy._ingest_quotes(quotes)
                if step >= 75:
                    fresh = strategy._fresh_quotes()
                    self.assertEqual({q.symbol for q in fresh}, set(watchlist))
                    missing = [q.symbol for q in fresh if not strategy._breakout_window(q)]
                    self.assertEqual(missing, [], f"step={step}")
                    checks += 1
                self.scheduler._schedule_market_reads(universe, subscribed, clock[0])
                if self.scheduler._market_read is not None:
                    self.scheduler._market_read.result(timeout=2)
            self.assertEqual(checks, 35)
            calls = self.scheduler.market_data.get_multi_price.call_args_list
            self.assertLessEqual(len(calls), 110)
            self.assertTrue(all(len(c.args[0]) <= 30 for c in calls))
            # Continuously streaming symbols do not consume the REST batch.
            self.assertTrue(all(not set(c.args[0]).intersection(watchlist[:10]) for c in calls))
            strategy.set_simulated_now(clock[0]+timedelta(seconds=6))
            self.assertEqual(strategy._fresh_quotes(), [])

    def test_reconcile_positions_after_sell_rate_limit_failure_syncs_account_positions(self):
        synced_positions = []
        self.scheduler.strategy = strategy_stub(
            positions={},
            sync_positions_from_account=lambda positions: synced_positions.extend(positions),
        )
        self.scheduler._fetch_balance_with_retry = lambda *args, **kwargs: SimpleNamespace(
            positions=["0010F0"],
        )
        result = OrderResult(
            success=False,
            side=OrderSide.SELL,
            message='{"rt_cd":"1","msg1":"초당 거래건수를 초과하였습니다.","message":"EGW00201"}',
        )

        self.scheduler._reconcile_pending_orders([result], wait=True)

        self.assertEqual(synced_positions, ["0010F0"])

    def test_reconcile_positions_after_sell_no_position_failure_syncs_account_positions(self):
        synced_positions = []
        self.scheduler.strategy = strategy_stub(
            positions={"015760": object()},
            sync_positions_from_account=lambda positions: synced_positions.extend(positions),
        )
        self.scheduler._fetch_balance_with_retry = lambda *args, **kwargs: SimpleNamespace(
            positions=[],
        )
        result = OrderResult(
            success=False,
            side=OrderSide.SELL,
            message='{"rt_cd":"1","msg_cd":"40240000","msg1":"모의투자 잔고내역이 없습니다."}',
            error_category="no_holding",
        )

        self.scheduler._reconcile_pending_orders([result], wait=True)

        self.assertEqual(synced_positions, [])

    def test_reconcile_positions_after_pending_fill_syncs_account_positions(self):
        synced_positions = []
        self.scheduler.strategy = strategy_stub(
            positions={},
            sync_positions_from_account=lambda positions: synced_positions.extend(positions),
        )
        self.scheduler._fetch_balance_with_retry = lambda *args, **kwargs: SimpleNamespace(
            positions=["005930"],
        )
        result = OrderResult(
            success=True,
            side=OrderSide.BUY,
            symbol="005930",
            quantity=0,
            price=0,
            requested_price=0,
            fill_mode="market_pending",
        )

        self.scheduler._reconcile_pending_orders([result], wait=True)

        self.assertEqual(synced_positions, ["005930"])

    def test_reconcile_positions_after_pending_fill_reconciles_before_sync(self):
        events = []
        self.scheduler.strategy = strategy_stub(
            positions={},
            reconcile_pending_fills_from_account=lambda results, positions: events.append(("reconcile", list(positions))) or [],
            sync_positions_from_account=lambda positions: events.append(("sync", list(positions))),
        )
        self.scheduler._fetch_balance_with_retry = lambda *args, **kwargs: SimpleNamespace(
            positions=["005930"],
        )
        result = OrderResult(
            success=True,
            side=OrderSide.BUY,
            symbol="005930",
            quantity=0,
            price=0,
            requested_price=0,
            fill_mode="market_pending",
        )

        self.scheduler._reconcile_pending_orders([result], wait=True)

        self.assertEqual(events, [("reconcile", ["005930"]), ("sync", ["005930"])])

    def test_pending_sell_syncs_account_before_confirming_realized_pnl(self):
        events = []
        self.scheduler.strategy = strategy_stub(
            positions={"017670": object()},
            reconcile_pending_fills_from_account=(
                lambda results, positions: events.append("reconcile") or []
            ),
            sync_positions_from_account=(
                lambda positions: events.append("sync")
            ),
        )
        self.scheduler._confirm_reconciled_sell_prices = (
            lambda results=None, **kwargs: events.append("confirm")
        )
        self.scheduler._fetch_balance_with_retry = lambda *args, **kwargs: SimpleNamespace(
            positions=[]
        )
        result = OrderResult(
            success=True,
            side=OrderSide.SELL,
            symbol="017670",
            quantity=0,
            requested_quantity=8,
            fill_mode="market_pending",
        )

        self.scheduler._reconcile_pending_orders([result], wait=True)

        self.assertEqual(events, ["reconcile", "sync", "confirm"])

    def test_reconciled_pending_buy_sends_buy_fill_alert(self):
        positions = {}
        inferred = OrderResult(
            success=True,
            symbol="085620",
            side=OrderSide.BUY,
            quantity=24,
            price=35_700,
            requested_price=0,
            fill_mode="account_reconciled",
            timestamp=datetime(2026, 6, 19, 11, 11, 32),
        )

        def reconcile_pending_fills_from_account(results, account_positions):
            positions["085620"] = SimpleNamespace(quantity=24, buy_price=35_700)
            return [inferred]

        self.scheduler.strategy = strategy_stub(
            positions=positions,
            daily_pnl=SimpleNamespace(realized_net_pnl=0),
            config=SimpleNamespace(commission_rate=0.00015, sell_tax_rate=0.002),
            reconcile_pending_fills_from_account=reconcile_pending_fills_from_account,
            sync_positions_from_account=lambda account_positions: None,
        )
        self.scheduler._fetch_balance_with_retry = lambda *args, **kwargs: SimpleNamespace(
            positions=[SimpleNamespace(symbol="085620", quantity=24, avg_price=35_700)],
        )
        pending = OrderResult(
            success=True,
            symbol="085620",
            side=OrderSide.BUY,
            quantity=0,
            price=0,
            reference_price=35_700,
            fill_mode="order_result_pending",
            timestamp=datetime(2026, 6, 19, 11, 11, 27),
        )

        self.scheduler._reconcile_pending_orders([pending], wait=True)

        self.scheduler._alerts.send.assert_called_once()
        kwargs = self.scheduler._alerts.send.call_args.kwargs
        self.assertEqual(kwargs["title"], "매수 체결")
        self.assertIn("085620 24주 @ 35,700원", kwargs["message"])

    def test_periodic_pending_buy_reconcile_confirms_delayed_fill(self):
        now = datetime.now()
        positions = {}
        pending = {
            "085620": {
                "plan": SimpleNamespace(quantity=24, entry_limit_price=35_700),
                "signal_price": 35_750,
                "created_at": now - timedelta(seconds=5),
                "order_no": "",
            }
        }

        def reconcile_pending(results, account_positions):
            pending.pop("085620", None)
            positions["085620"] = SimpleNamespace(quantity=24, buy_price=35_700)
            return [
                OrderResult(
                    success=True,
                    symbol="085620",
                    side=OrderSide.BUY,
                    quantity=24,
                    price=35_700,
                    requested_quantity=24,
                    fill_mode="account_reconciled",
                    timestamp=now,
                )
            ]

        self.scheduler.strategy = strategy_stub(
            _pending_entry_meta=pending,
            positions=positions,
            daily_pnl=SimpleNamespace(realized_net_pnl=0),
            config=SimpleNamespace(commission_rate=0.00015, sell_tax_rate=0.002),
            reconcile_pending_fills_from_account=reconcile_pending,
            sync_positions_from_account=lambda account_positions: None,
        )
        self.scheduler._fetch_balance_with_retry = lambda *args, **kwargs: SimpleNamespace(
            positions=[SimpleNamespace(symbol="085620", quantity=24, avg_price=35_700)],
        )

        reconciled = self.scheduler._reconcile_pending_orders(wait=True)

        self.assertTrue(reconciled)
        self.assertNotIn("085620", pending)
        self.scheduler._alerts.send.assert_called_once()
        self.assertEqual(
            self.scheduler._alerts.send.call_args.kwargs["title"],
            "매수 체결",
        )

    def test_confirmed_unfilled_buy_releases_block_without_balance_or_fill_alert(self):
        now = datetime.now()
        pending = {"152100": dict(order_no="known", created_at=now-timedelta(seconds=5),
                                 plan=SimpleNamespace(quantity=2, entry_limit_price=111280))}
        applied = Mock(side_effect=lambda result: pending.pop(result.symbol))
        self.scheduler.strategy = strategy_stub(_pending_entry_meta=pending, on_order_filled=applied)
        self.scheduler.trading = SimpleNamespace(resolve_pending_entry=Mock(side_effect=lambda r:
            replace(r, success=False, fill_mode="limit_cancelled", error_category="not_filled")))
        self.scheduler._pending_order_blocks[("152100", "buy")] = {"order_no": "known"}
        self.scheduler._fetch_balance_with_retry = Mock()

        self.assertTrue(self.scheduler._reconcile_pending_orders(wait=True))

        result = applied.call_args.args[0]
        self.assertFalse(result.success)
        self.assertEqual((result.order_no, result.requested_quantity, result.fill_mode),
                         ("known", 2, "limit_cancelled"))
        self.assertEqual(result.timestamp, now - timedelta(seconds=5))
        self.assertEqual(pending, {})
        self.assertEqual(self.scheduler._pending_order_blocks, {})
        self.scheduler._fetch_balance_with_retry.assert_not_called()
        self.scheduler._alerts.send.assert_not_called()

    def test_pending_buy_unknown_history_keeps_balance_reconciliation_and_poll_interval(self):
        for failure in (False, requests.exceptions.ReadTimeout("history timeout")):
            with self.subTest(failure=failure):
                now = datetime.now()
                pending = {"152100": dict(order_no="known", created_at=now-timedelta(seconds=200),
                                         plan=SimpleNamespace(quantity=2, entry_limit_price=111280))}
                self.scheduler._last_pending_entry_reconcile_at = None
                self.scheduler.strategy = strategy_stub(_pending_entry_meta=pending)
                confirm = Mock(return_value=None)
                if failure:
                    confirm.side_effect = failure
                self.scheduler.trading = SimpleNamespace(resolve_pending_entry=confirm)
                self.scheduler._fetch_balance_with_retry = Mock(return_value=SimpleNamespace(positions=[]))
                self.assertTrue(self.scheduler._reconcile_pending_orders(wait=True))
                self.assertFalse(self.scheduler._reconcile_pending_orders(wait=True))
                confirm.assert_called_once()
                self.scheduler._fetch_balance_with_retry.assert_called_once()
                self.assertIn("152100", pending)
                self.scheduler._alerts.send.assert_not_called()

    def test_send_order_result_alert_on_buy_fill(self):
        result = OrderResult(
            success=True,
            symbol="005930",
            side=OrderSide.BUY,
            quantity=3,
            price=71200,
            requested_price=0,
            timestamp=datetime(2026, 4, 9, 9, 1, 0),
        )
        self.scheduler.strategy = strategy_stub(
            daily_pnl=SimpleNamespace(realized_net_pnl=1250),
            config=SimpleNamespace(commission_rate=0.00015, sell_tax_rate=0.002),
        )
        after_position = SimpleNamespace(buy_price=71200)

        self.scheduler._send_order_result_alert(result, after_position=after_position)

        self.scheduler._alerts.send.assert_called_once()
        kwargs = self.scheduler._alerts.send.call_args.kwargs
        self.assertEqual(kwargs["title"], "매수 체결")
        self.assertIn("005930", kwargs["message"])
        self.assertIn("매수금액: 213,600원", kwargs["message"])
        self.assertIn("당일 누적순손익: 1,250원", kwargs["message"])
        self.assertNotIn("전략:", kwargs["message"])

    def test_send_order_result_alert_skips_non_fill_notices(self):
        result = OrderResult(
            success=True,
            symbol="005930",
            side=OrderSide.BUY,
            quantity=0,
            price=0,
            requested_price=0,
            fill_mode="market_pending",
            timestamp=datetime(2026, 4, 9, 9, 1, 0),
        )

        self.scheduler._send_order_result_alert(result)
        self.scheduler._send_order_result_alert(
            OrderResult(
                success=False,
                symbol="005930",
                side=OrderSide.BUY,
                quantity=0,
                price=0,
                requested_price=71_200,
                fill_mode="limit_cancelled",
                error_category="not_filled",
                message="EV 제한가 미체결 취소",
                timestamp=datetime(2026, 4, 9, 9, 1, 1),
            )
        )
        self.scheduler._send_order_result_alert(
            OrderResult(
                success=False,
                symbol="459550",
                side=OrderSide.BUY,
                quantity=0,
                price=0,
                error_category="rate_limit",
                message="[EGW00201] 초당 거래건수를 초과하였습니다.",
                timestamp=datetime(2026, 7, 27, 9, 2, 11),
            )
        )
        self.scheduler._send_order_result_alert(
            OrderResult(
                success=False,
                symbol="005930",
                side=OrderSide.BUY,
                quantity=0,
                price=0,
                error_category="other",
                message="temporary order API failure",
                timestamp=datetime(2026, 7, 27, 9, 2, 12),
            )
        )

        self.scheduler._alerts.send.assert_not_called()

    def test_market_pending_buy_does_not_use_provisional_position_as_a_fill(self):
        result = OrderResult(
            success=True,
            symbol="100790",
            side=OrderSide.BUY,
            quantity=0,
            price=0,
            requested_price=0,
            reference_price=41000,
            fill_mode="market_pending",
            timestamp=datetime(2026, 6, 9, 9, 1, 56),
        )
        self.scheduler.strategy = strategy_stub(
            daily_pnl=SimpleNamespace(realized_net_pnl=0),
            config=SimpleNamespace(commission_rate=0.00015, sell_tax_rate=0.002),
        )
        after_position = SimpleNamespace(quantity=12, buy_price=41000)

        self.scheduler._send_order_result_alert(result, after_position=after_position)

        self.scheduler._alerts.send.assert_not_called()

    def test_send_order_result_alert_on_sell_fill_includes_trade_and_daily_pnl(self):
        result = OrderResult(
            success=True,
            order_no="S1",
            fill_id="S1",
            symbol="005930",
            side=OrderSide.SELL,
            quantity=3,
            price=73000,
            requested_price=0,
            timestamp=datetime(2026, 4, 9, 9, 15, 0),
        )
        self.scheduler.strategy = strategy_stub(
            daily_pnl=SimpleNamespace(realized_net_pnl=4200),
            _sell_fill_ledger=[
                {
                    "fill_id": "S1",
                    "order_no": "S1",
                    "quantity": 3,
                    "sell_price": 73000,
                    "net_pnl": 4734,
                }
            ],
        )

        self.scheduler._send_order_result_alert(result)

        self.scheduler._alerts.send.assert_called_once()
        kwargs = self.scheduler._alerts.send.call_args.kwargs
        self.assertEqual(kwargs["title"], "매도 체결")
        self.assertIn("005930", kwargs["message"])
        self.assertIn("거래 순손익: 4,734원", kwargs["message"])
        self.assertIn("당일 누적순손익: 4,200원", kwargs["message"])
        self.assertNotIn("전략:", kwargs["message"])

    def test_pending_sell_does_not_send_order_received_alert(self):
        result = OrderResult(
            success=True,
            order_no="S1",
            symbol="005930",
            side=OrderSide.SELL,
            quantity=0,
            requested_quantity=3,
            reference_price=72_900,
            fill_mode="market_pending",
        )

        self.scheduler._send_order_result_alert(result)

        self.scheduler._alerts.send.assert_not_called()

    def test_reconciled_partial_sell_sends_partial_alert_with_remaining_quantity(self):
        self.scheduler.account = object()
        self.scheduler.strategy = strategy_stub(
            daily_pnl=SimpleNamespace(realized_net_pnl=-3_200),
            confirm_reconciled_sell_fills=lambda account, results=None, **kwargs: [
                {
                    "order_no": "S1",
                    "fill_id": "S1:25",
                    "symbol": "900300",
                    "quantity": 25,
                    "alert_quantity": 25,
                    "previous_price": 1_856,
                    "corrected_price": 1_850,
                    "alert_price": 1_850,
                    "delta_net_pnl": -10,
                    "net_pnl": -10,
                    "alert_net_pnl": -10,
                    "remaining_quantity": 514,
                    "partial": True,
                    "notify": True,
                    "requested_reason": "planned_stop",
                }
            ],
        )

        self.scheduler._confirm_reconciled_sell_prices()

        self.scheduler._alerts.send.assert_called_once()
        kwargs = self.scheduler._alerts.send.call_args.kwargs
        self.assertEqual(kwargs["title"], "매도 부분체결")
        self.assertIn("900300 25주 @ 1,850원", kwargs["message"])
        self.assertIn("미체결 잔량: 514주", kwargs["message"])

    def test_confirmed_reconciled_sell_sends_fill_alert(self):
        self.scheduler.account = object()
        self.scheduler.strategy = strategy_stub(
            daily_pnl=SimpleNamespace(realized_net_pnl=-9_369),
            confirm_reconciled_sell_fills=lambda account, results=None, **kwargs: [
                {
                    "order_no": "0000027235",
                    "symbol": "017670",
                    "quantity": 8,
                    "previous_price": 80_800,
                    "corrected_price": 80_900,
                    "delta_net_pnl": -3_088,
                    "net_pnl": -3_088,
                    "requested_reason": "planned_stop",
                }
            ],
        )

        self.scheduler._confirm_reconciled_sell_prices()

        self.scheduler._alerts.send.assert_called_once()
        kwargs = self.scheduler._alerts.send.call_args.kwargs
        self.assertEqual(kwargs["title"], "매도 체결")
        self.assertIn("017670 8주 @ 80,900원", kwargs["message"])
        self.assertIn("거래 순손익: -3,088원", kwargs["message"])
        self.assertIn("당일 누적순손익: -9,369원", kwargs["message"])

    def test_send_daily_halt_alert_if_needed(self):
        self.scheduler.strategy = strategy_stub(
            should_continue=lambda: False,
            daily_pnl=SimpleNamespace(realized_net_pnl=-3500),
        )

        self.scheduler._send_daily_halt_alert_if_needed()
        self.scheduler._send_daily_halt_alert_if_needed()

        self.scheduler._alerts.send.assert_called_once()
        kwargs = self.scheduler._alerts.send.call_args.kwargs
        self.assertEqual(kwargs["title"], "당일 거래 중지")
        self.assertIn("-3,500원", kwargs["message"])


if __name__ == "__main__":
    unittest.main()
