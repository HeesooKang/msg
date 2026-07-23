import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests

from src.models import Order, OrderResult, OrderSide
from src.scheduler import TradingScheduler


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


class SchedulerTimingTests(unittest.TestCase):
    def setUp(self):
        self.scheduler = TradingScheduler.__new__(TradingScheduler)
        self.scheduler.config = SimpleNamespace(off_hours_check_interval=1800, is_paper=True, trading_mode="paper")
        self.scheduler._shutdown = False
        self.scheduler._exit_priority_until = None
        self.scheduler._last_exit_priority_log_at = None
        self.scheduler._last_rate_limit_watchlist_log_at = None
        self.scheduler.client = SimpleNamespace(rate_limit_cooldown_remaining=lambda: 0.0)
        self.scheduler.market_data = SimpleNamespace(
            multi_price_backoff_remaining=lambda: 0.0,
            quote_cache_age_seconds=lambda symbol: None,
            get_fluctuation_ranking=lambda **kwargs: [],
            get_market_cap_ranking=lambda **kwargs: [],
            get_multi_price=lambda symbols: [],
            get_cached_quotes=lambda symbols, ttl_seconds: [],
        )
        self.scheduler.strategy = SimpleNamespace(positions={})
        self.scheduler._alerts = Mock()
        self.scheduler._last_halt_alert_day = None
        self.scheduler._watchlist_rotation_cursor = 0
        self.scheduler._last_runtime_pool_refresh_at = None
        self.scheduler._session_started_at = None
        self.scheduler._last_cached_quote_fallback_log_at = None
        self.scheduler._last_empty_quote_batch_log_at = None
        self.scheduler._last_tick_activity_log_at = None
        self.scheduler._pending_order_blocks = {}

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

    def test_pending_sell_block_symbols_for_positions_detects_halt_liquidation_wait(self):
        now = datetime(2026, 6, 8, 9, 11, 46)
        self.scheduler._remember_pending_order_results(
            [
                OrderResult(
                    success=True,
                    symbol="006220",
                    side=OrderSide.SELL,
                    quantity=0,
                    price=0,
                    fill_mode="market_pending",
                    timestamp=now,
                )
            ],
            now,
        )

        symbols = self.scheduler._pending_sell_block_symbols_for_positions(
            {"006220": object(), "036170": object()},
            now + timedelta(seconds=20),
        )
        other_symbols = self.scheduler._pending_sell_block_symbols_for_positions(
            {"005930": object()},
            now + timedelta(seconds=20),
        )

        self.assertEqual(symbols, ["006220"])
        self.assertEqual(other_symbols, [])

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
        confirmed = OrderResult(
            success=True,
            symbol="274090",
            side=OrderSide.SELL,
            quantity=26,
            requested_quantity=26,
            price=38650,
            fill_mode="account_reconciled_confirmed",
            timestamp=now + timedelta(seconds=10),
        )

        self.scheduler._clear_pending_order_blocks_for_results([confirmed])
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
        self.scheduler.account = SimpleNamespace(get_balance=lambda: responses.pop(0))
        self.scheduler.client = SimpleNamespace(rate_limit_cooldown_remaining=lambda: 3.2)
        self.scheduler._interruptible_sleep = lambda seconds: sleep_calls.append(seconds)

        balance = self.scheduler._fetch_balance_with_retry("세션 시작 잔고 조회", max_attempts=2, base_delay_seconds=2)

        self.assertIsNotNone(balance)
        self.assertEqual(sleep_calls, [4])

    def test_fetch_balance_with_retry_handles_transport_timeout_inside_retry_loop(self):
        sleep_calls = []
        responses = [
            requests.exceptions.Timeout("balance timeout"),
            SimpleNamespace(total_deposit=1, positions=[]),
        ]

        def get_balance():
            item = responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        self.scheduler.account = SimpleNamespace(get_balance=get_balance)
        self.scheduler._interruptible_sleep = lambda seconds: sleep_calls.append(seconds)

        balance = self.scheduler._fetch_balance_with_retry("세션 시작 잔고 조회", max_attempts=2, base_delay_seconds=2)

        self.assertIsNotNone(balance)
        self.assertEqual(sleep_calls, [2])

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

    def test_run_trading_session_does_not_send_kakao_alert_for_intraday_network_error(self):
        self.scheduler.market_data = SimpleNamespace(is_market_open=lambda: True)
        self.scheduler.strategy = SimpleNamespace(
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

    def test_run_trading_session_resumes_after_confirmed_pnl_releases_strategy_halt(self):
        runtime = {"continue": False}
        self.scheduler.market_data = SimpleNamespace(is_market_open=lambda: True)
        self.scheduler.strategy = SimpleNamespace(
            initialize=lambda: None,
            get_watchlist=lambda: ["067310"],
            sync_positions_from_account=lambda positions: None,
            should_continue=lambda: runtime["continue"],
            positions={},
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
        self.scheduler._confirm_reconciled_sell_prices = lambda results=None: runtime.__setitem__("continue", True)
        self.scheduler._interruptible_sleep = lambda seconds: None

        halted = self.scheduler._run_trading_session(10)

        self.assertFalse(halted)
        self.assertTrue(runtime["continue"])

    def test_run_trading_session_fetches_balance_before_runtime_pool_refresh(self):
        order = []
        self.scheduler.market_data = SimpleNamespace(is_market_open=lambda: True)
        self.scheduler.strategy = SimpleNamespace(
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
        self.scheduler._fetch_balance_with_retry = lambda *args, **kwargs: order.append("balance") or SimpleNamespace(
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
        self.assertEqual(order[:4], ["initialize", "balance", "sync", "refresh"])

    def test_run_trading_session_clears_stale_strategy_simulated_time_before_initialize(self):
        order = []
        self.scheduler.market_data = SimpleNamespace(is_market_open=lambda: True)
        self.scheduler.strategy = SimpleNamespace(
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

    def test_run_trading_session_skips_force_runtime_pool_refresh_when_state_snapshot_exists(self):
        self.scheduler.market_data = SimpleNamespace(is_market_open=lambda: True)
        refresh_runtime_pool = Mock(return_value=False)
        self.scheduler.strategy = SimpleNamespace(
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
        self.assertFalse(refresh_runtime_pool.call_args.kwargs["force"])

    def test_log_tick_activity_prefers_strategy_shortlist_symbols(self):
        now = datetime(2026, 4, 15, 9, 9, 0)
        self.scheduler.strategy = SimpleNamespace(
            positions={},
            _last_long_shortlist_symbols=["046970"],
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
        self.assertEqual(args[0], "틱 상태: 감시=%d 배치=%d 수신=%d 롱후보=%d 보유=%d")
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

    def test_effective_watchlist_uses_held_symbols_during_exit_priority(self):
        now = datetime(2026, 4, 7, 12, 0, 0)
        self.scheduler.strategy = SimpleNamespace(positions={"0010F0": object(), "066970": object()})
        self.scheduler._exit_priority_until = now.replace(second=30)

        watchlist = self.scheduler._effective_watchlist(["005930", "000660", "0010F0"], now)

        self.assertEqual(watchlist, ["0010F0", "066970"])

    def test_effective_watchlist_skips_new_entries_during_rate_limit_cooldown_without_positions(self):
        now = datetime(2026, 4, 7, 12, 0, 0)
        self.scheduler.client = SimpleNamespace(rate_limit_cooldown_remaining=lambda: 1.8)
        self.scheduler.strategy = SimpleNamespace(positions={})
        watchlist = [f"{i:06d}" for i in range(30)]

        limited = self.scheduler._effective_watchlist(watchlist, now)

        self.assertEqual(limited, [])

    def test_effective_watchlist_keeps_new_entries_during_multi_price_backoff(self):
        now = datetime(2026, 4, 7, 12, 0, 0)
        self.scheduler.market_data = SimpleNamespace(multi_price_backoff_remaining=lambda: 12.0)
        self.scheduler.strategy = SimpleNamespace(positions={})

        limited = self.scheduler._effective_watchlist(["005930", "000660"], now)

        self.assertEqual(limited, ["005930", "000660"])

    def test_effective_watchlist_uses_held_symbols_during_multi_price_backoff(self):
        now = datetime(2026, 4, 7, 12, 0, 0)
        self.scheduler.market_data = SimpleNamespace(multi_price_backoff_remaining=lambda: 12.0)
        self.scheduler.strategy = SimpleNamespace(positions={"005930": object(), "000660": object()})

        limited = self.scheduler._effective_watchlist(["005930", "000660", "051910"], now)

        self.assertEqual(limited, ["005930", "000660"])

    def test_build_quote_refresh_batch_caps_full_watchlist_to_single_chunk(self):
        now = datetime(2026, 4, 7, 10, 0, 0)
        self.scheduler.strategy = SimpleNamespace(positions={})
        watchlist = [f"{i:06d}" for i in range(30)]

        batch = self.scheduler._build_quote_refresh_batch(watchlist, now)

        self.assertEqual(len(batch), 12)
        self.assertEqual(batch, watchlist[:12])

    def test_build_quote_refresh_batch_uses_smaller_budget_during_session_warmup(self):
        now = datetime(2026, 4, 7, 10, 0, 20)
        self.scheduler.strategy = SimpleNamespace(positions={})
        self.scheduler._session_started_at = datetime(2026, 4, 7, 10, 0, 0)
        watchlist = [f"{i:06d}" for i in range(30)]

        batch = self.scheduler._build_quote_refresh_batch(watchlist, now)

        self.assertEqual(len(batch), 6)
        self.assertEqual(batch, watchlist[:6])

    def test_build_quote_refresh_batch_uses_larger_budget_during_opening_fast_warmup(self):
        now = datetime(2026, 4, 7, 9, 0, 20)
        self.scheduler.strategy = SimpleNamespace(
            positions={},
            config=SimpleNamespace(
                opening_fast_window_minutes=3,
                opening_fast_quote_warmup_seconds=2.0,
                opening_fast_initial_quote_budget=14,
            ),
        )
        self.scheduler._session_started_at = datetime(2026, 4, 7, 9, 0, 0)
        watchlist = [f"{i:06d}" for i in range(30)]

        batch = self.scheduler._build_quote_refresh_batch(watchlist, now)

        self.assertEqual(len(batch), 14)
        self.assertEqual(batch, watchlist[:14])

    def test_build_quote_refresh_batch_keeps_tiny_opening_start_buffer(self):
        now = datetime(2026, 4, 7, 9, 0, 1)
        self.scheduler.strategy = SimpleNamespace(
            positions={},
            config=SimpleNamespace(
                opening_fast_window_minutes=3,
                opening_fast_quote_warmup_seconds=2.0,
                opening_fast_initial_quote_budget=14,
            ),
        )
        self.scheduler._session_started_at = datetime(2026, 4, 7, 9, 0, 0)
        watchlist = [f"{i:06d}" for i in range(30)]

        batch = self.scheduler._build_quote_refresh_batch(watchlist, now)

        self.assertEqual(batch, [])

    def test_build_quote_refresh_batch_includes_positions_and_hot_symbols_first(self):
        now = datetime(2026, 4, 7, 9, 2, 0)
        self.scheduler.strategy = SimpleNamespace(
            positions={"005930": object(), "000660": object()},
            _latest_opening_fast_symbols={"051910"},
            _latest_opening_hot_symbols={"035420"},
            _latest_math_queue_symbols=["005490", "012330"],
            _latest_math_backfill_symbols=["034730"],
        )
        watchlist = ["005930", "000660", "051910", "035420", "005490", "012330", "034730", "003550"]

        batch = self.scheduler._build_quote_refresh_batch(watchlist, now)

        self.assertEqual(batch[:2], ["005930", "000660"])
        self.assertIn("051910", batch)
        self.assertIn("035420", batch)
        self.assertLessEqual(len(batch), 14)

    def test_build_quote_refresh_batch_rotates_background_symbols(self):
        now = datetime(2026, 4, 7, 10, 0, 0)
        self.scheduler.strategy = SimpleNamespace(positions={})
        watchlist = [f"{i:06d}" for i in range(20)]

        first = self.scheduler._build_quote_refresh_batch(watchlist, now)
        second = self.scheduler._build_quote_refresh_batch(watchlist, now)

        self.assertEqual(first[:4], watchlist[:4])
        self.assertEqual(second[:4], watchlist[12:16])
        self.assertNotEqual(first, second)

    def test_refresh_runtime_pool_if_needed_uses_fluctuation_ranking_for_discovery(self):
        now = datetime(2026, 4, 7, 9, 2, 0)
        update_runtime_pool = Mock()
        self.scheduler.strategy = SimpleNamespace(
            positions={},
            pool_override=[],
            config=SimpleNamespace(
                opening_candidate_window_minutes=20,
                opening_fast_window_minutes=3,
                opening_fast_fetch_count=80,
                opening_candidate_fetch_count=60,
                dynamic_pool_ranking_fetch_count=30,
                dynamic_pool_size=15,
                dynamic_pool_quote_min_change_rate=0.8,
                max_change_rate=10.0,
                min_price=5000,
                min_volume=180000,
                pool_refresh_interval=300,
            ),
            update_runtime_pool=update_runtime_pool,
        )
        self.scheduler.market_data = Mock()
        self.scheduler.market_data.get_fluctuation_ranking.return_value = [
            SimpleNamespace(symbol="009150"),
            SimpleNamespace(symbol="000660"),
        ]
        self.scheduler.market_data.get_market_cap_ranking.return_value = []

        self.scheduler._refresh_runtime_pool_if_needed(now)

        self.scheduler.market_data.get_fluctuation_ranking.assert_called_once()
        update_runtime_pool.assert_called_once_with(["009150", "000660"])

    def test_refresh_runtime_pool_if_needed_relaxes_filters_during_opening_fast(self):
        now = datetime(2026, 4, 7, 9, 0, 3)
        update_runtime_pool = Mock()
        self.scheduler._session_started_at = datetime(2026, 4, 7, 9, 0, 0)
        self.scheduler.strategy = SimpleNamespace(
            positions={},
            pool_override=[],
            config=SimpleNamespace(
                opening_candidate_window_minutes=20,
                opening_fast_window_minutes=3,
                opening_fast_fetch_count=80,
                opening_candidate_fetch_count=60,
                opening_fast_pool_warmup_seconds=2.0,
                opening_fast_pool_refresh_interval_seconds=20,
                opening_fast_min_change_rate=0.2,
                opening_fast_min_volume=30_000,
                dynamic_pool_ranking_fetch_count=30,
                dynamic_pool_size=15,
                dynamic_pool_quote_min_change_rate=0.8,
                max_change_rate=10.0,
                min_price=5000,
                min_volume=180000,
                pool_refresh_interval=300,
            ),
            update_runtime_pool=update_runtime_pool,
        )
        self.scheduler.market_data = Mock()
        self.scheduler.market_data.get_fluctuation_ranking.return_value = [
            SimpleNamespace(symbol="009150"),
            SimpleNamespace(symbol="000660"),
        ]
        self.scheduler.market_data.get_market_cap_ranking.return_value = []

        self.scheduler._refresh_runtime_pool_if_needed(now)

        self.scheduler.market_data.get_fluctuation_ranking.assert_called_once_with(
            count=80,
            min_change_rate=0.2,
            max_change_rate=30.0,
            min_price=0,
            min_volume=0,
        )
        update_runtime_pool.assert_called_once_with(["009150", "000660"])

    def test_refresh_runtime_pool_if_needed_keeps_existing_pool_when_fluctuation_empty(self):
        now = datetime(2026, 4, 7, 10, 0, 0)
        update_runtime_pool = Mock()
        self.scheduler.strategy = SimpleNamespace(
            positions={},
            pool_override=[],
            config=SimpleNamespace(
                opening_candidate_window_minutes=20,
                opening_fast_window_minutes=3,
                opening_fast_fetch_count=80,
                opening_candidate_fetch_count=60,
                dynamic_pool_ranking_fetch_count=30,
                dynamic_pool_size=15,
                dynamic_pool_quote_min_change_rate=0.8,
                max_change_rate=10.0,
                min_price=5000,
                min_volume=180000,
                pool_refresh_interval=300,
            ),
            update_runtime_pool=update_runtime_pool,
        )
        self.scheduler.market_data = Mock()
        self.scheduler.market_data.get_fluctuation_ranking.return_value = []
        self.scheduler.market_data.get_market_cap_ranking.return_value = [
            SimpleNamespace(symbol="005930"),
            SimpleNamespace(symbol="373220"),
        ]

        refreshed = self.scheduler._refresh_runtime_pool_if_needed(now)

        self.assertFalse(refreshed)
        self.scheduler.market_data.get_market_cap_ranking.assert_not_called()
        update_runtime_pool.assert_not_called()
        self.assertEqual(self.scheduler._last_runtime_pool_refresh_at, now)

    def test_refresh_runtime_pool_if_needed_force_bypasses_warmup_gate(self):
        now = datetime(2026, 4, 7, 10, 0, 5)
        update_runtime_pool = Mock()
        self.scheduler._session_started_at = datetime(2026, 4, 7, 10, 0, 0)
        self.scheduler.strategy = SimpleNamespace(
            positions={},
            pool_override=[],
            config=SimpleNamespace(
                opening_candidate_window_minutes=20,
                opening_fast_window_minutes=3,
                opening_fast_fetch_count=80,
                opening_candidate_fetch_count=60,
                dynamic_pool_ranking_fetch_count=30,
                dynamic_pool_size=15,
                dynamic_pool_quote_min_change_rate=0.8,
                max_change_rate=10.0,
                min_price=5000,
                min_volume=180000,
                pool_refresh_interval=300,
            ),
            update_runtime_pool=update_runtime_pool,
        )
        self.scheduler.market_data = Mock()
        self.scheduler.market_data.get_fluctuation_ranking.return_value = [
            SimpleNamespace(symbol="009150"),
            SimpleNamespace(symbol="000660"),
        ]
        self.scheduler.market_data.get_market_cap_ranking.return_value = []

        self.scheduler._refresh_runtime_pool_if_needed(now, force=True)

        update_runtime_pool.assert_called_once_with(["009150", "000660"])

    def test_refresh_runtime_pool_if_needed_skips_during_market_data_pressure(self):
        now = datetime(2026, 4, 7, 10, 0, 0)
        update_runtime_pool = Mock()
        self.scheduler.client = SimpleNamespace(rate_limit_cooldown_remaining=lambda: 1.0)
        self.scheduler.strategy = SimpleNamespace(
            positions={},
            pool_override=[],
            config=SimpleNamespace(
                opening_candidate_window_minutes=20,
                opening_fast_window_minutes=3,
                opening_fast_fetch_count=80,
                opening_candidate_fetch_count=60,
                dynamic_pool_ranking_fetch_count=30,
                dynamic_pool_size=15,
                dynamic_pool_quote_min_change_rate=0.8,
                max_change_rate=10.0,
                min_price=5000,
                min_volume=180000,
                pool_refresh_interval=300,
            ),
            update_runtime_pool=update_runtime_pool,
        )
        self.scheduler.market_data = Mock()

        self.scheduler._refresh_runtime_pool_if_needed(now)

        self.scheduler.market_data.get_fluctuation_ranking.assert_not_called()
        update_runtime_pool.assert_not_called()

    def test_refresh_runtime_pool_if_needed_skips_market_cap_fallback_when_fluctuation_rate_limited(self):
        now = datetime(2026, 4, 7, 10, 0, 0)
        update_runtime_pool = Mock()
        self.scheduler.strategy = SimpleNamespace(
            positions={},
            pool_override=[],
            config=SimpleNamespace(
                opening_candidate_window_minutes=20,
                opening_fast_window_minutes=3,
                opening_fast_fetch_count=80,
                opening_candidate_fetch_count=60,
                dynamic_pool_ranking_fetch_count=30,
                dynamic_pool_size=15,
                dynamic_pool_quote_min_change_rate=0.8,
                max_change_rate=10.0,
                min_price=5000,
                min_volume=180000,
                pool_refresh_interval=300,
            ),
            update_runtime_pool=update_runtime_pool,
        )
        self.scheduler.market_data = SimpleNamespace(
            get_fluctuation_ranking=lambda **kwargs: [],
            get_market_cap_ranking=Mock(),
            fluctuation_backoff_remaining=lambda: 25.0,
            multi_price_backoff_remaining=lambda: 0.0,
        )

        refreshed = self.scheduler._refresh_runtime_pool_if_needed(now)

        self.assertFalse(refreshed)
        self.scheduler.market_data.get_market_cap_ranking.assert_not_called()
        update_runtime_pool.assert_not_called()

    def test_load_quotes_for_refresh_batch_uses_cached_quotes_when_live_fetch_fails(self):
        cached_quote = SimpleNamespace(symbol="005930")
        self.scheduler.market_data = SimpleNamespace(
            get_multi_price=lambda symbols: [],
            get_cached_quotes=lambda symbols, ttl_seconds: [cached_quote],
        )

        quotes = self.scheduler._load_quotes_for_refresh_batch(["005930"], datetime(2026, 4, 7, 10, 0, 0))

        self.assertEqual(quotes, [cached_quote])

    def test_update_exit_priority_from_results_marks_sell_rate_limit(self):
        now = datetime(2026, 4, 7, 12, 0, 0)
        result = SimpleNamespace(
            success=False,
            side=OrderSide.SELL,
            message='{"rt_cd":"1","msg1":"초당 거래건수를 초과하였습니다.","message":"EGW00201"}',
        )

        self.scheduler._update_exit_priority_from_results([result], now)

        self.assertIsNotNone(self.scheduler._exit_priority_until)
        self.assertGreater(self.scheduler._exit_priority_until, now)

    def test_reconcile_positions_after_sell_rate_limit_failure_syncs_account_positions(self):
        synced_positions = []
        self.scheduler.strategy = SimpleNamespace(
            positions={},
            sync_positions_from_account=lambda positions: synced_positions.extend(positions),
        )
        self.scheduler._fetch_balance_with_retry = lambda *args, **kwargs: SimpleNamespace(
            positions=["0010F0"],
        )
        result = SimpleNamespace(
            success=False,
            side=OrderSide.SELL,
            message='{"rt_cd":"1","msg1":"초당 거래건수를 초과하였습니다.","message":"EGW00201"}',
        )

        self.scheduler._reconcile_positions_after_order_failures([result])

        self.assertEqual(synced_positions, ["0010F0"])

    def test_reconcile_positions_after_sell_no_position_failure_syncs_account_positions(self):
        synced_positions = []
        self.scheduler.strategy = SimpleNamespace(
            positions={"015760": object()},
            sync_positions_from_account=lambda positions: synced_positions.extend(positions),
        )
        self.scheduler._fetch_balance_with_retry = lambda *args, **kwargs: SimpleNamespace(
            positions=[],
        )
        result = SimpleNamespace(
            success=False,
            side=OrderSide.SELL,
            message='{"rt_cd":"1","msg_cd":"40240000","msg1":"모의투자 잔고내역이 없습니다."}',
            error_category="no_holding",
        )

        self.scheduler._reconcile_positions_after_order_failures([result])

        self.assertEqual(synced_positions, [])

    def test_reconcile_positions_after_pending_fill_syncs_account_positions(self):
        synced_positions = []
        self.scheduler.strategy = SimpleNamespace(
            positions={},
            sync_positions_from_account=lambda positions: synced_positions.extend(positions),
        )
        self.scheduler._fetch_balance_with_retry = lambda *args, **kwargs: SimpleNamespace(
            positions=["005930"],
        )
        result = SimpleNamespace(
            success=True,
            side=OrderSide.BUY,
            symbol="005930",
            quantity=0,
            price=0,
            requested_price=0,
            fill_mode="market_pending",
        )

        self.scheduler._reconcile_positions_after_order_failures([result])

        self.assertEqual(synced_positions, ["005930"])

    def test_reconcile_positions_after_pending_fill_reconciles_before_sync(self):
        events = []
        self.scheduler.strategy = SimpleNamespace(
            positions={},
            reconcile_pending_fills_from_account=lambda results, positions: events.append(("reconcile", list(positions))) or [],
            sync_positions_from_account=lambda positions: events.append(("sync", list(positions))),
        )
        self.scheduler._fetch_balance_with_retry = lambda *args, **kwargs: SimpleNamespace(
            positions=["005930"],
        )
        result = SimpleNamespace(
            success=True,
            side=OrderSide.BUY,
            symbol="005930",
            quantity=0,
            price=0,
            requested_price=0,
            fill_mode="market_pending",
        )

        self.scheduler._reconcile_positions_after_order_failures([result])

        self.assertEqual(events, [("reconcile", ["005930"]), ("sync", ["005930"])])

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

        self.scheduler.strategy = SimpleNamespace(
            positions=positions,
            daily_pnl=SimpleNamespace(realized_net_pnl=0),
            config=SimpleNamespace(commission_rate=0.00015, tax_slippage_rate=0.002),
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

        self.scheduler._reconcile_positions_after_order_failures([pending])

        self.scheduler._alerts.send.assert_called_once()
        kwargs = self.scheduler._alerts.send.call_args.kwargs
        self.assertEqual(kwargs["title"], "매수 체결")
        self.assertIn("085620 24주 @ 35,700원", kwargs["message"])

    def test_send_order_result_alert_on_buy_fill(self):
        result = SimpleNamespace(
            success=True,
            symbol="005930",
            side=OrderSide.BUY,
            quantity=3,
            price=71200,
            requested_price=0,
            timestamp=datetime(2026, 4, 9, 9, 1, 0),
        )
        self.scheduler.strategy = SimpleNamespace(
            daily_pnl=SimpleNamespace(realized_net_pnl=1250),
            config=SimpleNamespace(commission_rate=0.00015, tax_slippage_rate=0.002),
        )
        after_position = SimpleNamespace(
            entry_strategy_name="opening_conviction_long_strategy",
            entry_setup_name="opening_conviction",
            live_route="opening_conviction_long_strategy",
            buy_price=71200,
        )

        self.scheduler._send_order_result_alert(result, after_position=after_position)

        self.scheduler._alerts.send.assert_called_once()
        kwargs = self.scheduler._alerts.send.call_args.kwargs
        self.assertEqual(kwargs["title"], "매수 체결")
        self.assertIn("005930", kwargs["message"])
        self.assertIn("매수금액: 213,600원", kwargs["message"])
        self.assertIn("당일 누적순손익: 1,250원", kwargs["message"])
        self.assertNotIn("전략:", kwargs["message"])

    def test_send_order_result_alert_skips_pending_account_reconcile_notice(self):
        result = SimpleNamespace(
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

        self.scheduler._alerts.send.assert_not_called()

    def test_send_order_result_alert_uses_provisional_buy_position_for_market_pending(self):
        result = SimpleNamespace(
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
        self.scheduler.strategy = SimpleNamespace(
            daily_pnl=SimpleNamespace(realized_net_pnl=0),
            config=SimpleNamespace(commission_rate=0.00015, tax_slippage_rate=0.002),
        )
        after_position = SimpleNamespace(quantity=12, buy_price=41000)

        self.scheduler._send_order_result_alert(result, after_position=after_position)

        self.scheduler._alerts.send.assert_called_once()
        kwargs = self.scheduler._alerts.send.call_args.kwargs
        self.assertEqual(kwargs["title"], "매수 체결")
        self.assertIn("100790 12주 @ 41,000원", kwargs["message"])
        self.assertIn("매수금액: 492,000원", kwargs["message"])

    def test_send_order_result_alert_on_sell_fill_includes_trade_and_daily_pnl(self):
        result = SimpleNamespace(
            success=True,
            symbol="005930",
            side=OrderSide.SELL,
            quantity=3,
            price=73000,
            requested_price=0,
            timestamp=datetime(2026, 4, 9, 9, 15, 0),
        )
        self.scheduler.strategy = SimpleNamespace(
            daily_pnl=SimpleNamespace(realized_net_pnl=4200),
            config=SimpleNamespace(commission_rate=0.00015, tax_slippage_rate=0.002),
        )
        before_position = SimpleNamespace(
            buy_price=71200,
        )

        self.scheduler._send_order_result_alert(result, before_position=before_position)

        self.scheduler._alerts.send.assert_called_once()
        kwargs = self.scheduler._alerts.send.call_args.kwargs
        self.assertEqual(kwargs["title"], "매도 체결")
        self.assertIn("005930", kwargs["message"])
        self.assertIn("거래 순손익:", kwargs["message"])
        self.assertIn("당일 누적순손익: 4,200원", kwargs["message"])
        self.assertNotIn("전략:", kwargs["message"])

    def test_send_daily_halt_alert_if_needed(self):
        self.scheduler.strategy = SimpleNamespace(
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
