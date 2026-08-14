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
            multi_price_backoff_remaining=lambda: 0.0,
            get_fluctuation_symbols=lambda **kwargs: [],
            get_multi_price=lambda symbols: [],
        )
        self.scheduler.strategy = strategy_stub()
        self.scheduler._quote_stream = Mock()
        self.scheduler._quote_stream.stale_symbols.return_value = []
        self.scheduler._quote_stream.drain_quotes.return_value = []
        self.scheduler._alerts = Mock()
        self.scheduler._last_halt_alert_day = None
        self.scheduler._last_loop_error_alert_at = None
        self.scheduler._last_runtime_pool_refresh_at = None
        self.scheduler._session_started_at = None
        self.scheduler._market_closed_session_date = None
        self.scheduler._account_order_blocked_session_date = None
        self.scheduler._last_stream_stale_log_at = None
        self.scheduler._last_emergency_position_refresh_at = None
        self.scheduler._last_supplemental_quote_refresh_at = None
        self.scheduler._last_tick_activity_log_at = None
        self.scheduler._last_pending_entry_reconcile_at = None
        self.scheduler._last_pending_exit_reconcile_at = None
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

        reconciled = self.scheduler._reconcile_pending_exit_positions()

        self.assertTrue(reconciled)
        self.scheduler._fetch_balance_with_retry.assert_called_once_with(
            "pending 매도 잔고 확인",
            max_attempts=1,
            base_delay_seconds=1,
        )
        sync_positions.assert_called_once_with([])
        self.scheduler._confirm_reconciled_sell_prices.assert_called_once_with()

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

        reconciled = self.scheduler._reconcile_pending_exit_positions()

        self.assertTrue(reconciled)
        sync_positions.assert_called_once_with([])
        self.scheduler._confirm_reconciled_sell_prices.assert_called_once_with()

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

    def test_run_trading_session_resumes_after_confirmed_pnl_releases_strategy_halt(self):
        runtime = {"continue": False}
        self.scheduler.market_data = SimpleNamespace(is_market_open=lambda: True)
        self.scheduler.strategy = strategy_stub(
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
        self.scheduler.market_data = SimpleNamespace(is_market_open=lambda: True)
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

        self.scheduler.market_data.get_fluctuation_symbols.assert_called_once()
        update_runtime_pool.assert_called_once_with(["009150", "000660"])

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

        self.assertFalse(refreshed)
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

        quotes = self.scheduler._emergency_position_quotes(now)

        self.assertEqual(quotes, [held_quote])
        self.scheduler.market_data.get_multi_price.assert_called_once_with(
            ["005930"],
        )

    def test_stream_priority_keeps_positions_and_dynamic_movers_first(self):
        self.scheduler.strategy = strategy_stub(
            positions={"005930": object()},
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
            ["005930", "001210", "004410", "114800", "000660", "035420"],
        )

    def test_supplemental_quotes_cover_stale_and_unsubscribed_symbols_once_per_interval(self):
        now = datetime(2026, 8, 5, 10, 0, 0)
        quote = SimpleNamespace(symbol="001210")
        self.scheduler._quote_stream.stale_symbols.return_value = ["001210"]
        self.scheduler.market_data.get_multi_price = Mock(return_value=[quote])

        first = self.scheduler._supplemental_watchlist_quotes(
            now,
            ["001210", "004410", "005930"],
            ["001210", "004410"],
        )
        second = self.scheduler._supplemental_watchlist_quotes(
            now + timedelta(seconds=4),
            ["001210", "004410", "005930"],
            ["001210", "004410"],
        )

        self.assertEqual(first, [quote])
        self.assertEqual(second, [])
        self.scheduler.market_data.get_multi_price.assert_called_once_with(
            ["001210", "005930"]
        )

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

        self.scheduler._reconcile_positions_after_order_failures([result])

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

        self.scheduler._reconcile_positions_after_order_failures([result])

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

        self.scheduler._reconcile_positions_after_order_failures([result])

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

        self.scheduler._reconcile_positions_after_order_failures([result])

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
            lambda results=None: events.append("confirm")
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

        self.scheduler._reconcile_positions_after_order_failures([result])

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

        self.scheduler._reconcile_positions_after_order_failures([pending])

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

        reconciled = self.scheduler._reconcile_pending_entry_positions()

        self.assertTrue(reconciled)
        self.assertNotIn("085620", pending)
        self.scheduler._alerts.send.assert_called_once()
        self.assertEqual(
            self.scheduler._alerts.send.call_args.kwargs["title"],
            "매수 체결",
        )

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
            confirm_reconciled_sell_fills=lambda account, results=None: [
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
            confirm_reconciled_sell_fills=lambda account, results=None: [
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
