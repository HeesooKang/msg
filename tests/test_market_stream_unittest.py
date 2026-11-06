import asyncio
import json
import unittest
from dataclasses import replace
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import requests

from src.market_stream import (
    MAX_SYMBOLS_PER_CONNECTION,
    TRADE_ROW_WIDTH,
    TRADE_TR_ID,
    MarketQuoteStream,
)
from src.analytics.execution_policy import limit_fill, market_fill
from src.models import Quote


def config():
    return SimpleNamespace(
        api_key="test-key",
        api_secret="test-secret",
        base_url="https://example.invalid",
        ws_url="ws://example.invalid",
        request_connect_timeout=1.0,
        request_read_timeout=1.0,
    )


def realtime_message(rows):
    encoded = []
    for row in rows:
        encoded.extend(row)
    return f"0|{TRADE_TR_ID}|{len(rows):03d}|{'^'.join(encoded)}"


def trade_row(symbol, time_text, price):
    values = ["0"] * TRADE_ROW_WIDTH
    values[0] = str(symbol)
    values[1] = str(time_text)
    values[2] = str(price)
    values[5] = "2.50"
    values[7] = str(price - 1_000)
    values[8] = str(price + 500)
    values[9] = str(price - 1_500)
    values[10] = str(price + 5)
    values[11] = str(price - 5)
    values[12] = "25"
    values[13] = "12345"
    values[14] = "987654321"
    values[19] = "4000"
    values[20] = "6000"
    values[33] = "20260804"
    values[36] = "700"
    values[37] = "900"
    values[38] = "7000"
    values[39] = "9000"
    values[46] = "2"
    return values


class MarketQuoteStreamTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 8, 4, 9, 1, 2)
        self.stream = MarketQuoteStream(config())

    def test_approval_key_uses_long_setup_timeout_without_changing_request_defaults(self):
        response = Mock()
        response.json.return_value = {"approval_key": "test-approval"}
        for configured, expected in (((3.05, 10.0), (10.0, 60.0)), ((20.0, 90.0), (20.0, 90.0))):
            with self.subTest(configured=configured):
                self.stream.config.request_connect_timeout = configured[0]
                self.stream.config.request_read_timeout = configured[1]
                with patch("src.market_stream.requests.post", return_value=response) as post:
                    self.assertEqual(self.stream._approval_key(), "test-approval")

                self.assertEqual(post.call_args.args[0], "https://example.invalid/oauth2/Approval")
                self.assertEqual(post.call_args.kwargs["timeout"], expected)
                self.assertEqual(self.stream.config.request_connect_timeout, configured[0])
                self.assertEqual(self.stream.config.request_read_timeout, configured[1])

    def test_approval_timeout_retries_before_opening_websocket(self):
        response = Mock()
        response.json.return_value = {"approval_key": "test-approval"}
        with (
            patch("src.market_stream.requests.post", side_effect=[
                requests.exceptions.ReadTimeout("approval timeout"), response,
            ]) as post,
            patch("src.market_stream.asyncio.sleep", new_callable=AsyncMock) as sleep,
            patch.object(self.stream, "_run_connection", new_callable=AsyncMock) as connect,
        ):
            asyncio.run(self.stream._run())

        self.assertEqual(post.call_count, 2)
        self.assertTrue(all(call.kwargs["timeout"] == (10.0, 60.0) for call in post.call_args_list))
        sleep.assert_awaited_once_with(1.0)
        connect.assert_awaited_once()
        self.assertEqual(connect.call_args.args[1], "test-approval")

    def test_trade_frame_contains_executable_prices(self):
        self.stream.update_symbols(["005930"])
        self.stream.feed_realtime_message(
            realtime_message([trade_row("005930", "090102", 80_000)]),
            received_at=self.now,
        )

        quotes = self.stream.drain_quotes()

        self.assertEqual(len(quotes), 1)
        quote = quotes[0]
        self.assertEqual(quote.timestamp, self.now)
        self.assertEqual(quote.ask_price, 80_005)
        self.assertEqual(quote.bid_price, 79_995)
        self.assertEqual(quote.trade_volume, 25)
        self.assertEqual(quote.cumulative_volume, 12_345)
        self.assertEqual(quote.cumulative_sell_volume, 4_000)
        self.assertEqual(quote.cumulative_buy_volume, 6_000)
        self.assertEqual(quote.open_price, 79_000)
        self.assertEqual(quote.high_price, 80_500)
        self.assertEqual(quote.low_price, 78_500)
        self.assertEqual(quote.day_change_pct, 2.5)
        self.assertEqual(quote.ask_size, 700)
        self.assertEqual(quote.bid_size, 900)
        self.assertEqual(quote.total_ask_size, 7_000)
        self.assertEqual(quote.total_bid_size, 9_000)
        self.assertTrue(quote.book_available)
        self.assertTrue(quote.flow_available)
        self.assertTrue(quote.book_depth_available)
        self.assertTrue(quote.day_summary_available)

    def test_alphanumeric_stock_is_subscribed_and_parsed_without_code_conversion(self):
        self.stream.update_symbols(["18064K", "0000Z0"])
        self.stream.feed_realtime_message(realtime_message([
            trade_row(symbol, "090102", 50_000) for symbol in ("18064K", "0000Z0")
        ]), received_at=self.now)
        self.assertEqual([quote.symbol for quote in self.stream.drain_quotes()], ["0000Z0", "18064K"])

    def test_latest_book_survives_drain_and_cannot_be_mutated_by_order_worker(self):
        self.stream.update_symbols(["005930"])
        self.stream.feed_realtime_message(
            realtime_message([trade_row("005930", "090102", 80_000)]), received_at=self.now)
        self.stream.drain_quotes()
        latest = self.stream.latest_quote("005930", now=self.now)
        self.assertEqual(latest.bid_price, 79_995)
        latest.bid_price = 1
        self.assertEqual(self.stream.latest_quote("005930", now=self.now).bid_price, 79_995)
        self.assertIsNone(self.stream.latest_quote("005930", now=self.now + timedelta(seconds=6)))
        self.assertIsNone(self.stream.latest_quote("005930", now=self.now - timedelta(seconds=1)))
        self.stream.stop()
        self.assertIsNone(self.stream.latest_quote("005930", now=self.now))

    def test_rest_book_updates_execution_snapshot_without_impersonating_stream(self):
        q = Quote("033780", 179000, self.now, bid_price=178900, ask_price=179000,
                  book_available=True)
        self.stream.observe_quotes([q])
        q.bid_price = 1
        self.assertEqual(self.stream.latest_quote("033780", now=self.now).bid_price, 178900)
        fallen = replace(q, timestamp=self.now + timedelta(seconds=1), bid_price=178400)
        self.stream.observe_quotes([fallen])
        self.stream.observe_quotes([replace(q, bid_price=178900)])
        self.assertEqual(self.stream.latest_quote("033780", now=fallen.timestamp).bid_price, 178400)
        self.assertEqual(self.stream.drain_quotes(), [])
        self.assertEqual(self.stream.stale_symbols(["033780"], now=fallen.timestamp), ["033780"])

    def test_limit_up_trade_without_ask_is_kept_for_price_history(self):
        row = trade_row("005930", "090102", 80_000)
        row[10] = "0"
        self.stream.update_symbols(["005930"])

        parsed = self.stream.feed_realtime_message(
            realtime_message([row]),
            received_at=self.now,
        )

        self.assertEqual(parsed, 1)
        self.assertEqual(self.stream.drain_quotes()[0].ask_price, 0)

    def test_same_second_peak_and_latest_are_preserved_without_synthetic_prices(self):
        self.stream.update_symbols(["005930"])
        peak = trade_row("005930", "090102", 80_100)
        latest = trade_row("005930", "090102", 80_000)
        latest[36], latest[37] = "70", "90"
        for row in (peak, latest):
            self.stream.feed_realtime_message(realtime_message([row]), received_at=self.now)
        quotes = self.stream.drain_quotes()
        self.assertEqual([(q.current_price, q.ask_price, q.bid_price, q.ask_size, q.bid_size)
                          for q in quotes], [(80000, 80005, 79995, 70, 90)])
        self.assertEqual(quotes[0].ask_high_price, 80105)
        self.assertEqual(limit_fill(quotes[0], 80005, 100), (70, 80005))
        self.assertEqual(market_fill(quotes[0], 100), (90, 79995))
        self.assertTrue(all(q.timestamp == self.now for q in quotes))
        self.assertEqual(self.stream.drain_quotes(), [])

        # A rising last event is already the peak; emit it only once.
        self.stream.feed_realtime_message(realtime_message([latest, peak]), received_at=self.now)
        self.assertEqual(len(self.stream.drain_quotes()), 1)
        next_second = trade_row("005930", "090103", 80000)
        self.stream.feed_realtime_message(realtime_message([next_second]),
                                          received_at=self.now + timedelta(seconds=1))
        self.assertEqual(self.stream.drain_quotes()[0].ask_high_price, 80005)

    def test_trade_peak_is_preserved_even_when_the_book_is_below_the_last_trade(self):
        for drain_between in (False, True):
            with self.subTest(drain_between=drain_between):
                stream = MarketQuoteStream(config())
                stream.update_symbols(["005930"])
                peak = trade_row("005930", "090102", 80100)
                peak[10], peak[11] = "80005", "79995"
                last = trade_row("005930", "090102", 80000)
                stream.feed_realtime_message(realtime_message([peak]), received_at=self.now)
                if drain_between:
                    stream.drain_quotes()
                stream.feed_realtime_message(realtime_message([last]), received_at=self.now)
                q = stream.drain_quotes()[0]
                self.assertEqual((q.current_price, q.bid_price, q.ask_price), (80000, 79995, 80005))
                self.assertEqual(q.trade_high_price, 80100)
                self.assertEqual(q.ask_high_price, 80005)
                self.assertEqual(stream.latest_quote(q.symbol, now=self.now).trade_high_price, 80100)
                stream.observe_quotes([replace(q, current_price=80050, trade_high_price=0)])
                self.assertEqual(stream.latest_quote(q.symbol, now=self.now).trade_high_price, 80100)
                later = replace(q, timestamp=self.now + timedelta(seconds=1), trade_high_price=0)
                stream.observe_quotes([later])
                self.assertEqual(stream.latest_quote(q.symbol, now=later.timestamp).trade_high_price, 0)

    def test_rest_fractional_timestamp_cannot_erase_same_second_stream_peaks(self):
        for stream_first in (False, True):
            with self.subTest(stream_first=stream_first):
                stream = MarketQuoteStream(config())
                stream.update_symbols(["005930"])
                rest = Quote("005930", 80050, self.now + timedelta(microseconds=500000),
                             bid_price=80040, ask_price=80050, book_available=True,
                             ask_size=70, bid_size=90, book_depth_available=True)
                peak = realtime_message([trade_row(rest.symbol, "090102", 80100)])
                if stream_first:
                    stream.feed_realtime_message(peak, received_at=self.now)
                    stream.drain_quotes()
                    stream.observe_quotes([rest])
                else:
                    stream.observe_quotes([rest])
                    self.assertEqual(stream.feed_realtime_message(
                        peak, received_at=self.now + timedelta(microseconds=800000)), 1)
                    quotes = stream.drain_quotes()
                    self.assertEqual(len(quotes), 1)
                    self.assertEqual(quotes[0].trade_high_price, 80100)
                latest = stream.latest_quote(rest.symbol, now=rest.timestamp)
                self.assertEqual((latest.timestamp, latest.current_price, latest.bid_price,
                                  latest.ask_price, latest.ask_size, latest.bid_size),
                                 (rest.timestamp, 80050, 80040, 80050, 70, 90))
                self.assertEqual((latest.trade_high_price, latest.ask_high_price), (80100, 80105))
                self.assertFalse(latest.is_price_breakout(80000))

                # Keep the older observation for history without repricing the latest book.
                self.assertEqual(stream.feed_realtime_message(
                    realtime_message([trade_row(rest.symbol, "090101", 80200)]),
                    received_at=rest.timestamp), 1)
                history = stream.drain_quotes()
                self.assertIn(self.now - timedelta(seconds=1), [q.timestamp for q in history])
                self.assertEqual(stream.latest_quote(rest.symbol, now=rest.timestamp).trade_high_price, 80100)
                later = replace(rest, timestamp=self.now + timedelta(seconds=1))
                stream.observe_quotes([later])
                self.assertEqual(stream.latest_quote(rest.symbol, now=later.timestamp).trade_high_price, 0)
                self.assertIsNone(stream.latest_quote(rest.symbol, now=later.timestamp + timedelta(seconds=6)))

    def test_same_second_peak_survives_missing_ask_but_not_unsubscribe_or_reset(self):
        self.stream.update_symbols(["005930"])
        peak = trade_row("005930", "090102", 80_100)
        no_ask = trade_row("005930", "090102", 80_000)
        no_ask[10] = "0"
        self.stream.feed_realtime_message(realtime_message([peak, no_ask]), received_at=self.now)
        quotes = self.stream.drain_quotes()
        self.assertEqual([q.ask_price for q in quotes], [0])
        self.assertEqual(quotes[0].ask_high_price, 80105)
        for reset in (self.stream.stop, lambda: self.stream.update_symbols([])):
            self.stream.update_symbols(["005930"])
            self.stream.feed_realtime_message(realtime_message([peak, no_ask]), received_at=self.now)
            reset()
            self.assertEqual(self.stream.drain_quotes(), [])

    def test_multi_trade_frame_parses_each_declared_row(self):
        self.stream.update_symbols(["005930", "000660"])
        parsed = self.stream.feed_realtime_message(
            realtime_message(
                [
                    trade_row("005930", "090102", 80_000),
                    trade_row("000660", "090102", 200_000),
                ],
            ),
            received_at=self.now,
        )

        self.assertEqual(parsed, 2)
        self.assertEqual([quote.symbol for quote in self.stream.drain_quotes()], ["000660", "005930"])

    def test_legacy_and_current_frames_keep_every_row_aligned(self):
        symbols = [f"{i + 1:06d}" for i in range(40)]
        self.stream.update_symbols(symbols)
        for width in (46, 47):
            for count in (1, 2, 3, 5, 9, 40):
                with self.subTest(width=width, count=count):
                    rows = [
                        trade_row(symbol, "090102", 10_000 + i * 100)[:width]
                        for i, symbol in enumerate(symbols[:count])
                    ]
                    with patch("src.market_stream.logger.warning") as warning:
                        parsed = self.stream.feed_realtime_message(
                            realtime_message(rows), received_at=self.now,
                        )
                    self.assertEqual(parsed, count)
                    warning.assert_not_called()
                    quotes = self.stream.drain_quotes()
                    self.assertEqual([q.symbol for q in quotes], symbols[:count])
                    for i, quote in enumerate(quotes):
                        self.assertEqual(quote.current_price, 10_000 + i * 100)
                        self.assertEqual(quote.ask_price, 10_005 + i * 100)
                        self.assertEqual(quote.bid_price, 9_995 + i * 100)
                        self.assertEqual(quote.cumulative_buy_volume, 6_000)
                        self.assertEqual(quote.total_bid_size, 9_000)
                    self.assertEqual(self.stream.stale_symbols(symbols[:count], now=self.now), [])

    def test_unknown_frame_shape_warnings_are_throttled_without_blocking_valid_frames(self):
        self.stream.update_symbols(["005930"])
        row = trade_row("005930", "090102", 80_000)
        with (
            patch("src.market_stream.monotonic", side_effect=[100.0, 100.1, 129.9, 130.0]),
            patch("src.market_stream.logger.warning") as warning,
        ):
            for _ in range(4):
                self.assertEqual(self.stream.feed_realtime_message(
                    realtime_message([row + ["extra"]]), received_at=self.now,
                ), 0)
            self.assertEqual(warning.call_count, 2)
            self.assertEqual(self.stream.feed_realtime_message(
                realtime_message([row]), received_at=self.now,
            ), 1)
        self.assertEqual(self.stream.drain_quotes()[0].current_price, 80_000)

    def test_malformed_frame_cannot_shift_fields_into_another_symbol(self):
        self.stream.update_symbols(["005930", "252500"])
        row = trade_row("005930", "090102", 252_500)
        malformed = [
            realtime_message([row[:-1], row]),
            realtime_message([row + ["extra"], row]),
            realtime_message([row]).replace("|001|", "|002|"),
            realtime_message([row]).replace("|001|", "|0|"),
            realtime_message([row]).replace("|001|", "|1.0|"),
            "1" + realtime_message([row])[1:],
        ]
        for message in malformed:
            with self.subTest(message=message[:40]):
                self.assertEqual(
                    self.stream.feed_realtime_message(message, received_at=self.now), 0,
                )
                self.assertEqual(self.stream.drain_quotes(), [])
        self.assertEqual(self.stream.stale_symbols(["005930", "252500"], now=self.now),
                         ["005930", "252500"])

    def test_invalid_exchange_time_and_crossed_prices_do_not_become_fresh_quotes(self):
        self.stream.update_symbols(["252500"])
        for field, value in ((1, "99"), (1, "256199"), (33, "0"),
                             (33, "20260803"), (10, "99"), (11, "2177948"),
                             (2, "9999999")):
            with self.subTest(field=field, value=value):
                row = trade_row("252500", "090102", 2_235)
                row[field] = value
                self.assertEqual(self.stream.feed_realtime_message(
                    realtime_message([row]), received_at=self.now), 0)
                self.assertEqual(self.stream.drain_quotes(), [])
        valid = trade_row("252500", "090102", 2_235)
        self.assertEqual(self.stream.feed_realtime_message(
            realtime_message([valid]), received_at=self.now), 1)
        self.assertEqual(self.stream.drain_quotes()[0].current_price, 2_235)

    def test_subscription_list_preserves_priority_and_caps_single_connection(self):
        initial = [f"{index:06d}" for index in range(45)]
        self.stream.update_symbols(initial)
        self.stream.update_symbols(initial[5:] + ["900001", "900002"])

        self.assertEqual(
            self.stream.subscribed_symbols(),
            initial[5:5 + MAX_SYMBOLS_PER_CONNECTION],
        )

    def test_pingpong_payload_is_answered_without_parsing_as_market_data(self):
        class WebSocket:
            def __init__(self):
                self.payload = None

            async def pong(self, payload):
                self.payload = payload

        websocket = WebSocket()
        raw = json.dumps({"header": {"tr_id": "PINGPONG"}})

        asyncio.run(self.stream._handle_message(websocket, raw))

        self.assertEqual(websocket.payload, raw.encode("utf-8"))

    def test_dynamic_subscription_sync_sends_only_added_and_removed_symbols(self):
        class WebSocket:
            def __init__(self):
                self.messages = []

            async def send(self, message):
                self.messages.append(json.loads(message))

        websocket = WebSocket()
        subscribed = {"005930", "000660"}
        asyncio.run(
            self.stream._sync_subscriptions(
                websocket,
                "approval",
                subscribed,
                {"000660", "035420"},
            )
        )

        changes = [
            (
                message["header"]["tr_type"],
                message["body"]["input"]["tr_key"],
            )
            for message in websocket.messages
        ]
        self.assertEqual(
            changes,
            [("2", "005930"), ("1", "035420")],
        )
        self.assertEqual(subscribed, {"000660", "035420"})

    def test_connection_reconnects_and_restores_trade_subscriptions(self):
        stream = self.stream
        stream.update_symbols(["005930"])
        attempts = []
        sent = []
        subscribed = asyncio.Event()

        class WebSocket:
            async def send(self, message):
                sent.append(json.loads(message))
                subscribed.set()

            async def recv(self):
                await subscribed.wait()
                stream._stop_event.set()
                raise asyncio.TimeoutError

        class Connection:
            def __init__(self, attempt):
                self.attempt = attempt

            async def __aenter__(self):
                if self.attempt == 1:
                    raise ConnectionError("simulated disconnect")
                return WebSocket()

            async def __aexit__(self, *_args):
                return False

        class WebSockets:
            @staticmethod
            def connect(*_args, **_kwargs):
                attempt = len(attempts) + 1
                attempts.append(attempt)
                return Connection(attempt)

        with patch("src.market_stream.asyncio.sleep", new=AsyncMock()):
            asyncio.run(stream._run_connection(WebSockets, "approval"))

        self.assertEqual(attempts, [1, 2])
        self.assertEqual(
            {
                (message["body"]["input"]["tr_id"], message["body"]["input"]["tr_key"])
                for message in sent
            },
            {(TRADE_TR_ID, "005930")},
        )

    def test_stale_symbols_require_fresh_receive_and_exchange_times(self):
        self.stream.update_symbols(["005930"])
        self.stream.feed_realtime_message(
            realtime_message([trade_row("005930", "090102", 80_000)]),
            received_at=self.now,
        )

        self.assertEqual(
            self.stream.stale_symbols(
                ["005930", "000660"],
                now=self.now + timedelta(seconds=4),
            ),
            ["000660"],
        )
        self.assertEqual(self.stream.stale_symbols(["005930"], now=self.now+timedelta(seconds=2),
                                                  max_age_seconds=1), ["005930"])
        self.assertIsNotNone(self.stream.latest_quote("005930", now=self.now+timedelta(seconds=2)))
        # Refresh scheduling must not relax the execution freshness limit.
        self.assertEqual(self.stream.stale_symbols(["005930"], now=self.now+timedelta(seconds=6),
                                                  max_age_seconds=10), ["005930"])

        # Receiving a delayed or duplicate frame cannot refresh its exchange-time book.
        later = self.now + timedelta(seconds=6)
        self.stream.feed_realtime_message(
            realtime_message([trade_row("005930", "090102", 80_000)]), received_at=later,
        )
        self.assertEqual(self.stream.stale_symbols(["005930"], now=later), ["005930"])
        self.stream.feed_realtime_message(
            realtime_message([trade_row("005930", "090108", 80_000)]), received_at=later,
        )
        self.assertEqual(self.stream.stale_symbols(["005930"], now=later), [])
        self.assertEqual(self.stream.stale_symbols(["005930"], now=self.now), ["005930"])

    def test_rest_cannot_make_a_delayed_stream_event_healthy(self):
        self.stream.update_symbols(["005930"])
        rest = Quote("005930", 80000, self.now, bid_price=79995, ask_price=80005,
                     book_available=True)
        self.stream.observe_quotes([rest])
        self.stream.feed_realtime_message(
            realtime_message([trade_row("005930", "090055", 79000)]), received_at=self.now)
        self.assertEqual(self.stream.latest_quote("005930", now=self.now), rest)
        self.assertEqual(self.stream.stale_symbols(["005930"], now=self.now), ["005930"])
        self.assertEqual(self.stream.drain_quotes()[0].timestamp,
                         self.now - timedelta(seconds=7))

        newer = replace(rest, timestamp=self.now + timedelta(microseconds=500000),
                        current_price=81000, bid_price=80995, ask_price=81005)
        self.stream.observe_quotes([newer])
        self.stream.feed_realtime_message(
            realtime_message([trade_row("005930", "090102", 80000)]), received_at=newer.timestamp)
        historical = self.stream.drain_quotes()[0]
        self.assertEqual((historical.timestamp, historical.trade_high_price), (self.now, 80000))
        self.assertEqual(self.stream.latest_quote("005930", now=newer.timestamp).current_price, 81000)

    def test_subscription_updates_do_not_pause_prices_or_pingpong(self):
        stream = self.stream
        stream.update_symbols(["005930"])
        received = []

        async def scenario():
            sending = asyncio.Event()
            stopped = asyncio.Event()
            never = asyncio.Event()

            class WebSocket:
                async def send(self, message):
                    sending.set()
                    try:
                        await never.wait()
                    finally:
                        stopped.set()

                async def recv(self):
                    await sending.wait()
                    if not received:
                        received.append("trade")
                        return realtime_message([trade_row("005930", "090102", 80000)])
                    received.append("ping")
                    return json.dumps({"header": {"tr_id": "PINGPONG"}})

                async def pong(self, payload):
                    self_payload.append(payload)
                    stream._stop_event.set()

            class Connection:
                async def __aenter__(self):
                    return WebSocket()

                async def __aexit__(self, *args):
                    return False

            with patch("src.market_stream.datetime", wraps=datetime) as clock:
                clock.now.return_value = self.now
                await asyncio.wait_for(stream._run_connection(
                    SimpleNamespace(connect=lambda *args, **kwargs: Connection()), "approval"), 1)
            self.assertTrue(stopped.is_set())

        self_payload = []
        asyncio.run(scenario())
        self.assertEqual(received, ["trade", "ping"])
        self.assertEqual(len(self_payload), 1)
        self.assertEqual(self.stream.drain_quotes()[0].current_price, 80000)

    def test_stop_discards_snapshots_from_previous_session(self):
        self.stream.update_symbols(["005930"])
        self.stream.feed_realtime_message(
            realtime_message([trade_row("005930", "090102", 80_000)]),
            received_at=self.now,
        )

        self.stream.stop()

        self.assertEqual(self.stream.drain_quotes(), [])
        self.assertEqual(
            self.stream.stale_symbols(["005930"], now=self.now),
            ["005930"],
        )

    def test_stop_joins_old_producer_before_clearing_its_session_state(self):
        old = Mock()
        old.is_alive.return_value = True
        self.stream._thread = old
        self.stream.stop()
        old.join.assert_called_once_with()
        self.assertTrue(self.stream._stop_event.is_set())
        self.assertIsNone(self.stream._thread)


if __name__ == "__main__":
    unittest.main()
