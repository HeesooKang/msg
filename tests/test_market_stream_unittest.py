import asyncio
import json
import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src.market_stream import (
    MAX_SYMBOLS_PER_CONNECTION,
    TRADE_ROW_WIDTH,
    TRADE_TR_ID,
    MarketQuoteStream,
)


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
    return values


class MarketQuoteStreamTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 8, 4, 9, 1, 2)
        self.stream = MarketQuoteStream(config())

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
        self.assertEqual(quote.ask_size, 700)
        self.assertEqual(quote.bid_size, 900)
        self.assertEqual(quote.total_ask_size, 7_000)
        self.assertEqual(quote.total_bid_size, 9_000)
        self.assertTrue(quote.book_available)
        self.assertTrue(quote.flow_available)
        self.assertTrue(quote.book_depth_available)

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

        class WebSocket:
            async def send(self, message):
                sent.append(json.loads(message))

            async def recv(self):
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

    def test_stale_symbols_use_each_symbols_last_received_time(self):
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


if __name__ == "__main__":
    unittest.main()
