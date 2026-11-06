from __future__ import annotations

import asyncio
import json
import logging
import threading
from contextlib import suppress
from dataclasses import replace
from datetime import datetime
from time import monotonic
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import requests

from src.config import Config
from src.models import Quote


logger = logging.getLogger("kis_trader.market_stream")

TRADE_TR_ID = "H0STCNT0"
MAX_SYMBOLS_PER_CONNECTION = 40
# MARKET_CLS_CODE was appended to the original 46-field H0STCNT0 row.
TRADE_ROW_WIDTH = 47

def _integer(value: Any) -> int:
    try:
        return int(float(str(value or "0").replace(",", "")))
    except (TypeError, ValueError):
        return 0


def _number(value: Any) -> float:
    try:
        return float(str(value or "0").replace(",", ""))
    except (TypeError, ValueError):
        return 0.0


def _exchange_timestamp(date_text: str, time_text: str) -> datetime | None:
    normalized_date = str(date_text or "").strip()
    normalized_time = str(time_text or "").strip()
    if (
        len(normalized_date) != 8
        or len(normalized_time) != 6
        or not normalized_date.isascii()
        or not normalized_date.isdigit()
        or not normalized_time.isascii()
        or not normalized_time.isdigit()
    ):
        return None
    try:
        return datetime.strptime(normalized_date + normalized_time, "%Y%m%d%H%M%S")
    except ValueError:
        return None


def _quote_from_trade(values: Sequence[str], received_at: datetime) -> Quote | None:
    if len(values) not in (TRADE_ROW_WIDTH - 1, TRADE_ROW_WIDTH):
        return None

    def value(index: int) -> str:
        return values[index] if index < len(values) else ""

    symbol = str(value(0) or "").strip()
    current_price = max(0, _integer(value(2)))
    ask_price = max(0, _integer(value(10)))
    bid_price = max(0, _integer(value(11)))
    open_price = max(0, _integer(value(7)))
    high_price = max(0, _integer(value(8)))
    low_price = max(0, _integer(value(9)))
    timestamp = _exchange_timestamp(value(33), value(1))
    if (
        len(symbol) != 6
        or not symbol.isalnum()
        or not symbol.isascii()
        or timestamp is None
        or timestamp.date() != received_at.date()
    ):
        return None
    quote = Quote(
        symbol=symbol,
        current_price=current_price,
        timestamp=timestamp,
        ask_price=ask_price,
        bid_price=bid_price,
        trade_volume=max(0, _integer(value(12))),
        cumulative_volume=max(0, _integer(value(13))),
        cumulative_trade_amount=max(0, _integer(value(14))),
        cumulative_sell_volume=max(0, _integer(value(19))),
        cumulative_buy_volume=max(0, _integer(value(20))),
        open_price=open_price,
        high_price=high_price,
        low_price=low_price,
        day_change_pct=_number(value(5)),
        ask_size=max(0, _integer(value(36))),
        bid_size=max(0, _integer(value(37))),
        total_ask_size=max(0, _integer(value(38))),
        total_bid_size=max(0, _integer(value(39))),
        book_available=True,
        flow_available=bool(str(value(19)).strip() and str(value(20)).strip()),
        book_depth_available=all(
            str(value(index)).strip()
            for index in (36, 37, 38, 39)
        ),
        day_summary_available=bool(
            str(value(5)).strip()
            and open_price > 0
            and high_price > 0
            and low_price > 0
        ),
    )
    return quote if quote.has_valid_prices else None


class MarketQuoteStream:
    """KIS trade stream converted to exchange-time one-second quotes."""

    def __init__(self, config: Config, *, stale_seconds: float = 5.0):
        self.config = config
        self.stale_seconds = max(1.0, float(stale_seconds))
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._symbols: List[str] = []
        self._snapshots: Dict[tuple[str, datetime], Quote] = {}
        self._latest_quotes: Dict[str, Quote] = {}
        self._stream_observed_at: Dict[str, datetime] = {}
        self._connection_error = ""
        self._next_subscription_send_at = 0.0
        self._last_frame_warning_at = float("-inf")

    @staticmethod
    def _normalize_symbols(symbols: Iterable[str]) -> List[str]:
        normalized = []
        seen = set()
        for symbol in symbols:
            value = str(symbol or "").strip()
            if not value or value in seen:
                continue
            seen.add(value)
            normalized.append(value)
        return normalized

    def update_symbols(self, symbols: Iterable[str]) -> None:
        desired_order = self._normalize_symbols(symbols)[:MAX_SYMBOLS_PER_CONNECTION]
        desired = set(desired_order)
        with self._lock:
            removed = set(self._symbols) - desired
            self._symbols = desired_order
            for symbol in removed:
                self._latest_quotes.pop(symbol, None)
                self._stream_observed_at.pop(symbol, None)
            if removed:
                self._snapshots = {
                    key: quote
                    for key, quote in self._snapshots.items()
                    if key[0] not in removed
                }

    def subscribed_symbols(self) -> List[str]:
        with self._lock:
            return list(self._symbols)

    def _desired_symbols(self) -> set[str]:
        with self._lock:
            return set(self._symbols)

    def start(self, symbols: Iterable[str] = ()) -> None:
        self.update_symbols(symbols)
        if self._thread is not None and self._thread.is_alive():
            return
        self._clear_quote_state()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._thread_main,
            name="kis-market-stream",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            # Approval HTTP may outlive five seconds. Never clear its stop event
            # or start another producer while that old session is still alive.
            thread.join()
        self._thread = None
        self._clear_quote_state()

    def _clear_quote_state(self) -> None:
        with self._lock:
            self._snapshots = {}
            self._latest_quotes = {}
            self._stream_observed_at = {}
            self._connection_error = ""

    def _approval_key(self) -> str:
        payload = {
            "grant_type": "client_credentials",
            "appkey": self.config.api_key,
            "secretkey": self.config.api_secret,
        }
        response = requests.post(
            f"{self.config.base_url.rstrip('/')}/oauth2/Approval",
            data=json.dumps(payload),
            headers={"content-type": "application/json; charset=utf-8"},
            timeout=(
                max(10.0, float(self.config.request_connect_timeout)),
                max(60.0, float(self.config.request_read_timeout)),
            ),
        )
        response.raise_for_status()
        approval_key = str(response.json().get("approval_key") or "").strip()
        if not approval_key:
            raise RuntimeError("KIS WebSocket approval_key가 비어 있습니다.")
        return approval_key

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._run())
        except Exception:
            logger.exception("KIS 실시간 시세 스트림이 종료되었습니다.")

    async def _run(self) -> None:
        try:
            import websockets
        except ImportError as exc:
            raise RuntimeError(
                "실시간 시세 구동에 websockets 패키지가 필요합니다. requirements.txt를 설치해 주십시오."
            ) from exc

        approval_backoff = 1.0
        approval_key = ""
        while not self._stop_event.is_set() and not approval_key:
            try:
                approval_key = await asyncio.to_thread(self._approval_key)
            except Exception as exc:
                logger.warning(
                    "KIS WebSocket 승인키 재시도: %.1fs 후 (%s)",
                    approval_backoff,
                    f"{type(exc).__name__}: {exc}",
                )
                await asyncio.sleep(approval_backoff)
                approval_backoff = min(30.0, approval_backoff * 2.0)
        if not approval_key:
            return
        self._next_subscription_send_at = 0.0
        await self._run_connection(websockets, approval_key)

    def _subscription_message(
        self,
        approval_key: str,
        *,
        tr_id: str,
        symbol: str,
        subscribe: bool,
    ) -> str:
        return json.dumps(
            {
                "header": {
                    "approval_key": approval_key,
                    "custtype": "P",
                    "tr_type": "1" if subscribe else "2",
                    "content-type": "utf-8",
                },
                "body": {"input": {"tr_id": tr_id, "tr_key": symbol}},
            },
            separators=(",", ":"),
        )

    async def _sync_subscriptions(
        self,
        websocket,
        approval_key: str,
        subscribed: set[str],
        desired: set[str],
    ) -> None:
        for symbol in sorted(subscribed - desired):
            await self._send_subscription(
                websocket,
                self._subscription_message(
                    approval_key,
                    tr_id=TRADE_TR_ID,
                    symbol=symbol,
                    subscribe=False,
                ),
            )
            subscribed.remove(symbol)
        for symbol in sorted(desired - subscribed):
            await self._send_subscription(
                websocket,
                self._subscription_message(
                    approval_key,
                    tr_id=TRADE_TR_ID,
                    symbol=symbol,
                    subscribe=True,
                ),
            )
            subscribed.add(symbol)

    async def _send_subscription(self, websocket, message: str) -> None:
        loop = asyncio.get_running_loop()
        wait = self._next_subscription_send_at - loop.time()
        if wait > 0.0:
            await asyncio.sleep(wait)
        await websocket.send(message)
        self._next_subscription_send_at = loop.time() + 0.1

    async def _run_connection(self, websockets_module, approval_key: str) -> None:
        backoff = 1.0
        url = f"{self.config.ws_url.rstrip('/')}/tryitout"
        while not self._stop_event.is_set():
            if not self._desired_symbols():
                await asyncio.sleep(0.5)
                continue
            subscribed: set[str] = set()
            try:
                async with websockets_module.connect(
                    url,
                    ping_interval=None,
                    close_timeout=3,
                    max_queue=4096,
                ) as websocket:
                    with self._lock:
                        self._connection_error = ""
                    logger.info("KIS 실시간 시세 연결: 대상=%d종목", len(self._desired_symbols()))
                    backoff = 1.0
                    subscription_task = None
                    try:
                        while not self._stop_event.is_set():
                            desired = self._desired_symbols()
                            if not desired:
                                return
                            if subscription_task is not None and subscription_task.done():
                                subscription_task.result()
                                subscription_task = None
                            if subscription_task is None and subscribed != desired:
                                # Sending up to 40 subscriptions takes seconds;
                                # prices and PINGPONG must keep draining meanwhile.
                                subscription_task = asyncio.create_task(self._sync_subscriptions(
                                    websocket, approval_key, subscribed, desired))
                            try:
                                raw = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                            except asyncio.TimeoutError:
                                continue
                            await self._handle_message(websocket, raw)
                    finally:
                        if subscription_task is not None:
                            subscription_task.cancel()
                            with suppress(asyncio.CancelledError):
                                await subscription_task
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                with self._lock:
                    previous = self._connection_error
                    self._connection_error = message
                if previous != message:
                    logger.warning(
                        "KIS 실시간 시세 재연결 대기: %.1fs 사유=%s",
                        backoff,
                        message,
                    )
                await asyncio.sleep(backoff)
                backoff = min(30.0, backoff * 2.0)

    async def _handle_message(self, websocket, raw: str | bytes) -> None:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        if not raw:
            return
        if raw[0] in ("0", "1"):
            self.feed_realtime_message(raw, received_at=datetime.now())
            return
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return
        header = payload.get("header") if isinstance(payload, Mapping) else None
        tr_id = str((header or {}).get("tr_id") or "")
        if tr_id == "PINGPONG":
            await websocket.pong(raw.encode("utf-8"))
            return
        body = payload.get("body") if isinstance(payload, Mapping) else None
        if isinstance(body, Mapping) and str(body.get("rt_cd") or "0") != "0":
            logger.warning(
                "KIS 실시간 구독 응답 실패: tr_id=%s msg=%s",
                tr_id or "-",
                str(body.get("msg1") or "-")[:160],
            )

    def feed_realtime_message(self, raw: str, *, received_at: datetime) -> int:
        parts = str(raw or "").split("|", 3)
        if len(parts) != 4 or parts[0] != "0":
            return 0
        tr_id = parts[1]
        if tr_id != TRADE_TR_ID:
            return 0
        values = parts[3].split("^")
        if not parts[2].isascii() or not parts[2].isdigit():
            return 0
        declared_count = int(parts[2])
        if declared_count <= 0:
            return 0
        row_width, remainder = divmod(len(values), declared_count)
        if remainder or row_width not in (TRADE_ROW_WIDTH - 1, TRADE_ROW_WIDTH):
            now = monotonic()
            if now - self._last_frame_warning_at >= 30.0:
                logger.warning(
                    "KIS trade frame shape mismatch: rows=%d fields=%d",
                    declared_count,
                    len(values),
                )
                self._last_frame_warning_at = now
            return 0
        parsed_count = 0
        for index in range(declared_count):
            start = index * row_width
            parsed_count += self._merge_row(
                values[start:start + row_width],
                received_at,
            )
        return parsed_count

    def _merge_row(self, values: Sequence[str], received_at: datetime) -> int:
        quote = _quote_from_trade(values, received_at)
        if quote is None:
            return 0
        symbol = quote.symbol
        with self._lock:
            if symbol not in self._symbols:
                return 0
            previous = self._latest_quotes.get(symbol)
            # A new REST book must not make a delayed stream event look current.
            observed_at = min(received_at, quote.timestamp)
            self._stream_observed_at[symbol] = max(
                observed_at, self._stream_observed_at.get(symbol, observed_at))
            key = (symbol, quote.timestamp)
            previous_snapshot = self._snapshots.get(key)
            quote.observe_price_peaks(quote)
            for observed in (previous, previous_snapshot):
                if observed is not None and observed.timestamp <= quote.timestamp:
                    quote.observe_price_peaks(observed)
            self._snapshots[key] = quote
            latest = replace(quote if previous is None or quote.timestamp >= previous.timestamp else previous)
            if previous is not None:
                latest.observe_price_peaks(previous if latest.timestamp == quote.timestamp else quote)
            self._latest_quotes[symbol] = latest
            if len(self._snapshots) > 20_000:
                oldest = min(self._snapshots, key=lambda item: item[1])
                self._snapshots.pop(oldest, None)
            return 1

    def observe_quotes(self, quotes: Iterable[Quote]) -> None:
        """Publish REST books for execution without marking the stream healthy."""
        with self._lock:
            for quote in quotes:
                previous = self._latest_quotes.get(quote.symbol)
                latest = replace(quote if previous is None or quote.timestamp >= previous.timestamp else previous)
                if previous is not None:
                    latest.observe_price_peaks(previous if latest.timestamp == quote.timestamp else quote)
                self._latest_quotes[quote.symbol] = latest

    def latest_quote(self, symbol: str, *, now: datetime | None = None) -> Quote | None:
        current = now or datetime.now()
        with self._lock:
            quote = self._latest_quotes.get(symbol)
            if quote is None or not 0 <= (current - quote.timestamp).total_seconds() <= self.stale_seconds:
                return None
            return replace(quote)

    def drain_quotes(self) -> List[Quote]:
        with self._lock:
            quotes = sorted(
                self._snapshots.values(),
                key=lambda quote: (quote.timestamp, quote.symbol),
            )
            self._snapshots = {}
            return quotes

    def stale_symbols(
        self,
        symbols: Iterable[str],
        *,
        now: datetime | None = None,
        max_age_seconds: float | None = None,
    ) -> List[str]:
        current = now or datetime.now()
        age_limit = self.stale_seconds if max_age_seconds is None else min(
            self.stale_seconds, max(0.0, float(max_age_seconds)))
        normalized = self._normalize_symbols(symbols)
        with self._lock:
            return [
                symbol
                for symbol in normalized
                if symbol not in self._stream_observed_at
                or symbol not in self._latest_quotes
                or not 0 <= (current - self._stream_observed_at[symbol]).total_seconds() <= age_limit
                or not 0 <= (current - self._latest_quotes[symbol].timestamp).total_seconds() <= age_limit
            ]
