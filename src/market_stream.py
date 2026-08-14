from __future__ import annotations

import asyncio
import json
import logging
import threading
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import requests

from src.config import Config
from src.models import Quote


logger = logging.getLogger("kis_trader.market_stream")

TRADE_TR_ID = "H0STCNT0"
MAX_SYMBOLS_PER_CONNECTION = 40
TRADE_ROW_WIDTH = 46

def _integer(value: Any) -> int:
    try:
        return int(float(str(value or "0").replace(",", "")))
    except (TypeError, ValueError):
        return 0


def _exchange_timestamp(date_text: str, time_text: str, fallback: datetime) -> datetime:
    normalized_date = "".join(character for character in str(date_text or "") if character.isdigit())
    normalized_time = "".join(character for character in str(time_text or "") if character.isdigit())
    if len(normalized_date) != 8:
        normalized_date = fallback.strftime("%Y%m%d")
    if len(normalized_time) < 6:
        return fallback.replace(microsecond=0)
    try:
        return datetime.strptime(normalized_date + normalized_time[:6], "%Y%m%d%H%M%S")
    except ValueError:
        return fallback.replace(microsecond=0)


def _quote_from_trade(values: Sequence[str], received_at: datetime) -> Quote | None:
    def value(index: int) -> str:
        return values[index] if index < len(values) else ""

    symbol = str(value(0) or "").strip()
    current_price = max(0, _integer(value(2)))
    ask_price = max(0, _integer(value(10)))
    bid_price = max(0, _integer(value(11)))
    if not symbol or current_price <= 0:
        return None
    return Quote(
        symbol=symbol,
        current_price=current_price,
        timestamp=_exchange_timestamp(
            str(value(33) or ""),
            str(value(1) or ""),
            received_at,
        ),
        ask_price=ask_price,
        bid_price=bid_price,
        trade_volume=max(0, _integer(value(12))),
        cumulative_volume=max(0, _integer(value(13))),
        cumulative_trade_amount=max(0, _integer(value(14))),
        cumulative_sell_volume=max(0, _integer(value(19))),
        cumulative_buy_volume=max(0, _integer(value(20))),
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
    )


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
        self._latest_exchange_at: Dict[str, datetime] = {}
        self._received_at_by_symbol: Dict[str, datetime] = {}
        self._connection_error = ""
        self._next_subscription_send_at = 0.0

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
                self._latest_exchange_at.pop(symbol, None)
                self._received_at_by_symbol.pop(symbol, None)
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
            thread.join(timeout=5.0)
        self._thread = None
        self._clear_quote_state()

    def _clear_quote_state(self) -> None:
        with self._lock:
            self._snapshots = {}
            self._latest_exchange_at = {}
            self._received_at_by_symbol = {}
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
                float(self.config.request_connect_timeout),
                float(self.config.request_read_timeout),
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
                    while not self._stop_event.is_set():
                        desired = self._desired_symbols()
                        await self._sync_subscriptions(
                            websocket,
                            approval_key,
                            subscribed,
                            desired,
                        )
                        if not desired:
                            return
                        try:
                            raw = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        except asyncio.TimeoutError:
                            continue
                        await self._handle_message(websocket, raw)
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
        if len(parts) != 4:
            return 0
        tr_id = parts[1]
        if tr_id != TRADE_TR_ID:
            return 0
        values = parts[3].split("^")
        declared_count = max(1, _integer(parts[2]))
        parsed_count = 0
        for index in range(declared_count):
            start = index * TRADE_ROW_WIDTH
            if start >= len(values):
                break
            self._merge_row(
                values[start:start + TRADE_ROW_WIDTH],
                received_at,
            )
            parsed_count += 1
        return parsed_count

    def _merge_row(self, values: Sequence[str], received_at: datetime) -> None:
        quote = _quote_from_trade(values, received_at)
        if quote is None:
            return
        symbol = quote.symbol
        with self._lock:
            if symbol not in self._symbols:
                return
            previous = self._latest_exchange_at.get(symbol)
            if previous is not None and quote.timestamp < previous:
                return
            self._latest_exchange_at[symbol] = quote.timestamp
            self._received_at_by_symbol[symbol] = received_at
            key = (symbol, quote.timestamp)
            self._snapshots[key] = quote
            if len(self._snapshots) > 20_000:
                oldest = min(self._snapshots, key=lambda item: item[1])
                self._snapshots.pop(oldest, None)

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
    ) -> List[str]:
        current = now or datetime.now()
        normalized = self._normalize_symbols(symbols)
        with self._lock:
            return [
                symbol
                for symbol in normalized
                if symbol not in self._received_at_by_symbol
                or (current - self._received_at_by_symbol[symbol]).total_seconds() > self.stale_seconds
            ]
