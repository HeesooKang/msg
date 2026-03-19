from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.models import Quote


class QuoteTapeRecorder:
    def __init__(self, root: str | Path, *, enabled: bool = True):
        self.root = Path(root)
        self.enabled = enabled

    def _day_dir(self, now: datetime) -> Path:
        return self.root / now.strftime("%Y") / now.strftime("%m") / now.strftime("%d")

    def _append_rows(self, path: Path, rows: Iterable[dict[str, Any]]) -> None:
        if not self.enabled:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fp:
            for row in rows:
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    def record_quotes(
        self,
        now: datetime,
        quotes: Iterable[Quote],
        *,
        regime_label: str,
        bear_score: int,
        market_data_ready: bool,
    ) -> None:
        rows = []
        for quote in quotes:
            rows.append(
                {
                    "ts": now.isoformat(timespec="seconds"),
                    "symbol": quote.symbol,
                    "name": quote.name,
                    "current_price": int(quote.current_price or 0),
                    "change_rate": float(quote.change_rate or 0.0),
                    "open_price": int(quote.open_price or 0),
                    "high_price": int(quote.high_price or 0),
                    "low_price": int(quote.low_price or 0),
                    "volume": int(quote.volume or 0),
                    "trade_amount": int(getattr(quote, "trade_amount", 0) or 0),
                    "regime_label": regime_label,
                    "bear_score": int(bear_score),
                    "market_data_ready": bool(market_data_ready),
                }
            )
        self._append_rows(self._day_dir(now) / "quotes.ndjson", rows)

    def record_leaders(
        self,
        now: datetime,
        *,
        event: str,
        rows: Iterable[Mapping[str, Any] | Any],
    ) -> None:
        payload_rows = []
        for row in rows:
            if is_dataclass(row):
                row_payload = asdict(row)
            else:
                row_payload = dict(row)
            row_payload["ts"] = now.isoformat(timespec="seconds")
            row_payload["event"] = event
            payload_rows.append(row_payload)
        self._append_rows(self._day_dir(now) / "leaders.ndjson", payload_rows)
