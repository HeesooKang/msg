from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from src.analytics.price_prediction import (
    PREDICTION_FEATURES,
    executable_bid,
)
from src.models import Quote
from src.strategies.momentum_scalp_pnl import estimate_trade_net_pnl_unrounded


FORECAST_SCHEMA_VERSION = 3


def _has_current_features(item: Mapping[str, Any]) -> bool:
    compact_features = item.get("compact_features")
    return (
        isinstance(compact_features, Mapping)
        and set(compact_features) == set(PREDICTION_FEATURES)
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value or ""))
    except ValueError:
        return None


def _settlement_grace_seconds(horizon_seconds: int) -> int:
    horizon = max(1, int(horizon_seconds or 0))
    return max(15, min(60, int(round(horizon / 3.0))))


class ForecastOutcomeLedger:
    """Persist executable forecasts and settle each one on its first post-horizon bid."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self._active_date = ""
        self._pending: Dict[str, Dict[str, Any]] = {}
        self._outcomes: List[Dict[str, Any]] = []
        self._history_cache_date = ""
        self._history_cache: List[Dict[str, Any]] = []
        self._sequence = 0

    def _path_for_date(self, date_text: str) -> Path:
        year, month, _ = date_text.split("-")
        return self.root / year / month / f"forecast-outcomes.{date_text}.json"

    def _ensure_date(self, moment: datetime) -> None:
        date_text = moment.date().isoformat()
        if date_text == self._active_date:
            return
        if self._active_date and self._pending:
            self._save()
        self._active_date = date_text
        self._pending = {}
        self._outcomes = []
        self._sequence = 0
        path = self._path_for_date(date_text)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        schema_version = int(payload.get("schema_version", 0) or 0)
        if schema_version >= FORECAST_SCHEMA_VERSION:
            self._pending = {
                str(item.get("forecast_id") or ""): dict(item)
                for item in (payload.get("pending") or [])
                if str(item.get("forecast_id") or "")
            }
        self._outcomes = [
            dict(item)
            for item in (payload.get("outcomes") or [])
            if isinstance(item, Mapping)
        ]
        self._sequence = len(self._pending) + len(self._outcomes)

    def _save(self) -> None:
        if not self._active_date:
            return
        path = self._path_for_date(self._active_date)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": FORECAST_SCHEMA_VERSION,
            "date": self._active_date,
            "pending": list(self._pending.values()),
            "outcomes": list(self._outcomes),
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.replace(path)

    def settle(self, quotes: Sequence[Quote], *, now: datetime) -> List[Dict[str, Any]]:
        self._ensure_date(now)
        settled: List[Dict[str, Any]] = []
        ordered_quotes = sorted(
            quotes,
            key=lambda quote: (quote.timestamp, str(quote.symbol)),
        )
        for quote in ordered_quotes:
            symbol = str(quote.symbol or "")
            quote_at = quote.timestamp
            matching_ids = [
                forecast_id
                for forecast_id, pending in self._pending.items()
                if str(pending.get("symbol") or "") == symbol
            ]
            for forecast_id in matching_ids:
                pending = self._pending.get(forecast_id)
                if pending is None:
                    continue
                due_at = _parse_datetime(pending.get("due_at"))
                if due_at is None:
                    continue
                if quote_at < due_at:
                    continue
                entry_ask = max(
                    0,
                    int(pending.get("signal_entry_ask", 0) or 0),
                )
                outcome_bid = executable_bid(quote)
                if entry_ask <= 0 or outcome_bid <= 0:
                    continue
                commission_rate = max(0.0, _safe_float(pending.get("commission_rate")))
                sell_tax_rate = max(0.0, _safe_float(pending.get("sell_tax_rate")))
                actual_return = ((float(outcome_bid) / float(entry_ask)) - 1.0) * 100.0
                actual_net = estimate_trade_net_pnl_unrounded(
                    entry_price=entry_ask,
                    exit_price=outcome_bid,
                    quantity=1,
                    commission_rate=commission_rate,
                    sell_tax_rate=sell_tax_rate,
                )
                actual_net_return = actual_net / float(entry_ask) * 100.0
                signal_at = _parse_datetime(pending.get("signal_timestamp")) or quote_at
                elapsed_seconds = max(0, int((quote_at - signal_at).total_seconds()))
                horizon_seconds = max(1, int(pending.get("horizon_seconds", 0) or 0))
                training_eligible = elapsed_seconds <= (
                    horizon_seconds + _settlement_grace_seconds(horizon_seconds)
                )
                record = dict(pending)
                record.update(
                    {
                        "outcome_timestamp": quote_at.isoformat(timespec="seconds"),
                        "outcome_bid": int(outcome_bid),
                        "elapsed_seconds": elapsed_seconds,
                        "training_eligible": training_eligible,
                        "outcome_status": "on_horizon" if training_eligible else "late",
                        "actual_return_pct": round(actual_return, 8),
                        "actual_net_return_pct": round(actual_net_return, 8),
                        "profitable": bool(actual_net_return > 0.0),
                    }
                )
                self._outcomes.append(record)
                settled.append(record)
                self._pending.pop(forecast_id, None)
        if settled:
            self._save()
            self._history_cache_date = ""
        return settled

    def _pending_for_symbol(
        self,
        symbol: str,
        *,
        selected: bool,
        record_kind: str,
    ) -> Dict[str, Any] | None:
        return next(
            (
                item
                for item in self._pending.values()
                if str(item.get("symbol") or "") == symbol
                and _has_current_features(item)
                and bool(item.get("selected")) is bool(selected)
                and str(
                    item.get("record_kind")
                    or ("selected" if item.get("selected") else "observed")
                )
                == record_kind
            ),
            None,
        )

    def record(
        self,
        payload: Mapping[str, Any],
        *,
        now: datetime,
        selected: bool,
    ) -> str | None:
        self._ensure_date(now)
        symbol = str(payload.get("symbol") or "")
        entry_ask = max(
            0,
            int(payload.get("signal_entry_ask", 0) or 0),
        )
        horizon_seconds = max(1, int(payload.get("horizon_seconds", 0) or 0))
        if not symbol or entry_ask <= 0 or not _has_current_features(payload):
            return None
        record_kind = str(
            payload.get("record_kind")
            or ("selected" if selected else "observed")
        )
        existing = self._pending_for_symbol(
            symbol,
            selected=selected,
            record_kind=record_kind,
        )
        if existing is not None:
            return str(existing.get("forecast_id") or "") or None
        forecast_id = self._append_pending(payload, now=now, selected=selected)
        if forecast_id is not None:
            self._save()
        return forecast_id

    def _append_pending(
        self,
        payload: Mapping[str, Any],
        *,
        now: datetime,
        selected: bool,
    ) -> str | None:
        symbol = str(payload.get("symbol") or "")
        entry_ask = max(
            0,
            int(payload.get("signal_entry_ask", 0) or 0),
        )
        horizon_seconds = max(1, int(payload.get("horizon_seconds", 0) or 0))
        if not symbol or entry_ask <= 0 or not _has_current_features(payload):
            return None
        record_kind = str(
            payload.get("record_kind")
            or ("selected" if selected else "observed")
        )
        signal_at = _parse_datetime(payload.get("signal_timestamp")) or now
        self._sequence += 1
        forecast_id = str(payload.get("forecast_id") or "").strip() or (
            f"{symbol}:{signal_at.isoformat(timespec='seconds')}:{horizon_seconds}:"
            f"{record_kind}:{self._sequence}"
        )
        record = dict(payload)
        record.update(
            {
                "forecast_id": forecast_id,
                "symbol": symbol,
                "signal_entry_ask": entry_ask,
                "horizon_seconds": horizon_seconds,
                "signal_timestamp": signal_at.isoformat(timespec="seconds"),
                "due_at": (signal_at + timedelta(seconds=horizon_seconds)).isoformat(timespec="seconds"),
                "selected": bool(selected),
                "record_kind": record_kind,
            }
        )
        self._pending[forecast_id] = record
        return forecast_id

    def record_observation_sets(
        self,
        payloads: Sequence[Mapping[str, Any]],
        *,
        now: datetime,
    ) -> Dict[tuple[str, int], str]:
        """Record one non-overlapping multi-horizon sample per symbol."""
        self._ensure_date(now)
        grouped: Dict[str, Dict[int, Mapping[str, Any]]] = {}
        for payload in payloads:
            symbol = str(payload.get("symbol") or "")
            horizon = max(1, int(payload.get("horizon_seconds", 0) or 0))
            if symbol:
                grouped.setdefault(symbol, {})[horizon] = payload

        recorded: Dict[tuple[str, int], str] = {}
        changed = False
        for symbol, horizon_payloads in grouped.items():
            existing = self._pending_for_symbol(
                symbol,
                selected=False,
                record_kind="observed",
            )
            if existing is not None:
                continue
            for horizon, payload in sorted(horizon_payloads.items()):
                forecast_id = self._append_pending(
                    payload,
                    now=now,
                    selected=False,
                )
                if forecast_id is None:
                    continue
                recorded[(symbol, horizon)] = forecast_id
                changed = True
        if changed:
            self._save()
        return recorded

    def has_pending_selected(self, symbol: str, *, now: datetime) -> bool:
        self._ensure_date(now)
        return self._pending_for_symbol(
            str(symbol or "").strip(),
            selected=True,
            record_kind="selected",
        ) is not None

    def record_execution_outcome(
        self,
        *,
        symbol: str,
        forecast_id: str = "",
        entry_price: int,
        exit_price: int,
        quantity: int,
        net_pnl: int,
        now: datetime,
    ) -> bool:
        self._ensure_date(now)
        normalized_symbol = str(symbol or "").strip()
        resolved_entry = max(0, int(entry_price or 0))
        resolved_exit = max(0, int(exit_price or 0))
        resolved_quantity = max(0, int(quantity or 0))
        if not normalized_symbol or resolved_entry <= 0 or resolved_exit <= 0 or resolved_quantity <= 0:
            return False

        normalized_id = str(forecast_id or "").strip()
        target = self._pending.get(normalized_id) if normalized_id else None
        if target is None:
            target = next(
                (
                    item
                    for item in reversed(self._outcomes)
                    if str(item.get("symbol") or "") == normalized_symbol
                    and bool(item.get("selected"))
                    and (not normalized_id or str(item.get("forecast_id") or "") == normalized_id)
                ),
                None,
            )
        if target is None:
            return False
        invested = float(resolved_entry * resolved_quantity)
        signal_at = _parse_datetime(target.get("signal_timestamp")) or now
        target.update(
            {
                "execution_timestamp": now.isoformat(timespec="seconds"),
                "execution_elapsed_seconds": max(0, int((now - signal_at).total_seconds())),
                "execution_entry_price": resolved_entry,
                "execution_exit_price": resolved_exit,
                "execution_quantity": resolved_quantity,
                "execution_net_pnl": int(net_pnl),
                "execution_return_pct": round(((resolved_exit / resolved_entry) - 1.0) * 100.0, 8),
                "execution_net_return_pct": round(float(net_pnl) / invested * 100.0, 8),
                "execution_profitable": bool(net_pnl > 0),
            }
        )
        self._save()
        return True

    def historical_outcomes(self, *, as_of: datetime) -> List[Dict[str, Any]]:
        date_text = as_of.date().isoformat()
        if self._history_cache_date == date_text:
            return list(self._history_cache)
        outcomes: List[Dict[str, Any]] = []
        if self.root.exists():
            for path in sorted(self.root.rglob("forecast-outcomes.*.json")):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if int(payload.get("schema_version", 0) or 0) != FORECAST_SCHEMA_VERSION:
                    continue
                if str(payload.get("date") or "") >= date_text:
                    continue
                outcomes.extend(
                    dict(item)
                    for item in (payload.get("outcomes") or [])
                    if isinstance(item, Mapping)
                )
        self._history_cache_date = date_text
        self._history_cache = outcomes
        return list(outcomes)

    def training_outcomes(self, *, as_of: datetime) -> List[Dict[str, Any]]:
        self._ensure_date(as_of)
        outcomes = self.historical_outcomes(as_of=as_of)
        outcomes.extend(
            dict(item)
            for item in self._outcomes
            if (_parse_datetime(item.get("outcome_timestamp")) or as_of) <= as_of
        )
        return outcomes
