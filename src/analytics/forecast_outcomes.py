from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import exp, sqrt
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from src.models import Quote


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


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


def _weighted_median(values: Sequence[tuple[float, float]]) -> float:
    ordered = sorted((float(value), max(0.0, float(weight))) for value, weight in values)
    total_weight = sum(weight for _, weight in ordered)
    if not ordered or total_weight <= 0.0:
        return 0.0
    threshold = total_weight / 2.0
    running = 0.0
    for value, weight in ordered:
        running += weight
        if running >= threshold:
            return value
    return ordered[-1][0]


def _median_absolute_deviation(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    center = median(float(value) for value in values)
    return median(abs(float(value) - center) for value in values)


def _actual_return_pct(item: Mapping[str, Any]) -> float | None:
    if "actual_return_pct" in item:
        return _safe_float(item.get("actual_return_pct"))
    if "actual_net_return_pct" not in item:
        return None
    return _safe_float(item.get("actual_net_return_pct")) + max(
        0.0,
        _safe_float(item.get("round_trip_cost_pct")),
    )


@dataclass(frozen=True)
class WalkForwardCalibration:
    sample_count: int
    effective_sample_size: float
    raw_win_probability: float
    calibrated_win_probability: float
    raw_return_pct: float
    calibrated_return_pct: float

    @property
    def return_shift_pct(self) -> float:
        return float(self.calibrated_return_pct) - float(self.raw_return_pct)


def calibrate_walk_forward(
    outcomes: Iterable[Mapping[str, Any]],
    *,
    raw_win_probability: float,
    raw_return_pct: float,
    horizon_seconds: int,
    round_trip_cost_pct: float = 0.0,
    strategy_name: str = "",
) -> WalkForwardCalibration:
    raw_probability = _clip(raw_win_probability, 0.0, 1.0)
    raw_return = float(raw_return_pct)
    normalized_strategy = str(strategy_name or "")
    usable = [
        item
        for item in outcomes
        if int(item.get("horizon_seconds", 0) or 0) == int(horizon_seconds)
        and _actual_return_pct(item) is not None
        and "raw_win_probability" in item
        and "raw_predicted_return_pct" in item
        and (
            not normalized_strategy
            or str(item.get("strategy_name") or "") == normalized_strategy
        )
    ]
    if not usable:
        return WalkForwardCalibration(
            sample_count=0,
            effective_sample_size=0.0,
            raw_win_probability=raw_probability,
            calibrated_win_probability=raw_probability,
            raw_return_pct=raw_return,
            calibrated_return_pct=raw_return,
        )

    historical_probabilities = [_safe_float(item.get("raw_win_probability"), 0.5) for item in usable]
    historical_returns = [_safe_float(item.get("raw_predicted_return_pct")) for item in usable]
    statistical_floor = 1.0 / sqrt(len(usable) + 1.0)
    probability_scale = max(
        statistical_floor,
        1.4826 * _median_absolute_deviation(historical_probabilities),
    )
    return_scale = max(
        statistical_floor,
        1.4826 * _median_absolute_deviation(historical_returns),
    )
    raw_weights: List[float] = []
    for historical_probability, historical_return in zip(
        historical_probabilities,
        historical_returns,
    ):
        distance_squared = (
            ((historical_probability - raw_probability) / probability_scale) ** 2
            + ((historical_return - raw_return) / return_scale) ** 2
        )
        raw_weights.append(exp(-0.5 * distance_squared))

    # Repeated signals share both symbol-specific and session-wide noise.
    # Cap their combined influence so quote frequency cannot masquerade as evidence.
    group_totals: Dict[tuple[str, str], float] = defaultdict(float)
    group_keys: List[tuple[str, str]] = []
    for index, (item, weight) in enumerate(zip(usable, raw_weights)):
        timestamp = str(item.get("signal_timestamp") or "")
        symbol = str(item.get("symbol") or f"row-{index}")
        group_key = (timestamp[:10], symbol)
        group_keys.append(group_key)
        group_totals[group_key] += weight
    weights = [
        weight / max(1.0, group_totals[group_key])
        for weight, group_key in zip(raw_weights, group_keys)
    ]
    date_totals: Dict[str, float] = defaultdict(float)
    for (date_text, _symbol), weight in zip(group_keys, weights):
        date_totals[date_text] += weight
    weights = [
        weight / max(1.0, date_totals[date_text])
        for weight, (date_text, _symbol) in zip(weights, group_keys)
    ]
    effective_samples = sum(weights)
    if effective_samples <= 1e-9:
        return WalkForwardCalibration(
            sample_count=len(usable),
            effective_sample_size=round(effective_samples, 6),
            raw_win_probability=raw_probability,
            calibrated_win_probability=raw_probability,
            raw_return_pct=raw_return,
            calibrated_return_pct=raw_return,
        )

    execution_cost = max(0.0, float(round_trip_cost_pct))
    weighted_wins = sum(
        weight * (1.0 if float(_actual_return_pct(item) or 0.0) > execution_cost else 0.0)
        for item, weight in zip(usable, weights)
    )
    prior_strength = max(1.0, sqrt(effective_samples))
    calibrated_probability = (
        weighted_wins + prior_strength * raw_probability
    ) / (effective_samples + prior_strength)
    forecast_errors = [
        float(_actual_return_pct(item) or 0.0)
        - _safe_float(item.get("raw_predicted_return_pct"))
        for item in usable
    ]
    forecast_error = _weighted_median(list(zip(forecast_errors, weights)))
    residual_deviation = _weighted_median(
        [
            (abs(error - forecast_error), weight)
            for error, weight in zip(forecast_errors, weights)
        ]
    )
    # A noisy mix of over- and under-predictions should not create a large shift.
    residual_consistency = abs(forecast_error) / max(
        1e-12,
        abs(forecast_error) + 1.4826 * residual_deviation,
    )
    reliability = effective_samples / max(
        1.0,
        effective_samples + prior_strength,
    )
    calibrated_return = raw_return + reliability * residual_consistency * forecast_error
    return WalkForwardCalibration(
        sample_count=len(usable),
        effective_sample_size=round(effective_samples, 6),
        raw_win_probability=raw_probability,
        calibrated_win_probability=_clip(calibrated_probability, 0.0, 1.0),
        raw_return_pct=raw_return,
        calibrated_return_pct=float(calibrated_return),
    )


class ForecastOutcomeLedger:
    """Persist non-overlapping forecasts and settle them on the first post-horizon quote."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self._active_date = ""
        self._pending: Dict[str, Dict[str, Any]] = {}
        self._outcomes: List[Dict[str, Any]] = []
        self._history_cache_date = ""
        self._history_cache: List[Dict[str, Any]] = []

    def _path_for_date(self, date_text: str) -> Path:
        year, month, _ = date_text.split("-")
        return self.root / year / month / f"forecast-outcomes.{date_text}.json"

    def _ensure_date(self, moment: datetime) -> None:
        date_text = moment.date().isoformat()
        if date_text == self._active_date:
            return
        self._active_date = date_text
        self._pending = {}
        self._outcomes = []
        path = self._path_for_date(date_text)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        self._pending = {
            str(item.get("symbol") or ""): dict(item)
            for item in (payload.get("pending") or [])
            if str(item.get("symbol") or "")
        }
        self._outcomes = [dict(item) for item in (payload.get("outcomes") or [])]

    def _save(self) -> None:
        if not self._active_date:
            return
        path = self._path_for_date(self._active_date)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "date": self._active_date,
            "pending": list(self._pending.values()),
            "outcomes": list(self._outcomes),
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)

    def settle(self, quotes: Sequence[Quote], *, now: datetime) -> List[Dict[str, Any]]:
        self._ensure_date(now)
        settled: List[Dict[str, Any]] = []
        for quote in quotes:
            symbol = str(getattr(quote, "symbol", "") or "")
            pending = self._pending.get(symbol)
            if not pending:
                continue
            due_at = _parse_datetime(pending.get("due_at"))
            quote_at = _parse_datetime(getattr(quote, "timestamp", None)) or now
            if due_at is None or quote_at < due_at:
                continue
            signal_price = max(0, int(pending.get("signal_price", 0) or 0))
            outcome_price = max(0, int(getattr(quote, "current_price", 0) or 0))
            if signal_price <= 0 or outcome_price <= 0:
                continue
            actual_return = ((float(outcome_price) / float(signal_price)) - 1.0) * 100.0
            actual_net_return = actual_return - max(
                0.0,
                _safe_float(pending.get("round_trip_cost_pct")),
            )
            record = dict(pending)
            record.update(
                {
                    "outcome_timestamp": quote_at.isoformat(timespec="seconds"),
                    "outcome_price": outcome_price,
                    "elapsed_seconds": max(
                        0,
                        int((quote_at - (_parse_datetime(pending.get("signal_timestamp")) or quote_at)).total_seconds()),
                    ),
                    "actual_return_pct": round(actual_return, 6),
                    "actual_net_return_pct": round(actual_net_return, 6),
                    "raw_prediction_error_pct": round(
                        actual_return - _safe_float(pending.get("raw_predicted_return_pct")),
                        6,
                    ),
                    "calibrated_prediction_error_pct": round(
                        actual_return
                        - _safe_float(
                            pending.get(
                                "calibrated_predicted_return_pct",
                                pending.get("raw_predicted_return_pct"),
                            )
                        ),
                        6,
                    ),
                    "profitable": bool(actual_net_return > 0.0),
                }
            )
            self._outcomes.append(record)
            settled.append(record)
            self._pending.pop(symbol, None)
        if settled:
            self._save()
            self._history_cache_date = ""
        return settled

    def record(self, payload: Mapping[str, Any], *, now: datetime, selected: bool) -> bool:
        self._ensure_date(now)
        symbol = str(payload.get("symbol") or "")
        signal_price = max(0, int(payload.get("signal_price", 0) or 0))
        horizon_seconds = max(1, int(payload.get("horizon_seconds", 0) or 0))
        if not symbol or signal_price <= 0:
            return False
        if symbol in self._pending and not selected:
            return False
        signal_at = _parse_datetime(payload.get("signal_timestamp")) or now
        record = dict(payload)
        record.update(
            {
                "symbol": symbol,
                "signal_price": signal_price,
                "horizon_seconds": horizon_seconds,
                "signal_timestamp": signal_at.isoformat(timespec="seconds"),
                "due_at": (signal_at + timedelta(seconds=horizon_seconds)).isoformat(timespec="seconds"),
                "selected": bool(selected),
            }
        )
        self._pending[symbol] = record
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
                if str(payload.get("date") or "") >= date_text:
                    continue
                outcomes.extend(
                    dict(item)
                    for item in (payload.get("outcomes") or [])
                    if isinstance(item, dict)
                )
        self._history_cache_date = date_text
        self._history_cache = outcomes
        return list(outcomes)

    def calibrate(
        self,
        *,
        as_of: datetime,
        raw_win_probability: float,
        raw_return_pct: float,
        horizon_seconds: int,
        round_trip_cost_pct: float = 0.0,
        strategy_name: str = "",
    ) -> WalkForwardCalibration:
        return calibrate_walk_forward(
            self.historical_outcomes(as_of=as_of),
            raw_win_probability=raw_win_probability,
            raw_return_pct=raw_return_pct,
            horizon_seconds=horizon_seconds,
            round_trip_cost_pct=round_trip_cost_pct,
            strategy_name=strategy_name,
        )
