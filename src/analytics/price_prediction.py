from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from math import log, sqrt
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

from src.models import Quote


PREDICTION_HORIZONS_SECONDS = (30, 60, 120, 180)

PREDICTION_FEATURES = (
    "return_15s_pct",
    "return_60s_pct",
    "return_180s_pct",
    "pullback_60s_pct",
    "realized_volatility_pct",
    "volume_acceleration",
    "net_buy_volume_ratio",
    "book_imbalance",
    "relative_60s_pct",
    "spread_pct",
)
HORIZON_MODEL_FEATURE = "log_horizon_ratio"
MODEL_FEATURES = (*PREDICTION_FEATURES, HORIZON_MODEL_FEATURE)
LOOKBACK_SECONDS = (15, *PREDICTION_HORIZONS_SECONDS)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return default
    return resolved if np.isfinite(resolved) else default


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value or ""))
    except ValueError:
        return None


def _tick_size(price: int) -> int:
    value = max(1, int(price or 0))
    if value < 2_000:
        return 1
    if value < 5_000:
        return 5
    if value < 20_000:
        return 10
    if value < 50_000:
        return 50
    if value < 200_000:
        return 100
    if value < 500_000:
        return 500
    return 1_000


def executable_ask(quote: Quote) -> int:
    current = max(1, int(quote.current_price or 0))
    ask = max(0, int(quote.ask_price or 0))
    if ask > 0:
        return ask
    if bool(getattr(quote, "book_available", False)):
        return 0
    return current + _tick_size(current)


def executable_bid(quote: Quote) -> int:
    current = max(1, int(quote.current_price or 0))
    bid = max(0, int(quote.bid_price or 0))
    if bid > 0:
        return bid
    if bool(getattr(quote, "book_available", False)):
        return 0
    return max(1, current - _tick_size(current))


def _mid_price(quote: Quote) -> float:
    current = max(1, int(quote.current_price or 0))
    ask = max(0, int(quote.ask_price or 0))
    bid = max(0, int(quote.bid_price or 0))
    if ask > 0 and bid > 0:
        return (float(ask) + float(bid)) / 2.0
    return float(current)


def _quote_at_or_before(
    quotes: Sequence[Quote],
    target: datetime,
    *,
    max_age_seconds: float,
) -> Quote | None:
    prior = [quote for quote in quotes if quote.timestamp <= target]
    if not prior:
        return None
    resolved = max(prior, key=lambda quote: quote.timestamp)
    if (target - resolved.timestamp).total_seconds() > max_age_seconds:
        return None
    return resolved


def _return_pct(current: float, previous: float) -> float:
    if current <= 0.0 or previous <= 0.0:
        return 0.0
    return ((current / previous) - 1.0) * 100.0


def _horizon_model_value(horizon_seconds: int) -> float:
    horizon = max(1, int(horizon_seconds or 0))
    full_horizon = float(PREDICTION_HORIZONS_SECONDS[-1])
    horizon_range = abs(
        log(float(PREDICTION_HORIZONS_SECONDS[0]) / full_horizon)
    )
    return log(float(horizon) / full_horizon) / horizon_range


def _record_unavailable_features(item: Mapping[str, Any]) -> set[str]:
    raw = item.get("unavailable_features")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return {
            str(name)
            for name in raw
            if str(name) in PREDICTION_FEATURES
        }

    # Early v3 rows predate explicit source availability. Exact zero in both
    # microstructure fields was how REST supplementation represented missing
    # data, so do not train those placeholders as genuine neutral pressure.
    features = item.get("compact_features")
    if not isinstance(features, Mapping):
        return set()
    if (
        _safe_float(features.get("net_buy_volume_ratio")) == 0.0
        and _safe_float(features.get("book_imbalance")) == 0.0
    ):
        return {"net_buy_volume_ratio", "book_imbalance"}
    return set()


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    if len(values) == 0:
        return 0.0
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights)
    threshold = min(1.0, max(0.0, float(quantile))) * float(cumulative[-1])
    index = min(
        len(ordered_values) - 1,
        int(np.searchsorted(cumulative, threshold, side="left")),
    )
    return float(ordered_values[index])


@dataclass(frozen=True)
class FeatureRow:
    ready: bool
    reason: str
    compact_features: Dict[str, float] = field(default_factory=dict)
    unavailable_features: tuple[str, ...] = ()


@dataclass(frozen=True)
class Prediction:
    ready: bool
    reason: str
    horizon_seconds: int
    sample_count: int
    expected_net_return_pct: float
    lower_net_return_pct: float
    compact_features: Dict[str, float] = field(default_factory=dict)
    unavailable_features: tuple[str, ...] = ()


@dataclass(frozen=True)
class RidgeModel:
    coefficients: tuple[float, ...]
    medians: tuple[float, ...]
    scales: tuple[float, ...]
    feature_mins: tuple[float, ...]
    feature_maxs: tuple[float, ...]
    sample_count: int
    effective_sample_size: float
    target_min_pct: float
    target_max_pct: float
    residual_rms_pct: float
    residual_p10_pct: float


def build_feature_rows(
    quotes: Sequence[Quote],
    *,
    recent_quotes_by_symbol: Mapping[str, Sequence[Quote]],
) -> Dict[str, FeatureRow]:
    provisional: Dict[str, tuple[Dict[str, float], set[str]]] = {}
    for quote in quotes:
        symbol = str(quote.symbol or "")
        signal_at = quote.timestamp
        if not symbol or int(quote.current_price or 0) <= 0:
            continue

        history = [
            item
            for item in recent_quotes_by_symbol.get(symbol, ())
            if item.timestamp <= signal_at
        ]
        if not history or history[-1].timestamp != signal_at:
            history.append(quote)
        history.sort(key=lambda item: item.timestamp)

        current_mid = _mid_price(quote)
        past_quotes = {
            seconds: _quote_at_or_before(
                history,
                signal_at - timedelta(seconds=seconds),
                max_age_seconds=max(2.0, float(seconds) / 3.0),
            )
            for seconds in LOOKBACK_SECONDS
        }
        unavailable: set[str] = set()
        returns = {
            seconds: (
                _return_pct(current_mid, _mid_price(past_quotes[seconds]))
                if past_quotes[seconds] is not None
                else 0.0
            )
            for seconds in LOOKBACK_SECONDS
        }
        for seconds, feature_name in (
            (15, "return_15s_pct"),
            (60, "return_60s_pct"),
            (180, "return_180s_pct"),
        ):
            if past_quotes[seconds] is None:
                unavailable.add(feature_name)
        recent_window = [
            item
            for item in history
            if item.timestamp >= signal_at - timedelta(seconds=60)
        ]
        recent_high = max(_mid_price(item) for item in recent_window)
        if len({item.timestamp for item in recent_window}) < 2:
            unavailable.add("pullback_60s_pct")

        sampled = []
        for seconds in reversed(LOOKBACK_SECONDS):
            if past_quotes[seconds] is not None:
                sampled.append(past_quotes[seconds])
        sampled.append(quote)
        unique_sampled = []
        seen_timestamps = set()
        for item in sampled:
            if item.timestamp in seen_timestamps:
                continue
            seen_timestamps.add(item.timestamp)
            unique_sampled.append(item)
        interval_rates = []
        for previous, current in zip(unique_sampled, unique_sampled[1:]):
            elapsed = max(1.0, (current.timestamp - previous.timestamp).total_seconds())
            interval_rates.append(
                _return_pct(_mid_price(current), _mid_price(previous)) * 15.0 / elapsed
            )
        realized_volatility = (
            float(np.std(np.asarray(interval_rates, dtype=float)))
            if len(interval_rates) >= 2
            else 0.0
        )
        if len(interval_rates) < 2:
            unavailable.add("realized_volatility_pct")

        volume_now = max(0, int(getattr(quote, "cumulative_volume", 0) or 0))
        if past_quotes[15] is None or past_quotes[60] is None:
            volume_acceleration = 0.0
            unavailable.add("volume_acceleration")
        else:
            volume_15 = max(
                0,
                int(getattr(past_quotes[15], "cumulative_volume", 0) or 0),
            )
            volume_60 = max(
                0,
                int(getattr(past_quotes[60], "cumulative_volume", 0) or 0),
            )
            recent_volume_rate = max(0.0, float(volume_now - volume_15) / 15.0)
            prior_volume_rate = max(0.0, float(volume_15 - volume_60) / 45.0)
            volume_acceleration = float(
                np.log1p(recent_volume_rate) - np.log1p(prior_volume_rate)
            )

        flow_past = past_quotes[60]
        flow_available = bool(getattr(quote, "flow_available", False)) and (
            flow_past is not None
            and bool(getattr(flow_past, "flow_available", False))
        )
        if flow_available:
            cumulative_buy_now = max(
                0,
                int(getattr(quote, "cumulative_buy_volume", 0) or 0),
            )
            cumulative_sell_now = max(
                0,
                int(getattr(quote, "cumulative_sell_volume", 0) or 0),
            )
            cumulative_buy_60 = max(
                0,
                int(getattr(flow_past, "cumulative_buy_volume", 0) or 0),
            )
            cumulative_sell_60 = max(
                0,
                int(getattr(flow_past, "cumulative_sell_volume", 0) or 0),
            )
            buy_volume = max(0, cumulative_buy_now - cumulative_buy_60)
            sell_volume = max(0, cumulative_sell_now - cumulative_sell_60)
            directional_volume = buy_volume + sell_volume
            net_buy_volume_ratio = (
                float(buy_volume - sell_volume) / float(directional_volume)
                if directional_volume > 0
                else 0.0
            )
        else:
            net_buy_volume_ratio = 0.0
            unavailable.add("net_buy_volume_ratio")

        total_ask_size = max(
            0,
            int(getattr(quote, "total_ask_size", 0) or 0),
        )
        total_bid_size = max(
            0,
            int(getattr(quote, "total_bid_size", 0) or 0),
        )
        if total_ask_size + total_bid_size <= 0:
            total_ask_size = max(0, int(getattr(quote, "ask_size", 0) or 0))
            total_bid_size = max(0, int(getattr(quote, "bid_size", 0) or 0))
        book_size = total_ask_size + total_bid_size
        if bool(getattr(quote, "book_depth_available", False)) and book_size > 0:
            book_imbalance = (
                float(total_bid_size - total_ask_size) / float(book_size)
            )
        else:
            book_imbalance = 0.0
            unavailable.add("book_imbalance")

        ask = max(0, int(quote.ask_price or 0))
        bid = max(0, int(quote.bid_price or 0))
        if ask > 0 and bid > 0:
            observed_spread = max(0, ask - bid)
        else:
            observed_spread = _tick_size(int(current_mid))
            unavailable.add("spread_pct")
        provisional[symbol] = (
            {
                "return_15s_pct": returns[15],
                "return_60s_pct": returns[60],
                "return_180s_pct": returns[180],
                "pullback_60s_pct": _return_pct(current_mid, recent_high),
                "realized_volatility_pct": realized_volatility,
                "volume_acceleration": volume_acceleration,
                "net_buy_volume_ratio": net_buy_volume_ratio,
                "book_imbalance": book_imbalance,
                "spread_pct": float(observed_spread) / current_mid * 100.0,
            },
            unavailable,
        )

    ready_values = [
        values
        for values in provisional.values()
        if "return_60s_pct" not in values[1]
    ]
    median_return_60 = (
        float(np.median([values[0]["return_60s_pct"] for values in ready_values]))
        if ready_values
        else 0.0
    )
    rows: Dict[str, FeatureRow] = {}
    for symbol, (values, unavailable) in provisional.items():
        values["relative_60s_pct"] = values["return_60s_pct"] - median_return_60
        if "return_60s_pct" in unavailable:
            unavailable.add("relative_60s_pct")
        rows[symbol] = FeatureRow(
            ready=True,
            reason="ok",
            compact_features={
                name: round(values[name], 8)
                for name in PREDICTION_FEATURES
            },
            unavailable_features=tuple(sorted(unavailable)),
        )
    return rows


def fit_model(
    outcomes: Iterable[Mapping[str, Any]],
    *,
    as_of: datetime,
) -> RidgeModel | None:
    feature_rows = []
    feature_availability = []
    targets = []
    group_keys = []
    for item in outcomes:
        outcome_at = _parse_datetime(item.get("outcome_timestamp"))
        signal_at = _parse_datetime(item.get("signal_timestamp"))
        features = item.get("compact_features")
        item_horizon = int(item.get("horizon_seconds", 0) or 0)
        if (
            outcome_at is None
            or outcome_at > as_of
            or item_horizon not in PREDICTION_HORIZONS_SECONDS
            or item.get("training_eligible") is not True
            or item.get("actual_net_return_pct") is None
            or not isinstance(features, Mapping)
            or set(features) != set(PREDICTION_FEATURES)
        ):
            continue
        unavailable = _record_unavailable_features(item)
        feature_rows.append(
            [
                *[_safe_float(features[name]) for name in PREDICTION_FEATURES],
                _horizon_model_value(item_horizon),
            ]
        )
        feature_availability.append(
            [
                *[name not in unavailable for name in PREDICTION_FEATURES],
                True,
            ]
        )
        targets.append(_safe_float(item.get("actual_net_return_pct")))
        group_keys.append(
            (
                str(item.get("symbol") or ""),
                (signal_at or outcome_at).date().isoformat(),
            )
        )
    if not targets:
        return None

    feature_matrix = np.asarray(feature_rows, dtype=float)
    availability_matrix = np.asarray(feature_availability, dtype=bool)
    target_vector = np.asarray(targets, dtype=float)
    group_counts: Dict[tuple[str, str], int] = {}
    for key in group_keys:
        group_counts[key] = group_counts.get(key, 0) + 1
    current_date = as_of.date().isoformat()
    completed_group_counts = np.asarray(
        [
            count
            for (_, date_text), count in group_counts.items()
            if date_text < current_date
        ],
        dtype=float,
    )
    completed_group_size = (
        _weighted_quantile(
            completed_group_counts,
            np.ones(len(completed_group_counts), dtype=float),
            0.5,
        )
        if len(completed_group_counts)
        else 0.0
    )
    weights = np.asarray(
        [
            1.0
            / max(
                float(group_counts[key]),
                completed_group_size if key[1] == current_date else 0.0,
            )
            for key in group_keys
        ],
        dtype=float,
    )
    # An unfinished intraday group must not carry a full trading day's weight.
    effective_sample_size = max(1.0, float(np.sum(weights)))
    # Limit-move outcomes remain in residual risk but cannot dominate the fit.
    target_lower = _weighted_quantile(target_vector, weights, 0.01)
    target_upper = _weighted_quantile(target_vector, weights, 0.99)
    fit_targets = np.clip(target_vector, target_lower, target_upper)

    medians = []
    scales = []
    feature_mins = []
    feature_maxs = []
    for index in range(feature_matrix.shape[1]):
        if index == len(PREDICTION_FEATURES):
            medians.append(0.0)
            scales.append(1.0)
            feature_mins.append(-1.0)
            feature_maxs.append(0.0)
            continue
        observed = availability_matrix[:, index]
        if not np.any(observed):
            medians.append(0.0)
            scales.append(1.0)
            feature_mins.append(0.0)
            feature_maxs.append(0.0)
            continue
        values = feature_matrix[observed, index]
        observed_weights = weights[observed]
        median = _weighted_quantile(values, observed_weights, 0.5)
        mad = _weighted_quantile(
            np.abs(values - median),
            observed_weights,
            0.5,
        ) * 1.4826
        fallback_scale = sqrt(
            float(np.dot(observed_weights, np.square(values - median)))
            / max(1e-12, float(np.sum(observed_weights)))
        )
        medians.append(median)
        scales.append(mad if mad > 1e-6 else max(fallback_scale, 1e-6))
        feature_mins.append(float(np.min(values)))
        feature_maxs.append(float(np.max(values)))
    medians_array = np.asarray(medians, dtype=float)
    scales_array = np.asarray(scales, dtype=float)
    imputed_matrix = np.where(
        availability_matrix,
        feature_matrix,
        medians_array,
    )
    design = np.column_stack(
        (
            np.ones(len(target_vector)),
            (imputed_matrix - medians_array) / scales_array,
        )
    )
    weighted_design = design * weights[:, None]
    penalty = len(MODEL_FEATURES) / sqrt(float(effective_sample_size))
    regularization = penalty * np.eye(design.shape[1])
    gram = design.T @ weighted_design + regularization
    try:
        coefficients = np.linalg.solve(
            gram,
            design.T @ (weights * fit_targets),
        )
    except np.linalg.LinAlgError:
        return None

    fitted = design @ coefficients
    residuals = target_vector - fitted
    residual_rms = sqrt(
        float(np.dot(weights, np.square(residuals)))
        / max(1e-12, float(np.sum(weights)))
    )
    residual_p10 = _weighted_quantile(residuals, weights, 0.10)

    return RidgeModel(
        coefficients=tuple(float(value) for value in coefficients),
        medians=tuple(float(value) for value in medians_array),
        scales=tuple(float(value) for value in scales_array),
        feature_mins=tuple(feature_mins),
        feature_maxs=tuple(feature_maxs),
        sample_count=len(target_vector),
        effective_sample_size=effective_sample_size,
        target_min_pct=target_lower,
        target_max_pct=target_upper,
        residual_rms_pct=residual_rms,
        residual_p10_pct=residual_p10,
    )


def _not_ready(row: FeatureRow, reason: str, horizon: int) -> Prediction:
    return Prediction(
        ready=False,
        reason=reason,
        horizon_seconds=horizon,
        sample_count=0,
        expected_net_return_pct=0.0,
        lower_net_return_pct=0.0,
        compact_features=dict(row.compact_features),
        unavailable_features=tuple(row.unavailable_features),
    )


def predict_batch(
    rows: Mapping[str, FeatureRow],
    *,
    model: RidgeModel | None,
    evaluated_candidate_count: int,
    horizon_seconds: int = PREDICTION_HORIZONS_SECONDS[-1],
) -> Dict[str, Prediction]:
    horizon = max(30, int(horizon_seconds or PREDICTION_HORIZONS_SECONDS[-1]))
    candidate_count = max(1, int(evaluated_candidate_count or 0))
    predictions: Dict[str, Prediction] = {}
    for symbol, row in rows.items():
        if not row.ready:
            predictions[symbol] = _not_ready(row, row.reason, horizon)
            continue
        if model is None:
            predictions[symbol] = _not_ready(row, "no_settled_outcomes", horizon)
            continue

        unavailable = set(row.unavailable_features)
        features = np.asarray(
            [
                *[
                    (
                        float(model.medians[index])
                        if name in unavailable
                        else _safe_float(row.compact_features[name])
                    )
                    for index, name in enumerate(PREDICTION_FEATURES)
                ],
                _horizon_model_value(horizon),
            ],
            dtype=float,
        )
        bounded_features = np.clip(
            features,
            np.asarray(model.feature_mins),
            np.asarray(model.feature_maxs),
        )
        vector = np.concatenate(
            (
                [1.0],
                (bounded_features - np.asarray(model.medians))
                / np.asarray(model.scales),
            )
        )
        model_mean = float(np.dot(vector, np.asarray(model.coefficients)))
        bounded_mean = min(model.target_max_pct, max(model.target_min_pct, model_mean))
        mean_standard_error = model.residual_rms_pct / sqrt(
            float(model.effective_sample_size)
        )
        finite_sample_inflation = sqrt(
            float(model.effective_sample_size)
            / float(max(1, model.effective_sample_size - 2))
        )
        mean_standard_error *= finite_sample_inflation
        selection_adjustment = mean_standard_error * sqrt(
            2.0 * log(float(candidate_count))
        )
        expected = bounded_mean - selection_adjustment
        lower = expected + min(0.0, model.residual_p10_pct)
        predictions[symbol] = Prediction(
            ready=True,
            reason="ok",
            horizon_seconds=horizon,
            sample_count=model.sample_count,
            expected_net_return_pct=round(expected, 8),
            lower_net_return_pct=round(lower, 8),
            compact_features=dict(row.compact_features),
            unavailable_features=tuple(row.unavailable_features),
        )
    return predictions
