from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, sqrt
from statistics import mean, median
from typing import Any, Dict, Sequence

from src.models import Quote


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _sigmoid(value: float) -> float:
    clipped = _clip(value, -12.0, 12.0)
    return 1.0 / (1.0 + exp(-clipped))


def _linear_trend(times: Sequence[float], prices: Sequence[int]) -> tuple[float, float]:
    """Return percent-per-second slope and fit quality for a price path."""
    if len(times) < 2 or len(times) != len(prices) or prices[0] <= 0:
        return 0.0, 0.0
    x_mean = mean(times)
    values = [((float(price) / float(prices[0])) - 1.0) * 100.0 for price in prices]
    y_mean = mean(values)
    denominator = sum((value - x_mean) ** 2 for value in times)
    if denominator <= 0.0:
        return 0.0, 0.0
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(times, values)) / denominator
    fitted = [y_mean + slope * (x - x_mean) for x in times]
    total = sum((value - y_mean) ** 2 for value in values)
    residual = sum((value - estimate) ** 2 for value, estimate in zip(values, fitted))
    fit = 1.0 - (residual / total) if total > 1e-12 else 1.0
    return float(slope), _clip(fit, 0.0, 1.0)


def _flow_surge(values: Sequence[int], times: Sequence[float]) -> float:
    if len(values) < 4 or len(values) != len(times):
        return 1.0
    rates = []
    for previous, current, start, end in zip(values, values[1:], times, times[1:]):
        elapsed = max(0.001, float(end) - float(start))
        rates.append(max(0.0, float(current) - float(previous)) / elapsed)
    if len(rates) < 3:
        return 1.0
    split = max(1, len(rates) // 2)
    prior = median(rates[:split]) if rates[:split] else 0.0
    recent = median(rates[split:]) if rates[split:] else 0.0
    return _clip(recent / max(1.0, prior), 0.0, 8.0)


@dataclass(frozen=True)
class ShortHorizonPrediction:
    ready: bool
    reason: str
    horizon_seconds: int
    sample_count: int
    predicted_return_pct: float
    lower_bound_return_pct: float
    upper_bound_return_pct: float
    confidence: float
    direction_score: float
    volatility_pct: float
    features: Dict[str, float] = field(default_factory=dict)


def predict_short_horizon_return(
    quote: Quote,
    *,
    recent_quotes: Sequence[Quote],
    leader: Any = None,
    market_state: Dict[str, Any] | None = None,
    horizon_seconds: int = 180,
    min_samples: int = 5,
) -> ShortHorizonPrediction:
    """Estimate short-horizon return from repeated, independent price moves.

    The predictor deliberately gives little weight to one quote jump. A move
    becomes credible when several distinct price changes agree, while volume,
    leader rank, and market state only make small adjustments to that observed
    price path.
    """

    current_price = max(0, int(getattr(quote, "current_price", 0) or 0))
    horizon = max(30, int(horizon_seconds or 180))
    if current_price <= 0:
        return ShortHorizonPrediction(False, "invalid_price", horizon, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    window = [item for item in recent_quotes if int(getattr(item, "current_price", 0) or 0) > 0]
    if not window or window[-1] is not quote:
        last = window[-1] if window else None
        if last is None or getattr(last, "timestamp", None) != getattr(quote, "timestamp", None):
            window.append(quote)
    min_required = max(3, int(min_samples or 0))
    window = window[-max(15, min_required):]
    sample_count = len(window)
    if sample_count < min_required:
        return ShortHorizonPrediction(
            False,
            "insufficient_samples",
            horizon,
            sample_count,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    prices = [max(1, int(getattr(item, "current_price", 0) or 0)) for item in window]
    raw_timestamps = [getattr(item, "timestamp", None) for item in window]
    times = [0.0]
    for index in range(1, sample_count):
        elapsed = 0.0
        previous = raw_timestamps[index - 1]
        current = raw_timestamps[index]
        if previous is not None and current is not None:
            try:
                elapsed = float((current - previous).total_seconds())
            except TypeError:
                elapsed = 0.0
        if elapsed <= 0.0:
            elapsed = 10.0
        times.append(times[-1] + elapsed)

    returns = [
        ((float(current) / float(previous)) - 1.0) * 100.0
        for previous, current in zip(prices, prices[1:])
    ]
    if not returns:
        return ShortHorizonPrediction(False, "insufficient_returns", horizon, sample_count, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    nonzero_returns = [value for value in returns if abs(value) > 1e-9]
    positive_moves = [value for value in nonzero_returns if value > 0.0]
    negative_moves = [value for value in nonzero_returns if value < 0.0]
    movement_count = len(nonzero_returns)
    positive_move_ratio = len(positive_moves) / max(1, movement_count)
    negative_move_ratio = len(negative_moves) / max(1, movement_count)
    active_move_ratio = movement_count / max(1, len(returns))

    tail_returns = returns[-min(6, len(returns)):]
    tail_moves = [value for value in tail_returns if abs(value) > 1e-9]
    tail_positive = [value for value in tail_moves if value > 0.0]
    tail_negative = [value for value in tail_moves if value < 0.0]
    tail_positive_ratio = len(tail_positive) / max(1, len(tail_moves))
    tail_negative_ratio = len(tail_negative) / max(1, len(tail_moves))
    tail_active_move_ratio = len(tail_moves) / max(1, len(tail_returns))

    positive_move_support = _clip(
        ((len(positive_moves) - 0.75) / 3.25)
        * (0.42 + 0.58 * positive_move_ratio)
        * (0.35 + 0.65 * active_move_ratio),
        0.0,
        1.0,
    )
    tail_positive_support = _clip(
        ((len(tail_positive) - 0.50) / 2.75)
        * (0.38 + 0.62 * tail_positive_ratio)
        * (0.25 + 0.75 * tail_active_move_ratio),
        0.0,
        1.0,
    )
    negative_move_support = _clip(
        ((len(negative_moves) - 0.75) / 3.25)
        * (0.42 + 0.58 * negative_move_ratio)
        * (0.35 + 0.65 * active_move_ratio),
        0.0,
        1.0,
    )
    tail_negative_support = _clip(
        ((len(tail_negative) - 0.50) / 2.75)
        * (0.38 + 0.62 * tail_negative_ratio)
        * (0.25 + 0.75 * tail_active_move_ratio),
        0.0,
        1.0,
    )

    slow_slope, slow_fit = _linear_trend(times, prices)
    # The tail trend must be shorter than the full window even at the minimum
    # sample count. Otherwise a launch followed by a flat hold gives identical
    # slow/fast slopes and cannot be recognized as deceleration.
    fast_count = min(7, max(3, sample_count - 2))
    fast_times_raw = times[-fast_count:]
    fast_start = fast_times_raw[0]
    fast_times = [value - fast_start for value in fast_times_raw]
    fast_prices = prices[-fast_count:]
    fast_slope, fast_fit = _linear_trend(fast_times, fast_prices)
    terminal_count = min(3, sample_count)
    terminal_times_raw = times[-terminal_count:]
    terminal_start = terminal_times_raw[0]
    terminal_times = [value - terminal_start for value in terminal_times_raw]
    terminal_prices = prices[-terminal_count:]
    terminal_slope, terminal_fit = _linear_trend(terminal_times, terminal_prices)
    slow_projection = slow_slope * float(horizon)
    fast_projection = fast_slope * float(horizon)
    terminal_projection = terminal_slope * float(horizon)
    terminal_reversal = min(0.0, terminal_projection - fast_projection)
    raw_projection = 0.18 * slow_projection + 0.20 * fast_projection + 0.16 * terminal_reversal
    raw_projection = _clip(raw_projection, -3.6, 3.6)
    direction_support = (
        max(positive_move_support, 0.70 * tail_positive_support)
        if raw_projection >= 0.0
        else max(negative_move_support, 0.70 * tail_negative_support)
    )
    trend_edge = raw_projection * (0.08 + 0.92 * direction_support)

    short_anchor_index = max(0, sample_count - 4)
    short_return = ((float(prices[-1]) / float(prices[short_anchor_index])) - 1.0) * 100.0
    full_return = ((float(prices[-1]) / float(prices[0])) - 1.0) * 100.0
    if terminal_slope < fast_slope:
        acceleration_projection = (terminal_slope - fast_slope) * float(horizon)
    else:
        acceleration_projection = (fast_slope - slow_slope) * float(horizon)
    acceleration_edge = _clip(acceleration_projection * 0.10, -0.55, 0.55) * direction_support

    avg_return = mean(returns)
    variance = sum((value - avg_return) ** 2 for value in returns) / max(1, len(returns) - 1)
    volatility = variance ** 0.5
    avg_abs_return = mean(abs(value) for value in returns)
    weighted_return = sum((index + 1) * value for index, value in enumerate(returns)) / sum(range(1, len(returns) + 1))
    split = max(1, len(returns) // 2)
    early_velocity = mean(returns[:split]) if returns[:split] else 0.0
    late_velocity = mean(returns[split:]) if returns[split:] else 0.0
    acceleration = late_velocity - early_velocity
    up_ratio = sum(1 for value in returns if value > 0.0) / max(1, len(returns))

    positive_impulse = sum(positive_moves)
    dominant_positive_share = max(positive_moves) / max(1e-9, positive_impulse) if positive_moves else 0.0
    dominant_positive_index = max(
        (index for index, value in enumerate(returns) if value == max(positive_moves)),
        default=-1,
    ) if positive_moves else -1
    positive_follow_through = sum(
        1
        for value in returns[dominant_positive_index + 1:]
        if value > 0.0
    ) if dominant_positive_index >= 0 else 0
    dominant_age = len(returns) - 1 - dominant_positive_index if dominant_positive_index >= 0 else len(returns)
    recent_dominant_pressure = _clip((3.5 - float(dominant_age)) / 3.5, 0.0, 1.0)
    dominant_impulse_risk = _clip(
        (dominant_positive_share - 0.42) / 0.58,
        0.0,
        1.0,
    ) * recent_dominant_pressure * (1.0 - 0.42 * _clip(positive_follow_through / 2.0, 0.0, 1.0))
    prior_positive_impulse = sum(max(0.0, value) for value in returns[:dominant_positive_index]) if dominant_positive_index > 0 else 0.0
    dominant_value = max(positive_moves) if positive_moves else 0.0
    impulse_excess_risk = _clip(
        (dominant_value - max(0.10, prior_positive_impulse * 0.68))
        / max(0.16, avg_abs_return * 1.8),
        0.0,
        1.0,
    ) * recent_dominant_pressure * (1.0 - 0.50 * _clip(positive_follow_through / 2.0, 0.0, 1.0))
    prior_tail_returns = returns[-min(6, len(returns)):-1]
    if returns[-1] > 0.0 and prior_tail_returns:
        prior_tail_mean = mean(prior_tail_returns)
        prior_tail_variance = sum(
            (value - prior_tail_mean) ** 2 for value in prior_tail_returns
        ) / max(1, len(prior_tail_returns) - 1)
        prior_tail_std = prior_tail_variance ** 0.5
        immediate_positive_confirmation = _clip(
            max(0.0, prior_tail_returns[-1]) / max(0.16, returns[-1]),
            0.0,
            1.0,
        )
        latest_positive_surprise_risk = _clip(
            (returns[-1] - max(0.0, prior_tail_mean) - prior_tail_std)
            / max(0.16, prior_tail_std * 2.0),
            0.0,
            1.0,
        ) * (1.0 - immediate_positive_confirmation) ** 2
    else:
        latest_positive_surprise_risk = 0.0
    single_tick_impulse_risk = _clip(
        max(dominant_impulse_risk, impulse_excess_risk, latest_positive_surprise_risk),
        0.0,
        1.0,
    )

    recent_high = max(prices)
    recent_low = min(prices)
    recent_drawdown_pct = ((float(recent_high) / float(prices[-1])) - 1.0) * 100.0
    rebound_from_low_pct = ((float(prices[-1]) / float(recent_low)) - 1.0) * 100.0
    recent_swing_pct = ((float(recent_high) / float(recent_low)) - 1.0) * 100.0
    pre_rebound_pullback_pct = max(
        0.0,
        ((float(prices[0]) / float(recent_low)) - 1.0) * 100.0,
    )

    open_price = max(0, int(getattr(quote, "open_price", 0) or 0))
    high_price = max(0, int(getattr(quote, "high_price", 0) or 0))
    low_price = max(0, int(getattr(quote, "low_price", 0) or 0))
    day_span = max(1, high_price - low_price)
    day_range_position = (
        _clip((float(current_price) - float(low_price)) / float(day_span), 0.0, 1.0)
        if high_price > 0 and low_price > 0
        else 0.5
    )
    vs_open_pct = ((float(current_price) / float(open_price)) - 1.0) * 100.0 if open_price > 0 else 0.0
    day_high_retreat_pct = ((float(high_price) / float(current_price)) - 1.0) * 100.0 if high_price > current_price else 0.0

    volumes = [max(0, int(getattr(item, "volume", 0) or 0)) for item in window]
    amounts = [max(0, int(getattr(item, "trade_amount", 0) or 0)) for item in window]
    flow_surge = max(_flow_surge(volumes, times), _flow_surge(amounts, times))
    flow_confirmation = _clip((flow_surge - 1.0) / 2.2, 0.0, 1.0)
    flow_impulse_risk = _clip(
        single_tick_impulse_risk * _clip((flow_surge - 2.0) / 4.0, 0.0, 1.0),
        0.0,
        1.0,
    )

    gaps = [current - previous for previous, current in zip(times, times[1:])]
    typical_gap = median(gaps) if gaps else 10.0
    max_gap = max(gaps) if gaps else typical_gap
    irregular_gap_risk = _clip(
        (max_gap - max(24.0, typical_gap * 2.5)) / max(20.0, typical_gap * 2.5),
        0.0,
        1.0,
    )
    sparse_sampling_risk = _clip((typical_gap - 24.0) / 72.0, 0.0, 1.0)
    quote_gap_risk = max(irregular_gap_risk, sparse_sampling_risk)

    leader_percentile = _clip(_safe_float(getattr(leader, "leader_percentile", 0.0), 0.0), 0.0, 1.0)
    effective_score = _safe_float(getattr(leader, "effective_leader_score", 0.0), 0.0)
    leader_accel = _safe_float(getattr(leader, "recent_acceleration_pct", 0.0), 0.0)
    high_proximity = _clip(
        _safe_float(getattr(leader, "high_proximity", day_range_position), day_range_position),
        0.0,
        1.0,
    )
    absolute_leader_quality = _clip(
        0.46 * leader_percentile
        + 0.36 * _clip((effective_score + 0.05) / 1.10, 0.0, 1.0)
        + 0.18 * _clip((leader_accel + 0.05) / 0.85, 0.0, 1.0),
        0.0,
        1.0,
    )

    state = market_state or {}
    heat = _clip(_safe_float(state.get("tape_heat", 0.0), 0.0), 0.0, 1.0)
    caution = _clip(_safe_float(state.get("tape_caution", 0.0), 0.0), 0.0, 1.0)
    overheat = _clip(_safe_float(state.get("overheat", 0.0), 0.0), 0.0, 1.0)
    market_p90 = _safe_float(state.get("vs_open_p90", 0.0), 0.0)
    overextension_floor = max(4.0, market_p90 * 0.82 if market_p90 > 0.0 else 0.0)
    overextension_pct = max(0.0, vs_open_pct - overextension_floor)
    market_extension_pressure = _clip(overextension_pct / max(4.0, overextension_floor), 0.0, 1.0)
    prior_window_extension_pct = max(0.0, vs_open_pct - max(0.0, full_return))
    individual_extension_scale = max(1.0, abs(full_return) * 2.0 + volatility * 4.0)
    individual_extension_pressure = _clip(
        prior_window_extension_pct / individual_extension_scale,
        0.0,
        1.0,
    )
    extension_pressure = max(market_extension_pressure, individual_extension_pressure)
    late_acceleration_pressure = _clip(
        max(0.0, acceleration) / max(0.08, avg_abs_return * 1.4),
        0.0,
        1.0,
    )
    late_move_pressure = _clip(
        max(0.0, short_return) / max(0.15, abs(full_return) * 0.45 + volatility),
        0.0,
        1.0,
    )
    late_extension_risk = _clip(
        extension_pressure * late_acceleration_pressure * late_move_pressure,
        0.0,
        1.0,
    )
    rebound_structure_relief = _clip(
        pre_rebound_pullback_pct / max(0.20, rebound_from_low_pct * 0.55),
        0.0,
        1.0,
    )
    late_extension_risk *= 1.0 - rebound_structure_relief

    negative_tail_score = _clip(
        0.55 * tail_negative_support
        + 0.25 * _clip(recent_drawdown_pct / max(0.25, volatility * 2.5), 0.0, 1.0)
        + 0.20 * (1.0 if returns[-1] < 0.0 else 0.0),
        0.0,
        1.0,
    )
    deceleration_score = _clip(
        (slow_slope - terminal_slope) * float(horizon) / max(0.35, abs(slow_projection)),
        0.0,
        1.0,
    ) if slow_slope > 0.0 else 0.0
    effective_deceleration_score = deceleration_score * (
        1.0 - 0.90 * max(positive_move_support, tail_positive_support)
    )
    deceleration_pressure = effective_deceleration_score * (0.35 + 0.65 * extension_pressure)

    confirmed_tail_continuation_score = _clip(
        max(positive_move_support, tail_positive_support)
        * (0.42 + 0.30 * fast_fit + 0.18 * flow_confirmation + 0.10 * absolute_leader_quality)
        * (1.0 - 0.72 * max(single_tick_impulse_risk, late_extension_risk))
        * (1.0 - 0.55 * negative_tail_score),
        0.0,
        1.0,
    )
    high_hold_integrity = min(
        _clip((0.55 - recent_drawdown_pct) / 0.55, 0.0, 1.0),
        _clip((0.45 - day_high_retreat_pct) / 0.45, 0.0, 1.0),
    )
    confirmed_high_hold_continuation_score = _clip(
        _clip((max(high_proximity, day_range_position) - 0.86) / 0.14, 0.0, 1.0)
        * high_hold_integrity
        * max(positive_move_support, 0.75 * tail_positive_support)
        * (0.55 + 0.25 * fast_fit + 0.20 * flow_confirmation)
        * (1.0 - max(single_tick_impulse_risk, late_extension_risk)),
        0.0,
        1.0,
    )
    confirmed_intraday_flow_hold_score = _clip(
        confirmed_high_hold_continuation_score * (0.72 + 0.28 * flow_confirmation),
        0.0,
        1.0,
    )

    timestamp = getattr(quote, "timestamp", None)
    seconds_after_open = None
    if timestamp is not None and hasattr(timestamp, "hour"):
        seconds_after_open = timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second - 9 * 3600
    opening_phase_pressure = (
        _clip((420.0 - float(seconds_after_open)) / 420.0, 0.0, 1.0)
        if seconds_after_open is not None and seconds_after_open >= 0
        else 0.0
    )
    confirmed_opening_launch_score = _clip(
        opening_phase_pressure
        * max(positive_move_support, tail_positive_support)
        * (0.45 + 0.30 * fast_fit + 0.25 * absolute_leader_quality)
        * (1.0 - 0.75 * single_tick_impulse_risk)
        * (1.0 - 0.55 * negative_tail_score),
        0.0,
        1.0,
    )

    weak_absolute_leader_risk = _clip(
        (0.52 - absolute_leader_quality) / 0.52,
        0.0,
        1.0,
    ) * extension_pressure * (1.0 - confirmed_tail_continuation_score)
    chase_risk_score = _clip(
        0.52 * single_tick_impulse_risk
        + 0.45 * late_extension_risk
        + 0.30 * extension_pressure * (1.0 - max(positive_move_support, tail_positive_support))
        + 0.18 * quote_gap_risk
        + 0.24 * flow_impulse_risk,
        0.0,
        1.0,
    )
    rejection_risk_score = _clip(
        0.28 * negative_tail_score
        + 0.18 * _clip(recent_drawdown_pct / max(0.30, volatility * 2.8), 0.0, 1.0)
        + 0.44 * _clip(
            day_high_retreat_pct / max(0.45, recent_swing_pct + volatility),
            0.0,
            1.0,
        )
        + 0.10 * quote_gap_risk,
        0.0,
        1.0,
    )
    trap_risk_score = _clip(
        0.45 * chase_risk_score
        + 0.35 * rejection_risk_score
        + 0.20 * weak_absolute_leader_risk,
        0.0,
        1.0,
    )

    consistency_edge = (positive_move_ratio - negative_move_ratio) * active_move_ratio * 0.12
    flow_edge = flow_confirmation * max(positive_move_support, tail_positive_support) * 0.10
    leader_edge = (absolute_leader_quality - 0.50) * 0.12
    high_hold_edge = confirmed_high_hold_continuation_score * 0.16
    market_edge = 0.06 * heat - 0.07 * caution - 0.03 * overheat
    directional_evidence = (
        trend_edge * (1.0 - 0.80 * effective_deceleration_score)
        + acceleration_edge
        + consistency_edge
        + flow_edge
        + leader_edge
        + high_hold_edge
        + market_edge
    )
    directional_evidence_reliability = 1.0
    if directional_evidence > 0.0:
        directional_evidence_reliability = 1.0 - 0.85 * single_tick_impulse_risk
        directional_evidence *= directional_evidence_reliability
    directional_reversal_penalty = (
        0.95 * rejection_risk_score
        + 0.32 * day_high_retreat_pct * rejection_risk_score
        + 0.16 * weak_absolute_leader_risk
        + 0.55 * deceleration_pressure
    )
    predicted = directional_evidence - directional_reversal_penalty
    prediction_clip = _clip(2.6 + 0.45 * heat - 0.25 * caution, 2.0, 3.2)
    predicted = _clip(predicted, -prediction_clip, prediction_clip)

    future_steps = max(1.0, float(horizon) / max(5.0, typical_gap))
    # Three points are enough to spot a reversal, but not to establish that the
    # whole move is reliable. Keep terminal_fit diagnostic-only so a short,
    # perfectly straight burst cannot inflate confidence by itself.
    trend_fit = max(slow_fit, fast_fit)
    sample_quality = _clip((sample_count - min_required + 1) / 6.0, 0.0, 1.0)
    sample_uncertainty_floor = 0.26 + 0.42 * (1.0 - sample_quality)
    uncertainty = max(
        sample_uncertainty_floor,
        volatility * sqrt(future_steps) * (0.48 + 0.42 * (1.0 - trend_fit))
        + 0.34 * single_tick_impulse_risk
        + 1.65 * late_extension_risk
        + 0.62 * flow_impulse_risk
        + 0.40 * quote_gap_risk
        + 0.72 * rejection_risk_score
        + 0.14 * extension_pressure * (1.0 - direction_support)
        + 0.55 * deceleration_pressure,
    )
    uncertainty += 0.28 * abs(predicted) * (1.0 - sample_quality)
    lower = _clip(predicted - uncertainty, -5.0, 5.0)
    upper = _clip(predicted + uncertainty, -5.0, 5.0)
    direction_score = _sigmoid(predicted / max(0.16, uncertainty * 0.85))
    confidence = _clip(
        0.18
        + 0.18 * sample_quality
        + 0.24 * direction_support
        + 0.15 * trend_fit
        + 0.08 * flow_confirmation
        + 0.08 * absolute_leader_quality
        + 0.09 * confirmed_high_hold_continuation_score
        - 0.28 * single_tick_impulse_risk
        - 0.38 * late_extension_risk
        - 0.20 * flow_impulse_risk
        - 0.16 * quote_gap_risk
        - 0.24 * rejection_risk_score
        - 0.08 * weak_absolute_leader_risk
        - 0.22 * deceleration_pressure,
        0.0,
        1.0,
    )

    follow_through_score = _clip(
        0.50 * max(positive_move_support, tail_positive_support)
        + 0.22 * fast_fit
        + 0.16 * flow_confirmation
        + 0.12 * absolute_leader_quality
        - 0.42 * single_tick_impulse_risk
        - 0.55 * late_extension_risk
        - 0.30 * negative_tail_score,
        0.0,
        1.0,
    )
    continuation_quality = _clip(
        0.45 * follow_through_score
        + 0.25 * confirmed_tail_continuation_score
        + 0.15 * confirmed_high_hold_continuation_score
        + 0.15 * absolute_leader_quality
        - 0.28 * trap_risk_score,
        0.0,
        1.0,
    )
    tail_flow_spike_risk = flow_impulse_risk
    high_hold_flow_relief = confirmed_high_hold_continuation_score * flow_confirmation
    intraday_high_hold_quality = _clip(
        0.55 * high_hold_integrity
        + 0.25 * max(positive_move_support, tail_positive_support)
        + 0.20 * flow_confirmation
        - 0.30 * negative_tail_score,
        0.0,
        1.0,
    )

    return ShortHorizonPrediction(
        ready=True,
        reason="ok",
        horizon_seconds=horizon,
        sample_count=sample_count,
        predicted_return_pct=round(predicted, 6),
        lower_bound_return_pct=round(lower, 6),
        upper_bound_return_pct=round(upper, 6),
        confidence=round(confidence, 6),
        direction_score=round(direction_score, 6),
        volatility_pct=round(volatility, 6),
        features={
            "weighted_return_pct": round(weighted_return, 6),
            "late_velocity_pct": round(late_velocity, 6),
            "acceleration_pct": round(acceleration, 6),
            "short_return_pct": round(short_return, 6),
            "full_return_pct": round(full_return, 6),
            "up_ratio": round(up_ratio, 6),
            "active_move_ratio": round(active_move_ratio, 6),
            "positive_move_support": round(positive_move_support, 6),
            "tail_positive_support": round(tail_positive_support, 6),
            "single_tick_impulse_risk": round(single_tick_impulse_risk, 6),
            "slow_trend_pct_per_second": round(slow_slope, 8),
            "fast_trend_pct_per_second": round(fast_slope, 8),
            "terminal_trend_pct_per_second": round(terminal_slope, 8),
            "slow_trend_fit": round(slow_fit, 6),
            "fast_trend_fit": round(fast_fit, 6),
            "terminal_trend_fit": round(terminal_fit, 6),
            "flow_surge": round(flow_surge, 6),
            "flow_confirmation": round(flow_confirmation, 6),
            "tail_flow_confirmation": round(flow_confirmation, 6),
            "tail_sequence_confirmation": round(tail_positive_support, 6),
            "tail_flow_spike_risk": round(tail_flow_spike_risk, 6),
            "recent_drawdown_pct": round(recent_drawdown_pct, 6),
            "rebound_from_low_pct": round(rebound_from_low_pct, 6),
            "recent_swing_pct": round(recent_swing_pct, 6),
            "pre_rebound_pullback_pct": round(pre_rebound_pullback_pct, 6),
            "vs_open_pct": round(vs_open_pct, 6),
            "day_range_position": round(day_range_position, 6),
            "day_high_retreat_pct": round(day_high_retreat_pct, 6),
            "overextension_pct": round(overextension_pct, 6),
            "overextension_floor_pct": round(overextension_floor, 6),
            "prior_window_extension_pct": round(prior_window_extension_pct, 6),
            "late_extension_risk": round(late_extension_risk, 6),
            "dynamic_momentum_context": round(1.0 + market_edge, 6),
            "dynamic_risk_context": round(1.0 + trap_risk_score, 6),
            "dynamic_volatility_context": round(1.0 + uncertainty, 6),
            "dynamic_prediction_clip_pct": round(prediction_clip, 6),
            "directional_evidence_pct": round(directional_evidence, 6),
            "directional_evidence_reliability": round(directional_evidence_reliability, 6),
            "directional_reversal_penalty_pct": round(directional_reversal_penalty, 6),
            "deceleration_score": round(deceleration_score, 6),
            "effective_deceleration_score": round(effective_deceleration_score, 6),
            "deceleration_pressure": round(deceleration_pressure, 6),
            "negative_tail_score": round(negative_tail_score, 6),
            "confirmed_tail_continuation_score": round(confirmed_tail_continuation_score, 6),
            "confirmed_tail_edge_pct": round(max(0.0, trend_edge) * confirmed_tail_continuation_score, 6),
            "confirmed_high_hold_continuation_score": round(confirmed_high_hold_continuation_score, 6),
            "high_hold_flow_relief": round(high_hold_flow_relief, 6),
            "intraday_high_hold_quality": round(intraday_high_hold_quality, 6),
            "confirmed_intraday_flow_hold_score": round(confirmed_intraday_flow_hold_score, 6),
            "intraday_flow_hold_edge_pct": round(high_hold_edge, 6),
            "confirmed_opening_launch_score": round(confirmed_opening_launch_score, 6),
            "opening_price_action_quality": round(max(positive_move_support, tail_positive_support), 6),
            "opening_high_hold_quality": round(confirmed_high_hold_continuation_score, 6),
            "opening_hold_absorption_score": round(confirmed_high_hold_continuation_score * (1.0 - negative_tail_score), 6),
            "opening_leader_confirmation": round(absolute_leader_quality, 6),
            "opening_impulse_risk_for_confirmation": round(single_tick_impulse_risk, 6),
            "opening_launch_edge_pct": round(max(0.0, trend_edge) * confirmed_opening_launch_score, 6),
            "absolute_leader_quality": round(absolute_leader_quality, 6),
            "raw_leader_support_quality": round(absolute_leader_quality, 6),
            "leader_support_quality": round(absolute_leader_quality * (1.0 - weak_absolute_leader_risk), 6),
            "leader_run_pressure": round(extension_pressure, 6),
            "weak_leader_context_pressure": round(extension_pressure, 6),
            "weak_absolute_leader_risk": round(weak_absolute_leader_risk, 6),
            "leader_quality_gap_risk": round(weak_absolute_leader_risk, 6),
            "late_rebound_distribution_risk": round(rejection_risk_score * (1.0 - tail_positive_support), 6),
            "weak_opening_leader_risk": round(weak_absolute_leader_risk * opening_phase_pressure, 6),
            "unconfirmed_impulse_risk": round(single_tick_impulse_risk, 6),
            "raw_unconfirmed_impulse_risk": round(single_tick_impulse_risk, 6),
            "opening_instability_risk": round(
                max(chase_risk_score, 0.70 * single_tick_impulse_risk + 0.30 * flow_impulse_risk)
                * opening_phase_pressure,
                6,
            ),
            "intraday_impulse_instability_risk": round(
                max(single_tick_impulse_risk, flow_impulse_risk, quote_gap_risk * tail_positive_support),
                6,
            ),
            "raw_intraday_impulse_instability_risk": round(
                max(single_tick_impulse_risk, flow_impulse_risk, quote_gap_risk * tail_positive_support),
                6,
            ),
            "quote_gap_risk": round(quote_gap_risk, 6),
            "follow_through_score": round(follow_through_score, 6),
            "continuation_quality": round(continuation_quality, 6),
            "rejection_risk_score": round(rejection_risk_score, 6),
            "chase_risk_score": round(chase_risk_score, 6),
            "trap_risk_score": round(trap_risk_score, 6),
            "heat": round(heat, 6),
            "caution": round(caution, 6),
        },
    )
