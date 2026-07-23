from __future__ import annotations

from statistics import mean
from typing import Any, Dict, List, Optional

from src.analytics.math_signals import LeaderSignal
from src.models import Quote


def symbol_micro_edge_metrics(strategy: Any, quote: Quote, *, leader: Optional[LeaderSignal] = None) -> Dict[str, Any]:
    self = strategy
    if not bool(getattr(self.config, "enable_symbol_micro_edge", True)):
        return {"micro_ready": False}
    current_price = max(0, int(quote.current_price or 0))
    if current_price <= 0:
        return {"micro_ready": False}

    recent_quotes = list(self._recent_quotes.get(quote.symbol, []))
    if not recent_quotes or int(getattr(recent_quotes[-1], "current_price", 0) or 0) != current_price:
        recent_quotes.append(quote)
    window = recent_quotes[-max(4, int(self.config.setup_recent_quote_window)) :]
    prices = [int(item.current_price or 0) for item in window if int(item.current_price or 0) > 0]
    min_history = max(3, int(getattr(self.config, "symbol_micro_edge_min_history", 4) or 4))
    if len(prices) < min_history:
        return {"micro_ready": False, "micro_sample_count": float(len(prices))}

    returns: List[float] = []
    for previous, current in zip(prices, prices[1:]):
        if previous > 0:
            returns.append(((current - previous) / previous) * 100.0)
    if not returns:
        return {"micro_ready": False, "micro_sample_count": float(len(prices))}

    split = max(1, len(returns) // 2)
    early_returns = returns[:split]
    late_returns = returns[split:]
    short_anchor = prices[-min(len(prices), 4)]
    full_anchor = prices[0]
    short_return_pct = ((prices[-1] - short_anchor) / max(1, short_anchor)) * 100.0
    full_return_pct = ((prices[-1] - full_anchor) / max(1, full_anchor)) * 100.0
    late_velocity = mean(late_returns) if late_returns else 0.0
    early_velocity = mean(early_returns) if early_returns else 0.0
    acceleration = late_velocity - early_velocity
    up_ratio = sum(1 for value in returns if value > 0.0) / max(1, len(returns))
    last_return = returns[-1]
    weighted_momentum = sum((idx + 1) * value for idx, value in enumerate(returns)) / max(
        1,
        sum(range(1, len(returns) + 1)),
    )
    avg_abs_return = mean(abs(value) for value in returns)
    return_noise = mean(abs(value - weighted_momentum) for value in returns)
    momentum_clarity = self._clip_float(abs(weighted_momentum) / max(0.05, avg_abs_return), 0.0, 1.0)
    noise_penalty = self._clip_float(return_noise / max(0.12, avg_abs_return * 1.35), 0.0, 1.0)
    down_streak = 0
    for value in reversed(returns):
        if value < 0.0:
            down_streak += 1
        else:
            break
    up_streak = 0
    for value in reversed(returns):
        if value > 0.0:
            up_streak += 1
        else:
            break

    volumes = [max(0, int(getattr(item, "volume", 0) or 0)) for item in window]
    volume_deltas = [
        max(0, current - previous)
        for previous, current in zip(volumes, volumes[1:])
        if current > 0 or previous > 0
    ]
    volume_surge = 1.0
    if len(volume_deltas) >= 2:
        recent_volume = max(0, volume_deltas[-1])
        prior_volume = mean(volume_deltas[:-1]) if volume_deltas[:-1] else 0.0
        volume_surge = recent_volume / max(1.0, prior_volume)

    amounts = [max(0, int(getattr(item, "trade_amount", 0) or 0)) for item in window]
    amount_deltas = [
        max(0, current - previous)
        for previous, current in zip(amounts, amounts[1:])
        if current > 0 or previous > 0
    ]
    amount_surge = 1.0
    if len(amount_deltas) >= 2:
        recent_amount = max(0, amount_deltas[-1])
        prior_amount = mean(amount_deltas[:-1]) if amount_deltas[:-1] else 0.0
        amount_surge = recent_amount / max(1.0, prior_amount)
    flow_surge = max(volume_surge, amount_surge)

    recent_high = max(prices)
    recent_low = min(prices)
    recent_high_idx = max(idx for idx, price in enumerate(prices) if price == recent_high)
    recent_low_idx = max(idx for idx, price in enumerate(prices) if price == recent_low)
    high_age_ticks = len(prices) - 1 - recent_high_idx
    low_age_ticks = len(prices) - 1 - recent_low_idx
    recent_drawdown_pct = ((recent_high - prices[-1]) / recent_high * 100.0) if recent_high > 0 else 0.0
    rebound_from_recent_low_pct = ((prices[-1] - recent_low) / recent_low * 100.0) if recent_low > 0 else 0.0
    day_high = max(0, int(quote.high_price or 0))
    day_low = max(0, int(quote.low_price or 0))
    day_span = max(1, day_high - day_low)
    day_discount_pct = ((day_high - current_price) / day_high * 100.0) if day_high > current_price else 0.0
    day_range_position = (current_price - day_low) / day_span if day_high > 0 and day_low > 0 else 0.5
    leader_vs_open = float(getattr(leader, "vs_open_pct", 0.0) or 0.0) if leader is not None else (
        ((current_price - int(quote.open_price or 0)) / max(1, int(quote.open_price or 0))) * 100.0
        if int(quote.open_price or 0) > 0
        else 0.0
    )
    leader_percentile = float(getattr(leader, "leader_percentile", 0.0) or 0.0) if leader is not None else 0.0
    high_proximity = float(getattr(leader, "high_proximity", 0.0) or 0.0) if leader is not None else day_range_position
    market_p90 = float(self._adaptive_market_state.get("vs_open_p90", 0.0) or 0.0)
    overextension_pct = max(0.0, leader_vs_open - max(6.0, market_p90 * 0.95))
    rejection_pressure = max(0.0, max(day_discount_pct, recent_drawdown_pct) - rebound_from_recent_low_pct * 0.35)
    volume_fade = 1.0 if flow_surge < 0.70 and late_velocity <= 0.0 else 0.0
    failed_breakout_score = (
        0.34 * self._clip_float(recent_drawdown_pct / 0.90, 0.0, 1.0)
        + 0.24 * self._clip_float(down_streak / 3.0, 0.0, 1.0)
        + 0.18 * self._clip_float((-weighted_momentum) / 0.28, 0.0, 1.0)
        + 0.14 * self._clip_float(overextension_pct / 8.0, 0.0, 1.0)
        + 0.10 * self._clip_float((2.0 - high_age_ticks) / 2.0, 0.0, 1.0)
    )
    bounce_confirmation = (
        0.30 * self._clip_float(up_streak / 2.0, 0.0, 1.0)
        + 0.24 * self._clip_float(weighted_momentum / 0.28, 0.0, 1.0)
        + 0.18 * self._clip_float(rebound_from_recent_low_pct / 0.75, 0.0, 1.0)
        + 0.16 * self._clip_float((flow_surge - 0.75) / 1.50, 0.0, 1.0)
        + 0.12 * self._clip_float((2.0 - low_age_ticks) / 2.0, 0.0, 1.0)
    )

    launch_score = (
        0.22 * self._clip_float(short_return_pct / 1.20, 0.0, 1.0)
        + 0.18 * self._clip_float(acceleration / 0.35, 0.0, 1.0)
        + 0.15 * self._clip_float((up_ratio - 0.45) / 0.55, 0.0, 1.0)
        + 0.14 * self._clip_float((flow_surge - 0.80) / 1.80, 0.0, 1.0)
        + 0.10 * self._clip_float(rebound_from_recent_low_pct / 1.40, 0.0, 1.0)
        + 0.11 * self._clip_float(leader_percentile, 0.0, 1.0)
        + 0.06 * self._clip_float(day_range_position, 0.0, 1.0)
        + 0.04 * bounce_confirmation
    )
    downside_score = (
        0.24 * self._clip_float((-short_return_pct) / 0.80, 0.0, 1.0)
        + 0.20 * self._clip_float((-acceleration) / 0.35, 0.0, 1.0)
        + 0.15 * self._clip_float(rejection_pressure / 1.20, 0.0, 1.0)
        + 0.14 * self._clip_float((0.55 - up_ratio) / 0.55, 0.0, 1.0)
        + 0.10 * self._clip_float(overextension_pct / 8.0, 0.0, 1.0)
        + 0.08 * volume_fade
        + 0.04 * self._clip_float(high_proximity - 0.98, 0.0, 1.0)
        + 0.05 * failed_breakout_score
    )
    positive_consistency = up_ratio if weighted_momentum >= 0.0 else 0.0
    trend_quality = self._clip_float(
        0.55 * positive_consistency
        + 0.25 * momentum_clarity
        + 0.20 * self._clip_float(weighted_momentum / 0.28, 0.0, 1.0)
        - 0.30 * noise_penalty,
        0.0,
        1.0,
    )
    direction_margin = self._clip_float((launch_score - downside_score + 0.25) / 0.75, 0.0, 1.0)
    flow_confirmation = self._clip_float((flow_surge - 0.75) / 1.50, 0.0, 1.0)
    signal_conflict_score = self._clip_float(
        min(launch_score, downside_score) * 1.35 - abs(launch_score - downside_score),
        0.0,
        1.0,
    )
    precision_score = self._clip_float(
        0.28 * trend_quality
        + 0.22 * direction_margin
        + 0.18 * bounce_confirmation
        + 0.14 * flow_confirmation
        + 0.10 * self._clip_float(leader_percentile, 0.0, 1.0)
        + 0.08 * self._clip_float(float(getattr(leader, "recent_acceleration_pct", 0.0) or 0.0) / 0.35, 0.0, 1.0)
        - 0.20 * failed_breakout_score
        - 0.16 * signal_conflict_score
        - 0.10 * volume_fade,
        0.0,
        1.0,
    )
    net_score = self._clip_float(launch_score - downside_score, -1.0, 1.0)
    fast_launch = (
        launch_score >= 0.58
        and net_score >= 0.18
        and late_velocity > 0.0
        and up_ratio >= 0.60
    )
    reversal_risk = (
        downside_score >= 0.64
        and net_score <= -0.12
        and (last_return <= 0.0 or late_velocity <= 0.0)
    )
    return {
        "micro_ready": True,
        "micro_sample_count": float(len(prices)),
        "micro_short_return_pct": round(short_return_pct, 6),
        "micro_full_return_pct": round(full_return_pct, 6),
        "micro_velocity_pct": round(late_velocity, 6),
        "micro_acceleration_pct": round(acceleration, 6),
        "micro_up_tick_ratio": round(up_ratio, 6),
        "micro_last_return_pct": round(last_return, 6),
        "micro_weighted_momentum_pct": round(weighted_momentum, 6),
        "micro_momentum_clarity": round(momentum_clarity, 6),
        "micro_noise_penalty": round(noise_penalty, 6),
        "micro_down_streak": float(down_streak),
        "micro_up_streak": float(up_streak),
        "micro_flow_surge": round(flow_surge, 6),
        "micro_recent_drawdown_pct": round(recent_drawdown_pct, 6),
        "micro_rebound_from_low_pct": round(rebound_from_recent_low_pct, 6),
        "micro_rejection_pressure": round(rejection_pressure, 6),
        "micro_overextension_pct": round(overextension_pct, 6),
        "micro_failed_breakout_score": round(failed_breakout_score, 6),
        "micro_bounce_confirmation": round(bounce_confirmation, 6),
        "micro_trend_quality": round(trend_quality, 6),
        "micro_signal_conflict_score": round(signal_conflict_score, 6),
        "micro_precision_score": round(precision_score, 6),
        "micro_launch_score": round(launch_score, 6),
        "micro_downside_score": round(downside_score, 6),
        "micro_net_score": round(net_score, 6),
        "micro_fast_launch": 1.0 if fast_launch else 0.0,
        "micro_reversal_risk": 1.0 if reversal_risk else 0.0,
    }
