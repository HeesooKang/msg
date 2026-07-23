from __future__ import annotations

from typing import Any, Dict

from src.analytics.price_prediction import predict_short_horizon_return
from src.models import Quote
from src.strategies.momentum_scalp_micro import symbol_micro_edge_metrics
from src.strategies.momentum_scalp_types import PositionState




def long_exit_hold_profile(
    strategy: Any,
    pos: PositionState,
    quote: Quote,
    *,
    estimated_net_pnl: int,
    gain_pct: float,
    take_profit_pct: float,
) -> Dict[str, float]:
    """Recalculate long exit hold limits from live micro/prediction evidence."""
    max_hold_config = max(1, int(getattr(strategy.config, "max_position_holding_minutes", 50) or 50))
    planned_hold = int(getattr(pos, "adaptive_max_hold_minutes", 0) or 0)
    if planned_hold <= 0:
        strategy_bonus = 0.05 if str(getattr(pos, "entry_strategy_name", "") or "") == "opening_conviction_long_strategy" else 0.0
        score_hint = strategy._clip_float(float(getattr(pos, "conviction_score", 0.0) or 0.0), 0.0, 1.0)
        planned_hold = int(round(max_hold_config * strategy._clip_float(0.34 + strategy_bonus + 0.18 * score_hint, 0.22, 0.72)))

    leader = strategy._leader_signal_for_quote(quote)
    metrics = symbol_micro_edge_metrics(strategy, quote, leader=leader)
    adaptive = strategy._adaptive_market_entry_thresholds()
    heat = strategy._clip_float(float(adaptive.get("heat", 0.0) or 0.0), 0.0, 1.0)
    caution = strategy._clip_float(float(adaptive.get("caution", 0.0) or 0.0), 0.0, 1.0)
    overheat = strategy._clip_float(float(adaptive.get("overheat", 0.0) or 0.0), 0.0, 1.0)
    micro_ready = bool(metrics.get("micro_ready", False))
    micro_net = float(metrics.get("micro_net_score", 0.0) or 0.0)
    micro_precision = strategy._clip_float(float(metrics.get("micro_precision_score", 0.0) or 0.0), 0.0, 1.0)
    micro_downside = strategy._clip_float(float(metrics.get("micro_downside_score", 0.0) or 0.0), 0.0, 1.0)
    failed_breakout = strategy._clip_float(float(metrics.get("micro_failed_breakout_score", 0.0) or 0.0), 0.0, 1.0)
    signal_conflict = strategy._clip_float(float(metrics.get("micro_signal_conflict_score", 0.0) or 0.0), 0.0, 1.0)
    bounce = strategy._clip_float(float(metrics.get("micro_bounce_confirmation", 0.0) or 0.0), 0.0, 1.0)

    prediction = predict_short_horizon_return(
        quote,
        recent_quotes=list(strategy._recent_quotes.get(quote.symbol, [])),
        leader=leader,
        market_state=dict(getattr(strategy, "_adaptive_market_state", {}) or {}),
        horizon_seconds=int(getattr(strategy.config, "price_prediction_horizon_seconds", 180) or 180),
        min_samples=int(getattr(strategy.config, "price_prediction_min_samples", 5) or 5),
    )
    features = dict(getattr(prediction, "features", {}) or {})
    prediction_risk = max(
        float(features.get("rejection_risk_score", 0.0) or 0.0),
        float(features.get("chase_risk_score", 0.0) or 0.0),
        float(features.get("trap_risk_score", 0.0) or 0.0),
    )
    prediction_quality = (
        strategy._clip_float(
            (
                float(features.get("continuation_quality", 0.0) or 0.0)
                + float(features.get("follow_through_score", 0.0) or 0.0)
                + strategy._clip_float(float(getattr(prediction, "predicted_return_pct", 0.0) or 0.0) / 1.20, 0.0, 1.0)
                + strategy._clip_float(float(getattr(prediction, "lower_bound_return_pct", 0.0) or 0.0) / 0.45, -0.4, 1.0)
                + strategy._clip_float(float(getattr(prediction, "confidence", 0.0) or 0.0), 0.0, 1.0)
                + strategy._clip_float(float(getattr(prediction, "direction_score", 0.0) or 0.0), 0.0, 1.0)
            )
            / 6.0,
            0.0,
            1.0,
        )
        if prediction.ready
        else 0.0
    )
    micro_quality = (
        strategy._clip_float(
            (strategy._clip_float((micro_net + 0.20) / 1.10, 0.0, 1.0) + micro_precision + bounce) / 3.0,
            0.0,
            1.0,
        )
        if micro_ready
        else 0.20
    )
    quality = strategy._clip_float(
        0.48 * micro_quality
        + 0.36 * prediction_quality
        + 0.10 * strategy._clip_float(float(getattr(pos, "leader_percentile", 0.0) or 0.0), 0.0, 1.0)
        + 0.06 * heat,
        0.0,
        1.0,
    )
    risk = strategy._clip_float(
        max(micro_downside, failed_breakout, signal_conflict, prediction_risk) + 0.12 * caution + 0.12 * overheat,
        0.0,
        1.0,
    )
    profit_progress = strategy._clip_float(float(gain_pct) / max(0.10, float(take_profit_pct)), -1.0, 1.0)
    loss_room = max(0, int(strategy._daily_loss_room()))
    daily_loss_budget = abs(
        int(
            strategy.config.daily_total_loss_limit
            if strategy.config.daily_total_loss_limit is not None
            else strategy.config.daily_loss_limit
        )
    )
    loss_pressure = 1.0 - strategy._clip_float(float(loss_room) / max(1.0, float(daily_loss_budget)), 0.0, 1.0)
    extension = int(
        round(
            max_hold_config
            * (
                0.16 * quality
                - 0.20 * risk
                + 0.06 * heat
                - 0.08 * caution
                + 0.08 * max(0.0, profit_progress)
                - 0.08 * loss_pressure
            )
        )
    )
    dynamic_max_hold = max(1, min(max_hold_config, planned_hold + extension))
    if estimated_net_pnl <= 0 and risk > quality:
        dynamic_max_hold = min(dynamic_max_hold, max(1, planned_hold))

    remaining_minutes = max(1, strategy._market_close_minutes_since_open() - strategy._minutes_since_market_open())
    if remaining_minutes <= 3:
        dynamic_max_hold = min(dynamic_max_hold, remaining_minutes)

    remaining_target = strategy._daily_target_remaining_net()
    strong_profit_lock = int(estimated_net_pnl) >= max(1_800, int(remaining_target * 0.35)) or gain_pct >= max(
        float(take_profit_pct) * 0.72,
        float(take_profit_pct) - 0.25,
    )
    urgent_risk_lock = risk >= 0.76 and quality <= 0.38
    min_trailing_seconds = 0.0
    if not strong_profit_lock and not urgent_risk_lock:
        min_trailing_seconds = strategy._clip_float(
            90.0 + 80.0 * quality - 55.0 * risk - 45.0 * max(0.0, profit_progress),
            30.0,
            180.0,
        )

    return {
        "max_hold_minutes": float(dynamic_max_hold),
        "min_trailing_seconds": float(min_trailing_seconds),
        "quality": float(quality),
        "risk": float(risk),
    }
