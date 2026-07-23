from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExitDecision:
    reason: str = ""
    quantity: int = 0

    @property
    def should_exit(self) -> bool:
        return bool(self.reason and self.quantity > 0)


@dataclass(frozen=True)
class LongExitSnapshot:
    quantity: int
    held_minutes: float
    held_seconds: float
    gain_pct: float
    trail_drawdown_pct: float
    unrealized_pnl: int
    estimated_net_pnl: int
    net_stop_amount: int
    take_profit_pct: float
    adaptive_target: bool
    max_hold_minutes: float
    min_trailing_seconds: float
    trailing_activation_pct: float
    trailing_stop_pct: float
    adaptive_stop_loss_pct: float


def decide_long_exit(
    snapshot: LongExitSnapshot,
    *,
    partial_exit_done: bool,
    partial_exit_ratio: float,
) -> ExitDecision:
    qty = max(0, int(snapshot.quantity or 0))
    if qty <= 0:
        return ExitDecision()

    stop = max(1, int(snapshot.net_stop_amount or 0))
    if int(snapshot.estimated_net_pnl) <= -stop:
        return ExitDecision(reason="protective_stop_net", quantity=qty)
    if int(snapshot.unrealized_pnl) <= -stop:
        return ExitDecision(reason="protective_stop", quantity=qty)

    adaptive_stop = float(snapshot.adaptive_stop_loss_pct or 0.0)
    if adaptive_stop != 0.0 and float(snapshot.gain_pct) <= -abs(adaptive_stop):
        return ExitDecision(reason="adaptive_stop_loss", quantity=qty)

    take_profit = float(snapshot.take_profit_pct or 0.0)
    if (
        not bool(partial_exit_done)
        and not bool(snapshot.adaptive_target)
        and float(snapshot.gain_pct) >= max(0.6, take_profit * 0.6)
        and qty > 1
    ):
        partial_qty = max(1, int(round(qty * float(partial_exit_ratio or 0.0))))
        return ExitDecision(reason="partial_take_profit", quantity=partial_qty)

    trailing_activation = float(snapshot.trailing_activation_pct or 0.0)
    trailing_stop = -abs(float(snapshot.trailing_stop_pct or 0.0))
    if float(snapshot.gain_pct) >= trailing_activation and float(snapshot.trail_drawdown_pct) <= trailing_stop:
        if float(snapshot.min_trailing_seconds or 0.0) > 0.0 and float(snapshot.held_seconds) < float(snapshot.min_trailing_seconds):
            return ExitDecision()
        return ExitDecision(reason="trailing_stop", quantity=qty)

    if float(snapshot.gain_pct) >= take_profit:
        return ExitDecision(reason="take_profit", quantity=qty)
    if float(snapshot.held_minutes) >= float(snapshot.max_hold_minutes):
        if int(snapshot.estimated_net_pnl) <= 0 and float(snapshot.gain_pct) > 0.0:
            return ExitDecision()
        return ExitDecision(reason="time_exit", quantity=qty)
    return ExitDecision()
