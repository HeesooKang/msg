from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple


DAILY_PNL_SNAPSHOT_FIELDS: Tuple[str, ...] = (
    "realized_gross_pnl",
    "realized_net_pnl",
    "fees_paid",
    "taxes_paid",
    "trade_count",
    "win_count",
    "loss_count",
    "breakeven_count",
    "winning_net_pnl_sum",
    "losing_net_pnl_sum",
    "largest_win_net",
    "largest_loss_net",
)


def empty_daily_pnl_snapshot() -> Dict[str, int]:
    snapshot = {field: 0 for field in DAILY_PNL_SNAPSHOT_FIELDS}
    snapshot["breaker_excluded_realized_net_pnl"] = 0
    return snapshot


def rebuild_daily_pnl_snapshot_from_ledgers(
    *,
    seed_snapshot: Mapping[str, Any],
    sell_fill_ledger: list[Mapping[str, Any]],
    closed_trade_ledger: Mapping[str, Mapping[str, Any]],
) -> Dict[str, int]:
    snapshot = empty_daily_pnl_snapshot()
    snapshot.update(
        {
            field: int((seed_snapshot or {}).get(field, 0) or 0)
            for field in DAILY_PNL_SNAPSHOT_FIELDS
        }
    )
    snapshot["breaker_excluded_realized_net_pnl"] = int(
        (seed_snapshot or {}).get("breaker_excluded_realized_net_pnl", 0) or 0
    )

    for entry in sell_fill_ledger or []:
        gross_pnl = int(entry.get("gross_pnl", 0) or 0)
        net_pnl = int(entry.get("net_pnl", 0) or 0)
        fees = int(entry.get("fees", 0) or 0)
        taxes = int(entry.get("taxes", 0) or 0)
        snapshot["realized_gross_pnl"] += gross_pnl
        snapshot["realized_net_pnl"] += net_pnl
        snapshot["fees_paid"] += fees
        snapshot["taxes_paid"] += taxes
        if not bool(entry.get("counts_for_daily_breaker", True)):
            snapshot["breaker_excluded_realized_net_pnl"] += net_pnl

    for record in (closed_trade_ledger or {}).values():
        net_pnl = int(record.get("net_pnl", 0) or 0)
        snapshot["trade_count"] += 1
        if net_pnl > 0:
            snapshot["win_count"] += 1
            snapshot["winning_net_pnl_sum"] += net_pnl
            snapshot["largest_win_net"] = max(int(snapshot.get("largest_win_net", 0) or 0), net_pnl)
        elif net_pnl < 0:
            snapshot["loss_count"] += 1
            snapshot["losing_net_pnl_sum"] += net_pnl
            snapshot["largest_loss_net"] = min(int(snapshot.get("largest_loss_net", 0) or 0), net_pnl)
        else:
            snapshot["breakeven_count"] += 1

    return snapshot
