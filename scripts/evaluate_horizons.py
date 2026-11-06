"""Compare fixed forecast horizons on the same past-only, nonoverlapping signals.

These are ask-to-endpoint-bid forecasts, NOT trade/stop/fill simulations. A gap
inside a horizon need not erase an observed endpoint, but prevents path claims.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path

from scripts.evaluate_return_distribution import evaluate, executable, summarize
from scripts.replay_execution_policy import batches
from src.analytics.execution_policy import limit_fill
from src.strategies.momentum_scalp_pnl import estimate_trade_net_pnl_unrounded


def select_signals(rows, horizons):
    selected, next_at = [], {}
    separation = timedelta(seconds=max(horizons) + 5)
    for row in sorted(rows, key=lambda r: (r["signal_at"], r["symbol"])):
        if row.get("cohort") != "sample":
            continue
        at = datetime.fromisoformat(row["signal_at"])
        if at + separation >= at.replace(hour=15, minute=20, second=0, microsecond=0):
            continue
        if at < next_at.get(row["symbol"], datetime.min):
            continue
        # Never select using future outcome availability or profitability.
        selected.append({k: v for k, v in row.items() if k not in
            {"status", "reason", "exit_at", "exit_price", "exit_reason", "net_return"}
            and not k.startswith("terminal_")})
        next_at[row["symbol"]] = at + separation
    return selected


def label_endpoints(stream, signals, horizons):
    starts = {(r["symbol"], r["signal_at"]): r for r in signals}
    pending, result, matched = defaultdict(list), [], set()

    def finish(item, quote=None, reason="endpoint_missing"):
        row = dict(item["row"], horizon_seconds=item["horizon"],
                   continuous_path=item["continuous"], maximum_gap_seconds=item["gap"])
        if quote is None:
            row.update(status="unknown", reason=reason)
        else:
            net = estimate_trade_net_pnl_unrounded(entry_price=row["entry"], exit_price=quote.bid_price,
                quantity=1, commission_rate=row["commission"], sell_tax_rate=row["tax"])
            row.update(status="complete", exit_at=quote.timestamp.isoformat(), exit_price=quote.bid_price,
                       net_return=net/row["entry"], exit_reason="endpoint",
                       endpoint_delay_seconds=(quote.timestamp-item["due"]).total_seconds())
        result.append(row)

    for batch, now, quotes in stream:
        if batch.get("reset_history"):
            for entries in pending.values():
                for item in entries:
                    item["continuous"] = False
        for quote in sorted(quotes, key=lambda q: (q.timestamp, q.symbol)):
            entries = pending[quote.symbol]
            for item in list(entries):
                if quote.timestamp <= item["last"]:
                    continue
                gap = (quote.timestamp-item["last"]).total_seconds()
                item["gap"] = max(item["gap"], gap)
                item["continuous"] &= gap <= 5 and executable(quote) and 0 <= (now-quote.timestamp).total_seconds() <= 5
                item["last"] = quote.timestamp
                if (item["due"] <= quote.timestamp <= now <= item["due"]+timedelta(seconds=5)
                    and executable(quote)):
                    finish(item, quote)
                    entries.remove(item)
            key = quote.symbol, quote.timestamp.isoformat()
            if key not in matched and key in starts:
                row = starts[key]
                bid = round(row["entry"]*(1-row["features"][6]))
                if quote.ask_price != row["entry"] or quote.bid_price != bid or not executable(quote):
                    continue
                matched.add(key)
                for horizon in horizons:
                    entries.append(dict(row=row, horizon=horizon, due=quote.timestamp+timedelta(seconds=horizon),
                                        last=quote.timestamp, gap=0., continuous=True))
        for symbol, entries in list(pending.items()):
            for item in list(entries):
                if now > item["due"]+timedelta(seconds=5):
                    finish(item)
                    entries.remove(item)
            if not entries:
                del pending[symbol]
    for entries in pending.values():
        for item in entries:
            finish(item, reason="archive_end")
    for key in starts.keys()-matched:
        for horizon in horizons:
            result.append(dict(starts[key], horizon_seconds=horizon, status="unknown",
                               continuous_path=False, reason="signal_not_found"))
    return result


def label_limit_entries(stream, endpoints):
    """Compare anchored limits with the first post-submission executable ask.

    This is not a resting-order queue model. Non-crossing quotes mean no fill
    under this one-observation policy, not a known historical broker outcome.
    The original endpoint stays fixed even when the entry price is cheaper.
    """
    starts = {(r["symbol"], r["signal_at"]): r for r in endpoints}
    if len(starts) != len(endpoints):
        raise ValueError("duplicate_entry_signals")
    pending, matched, result = defaultdict(list), set(), []

    def finish(original, now=None, quote=None, reason="entry_quote_missing"):
        bid = round(original["entry"] * (1-original["features"][6]))
        for action, limit in (("ask", original["entry"]), ("bid", bid),
                              ("next_ask", quote.ask_price if quote is not None else original["entry"])):
            row = {k: v for k, v in original.items() if k not in
                   {"status", "reason", "exit_at", "exit_price", "exit_reason", "net_return"}}
            row.update(action=action, entry_limit=limit, status="unknown", reason=reason)
            if quote is not None:
                quantity, price = limit_fill(quote, limit, 1)
                row.update(filled=bool(quantity), entry_checked_at=now.isoformat())
                if not quantity:
                    row.update(status="complete", reason="limit_not_crossed", net_return=0.,
                               exit_at=now.isoformat())
                else:
                    row.update(fill_price=price, fill_at=quote.timestamp.isoformat())
                    if original["status"] != "complete":
                        row["reason"] = "endpoint_missing"
                    else:
                        if datetime.fromisoformat(original["exit_at"]) <= now:
                            raise ValueError("endpoint_precedes_entry")
                        net = estimate_trade_net_pnl_unrounded(entry_price=price,
                            exit_price=original["exit_price"], quantity=1,
                            commission_rate=original["commission"], sell_tax_rate=original["tax"])
                        row.update(status="complete", reason="endpoint", exit_at=original["exit_at"],
                                   exit_price=original["exit_price"], net_return=net/limit)
            result.append(row)

    for batch, now, quotes in stream:
        if batch.get("reset_history"):
            for entries in pending.values():
                for item in entries:
                    finish(item["row"], reason="history_reset")
            pending.clear()
        for quote in sorted(quotes, key=lambda q: (q.timestamp, q.symbol)):
            for item in list(pending.get(quote.symbol, [])):
                if quote.timestamp <= item["observed_at"]:
                    continue
                fresh = quote.timestamp <= now < item["deadline"] and executable(quote)
                finish(item["row"], now, quote if fresh else None,
                       reason="entry_quote_invalid_or_late")
                pending[quote.symbol].remove(item)
            key = quote.symbol, quote.timestamp.isoformat()
            if key not in matched and key in starts:
                row = starts[key]
                bid = round(row["entry"] * (1-row["features"][6]))
                if quote.ask_price != row["entry"] or quote.bid_price != bid or not executable(quote):
                    continue
                matched.add(key)
                pending[quote.symbol].append(dict(row=row, observed_at=now,
                    deadline=quote.timestamp+timedelta(seconds=5)))
        for symbol, entries in list(pending.items()):
            for item in list(entries):
                if now >= item["deadline"]:
                    finish(item["row"])
                    entries.remove(item)
            if not entries:
                del pending[symbol]
    for entries in pending.values():
        for item in entries:
            finish(item["row"], reason="archive_end")
    for key in starts.keys()-matched:
        finish(starts[key], reason="signal_not_found")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--input-snapshots", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("reports/market-quotes"))
    parser.add_argument("--horizon", type=int, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    horizons = sorted(set(args.horizon or (180, 900, 1800)))
    if min(horizons) <= 0:
        parser.error("horizons must be positive")
    dataset = json.loads(args.dataset.read_text())
    fixed = {row["day"]: row for row in json.loads(args.input_snapshots.read_text())["inputs"]}
    sources = [Path(__file__), Path("scripts/evaluate_return_distribution.py"),
               Path("scripts/return_distribution.py"), Path("scripts/replay_execution_policy.py"),
               Path("src/strategies/momentum_scalp_pnl.py")]
    source_text = {str(p): p.read_text() for p in sources}
    rows, inputs = [], []
    for day in sorted({r["day"] for r in dataset["observations"]}):
        signals = select_signals([r for r in dataset["observations"] if r["day"] == day], horizons)
        path = args.root / day[:4] / day[5:7] / f"quote-batches.{day}.jsonl.gz"
        snapshot = path.read_bytes()[:fixed[day]["input_bytes"]]
        digest = hashlib.sha256(snapshot).hexdigest()
        if digest != fixed[day]["input_sha256"]:
            raise ValueError(f"input_snapshot_mismatch: {day}")
        labelled = label_endpoints(batches(args.root, day, snapshot=snapshot), signals, horizons)
        rows.extend(labelled)
        summary = dict(day=day, signals=len(signals), observations=len(labelled), input_sha256=digest,
            input_bytes=len(snapshot), statuses=dict(Counter(r["status"] for r in labelled)))
        inputs.append(summary)
        print(json.dumps(summary), flush=True)
    args.output.with_suffix(".dataset.json").write_text(json.dumps(dict(observations=rows, inputs=inputs)))
    results = []
    for horizon in horizons:
        predictions = evaluate([r for r in rows if r["horizon_seconds"] == horizon], price_move=True)
        summary = summarize(predictions)
        results.append(dict(horizon_seconds=horizon, summary=summary, predictions=predictions))
        print(json.dumps(dict(horizon=horizon, summary=summary["positive_ev"])), flush=True)
    args.output.write_text(json.dumps(dict(results=results, inputs=inputs, horizons=horizons,
        source_text=source_text, source_sha256={name: hashlib.sha256(text.encode()).hexdigest()
                                               for name, text in source_text.items()},
        source_changed_during_run=any(p.read_text()!=source_text[str(p)] for p in sources),
        decision="not_deployed", limitations=["conditional_ask_entry_not_limit_fills",
            "endpoint_forecast_not_stop_or_daily_limit_portfolio",
            "intermediate_gaps_flagged_not_interpolated", "same_signals_for_all_horizons",
            "nonoverlapping_by_symbol_at_longest_horizon", "reused_dates_exploratory",
            "all_horizons_reported_no_selective_hiding", "no_runtime_change"])))


if __name__ == "__main__":
    main()
