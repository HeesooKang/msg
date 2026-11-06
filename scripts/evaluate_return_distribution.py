"""Full-watchlist, chronological policy-outcome research. No broker or state writes.

Entry at the observed ask is conditional on a fill, NOT a simulated portfolio.
Exit uses the first observed executable bid AFTER the frozen policy's trigger.
Unknown books/gaps are censored, never silently labelled as zero-return trades.
Install optional dependencies with: venv/bin/pip install -r requirements-research.txt
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import asdict
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
from math import sqrt

import numpy as np

from scripts.replay_execution_policy import batches
from scripts.return_distribution import ExecutionReturnDistribution, ReturnDistribution
from src.instruments import tick_size, transaction_tax_rate
from src.models import Quote
from src.strategies.momentum_scalp_pnl import estimate_trade_net_pnl_unrounded


FEATURE_NAMES = ("return_15s", "return_30s", "return_60s", "acceleration",
                 "volatility", "drawdown", "spread", "imbalance", "mean_imbalance",
                 "imbalance_change", "target_return", "stop_return", "exit_allowance",
                 "breakout_distance", "sell_tax", "trade_imbalance_15s",
                 "trade_imbalance_60s", "turnover_acceleration", "flow_observed")


def executable(q: Quote) -> bool:
    return (q.has_valid_prices and q.book_available and q.book_depth_available
            and 0 < q.bid_price <= q.ask_price and min(q.bid_size, q.ask_size) > 0)


def observe(history: list[Quote], q: Quote, *, commission: float, tax: float) -> dict | None:
    cutoff = q.timestamp - timedelta(seconds=60)
    anchors = [p for p in history if p.timestamp <= cutoff]
    if not executable(q) or not anchors or (cutoff - anchors[-1].timestamp).total_seconds() > 5:
        return None
    window = [p for p in history if anchors[-1].timestamp <= p.timestamp < q.timestamp]
    path = window + [q]
    if any(not executable(p) for p in path) or any(
            not 0 < (b.timestamp - a.timestamp).total_seconds() <= 5
            for a, b in zip(path, path[1:])):
        return None
    seconds = np.array([(b.timestamp - max(cutoff, a.timestamp)).total_seconds()
                        for a, b in zip(path, path[1:])])
    bids = np.array([p.bid_price for p in window], dtype=float)
    imbalances = np.array([(p.bid_size - p.ask_size) / (p.bid_size + p.ask_size)
                          for p in window])
    mean_bid = float(np.average(bids, weights=seconds))
    target = 2 * window[-1].bid_price - mean_bid
    target = max(0, int(target // tick_size(target, q.security_group) * tick_size(target, q.security_group)))
    drawdown = float(np.max(np.maximum.accumulate(bids) - bids))
    stop = q.bid_price - max(drawdown, tick_size(q.bid_price, q.security_group))
    stop_tick = tick_size(stop, q.security_group)
    stop = max(stop_tick, int(stop // stop_tick * stop_tick))
    allowance = max(stop_tick, max(p.ask_price - p.bid_price for p in path),
                    max(a.bid_price - b.bid_price for a, b in zip(path, path[1:])))
    returns = []
    for age in (15, 30, 60):
        anchor = [p for p in window if p.timestamp <= q.timestamp - timedelta(seconds=age)][-1]
        returns.append(q.bid_price / anchor.bid_price - 1)
    steps = np.diff(np.log([p.bid_price for p in path]))
    imbalance = (q.bid_size - q.ask_size) / (q.bid_size + q.ask_size)
    mean_imbalance = float(np.average(imbalances, weights=seconds))
    level = max(max(p.current_price, p.trade_high_price, p.ask_high_price, p.ask_price) for p in window)
    target_fill = max(1, target - allowance)
    target_fill = max(tick_size(target_fill, q.security_group),
                      target_fill // tick_size(target_fill, q.security_group) * tick_size(target_fill, q.security_group))
    target_net = estimate_trade_net_pnl_unrounded(entry_price=q.ask_price,
        exit_price=target_fill, quantity=1, commission_rate=commission, sell_tax_rate=tax)
    flow = []
    amounts = []
    for age in (15, 60):
        anchor = [p for p in window if p.timestamp <= q.timestamp - timedelta(seconds=age)][-1]
        buys = q.cumulative_buy_volume - anchor.cumulative_buy_volume
        sells = q.cumulative_sell_volume - anchor.cumulative_sell_volume
        amount = q.cumulative_trade_amount - anchor.cumulative_trade_amount
        observed = [p for p in path if p.timestamp >= anchor.timestamp and p.flow_available]
        valid = (q.flow_available and anchor.flow_available and min(buys, sells, amount) >= 0
                 and all(b.cumulative_buy_volume >= a.cumulative_buy_volume
                         and b.cumulative_sell_volume >= a.cumulative_sell_volume
                         and b.cumulative_trade_amount >= a.cumulative_trade_amount
                         for a, b in zip(observed, observed[1:])))
        if not valid:
            flow = []
            break
        flow.append((buys-sells)/(buys+sells) if buys+sells else 0.)
        amounts.append(amount / (q.timestamp-anchor.timestamp).total_seconds())
    flow_features = [*flow, amounts[0]/amounts[1]-1 if amounts[1] else 0., 1.] if flow else [0.] * 4
    return dict(symbol=q.symbol, day=q.timestamp.date().isoformat(),
        signal_at=q.timestamp.isoformat(), entry=q.ask_price, target=target, stop=stop,
        commission=commission, tax=tax, target_net_return=target_net / q.ask_price,
        price_scale=max(sqrt(float(steps @ steps)), tick_size(q.bid_price, q.security_group)/q.bid_price),
        exit_allowance=allowance,
        breakout=q.is_price_breakout(level),
        features=[*returns, returns[0] * 2 - returns[1], sqrt(float(steps @ steps)),
                  drawdown / q.ask_price, (q.ask_price-q.bid_price) / q.ask_price,
                  imbalance, mean_imbalance, imbalance-mean_imbalance,
                  target / q.ask_price - 1, stop / q.ask_price - 1,
                  allowance / q.ask_price, q.bid_price / level - 1, tax, *flow_features])


def collect(root: Path, day: str, groups: dict) -> tuple[list[dict], dict]:
    histories, pending, next_sample = defaultdict(deque), {}, {}
    rows, reasons = [], Counter()
    path = root / day[:4] / day[5:7] / f"quote-batches.{day}.jsonl.gz"
    snapshot = path.read_bytes()

    def unknown(symbol, reason):
        item = pending.pop(symbol)
        row = item["row"]
        if "exit_at" not in row:
            row.update(status="unknown", reason=reason)
        row.update(terminal_status="unknown", terminal_reason=reason)
        rows.append(row)
        reasons[reason] += 1

    for batch, now, quotes in batches(root, day, snapshot=snapshot):
        if batch.get("reset_history"):
            for symbol in list(pending):
                unknown(symbol, "history_reset")
            histories.clear()
        universe = set(batch.get("universe") or [])
        for q in sorted(quotes, key=lambda p: (p.timestamp, p.symbol)):
            q.security_group = q.security_group or groups.get(q.symbol, "")
            history = histories[q.symbol]
            if history and q.timestamp <= history[-1].timestamp:
                if q.timestamp == history[-1].timestamp:
                    q.observe_price_peaks(history.pop())
                    history.append(q)
                continue
            for key in ((q.symbol, "sample"), (q.symbol, "entry")):
                if key not in pending:
                    continue
                item = pending[key]
                if ((q.timestamp - item["last"]).total_seconds() > 5
                    or (now - q.timestamp).total_seconds() > 5 or not executable(q)):
                    unknown(key, "unobserved_execution_path")
                else:
                    row = item["row"]
                    net = estimate_trade_net_pnl_unrounded(entry_price=row["entry"],
                        exit_price=q.bid_price, quantity=1,
                        commission_rate=row["commission"], sell_tax_rate=row["tax"])
                    if "exit_at" not in row:
                        if item["trigger"] is not None:
                            row.update(status="complete", exit_at=q.timestamp.isoformat(),
                                       exit_price=q.bid_price, net_return=net/row["entry"],
                                       exit_reason=item["trigger"])
                        else:
                            item["trigger"] = ("stop" if q.bid_price <= row["stop"] else
                                "target" if q.bid_price >= row["target"] and net > 0 else
                                "expiry" if q.timestamp >= item["expires"] else None)
                    if item["terminal_trigger"]:
                        row.update(terminal_status="complete", terminal_at=q.timestamp.isoformat(),
                                   terminal_net_return=net/row["entry"])
                        rows.append(row)
                        pending.pop(key)
                    elif q.timestamp >= item["expires"]:
                        item["terminal_trigger"] = True
                    item["last"] = q.timestamp
            if (q.symbol in universe
                and (now-q.timestamp).total_seconds() <= 5
                and q.timestamp.hour >= 9 and (q.timestamp.hour, q.timestamp.minute) < (15, 17)
                and q.ask_price * 1.000142 <= 1_000_000):
                keys = [(q.symbol, kind) for kind in ("sample", "entry")
                        if (q.symbol, kind) not in pending
                        and q.timestamp >= next_sample.get((q.symbol, kind), datetime.min)]
                previous_peak = (max(history[-1].current_price, history[-1].ask_price,
                                     history[-1].ask_high_price, history[-1].trade_high_price)
                                 if history else 0)
                keys = [key for key in keys if key[1] == "sample" or q.is_price_breakout(previous_peak)]
                row = observe(list(history), q, commission=.000142,
                              tax=transaction_tax_rate(q.security_group, .0015)) if keys else None
                if row:
                    for key in keys:
                        if key[1] == "entry" and not (row["breakout"] and row["target_net_return"] > 0):
                            continue
                        expires = q.timestamp + timedelta(seconds=180)
                        pending[key] = dict(row=dict(row, cohort=key[1]), last=q.timestamp, expires=expires,
                                            trigger=None, terminal_trigger=False)
                        next_sample[key] = expires + timedelta(seconds=5)
            history.append(q)
            while len(history) > 1 and (q.timestamp-history[1].timestamp).total_seconds() > 60:
                history.popleft()
        for symbol, item in list(pending.items()):
            if (now-item["last"]).total_seconds() > 5:
                unknown(symbol, "quote_gap")
    for symbol in list(pending):
        unknown(symbol, "archive_end")
    return rows, dict(day=day, observations=len(rows), complete=sum(r["status"] == "complete" for r in rows),
                      terminal_complete=sum(r["terminal_status"] == "complete" for r in rows),
                      unknown=dict(reasons), sha256=hashlib.sha256(snapshot).hexdigest())


def evaluate(rows: list[dict], *, price_move=False, training_rows=None) -> list[dict]:
    result = []
    training = rows if training_rows is None else training_rows
    for day in sorted({r["day"] for r in rows}):
        train = [r for r in training if r["day"] < day and r.get("cohort", "sample") == "sample"]
        test = [r for r in rows if r["day"] == day]
        if not any(r["status"] == "complete" for r in train):
            continue
        model = ReturnDistribution(price_move=price_move).fit(train, before=day + "T00:00:00")
        estimates = model.predict([r["features"] for r in test], contexts=test)
        for row, estimate in zip(test, estimates):
            result.append(dict(row, prediction=asdict(estimate),
                               training_through=model.training_through, training_count=model.training_count,
                               baseline_mean=model.baseline_mean, baseline_loss=model.baseline_loss))
        print(json.dumps(dict(day=day, training=model.training_count, evaluated=len(test))), flush=True)
    return result


def summarize(predictions: list[dict]) -> dict:
    def metrics(rows):
        complete = [r for r in rows if r["status"] == "complete"]
        if not complete:
            return dict(observations=len(rows), complete=0)
        actual = np.array([r["net_return"] for r in complete])
        predicted = np.array([r["prediction"]["expected_return"] for r in complete])
        p_loss = np.array([r["prediction"]["loss_probability"] for r in complete])
        baseline = np.array([r["baseline_mean"] for r in complete])
        baseline_loss = np.array([r["baseline_loss"] for r in complete])
        return dict(observations=len(rows), complete=len(complete), unknown=len(rows)-len(complete),
            actual_mean_bps=float(actual.mean()*10000), predicted_mean_bps=float(predicted.mean()*10000),
            loss_rate=float((actual < 0).mean()), mean_predicted_loss=float(p_loss.mean()),
            brier=float(np.mean((p_loss-(actual < 0))**2)),
            baseline_brier=float(np.mean((baseline_loss-(actual < 0))**2)),
            mean_squared_error=float(np.mean((predicted-actual)**2)),
            baseline_mean_squared_error=float(np.mean((baseline-actual)**2)))
    return dict(all=metrics(predictions),
        positive_ev=metrics([r for r in predictions if r["prediction"]["expected_return"] > 0]),
        existing_entry=metrics([r for r in predictions if r.get("cohort") == "entry"]),
        positive_ev_entry=metrics([r for r in predictions if r.get("cohort") == "entry"
                                  and r["prediction"]["expected_return"] > 0]),
        all_by_day={day: metrics([r for r in predictions if r["day"] == day])
                    for day in sorted({r["day"] for r in predictions})},
        by_day={day: metrics([r for r in predictions if r["day"] == day
                              and r["prediction"]["expected_return"] > 0])
                for day in sorted({r["day"] for r in predictions})})


def summarize_execution_predictions(items: list[dict]) -> dict:
    complete = [r for r in items if r["status"] == "complete"]
    filled = [r for r in complete if r["filled"]]
    actual = [r["quantity"]*r["entry_limit"]*r["net_return"] for r in complete]
    wins, losses = [p for p in actual if p > 0], [p for p in actual if p < 0]
    errors = [(r["prediction"]["loss_probability_if_filled"] - (r["net_return"] < 0))**2
              for r in filled if r["prediction"]["loss_probability_if_filled"] is not None]
    return dict(observations=len(items), complete=len(complete), unknown=len(items)-len(complete),
        filled=len(filled), wins=len(wins), losses=len(losses),
        average_win_net=float(np.mean(wins)) if wins else None,
        average_loss_net=float(np.mean(losses)) if losses else None,
        expected_net_sum=sum(r["expected_net"] for r in complete),
        independent_trade_net_sum=sum(actual),
        mean_net_overprediction=float(np.mean([r["expected_net"]-p for r, p in zip(complete, actual)]))
            if complete else None,
        loss_rate_filled=float(np.mean([r["net_return"] < 0 for r in filled])) if filled else None,
        loss_brier_filled=float(np.mean(errors)) if errors else None)


def evaluate_execution(rows: list[dict], groups: dict) -> dict:
    """Walk forward without increasing limit prices or inventing missing fills."""
    predictions = []
    for day in sorted({r["day"] for r in rows})[1:]:
        model = ExecutionReturnDistribution().fit(rows, before=day + "T00:00:00")
        # A sample and an entry probe at the same instant are one opportunity.
        test = {}
        for row in rows:
            if row["day"] == day and row["limit_ticks"] == 0:
                key = row["symbol"], row["signal_at"]
                if key not in test or row["cohort"] == "entry":
                    test[key] = row
        items = list(test.values())
        for row, estimate in zip(items, model.predict(items)):
            group = groups.get(row["symbol"], "")
            tick = tick_size(row["stop"], group)
            stop_fill = max(tick, int((row["stop"]-row["exit_allowance"]) // tick * tick))
            risk = max(0., -estimate_trade_net_pnl_unrounded(entry_price=row["entry_limit"],
                exit_price=stop_fill, quantity=1, commission_rate=row["commission"], sell_tax_rate=row["tax"]))
            capital = int(1_000_000 // (row["entry_limit"] * (1+row["commission"])))
            quantity = min(capital, int(5000 / risk)) if risk > 0 else capital
            predictions.append(dict(row, prediction=asdict(estimate), quantity=quantity,
                expected_net=quantity*row["entry_limit"]*estimate.expected_return,
                training_through=model.training_through, training_count=model.training_count))
        print(json.dumps(dict(day=day, training=model.training_count, evaluated=len(items))), flush=True)

    selected = [r for r in predictions if r["expected_net"] > 0 and r["quantity"] > 0]
    return dict(predictions=predictions, selected=selected, summary=summarize_execution_predictions(selected),
        all_summary=summarize_execution_predictions(predictions),
        by_day={day: summarize_execution_predictions([r for r in selected if r["day"] == day])
                for day in sorted({r["day"] for r in predictions})},
        decision="not_deployed", limitations=[
            "next_snapshot_execution_not_broker_latency_or_queue",
            "unknown_outcomes_excluded_not_zero; possible_informative_censoring",
            "independent_1m_capital_5000_risk_per_observation_not_a_portfolio",
            "no_overlap_capital_or_daily_stop_simulation",
            "reused_dates_exploratory_not_untouched_holdout",
            "no_runtime_deployment"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("reports/market-quotes"))
    parser.add_argument("--date", action="append", default=[])
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--policy", choices=("planned", "expiry", "execution"), default="planned")
    parser.add_argument("--price-move", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.policy == "execution" and args.dataset is None:
        parser.error("execution requires a next-snapshot execution dataset")
    if args.dataset:
        dataset = json.loads(args.dataset.read_text())
        rows, inputs = dataset["observations"], dataset["inputs"]
    else:
        cache = Path("state/instruments.json")
        groups = json.loads(cache.read_text()).get("groups", {}) if cache.exists() else {}
        rows, inputs = [], []
        for day in sorted(set(args.date)):
            observations, summary = collect(args.root, day, groups)
            rows.extend(observations)
            inputs.append(summary)
            print(json.dumps(summary), flush=True)
        args.output.with_suffix(".dataset.json").write_text(json.dumps(
            dict(features=FEATURE_NAMES, inputs=inputs, observations=rows)))
    if args.policy == "execution":
        if not all("entry_limit" in r and "limit_ticks" in r for r in rows):
            parser.error("dataset does not contain limit-execution outcomes")
        cache = Path("state/instruments.json")
        groups = json.loads(cache.read_text()).get("groups", {}) if cache.exists() else {}
        sources = (Path(__file__), Path("scripts/return_distribution.py"))
        source_text = {str(p): p.read_text() for p in sources}
        result = evaluate_execution(rows, groups)
        result.update(inputs=inputs, source_dataset=str(args.dataset),
            dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
            source_text=source_text,
            source_sha256={name: hashlib.sha256(text.encode()).hexdigest() for name, text in source_text.items()},
            source_changed_during_run=any(p.read_text() != source_text[str(p)] for p in sources))
        args.output.write_text(json.dumps(result))
        return
    if args.policy == "expiry":
        terminal_rows = []
        for row in rows:
            item = {k: v for k, v in row.items()
                    if k not in {"exit_at", "exit_price", "net_return", "exit_reason"}}
            item["status"] = row["terminal_status"]
            if item["status"] == "complete":
                item.update(exit_at=row["terminal_at"], net_return=row["terminal_net_return"])
            terminal_rows.append(item)
        rows = terminal_rows
    predictions = evaluate(rows, price_move=args.price_move)
    result = dict(features=FEATURE_NAMES, inputs=inputs, predictions=predictions,
                  policy=args.policy,
                  price_move=args.price_move,
                  summary=summarize(predictions),
                  decision="not_deployed",
                  sources=["https://jmlr.org/papers/volume7/meinshausen06a/meinshausen06a.pdf",
                           "https://arxiv.org/abs/1512.03492",
                           "https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf"],
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (Path(__file__), Path("scripts/return_distribution.py"))},
                  limitations=["conditional_ask_fill_not_portfolio",
                               "unknown_paths_censored_and_reported",
                               "loss_probability_is_conditional_on_observable_outcomes",
                               "fixed_time_samples_are_not_all_breakout_signals",
                               "sizing_liquidity_daily_limits_not_simulated",
                               "weighted_forest_adaptation_not_a_profitability_theorem",
                               "reused_dates_not_untouched_holdout",
                               "no_runtime_deployment"])
    args.output.write_text(json.dumps(result))


if __name__ == "__main__":
    main()
