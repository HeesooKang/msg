"""Compare conditional-return entries with the current bot, offline only.

Both portfolios share fills and daily controls. The optional endpoint experiment
uses expiry-only exits and never reduces the existing planning risk estimate.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import timedelta
import hashlib
import json
import logging
from math import ceil
from pathlib import Path

from scripts.evaluate_return_distribution import executable, observe
from scripts.replay_execution_policy import ReplayStrategy, replay_day
from scripts.return_distribution import ReturnDistribution
from src.models import OrderSide
from src.strategies.momentum_scalp import MAX_HOLD_SECONDS
from src.strategies.momentum_scalp_types import TradePlan


def limit_entry_rejection(order, quote):
    if quote is None or quote.symbol != order.symbol or not executable(quote):
        return "no_executable_book"
    if quote.ask_price > order.price:
        return "ask_above_planned_limit"
    return None


def candidate_strategy(model, *, expiry_only=False):
    class ConditionalReturnStrategy(ReplayStrategy):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._estimates = {}
            self._batch_predictions_ready = False
            self.diagnostics = dict(selected_forecasts=[], prediction_rows=0)

        def on_batch_tick(self, quotes):
            self._batch_predictions_ready = False
            orders = super().on_batch_tick(quotes)
            for order in orders:
                if order.side == OrderSide.BUY:
                    row, prediction, _ = self._estimates[order.symbol]
                    self.diagnostics["selected_forecasts"].append(dict(
                        symbol=order.symbol, signal_at=row["signal_at"], submitted_at=self._now().isoformat(),
                        quantity=order.quantity, entry_limit=order.price,
                        expected_net=prediction.expected_return * order.price * order.quantity,
                        prediction=asdict(prediction), training_through=model.training_through))
                    order.requested_reason = "conditional_return"
            return orders

        def _prepare_predictions(self):
            missing = []
            current = set()
            for quote in self._fresh_quotes():
                if quote.symbol in self.positions:
                    continue
                if not executable(quote) or quote.ask_price * (1+self.config.commission_rate) > self._available_capital():
                    continue
                row = observe(list(self._recent_window(quote.symbol)), quote,
                              commission=self.config.commission_rate,
                              tax=self._sell_tax_rate_for_symbol(quote.symbol))
                if row is None:
                    continue
                key = (row["signal_at"], row["entry"], tuple(row["features"]))
                current.add(quote.symbol)
                cached = self._estimates.get(quote.symbol)
                if cached is None or cached[2] != key:
                    missing.append((row, key))
            self._estimates = {symbol: item for symbol, item in self._estimates.items() if symbol in current}
            rows = [item[0] for item in missing]
            predictions = model.predict([r["features"] for r in rows], contexts=rows)
            self.diagnostics["prediction_rows"] += len(rows)
            for (row, key), prediction in zip(missing, predictions):
                self._estimates[row["symbol"]] = (row, prediction, key)
            self._batch_predictions_ready = True

        def _build_trade_plan(self, quote, *, now):
            if not self._batch_predictions_ready:
                self._prepare_predictions()
            plan = TradePlan(quote.symbol, False, "price_window_incomplete", 0, quote.ask_price,
                0, 0, quote.timestamp, quote.timestamp + timedelta(seconds=MAX_HOLD_SECONDS))
            cached = self._estimates.get(quote.symbol)
            if cached is None:
                return plan
            row, prediction, _ = cached
            plan = replace(plan, reason="non_positive_ev", target_price=row["target"],
                           stop_price=row["stop"], exit_price_allowance=row["exit_allowance"],
                           breakout_price=round(quote.bid_price / (1+row["features"][13])))
            if prediction.expected_return <= 0:
                return plan
            risk = self._risk_per_share(quote, quote.ask_price, plan.stop_price,
                                       exit_allowance=plan.exit_price_allowance)
            if expiry_only:
                risk = max(risk, max(0., -prediction.lower_return)*quote.ask_price
                           + plan.exit_price_allowance*(1-self.config.commission_rate
                                                         -self._sell_tax_rate_for_symbol(quote.symbol)))
            capital_quantity = int(self._available_capital() // (quote.ask_price * (1+self.config.commission_rate)))
            room = self._remaining_daily_loss_room()
            quantity = min(capital_quantity, int(room/risk)) if risk > 0 else capital_quantity
            committed = ceil(risk*quantity)
            if quantity <= 0 or committed > room:
                return replace(plan, reason="insufficient_capital_or_loss_room")
            return replace(plan, allowed=True, reason="conditional_return", quantity=quantity,
                           committed_risk_net_abs=committed)

        def _entry_plan_priority(self, plan):
            prediction = self._estimates[plan.symbol][1]
            return (-prediction.expected_return * plan.entry_limit_price * plan.quantity,
                    plan.committed_risk_net_abs, plan.symbol)

        def _position_exit_order(self, position, quote, *, now, observed_quotes=()):
            plan = position.plan
            if not expiry_only or plan is None or not plan.allowed or plan.breakout_price <= 0:
                return super()._position_exit_order(position, quote, now=now,
                                                    observed_quotes=observed_quotes)
            if position.pending_exit_started_at is not None:
                return None
            if now >= plan.expires_at:
                return self._make_market_sell(position, quote, reason="holding_expiry")
            return None

    return ConditionalReturnStrategy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("reports/market-quotes"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--date", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", choices=("planned", "expiry"), default="planned")
    parser.add_argument("--input-snapshots", type=Path)
    parser.add_argument("--observed-holds", action="store_true",
                        help="replay recorded observations; do not assume missing intrahold prices")
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    sources = (Path(__file__), Path("scripts/return_distribution.py"),
               Path("scripts/evaluate_return_distribution.py"), Path("scripts/replay_execution_policy.py"),
               Path("src/strategies/momentum_scalp.py"), Path("src/analytics/execution_policy.py"))
    source_text = {str(p): p.read_text() for p in sources}
    source_sha256 = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in source_text.items()}
    dataset = json.loads(args.dataset.read_text())
    training = [r for r in dataset["observations"] if r.get("cohort", "sample") == "sample"]
    if args.policy == "expiry":
        training = [r for r in training if r.get("horizon_seconds") == MAX_HOLD_SECONDS]
        if not training or any(r.get("exit_reason") != "endpoint" for r in training if r["status"] == "complete"):
            parser.error("expiry requires endpoint labels with the same holding horizon")
    fixed = ({r["day"]: r for r in json.loads(args.input_snapshots.read_text())["inputs"]}
             if args.input_snapshots else {})
    cache = Path("state/instruments.json")
    groups = json.loads(cache.read_text()).get("groups", {}) if cache.exists() else {}
    results = []
    for day in sorted(set(args.date)):
        model = ReturnDistribution(price_move=True).fit(training, before=day+"T00:00:00")
        snapshot = (args.root / day[:4] / day[5:7] / f"quote-batches.{day}.jsonl.gz").read_bytes()
        if fixed:
            snapshot = snapshot[:fixed[day]["input_bytes"]]
            if hashlib.sha256(snapshot).hexdigest() != fixed[day]["input_sha256"]:
                raise ValueError(f"input_snapshot_mismatch: {day}")
        for policy in ("current", "conditional_return"):
            options = (dict(strategy_type=candidate_strategy(model, expiry_only=args.policy == "expiry"),
                            entry_quote_check=limit_entry_rejection)
                       if policy == "conditional_return" else {})
            for cost in ("configured", "one_tick_stress"):
                result = replay_day(args.root, day, instrument_groups=groups,
                    input_snapshot=snapshot, require_fresh_hold=not args.observed_holds,
                    adverse_sell_ticks=int(cost == "one_tick_stress"), **options)
                result.update(policy=policy, cost=cost, training_through=model.training_through,
                              candidate_exit_policy=args.policy, observed_holds=args.observed_holds)
                results.append(result)
                args.output.write_text(json.dumps(dict(results=results, decision="not_deployed")))
                print(json.dumps({k:v for k,v in result.items() if k not in
                                  {"orders", "sell_fills", "diagnostics"}}), flush=True)
    args.output.write_text(json.dumps(dict(results=results, decision="not_deployed",
        source_sha256=source_sha256, source_text=source_text,
        source_changed_during_run=any(p.read_text() != source_text[str(p)] for p in sources),
        limitations=["reused_dates_not_untouched_validation", "top_book_fills_not_broker_queue",
                     "entry_recheck_uses_next_exchange_quote_not_measured_broker_latency",
                     "unknown_executions_stop_portfolio", "no_deployment_or_restart"])))


if __name__ == "__main__":
    main()
