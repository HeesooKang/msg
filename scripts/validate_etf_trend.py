"""Offline frozen trend hypothesis; never imported by the trading runtime."""

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
import hashlib
import json
from math import ceil
from pathlib import Path

from scripts.replay_execution_policy import ReplayStrategy, batches, replay_day
from src.models import Order, OrderSide, OrderType
from src.strategies.momentum_scalp_types import TradePlan

SYMBOLS = frozenset(("152100", "292190", "114800", "123310"))
ENTRY_BARS = 20
STOP_BARS = 5
CONTROL_SEEDS = tuple(range(20))
FROZEN_ON = "2026-09-21"


def comparison_protocol(coverage, reference=None):
    days = [row["date"] for row in coverage]
    if not days or len(days) != len(set(days)):
        raise ValueError("provide distinct evaluation dates")
    if any(datetime.strptime(day, "%Y-%m-%d").strftime("%Y-%m-%d") != day for day in days):
        raise ValueError("dates must use YYYY-MM-DD")
    rules = dict(entry_bars=ENTRY_BARS, stop_bars=STOP_BARS, symbols=sorted(SYMBOLS))
    if reference is None:
        if max(days) > FROZEN_ON:
            raise ValueError("later dates require --reference with the frozen diagnostic report")
        diagnostic = [row for row in coverage if row["date"] < FROZEN_ON]
        comparison_dates = [day for day in days if day == FROZEN_ON]
    else:
        if reference.get("frozen_rules") != rules:
            raise ValueError("reference rules differ from the frozen hypothesis")
        historical = reference.get("coverage", [])
        if not historical or min(days) <= max(FROZEN_ON, *(r["date"] for r in historical)):
            raise ValueError("evaluation dates must follow all reference dates")
        diagnostic = [row for row in historical if row["date"] < FROZEN_ON]
        comparison_dates = days
    ready = sum(s.get("ready_minutes", 0) for r in diagnostic for s in r["symbols"].values())
    signals = sum(s.get("breakout_minutes", 0) for r in diagnostic for s in r["symbols"].values())
    rate = signals / ready if ready else None
    if rate is not None and not 0 <= rate <= 1:
        raise ValueError("invalid diagnostic signal frequency")
    if reference is not None and (rate is None or reference.get("control_rate") != rate):
        raise ValueError("reference lacks a reproducible frozen control rate")
    return dict(control_rate=rate, comparison_dates=comparison_dates,
                diagnostic_dates=[row["date"] for row in diagnostic])


@dataclass
class BidBar:
    minute: datetime
    first_at: datetime
    last_at: datetime
    high: int
    low: int
    close: int
    max_gap_seconds: float = 0.


class BidChannels:
    def __init__(self, symbols=SYMBOLS):
        self.symbols = None if symbols is None else frozenset(symbols)
        self.current = {}
        self.closed = defaultdict(lambda: deque(maxlen=ENTRY_BARS + 1))
        self.reasons = Counter()

    def update(self, quotes, now, freshness=5):
        for q in sorted(quotes, key=lambda q: (q.timestamp, q.symbol)):
            if self.symbols is not None and q.symbol not in self.symbols:
                continue
            if not (q.has_valid_prices and q.book_available and 0 < q.bid_price <= q.ask_price
                    and q.timestamp.date() == now.date()
                    and 0 <= (now - q.timestamp).total_seconds() <= freshness):
                self.reasons["invalid_or_stale_quote"] += 1
                continue
            minute = q.timestamp.replace(second=0, microsecond=0)
            history = self.closed[q.symbol]
            if history and minute <= history[-1].minute:
                self.reasons["closed_bar_revision_ignored"] += 1
                continue
            bar = self.current.get(q.symbol)
            if bar is not None and q.timestamp <= bar.last_at:
                continue
            if bar is not None and minute > bar.minute:
                self._close(q.symbol, freshness)
                bar = None
            if bar is None:
                self.current[q.symbol] = BidBar(minute, q.timestamp, q.timestamp,
                                                q.bid_price, q.bid_price, q.bid_price)
            else:
                bar.max_gap_seconds = max(bar.max_gap_seconds, (q.timestamp-bar.last_at).total_seconds())
                bar.last_at = q.timestamp
                bar.high = max(bar.high, q.bid_price)
                bar.low = min(bar.low, q.bid_price)
                bar.close = q.bid_price
        for symbol, bar in list(self.current.items()):
            if bar.minute + timedelta(minutes=1) <= now:
                self._close(symbol, freshness)

    def _close(self, symbol, freshness):
        bar = self.current.pop(symbol)
        end = bar.minute + timedelta(minutes=1)
        if bar.max_gap_seconds > freshness:
            self.closed[symbol].clear()
            self.reasons["incomplete_minute_path"] += 1
            return
        if ((bar.first_at - bar.minute).total_seconds() > freshness
                or (end - bar.last_at).total_seconds() > freshness):
            self.closed[symbol].clear()
            self.reasons["incomplete_minute_boundary"] += 1
            return
        previous = self.closed[symbol]
        if previous and bar.minute != previous[-1].minute + timedelta(minutes=1):
            previous.clear()
            self.reasons["missing_minute"] += 1
        previous.append(bar)

    def latest(self, symbol, now, count):
        bars = list(self.closed[symbol])
        current_minute = now.replace(second=0, microsecond=0)
        if len(bars) < count or bars[-1].minute != current_minute - timedelta(minutes=1):
            return []
        return bars[-count:]


class ETFTrendReplay(ReplayStrategy):
    """Research adapter reusing the production fills, costs and account ledger."""

    def __init__(self, market_data, config, *, control_rate=None, control_seed=0, symbols=SYMBOLS):
        super().__init__(market_data, config)
        self.symbols = None if symbols is None else frozenset(symbols)
        self.channels = BidChannels(self.symbols)
        self.attempted = {}
        self.diagnostics = dict(ready=0, signals=0, evaluated_minutes=0, decisions=[])
        self._counted = {}
        self._bar_day = None
        self.control_rate = control_rate
        self.control_seed = control_seed

    def reset_observations(self):
        self.channels = BidChannels(self.symbols)
        self._counted.clear()

    def _entry_signal(self, symbol, bar, high):
        if self.control_rate is None:
            return bar.close > high
        key = f"{self.control_seed}:{symbol}:{bar.minute.isoformat()}".encode()
        draw = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
        return draw < self.control_rate

    def _plan(self, quote, signal, stop, now):
        # In this adapter TradePlan.breakout_price stores the actual stop,
        # not the separate high used to recognize a signal.
        entry = quote.ask_price
        risk = self._risk_per_share(quote, entry, stop)
        capital = int(self._available_capital() // (entry * (1 + self.config.commission_rate)))
        room = self._remaining_daily_loss_room()
        quantity = min(capital, int(room / risk)) if risk > 0 else capital
        if quote.book_depth_available:
            quantity = min(quantity, max(0, quote.ask_size), max(0, quote.bid_size))
        committed = ceil(risk * max(0, quantity))
        allowed = quote.bid_price > stop and quantity > 0 and committed <= room
        return TradePlan(quote.symbol, allowed, "price_breakout", max(0, quantity),
                         entry, stop, committed, signal.minute + timedelta(minutes=1),
                         self._closing_auction_start(now))

    def on_batch_tick(self, quotes):
        self._ensure_active_day()
        now = self._now()
        if self._bar_day != now.date():
            self.reset_observations()
            self.attempted.clear()
            self._bar_day = now.date()
        quotes = [q for q in quotes if q.has_valid_prices and q.timestamp <= now]
        self._ingest_quotes(quotes)
        self._update_daily_halt_state()
        if self._halted:
            return self._liquidate_all()
        fresh = {q.symbol: q for q in self._fresh_quotes()
                 if q.book_available and 0 < q.bid_price <= q.ask_price}
        if (self.positions and not self._pending_entry_meta and not self._pending_sell_fills
                and all((quote := fresh.get(symbol)) is not None
                        and quote.book_depth_available and quote.bid_size >= position.quantity
                        for symbol, position in self.positions.items())
                and self.daily_pnl.realized_net_pnl + self._unrealized_net_pnl()
                >= int(self.config.daily_profit_target)):
            return [order for position in self.positions.values()
                    if (order := self._make_market_sell(position, fresh[position.symbol],
                        reason="daily_profit_target")) is not None]
        exits = []
        # A received touch of the old stop precedes any subsequent bar update.
        for symbol, position in self.positions.items():
            touched = next((q for q in sorted(quotes, key=lambda q: q.timestamp)
                            if q.symbol == symbol and q.timestamp >= position.buy_time
                            and q.book_available and 0 < q.bid_price <= position.plan.breakout_price), None)
            if touched is not None:
                order = self._make_market_sell(position, touched, reason="trailing_low")
                if order is not None:
                    exits.append(order)
        self.channels.update(quotes, now, self.config.quote_freshness_seconds)
        for symbol, position in self.positions.items():
            if any(o.symbol == symbol for o in exits):
                continue
            bars = self.channels.latest(symbol, now, STOP_BARS)
            if bars:
                stop = max(position.plan.breakout_price, min(b.low for b in bars))
                position.plan = replace(position.plan, breakout_price=stop)
            quote = fresh.get(symbol)
            reason = ("session_end" if now >= self._closing_auction_start(now)
                      else "trailing_low" if quote and quote.bid_price <= position.plan.breakout_price else "")
            if reason:
                order = self._make_market_sell(position, quote, reason=reason)
                if order is not None:
                    exits.append(order)
        if exits:
            return exits
        if now >= self._closing_auction_start(now) - timedelta(minutes=1):
            return []
        candidates = []
        for symbol in self.get_watchlist():
            if self.symbols is not None and symbol not in self.symbols:
                continue
            bars = self.channels.latest(symbol, now, ENTRY_BARS + 1)
            if not bars:
                continue
            last = bars[-1]
            high = max(b.high for b in bars[:-1])
            stop = min(b.low for b in bars[-STOP_BARS:])
            signal = self._entry_signal(symbol, last, high)
            if self._counted.get(symbol) != last.minute:
                self._counted[symbol] = last.minute
                self.diagnostics["ready"] += 1
                self.diagnostics["signals"] += int(signal)
                self.diagnostics["evaluated_minutes"] += 1
            quote = fresh.get(symbol)
            if (not signal or symbol in self.positions or quote is None
                    or self.attempted.get(symbol) == last.minute):
                continue
            plan = self._plan(quote, last, stop, now)
            if plan.allowed:
                candidates.append((quote, plan, last, high))
        if not candidates or not self._entry_submission_available():
            return []
        quote, plan, signal, high = min(candidates, key=lambda item: (
            (item[0].ask_price - item[0].bid_price) / item[0].ask_price,
            -(item[2].close / item[3] - 1), item[0].symbol))
        self.attempted[quote.symbol] = signal.minute
        self._pending_entry_meta[quote.symbol] = dict(
            plan=plan, signal_price=quote.current_price, created_at=now, order_no="")
        self.diagnostics["decisions"].append(dict(
            at=now.isoformat(), symbol=quote.symbol, signal_minute=signal.minute.isoformat(),
            channel_high=high, signal_close=signal.close, stop=plan.breakout_price,
            ask=plan.entry_limit_price, quantity=plan.quantity, risk=plan.committed_risk_net_abs))
        return [Order(quote.symbol, OrderSide.BUY, OrderType.LIMIT, plan.quantity,
                      plan.entry_limit_price, quote.current_price, "price_breakout")]


def audit_observations(root, day):
    channels = BidChannels()
    counts = defaultdict(Counter)
    counted, last = {}, {}
    boundary_reasons = Counter()
    for batch, now, quotes in batches(root, day):
        if batch.get("reset_history"):
            boundary_reasons.update(channels.reasons)
            channels = BidChannels()
            counted.clear()
            last.clear()
        channels.update(quotes, now)
        for q in quotes:
            if q.symbol not in SYMBOLS or not q.book_available or not 0 < q.bid_price <= q.ask_price:
                continue
            if not 0 <= (now - q.timestamp).total_seconds() <= 5:
                counts[q.symbol]["stale_quotes"] += 1
                continue
            previous = last.get(q.symbol)
            if previous is not None and q.timestamp <= previous:
                continue
            if previous is not None and (q.timestamp - previous).total_seconds() > 5:
                counts[q.symbol]["quote_gaps_over_5s"] += 1
            last[q.symbol] = q.timestamp
            counts[q.symbol]["fresh_quotes"] += 1
        for symbol in SYMBOLS:
            bars = channels.latest(symbol, now, ENTRY_BARS + 1)
            if not bars or counted.get(symbol) == bars[-1].minute or (now.hour, now.minute) >= (15, 19):
                continue
            if batch.get("universe") is not None and symbol not in batch["universe"]:
                continue
            counted[symbol] = bars[-1].minute
            counts[symbol]["ready_minutes"] += 1
            counts[symbol]["breakout_minutes"] += int(bars[-1].close > max(b.high for b in bars[:-1]))
    boundary_reasons.update(channels.reasons)
    return dict(date=day, symbols={s:dict(counts[s]) for s in sorted(SYMBOLS)},
                boundary_reasons=dict(boundary_reasons))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("reports/forecast-outcomes"))
    parser.add_argument("--date", action="append", required=True)
    parser.add_argument("--reference", type=Path,
                        help="frozen historical report supplying the control rate for later dates")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference_bytes = args.reference.read_bytes() if args.reference else None
    reference = json.loads(reference_bytes) if reference_bytes is not None else None
    try:
        comparison_protocol([dict(date=day, symbols={}) for day in args.date], reference)
    except ValueError as exc:
        parser.error(str(exc))
    sources = [Path(__file__), Path("scripts/replay_execution_policy.py"),
               Path("src/strategies/momentum_scalp.py"), Path("src/strategies/momentum_scalp_pnl.py"),
               Path("docs/etf_trend_validation.md")]
    source_hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    coverage = [audit_observations(args.root, day) for day in args.date]
    protocol = comparison_protocol(coverage, reference)
    control_rate = protocol["control_rate"]
    print(json.dumps(dict(coverage=coverage, control_rate=control_rate)), flush=True)
    reports = []
    for day in args.date:
        for name, policy, fee, slip in (("baseline", ReplayStrategy, 1, 0),
                                      ("candidate", ETFTrendReplay, 1, 0),
                                      ("candidate_cost_stress", ETFTrendReplay, 2, 1)):
            result = replay_day(args.root, day, instrument_groups={s:"EF" for s in SYMBOLS},
                strategy_type=policy, universe_symbols=SYMBOLS,
                commission_multiplier=fee, adverse_sell_ticks=slip, require_fresh_hold=True)
            result["policy"] = name
            reports.append(result)
            print(json.dumps({k:v for k,v in result.items()
                              if k not in {"orders", "sell_fills", "diagnostics"}}), flush=True)
    if control_rate is not None:
        for day in protocol["comparison_dates"]:
            for seed in CONTROL_SEEDS:
                def control(market_data, config):
                    return ETFTrendReplay(market_data, config, control_rate=control_rate, control_seed=seed)
                result = replay_day(args.root, day, instrument_groups={s:"EF" for s in SYMBOLS},
                    strategy_type=control, universe_symbols=SYMBOLS, require_fresh_hold=True)
                result.update(policy="random_entry_control", seed=seed)
                reports.append(result)
                print(json.dumps({k:v for k,v in result.items()
                                  if k not in {"orders", "sell_fills", "diagnostics"}}), flush=True)
    current_hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    if current_hashes != source_hashes:
        raise RuntimeError("source changed during evaluation; rerun without concurrent edits")
    result = dict(reports=reports, frozen_rules=dict(entry_bars=ENTRY_BARS, stop_bars=STOP_BARS,
        symbols=sorted(SYMBOLS)), source_sha256=source_hashes,
        coverage=coverage, control_rate=control_rate, control_seeds=CONTROL_SEEDS,
        comparison_protocol=protocol,
        reference_report=(dict(path=str(args.reference), sha256=hashlib.sha256(reference_bytes).hexdigest())
                          if reference_bytes is not None else None),
        independent_holdout=False, runtime_adopted=False,
        limitations=["a later date alone does not prove an untouched independent holdout",
                     "next observed top-of-book is not a broker queue simulation",
                     "possible fund withholding is not in these raw quote files",
                     "incomplete replay net_total is null, not known-prefix profit"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
