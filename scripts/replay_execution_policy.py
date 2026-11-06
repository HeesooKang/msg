"""Offline live-strategy replay. Unknown executions stop the portfolio.

python -m scripts.replay_execution_policy --root reports/market-quotes
       --date YYYY-MM-DD --output /private/tmp/breakout-replay.json
No credentials, broker calls, model fitting, or original state writes.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
from io import BytesIO, TextIOWrapper
import json
from datetime import datetime
from pathlib import Path
import tempfile
from types import SimpleNamespace

from src.analytics.execution_policy import limit_fill, market_fill
from src.instruments import tick_size
from src.models import OrderResult, OrderSide, Quote
from src.strategies.momentum_scalp import MomentumScalpStrategy
from src.strategies.momentum_scalp_types import MomentumScalpConfig
from src.trading import TradingAPI


class MemoryArchive:
    def record(self, *args, **kwargs):
        pass


class ReplayStrategy(MomentumScalpStrategy):
    def get_watchlist(self):
        return list(dict.fromkeys([*self.fixed_pool, *self.positions]))

    def _save_daily_state(self):
        pass


def batches(root: Path, day: str, *, snapshot=None):
    path = root / day[:4] / day[5:7] / f"quote-batches.{day}.jsonl.gz"
    compressed = path.read_bytes() if snapshot is None else snapshot
    with TextIOWrapper(gzip.GzipFile(fileobj=BytesIO(compressed)), encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            at = datetime.fromisoformat(item["observed_timestamp"])
            quotes = [Quote(**dict(q, timestamp=datetime.fromisoformat(q["timestamp"]))) for q in item["quotes"]]
            yield item, at, [q for q in quotes if q.has_valid_prices and q.timestamp <= at]


def replay_day(root: Path, day: str, *, instrument_groups=None,
               strategy_type=ReplayStrategy, universe_symbols=None,
               commission_multiplier=1.0, adverse_sell_ticks=0,
               require_fresh_hold=False, input_snapshot=None,
               entry_quote_check=TradingAPI.entry_quote_rejection):
    with tempfile.TemporaryDirectory() as directory:
        config = MomentumScalpConfig(static_watchlist=[], inverse_etfs=[],
            commission_rate=.000142 * commission_multiplier, sell_tax_rate=.0015,
            daily_state_path=f"{directory}/state.json", quote_archive_root=f"{directory}/quotes")
        market_data = SimpleNamespace(client=SimpleNamespace(
            config=SimpleNamespace(is_paper=True), rate_limit_cooldown_remaining=lambda: 0.0))
        strategy = strategy_type(market_data, config)
        strategy.instrument_groups.update(instrument_groups or {})
        strategy._quote_archive = MemoryArchive()
        pending, attempts, seen = [], [], set()
        unknown = None
        universe_verified = True
        frames = 0
        last_at = None
        first_at = None
        last_bid_at = {}
        peak_equity, max_drawdown = 0, 0
        input_path = root / day[:4] / day[5:7] / f"quote-batches.{day}.jsonl.gz"
        snapshot = input_path.read_bytes() if input_snapshot is None else input_snapshot
        for batch, now, quotes in batches(root, day, snapshot=snapshot):
            frames += 1
            last_at = now
            first_at = first_at or now
            strategy.set_simulated_now(now)
            strategy._ensure_active_day()
            if batch.get("reset_history"):
                strategy._quotes_cache.clear()
                strategy._recent_quotes.clear()
                if hasattr(strategy, "reset_observations"):
                    strategy.reset_observations()
            for q in quotes:
                q.security_group = q.security_group or strategy.instrument_groups.get(q.symbol, "")
            seen.update(q.symbol for q in quotes)
            universe = batch.get("universe")
            if universe is None:
                universe_verified = False
                universe = sorted(seen)
            if universe_symbols is not None:
                universe = [s for s in universe if s in universe_symbols]
            strategy.fixed_pool = list(universe)
            strategy._pool = list(universe)
            if require_fresh_hold:
                for symbol, position in strategy.positions.items():
                    after = last_bid_at.get(symbol, position.buy_time)
                    for q in sorted((q for q in quotes if q.symbol == symbol
                                     and q.book_available and q.bid_price > 0
                                     and q.timestamp > after), key=lambda q: q.timestamp):
                        if (q.timestamp - after).total_seconds() > config.quote_freshness_seconds:
                            unknown = dict(symbol=symbol, reason="held_quote_gap")
                            break
                        after = q.timestamp
                    if (now - after).total_seconds() > config.quote_freshness_seconds:
                        unknown = dict(symbol=symbol, reason="held_quote_stale")
                    if unknown:
                        break
                if unknown:
                    break
            for q in quotes:
                if q.book_available and 0 < q.bid_price <= q.ask_price:
                    previous = last_bid_at.get(q.symbol)
                    if previous is None or q.timestamp > previous:
                        last_bid_at[q.symbol] = q.timestamp
            strategy._ingest_quotes(quotes)
            for attempt in list(pending):
                order = attempt["order"]
                matching = sorted((q for q in quotes if q.symbol == order.symbol
                                   and q.timestamp > attempt["after"]), key=lambda q: q.timestamp)
                for q in matching:
                    if (q.timestamp - attempt["after"]).total_seconds() > config.quote_freshness_seconds:
                        unknown = dict(order_id=attempt["id"], reason="next_execution_quote_late")
                        break
                    if (now - q.timestamp).total_seconds() > config.quote_freshness_seconds:
                        unknown = dict(order_id=attempt["id"], reason="execution_quote_received_late")
                        break
                    if order.side == OrderSide.BUY:
                        expired = order.submit_before is not None and now.timestamp() >= order.submit_before
                        rejected = ("entry_submission_expired" if expired
                                    else entry_quote_check(order, q))
                        if rejected:
                            strategy.on_order_filled(OrderResult(success=False,
                                symbol=order.symbol, side=order.side, requested_quantity=order.quantity,
                                requested_reason=order.requested_reason, timestamp=now,
                                error_code="LOCAL_ORDER_EXPIRED" if expired else "LOCAL_ORDER_INVALIDATED",
                                error_category="stale_order", message=rejected))
                            attempt["status"] = "expired" if expired else "rejected"
                            attempt["rejection_reason"] = rejected
                            pending.remove(attempt)
                            break
                    remaining = order.quantity - attempt["filled"]
                    fill = (limit_fill(q, order.price, remaining) if order.side == OrderSide.BUY
                            else market_fill(q, remaining))
                    if fill is None:
                        unknown = dict(order_id=attempt["id"], reason="execution_depth_unknown")
                        break
                    quantity, price = fill
                    if quantity > 0 and order.side == OrderSide.SELL:
                        price = max(1, price - adverse_sell_ticks * tick_size(price, q.security_group))
                    cancelled = order.side == OrderSide.BUY and quantity == 0
                    strategy.on_order_filled(OrderResult(success=not cancelled,
                        symbol=order.symbol, side=order.side, quantity=quantity, price=price,
                        requested_quantity=remaining, order_no=attempt["id"],
                        fill_id=f'{attempt["id"]}:{q.timestamp.isoformat()}', timestamp=now,
                        fill_mode="limit_cancelled" if cancelled else ("limit" if order.side == OrderSide.BUY else "market"),
                        requested_reason=order.requested_reason))
                    attempt["filled"] += quantity
                    attempt["after"] = q.timestamp
                    attempt["status"] = "unfilled" if cancelled else ("filled" if quantity == remaining else "partial")
                    # Buy remainder is cancelled; a market sell remainder stays pending.
                    if order.side == OrderSide.BUY or quantity == remaining:
                        pending.remove(attempt)
                        break
                if unknown:
                    break
                if attempt in pending and (now - attempt["after"]).total_seconds() > config.quote_freshness_seconds:
                    unknown = dict(order_id=attempt["id"], reason="execution_quote_missing")
                    break
            if unknown:
                break
            for order in strategy.on_batch_tick(quotes):
                if any(a["order"].symbol == order.symbol and a["order"].side == order.side for a in pending):
                    continue
                attempt = dict(id=f"replay-{len(attempts)}", at=now, after=now,
                               order=order, status="pending", filled=0)
                pending.append(attempt)
                attempts.append(attempt)
                if order.side == OrderSide.SELL:
                    strategy.on_order_filled(OrderResult(success=True, symbol=order.symbol,
                        side=order.side, order_no=attempt["id"], quantity=0, price=0,
                        reference_price=order.reference_price, requested_quantity=order.quantity,
                        fill_mode="market_pending", requested_reason=order.requested_reason))
            equity = strategy.daily_pnl.realized_net_pnl + strategy._unrealized_net_pnl()
            peak_equity = max(peak_equity, equity)
            max_drawdown = max(max_drawdown, peak_equity - equity)
            if strategy._halted and not strategy.positions and not pending:
                break
        complete = frames > 0 and unknown is None and not pending and not strategy.positions
        return dict(date=day, frames=frames, start=first_at.isoformat() if first_at else None,
            through=last_at.isoformat() if last_at else None,
            input_sha256=hashlib.sha256(snapshot).hexdigest(),
            input_bytes=len(snapshot),
            universe_verified=universe_verified, complete=bool(complete), unknown_execution=unknown,
            realized_net_known_prefix=strategy.daily_pnl.realized_net_pnl,
            net_total=strategy.daily_pnl.realized_net_pnl if complete else None,
            open_positions=len(strategy.positions), unresolved_orders=len(pending),
            halt_reason=strategy._halt_reason, sell_fills=strategy._sell_fill_ledger,
            max_drawdown_known_prefix=max_drawdown,
            diagnostics=getattr(strategy, "diagnostics", {}),
            orders=[dict(order_id=a["id"], timestamp=a["at"].isoformat(), symbol=a["order"].symbol,
                         side=a["order"].side.value, requested=a["order"].quantity, filled=a["filled"],
                         reason=a["order"].requested_reason, status=a["status"],
                         rejection_reason=a.get("rejection_reason", "")) for a in attempts])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("reports/market-quotes"))
    parser.add_argument("--date", required=True, action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instrument-cache", type=Path, default=Path("state/instruments.json"))
    args = parser.parse_args()
    groups = json.loads(args.instrument_cache.read_text()).get("groups", {}) if args.instrument_cache.exists() else {}
    reports = []
    for day in args.date:
        result = replay_day(args.root, day, instrument_groups=groups)
        reports.append(result)
        print(json.dumps({k:v for k,v in result.items() if k not in {"orders", "sell_fills"}}), flush=True)
    sources = [Path(__file__), Path("src/strategies/momentum_scalp.py"),
               Path("src/strategies/momentum_scalp_types.py"),
               Path("src/analytics/execution_policy.py"), Path("src/strategies/momentum_scalp_pnl.py"),
               Path("src/trading.py"), Path("src/models.py")]
    report = dict(days=reports, source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        assumptions=["next observed executable quote, not a broker queue simulation",
                     "buy submission and fill use the next quote with the live entry check and received-time deadline",
                     "unknown execution stops replay; known prefix is not a whole-day return",
                     "missing historical universe is not an exact watchlist replay",
                     "1000000 capital, +10000 realized target, -5000 total loss stop, no bypass"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
