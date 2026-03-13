"""1분봉 리플레이용 백테스트 엔진."""

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from src.backtest.engine import BacktestResult, DailyRecord, TradeRecord
from src.models import Order, OrderResult, OrderSide, Quote
from src.strategy import BaseStrategy

logger = logging.getLogger("kis_trader.backtest.intraday")


@dataclass
class _DayMeta:
    prev_close: int
    day_open: int


class IntradayBacktestEngine:
    """1분봉 데이터를 사용해 전략을 리플레이한다."""

    def __init__(
        self,
        strategy: BaseStrategy,
        data: Dict[str, pd.DataFrame],
        initial_capital: int = 1_000_000,
        slippage_bps: int = 0,
        commission_rate: float = 0.00015,
        tax_rate: float = 0.002,
    ):
        self.strategy = strategy
        self.data = self._prepare_data(data)
        self.initial_capital = initial_capital
        self.slippage_bps = slippage_bps
        self.commission_rate = commission_rate
        self.tax_rate = tax_rate

        self._capital = initial_capital
        self._positions: Dict[str, dict] = {}
        self._daily_pnl = 0
        self._pending_orders: List[Order] = []
        self._day_meta = self._build_day_meta()

    def run(self, start_date: str, end_date: str) -> BacktestResult:
        trading_days = self._get_trading_days(start_date, end_date)
        logger.info("1분봉 백테스트 시작: %s ~ %s (%d 거래일)", start_date, end_date, len(trading_days))

        result = BacktestResult(
            initial_capital=self.initial_capital,
            final_capital=self.initial_capital,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
        )
        self._capital = self.initial_capital

        for day in trading_days:
            day_frames = self._day_frames(day)
            if not day_frames:
                continue

            self._daily_pnl = 0
            day_trades = 0
            first_timestamp = min(df["timestamp"].min() for df in day_frames.values())
            self._set_strategy_time(first_timestamp.to_pydatetime())
            self.strategy.initialize()
            self._load_strategy_avg_volumes(day)

            for ts in self._day_timestamps(day_frames):
                tick_quotes = self._build_quotes_for_timestamp(day, ts, day_frames)
                if not tick_quotes:
                    continue

                self._set_strategy_time(ts.to_pydatetime())
                day_trades += self._fill_pending_orders(tick_quotes, result, trade_date=day)
                self._pending_orders = self.strategy.on_batch_tick(tick_quotes)

            if self._pending_orders:
                last_quotes = self._build_quotes_for_timestamp(day, self._day_timestamps(day_frames)[-1], day_frames)
                day_trades += self._fill_pending_orders(last_quotes, result, trade_date=day)

            if self._positions:
                last_ts = self._day_timestamps(day_frames)[-1]
                close_quotes = self._build_quotes_for_timestamp(day, last_ts, day_frames)
                quote_map = {q.symbol: q for q in close_quotes}
                self._set_strategy_time(last_ts.to_pydatetime())
                for symbol in list(self._positions.keys()):
                    q = quote_map.get(symbol)
                    if not q:
                        continue
                    pos = self._positions.pop(symbol)
                    fill_price = q.current_price
                    gross_proceeds = fill_price * pos["qty"]
                    sell_commission = int(gross_proceeds * self.commission_rate)
                    sell_tax = int(gross_proceeds * self.tax_rate)
                    net_proceeds = gross_proceeds - sell_commission - sell_tax
                    buy_comm = pos.get("buy_comm", 0)
                    net_pnl = net_proceeds - (pos["price"] * pos["qty"] + buy_comm)

                    self._capital += net_proceeds
                    self._daily_pnl += net_pnl
                    if net_pnl > 0:
                        result.winning_trades += 1
                    elif net_pnl < 0:
                        result.losing_trades += 1

                    self.strategy.on_order_filled(
                        OrderResult(
                            success=True,
                            symbol=symbol,
                            side=OrderSide.SELL,
                            quantity=pos["qty"],
                            price=fill_price,
                        )
                    )
                    result.trade_records.append(
                        TradeRecord(
                            date=day,
                            symbol=symbol,
                            side="sell",
                            quantity=pos["qty"],
                            price=fill_price,
                            pnl=net_pnl,
                        )
                    )
                    result.total_trades += 1
                    day_trades += 1

            result.daily_records.append(
                DailyRecord(
                    date=day,
                    capital=self._capital,
                    realized_pnl=self._daily_pnl,
                    trade_count=day_trades,
                    positions_held=len(self._positions),
                )
            )

        result.final_capital = self._capital
        self._set_strategy_time(None)
        logger.info("1분봉 백테스트 완료")
        return result

    def _prepare_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        prepared: Dict[str, pd.DataFrame] = {}
        for symbol, df in data.items():
            if df.empty or "timestamp" not in df.columns:
                continue
            work = df.copy()
            work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
            work = work.dropna(subset=["timestamp"]).sort_values("timestamp")
            if "trade_date" not in work.columns:
                work["trade_date"] = work["timestamp"].dt.strftime("%Y%m%d")
            prepared[symbol] = work.reset_index(drop=True)
        return prepared

    def _build_day_meta(self) -> Dict[str, Dict[str, _DayMeta]]:
        meta: Dict[str, Dict[str, _DayMeta]] = defaultdict(dict)
        for symbol, df in self.data.items():
            for trade_date, day_df in df.groupby("trade_date"):
                day_df = day_df.sort_values("timestamp")
                prev_close = int(day_df["close"].iloc[0])
                prior = df[df["trade_date"] < trade_date]
                if not prior.empty:
                    prev_close = int(prior.sort_values("timestamp")["close"].iloc[-1])
                meta[symbol][trade_date] = _DayMeta(
                    prev_close=prev_close,
                    day_open=int(day_df["open"].iloc[0]),
                )
        return meta

    def _get_trading_days(self, start: str, end: str) -> List[str]:
        days = set()
        for df in self.data.values():
            dates = df["trade_date"][(df["trade_date"] >= start) & (df["trade_date"] <= end)]
            days.update(dates.tolist())
        return sorted(days)

    def _day_frames(self, day: str) -> Dict[str, pd.DataFrame]:
        return {
            symbol: df[df["trade_date"] == day].copy()
            for symbol, df in self.data.items()
            if not df[df["trade_date"] == day].empty
        }

    @staticmethod
    def _day_timestamps(day_frames: Dict[str, pd.DataFrame]) -> List[pd.Timestamp]:
        timestamps = set()
        for df in day_frames.values():
            timestamps.update(df["timestamp"].tolist())
        return sorted(timestamps)

    def _build_quotes_for_timestamp(
        self,
        day: str,
        ts: pd.Timestamp,
        day_frames: Dict[str, pd.DataFrame],
    ) -> List[Quote]:
        quotes: List[Quote] = []
        for symbol, df in day_frames.items():
            row_df = df[df["timestamp"] == ts]
            if row_df.empty:
                continue
            row = row_df.iloc[-1]
            meta = self._day_meta.get(symbol, {}).get(day)
            prev_close = meta.prev_close if meta is not None else int(row["open"])
            current_price = int(row["close"])
            change = current_price - prev_close
            change_rate = (change / prev_close * 100) if prev_close > 0 else 0.0
            quotes.append(
                Quote(
                    symbol=symbol,
                    name=symbol,
                    current_price=current_price,
                    change=change,
                    change_rate=change_rate,
                    open_price=int(meta.day_open if meta is not None else row["open"]),
                    high_price=int(row["high"]),
                    low_price=int(row["low"]),
                    volume=int(row["cumulative_volume"]),
                    trade_amount=int(current_price * max(0, int(row["volume"]))),
                    timestamp=ts.to_pydatetime(),
                )
            )
        return quotes

    def _fill_pending_orders(
        self,
        quotes: List[Quote],
        result: BacktestResult,
        trade_date: str,
    ) -> int:
        quote_map = {q.symbol: q for q in quotes}
        filled_count = 0

        for order in self._pending_orders:
            q = quote_map.get(order.symbol)
            if not q or q.current_price <= 0:
                continue

            if order.side == OrderSide.BUY:
                fill_price = int(q.current_price * (1 + self.slippage_bps / 10000))
                gross_cost = fill_price * order.quantity
                buy_commission = int(gross_cost * self.commission_rate)
                total_cost = gross_cost + buy_commission
                if total_cost > self._capital:
                    continue
                self._capital -= total_cost
                self._positions[order.symbol] = {
                    "price": fill_price,
                    "qty": order.quantity,
                    "buy_comm": buy_commission,
                }
                self.strategy.on_order_filled(
                    OrderResult(
                        success=True,
                        symbol=order.symbol,
                        side=OrderSide.BUY,
                        quantity=order.quantity,
                        price=fill_price,
                    )
                )
                result.trade_records.append(
                    TradeRecord(
                        date=trade_date,
                        symbol=order.symbol,
                        side="buy",
                        quantity=order.quantity,
                        price=fill_price,
                    )
                )
                result.total_trades += 1
                filled_count += 1
                continue

            pos = self._positions.pop(order.symbol, None)
            if not pos:
                continue

            fill_price = int(q.current_price * (1 - self.slippage_bps / 10000))
            gross_proceeds = fill_price * order.quantity
            sell_commission = int(gross_proceeds * self.commission_rate)
            sell_tax = int(gross_proceeds * self.tax_rate)
            net_proceeds = gross_proceeds - sell_commission - sell_tax
            buy_comm = pos.get("buy_comm", 0)
            net_pnl = net_proceeds - (pos["price"] * order.quantity + buy_comm)

            self._capital += net_proceeds
            self._daily_pnl += net_pnl
            if net_pnl > 0:
                result.winning_trades += 1
            elif net_pnl < 0:
                result.losing_trades += 1

            self.strategy.on_order_filled(
                OrderResult(
                    success=True,
                    symbol=order.symbol,
                    side=OrderSide.SELL,
                    quantity=order.quantity,
                    price=fill_price,
                )
            )
            result.trade_records.append(
                TradeRecord(
                    date=trade_date,
                    symbol=order.symbol,
                    side="sell",
                    quantity=order.quantity,
                    price=fill_price,
                    pnl=net_pnl,
                )
            )
            result.total_trades += 1
            filled_count += 1

        self._pending_orders = []
        return filled_count

    def _set_strategy_time(self, now: Optional[datetime]):
        if hasattr(self.strategy, "set_simulated_now"):
            self.strategy.set_simulated_now(now)

    def _load_strategy_avg_volumes(self, day: str):
        if not hasattr(self.strategy, "load_avg_volumes"):
            return

        avg_volumes: Dict[str, int] = {}
        for symbol, df in self.data.items():
            prior = df[df["trade_date"] < day]
            if prior.empty:
                continue
            daily_volume = prior.groupby("trade_date")["volume"].sum().tail(5)
            if daily_volume.empty:
                continue
            avg_volumes[symbol] = int(daily_volume.mean())
        self.strategy.load_avg_volumes(avg_volumes)
