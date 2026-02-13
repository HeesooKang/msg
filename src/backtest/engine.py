"""백테스트 엔진: 일봉 OHLCV로 전략을 시뮬레이션한다."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List

import pandas as pd

from src.models import Order, OrderResult, OrderSide, Quote
from src.strategy import BaseStrategy

logger = logging.getLogger("kis_trader.backtest.engine")


@dataclass
class TradeRecord:
    """체결 기록."""
    date: str
    symbol: str
    side: str
    quantity: int
    price: int
    pnl: int = 0


@dataclass
class DailyRecord:
    """일별 기록."""
    date: str
    capital: int
    realized_pnl: int
    trade_count: int
    positions_held: int


@dataclass
class BacktestResult:
    """백테스트 결과."""
    initial_capital: int
    final_capital: int
    total_trades: int
    winning_trades: int
    losing_trades: int
    daily_records: List[DailyRecord] = field(default_factory=list)
    trade_records: List[TradeRecord] = field(default_factory=list)

    @property
    def total_return_pct(self) -> float:
        if self.initial_capital == 0:
            return 0
        return (self.final_capital - self.initial_capital) / self.initial_capital * 100

    @property
    def win_rate(self) -> float:
        total = self.winning_trades + self.losing_trades
        return self.winning_trades / total * 100 if total > 0 else 0

    @property
    def avg_win(self) -> float:
        wins = [t.pnl for t in self.trade_records if t.side == "sell" and t.pnl > 0]
        return sum(wins) / len(wins) if wins else 0

    @property
    def avg_loss(self) -> float:
        losses = [t.pnl for t in self.trade_records if t.side == "sell" and t.pnl < 0]
        return sum(losses) / len(losses) if losses else 0

    @property
    def max_drawdown_pct(self) -> float:
        if not self.daily_records:
            return 0
        peak = self.initial_capital
        max_dd = 0
        for rec in self.daily_records:
            if rec.capital > peak:
                peak = rec.capital
            dd = (peak - rec.capital) / peak * 100
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def days_target_hit(self, target: int = 10_000) -> int:
        return sum(1 for r in self.daily_records if r.realized_pnl >= target)

    def days_loss_limit_hit(self, limit: int = -5_000) -> int:
        return sum(1 for r in self.daily_records if r.realized_pnl <= limit)


class BacktestEngine:
    """일봉 데이터를 사용해 전략을 백테스트한다.

    하루를 4틱(시가→첫 극단→둘째 극단→종가)으로 시뮬레이션.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        data: Dict[str, pd.DataFrame],
        initial_capital: int = 1_000_000,
        slippage_bps: int = 0,
        commission_rate: float = 0.00015,   # 0.015% 수수료 (매수/매도)
        tax_rate: float = 0.002,            # 0.20% 세금+슬리피지 (매도 시)
    ):
        self.strategy = strategy
        self.data = self._prepare_data(data)
        self.initial_capital = initial_capital
        self.slippage_bps = slippage_bps
        self.commission_rate = commission_rate
        self.tax_rate = tax_rate

        self._capital = initial_capital
        self._positions: Dict[str, dict] = {}  # {symbol: {price, qty, buy_comm}}
        self._daily_pnl = 0
        self._pending_orders: List[Order] = []

    def run(self, start_date: str, end_date: str) -> BacktestResult:
        """백테스트를 실행한다."""
        trading_days = self._get_trading_days(start_date, end_date)
        logger.info("백테스트 시작: %s ~ %s (%d 거래일)", start_date, end_date, len(trading_days))

        result = BacktestResult(
            initial_capital=self.initial_capital,
            final_capital=self.initial_capital,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
        )

        self._capital = self.initial_capital

        for day in trading_days:
            self._daily_pnl = 0
            day_trades = 0

            # 전략 초기화 (매일 리셋)
            self.strategy.initialize()

            # 하루 4틱 시뮬레이션
            ticks = self._generate_day_ticks(day)
            for tick_quotes in ticks:
                if not tick_quotes:
                    continue

                # 이전 틱의 대기 주문 체결
                self._fill_pending_orders(tick_quotes, result)

                # 전략 실행
                orders = self.strategy.on_batch_tick(tick_quotes)
                self._pending_orders = orders

            # 대기 주문은 종가에 체결 시도
            if self._pending_orders:
                close_quotes = ticks[-1] if ticks else []
                self._fill_pending_orders(close_quotes, result)

            # 장마감 강제 청산: 잔여 포지션을 종가에 매도 (오버나잇 없음)
            if self._positions and ticks:
                close_quotes = ticks[-1]
                quote_map = {q.symbol: q for q in close_quotes}
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

                    fill_result = OrderResult(
                        success=True, symbol=symbol, side=OrderSide.SELL,
                        quantity=pos["qty"], price=fill_price,
                    )
                    self.strategy.on_order_filled(fill_result)

                    result.trade_records.append(TradeRecord(
                        date=day, symbol=symbol, side="sell",
                        quantity=pos["qty"], price=fill_price, pnl=net_pnl,
                    ))
                    result.total_trades += 1

            # 일별 기록
            portfolio_value = self._capital
            result.daily_records.append(DailyRecord(
                date=day,
                capital=portfolio_value,
                realized_pnl=self._daily_pnl,
                trade_count=day_trades,
                positions_held=len(self._positions),
            ))

        # 최종 자본 계산 (매일 청산하므로 잔여 포지션 없음)
        result.final_capital = self._capital

        logger.info("백테스트 완료")
        return result

    def _prepare_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """데이터를 날짜별로 접근 가능하도록 정리한다."""
        prepared = {}
        for symbol, df in data.items():
            if "stck_bsop_date" not in df.columns:
                continue
            df = df.copy()
            for col in ["stck_oprc", "stck_hgpr", "stck_lwpr", "stck_clpr", "acml_vol"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
            df = df.set_index("stck_bsop_date")
            prepared[symbol] = df
        return prepared

    def _get_trading_days(self, start: str, end: str) -> List[str]:
        """데이터에 존재하는 거래일 목록을 반환한다."""
        all_dates = set()
        for df in self.data.values():
            dates = df.index[(df.index >= start) & (df.index <= end)]
            all_dates.update(dates)
        return sorted(all_dates)

    def _generate_day_ticks(self, day: str) -> List[List[Quote]]:
        """하루를 4틱으로 변환한다: 시가→첫극단→둘째극단→종가."""
        ticks = [[], [], [], []]  # open, first_ext, second_ext, close

        for symbol, df in self.data.items():
            if day not in df.index:
                continue

            row = df.loc[day]
            o = int(row.get("stck_oprc", 0))
            h = int(row.get("stck_hgpr", 0))
            l = int(row.get("stck_lwpr", 0))
            c = int(row.get("stck_clpr", 0))
            v = int(row.get("acml_vol", 0))

            if o <= 0 or c <= 0:
                continue

            prev_close = int(row.get("stck_prdy_clpr", o))
            if prev_close <= 0:
                prev_close = o

            # 상승일: O → L → H → C, 하락일: O → H → L → C
            if c >= o:
                prices = [o, l, h, c]
            else:
                prices = [o, h, l, c]

            for i, price in enumerate(prices):
                change = price - prev_close
                change_rate = change / prev_close * 100 if prev_close > 0 else 0

                ticks[i].append(Quote(
                    symbol=symbol,
                    name=symbol,
                    current_price=price,
                    change=change,
                    change_rate=change_rate,
                    open_price=o,
                    high_price=h if i >= 2 else max(o, price),
                    low_price=l if i >= 2 else min(o, price),
                    volume=v * (i + 1) // 4,
                    trade_amount=0,
                ))

        return ticks

    def _fill_pending_orders(self, quotes: List[Quote], result: BacktestResult):
        """대기 주문을 현재 틱 가격에 체결한다."""
        quote_map = {q.symbol: q for q in quotes}

        for order in self._pending_orders:
            q = quote_map.get(order.symbol)
            if not q or q.current_price <= 0:
                continue

            if order.side == OrderSide.BUY:
                # 슬리피지 적용
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

                fill_result = OrderResult(
                    success=True, symbol=order.symbol, side=OrderSide.BUY,
                    quantity=order.quantity, price=fill_price,
                )
                self.strategy.on_order_filled(fill_result)

                result.trade_records.append(TradeRecord(
                    date="", symbol=order.symbol, side="buy",
                    quantity=order.quantity, price=fill_price,
                ))
                result.total_trades += 1

            elif order.side == OrderSide.SELL:
                pos = self._positions.pop(order.symbol, None)
                if not pos:
                    continue

                fill_price = int(q.current_price * (1 - self.slippage_bps / 10000))
                gross_proceeds = fill_price * order.quantity
                sell_commission = int(gross_proceeds * self.commission_rate)
                sell_tax = int(gross_proceeds * self.tax_rate)
                net_proceeds = gross_proceeds - sell_commission - sell_tax

                # 순손익 = 매도순수익 - 매수총비용
                buy_comm = pos.get("buy_comm", 0)
                net_pnl = net_proceeds - (pos["price"] * order.quantity + buy_comm)

                self._capital += net_proceeds
                self._daily_pnl += net_pnl

                if net_pnl > 0:
                    result.winning_trades += 1
                elif net_pnl < 0:
                    result.losing_trades += 1

                fill_result = OrderResult(
                    success=True, symbol=order.symbol, side=OrderSide.SELL,
                    quantity=order.quantity, price=fill_price,
                )
                self.strategy.on_order_filled(fill_result)

                result.trade_records.append(TradeRecord(
                    date="", symbol=order.symbol, side="sell",
                    quantity=order.quantity, price=fill_price, pnl=net_pnl,
                ))
                result.total_trades += 1

        self._pending_orders = []
