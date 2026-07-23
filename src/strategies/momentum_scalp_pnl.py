from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class TradePnL:
    gross_pnl: int
    net_pnl: int
    fees: int
    taxes: int


def calculate_trade_pnl_from_prices(
    *,
    entry_price: int,
    exit_price: int,
    quantity: int,
    commission_rate: float,
    tax_slippage_rate: float,
) -> TradePnL:
    qty = max(0, int(quantity or 0))
    if qty <= 0:
        return TradePnL(gross_pnl=0, net_pnl=0, fees=0, taxes=0)
    gross_buy = max(0, int(entry_price or 0)) * qty
    gross_sell = max(0, int(exit_price or 0)) * qty
    gross_pnl = gross_sell - gross_buy
    fees = int((gross_buy + gross_sell) * float(commission_rate))
    taxes = int(gross_sell * float(tax_slippage_rate))
    return TradePnL(
        gross_pnl=int(gross_pnl),
        net_pnl=int(gross_pnl - fees - taxes),
        fees=int(fees),
        taxes=int(taxes),
    )


def estimate_trade_net_pnl_from_prices(
    *,
    entry_price: int,
    exit_price: int,
    quantity: int,
    commission_rate: float,
    tax_slippage_rate: float,
) -> int:
    """Return net PnL after buy/sell commission and sell-side tax/slippage."""
    return calculate_trade_pnl_from_prices(
        entry_price=entry_price,
        exit_price=exit_price,
        quantity=quantity,
        commission_rate=commission_rate,
        tax_slippage_rate=tax_slippage_rate,
    ).net_pnl


def estimate_trade_net_pnl_unrounded(
    *,
    entry_price: int,
    exit_price: int,
    quantity: int,
    commission_rate: float,
    tax_slippage_rate: float,
) -> float:
    """Return economic PnL without won-level truncation artifacts."""
    qty = max(0, int(quantity or 0))
    if qty <= 0:
        return 0.0
    gross_buy = float(max(0, int(entry_price or 0)) * qty)
    gross_sell = float(max(0, int(exit_price or 0)) * qty)
    fees = (gross_buy + gross_sell) * max(0.0, float(commission_rate))
    taxes = gross_sell * max(0.0, float(tax_slippage_rate))
    return gross_sell - gross_buy - fees - taxes


def projected_entry_price(*, current_price: int, entry_market_slippage_rate: float) -> int:
    return max(1, int(round(max(1, int(current_price or 0)) * (1.0 + float(entry_market_slippage_rate)))))


def projected_take_profit_exit_price(
    *,
    entry_price: int,
    take_profit_pct: float,
    exit_market_slippage_rate: float,
) -> int:
    return max(
        1,
        int(round(max(1, int(entry_price or 0)) * (1.0 + (float(take_profit_pct) / 100.0) - float(exit_market_slippage_rate)))),
    )


def net_stop_exit_price_for_loss(
    *,
    entry_price: int,
    quantity: int,
    stop_loss_abs: int,
    commission_rate: float,
    tax_slippage_rate: float,
) -> int:
    """Find the highest exit price whose net PnL is at or below stop_loss_abs."""
    qty = max(0, int(quantity or 0))
    if qty <= 0:
        return max(1, int(entry_price or 1))
    stop_loss = max(1, int(stop_loss_abs or 0))
    entry = max(1, int(entry_price or 1))
    if estimate_trade_net_pnl_from_prices(
        entry_price=entry,
        exit_price=entry,
        quantity=qty,
        commission_rate=commission_rate,
        tax_slippage_rate=tax_slippage_rate,
    ) <= -stop_loss:
        return entry

    low = 1
    high = entry
    best = low
    while low <= high:
        mid = (low + high) // 2
        net_pnl = estimate_trade_net_pnl_from_prices(
            entry_price=entry,
            exit_price=mid,
            quantity=qty,
            commission_rate=commission_rate,
            tax_slippage_rate=tax_slippage_rate,
        )
        if net_pnl <= -stop_loss:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return max(1, int(best))


def expected_entry_outcome_metrics(
    *,
    current_price: int,
    quantity: int,
    take_profit_pct: float,
    stop_loss_abs: int,
    entry_market_slippage_rate: float,
    exit_market_slippage_rate: float,
    commission_rate: float,
    tax_slippage_rate: float,
    expected_stop_uses_net_cap: bool,
) -> Dict[str, float]:
    """Project entry, target exit, stop exit, and their net PnL with one cost model."""
    qty = max(0, int(quantity or 0))
    entry_price = projected_entry_price(
        current_price=current_price,
        entry_market_slippage_rate=entry_market_slippage_rate,
    )
    target_exit_price = projected_take_profit_exit_price(
        entry_price=entry_price,
        take_profit_pct=take_profit_pct,
        exit_market_slippage_rate=exit_market_slippage_rate,
    )
    stop_amount = max(1, int(stop_loss_abs or 0))
    if expected_stop_uses_net_cap:
        stop_exit_price = net_stop_exit_price_for_loss(
            entry_price=entry_price,
            quantity=qty,
            stop_loss_abs=stop_amount,
            commission_rate=commission_rate,
            tax_slippage_rate=tax_slippage_rate,
        )
    else:
        stop_exit_price = max(
            1,
            int(round(((entry_price * max(1, qty)) - stop_amount) / max(1, qty))),
        )
        stop_exit_price = max(1, int(round(stop_exit_price * (1.0 - float(exit_market_slippage_rate)))))

    target_net = estimate_trade_net_pnl_from_prices(
        entry_price=entry_price,
        exit_price=target_exit_price,
        quantity=qty,
        commission_rate=commission_rate,
        tax_slippage_rate=tax_slippage_rate,
    )
    stop_net = estimate_trade_net_pnl_from_prices(
        entry_price=entry_price,
        exit_price=stop_exit_price,
        quantity=qty,
        commission_rate=commission_rate,
        tax_slippage_rate=tax_slippage_rate,
    )
    return {
        "entry_price": float(entry_price),
        "target_exit_price": float(target_exit_price),
        "stop_exit_price": float(stop_exit_price),
        "target_net_pnl": float(target_net),
        "stop_net_pnl": float(stop_net),
    }
