from __future__ import annotations

from dataclasses import dataclass


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
    sell_tax_rate: float,
) -> TradePnL:
    """Calculate the won-level result with the same costs used by live fills."""
    qty = max(0, int(quantity or 0))
    if qty <= 0:
        return TradePnL(gross_pnl=0, net_pnl=0, fees=0, taxes=0)
    buy_amount = max(0, int(entry_price or 0)) * qty
    sell_amount = max(0, int(exit_price or 0)) * qty
    tax_rate = float(sell_tax_rate)
    gross_pnl = sell_amount - buy_amount
    fees = int((buy_amount + sell_amount) * max(0.0, float(commission_rate)))
    taxes = int(sell_amount * max(0.0, tax_rate))
    return TradePnL(
        gross_pnl=gross_pnl,
        net_pnl=gross_pnl - fees - taxes,
        fees=fees,
        taxes=taxes,
    )


def estimate_trade_net_pnl_unrounded(
    *,
    entry_price: int,
    exit_price: int,
    quantity: int,
    commission_rate: float,
    sell_tax_rate: float,
) -> float:
    qty = max(0, int(quantity or 0))
    buy_amount = float(max(0, int(entry_price or 0)) * qty)
    sell_amount = float(max(0, int(exit_price or 0)) * qty)
    fees = (buy_amount + sell_amount) * max(0.0, float(commission_rate))
    taxes = sell_amount * max(0.0, float(sell_tax_rate))
    return sell_amount - buy_amount - fees - taxes
