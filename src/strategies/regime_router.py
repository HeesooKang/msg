from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from src.models import Order, Quote

if TYPE_CHECKING:
    from src.strategies.momentum_scalp import MomentumScalpStrategy

class RegimeStrategyRouter:
    def evaluate_long_exit(
        self,
        strategy: "MomentumScalpStrategy",
        quote: Quote,
    ) -> Optional[Order]:
        pos = strategy.positions.get(quote.symbol)
        if pos is None:
            return None
        return strategy._default_long_exit(quote)
