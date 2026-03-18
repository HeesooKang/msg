from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, TYPE_CHECKING

from src.models import Order, Quote

if TYPE_CHECKING:
    from src.strategies.momentum_scalp import MomentumScalpStrategy


@dataclass
class EntryDecision:
    allowed: bool
    strategy_name: str = ""
    setup_name: str = ""
    payload: str = ""
    reject_reason: str = ""


class RegimeSubStrategy:
    name = "unknown_strategy"
    profile_name = "unknown"

    def evaluate_long_entry(
        self,
        strategy: "MomentumScalpStrategy",
        quote: Quote,
        score: float,
    ) -> EntryDecision:
        return EntryDecision(False, strategy_name=self.name, reject_reason="long_unsupported")

    def evaluate_inverse_entry(
        self,
        strategy: "MomentumScalpStrategy",
        quote: Quote,
        score: float,
    ) -> EntryDecision:
        return EntryDecision(False, strategy_name=self.name, reject_reason="inverse_unsupported")

    def evaluate_long_exit(
        self,
        strategy: "MomentumScalpStrategy",
        quote: Quote,
    ) -> Optional[Order]:
        return strategy._default_long_exit(quote)

    def evaluate_inverse_exit(
        self,
        strategy: "MomentumScalpStrategy",
        quote: Quote,
    ) -> Optional[Order]:
        return strategy._default_inverse_exit(quote)


class BullBreakoutSubStrategy(RegimeSubStrategy):
    name = "bull_breakout_strategy"
    profile_name = "bull"

    def evaluate_long_entry(
        self,
        strategy: "MomentumScalpStrategy",
        quote: Quote,
        score: float,
    ) -> EntryDecision:
        ok, setup_name, payload = strategy._passes_bull_breakout_setup(quote, score)
        return EntryDecision(
            allowed=ok,
            strategy_name=self.name,
            setup_name=setup_name,
            payload=payload,
            reject_reason=payload if not ok else "",
        )


class NeutralPullbackSubStrategy(RegimeSubStrategy):
    name = "neutral_pullback_strategy"
    profile_name = "neutral"

    def evaluate_long_entry(
        self,
        strategy: "MomentumScalpStrategy",
        quote: Quote,
        score: float,
    ) -> EntryDecision:
        ok, setup_name, payload = strategy._passes_neutral_pullback_reclaim_setup(quote, score)
        return EntryDecision(
            allowed=ok,
            strategy_name=self.name,
            setup_name=setup_name,
            payload=payload,
            reject_reason=payload if not ok else "",
        )


class SoftBearInverseSubStrategy(RegimeSubStrategy):
    name = "soft_bear_inverse_strategy"
    profile_name = "soft_bear"

    def evaluate_inverse_entry(
        self,
        strategy: "MomentumScalpStrategy",
        quote: Quote,
        score: float,
    ) -> EntryDecision:
        ok, setup_name, payload = strategy._passes_soft_bear_inverse_setup(quote, score)
        return EntryDecision(
            allowed=ok,
            strategy_name=self.name,
            setup_name=setup_name,
            payload=payload,
            reject_reason=payload if not ok else "",
        )


class HardBearInverseSubStrategy(RegimeSubStrategy):
    name = "hard_bear_inverse_strategy"
    profile_name = "bear"

    def evaluate_inverse_entry(
        self,
        strategy: "MomentumScalpStrategy",
        quote: Quote,
        score: float,
    ) -> EntryDecision:
        ok, setup_name, payload = strategy._passes_hard_bear_inverse_setup(quote, score)
        return EntryDecision(
            allowed=ok,
            strategy_name=self.name,
            setup_name=setup_name,
            payload=payload,
            reject_reason=payload if not ok else "",
        )


class RegimeStrategyRouter:
    def __init__(self):
        self._profile_strategies: Dict[str, RegimeSubStrategy] = {
            "bull": BullBreakoutSubStrategy(),
            "neutral": NeutralPullbackSubStrategy(),
            "soft_bear": SoftBearInverseSubStrategy(),
            "bear": HardBearInverseSubStrategy(),
        }
        self._name_strategies: Dict[str, RegimeSubStrategy] = {
            strategy.name: strategy for strategy in self._profile_strategies.values()
        }

    def strategy_for_profile(self, profile_name: str) -> RegimeSubStrategy:
        return self._profile_strategies.get(
            profile_name,
            self._profile_strategies["neutral"],
        )

    def strategy_for_position(
        self,
        strategy_name: str,
        regime_label: str,
        fallback_profile: str,
    ) -> RegimeSubStrategy:
        if strategy_name and strategy_name in self._name_strategies:
            return self._name_strategies[strategy_name]
        if regime_label and regime_label in self._profile_strategies:
            return self._profile_strategies[regime_label]
        return self.strategy_for_profile(fallback_profile)

    def evaluate_long_entry(
        self,
        strategy: "MomentumScalpStrategy",
        quote: Quote,
        score: float,
    ) -> EntryDecision:
        strategy_name = strategy._current_profile_entry_strategy_name(is_inverse=False)
        if strategy_name and strategy_name in self._name_strategies:
            return self._name_strategies[strategy_name].evaluate_long_entry(strategy, quote, score)
        profile_name = strategy._resolve_regime_profile_name()
        return self.strategy_for_profile(profile_name).evaluate_long_entry(strategy, quote, score)

    def evaluate_inverse_entry(
        self,
        strategy: "MomentumScalpStrategy",
        quote: Quote,
        score: float,
    ) -> EntryDecision:
        profile_name = strategy._resolve_regime_profile_name()
        return self.strategy_for_profile(profile_name).evaluate_inverse_entry(strategy, quote, score)

    def evaluate_long_exit(
        self,
        strategy: "MomentumScalpStrategy",
        quote: Quote,
    ) -> Optional[Order]:
        pos = strategy.positions.get(quote.symbol)
        profile_name = strategy._resolve_regime_profile_name()
        if pos is None:
            return None
        return self.strategy_for_position(
            getattr(pos, "entry_strategy_name", ""),
            getattr(pos, "regime_label", ""),
            profile_name,
        ).evaluate_long_exit(strategy, quote)

    def evaluate_inverse_exit(
        self,
        strategy: "MomentumScalpStrategy",
        quote: Quote,
    ) -> Optional[Order]:
        pos = strategy.positions.get(quote.symbol)
        profile_name = strategy._resolve_regime_profile_name()
        if pos is None:
            return None
        return self.strategy_for_position(
            getattr(pos, "entry_strategy_name", ""),
            getattr(pos, "regime_label", ""),
            profile_name,
        ).evaluate_inverse_exit(strategy, quote)
