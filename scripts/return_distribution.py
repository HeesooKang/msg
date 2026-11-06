"""Offline conditional net-return distributions (Meinshausen, JMLR, 2006).

Leaf outcomes, not the dispersion of tree means, define the loss probability.
Fitting is offline; this module neither connects to a broker nor places orders.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import sqrt

import numpy as np
from sklearn.ensemble import RandomForestRegressor


@dataclass(frozen=True)
class ReturnEstimate:
    expected_return: float
    loss_probability: float
    lower_return: float


class _OutcomeForest:
    def _fit_outcomes(self, rows, x, y, weights, *, clusters=None):
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("non_finite_training_data")
        mass = weights if clusters is None else np.array(list(clusters.values()))
        mass = mass / mass.sum()
        effective_n = 1 / np.square(mass).sum()
        self.forest = RandomForestRegressor(n_estimators=64, random_state=0,
            bootstrap=False, max_features="sqrt",
            min_weight_fraction_leaf=min(.5, 1 / sqrt(effective_n)), n_jobs=1)
        self.forest.fit(x, y, sample_weight=weights)
        self.leaves = []
        for column in self.forest.apply(x).T:
            nodes = {}
            for leaf in np.unique(column):
                indices = np.flatnonzero(column == leaf)
                nodes[int(leaf)] = (indices, weights[indices] / weights[indices].sum())
            self.leaves.append(nodes)
        self.training_count = len(rows)
        self.training_through = max(r["exit_at"] for r in rows)

    def _outcome_weights(self, features):
        x = np.asarray(features, dtype=float)
        if not np.isfinite(x).all():
            raise ValueError("non_finite_features")
        for leaf_ids in self.forest.apply(x):
            mass = np.zeros(self.training_count)
            for nodes, leaf in zip(self.leaves, leaf_ids):
                indices, weights = nodes[int(leaf)]
                mass[indices] += weights / len(self.leaves)
            yield mass


class ReturnDistribution(_OutcomeForest):
    def __init__(self, *, price_move: bool = False):
        self.price_move = price_move

    @staticmethod
    def _price_features(row: dict) -> list[float]:
        values = list(row["features"])
        # Taxes are applied exactly to scenarios, not inferred from stock identity.
        values[14] = 0.
        return values

    def fit(self, observations: list[dict], *, before: str) -> "ReturnDistribution":
        # Filter BEFORE computing weights or fitting: later rows cannot affect a model.
        rows = [r for r in observations if r.get("status") == "complete"
                and r["exit_at"] < before and r["signal_at"] < before]
        if not rows:
            raise ValueError("no_completed_training_outcomes")
        counts = Counter((r["day"], r["symbol"]) for r in rows)
        symbols = Counter(day for day, _ in counts)
        weights = np.array([1 / (symbols[r["day"]] * counts[r["day"], r["symbol"]])
                            for r in rows])
        weights /= weights.sum()
        net_returns = np.array([r["net_return"] for r in rows], dtype=float)
        x = np.array([self._price_features(r) if self.price_move else r["features"]
                      for r in rows], dtype=float)
        self.y = net_returns.copy()
        if self.price_move:
            # Invert the exact unrounded cost equation; learn bid-to-bid movement,
            # then apply the query's known spread/tax/commission only once.
            self.y = np.array([((r["net_return"] + 1 + r["commission"])
                / (1 - r["commission"] - r["tax"]) / (1 - r["features"][6]) - 1)
                for r in rows])
        self._fit_outcomes(rows, x, self.y, weights)
        self.order = np.argsort(self.y)
        self.baseline_mean = float(weights @ net_returns)
        self.baseline_loss = float(weights[net_returns < 0].sum())
        return self

    def predict(self, features: list[list[float]], *, contexts: list[dict] | None = None) -> list[ReturnEstimate]:
        if not features:
            return []
        if self.price_move:
            if contexts is None or len(contexts) != len(features):
                raise ValueError("price_context_required")
            features = [self._price_features(row) for row in contexts]
        result = []
        for number, mass in enumerate(self._outcome_weights(features)):
            cumulative = np.cumsum(mass[self.order])
            index = min(len(self.order) - 1, int(np.searchsorted(cumulative, .1)))
            outcomes = self.y
            if self.price_move:
                row = contexts[number]
                bid_ratio = 1 - row["features"][6]
                # Volatility is an input, not a multiplier that invents larger
                # future returns outside the observed outcome distribution.
                exit_ratio = np.maximum(0., bid_ratio * (1 + self.y))
                outcomes = exit_ratio * (1 - row["commission"] - row["tax"]) - (1 + row["commission"])
            result.append(ReturnEstimate(float(mass @ outcomes),
                float(mass[outcomes < 0].sum()), float(outcomes[self.order[index]])))
        return result


@dataclass(frozen=True)
class ExecutionEstimate:
    expected_return: float
    fill_probability: float
    expected_return_if_filled: float | None
    loss_probability_if_filled: float | None
    lower_return_if_filled: float | None


class ExecutionReturnDistribution(_OutcomeForest):
    """Joint fill/entry/exit scenarios, conditional on observable executions.

    Unknown outcomes are NOT no-fills. Quote replay is not a broker-fill model;
    this research estimator must not be described as validated live execution.
    """

    @staticmethod
    def _features(row):
        values = list(row["features"])
        scale = float(row["price_scale"])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("invalid_price_scale")
        # Retain absolute volatility: equal standardized patterns need not have
        # equal executable gains. Known fees/taxes are applied to scenarios.
        for index in (0, 1, 2, 3, 5, 6, 10, 11, 12, 13, 19):
            values[index] /= scale
        values[14] = 0.
        return values

    def fit(self, observations, *, before):
        unique = {}
        for row in observations:
            if (row.get("cohort") != "sample" or row.get("status") != "complete"
                    or row["exit_at"] >= before or row["signal_at"] >= before):
                continue
            key = (row["day"], row["symbol"], row["signal_at"], row["entry_limit"])
            if key in unique and unique[key] != row:
                raise ValueError("conflicting_execution_outcomes")
            unique[key] = row
        rows = list(unique.values())
        if not rows:
            raise ValueError("no_completed_training_outcomes")
        counts = Counter((r["day"], r["symbol"]) for r in rows)
        days = Counter(day for day, _ in counts)
        weights = np.array([1 / (days[r["day"]] * counts[r["day"], r["symbol"]]) for r in rows])
        weights /= weights.sum()
        self.filled = np.array([r["filled"] for r in rows], dtype=bool)
        self.entry_changes = np.zeros(len(rows))
        self.exit_changes = np.zeros(len(rows))
        gross = np.zeros(len(rows))
        clusters = defaultdict(float)
        for i, (row, weight) in enumerate(zip(rows, weights)):
            bid = row["entry"] * (1 - row["features"][6])
            scale = bid * row["price_scale"]
            if not np.isfinite(scale) or scale <= 0 or row["entry_limit"] <= 0:
                raise ValueError("invalid_execution_prices")
            clusters[row["day"], row["symbol"], row["signal_at"]] += weight
            if self.filled[i]:
                if not 0 < row["fill_price"] <= row["entry_limit"] or row["exit_price"] <= 0:
                    raise ValueError("invalid_execution_prices")
                self.entry_changes[i] = (row["fill_price"] - row["entry_limit"]) / scale
                self.exit_changes[i] = (row["exit_price"] - bid) / scale
                gross[i] = (row["exit_price"] - row["fill_price"]) / scale
        x = np.asarray([self._features(r) for r in rows])
        # Non-fills participate in fitting as zero cash flow, not just in a
        # separate probability multiplier fitted on a different population.
        self._fit_outcomes(rows, x, gross, weights, clusters=clusters)
        return self

    def predict(self, rows):
        if not rows:
            return []
        result = []
        for row, mass in zip(rows, self._outcome_weights([self._features(r) for r in rows])):
            limit = row["entry_limit"]
            bid = row["entry"] * (1 - row["features"][6])
            scale = bid * row["price_scale"]
            if not 0 < bid <= row["entry"] <= limit:
                raise ValueError("invalid_execution_prices")
            # Keep each entry and exit paired. A cheap fill must not invent a
            # higher exit, or erase the adverse move that made that fill possible.
            buys = np.clip(limit + scale * self.entry_changes, 1., limit)
            sells = np.maximum(0., bid + scale * self.exit_changes)
            net = (sells * (1-row["commission"]-row["tax"])
                   - buys * (1+row["commission"])) / limit
            net[~self.filled] = 0.
            fill_probability = float(mass[self.filled].sum())
            if fill_probability <= 0:
                result.append(ExecutionEstimate(0., 0., None, None, None))
                continue
            conditional = mass[self.filled] / fill_probability
            outcomes = net[self.filled]
            order = np.argsort(outcomes)
            index = min(len(order)-1, int(np.searchsorted(np.cumsum(conditional[order]), .1)))
            result.append(ExecutionEstimate(float(mass @ net), fill_probability,
                float(conditional @ outcomes), float(conditional[outcomes < 0].sum()),
                float(outcomes[order[index]])))
        return result
