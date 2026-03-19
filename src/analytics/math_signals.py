from __future__ import annotations

import json
from dataclasses import dataclass
from math import exp, log1p
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.models import Quote


DAILY_REPORT_PREFIX = "daily-scorecard."


@dataclass
class LeaderSignal:
    symbol: str
    leader_score: float
    leader_percentile: float
    entry_grade: str
    change_rate: float
    trade_amount: int
    vs_open_pct: float
    high_proximity: float
    volume_vs_avg: float
    reclaim_speed_ticks: int


@dataclass
class RegimeProbabilities:
    bull_prob: float
    neutral_prob: float
    soft_bear_prob: float
    bear_prob: float

    def dominant_profile(self) -> str:
        ranked = [
            ("bull", self.bull_prob),
            ("neutral", self.neutral_prob),
            ("soft_bear", self.soft_bear_prob),
            ("bear", self.bear_prob),
        ]
        ranked.sort(key=lambda item: (item[1], item[0]), reverse=True)
        return ranked[0][0]


@dataclass
class ExpectedValueEstimate:
    strategy_name: str
    regime_label: str
    hour_bucket: str
    entry_grade: str
    entry_ev: float
    p_win: float
    confidence: str
    closed_trades: int


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _robust_z_scores(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    if len(values) == 1:
        return [0.0]

    center = median(values)
    deviations = [abs(value - center) for value in values]
    mad = median(deviations)
    if mad > 0:
        scale = 1.4826 * mad
        if scale > 0:
            return [round((value - center) / scale, 6) for value in values]

    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
    std = variance ** 0.5
    if std <= 0:
        return [0.0 for _ in values]
    return [round((value - mean) / std, 6) for value in values]


def _recent_reclaim_speed_ticks(recent_quotes: Sequence[Quote]) -> int:
    prices = [int(item.current_price) for item in recent_quotes if int(item.current_price) > 0]
    if len(prices) < 4:
        return 99

    prior_prices = prices[:-1]
    local_high = max(prior_prices)
    local_high_idx = max(idx for idx, price in enumerate(prior_prices) if price == local_high)
    pullback_window = prior_prices[local_high_idx + 1 :]
    if not pullback_window:
        return 99

    pullback_low = min(pullback_window)
    pullback_low_idx = max(idx for idx, price in enumerate(prior_prices) if idx > local_high_idx and price == pullback_low)
    reclaim_ticks = max(1, len(prices) - 1 - pullback_low_idx)
    return reclaim_ticks


def build_leader_signals(
    quotes: Sequence[Quote],
    *,
    avg_volumes: Optional[Dict[str, int]] = None,
    recent_quotes_by_symbol: Optional[Dict[str, Sequence[Quote]]] = None,
) -> Dict[str, LeaderSignal]:
    regular_quotes = [quote for quote in quotes if int(getattr(quote, "current_price", 0) or 0) > 0]
    if not regular_quotes:
        return {}

    avg_volumes = avg_volumes or {}
    recent_quotes_by_symbol = recent_quotes_by_symbol or {}

    features: List[Dict[str, Any]] = []
    for quote in regular_quotes:
        current_price = int(quote.current_price or 0)
        open_price = int(quote.open_price or 0)
        high_price = int(quote.high_price or 0)
        low_price = int(quote.low_price or 0)
        trade_amount = int(getattr(quote, "trade_amount", 0) or 0)
        if trade_amount <= 0:
            trade_amount = max(0, current_price) * max(0, int(quote.volume or 0))
        vs_open_pct = ((current_price - open_price) / open_price * 100) if open_price > 0 else 0.0
        price_range = max(1, high_price - low_price)
        high_proximity = _clip((current_price - low_price) / price_range, 0.0, 1.0) if high_price > 0 else 0.0
        avg_volume = int(avg_volumes.get(quote.symbol, 0) or 0)
        volume_vs_avg = (int(quote.volume or 0) / avg_volume) if avg_volume > 0 else 1.0
        reclaim_speed_ticks = _recent_reclaim_speed_ticks(recent_quotes_by_symbol.get(quote.symbol, []))
        features.append(
            {
                "symbol": quote.symbol,
                "quote": quote,
                "change_rate": float(quote.change_rate or 0.0),
                "trade_amount_log": log1p(max(0, trade_amount)),
                "trade_amount": trade_amount,
                "vs_open_pct": vs_open_pct,
                "high_proximity": high_proximity,
                "volume_vs_avg": volume_vs_avg,
                "reclaim_speed_feature": float(-reclaim_speed_ticks),
                "reclaim_speed_ticks": reclaim_speed_ticks,
            }
        )

    z_change = _robust_z_scores([item["change_rate"] for item in features])
    z_trade = _robust_z_scores([item["trade_amount_log"] for item in features])
    z_vs_open = _robust_z_scores([item["vs_open_pct"] for item in features])
    z_high = _robust_z_scores([item["high_proximity"] for item in features])
    z_volume = _robust_z_scores([item["volume_vs_avg"] for item in features])
    z_reclaim = _robust_z_scores([item["reclaim_speed_feature"] for item in features])

    raw_scores: List[Tuple[str, float]] = []
    for idx, item in enumerate(features):
        score = (
            0.25 * z_change[idx]
            + 0.25 * z_trade[idx]
            + 0.15 * z_vs_open[idx]
            + 0.10 * z_high[idx]
            + 0.15 * z_volume[idx]
            + 0.10 * z_reclaim[idx]
        )
        raw_scores.append((item["symbol"], score))

    ranked_scores = sorted(raw_scores, key=lambda item: (item[1], item[0]), reverse=True)
    total = len(ranked_scores)
    percentiles = {
        symbol: (1.0 if total <= 1 else (total - rank) / (total - 1))
        for rank, (symbol, _) in enumerate(ranked_scores, start=1)
    }
    score_map = {symbol: score for symbol, score in raw_scores}

    signals: Dict[str, LeaderSignal] = {}
    for item in features:
        symbol = item["symbol"]
        percentile = percentiles.get(symbol, 0.0)
        if percentile >= 0.85:
            entry_grade = "A"
        elif percentile >= 0.60:
            entry_grade = "B"
        else:
            entry_grade = "C"
        signals[symbol] = LeaderSignal(
            symbol=symbol,
            leader_score=round(score_map.get(symbol, 0.0), 6),
            leader_percentile=round(percentile, 6),
            entry_grade=entry_grade,
            change_rate=round(item["change_rate"], 6),
            trade_amount=int(item["trade_amount"]),
            vs_open_pct=round(item["vs_open_pct"], 6),
            high_proximity=round(item["high_proximity"], 6),
            volume_vs_avg=round(item["volume_vs_avg"], 6),
            reclaim_speed_ticks=int(item["reclaim_speed_ticks"]),
        )
    return signals


def compute_regime_probabilities(
    *,
    index_gap_ma20_pct: float,
    index_gap_ma5_pct: float,
    avg_change: float,
    decliner_ratio: float,
    strong_leader_count: int,
    strong_leader_avg_score: float,
) -> RegimeProbabilities:
    gap20 = _clip(index_gap_ma20_pct / 3.0, -2.5, 2.5)
    gap5 = _clip(index_gap_ma5_pct / 2.0, -2.5, 2.5)
    avg_change_norm = _clip(avg_change / 2.0, -2.5, 2.5)
    decliner_norm = _clip((decliner_ratio - 0.5) / 0.2, -2.5, 2.5)
    leader_count_norm = _clip(strong_leader_count / 3.0, 0.0, 2.5)
    leader_score_norm = _clip(strong_leader_avg_score / 2.5, -2.5, 2.5)

    bull_logit = (
        1.2 * gap20
        + 0.8 * gap5
        + 0.7 * avg_change_norm
        + 0.6 * leader_count_norm
        + 0.5 * leader_score_norm
        - 1.0 * decliner_norm
    )
    neutral_logit = (
        0.4
        - 0.5 * abs(avg_change_norm)
        - 0.4 * abs(gap5)
        - 0.2 * abs(decliner_norm)
    )
    soft_bear_logit = (
        0.8 * decliner_norm
        - 0.5 * avg_change_norm
        - 0.3 * leader_count_norm
        - 0.2 * gap20
    )
    bear_logit = (
        1.0 * decliner_norm
        - 0.8 * avg_change_norm
        - 0.7 * gap20
        - 0.4 * leader_score_norm
        - 0.4 * gap5
    )
    logits = [bull_logit, neutral_logit, soft_bear_logit, bear_logit]
    max_logit = max(logits)
    exp_values = [exp(value - max_logit) for value in logits]
    denom = sum(exp_values) or 1.0
    probs = [value / denom for value in exp_values]
    return RegimeProbabilities(
        bull_prob=round(probs[0], 6),
        neutral_prob=round(probs[1], 6),
        soft_bear_prob=round(probs[2], 6),
        bear_prob=round(probs[3], 6),
    )


def load_recent_scorecards(report_root: Path, window_days: int) -> List[Dict[str, Any]]:
    files = sorted(report_root.rglob(f"{DAILY_REPORT_PREFIX}*.json"), key=lambda path: path.name)
    if window_days > 0:
        files = files[-window_days:]
    scorecards: List[Dict[str, Any]] = []
    for path in files:
        try:
            scorecards.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    scorecards.sort(key=lambda item: item.get("date", ""))
    return scorecards


def build_entry_ev_table(
    scorecards: Sequence[Dict[str, Any]],
    *,
    window_days: int = 5,
    min_samples: int = 4,
) -> Dict[Tuple[str, str, str, str], ExpectedValueEstimate]:
    if not scorecards:
        return {}
    window_cards = list(scorecards[-window_days:]) if window_days > 0 else list(scorecards)

    bucket_rollup: Dict[Tuple[str, str, str, str], Dict[str, float]] = {}
    strategy_rollup: Dict[str, Dict[str, float]] = {}

    for card in window_cards:
        trade_records = (card.get("log_analysis") or {}).get("trade_records") or []
        for record in trade_records:
            strategy_name = str(record.get("strategy_name") or "unknown_strategy")
            regime_label = str(record.get("regime_label") or "unknown")
            hour_bucket = str(record.get("hour_bucket") or "unknown")
            entry_grade = str(record.get("entry_grade_math") or "unknown")
            net_pnl = _safe_float(record.get("net_pnl"))

            key = (strategy_name, regime_label, hour_bucket, entry_grade)
            bucket = bucket_rollup.setdefault(
                key,
                {"trades": 0, "wins": 0, "sum_win": 0.0, "losses": 0, "sum_loss": 0.0},
            )
            bucket["trades"] += 1
            if net_pnl > 0:
                bucket["wins"] += 1
                bucket["sum_win"] += net_pnl
            elif net_pnl < 0:
                bucket["losses"] += 1
                bucket["sum_loss"] += net_pnl

            strategy_bucket = strategy_rollup.setdefault(
                strategy_name,
                {"trades": 0, "wins": 0, "sum_win": 0.0, "losses": 0, "sum_loss": 0.0},
            )
            strategy_bucket["trades"] += 1
            if net_pnl > 0:
                strategy_bucket["wins"] += 1
                strategy_bucket["sum_win"] += net_pnl
            elif net_pnl < 0:
                strategy_bucket["losses"] += 1
                strategy_bucket["sum_loss"] += net_pnl

    estimates: Dict[Tuple[str, str, str, str], ExpectedValueEstimate] = {}
    for key, bucket in bucket_rollup.items():
        strategy_name, regime_label, hour_bucket, entry_grade = key
        strategy_stats = strategy_rollup.get(strategy_name, {})
        trades = int(bucket["trades"])
        wins = int(bucket["wins"])
        losses = int(bucket["losses"])

        bucket_avg_win = (bucket["sum_win"] / wins) if wins > 0 else 0.0
        bucket_avg_loss = (bucket["sum_loss"] / losses) if losses > 0 else 0.0
        strategy_avg_win = (
            strategy_stats["sum_win"] / strategy_stats["wins"]
            if strategy_stats.get("wins", 0) > 0
            else 0.0
        )
        strategy_avg_loss = (
            strategy_stats["sum_loss"] / strategy_stats["losses"]
            if strategy_stats.get("losses", 0) > 0
            else 0.0
        )

        shrink_weight = min(1.0, trades / max(1, min_samples))
        avg_win = (bucket_avg_win * shrink_weight) + (strategy_avg_win * (1 - shrink_weight))
        avg_loss = (bucket_avg_loss * shrink_weight) + (strategy_avg_loss * (1 - shrink_weight))
        p_win = (wins + 1) / (trades + 2)
        entry_ev = (p_win * avg_win) + ((1 - p_win) * avg_loss)

        if trades < min_samples:
            confidence = "low"
        elif trades < (min_samples * 2):
            confidence = "medium"
        else:
            confidence = "high"

        estimates[key] = ExpectedValueEstimate(
            strategy_name=strategy_name,
            regime_label=regime_label,
            hour_bucket=hour_bucket,
            entry_grade=entry_grade,
            entry_ev=round(entry_ev, 2),
            p_win=round(p_win, 6),
            confidence=confidence,
            closed_trades=trades,
        )

    return estimates


def estimate_entry_ev(
    ev_table: Dict[Tuple[str, str, str, str], ExpectedValueEstimate],
    *,
    strategy_name: str,
    regime_label: str,
    hour_bucket: str,
    entry_grade: str,
) -> ExpectedValueEstimate:
    if not ev_table:
        return ExpectedValueEstimate(
            strategy_name=strategy_name,
            regime_label=regime_label,
            hour_bucket=hour_bucket,
            entry_grade=entry_grade,
            entry_ev=0.0,
            p_win=0.5,
            confidence="none",
            closed_trades=0,
        )

    lookup_order = [
        (strategy_name, regime_label, hour_bucket, entry_grade),
        (strategy_name, regime_label, hour_bucket, "unknown"),
        (strategy_name, regime_label, "unknown", entry_grade),
        (strategy_name, regime_label, "unknown", "unknown"),
        (strategy_name, "unknown", hour_bucket, entry_grade),
        (strategy_name, "unknown", "unknown", entry_grade),
        (strategy_name, "unknown", "unknown", "unknown"),
    ]
    for key in lookup_order:
        if key in ev_table:
            return ev_table[key]

    return ExpectedValueEstimate(
        strategy_name=strategy_name,
        regime_label=regime_label,
        hour_bucket=hour_bucket,
        entry_grade=entry_grade,
        entry_ev=0.0,
        p_win=0.5,
        confidence="none",
        closed_trades=0,
    )
