import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_REPORT_ROOT = Path("reports")
DEFAULT_LOG_ROOT = Path("logs")
DAILY_REPORT_PREFIX = "daily-scorecard."
READINESS_REPORT_JSON = "real-trade-readiness.json"
READINESS_REPORT_MD = "real-trade-readiness.md"
STRATEGY_GATE_REPORT_JSON = "strategy-gates.json"
STRATEGY_GATE_REPORT_MD = "strategy-gates.md"
MATH_SHADOW_REPORT_JSON = "math-shadow-report.json"
MATH_SHADOW_REPORT_MD = "math-shadow-report.md"
PAPER_GATE_WINDOW_DAYS = 5
PAPER_GATE_MIN_POSITIVE_DAYS = 3
PAPER_GATE_MIN_TOTAL_NET_PNL = 10_000
PAPER_GATE_DAILY_LOSS_LIMIT = -5_000
PAPER_GATE_DAILY_TARGET = 10_000
DEFAULT_STRATEGY_GATE_WINDOW_DAYS = 5
DEFAULT_STRATEGY_GATE_MIN_CLOSED_TRADES = 4
REAL_MONEY_STAGE_RULES = {
    1: {
        "label": "stage1",
        "capital_scale": 0.25,
        "days_required": 5,
        "daily_loss_limit": -1_250,
        "profit_protect_threshold": 2_000,
        "daily_profit_target": 2_500,
    },
    2: {
        "label": "stage2",
        "capital_scale": 0.50,
        "days_required": 5,
        "daily_loss_limit": -2_500,
        "profit_protect_threshold": 4_000,
        "daily_profit_target": 5_000,
    },
    3: {
        "label": "stage3",
        "capital_scale": 1.00,
        "days_required": 0,
        "daily_loss_limit": -5_000,
        "profit_protect_threshold": 8_000,
        "daily_profit_target": 10_000,
    },
}

_LOG_MESSAGE_RE = re.compile(r"^\d{4}-\d{2}-\d{2} [^[]+\[[A-Z]+\] [^:]+: (?P<message>.*)$")
_SETUP_NAME_RE = re.compile(r"setup_name=([a-zA-Z0-9_]+)")
_ENTRY_REASON_RE = re.compile(r"entry_reason=([a-zA-Z0-9_]+)")
_REJECT_REASON_RE = re.compile(r"reject_reason=([a-zA-Z0-9_]+)")
_REGIME_LABEL_RE = re.compile(r"regime_label=([a-zA-Z0-9_]+)")
_LONG_SIGNAL_SYMBOL_RE = re.compile(r"매수 신호:\s*.+?\(([0-9A-Z]+)\)")
_LONG_SIGNAL_SYMBOL_PLAIN_RE = re.compile(r"매수 신호:\s*([0-9A-Z]{5,6})\s+\d+주")
_INV_SIGNAL_SYMBOL_RE = re.compile(r"\[INV\]\s*매수 신호:\s*([0-9A-Z]+)")
_FULL_SELL_SYMBOL_RE = re.compile(r"(?:^|\s)(?:\[INV\]\s*)?매도 체결:\s*([0-9A-Z]+)\s")
_PARTIAL_SELL_SYMBOL_RE = re.compile(r"(?:^|\s)(?:\[INV\]\s*)?부분매도 체결:\s*([0-9A-Z]+)\s")
_CORRECTED_SELL_SYMBOL_RE = re.compile(r"(?:^|\s)(?:\[INV\]\s*)?매도 체결 정정:\s*([0-9A-Z]+)\s")
_SELL_NET_PNL_RE = re.compile(r"순손익:\s*([-\d,]+)원")
_SELL_PNL_RE = re.compile(r"손익:\s*([-\d,]+)원")
_RISK_STAGE_RE = re.compile(r"리스크 단계 전환:\s*([a-zA-Z0-9_]+)")
_SHADOW_OUTCOME_RE = re.compile(r"outcome=([a-zA-Z0-9_]+)")
_QUEUE_COUNT_RE = re.compile(r":\s*(\d+)개")


def _parse_line_timestamp(raw_line: str) -> Optional[datetime]:
    try:
        return datetime.strptime(raw_line[:19], "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_strategy_snapshot(strategy: Any) -> Dict[str, Any]:
    daily_pnl = getattr(strategy, "daily_pnl", None)
    positions = getattr(strategy, "positions", {}) or {}

    if daily_pnl is None:
        return {
            "realized_gross_pnl": 0,
            "realized_net_pnl": 0,
            "fees_paid": 0,
            "taxes_paid": 0,
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0,
            "breakeven_count": 0,
            "winning_net_pnl_sum": 0,
            "losing_net_pnl_sum": 0,
            "largest_win_net": 0,
            "largest_loss_net": 0,
            "halted": bool(getattr(strategy, "_halted", False)),
            "open_positions_count": len(positions),
        }

    return {
        "realized_gross_pnl": _safe_int(getattr(daily_pnl, "realized_gross_pnl", 0)),
        "realized_net_pnl": _safe_int(getattr(daily_pnl, "realized_net_pnl", 0)),
        "fees_paid": _safe_int(getattr(daily_pnl, "fees_paid", 0)),
        "taxes_paid": _safe_int(getattr(daily_pnl, "taxes_paid", 0)),
        "trade_count": _safe_int(getattr(daily_pnl, "trade_count", 0)),
        "win_count": _safe_int(getattr(daily_pnl, "win_count", 0)),
        "loss_count": _safe_int(getattr(daily_pnl, "loss_count", 0)),
        "breakeven_count": _safe_int(getattr(daily_pnl, "breakeven_count", 0)),
        "winning_net_pnl_sum": _safe_int(getattr(daily_pnl, "winning_net_pnl_sum", 0)),
        "losing_net_pnl_sum": _safe_int(getattr(daily_pnl, "losing_net_pnl_sum", 0)),
        "largest_win_net": _safe_int(getattr(daily_pnl, "largest_win_net", 0)),
        "largest_loss_net": _safe_int(getattr(daily_pnl, "largest_loss_net", 0)),
        "halted": bool(getattr(strategy, "_halted", False)),
        "open_positions_count": len(positions),
    }


def _compute_trade_metrics(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    trade_count = _safe_int(snapshot.get("trade_count"))
    win_count = _safe_int(snapshot.get("win_count"))
    loss_count = _safe_int(snapshot.get("loss_count"))
    winning_sum = _safe_int(snapshot.get("winning_net_pnl_sum"))
    losing_sum = _safe_int(snapshot.get("losing_net_pnl_sum"))
    realized_net_pnl = _safe_int(snapshot.get("realized_net_pnl"))

    win_rate = (win_count / trade_count) if trade_count > 0 else 0.0
    avg_net = (realized_net_pnl / trade_count) if trade_count > 0 else 0.0
    avg_win = (winning_sum / win_count) if win_count > 0 else 0.0
    avg_loss = (losing_sum / loss_count) if loss_count > 0 else 0.0

    profit_factor: Optional[float]
    if losing_sum < 0:
        profit_factor = winning_sum / abs(losing_sum) if winning_sum > 0 else 0.0
    elif winning_sum > 0:
        profit_factor = None
    else:
        profit_factor = 0.0

    return {
        "closed_trades": trade_count,
        "wins": win_count,
        "losses": loss_count,
        "breakeven": _safe_int(snapshot.get("breakeven_count")),
        "win_rate": round(win_rate, 4),
        "average_net_per_trade": round(avg_net, 2),
        "average_win_net": round(avg_win, 2),
        "average_loss_net": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 4) if profit_factor is not None else None,
    }


def build_daily_scorecard(
    strategy: Any,
    balance: Any,
    session_pnl: int,
    trading_mode: str,
    generated_at: Optional[datetime] = None,
    log_root: Path = DEFAULT_LOG_ROOT,
) -> Dict[str, Any]:
    now = generated_at or datetime.now()
    snapshot = _extract_strategy_snapshot(strategy)
    report_date = now.date().isoformat()

    scorecard = {
        "date": report_date,
        "generated_at": now.isoformat(timespec="seconds"),
        "trading_mode": trading_mode,
        "balance": {
            "total_eval_amount": _safe_int(getattr(balance, "total_eval_amount", 0)),
            "total_deposit": _safe_int(getattr(balance, "total_deposit", 0)),
            "total_profit_loss": _safe_int(getattr(balance, "total_profit_loss", 0)),
            "total_profit_rate": round(_safe_float(getattr(balance, "total_profit_rate", 0.0)), 4),
        },
        "pnl": {
            "session_pnl": _safe_int(session_pnl),
            "realized_gross_pnl": _safe_int(snapshot.get("realized_gross_pnl")),
            "realized_net_pnl": _safe_int(snapshot.get("realized_net_pnl")),
            "fees_paid": _safe_int(snapshot.get("fees_paid")),
            "taxes_paid": _safe_int(snapshot.get("taxes_paid")),
            "winning_net_pnl_sum": _safe_int(snapshot.get("winning_net_pnl_sum")),
            "losing_net_pnl_sum": _safe_int(snapshot.get("losing_net_pnl_sum")),
            "largest_win_net": _safe_int(snapshot.get("largest_win_net")),
            "largest_loss_net": _safe_int(snapshot.get("largest_loss_net")),
        },
        "trades": _compute_trade_metrics(snapshot),
        "strategy": {
            "halted": bool(snapshot.get("halted", False)),
            "open_positions_count": _safe_int(snapshot.get("open_positions_count")),
            "real_money_stage": _safe_int(getattr(strategy, "_real_money_stage", 0)),
            "capital_scale": round(_safe_float(getattr(strategy, "_capital_scale", 1.0), 1.0), 4),
        },
    }
    scorecard["log_analysis"] = analyze_trading_log(report_date=report_date, log_root=log_root)
    return _reconcile_scorecard_with_log(scorecard)


def _scorecard_paths(report_root: Path, report_date: str) -> Dict[str, Path]:
    year, month, _ = report_date.split("-")
    base_dir = report_root / year / month
    return {
        "json": base_dir / f"{DAILY_REPORT_PREFIX}{report_date}.json",
        "md": base_dir / f"{DAILY_REPORT_PREFIX}{report_date}.md",
    }


def _format_currency(value: Any) -> str:
    return f"{_safe_int(value):,}원"


def _format_ratio(value: Any) -> str:
    return f"{_safe_float(value) * 100:.1f}%"


def _format_profit_factor(value: Any) -> str:
    if value is None:
        return "무손실"
    return f"{_safe_float(value):.2f}"


def _log_candidates(log_root: Path, report_date: str) -> List[Path]:
    year, month, _ = report_date.split("-")
    return [
        log_root / year / month / f"trading.log.{report_date}",
        log_root / f"trading.log.{report_date}",
        log_root / "trading.log",
    ]


def _extract_log_message(line: str) -> str:
    match = _LOG_MESSAGE_RE.match(line.strip())
    return match.group("message") if match else line.strip()


def _extract_signal_symbol(message: str) -> Optional[str]:
    inv_match = _INV_SIGNAL_SYMBOL_RE.search(message)
    if inv_match:
        return inv_match.group(1)
    long_match = _LONG_SIGNAL_SYMBOL_RE.search(message)
    if long_match:
        return long_match.group(1)
    plain_match = _LONG_SIGNAL_SYMBOL_PLAIN_RE.search(message)
    if plain_match:
        return plain_match.group(1)
    return None


def _extract_full_sell_symbol(message: str) -> Optional[str]:
    match = _FULL_SELL_SYMBOL_RE.search(message)
    return match.group(1) if match else None


def _extract_partial_sell_symbol(message: str) -> Optional[str]:
    match = _PARTIAL_SELL_SYMBOL_RE.search(message)
    return match.group(1) if match else None


def _extract_corrected_sell_symbol(message: str) -> Optional[str]:
    match = _CORRECTED_SELL_SYMBOL_RE.search(message)
    return match.group(1) if match else None


def _extract_sell_net_pnl(message: str) -> int:
    match = _SELL_NET_PNL_RE.search(message)
    if match:
        return _safe_int(match.group(1).replace(",", ""))
    gross_match = _SELL_PNL_RE.search(message)
    if gross_match:
        return _safe_int(gross_match.group(1).replace(",", ""))
    return 0


def _extract_context_token(message: str, key: str, default: str = "") -> str:
    marker = f"{key}="
    for chunk in str(message or "").split():
        if chunk.startswith(marker):
            return chunk[len(marker):].rstrip(",)")
    return default


def _extract_context_token_any(message: str, keys: List[str], default: str = "") -> str:
    for key in keys:
        value = _extract_context_token(message, key, "")
        if value:
            return value
    return default


def _normalize_queue_source(raw: str) -> str:
    return str(raw or "").strip()


def analyze_trading_log(
    report_date: str,
    log_root: Path = DEFAULT_LOG_ROOT,
) -> Dict[str, Any]:
    lines: List[str] = []
    selected_path: Optional[Path] = None

    for candidate in _log_candidates(log_root, report_date):
        if not candidate.exists():
            continue
        try:
            filtered = [
                line.rstrip("\n")
                for line in candidate.read_text(encoding="utf-8").splitlines()
                if line.startswith(report_date)
            ]
        except OSError:
            continue
        if filtered:
            selected_path = candidate
            lines = filtered
            break

    entry_by_setup: Dict[str, int] = defaultdict(int)
    entry_by_strategy: Dict[str, int] = defaultdict(int)
    entry_by_reason: Dict[str, int] = defaultdict(int)
    entry_by_regime: Dict[str, int] = defaultdict(int)
    reject_by_reason: Dict[str, int] = defaultdict(int)
    setup_pnl: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"closed_trades": 0, "net_pnl": 0, "wins": 0, "losses": 0}
    )
    strategy_pnl: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"closed_trades": 0, "net_pnl": 0, "wins": 0, "losses": 0}
    )
    regime_pnl: Dict[str, int] = defaultdict(int)
    symbol_pnl: Dict[str, int] = defaultdict(int)
    risk_stage_transitions: Dict[str, int] = defaultdict(int)
    strategy_hourly: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: {"closed_trades": 0, "net_pnl": 0.0})
    )
    trade_records: List[Dict[str, Any]] = []
    shadow_blocked: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "total": 0,
            "outcomes": defaultdict(int),
            "by_reason": defaultdict(int),
        }
    )
    math_grade_entries: Dict[str, int] = defaultdict(int)
    math_grade_pnl: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"closed_trades": 0, "net_pnl": 0.0}
    )
    math_ev_buckets: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"closed_trades": 0, "net_pnl": 0.0}
    )
    math_shadow_gate_counts: Dict[str, int] = defaultdict(int)
    math_shadow_reason_counts: Dict[str, int] = defaultdict(int)
    regime_shadow_disagreements: Dict[str, int] = defaultdict(int)
    funnel_queue_counts: Dict[str, int] = defaultdict(int)
    precheck_reached_counts: Dict[str, int] = defaultdict(int)
    precheck_blocked_counts: Dict[str, int] = defaultdict(int)
    policy_blocked_counts: Dict[str, int] = defaultdict(int)
    router_reached_counts: Dict[str, int] = defaultdict(int)
    router_blocked_counts: Dict[str, int] = defaultdict(int)
    hard_guard_blocked_counts: Dict[str, int] = defaultdict(int)
    funnel_admission_counts: Dict[str, int] = defaultdict(int)
    conviction_tier_pnl: Dict[str, Dict[str, float]] = defaultdict(lambda: {"closed_trades": 0, "net_pnl": 0.0})
    bull_risk_mode_pnl: Dict[str, Dict[str, float]] = defaultdict(lambda: {"closed_trades": 0, "net_pnl": 0.0})
    post_loss_admission_class_pnl: Dict[str, Dict[str, float]] = defaultdict(lambda: {"closed_trades": 0, "net_pnl": 0.0})
    candidate_class_pnl: Dict[str, Dict[str, float]] = defaultdict(lambda: {"closed_trades": 0, "net_pnl": 0.0})
    live_route_pnl: Dict[str, Dict[str, float]] = defaultdict(lambda: {"closed_trades": 0, "net_pnl": 0.0})
    queue_source_pnl: Dict[str, Dict[str, float]] = defaultdict(lambda: {"closed_trades": 0, "net_pnl": 0.0})
    stop_triggered_count = 0
    stop_limit_fallback_count = 0
    stop_overshoots: List[int] = []
    active_entries: Dict[str, Dict[str, str]] = {}
    partial_sell_net_by_symbol: Dict[str, int] = {}
    trade_record_index_by_order_no: Dict[str, int] = {}
    closed_trade_flag_by_order_no: Dict[str, bool] = {}
    daily_hard_stop_triggered = False
    daily_profit_target_triggered = False
    candidate_reject_reason_outcomes: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def _record_sell_trade(
        *,
        symbol: str,
        entry_meta: Dict[str, str],
        message: str,
        line_ts: Optional[datetime],
        count_as_closed_trade: bool,
        net_pnl_override: Optional[int] = None,
        trade_outcome_net_pnl: Optional[int] = None,
    ) -> None:
        nonlocal stop_triggered_count, stop_limit_fallback_count
        strategy_name = entry_meta.get("strategy_name") or _extract_context_token(
            message,
            "strategy_name",
            "unknown_strategy",
        )
        setup_name = entry_meta.get("setup_name") or _extract_context_token(message, "setup_name", "unknown")
        regime_label = entry_meta.get("regime_label") or _extract_context_token(message, "regime_label", "unknown")
        entry_grade_math = entry_meta.get("entry_grade_math") or _extract_context_token(
            message,
            "entry_grade_math",
            "unknown",
        )
        conviction_tier = entry_meta.get("conviction_tier") or _extract_context_token(message, "conviction_tier", "none")
        bull_risk_mode = entry_meta.get("bull_risk_mode") or _extract_context_token(message, "bull_risk_mode", "normal")
        post_loss_admission_class = entry_meta.get("post_loss_admission_class") or _extract_context_token(
            message,
            "post_loss_admission_class",
            "general",
        )
        candidate_class = entry_meta.get("candidate_class") or entry_meta.get("live_candidate_class") or _extract_context_token(
            message,
            "candidate_class",
            _extract_context_token(message, "live_candidate_class", "unclassified"),
        )
        execution_mode = entry_meta.get("execution_mode") or entry_meta.get("route_mode") or _extract_context_token(
            message,
            "execution_mode",
            _extract_context_token(message, "route_mode", "live"),
        )
        live_route = entry_meta.get("live_route") or _extract_context_token(
            message,
            "live_route",
            strategy_name,
        )
        queue_source = _normalize_queue_source(
            entry_meta.get("queue_source") or _extract_context_token(message, "queue_source", "")
        ) or "unknown"
        protective_exit_mode = _extract_context_token(message, "protective_exit_mode", "")
        fill_mode = _extract_context_token(message, "fill_mode", "")
        order_no = _extract_context_token(message, "order_no", "")
        stop_overshoot_krw = _safe_int(_extract_context_token(message, "stop_overshoot_krw", "0"))
        net_pnl = _extract_sell_net_pnl(message) if net_pnl_override is None else int(net_pnl_override)
        outcome_net_pnl = net_pnl if trade_outcome_net_pnl is None else int(trade_outcome_net_pnl)
        metrics = setup_pnl[setup_name]
        metrics["net_pnl"] += net_pnl
        strategy_metrics = strategy_pnl[strategy_name]
        strategy_metrics["net_pnl"] += net_pnl
        regime_pnl[regime_label] += net_pnl
        symbol_pnl[symbol] += net_pnl
        if count_as_closed_trade:
            metrics["closed_trades"] += 1
            strategy_metrics["closed_trades"] += 1
            if outcome_net_pnl > 0:
                metrics["wins"] += 1
                strategy_metrics["wins"] += 1
            elif outcome_net_pnl < 0:
                metrics["losses"] += 1
                strategy_metrics["losses"] += 1
        if line_ts is not None:
            hour_key = f"{line_ts.hour:02d}"
            bucket = strategy_hourly[strategy_name][hour_key]
            bucket["net_pnl"] += net_pnl
            if count_as_closed_trade:
                bucket["closed_trades"] += 1
        math_grade_metrics = math_grade_pnl[entry_grade_math]
        math_grade_metrics["net_pnl"] += net_pnl
        entry_ev = _safe_float(
            entry_meta.get("entry_ev")
            or _extract_context_token(message, "entry_ev", "0")
        )
        ev_bucket = "positive" if entry_ev > 0 else "negative" if entry_ev < 0 else "zero"
        math_ev_metrics = math_ev_buckets[ev_bucket]
        math_ev_metrics["net_pnl"] += net_pnl
        conviction_metrics = conviction_tier_pnl[conviction_tier or "none"]
        conviction_metrics["net_pnl"] += net_pnl
        risk_mode_metrics = bull_risk_mode_pnl[bull_risk_mode or "normal"]
        risk_mode_metrics["net_pnl"] += net_pnl
        post_loss_metrics = post_loss_admission_class_pnl[post_loss_admission_class or "general"]
        post_loss_metrics["net_pnl"] += net_pnl
        candidate_metrics = candidate_class_pnl[candidate_class or "unclassified"]
        candidate_metrics["net_pnl"] += net_pnl
        live_route_metrics = live_route_pnl[live_route or "unknown"]
        live_route_metrics["net_pnl"] += net_pnl
        queue_metrics = queue_source_pnl[queue_source or "unknown"]
        queue_metrics["net_pnl"] += net_pnl
        if count_as_closed_trade:
            math_grade_metrics["closed_trades"] += 1
            math_ev_metrics["closed_trades"] += 1
            conviction_metrics["closed_trades"] += 1
            risk_mode_metrics["closed_trades"] += 1
            post_loss_metrics["closed_trades"] += 1
            candidate_metrics["closed_trades"] += 1
            live_route_metrics["closed_trades"] += 1
            queue_metrics["closed_trades"] += 1
        if protective_exit_mode:
            stop_triggered_count += 1
            if fill_mode == "limit_then_market":
                stop_limit_fallback_count += 1
            stop_overshoots.append(max(0, stop_overshoot_krw))
        trade_records.append(
            {
                "symbol": symbol,
                "strategy_name": strategy_name,
                "setup_name": setup_name,
                "regime_label": regime_label,
                "hour_bucket": f"{line_ts.hour:02d}" if line_ts is not None else "unknown",
                "entry_grade_math": entry_grade_math,
                "conviction_tier": conviction_tier or "none",
                "bull_risk_mode": bull_risk_mode or "normal",
                "post_loss_admission_class": post_loss_admission_class or "general",
                "candidate_class": candidate_class or "unclassified",
                "execution_mode": execution_mode or "unknown",
                "live_route": live_route or "unknown",
                "order_no": order_no or "",
                "volume_gate_threshold_used": _safe_float(entry_meta.get("volume_gate_threshold_used")),
                "conviction_score": _safe_float(
                    entry_meta.get("conviction_score", entry_meta.get("opening_conviction_score"))
                ),
                "conviction_rank": _safe_int(
                    entry_meta.get("conviction_rank", entry_meta.get("opening_conviction_rank"))
                ),
                "count_as_closed_trade": bool(count_as_closed_trade),
                "net_pnl": net_pnl,
                "trade_outcome_net_pnl": outcome_net_pnl,
                "leader_score": _safe_float(entry_meta.get("leader_score")),
                "effective_leader_score": _safe_float(entry_meta.get("effective_leader_score")),
                "leader_pct": _safe_float(entry_meta.get("leader_pct")),
                "recent_acceleration_pct": _safe_float(entry_meta.get("recent_acceleration_pct")),
                "entry_ev": entry_ev,
                "entry_ev_conf": entry_meta.get("entry_ev_conf") or "none",
                "price_prediction_return_pct": _safe_float(
                    entry_meta.get("price_prediction_return_pct")
                ),
                "price_prediction_lower_pct": _safe_float(
                    entry_meta.get("price_prediction_lower_pct")
                ),
                "price_prediction_confidence": _safe_float(
                    entry_meta.get("price_prediction_confidence")
                ),
                "entry_prediction_win_probability": _safe_float(
                    entry_meta.get("entry_prediction_win_probability")
                ),
                "queue_source": queue_source,
                "math_dominant_profile": entry_meta.get("math_dominant_profile") or "",
                "size_multiplier": _safe_float(entry_meta.get("size_multiplier"), 1.0),
                "protective_exit_mode": protective_exit_mode,
                "fill_mode": fill_mode,
                "stop_overshoot_krw": stop_overshoot_krw,
                "bull_prob": _safe_float(entry_meta.get("bull_prob")),
                "neutral_prob": _safe_float(entry_meta.get("neutral_prob")),
                "soft_bear_prob": _safe_float(entry_meta.get("soft_bear_prob")),
                "bear_prob": _safe_float(entry_meta.get("bear_prob")),
                "shock_score": _safe_float(entry_meta.get("shock_score")),
                "shock_confidence": _safe_float(entry_meta.get("shock_confidence")),
                "adaptive_take_profit_pct": _safe_float(entry_meta.get("adaptive_take_profit_pct")),
                "adaptive_stop_loss_pct": _safe_float(entry_meta.get("adaptive_stop_loss_pct")),
                "adaptive_trailing_activation_pct": _safe_float(entry_meta.get("adaptive_trailing_activation_pct")),
                "adaptive_trailing_stop_pct": _safe_float(entry_meta.get("adaptive_trailing_stop_pct")),
                "adaptive_max_hold_minutes": _safe_int(entry_meta.get("adaptive_max_hold_minutes")),
            }
        )
        if order_no:
            trade_record_index_by_order_no[order_no] = len(trade_records) - 1
            closed_trade_flag_by_order_no[order_no] = bool(count_as_closed_trade)

    def _apply_sell_trade_correction(message: str, line_ts: Optional[datetime]) -> None:
        symbol = _extract_corrected_sell_symbol(message)
        order_no = _extract_context_token(message, "order_no", "")
        if not symbol or not order_no:
            return
        record_index = trade_record_index_by_order_no.get(order_no)
        if record_index is None or record_index >= len(trade_records):
            return
        record = trade_records[record_index]
        previous_net = _safe_int(record.get("net_pnl", 0))
        previous_outcome_net = _safe_int(record.get("trade_outcome_net_pnl", previous_net))
        corrected_net = _safe_int(_extract_context_token(message, "corrected_net_pnl", str(previous_net)))
        delta_net = corrected_net - previous_net
        if delta_net == 0:
            record["order_no"] = order_no
            record["fill_mode"] = "account_reconciled_confirmed"
            return

        strategy_name = str(record.get("strategy_name", "unknown_strategy") or "unknown_strategy")
        setup_name = str(record.get("setup_name", "unknown") or "unknown")
        regime_label = str(record.get("regime_label", "unknown") or "unknown")
        entry_grade_math = str(record.get("entry_grade_math", "unknown") or "unknown")
        conviction_tier = str(record.get("conviction_tier", "none") or "none")
        bull_risk_mode = str(record.get("bull_risk_mode", "normal") or "normal")
        post_loss_admission_class = str(record.get("post_loss_admission_class", "general") or "general")
        candidate_class = str(record.get("candidate_class", "unclassified") or "unclassified")
        live_route = str(record.get("live_route", "unknown") or "unknown")
        queue_source = _normalize_queue_source(record.get("queue_source", "unknown")) or "unknown"
        hour_key = str(record.get("hour_bucket", "unknown") or "unknown")
        entry_ev = _safe_float(record.get("entry_ev"))
        ev_bucket = "positive" if entry_ev > 0 else "negative" if entry_ev < 0 else "zero"
        was_closed_trade = bool(closed_trade_flag_by_order_no.get(order_no, False))

        setup_pnl[setup_name]["net_pnl"] += delta_net
        strategy_pnl[strategy_name]["net_pnl"] += delta_net
        regime_pnl[regime_label] += delta_net
        symbol_pnl[symbol] += delta_net
        if line_ts is not None and hour_key != "unknown":
            strategy_hourly[strategy_name][hour_key]["net_pnl"] += delta_net
        math_grade_pnl[entry_grade_math]["net_pnl"] += delta_net
        math_ev_buckets[ev_bucket]["net_pnl"] += delta_net
        conviction_tier_pnl[conviction_tier]["net_pnl"] += delta_net
        bull_risk_mode_pnl[bull_risk_mode]["net_pnl"] += delta_net
        post_loss_admission_class_pnl[post_loss_admission_class]["net_pnl"] += delta_net
        candidate_class_pnl[candidate_class]["net_pnl"] += delta_net
        live_route_pnl[live_route]["net_pnl"] += delta_net
        queue_source_pnl[queue_source]["net_pnl"] += delta_net

        if was_closed_trade:
            corrected_outcome_net = previous_outcome_net + delta_net
            if previous_outcome_net > 0:
                setup_pnl[setup_name]["wins"] -= 1
                strategy_pnl[strategy_name]["wins"] -= 1
            elif previous_outcome_net < 0:
                setup_pnl[setup_name]["losses"] -= 1
                strategy_pnl[strategy_name]["losses"] -= 1
            if corrected_outcome_net > 0:
                setup_pnl[setup_name]["wins"] += 1
                strategy_pnl[strategy_name]["wins"] += 1
            elif corrected_outcome_net < 0:
                setup_pnl[setup_name]["losses"] += 1
                strategy_pnl[strategy_name]["losses"] += 1
            record["trade_outcome_net_pnl"] = corrected_outcome_net

        record["net_pnl"] = corrected_net
        record["fill_mode"] = "account_reconciled_confirmed"

    for raw_line in lines:
        message = _extract_log_message(raw_line)
        if not message:
            continue
        line_ts = _parse_line_timestamp(raw_line)

        risk_stage_match = _RISK_STAGE_RE.search(message)
        if risk_stage_match:
            risk_stage_transitions[risk_stage_match.group(1)] += 1

        if "일일 총손익 하드스탑 도달!" in message:
            daily_hard_stop_triggered = True
        if "일일 총손익 목표 달성!" in message:
            daily_profit_target_triggered = True

        reject_match = _REJECT_REASON_RE.search(message)
        if reject_match and not message.startswith("진입 거부 요약:"):
            reject_by_reason[reject_match.group(1)] += 1
        math_shadow_gate = _extract_context_token_any(message, ["shadow_gate", "math_shadow_gate"], "")
        math_shadow_reason = _extract_context_token_any(message, ["shadow_reason", "math_shadow_reason"], "")
        if math_shadow_gate:
            math_shadow_gate_counts[math_shadow_gate] += 1
        if math_shadow_reason:
            math_shadow_reason_counts[math_shadow_reason] += 1
        if _extract_context_token_any(message, ["regime_shadow_disagreement", "math_regime_shadow_disagreement"], "") == "1":
            discrete_regime = _extract_context_token(message, "discrete_regime", "unknown")
            shadow_regime = _extract_context_token(message, "shadow_regime", "unknown")
            regime_shadow_disagreements[f"{discrete_regime}->{shadow_regime}"] += 1
        if message.startswith("후보 평가 요약:"):
            stage = _extract_context_token(message, "stage", "")
            tokens = dict(re.findall(r"([A-Za-z0-9_]+)=([^ ]+)", message))
            queue_payload = {
                key: _safe_int(value)
                for key, value in tokens.items()
                if key not in {"stage", "window_start", "window_end", "total"}
                and not key.startswith("reason_")
            }
            if stage == "queue_entered":
                for key, value in queue_payload.items():
                    funnel_queue_counts[_normalize_queue_source(key) or key] += value
            elif stage == "precheck_reached":
                for key, value in queue_payload.items():
                    precheck_reached_counts[_normalize_queue_source(key) or key] += value
            elif stage == "precheck_blocked":
                for key, value in queue_payload.items():
                    precheck_blocked_counts[_normalize_queue_source(key) or key] += value
            elif stage == "policy_blocked":
                for key, value in queue_payload.items():
                    policy_blocked_counts[_normalize_queue_source(key) or key] += value
            elif stage == "router_reached":
                for key, value in queue_payload.items():
                    router_reached_counts[_normalize_queue_source(key) or key] += value
            elif stage == "router_blocked":
                for key, value in queue_payload.items():
                    router_blocked_counts[_normalize_queue_source(key) or key] += value
            elif stage == "admission_passed":
                funnel_admission_counts["passed"] += _safe_int(tokens.get("total"))
            elif stage == "admission_blocked":
                funnel_admission_counts["blocked"] += _safe_int(tokens.get("total"))
            elif stage == "hard_guard_blocked":
                for key, value in queue_payload.items():
                    hard_guard_blocked_counts[_normalize_queue_source(key) or key] += value
        elif message.startswith("진입 거부 요약:"):
            reject_reason = _extract_context_token(message, "reject_reason", "")
            total = _safe_int(_extract_context_token(message, "total", "0"))
            if reject_reason and total > 0:
                reject_by_reason[reject_reason] += total

        setup_match = _SETUP_NAME_RE.search(message)
        entry_reason_match = _ENTRY_REASON_RE.search(message)
        regime_match = _REGIME_LABEL_RE.search(message)
        is_ev_signal = message.startswith("EV 매수 신호:")
        if "매수 신호:" in message and (setup_match or is_ev_signal):
            setup_name = setup_match.group(1) if setup_match else "expected_value"
            strategy_name = _extract_context_token(
                message,
                "strategy_name",
                _extract_context_token(message, "route", "unknown_strategy"),
            )
            entry_by_setup[setup_name] += 1
            entry_by_strategy[strategy_name] += 1
            if entry_reason_match:
                entry_by_reason[entry_reason_match.group(1)] += 1
            elif is_ev_signal:
                entry_by_reason["expected_value"] += 1
            regime_label = regime_match.group(1) if regime_match else "unknown"
            entry_by_regime[regime_label] += 1
            entry_grade_math = _extract_context_token_any(message, ["entry_grade_signal", "entry_grade_math"], "unknown")
            math_grade_entries[entry_grade_math] += 1
            symbol = _extract_signal_symbol(message)
            if symbol:
                partial_sell_net_by_symbol.pop(symbol, None)
                active_entries[symbol] = {
                    "strategy_name": strategy_name,
                    "setup_name": setup_name,
                    "regime_label": regime_label,
                    "entry_grade_math": entry_grade_math,
                    "leader_score": _extract_context_token(message, "leader_score", "0"),
                    "effective_leader_score": _extract_context_token(message, "effective_leader_score", "0"),
                    "leader_pct": _extract_context_token(message, "leader_pct", "0"),
                    "recent_acceleration_pct": _extract_context_token(message, "recent_accel", "0"),
                    "entry_ev": _extract_context_token(
                        message,
                        "entry_ev",
                        _extract_context_token(message, "exp", "0"),
                    ),
                    "entry_ev_conf": _extract_context_token(
                        message,
                        "entry_ev_conf",
                        "live_plan" if is_ev_signal else "none",
                    ),
                    "conviction_tier": _extract_context_token(message, "conviction_tier", "none"),
                    "bull_risk_mode": _extract_context_token(message, "bull_risk_mode", "normal"),
                    "post_loss_admission_class": _extract_context_token(message, "post_loss_admission_class", "general"),
                    "candidate_class": _extract_context_token(
                        message,
                        "candidate_class",
                        _extract_context_token(message, "live_candidate_class", "unclassified"),
                    ),
                    "execution_mode": _extract_context_token(
                        message,
                        "execution_mode",
                        _extract_context_token(message, "route_mode", "live"),
                    ),
                    "live_route": _extract_context_token(
                        message,
                        "live_route",
                        _extract_context_token(message, "route", strategy_name),
                    ),
                    "volume_gate_threshold_used": _extract_context_token(message, "volume_gate_threshold_used", "0"),
                    "conviction_score": _extract_context_token(
                        message,
                        "conviction_score",
                        _extract_context_token(
                            message,
                            "opening_conviction_score",
                            _extract_context_token(message, "score", "0"),
                        ),
                    ),
                    "conviction_rank": _extract_context_token(
                        message,
                        "conviction_rank",
                        _extract_context_token(
                            message,
                            "opening_conviction_rank",
                            _extract_context_token(message, "rank", "0"),
                        ),
                    ),
                    "queue_source": _normalize_queue_source(
                        _extract_context_token(
                            message,
                            "queue_source",
                            _extract_context_token(message, "source", ""),
                        )
                    ),
                    "price_prediction_return_pct": _extract_context_token(message, "pred", "0"),
                    "price_prediction_lower_pct": _extract_context_token(message, "lower", "0"),
                    "price_prediction_confidence": _extract_context_token(message, "conf", "0"),
                    "entry_prediction_win_probability": _extract_context_token(message, "win", "0"),
                    "math_dominant_profile": _extract_context_token_any(message, ["dominant_profile", "math_dominant_profile"], ""),
                    "size_multiplier": _extract_context_token(message, "size_multiplier", "1"),
                    "bull_prob": _extract_context_token(message, "bull_prob", "0"),
                    "neutral_prob": _extract_context_token(message, "neutral_prob", "0"),
                    "soft_bear_prob": _extract_context_token(message, "soft_bear_prob", "0"),
                    "bear_prob": _extract_context_token(message, "bear_prob", "0"),
                    "shock_score": _extract_context_token(message, "shock_score", "0"),
                    "shock_confidence": _extract_context_token(message, "shock_conf", "0"),
                    "adaptive_take_profit_pct": _extract_context_token(message, "adaptive_tp", "0"),
                    "adaptive_stop_loss_pct": _extract_context_token(message, "adaptive_sl", "0"),
                    "adaptive_trailing_activation_pct": _extract_context_token(message, "adaptive_trail_act", "0"),
                    "adaptive_trailing_stop_pct": _extract_context_token(message, "adaptive_trail", "0"),
                    "adaptive_max_hold_minutes": _extract_context_token(message, "adaptive_hold", "0"),
                }

        if "부분매도 체결:" in message:
            symbol = _extract_partial_sell_symbol(message)
            if not symbol:
                continue
            partial_net_pnl = _extract_sell_net_pnl(message)
            partial_sell_net_by_symbol[symbol] = partial_sell_net_by_symbol.get(symbol, 0) + partial_net_pnl
            _record_sell_trade(
                symbol=symbol,
                entry_meta=active_entries.get(symbol, {}),
                message=message,
                line_ts=line_ts,
                count_as_closed_trade=False,
            )
        elif "매도 체결 정정:" in message:
            _apply_sell_trade_correction(message, line_ts)
        elif "매도 체결:" in message:
            symbol = _extract_full_sell_symbol(message)
            if not symbol:
                continue
            exit_reason = _extract_context_token(message, "exit_reason", "")
            is_partial_legacy_log = exit_reason == "partial_take_profit"
            entry_meta = active_entries.get(symbol, {})
            if not is_partial_legacy_log:
                entry_meta = active_entries.pop(symbol, {})
            cumulative_trade_net_pnl = _extract_sell_net_pnl(message)
            prior_partial_net_pnl = partial_sell_net_by_symbol.pop(symbol, 0) if not is_partial_legacy_log else 0
            _record_sell_trade(
                symbol=symbol,
                entry_meta=entry_meta,
                message=message,
                line_ts=line_ts,
                count_as_closed_trade=not is_partial_legacy_log,
                net_pnl_override=(
                    cumulative_trade_net_pnl - prior_partial_net_pnl
                    if not is_partial_legacy_log and prior_partial_net_pnl != 0
                    else None
                ),
                trade_outcome_net_pnl=(cumulative_trade_net_pnl if not is_partial_legacy_log else None),
            )

        if "그림자 후보 종료:" in message:
            strategy_name = _extract_context_token(message, "strategy_name", "unknown_strategy")
            shadow_reason = _extract_context_token(message, "shadow_reason", "unknown")
            outcome_match = _SHADOW_OUTCOME_RE.search(message)
            outcome = outcome_match.group(1) if outcome_match else "unknown"
            shadow_metrics = shadow_blocked[strategy_name]
            shadow_metrics["total"] += 1
            shadow_metrics["outcomes"][outcome] += 1
            shadow_metrics["by_reason"][shadow_reason] += 1
            candidate_reject_reason_outcomes[shadow_reason][outcome] += 1

    sorted_symbols = sorted(symbol_pnl.items(), key=lambda item: (item[1], item[0]), reverse=True)
    top_winners = [
        {"symbol": symbol, "net_pnl": net_pnl}
        for symbol, net_pnl in sorted_symbols[:5]
    ]
    top_losers = [
        {"symbol": symbol, "net_pnl": net_pnl}
        for symbol, net_pnl in sorted(symbol_pnl.items(), key=lambda item: (item[1], item[0]))[:5]
    ]
    resolved_queue_counts = dict(sorted(funnel_queue_counts.items()))
    resolved_queue_eval_reached_counts = {
        key: min(_safe_int(router_reached_counts.get(key)), _safe_int(value))
        for key, value in sorted(resolved_queue_counts.items())
    }
    resolved_admission_counts = dict(sorted(funnel_admission_counts.items()))
    realized_net_pnl = sum(
        _safe_int(metrics.get("net_pnl"))
        for metrics in strategy_pnl.values()
    )
    closed_trades = sum(
        _safe_int(metrics.get("closed_trades"))
        for metrics in strategy_pnl.values()
    )
    wins = sum(_safe_int(metrics.get("wins")) for metrics in strategy_pnl.values())
    losses = sum(_safe_int(metrics.get("losses")) for metrics in strategy_pnl.values())
    realized_fills = [_safe_int(record.get("net_pnl")) for record in trade_records]
    winning_net_pnl_sum = sum(value for value in realized_fills if value > 0)
    losing_net_pnl_sum = sum(value for value in realized_fills if value < 0)

    return {
        "log_path": str(selected_path) if selected_path else None,
        "realized": {
            "source": "sell_fill_log",
            "has_activity": bool(trade_records or closed_trades or realized_net_pnl),
            "realized_net_pnl": int(realized_net_pnl),
            "closed_trades": int(closed_trades),
            "wins": int(wins),
            "losses": int(losses),
            "breakeven": max(0, int(closed_trades - wins - losses)),
            "winning_net_pnl_sum": int(winning_net_pnl_sum),
            "losing_net_pnl_sum": int(losing_net_pnl_sum),
            "largest_win_net": max([0, *realized_fills]),
            "largest_loss_net": min([0, *realized_fills]),
        },
        "entries": {
            "total": int(sum(entry_by_setup.values())),
            "by_setup": dict(sorted(entry_by_setup.items())),
            "by_strategy": dict(sorted(entry_by_strategy.items())),
            "by_entry_reason": dict(sorted(entry_by_reason.items())),
            "by_regime": dict(sorted(entry_by_regime.items())),
        },
        "rejections": {
            "total": int(sum(reject_by_reason.values())),
            "by_reason": dict(sorted(reject_by_reason.items())),
        },
        "strategy_pnl": {
            strategy: metrics
            for strategy, metrics in sorted(strategy_pnl.items())
        },
        "strategy_hourly_pnl": {
            strategy: {
                hour: {
                    "closed_trades": int(metrics["closed_trades"]),
                    "net_pnl": int(metrics["net_pnl"]),
                    "expectancy": round(
                        metrics["net_pnl"] / metrics["closed_trades"],
                        2,
                    ) if metrics["closed_trades"] else 0.0,
                }
                for hour, metrics in sorted(hours.items())
            }
            for strategy, hours in sorted(strategy_hourly.items())
        },
        "setup_pnl": {
            setup: metrics
            for setup, metrics in sorted(setup_pnl.items())
        },
        "regime_pnl": dict(sorted(regime_pnl.items())),
        "shadow_blocked": {
            strategy: {
                "total": int(metrics["total"]),
                "outcomes": dict(sorted(metrics["outcomes"].items())),
                "by_reason": dict(sorted(metrics["by_reason"].items())),
            }
            for strategy, metrics in sorted(shadow_blocked.items())
        },
        "trade_records": trade_records,
        "math_shadow": {
            "entries_by_grade": dict(sorted(math_grade_entries.items())),
            "grade_pnl": {
                grade: {
                    "closed_trades": int(metrics["closed_trades"]),
                    "net_pnl": int(metrics["net_pnl"]),
                    "expectancy": round(
                        metrics["net_pnl"] / metrics["closed_trades"],
                        2,
                    ) if metrics["closed_trades"] else 0.0,
                }
                for grade, metrics in sorted(math_grade_pnl.items())
            },
            "ev_buckets": {
                bucket: {
                    "closed_trades": int(metrics["closed_trades"]),
                    "net_pnl": int(metrics["net_pnl"]),
                    "expectancy": round(
                        metrics["net_pnl"] / metrics["closed_trades"],
                        2,
                    ) if metrics["closed_trades"] else 0.0,
                }
                for bucket, metrics in sorted(math_ev_buckets.items())
            },
            "shadow_gate_counts": dict(sorted(math_shadow_gate_counts.items())),
            "shadow_reason_counts": dict(sorted(math_shadow_reason_counts.items())),
            "candidate_reject_reason_outcomes": {
                reason: dict(sorted(outcomes.items()))
                for reason, outcomes in sorted(candidate_reject_reason_outcomes.items())
            },
            "regime_shadow_disagreements": dict(sorted(regime_shadow_disagreements.items())),
            "conviction_tier_pnl": {
                tier: {
                    "closed_trades": int(metrics["closed_trades"]),
                    "net_pnl": int(metrics["net_pnl"]),
                    "expectancy": round(metrics["net_pnl"] / metrics["closed_trades"], 2)
                    if metrics["closed_trades"]
                    else 0.0,
                }
                for tier, metrics in sorted(conviction_tier_pnl.items())
            },
            "bull_risk_mode_pnl": {
                mode: {
                    "closed_trades": int(metrics["closed_trades"]),
                    "net_pnl": int(metrics["net_pnl"]),
                    "expectancy": round(metrics["net_pnl"] / metrics["closed_trades"], 2)
                    if metrics["closed_trades"]
                    else 0.0,
                }
                for mode, metrics in sorted(bull_risk_mode_pnl.items())
            },
            "post_loss_admission_class_pnl": {
                admission_class: {
                    "closed_trades": int(metrics["closed_trades"]),
                    "net_pnl": int(metrics["net_pnl"]),
                    "expectancy": round(metrics["net_pnl"] / metrics["closed_trades"], 2)
                    if metrics["closed_trades"]
                    else 0.0,
                }
                for admission_class, metrics in sorted(post_loss_admission_class_pnl.items())
            },
            "candidate_class_pnl": {
                candidate_class: {
                    "closed_trades": int(metrics["closed_trades"]),
                    "net_pnl": int(metrics["net_pnl"]),
                    "expectancy": round(metrics["net_pnl"] / metrics["closed_trades"], 2)
                    if metrics["closed_trades"]
                    else 0.0,
                }
                for candidate_class, metrics in sorted(candidate_class_pnl.items())
            },
            "live_route_pnl": {
                route: {
                    "closed_trades": int(metrics["closed_trades"]),
                    "net_pnl": int(metrics["net_pnl"]),
                    "expectancy": round(metrics["net_pnl"] / metrics["closed_trades"], 2)
                    if metrics["closed_trades"]
                    else 0.0,
                }
                for route, metrics in sorted(live_route_pnl.items())
            },
            "queue_source_pnl": {
                key: {
                    "closed_trades": int(metrics["closed_trades"]),
                    "net_pnl": int(metrics["net_pnl"]),
                    "expectancy": round(metrics["net_pnl"] / metrics["closed_trades"], 2)
                    if metrics["closed_trades"]
                    else 0.0,
                }
                for key, metrics in sorted(queue_source_pnl.items())
            },
            "stop_stats": {
                "stop_triggered_count": int(stop_triggered_count),
                "stop_limit_fallback_count": int(stop_limit_fallback_count),
                "stop_overshoot_avg_krw": round(sum(stop_overshoots) / len(stop_overshoots), 2) if stop_overshoots else 0.0,
                "stop_overshoot_max_krw": max(stop_overshoots) if stop_overshoots else 0,
            },
            "queue_counts": resolved_queue_counts,
            "precheck_reached_counts": dict(sorted(precheck_reached_counts.items())),
            "precheck_blocked_counts": dict(sorted(precheck_blocked_counts.items())),
            "policy_blocked_counts": dict(sorted(policy_blocked_counts.items())),
            "router_reached_counts": dict(sorted(router_reached_counts.items())),
            "router_blocked_counts": dict(sorted(router_blocked_counts.items())),
            "hard_guard_blocked_counts": dict(sorted(hard_guard_blocked_counts.items())),
            "queue_eval_reached_counts": resolved_queue_eval_reached_counts,
            "admission_counts": resolved_admission_counts,
        },
        "symbols": {
            "net_pnl": dict(sorted(symbol_pnl.items())),
            "top_winners": top_winners,
            "top_losers": top_losers,
        },
        "risk_events": {
            "risk_stage_transitions": dict(sorted(risk_stage_transitions.items())),
            "risk_stage1_block_count": int(reject_by_reason.get("risk_stage1_block", 0)),
            "daily_hard_stop_triggered": daily_hard_stop_triggered,
            "daily_profit_target_triggered": daily_profit_target_triggered,
        },
    }


def _derive_log_realized_summary(log_analysis: Dict[str, Any]) -> Dict[str, Any]:
    realized = log_analysis.get("realized")
    if isinstance(realized, dict):
        return dict(realized)

    strategy_pnl = log_analysis.get("strategy_pnl") or {}
    closed_trades = sum(
        _safe_int(metrics.get("closed_trades"))
        for metrics in strategy_pnl.values()
        if isinstance(metrics, dict)
    )
    wins = sum(
        _safe_int(metrics.get("wins"))
        for metrics in strategy_pnl.values()
        if isinstance(metrics, dict)
    )
    losses = sum(
        _safe_int(metrics.get("losses"))
        for metrics in strategy_pnl.values()
        if isinstance(metrics, dict)
    )
    realized_net_pnl = sum(
        _safe_int(metrics.get("net_pnl"))
        for metrics in strategy_pnl.values()
        if isinstance(metrics, dict)
    )
    records = [
        record
        for record in (log_analysis.get("trade_records") or [])
        if isinstance(record, dict)
    ]
    realized_fills = [_safe_int(record.get("net_pnl")) for record in records]
    winning_net_pnl_sum = sum(value for value in realized_fills if value > 0)
    losing_net_pnl_sum = sum(value for value in realized_fills if value < 0)
    unattributed_net = realized_net_pnl - sum(realized_fills)
    if unattributed_net > 0:
        winning_net_pnl_sum += unattributed_net
    elif unattributed_net < 0:
        losing_net_pnl_sum += unattributed_net
    return {
        "source": "sell_fill_log",
        "has_activity": bool(records or closed_trades or realized_net_pnl),
        "realized_net_pnl": int(realized_net_pnl),
        "closed_trades": int(closed_trades),
        "wins": int(wins),
        "losses": int(losses),
        "breakeven": max(0, int(closed_trades - wins - losses)),
        "winning_net_pnl_sum": int(winning_net_pnl_sum),
        "losing_net_pnl_sum": int(losing_net_pnl_sum),
        "largest_win_net": max([0, *realized_fills]),
        "largest_loss_net": min([0, *realized_fills]),
    }


def _reconcile_scorecard_with_log(scorecard: Dict[str, Any]) -> Dict[str, Any]:
    log_analysis = scorecard.get("log_analysis") or {}
    realized = _derive_log_realized_summary(log_analysis)
    if not bool(realized.get("has_activity")):
        return scorecard

    pnl = scorecard.setdefault("pnl", {})
    trades_before = dict(scorecard.get("trades") or {})
    snapshot_net = _safe_int(pnl.get("realized_net_pnl"))
    snapshot_session = _safe_int(pnl.get("session_pnl"))
    log_net = _safe_int(realized.get("realized_net_pnl"))
    pnl["realized_net_pnl"] = log_net
    pnl["session_pnl"] = log_net
    pnl["winning_net_pnl_sum"] = _safe_int(realized.get("winning_net_pnl_sum"))
    pnl["losing_net_pnl_sum"] = _safe_int(realized.get("losing_net_pnl_sum"))
    pnl["largest_win_net"] = _safe_int(realized.get("largest_win_net"))
    pnl["largest_loss_net"] = _safe_int(realized.get("largest_loss_net"))
    pnl["realized_source"] = "sell_fill_log"
    scorecard["trades"] = _compute_trade_metrics(
        {
            "realized_net_pnl": log_net,
            "trade_count": _safe_int(realized.get("closed_trades")),
            "win_count": _safe_int(realized.get("wins")),
            "loss_count": _safe_int(realized.get("losses")),
            "breakeven_count": _safe_int(realized.get("breakeven")),
            "winning_net_pnl_sum": _safe_int(realized.get("winning_net_pnl_sum")),
            "losing_net_pnl_sum": _safe_int(realized.get("losing_net_pnl_sum")),
        }
    )
    scorecard["reconciliation"] = {
        "authoritative_source": "sell_fill_log",
        "strategy_snapshot_realized_net_pnl": snapshot_net,
        "strategy_snapshot_session_pnl": snapshot_session,
        "strategy_snapshot_closed_trades": _safe_int(trades_before.get("closed_trades")),
        "net_pnl_difference": int(log_net - snapshot_net),
    }
    return scorecard


def render_daily_scorecard_markdown(scorecard: Dict[str, Any]) -> str:
    pnl = scorecard.get("pnl", {})
    trades = scorecard.get("trades", {})
    balance = scorecard.get("balance", {})
    strategy = scorecard.get("strategy", {})
    log_analysis = scorecard.get("log_analysis", {})
    paper_gate = scorecard.get("paper_gate", {})

    lines = [
        f"# 일별 성적표 {scorecard.get('date', '')}",
        "",
        f"- 생성 시각: {scorecard.get('generated_at', '')}",
        f"- 모드: {scorecard.get('trading_mode', '')}",
        f"- 세션 손익: {_format_currency(pnl.get('session_pnl'))}",
        f"- 순실현손익: {_format_currency(pnl.get('realized_net_pnl'))}",
        f"- 총실현손익: {_format_currency(pnl.get('realized_gross_pnl'))}",
        f"- 수수료: {_format_currency(pnl.get('fees_paid'))}",
        f"- 세금/슬리피지: {_format_currency(pnl.get('taxes_paid'))}",
        f"- 청산 체결 수: {_safe_int(trades.get('closed_trades'))}건",
        f"- 승/패/보합: {_safe_int(trades.get('wins'))}/{_safe_int(trades.get('losses'))}/{_safe_int(trades.get('breakeven'))}",
        f"- 승률: {_format_ratio(trades.get('win_rate'))}",
        f"- 평균 순손익/거래: {_safe_float(trades.get('average_net_per_trade')):,.2f}원",
        f"- 평균 순이익/승: {_safe_float(trades.get('average_win_net')):,.2f}원",
        f"- 평균 순손실/패: {_safe_float(trades.get('average_loss_net')):,.2f}원",
        f"- Profit Factor: {_format_profit_factor(trades.get('profit_factor'))}",
        f"- 최대 순이익 체결: {_format_currency(pnl.get('largest_win_net'))}",
        f"- 최대 순손실 체결: {_format_currency(pnl.get('largest_loss_net'))}",
        f"- 평가금액: {_format_currency(balance.get('total_eval_amount'))}",
        f"- 평가손익: {_format_currency(balance.get('total_profit_loss'))}",
        f"- 평가수익률: {_safe_float(balance.get('total_profit_rate')):.2f}%",
        f"- 미청산 포지션 수: {_safe_int(strategy.get('open_positions_count'))}개",
        f"- 거래중지 상태: {'예' if strategy.get('halted') else '아니오'}",
        "",
    ]

    entries = log_analysis.get("entries", {})
    rejections = log_analysis.get("rejections", {})
    strategy_pnl = log_analysis.get("strategy_pnl", {})
    setup_pnl = log_analysis.get("setup_pnl", {})
    regime_pnl = log_analysis.get("regime_pnl", {})
    strategy_hourly_pnl = log_analysis.get("strategy_hourly_pnl", {})
    shadow_blocked = log_analysis.get("shadow_blocked", {})
    math_shadow = log_analysis.get("math_shadow", {})
    symbols = log_analysis.get("symbols", {})
    risk_events = log_analysis.get("risk_events", {})
    if entries or rejections or setup_pnl:
        setup_entry_summary = ", ".join(
            f"{key} {_safe_int(value)}건"
            for key, value in entries.get("by_setup", {}).items()
        ) or "-"
        strategy_entry_summary = ", ".join(
            f"{key} {_safe_int(value)}건"
            for key, value in entries.get("by_strategy", {}).items()
        ) or "-"
        rejection_summary = ", ".join(
            f"{key} {_safe_int(value)}건"
            for key, value in rejections.get("by_reason", {}).items()
        ) or "-"
        strategy_pnl_summary = ", ".join(
            f"{key} {_format_currency(value.get('net_pnl'))} / {_safe_int(value.get('closed_trades'))}건"
            for key, value in strategy_pnl.items()
        ) or "-"
        setup_pnl_summary = ", ".join(
            f"{key} {_format_currency(value.get('net_pnl'))} / {_safe_int(value.get('closed_trades'))}건"
            for key, value in setup_pnl.items()
        ) or "-"
        regime_pnl_summary = ", ".join(
            f"{key} {_format_currency(value)}"
            for key, value in regime_pnl.items()
        ) or "-"
        hourly_expectancy_summary = ", ".join(
            f"{strategy} "
            + "/".join(
                f"{hour}시 {metrics.get('expectancy', 0.0):,.1f}원"
                for hour, metrics in hourly.items()
            )
            for strategy, hourly in strategy_hourly_pnl.items()
        ) or "-"
        shadow_summary = ", ".join(
            f"{strategy} {data.get('total', 0)}건 "
            + "/".join(
                f"{outcome} {count}건"
                for outcome, count in data.get("outcomes", {}).items()
            )
            for strategy, data in shadow_blocked.items()
        ) or "-"
        top_winners_summary = ", ".join(
            f"{item.get('symbol')} {_format_currency(item.get('net_pnl'))}"
            for item in symbols.get("top_winners", [])
        ) or "-"
        top_losers_summary = ", ".join(
            f"{item.get('symbol')} {_format_currency(item.get('net_pnl'))}"
            for item in symbols.get("top_losers", [])
        ) or "-"
        math_grade_summary = ", ".join(
            f"{grade} {_safe_int(value)}건"
            for grade, value in (math_shadow.get("entries_by_grade") or {}).items()
        ) or "-"
        math_ev_summary = ", ".join(
            f"{bucket} {_format_currency(value.get('net_pnl'))} / {_safe_int(value.get('closed_trades'))}건"
            for bucket, value in (math_shadow.get("ev_buckets") or {}).items()
        ) or "-"
        conviction_tier_summary = ", ".join(
            f"{tier} {_format_currency(value.get('net_pnl'))} / {_safe_int(value.get('closed_trades'))}건"
            for tier, value in (math_shadow.get("conviction_tier_pnl") or {}).items()
        ) or "-"
        candidate_class_summary = ", ".join(
            f"{candidate_class} {_format_currency(value.get('net_pnl'))} / {_safe_int(value.get('closed_trades'))}건"
            for candidate_class, value in (math_shadow.get("candidate_class_pnl") or {}).items()
        ) or "-"
        queue_source_summary = ", ".join(
            f"{key} {_format_currency(value.get('net_pnl'))} / {_safe_int(value.get('closed_trades'))}건"
            for key, value in (math_shadow.get("queue_source_pnl") or {}).items()
        ) or "-"
        candidate_reject_reason_summary = ", ".join(
            f"{reason} "
            + "/".join(
                f"{outcome} {count}건"
                for outcome, count in (outcomes or {}).items()
            )
            for reason, outcomes in (math_shadow.get("candidate_reject_reason_outcomes") or {}).items()
        ) or "-"
        math_regime_summary = ", ".join(
            f"{key} {value}건"
            for key, value in (math_shadow.get("regime_shadow_disagreements") or {}).items()
        ) or "-"
        stop_stats = math_shadow.get("stop_stats") or {}
        lines.extend(
            [
                "## 로그 분석",
                f"- 분석 로그: {log_analysis.get('log_path') or '-'}",
                f"- 진입 신호 수: {_safe_int(entries.get('total'))}건",
                f"- 진입 셋업별: {setup_entry_summary}",
                f"- 진입 전략별: {strategy_entry_summary}",
                f"- 차단 사유 수: {_safe_int(rejections.get('total'))}건",
                f"- 차단 사유별: {rejection_summary}",
                f"- 전략별 순손익: {strategy_pnl_summary}",
                f"- 전략별 시간대 기대값: {hourly_expectancy_summary}",
                f"- 셋업별 순손익: {setup_pnl_summary}",
                f"- 레짐별 순손익: {regime_pnl_summary}",
                f"- 그림자 차단 후보 결과: {shadow_summary}",
                f"- 보조 신호 등급 분포: {math_grade_summary}",
                f"- 보조 신호 EV 버킷: {math_ev_summary}",
                f"- conviction tier별 손익: {conviction_tier_summary}",
                f"- 실전 후보 class별 손익: {candidate_class_summary}",
                f"- 후보 source별 손익: {queue_source_summary}",
                f"- 실전 제외 사유별 결과: {candidate_reject_reason_summary}",
                f"- 손절 overshoot: 트리거 {_safe_int(stop_stats.get('stop_triggered_count'))}건 / 폴백 {_safe_int(stop_stats.get('stop_limit_fallback_count'))}건 / 평균 {_safe_float(stop_stats.get('stop_overshoot_avg_krw')):,.2f}원 / 최대 {_format_currency(stop_stats.get('stop_overshoot_max_krw'))}",
                f"- 보조 신호 레짐 불일치: {math_regime_summary}",
                f"- 종목별 상위: {top_winners_summary}",
                f"- 종목별 하위: {top_losers_summary}",
                f"- 손실 1단계 진입 차단: {_safe_int(risk_events.get('risk_stage1_block_count'))}건",
                f"- 총손익 하드스탑 발생: {'예' if risk_events.get('daily_hard_stop_triggered') else '아니오'}",
                "",
            ]
        )
    if paper_gate:
        lines.extend(
            [
                "## 최근 5거래일 Gate",
                f"- 구간: {paper_gate.get('first_date') or '-'} ~ {paper_gate.get('last_date') or '-'}",
                f"- 누적 순손익: {_format_currency(paper_gate.get('rolling_net_pnl'))}",
                f"- 플러스 일수: {_safe_int(paper_gate.get('positive_days'))}일",
                f"- 하드스탑 발생: {_safe_int(paper_gate.get('hard_stop_days'))}일",
                f"- 목표 달성 일수: {_safe_int(paper_gate.get('target_hit_days'))}일",
                f"- Gate 통과: {'예' if paper_gate.get('passed') else '아니오'}",
                "",
            ]
        )
    return "\n".join(lines)


def evaluate_strategy_gates(
    scorecards: List[Dict[str, Any]],
    *,
    window_days: int = DEFAULT_STRATEGY_GATE_WINDOW_DAYS,
    min_closed_trades: int = DEFAULT_STRATEGY_GATE_MIN_CLOSED_TRADES,
    disable_expectancy_threshold: float = -150.0,
) -> Dict[str, Any]:
    window_cards = scorecards[-window_days:] if window_days > 0 else list(scorecards)
    strategy_rollup: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "closed_trades": 0,
            "net_pnl": 0.0,
            "wins": 0,
            "losses": 0,
            "hourly": defaultdict(lambda: {"closed_trades": 0, "net_pnl": 0.0}),
            "shadow_outcomes": defaultdict(int),
            "shadow_reasons": defaultdict(int),
        }
    )

    for card in window_cards:
        log_analysis = card.get("log_analysis", {})
        trade_records = (log_analysis.get("trade_records") or [])
        if trade_records:
            for record in trade_records:
                setup_name = str(record.get("setup_name") or "")
                queue_source = str(record.get("queue_source") or "")
                if setup_name == "restored_position" or queue_source == "account_restore":
                    continue
                strategy_name = str(record.get("strategy_name") or "unknown_strategy")
                rollup = strategy_rollup[strategy_name]
                net_pnl = _safe_float(record.get("net_pnl"))
                rollup["closed_trades"] += 1
                rollup["net_pnl"] += net_pnl
                if net_pnl > 0:
                    rollup["wins"] += 1
                elif net_pnl < 0:
                    rollup["losses"] += 1
                hour = str(record.get("hour_bucket") or "unknown")
                bucket = rollup["hourly"][hour]
                bucket["closed_trades"] += 1
                bucket["net_pnl"] += net_pnl
        else:
            for strategy_name, metrics in (log_analysis.get("strategy_pnl") or {}).items():
                rollup = strategy_rollup[strategy_name]
                rollup["closed_trades"] += _safe_int(metrics.get("closed_trades"))
                rollup["net_pnl"] += _safe_float(metrics.get("net_pnl"))
                rollup["wins"] += _safe_int(metrics.get("wins"))
                rollup["losses"] += _safe_int(metrics.get("losses"))
            for strategy_name, hours in (log_analysis.get("strategy_hourly_pnl") or {}).items():
                rollup = strategy_rollup[strategy_name]
                for hour, metrics in hours.items():
                    bucket = rollup["hourly"][hour]
                    bucket["closed_trades"] += _safe_int(metrics.get("closed_trades"))
                    bucket["net_pnl"] += _safe_float(metrics.get("net_pnl"))
        for strategy_name, shadow in (log_analysis.get("shadow_blocked") or {}).items():
            rollup = strategy_rollup[strategy_name]
            for outcome, count in (shadow.get("outcomes") or {}).items():
                rollup["shadow_outcomes"][outcome] += _safe_int(count)
            for reason, count in (shadow.get("by_reason") or {}).items():
                rollup["shadow_reasons"][reason] += _safe_int(count)

    now_iso = datetime.now().isoformat(timespec="seconds")
    strategies: Dict[str, Any] = {}
    for strategy_name, metrics in sorted(strategy_rollup.items()):
        closed_trades = _safe_int(metrics.get("closed_trades"))
        net_pnl = _safe_float(metrics.get("net_pnl"))
        expectancy = round(net_pnl / closed_trades, 2) if closed_trades > 0 else 0.0
        win_rate = round(_safe_int(metrics.get("wins")) / closed_trades, 4) if closed_trades > 0 else 0.0
        enabled = not (closed_trades >= min_closed_trades and expectancy <= float(disable_expectancy_threshold))
        reason = "negative_expectancy" if not enabled else "pass"
        strategies[strategy_name] = {
            "enabled": enabled,
            "reason": reason,
            "closed_trades": closed_trades,
            "expectancy": expectancy,
            "avg_net_pnl": expectancy,
            "win_rate": win_rate,
            "hour_bucket_expectancy": {
                hour: round(bucket["net_pnl"] / bucket["closed_trades"], 2)
                if bucket["closed_trades"]
                else 0.0
                for hour, bucket in sorted(metrics["hourly"].items())
            },
            "shadow_blocked_summary": {
                "outcomes": dict(sorted(metrics["shadow_outcomes"].items())),
                "by_reason": dict(sorted(metrics["shadow_reasons"].items())),
            },
            "updated_at": now_iso,
        }

    return {
        "generated_at": now_iso,
        "window_days": window_days,
        "min_closed_trades": min_closed_trades,
        "strategies": strategies,
    }


def render_strategy_gates_markdown(payload: Dict[str, Any]) -> str:
    lines = [
        "# 전략 자동 게이트",
        "",
        f"- 생성 시각: {payload.get('generated_at', '')}",
        f"- 최근 창: {_safe_int(payload.get('window_days'))}거래일",
        f"- 최소 청산 체결: {_safe_int(payload.get('min_closed_trades'))}건",
        "",
    ]
    strategies = payload.get("strategies", {})
    if not strategies:
        lines.append("- 집계된 전략이 없습니다.")
        lines.append("")
        return "\n".join(lines)

    for strategy_name, metrics in strategies.items():
        hourly = metrics.get("hour_bucket_expectancy", {})
        shadow = metrics.get("shadow_blocked_summary", {})
        lines.extend(
            [
                f"## {strategy_name}",
                f"- 활성화: {'예' if metrics.get('enabled') else '아니오'}",
                f"- 사유: {metrics.get('reason', '-')}",
                f"- 청산 체결: {_safe_int(metrics.get('closed_trades'))}건",
                f"- 기대값: {_safe_float(metrics.get('expectancy')):,.2f}원",
                f"- 승률: {_format_ratio(metrics.get('win_rate'))}",
                f"- 시간대 기대값: "
                + (", ".join(f"{hour}시 {value:,.1f}원" for hour, value in hourly.items()) or "-"),
                f"- 그림자 결과: "
                + (
                    ", ".join(f"{key} {value}건" for key, value in (shadow.get('outcomes') or {}).items())
                    or "-"
                ),
                "",
            ]
        )
    return "\n".join(lines)


def evaluate_math_shadow_report(
    scorecards: List[Dict[str, Any]],
    *,
    window_days: int = 5,
    min_closed_trades: int = 4,
) -> Dict[str, Any]:
    window_cards = scorecards[-window_days:] if window_days > 0 else list(scorecards)
    grade_rollup: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"entries": 0, "closed_trades": 0, "net_pnl": 0.0}
    )
    ev_rollup: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"closed_trades": 0, "net_pnl": 0.0}
    )
    leader_percentile_rollup: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"closed_trades": 0, "net_pnl": 0.0}
    )
    shock_score_rollup: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"closed_trades": 0, "net_pnl": 0.0}
    )
    adaptive_hold_rollup: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"closed_trades": 0, "net_pnl": 0.0}
    )
    regime_profile_rollup: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"closed_trades": 0, "net_pnl": 0.0}
    )
    conviction_tier_rollup: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"closed_trades": 0, "net_pnl": 0.0}
    )
    bull_risk_mode_rollup: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"closed_trades": 0, "net_pnl": 0.0}
    )
    post_loss_admission_class_rollup: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"closed_trades": 0, "net_pnl": 0.0}
    )
    candidate_class_rollup: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"closed_trades": 0, "net_pnl": 0.0}
    )
    queue_source_rollup: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"closed_trades": 0, "net_pnl": 0.0}
    )
    shadow_gate_counts: Dict[str, int] = defaultdict(int)
    regime_disagreements: Dict[str, int] = defaultdict(int)
    queue_counts: Dict[str, int] = defaultdict(int)
    precheck_reached_counts: Dict[str, int] = defaultdict(int)
    precheck_blocked_counts: Dict[str, int] = defaultdict(int)
    policy_blocked_counts: Dict[str, int] = defaultdict(int)
    router_reached_counts: Dict[str, int] = defaultdict(int)
    router_blocked_counts: Dict[str, int] = defaultdict(int)
    hard_guard_blocked_counts: Dict[str, int] = defaultdict(int)
    queue_eval_reached_counts: Dict[str, int] = defaultdict(int)
    admission_counts: Dict[str, int] = defaultdict(int)
    leader_hits: List[Dict[str, Any]] = []
    stop_triggered_count = 0
    stop_limit_fallback_count = 0
    stop_overshoots: List[int] = []
    candidate_reject_reason_outcomes_rollup: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for card in window_cards:
        log_analysis = card.get("log_analysis", {})
        math_shadow = log_analysis.get("math_shadow", {})
        for grade, count in (math_shadow.get("entries_by_grade") or {}).items():
            grade_rollup[grade]["entries"] += _safe_int(count)
        for grade, metrics in (math_shadow.get("grade_pnl") or {}).items():
            grade_rollup[grade]["closed_trades"] += _safe_int(metrics.get("closed_trades"))
            grade_rollup[grade]["net_pnl"] += _safe_float(metrics.get("net_pnl"))
        for bucket, metrics in (math_shadow.get("ev_buckets") or {}).items():
            ev_rollup[bucket]["closed_trades"] += _safe_int(metrics.get("closed_trades"))
            ev_rollup[bucket]["net_pnl"] += _safe_float(metrics.get("net_pnl"))
        for key, value in (math_shadow.get("shadow_gate_counts") or {}).items():
            shadow_gate_counts[key] += _safe_int(value)
        for reason, outcomes in (math_shadow.get("candidate_reject_reason_outcomes") or {}).items():
            for outcome, count in (outcomes or {}).items():
                candidate_reject_reason_outcomes_rollup[str(reason)][str(outcome)] += _safe_int(count)
        for key, value in (math_shadow.get("regime_shadow_disagreements") or {}).items():
            regime_disagreements[key] += _safe_int(value)
        for key, value in (math_shadow.get("queue_counts") or {}).items():
            queue_counts[key] += _safe_int(value)
        for key, value in (math_shadow.get("precheck_reached_counts") or {}).items():
            precheck_reached_counts[key] += _safe_int(value)
        for key, value in (math_shadow.get("precheck_blocked_counts") or {}).items():
            precheck_blocked_counts[key] += _safe_int(value)
        for key, value in (math_shadow.get("policy_blocked_counts") or {}).items():
            policy_blocked_counts[key] += _safe_int(value)
        for key, value in (math_shadow.get("router_reached_counts") or {}).items():
            router_reached_counts[key] += _safe_int(value)
        for key, value in (math_shadow.get("router_blocked_counts") or {}).items():
            router_blocked_counts[key] += _safe_int(value)
        for key, value in (math_shadow.get("hard_guard_blocked_counts") or {}).items():
            hard_guard_blocked_counts[key] += _safe_int(value)
        for key, value in (math_shadow.get("queue_eval_reached_counts") or {}).items():
            queue_eval_reached_counts[key] += _safe_int(value)
        for key, value in (math_shadow.get("admission_counts") or {}).items():
            admission_counts[key] += _safe_int(value)
        for record in (log_analysis.get("trade_records") or []):
            if _safe_float(record.get("leader_pct")) >= 0.95:
                leader_hits.append(
                    {
                        "date": card.get("date"),
                        "symbol": record.get("symbol"),
                        "strategy_name": record.get("strategy_name"),
                        "entry_grade_math": record.get("entry_grade_math"),
                        "leader_pct": round(_safe_float(record.get("leader_pct")), 4),
                        "entry_ev": round(_safe_float(record.get("entry_ev")), 2),
                        "net_pnl": _safe_int(record.get("net_pnl")),
                    }
                )
            leader_pct = _safe_float(record.get("leader_pct"))
            if leader_pct >= 0.90:
                pct_bucket = "0.90-1.00"
            elif leader_pct >= 0.80:
                pct_bucket = "0.80-0.89"
            else:
                pct_bucket = "<0.80"
            leader_percentile_rollup[pct_bucket]["closed_trades"] += 1
            leader_percentile_rollup[pct_bucket]["net_pnl"] += _safe_float(record.get("net_pnl"))

            shock_score = _safe_float(record.get("shock_score"))
            if shock_score >= 1.50:
                shock_bucket = "1.50+"
            elif shock_score >= 1.00:
                shock_bucket = "1.00-1.49"
            elif shock_score >= 0.50:
                shock_bucket = "0.50-0.99"
            else:
                shock_bucket = "<0.50"
            shock_score_rollup[shock_bucket]["closed_trades"] += 1
            shock_score_rollup[shock_bucket]["net_pnl"] += _safe_float(record.get("net_pnl"))

            adaptive_hold = _safe_int(record.get("adaptive_max_hold_minutes"))
            if adaptive_hold <= 0:
                hold_bucket = "none"
            elif adaptive_hold <= 12:
                hold_bucket = "<=12"
            elif adaptive_hold <= 18:
                hold_bucket = "13-18"
            else:
                hold_bucket = "19+"
            adaptive_hold_rollup[hold_bucket]["closed_trades"] += 1
            adaptive_hold_rollup[hold_bucket]["net_pnl"] += _safe_float(record.get("net_pnl"))

            dominant_profile = str(record.get("math_dominant_profile") or "").strip()
            if not dominant_profile:
                probs = {
                    "bull": _safe_float(record.get("bull_prob")),
                    "neutral": _safe_float(record.get("neutral_prob")),
                    "soft_bear": _safe_float(record.get("soft_bear_prob")),
                    "bear": _safe_float(record.get("bear_prob")),
                }
                dominant_profile = sorted(probs.items(), key=lambda item: (item[1], item[0]), reverse=True)[0][0]
            regime_profile_rollup[dominant_profile]["closed_trades"] += 1
            regime_profile_rollup[dominant_profile]["net_pnl"] += _safe_float(record.get("net_pnl"))
            conviction_tier = str(record.get("conviction_tier") or "none")
            conviction_tier_rollup[conviction_tier]["closed_trades"] += 1
            conviction_tier_rollup[conviction_tier]["net_pnl"] += _safe_float(record.get("net_pnl"))
            bull_risk_mode = str(record.get("bull_risk_mode") or "normal")
            bull_risk_mode_rollup[bull_risk_mode]["closed_trades"] += 1
            bull_risk_mode_rollup[bull_risk_mode]["net_pnl"] += _safe_float(record.get("net_pnl"))
            post_loss_admission_class = str(record.get("post_loss_admission_class") or "general")
            post_loss_admission_class_rollup[post_loss_admission_class]["closed_trades"] += 1
            post_loss_admission_class_rollup[post_loss_admission_class]["net_pnl"] += _safe_float(record.get("net_pnl"))
            candidate_class = str(record.get("candidate_class") or "unclassified")
            candidate_class_rollup[candidate_class]["closed_trades"] += 1
            candidate_class_rollup[candidate_class]["net_pnl"] += _safe_float(record.get("net_pnl"))
            queue_source = str(record.get("queue_source") or "unknown")
            queue_source_rollup[queue_source]["closed_trades"] += 1
            queue_source_rollup[queue_source]["net_pnl"] += _safe_float(record.get("net_pnl"))
            if str(record.get("protective_exit_mode") or ""):
                stop_triggered_count += 1
                if str(record.get("fill_mode") or "") == "limit_then_market":
                    stop_limit_fallback_count += 1
                stop_overshoots.append(max(0, _safe_int(record.get("stop_overshoot_krw"))))

    grade_summary = {
        grade: {
            "entries": int(metrics["entries"]),
            "closed_trades": int(metrics["closed_trades"]),
            "net_pnl": int(metrics["net_pnl"]),
            "expectancy": round(metrics["net_pnl"] / metrics["closed_trades"], 2)
            if metrics["closed_trades"]
            else 0.0,
        }
        for grade, metrics in sorted(grade_rollup.items())
    }
    ev_summary = {
        bucket: {
            "closed_trades": int(metrics["closed_trades"]),
            "net_pnl": int(metrics["net_pnl"]),
            "expectancy": round(metrics["net_pnl"] / metrics["closed_trades"], 2)
            if metrics["closed_trades"]
            else 0.0,
        }
        for bucket, metrics in sorted(ev_rollup.items())
    }
    leader_percentile_summary = {
        bucket: {
            "closed_trades": int(metrics["closed_trades"]),
            "net_pnl": int(metrics["net_pnl"]),
            "expectancy": round(metrics["net_pnl"] / metrics["closed_trades"], 2)
            if metrics["closed_trades"]
            else 0.0,
        }
        for bucket, metrics in sorted(leader_percentile_rollup.items())
    }
    shock_score_summary = {
        bucket: {
            "closed_trades": int(metrics["closed_trades"]),
            "net_pnl": int(metrics["net_pnl"]),
            "expectancy": round(metrics["net_pnl"] / metrics["closed_trades"], 2)
            if metrics["closed_trades"]
            else 0.0,
        }
        for bucket, metrics in sorted(shock_score_rollup.items())
    }
    adaptive_hold_summary = {
        bucket: {
            "closed_trades": int(metrics["closed_trades"]),
            "net_pnl": int(metrics["net_pnl"]),
            "expectancy": round(metrics["net_pnl"] / metrics["closed_trades"], 2)
            if metrics["closed_trades"]
            else 0.0,
        }
        for bucket, metrics in sorted(adaptive_hold_rollup.items())
    }
    regime_profile_summary = {
        regime: {
            "closed_trades": int(metrics["closed_trades"]),
            "net_pnl": int(metrics["net_pnl"]),
            "expectancy": round(metrics["net_pnl"] / metrics["closed_trades"], 2)
            if metrics["closed_trades"]
            else 0.0,
        }
        for regime, metrics in sorted(regime_profile_rollup.items())
    }
    conviction_tier_summary = {
        tier: {
            "closed_trades": int(metrics["closed_trades"]),
            "net_pnl": int(metrics["net_pnl"]),
            "expectancy": round(metrics["net_pnl"] / metrics["closed_trades"], 2)
            if metrics["closed_trades"]
            else 0.0,
        }
        for tier, metrics in sorted(conviction_tier_rollup.items())
    }
    bull_risk_mode_summary = {
        mode: {
            "closed_trades": int(metrics["closed_trades"]),
            "net_pnl": int(metrics["net_pnl"]),
            "expectancy": round(metrics["net_pnl"] / metrics["closed_trades"], 2)
            if metrics["closed_trades"]
            else 0.0,
        }
        for mode, metrics in sorted(bull_risk_mode_rollup.items())
    }
    post_loss_admission_class_summary = {
        admission_class: {
            "closed_trades": int(metrics["closed_trades"]),
            "net_pnl": int(metrics["net_pnl"]),
            "expectancy": round(metrics["net_pnl"] / metrics["closed_trades"], 2)
            if metrics["closed_trades"]
            else 0.0,
        }
        for admission_class, metrics in sorted(post_loss_admission_class_rollup.items())
    }
    candidate_class_summary = {
        candidate_class: {
            "closed_trades": int(metrics["closed_trades"]),
            "net_pnl": int(metrics["net_pnl"]),
            "expectancy": round(metrics["net_pnl"] / metrics["closed_trades"], 2)
            if metrics["closed_trades"]
            else 0.0,
        }
        for candidate_class, metrics in sorted(candidate_class_rollup.items())
    }
    queue_source_summary = {
        key: {
            "closed_trades": int(metrics["closed_trades"]),
            "net_pnl": int(metrics["net_pnl"]),
            "expectancy": round(metrics["net_pnl"] / metrics["closed_trades"], 2)
            if metrics["closed_trades"]
            else 0.0,
        }
        for key, metrics in sorted(queue_source_rollup.items())
    }
    promotion_ready = any(
        metrics["closed_trades"] >= min_closed_trades and metrics["expectancy"] > 0
        for metrics in grade_summary.values()
    ) or any(
        metrics["closed_trades"] >= min_closed_trades and metrics["expectancy"] > 0
        for metrics in ev_summary.values()
    )
    if queue_counts:
        resolved_queue_eval_reached_counts = {
            key: min(_safe_int(router_reached_counts.get(key)), _safe_int(value))
            for key, value in sorted(queue_counts.items())
        }
    else:
        resolved_queue_eval_reached_counts = dict(sorted(queue_eval_reached_counts.items()))
    queue_total = sum(_safe_int(value) for value in queue_counts.values())
    queue_eval_total = sum(_safe_int(value) for value in resolved_queue_eval_reached_counts.values())
    queue_eval_reach_rate = round((queue_eval_total / queue_total), 4) if queue_total else 0.0
    admission_total = _safe_int(admission_counts.get("passed")) + _safe_int(admission_counts.get("blocked"))
    admission_pass_rate = round(
        (_safe_int(admission_counts.get("passed")) / admission_total),
        4,
    ) if admission_total else 0.0

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "window_days": window_days,
        "min_closed_trades": min_closed_trades,
        "grades": grade_summary,
        "ev_buckets": ev_summary,
        "leader_percentile_buckets": leader_percentile_summary,
        "shock_score_buckets": shock_score_summary,
        "adaptive_hold_buckets": adaptive_hold_summary,
        "dominant_regime_profiles": regime_profile_summary,
        "conviction_tiers": conviction_tier_summary,
        "bull_risk_modes": bull_risk_mode_summary,
        "post_loss_admission_classes": post_loss_admission_class_summary,
        "candidate_classes": candidate_class_summary,
        "queue_source_pnl": queue_source_summary,
        "stop_stats": {
            "stop_triggered_count": int(stop_triggered_count),
            "stop_limit_fallback_count": int(stop_limit_fallback_count),
            "stop_overshoot_avg_krw": round(sum(stop_overshoots) / len(stop_overshoots), 2) if stop_overshoots else 0.0,
            "stop_overshoot_max_krw": max(stop_overshoots) if stop_overshoots else 0,
        },
        "shadow_gate_counts": dict(sorted(shadow_gate_counts.items())),
        "candidate_reject_reason_outcomes": {
            reason: dict(sorted(outcomes.items()))
            for reason, outcomes in sorted(candidate_reject_reason_outcomes_rollup.items())
        },
        "regime_shadow_disagreements": dict(sorted(regime_disagreements.items())),
        "queue_counts": dict(sorted(queue_counts.items())),
        "precheck_reached_counts": dict(sorted(precheck_reached_counts.items())),
        "precheck_blocked_counts": dict(sorted(precheck_blocked_counts.items())),
        "policy_blocked_counts": dict(sorted(policy_blocked_counts.items())),
        "router_reached_counts": dict(sorted(router_reached_counts.items())),
        "router_blocked_counts": dict(sorted(router_blocked_counts.items())),
        "hard_guard_blocked_counts": dict(sorted(hard_guard_blocked_counts.items())),
        "queue_eval_reached_counts": resolved_queue_eval_reached_counts,
        "queue_eval_reach_rate": queue_eval_reach_rate,
        "admission_counts": dict(sorted(admission_counts.items())),
        "admission_pass_rate": admission_pass_rate,
        "top_leader_trades": sorted(
            leader_hits,
            key=lambda item: (item.get("leader_pct", 0.0), item.get("net_pnl", 0)),
            reverse=True,
        )[:10],
        "shadow_ready_for_promotion": promotion_ready,
    }


def render_math_shadow_markdown(payload: Dict[str, Any]) -> str:
    lines = [
        "# Signal Shadow Report",
        "",
        f"- 생성 시각: {payload.get('generated_at', '')}",
        f"- 최근 창: {_safe_int(payload.get('window_days'))}거래일",
        f"- 최소 청산 체결: {_safe_int(payload.get('min_closed_trades'))}건",
        f"- 승격 후보 여부: {'예' if payload.get('shadow_ready_for_promotion') else '아니오'}",
        "",
        "## Leader Grade",
    ]
    grades = payload.get("grades", {})
    if grades:
        for grade, metrics in grades.items():
            lines.append(
                f"- {grade}: 진입 {_safe_int(metrics.get('entries'))}건 / 청산 {_safe_int(metrics.get('closed_trades'))}건 / "
                f"순손익 {_format_currency(metrics.get('net_pnl'))} / 기대값 {_safe_float(metrics.get('expectancy')):,.2f}원"
            )
    else:
        lines.append("- 집계 없음")
    lines.extend(["", "## EV Buckets"])
    ev_buckets = payload.get("ev_buckets", {})
    if ev_buckets:
        for bucket, metrics in ev_buckets.items():
            lines.append(
                f"- {bucket}: 청산 {_safe_int(metrics.get('closed_trades'))}건 / "
                f"순손익 {_format_currency(metrics.get('net_pnl'))} / 기대값 {_safe_float(metrics.get('expectancy')):,.2f}원"
            )
    else:
        lines.append("- 집계 없음")
    lines.extend(["", "## Conviction Tier"])
    conviction_tiers = payload.get("conviction_tiers", {})
    if conviction_tiers:
        for tier, metrics in conviction_tiers.items():
            lines.append(
                f"- {tier}: 청산 {_safe_int(metrics.get('closed_trades'))}건 / "
                f"순손익 {_format_currency(metrics.get('net_pnl'))} / 기대값 {_safe_float(metrics.get('expectancy')):,.2f}원"
            )
    else:
        lines.append("- 집계 없음")
    lines.extend(["", "## Bull Risk Mode"])
    bull_risk_modes = payload.get("bull_risk_modes", {})
    if bull_risk_modes:
        for mode, metrics in bull_risk_modes.items():
            lines.append(
                f"- {mode}: 청산 {_safe_int(metrics.get('closed_trades'))}건 / "
                f"순손익 {_format_currency(metrics.get('net_pnl'))} / 기대값 {_safe_float(metrics.get('expectancy')):,.2f}원"
            )
    else:
        lines.append("- 집계 없음")
    lines.extend(["", "## Post-loss Admission"])
    post_loss_admission_classes = payload.get("post_loss_admission_classes", {})
    if post_loss_admission_classes:
        for admission_class, metrics in post_loss_admission_classes.items():
            lines.append(
                f"- {admission_class}: 청산 {_safe_int(metrics.get('closed_trades'))}건 / "
                f"순손익 {_format_currency(metrics.get('net_pnl'))} / 기대값 {_safe_float(metrics.get('expectancy')):,.2f}원"
            )
    else:
        lines.append("- 집계 없음")
    lines.extend(["", "## Candidate Class"])
    candidate_classes = payload.get("candidate_classes", {})
    if candidate_classes:
        for candidate_class, metrics in candidate_classes.items():
            lines.append(
                f"- {candidate_class}: 청산 {_safe_int(metrics.get('closed_trades'))}건 / "
                f"순손익 {_format_currency(metrics.get('net_pnl'))} / 기대값 {_safe_float(metrics.get('expectancy')):,.2f}원"
            )
    else:
        lines.append("- 집계 없음")
    lines.extend(["", "## Candidate Queue"])
    queue_counts = payload.get("queue_counts", {})
    precheck_reached = payload.get("precheck_reached_counts", {})
    precheck_blocked = payload.get("precheck_blocked_counts", {})
    policy_blocked = payload.get("policy_blocked_counts", {})
    router_reached = payload.get("router_reached_counts", {})
    router_blocked = payload.get("router_blocked_counts", {})
    hard_guard_blocked = payload.get("hard_guard_blocked_counts", {})
    queue_eval_reached = payload.get("queue_eval_reached_counts", {})
    admission_counts = payload.get("admission_counts", {})
    if queue_counts:
        for key, value in queue_counts.items():
            reached = _safe_int(queue_eval_reached.get(key))
            lines.append(f"- {key}: 큐 {_safe_int(value)}건 / 실제 평가 도달 {reached}건")
    else:
        lines.append("- 집계 없음")
    if precheck_reached:
        lines.append(
            "- precheck 도달: "
            + ", ".join(f"{key} {_safe_int(value)}건" for key, value in sorted(precheck_reached.items()))
        )
    if precheck_blocked:
        lines.append(
            "- precheck 차단: "
            + ", ".join(f"{key} {_safe_int(value)}건" for key, value in sorted(precheck_blocked.items()))
        )
    if policy_blocked:
        lines.append(
            "- policy 차단: "
            + ", ".join(f"{key} {_safe_int(value)}건" for key, value in sorted(policy_blocked.items()))
        )
    if router_reached:
        lines.append(
            "- router 도달: "
            + ", ".join(f"{key} {_safe_int(value)}건" for key, value in sorted(router_reached.items()))
        )
    if router_blocked:
        lines.append(
            "- router 차단: "
            + ", ".join(f"{key} {_safe_int(value)}건" for key, value in sorted(router_blocked.items()))
        )
    if hard_guard_blocked:
        lines.append(
            "- hard guard 차단: "
            + ", ".join(f"{key} {_safe_int(value)}건" for key, value in sorted(hard_guard_blocked.items()))
        )
    lines.append(
        f"- 큐 대비 실제 평가 도달율: {_safe_float(payload.get('queue_eval_reach_rate')) * 100:.2f}%"
    )
    lines.append(
        f"- admission 통과율: {_safe_float(payload.get('admission_pass_rate')) * 100:.2f}% "
        f"(통과 {_safe_int(admission_counts.get('passed'))}건 / 차단 {_safe_int(admission_counts.get('blocked'))}건)"
    )
    lines.extend(["", "## Queue Source PnL"])
    queue_source_pnl = payload.get("queue_source_pnl", {})
    if queue_source_pnl:
        for key, metrics in queue_source_pnl.items():
            lines.append(
                f"- {key}: 청산 {_safe_int(metrics.get('closed_trades'))}건 / "
                f"순손익 {_format_currency(metrics.get('net_pnl'))} / 기대값 {_safe_float(metrics.get('expectancy')):,.2f}원"
            )
    else:
        lines.append("- 집계 없음")
    lines.extend(["", "## Candidate Reject Outcomes"])
    candidate_reject_reason_outcomes = payload.get("candidate_reject_reason_outcomes", {})
    if candidate_reject_reason_outcomes:
        for reason, outcomes in candidate_reject_reason_outcomes.items():
            summary = ", ".join(
                f"{outcome} {_safe_int(count)}건"
                for outcome, count in sorted((outcomes or {}).items())
            ) or "-"
            lines.append(f"- {reason}: {summary}")
    else:
        lines.append("- 집계 없음")
    stop_stats = payload.get("stop_stats", {})
    lines.extend(
        [
            "",
            "## Stop Stats",
            f"- 손절 트리거: {_safe_int(stop_stats.get('stop_triggered_count'))}건",
            f"- 지정가 후 시장가 폴백: {_safe_int(stop_stats.get('stop_limit_fallback_count'))}건",
            f"- 손절 overshoot 평균: {_safe_float(stop_stats.get('stop_overshoot_avg_krw')):,.2f}원",
            f"- 손절 overshoot 최대: {_format_currency(stop_stats.get('stop_overshoot_max_krw'))}",
        ]
    )
    lines.extend(["", "## Leader Percentile Buckets"])
    leader_pct_buckets = payload.get("leader_percentile_buckets", {})
    if leader_pct_buckets:
        for bucket, metrics in leader_pct_buckets.items():
            lines.append(
                f"- {bucket}: 청산 {_safe_int(metrics.get('closed_trades'))}건 / "
                f"순손익 {_format_currency(metrics.get('net_pnl'))} / 기대값 {_safe_float(metrics.get('expectancy')):,.2f}원"
            )
    else:
        lines.append("- 집계 없음")
    lines.extend(["", "## Shock Score Buckets"])
    shock_buckets = payload.get("shock_score_buckets", {})
    if shock_buckets:
        for bucket, metrics in shock_buckets.items():
            lines.append(
                f"- {bucket}: 청산 {_safe_int(metrics.get('closed_trades'))}건 / "
                f"순손익 {_format_currency(metrics.get('net_pnl'))} / 기대값 {_safe_float(metrics.get('expectancy')):,.2f}원"
            )
    else:
        lines.append("- 집계 없음")
    lines.extend(["", "## Adaptive Hold Buckets"])
    hold_buckets = payload.get("adaptive_hold_buckets", {})
    if hold_buckets:
        for bucket, metrics in hold_buckets.items():
            lines.append(
                f"- {bucket}: 청산 {_safe_int(metrics.get('closed_trades'))}건 / "
                f"순손익 {_format_currency(metrics.get('net_pnl'))} / 기대값 {_safe_float(metrics.get('expectancy')):,.2f}원"
            )
    else:
        lines.append("- 집계 없음")
    lines.extend(["", "## Dominant Regime Profiles"])
    regime_profiles = payload.get("dominant_regime_profiles", {})
    if regime_profiles:
        for regime, metrics in regime_profiles.items():
            lines.append(
                f"- {regime}: 청산 {_safe_int(metrics.get('closed_trades'))}건 / "
                f"순손익 {_format_currency(metrics.get('net_pnl'))} / 기대값 {_safe_float(metrics.get('expectancy')):,.2f}원"
            )
    else:
        lines.append("- 집계 없음")
    lines.extend(["", "## Shadow Gates"])
    shadow_gates = payload.get("shadow_gate_counts", {})
    if shadow_gates:
        for key, value in shadow_gates.items():
            lines.append(f"- {key}: {_safe_int(value)}건")
    else:
        lines.append("- 집계 없음")
    lines.extend(["", "## Regime Disagreements"])
    disagreements = payload.get("regime_shadow_disagreements", {})
    if disagreements:
        for key, value in disagreements.items():
            lines.append(f"- {key}: {_safe_int(value)}건")
    else:
        lines.append("- 집계 없음")
    lines.extend(["", "## Top Leader Trades"])
    top_trades = payload.get("top_leader_trades", {})
    if top_trades:
        for item in top_trades:
            lines.append(
                f"- {item.get('date')} {item.get('symbol')} "
                f"{item.get('strategy_name')} grade={item.get('entry_grade_math')} "
                f"leader_pct={_safe_float(item.get('leader_pct')):.4f} "
                f"entry_ev={_safe_float(item.get('entry_ev')):,.2f} "
                f"net={_format_currency(item.get('net_pnl'))}"
            )
    else:
        lines.append("- 집계 없음")
    lines.append("")
    return "\n".join(lines)


def write_daily_scorecard(scorecard: Dict[str, Any], report_root: Path = DEFAULT_REPORT_ROOT) -> Dict[str, Path]:
    paths = _scorecard_paths(report_root, scorecard["date"])
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    paths["json"].write_text(
        json.dumps(scorecard, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths["md"].write_text(
        render_daily_scorecard_markdown(scorecard),
        encoding="utf-8",
    )
    return paths


def write_strategy_gates_report(payload: Dict[str, Any], report_root: Path = DEFAULT_REPORT_ROOT) -> Dict[str, Path]:
    report_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": report_root / STRATEGY_GATE_REPORT_JSON,
        "md": report_root / STRATEGY_GATE_REPORT_MD,
    }
    paths["json"].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths["md"].write_text(
        render_strategy_gates_markdown(payload),
        encoding="utf-8",
    )
    return paths


def write_math_shadow_report(payload: Dict[str, Any], report_root: Path = DEFAULT_REPORT_ROOT) -> Dict[str, Path]:
    report_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": report_root / MATH_SHADOW_REPORT_JSON,
        "md": report_root / MATH_SHADOW_REPORT_MD,
    }
    paths["json"].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths["md"].write_text(
        render_math_shadow_markdown(payload),
        encoding="utf-8",
    )
    return paths


def load_scorecards(report_root: Path = DEFAULT_REPORT_ROOT, limit: int = 60) -> List[Dict[str, Any]]:
    files = sorted(
        report_root.rglob(f"{DAILY_REPORT_PREFIX}*.json"),
        key=lambda path: path.name,
    )
    if limit > 0:
        files = files[-limit:]

    scorecards = []
    for path in files:
        try:
            scorecards.append(
                _reconcile_scorecard_with_log(
                    json.loads(path.read_text(encoding="utf-8"))
                )
            )
        except (OSError, json.JSONDecodeError):
            continue
    scorecards.sort(key=lambda item: item.get("date", ""))
    return scorecards


def _count_streak(scorecards: Iterable[Dict[str, Any]], positive: bool) -> int:
    streak = 0
    for card in reversed(list(scorecards)):
        day_net = _safe_int(card.get("pnl", {}).get("realized_net_pnl"))
        if positive and day_net > 0:
            streak += 1
            continue
        if not positive and day_net < 0:
            streak += 1
            continue
        break
    return streak


def _merge_scorecards(scorecards: List[Dict[str, Any]], current: Dict[str, Any]) -> List[Dict[str, Any]]:
    merged = [card for card in scorecards if card.get("date") != current.get("date")]
    merged.append(current)
    merged.sort(key=lambda item: item.get("date", ""))
    return merged


def _paper_scorecards(scorecards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [card for card in scorecards if card.get("trading_mode") == "paper"]


def _real_stage_scorecards(scorecards: List[Dict[str, Any]], stage: int) -> List[Dict[str, Any]]:
    return [
        card
        for card in scorecards
        if card.get("trading_mode") == "real"
        and _safe_int(card.get("strategy", {}).get("real_money_stage")) == stage
    ]


def evaluate_paper_trading_gate(
    scorecards: List[Dict[str, Any]],
    *,
    window_days: int = PAPER_GATE_WINDOW_DAYS,
    min_positive_days: int = PAPER_GATE_MIN_POSITIVE_DAYS,
    min_total_net_pnl: int = PAPER_GATE_MIN_TOTAL_NET_PNL,
    daily_loss_limit: int = PAPER_GATE_DAILY_LOSS_LIMIT,
    daily_profit_target: int = PAPER_GATE_DAILY_TARGET,
) -> Dict[str, Any]:
    paper_cards = _paper_scorecards(scorecards)
    window_cards = paper_cards[-window_days:]
    recorded_days = len(window_cards)
    positive_days = 0
    hard_stop_days = 0
    target_hit_days = 0
    total_net_pnl = 0
    max_day_loss = 0

    for card in window_cards:
        pnl = card.get("pnl", {})
        realized_net = _safe_int(pnl.get("realized_net_pnl", pnl.get("session_pnl", 0)))
        total_net_pnl += realized_net
        max_day_loss = min(max_day_loss, realized_net)
        if realized_net > 0:
            positive_days += 1

        risk_events = card.get("log_analysis", {}).get("risk_events", {})
        if bool(risk_events.get("daily_hard_stop_triggered")):
            hard_stop_days += 1
        if realized_net >= daily_profit_target or bool(risk_events.get("daily_profit_target_triggered")):
            target_hit_days += 1

    no_loss_breach = recorded_days == 0 or all(
        _safe_int(card.get("pnl", {}).get("realized_net_pnl", card.get("pnl", {}).get("session_pnl", 0))) >= daily_loss_limit
        for card in window_cards
    )
    criteria = {
        "window_filled": {
            "ok": recorded_days >= window_days,
            "actual": recorded_days,
            "target": window_days,
        },
        "daily_loss_respected": {
            "ok": no_loss_breach,
            "actual": max_day_loss,
            "target": daily_loss_limit,
        },
        "hard_stop_zero": {
            "ok": hard_stop_days == 0,
            "actual": hard_stop_days,
            "target": 0,
        },
        "positive_days": {
            "ok": positive_days >= min_positive_days,
            "actual": positive_days,
            "target": min_positive_days,
        },
        "rolling_net_pnl": {
            "ok": total_net_pnl >= min_total_net_pnl,
            "actual": total_net_pnl,
            "target": min_total_net_pnl,
        },
        "target_hit_days": {
            "ok": target_hit_days >= 1,
            "actual": target_hit_days,
            "target": 1,
        },
    }
    return {
        "window_days": window_days,
        "recorded_days": recorded_days,
        "first_date": window_cards[0]["date"] if window_cards else None,
        "last_date": window_cards[-1]["date"] if window_cards else None,
        "positive_days": positive_days,
        "hard_stop_days": hard_stop_days,
        "target_hit_days": target_hit_days,
        "rolling_net_pnl": total_net_pnl,
        "max_day_loss": max_day_loss,
        "criteria": criteria,
        "passed": recorded_days >= window_days and all(item.get("ok", False) for item in criteria.values()),
    }


def _evaluate_real_stage_window(scorecards: List[Dict[str, Any]], stage: int) -> Dict[str, Any]:
    rules = REAL_MONEY_STAGE_RULES[stage]
    stage_cards = _real_stage_scorecards(scorecards, stage)
    window_cards = stage_cards[-rules["days_required"]:] if rules["days_required"] > 0 else stage_cards
    hard_stop_days = sum(
        1
        for card in window_cards
        if bool(card.get("log_analysis", {}).get("risk_events", {}).get("daily_hard_stop_triggered"))
    )
    cumulative_net = sum(
        _safe_int(card.get("pnl", {}).get("realized_net_pnl", card.get("pnl", {}).get("session_pnl", 0)))
        for card in window_cards
    )
    return {
        "stage": stage,
        "label": rules["label"],
        "recorded_days": len(window_cards),
        "required_days": rules["days_required"],
        "cumulative_net_pnl": cumulative_net,
        "hard_stop_days": hard_stop_days,
        "passed": (
            len(window_cards) >= rules["days_required"]
            and cumulative_net > 0
            and hard_stop_days == 0
        ) if rules["days_required"] > 0 else False,
    }


def build_real_money_promotion_status(scorecards: List[Dict[str, Any]], paper_gate: Dict[str, Any]) -> Dict[str, Any]:
    stage_results = {
        str(stage): _evaluate_real_stage_window(scorecards, stage)
        for stage in (1, 2)
    }
    current_stage_allowed = 0
    if paper_gate.get("passed"):
        current_stage_allowed = 1
        if stage_results["1"]["passed"]:
            current_stage_allowed = 2
        if stage_results["2"]["passed"]:
            current_stage_allowed = 3

    return {
        "paper_gate_passed": bool(paper_gate.get("passed")),
        "current_stage_allowed": current_stage_allowed,
        "eligible_for_real_money_stage1": current_stage_allowed >= 1,
        "stages": {
            str(stage): {
                **REAL_MONEY_STAGE_RULES[stage],
                **stage_results.get(str(stage), {}),
            }
            for stage in REAL_MONEY_STAGE_RULES
        },
    }


def evaluate_real_trading_readiness(
    scorecards: List[Dict[str, Any]],
    *,
    min_days: int = 20,
    min_closed_trades: int = 100,
    min_profit_factor: float = 1.3,
) -> Dict[str, Any]:
    recorded_days = len(scorecards)
    profitable_days = 0
    losing_days = 0
    flat_days = 0
    total_net_pnl = 0
    total_closed_trades = 0
    total_winning_net = 0
    total_losing_net = 0

    for card in scorecards:
        pnl = card.get("pnl", {})
        trades = card.get("trades", {})
        realized_net = _safe_int(pnl.get("realized_net_pnl", pnl.get("session_pnl", 0)))
        closed_trades = _safe_int(trades.get("closed_trades"))

        total_net_pnl += realized_net
        total_closed_trades += closed_trades
        total_winning_net += _safe_int(pnl.get("winning_net_pnl_sum"))
        total_losing_net += _safe_int(pnl.get("losing_net_pnl_sum"))

        if realized_net > 0:
            profitable_days += 1
        elif realized_net < 0:
            losing_days += 1
        else:
            flat_days += 1

    if total_losing_net < 0:
        aggregate_profit_factor: Optional[float] = total_winning_net / abs(total_losing_net) if total_winning_net > 0 else 0.0
    elif total_winning_net > 0:
        aggregate_profit_factor = None
    else:
        aggregate_profit_factor = 0.0

    expectancy = (total_net_pnl / total_closed_trades) if total_closed_trades > 0 else 0.0
    sample_size_ok = recorded_days >= min_days or total_closed_trades >= min_closed_trades
    net_pnl_ok = total_net_pnl > 0
    expectancy_ok = expectancy > 0
    if aggregate_profit_factor is None:
        profit_factor_ok = total_winning_net > 0
    else:
        profit_factor_ok = aggregate_profit_factor >= min_profit_factor

    latest_date = scorecards[-1]["date"] if scorecards else None
    first_date = scorecards[0]["date"] if scorecards else None

    paper_gate = evaluate_paper_trading_gate(scorecards)
    promotion = build_real_money_promotion_status(scorecards, paper_gate)

    readiness = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "window": {
            "first_date": first_date,
            "last_date": latest_date,
            "recorded_days": recorded_days,
            "profitable_days": profitable_days,
            "losing_days": losing_days,
            "flat_days": flat_days,
            "current_profit_streak": _count_streak(scorecards, positive=True),
            "current_loss_streak": _count_streak(scorecards, positive=False),
        },
        "aggregate": {
            "total_net_pnl": total_net_pnl,
            "total_closed_trades": total_closed_trades,
            "total_winning_net_pnl": total_winning_net,
            "total_losing_net_pnl": total_losing_net,
            "profit_factor": round(aggregate_profit_factor, 4) if aggregate_profit_factor is not None else None,
            "expectancy_net_per_trade": round(expectancy, 2),
        },
        "criteria": {
            "sample_size": {
                "ok": sample_size_ok,
                "recorded_days": recorded_days,
                "min_days": min_days,
                "total_closed_trades": total_closed_trades,
                "min_closed_trades": min_closed_trades,
                "rule": "recorded_days >= min_days or total_closed_trades >= min_closed_trades",
            },
            "net_pnl_positive": {
                "ok": net_pnl_ok,
                "actual": total_net_pnl,
                "target": 0,
            },
            "profit_factor": {
                "ok": profit_factor_ok,
                "actual": round(aggregate_profit_factor, 4) if aggregate_profit_factor is not None else None,
                "target": min_profit_factor,
            },
            "expectancy_positive": {
                "ok": expectancy_ok,
                "actual": round(expectancy, 2),
                "target": 0,
            },
        },
        "paper_gate": paper_gate,
        "promotion": promotion,
    }
    readiness["ready_for_real_trading"] = (
        all(item.get("ok", False) for item in readiness["criteria"].values())
        and promotion.get("current_stage_allowed", 0) >= 3
    )
    return readiness


def render_readiness_markdown(readiness: Dict[str, Any]) -> str:
    window = readiness.get("window", {})
    aggregate = readiness.get("aggregate", {})
    criteria = readiness.get("criteria", {})
    paper_gate = readiness.get("paper_gate", {})
    promotion = readiness.get("promotion", {})

    def mark(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    lines = [
        "# 실투자 전환 게이트",
        "",
        f"- 생성 시각: {readiness.get('generated_at', '')}",
        f"- 분석 구간: {window.get('first_date') or '-'} ~ {window.get('last_date') or '-'}",
        f"- 기록 일수: {_safe_int(window.get('recorded_days'))}일",
        f"- 수익일/손실일/보합일: {_safe_int(window.get('profitable_days'))}/{_safe_int(window.get('losing_days'))}/{_safe_int(window.get('flat_days'))}",
        f"- 현재 수익 연속일수: {_safe_int(window.get('current_profit_streak'))}일",
        f"- 현재 손실 연속일수: {_safe_int(window.get('current_loss_streak'))}일",
        f"- 누적 순손익: {_format_currency(aggregate.get('total_net_pnl'))}",
        f"- 누적 청산 체결 수: {_safe_int(aggregate.get('total_closed_trades'))}건",
        f"- 누적 Profit Factor: {_format_profit_factor(aggregate.get('profit_factor'))}",
        f"- 거래당 기대값: {_safe_float(aggregate.get('expectancy_net_per_trade')):,.2f}원",
        "",
        "## Paper Gate",
        f"- 최근 { _safe_int(paper_gate.get('window_days')) }거래일: {paper_gate.get('first_date') or '-'} ~ {paper_gate.get('last_date') or '-'}",
        f"- 누적 순손익: {_format_currency(paper_gate.get('rolling_net_pnl'))}",
        f"- 플러스 일수: {_safe_int(paper_gate.get('positive_days'))}일",
        f"- 하드스탑 발생: {_safe_int(paper_gate.get('hard_stop_days'))}일",
        f"- 목표 달성 일수: {_safe_int(paper_gate.get('target_hit_days'))}일",
        f"- Paper Gate 통과: {mark(bool(paper_gate.get('passed')))}",
        "",
        "## 실계좌 단계",
        f"- 현재 허용 단계: {promotion.get('current_stage_allowed', 0)}",
        f"- 1단계 진입 가능: {mark(bool(promotion.get('eligible_for_real_money_stage1')))}",
        "",
        "## 판정",
        f"- 최종 판정: {'실투자 전환 가능' if readiness.get('ready_for_real_trading') else '실투자 전환 보류'}",
        f"- 표본 크기: {mark(criteria.get('sample_size', {}).get('ok', False))}",
        f"- 누적 순손익 플러스: {mark(criteria.get('net_pnl_positive', {}).get('ok', False))}",
        f"- Profit Factor 기준 충족: {mark(criteria.get('profit_factor', {}).get('ok', False))}",
        f"- 거래당 기대값 플러스: {mark(criteria.get('expectancy_positive', {}).get('ok', False))}",
        "",
    ]
    for stage in (1, 2, 3):
        stage_payload = promotion.get("stages", {}).get(str(stage), {})
        lines.append(
            "- 단계 {stage}: scale {scale:.0%}, 손실한도 {loss}, 수익보호 {protect}, 목표 {target}, "
            "기록 {days}일, 누적 {net}, 하드스탑 {hard}, 통과 {passed}".format(
                stage=stage,
                scale=_safe_float(stage_payload.get("capital_scale"), 0.0),
                loss=_format_currency(stage_payload.get("daily_loss_limit")),
                protect=_format_currency(stage_payload.get("profit_protect_threshold")),
                target=_format_currency(stage_payload.get("daily_profit_target")),
                days=_safe_int(stage_payload.get("recorded_days")),
                net=_format_currency(stage_payload.get("cumulative_net_pnl")),
                hard=_safe_int(stage_payload.get("hard_stop_days")),
                passed=mark(bool(stage_payload.get("passed", stage == 3 and promotion.get("current_stage_allowed", 0) >= 3))),
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_readiness_report(readiness: Dict[str, Any], report_root: Path = DEFAULT_REPORT_ROOT) -> Dict[str, Path]:
    report_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": report_root / READINESS_REPORT_JSON,
        "md": report_root / READINESS_REPORT_MD,
    }
    paths["json"].write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths["md"].write_text(
        render_readiness_markdown(readiness),
        encoding="utf-8",
    )
    return paths


def update_performance_reports(
    strategy: Any,
    balance: Any,
    session_pnl: int,
    trading_mode: str,
    report_root: Path = DEFAULT_REPORT_ROOT,
    log_root: Path = DEFAULT_LOG_ROOT,
) -> Dict[str, Dict[str, Path]]:
    existing_scorecards = load_scorecards(report_root=report_root)
    scorecard = build_daily_scorecard(
        strategy=strategy,
        balance=balance,
        session_pnl=session_pnl,
        trading_mode=trading_mode,
        log_root=log_root,
    )
    merged_scorecards = _merge_scorecards(existing_scorecards, scorecard)
    scorecard["paper_gate"] = evaluate_paper_trading_gate(merged_scorecards)
    scorecard_paths = write_daily_scorecard(scorecard, report_root=report_root)
    strategy_cfg = getattr(strategy, "cfg", None) or getattr(strategy, "config", None)
    strategy_gates = evaluate_strategy_gates(
        merged_scorecards,
        window_days=_safe_int(
            getattr(strategy_cfg, "strategy_gate_window_days", DEFAULT_STRATEGY_GATE_WINDOW_DAYS),
            DEFAULT_STRATEGY_GATE_WINDOW_DAYS,
        ),
        min_closed_trades=_safe_int(
            getattr(strategy_cfg, "strategy_gate_min_closed_trades", DEFAULT_STRATEGY_GATE_MIN_CLOSED_TRADES),
            DEFAULT_STRATEGY_GATE_MIN_CLOSED_TRADES,
        ),
        disable_expectancy_threshold=_safe_float(
            getattr(strategy_cfg, "strategy_gate_disable_expectancy_threshold", -150.0),
            -150.0,
        ),
    )
    strategy_gate_paths = write_strategy_gates_report(strategy_gates, report_root=report_root)
    math_shadow = evaluate_math_shadow_report(
        merged_scorecards,
        window_days=_safe_int(
            getattr(strategy_cfg, "ev_window_days", DEFAULT_STRATEGY_GATE_WINDOW_DAYS),
            DEFAULT_STRATEGY_GATE_WINDOW_DAYS,
        ),
        min_closed_trades=_safe_int(
            getattr(strategy_cfg, "ev_min_samples", DEFAULT_STRATEGY_GATE_MIN_CLOSED_TRADES),
            DEFAULT_STRATEGY_GATE_MIN_CLOSED_TRADES,
        ),
    )
    math_shadow_paths = write_math_shadow_report(math_shadow, report_root=report_root)
    readiness = evaluate_real_trading_readiness(merged_scorecards)
    readiness_paths = write_readiness_report(readiness, report_root=report_root)
    return {
        "scorecard": scorecard_paths,
        "strategy_gates": strategy_gate_paths,
        "math_shadow": math_shadow_paths,
        "readiness": readiness_paths,
    }


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _print_today_scorecard(report_root: Path, report_date: Optional[str] = None) -> int:
    target_date = report_date or datetime.now().date().isoformat()
    paths = _scorecard_paths(report_root, target_date)
    payload = _load_json(paths["json"])
    if payload is None:
        print("오늘 성적표가 아직 없습니다.")
        return 1

    pnl = payload.get("pnl", {})
    trades = payload.get("trades", {})
    paper_gate = payload.get("paper_gate", {})
    print("=== 오늘 성적표 ===")
    print(f"→ 날짜: {payload.get('date', '')}")
    print(f"→ 생성 시각: {payload.get('generated_at', '')}")
    print(f"→ 세션 손익: {_format_currency(pnl.get('session_pnl'))}")
    print(f"→ 순실현손익: {_format_currency(pnl.get('realized_net_pnl'))}")
    print(f"→ 청산 체결: {_safe_int(trades.get('closed_trades'))}건")
    print(
        "→ 승/패/보합: "
        f"{_safe_int(trades.get('wins'))}/{_safe_int(trades.get('losses'))}/{_safe_int(trades.get('breakeven'))}"
    )
    print(f"→ Profit Factor: {_format_profit_factor(trades.get('profit_factor'))}")
    print(f"→ 평균 순손익/거래: {_safe_float(trades.get('average_net_per_trade')):,.2f}원")
    if paper_gate:
        print(
            "→ 최근 5거래일 Gate: "
            f"{'PASS' if paper_gate.get('passed') else 'FAIL'} "
            f"(누적 {_format_currency(paper_gate.get('rolling_net_pnl'))}, "
            f"플러스 {_safe_int(paper_gate.get('positive_days'))}일)"
        )
    print(f"→ 상세 파일: {paths['md']}")
    return 0


def _print_readiness(report_root: Path) -> int:
    readiness_path = report_root / READINESS_REPORT_JSON
    payload = _load_json(readiness_path)
    if payload is None:
        scorecards = load_scorecards(report_root=report_root)
        if not scorecards:
            print("실투자 전환 게이트를 계산할 성적표가 아직 없습니다.")
            return 1
        payload = evaluate_real_trading_readiness(scorecards)

    window = payload.get("window", {})
    aggregate = payload.get("aggregate", {})
    paper_gate = payload.get("paper_gate", {})
    promotion = payload.get("promotion", {})
    print("=== 실투자 전환 게이트 ===")
    print(
        "→ 구간: "
        f"{window.get('first_date') or '-'} ~ {window.get('last_date') or '-'}"
    )
    print(f"→ 기록 일수: {_safe_int(window.get('recorded_days'))}일")
    print(f"→ 누적 순손익: {_format_currency(aggregate.get('total_net_pnl'))}")
    print(f"→ 누적 청산 체결: {_safe_int(aggregate.get('total_closed_trades'))}건")
    print(f"→ Profit Factor: {_format_profit_factor(aggregate.get('profit_factor'))}")
    print(f"→ 거래당 기대값: {_safe_float(aggregate.get('expectancy_net_per_trade')):,.2f}원")
    print(
        "→ Paper Gate: "
        f"{'PASS' if paper_gate.get('passed') else 'FAIL'} "
        f"(최근 {_safe_int(paper_gate.get('recorded_days'))}일, "
        f"누적 {_format_currency(paper_gate.get('rolling_net_pnl'))})"
    )
    print(f"→ 현재 허용 실계좌 단계: {promotion.get('current_stage_allowed', 0)}")
    print(
        "→ 최종 판정: "
        f"{'실투자 전환 가능' if payload.get('ready_for_real_trading') else '실투자 전환 보류'}"
    )
    print(f"→ 상세 파일: {report_root / READINESS_REPORT_MD}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="성과 리포트 조회")
    parser.add_argument(
        "--report-root",
        default=str(DEFAULT_REPORT_ROOT),
        help="리포트 루트 경로",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    today_parser = subparsers.add_parser("today", help="오늘 성적표 출력")
    today_parser.add_argument("--date", help="조회할 날짜 (YYYY-MM-DD)")

    subparsers.add_parser("gate", help="실투자 전환 게이트 출력")

    args = parser.parse_args(argv)
    report_root = Path(args.report_root).expanduser()

    if args.command == "today":
        return _print_today_scorecard(report_root=report_root, report_date=args.date)
    if args.command == "gate":
        return _print_readiness(report_root=report_root)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
