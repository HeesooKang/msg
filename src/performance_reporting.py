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
PAPER_GATE_WINDOW_DAYS = 5
PAPER_GATE_MIN_POSITIVE_DAYS = 3
PAPER_GATE_MIN_TOTAL_NET_PNL = 10_000
PAPER_GATE_DAILY_LOSS_LIMIT = -5_000
PAPER_GATE_DAILY_TARGET = 10_000
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
_INV_SIGNAL_SYMBOL_RE = re.compile(r"\[INV\]\s*매수 신호:\s*([0-9A-Z]+)")
_SELL_SYMBOL_RE = re.compile(r"(?:\[INV\]\s*)?매도 체결:\s*([0-9A-Z]+)\s")
_SELL_NET_PNL_RE = re.compile(r"순손익:\s*([-\d,]+)원")
_SELL_PNL_RE = re.compile(r"손익:\s*([-\d,]+)원")
_RISK_STAGE_RE = re.compile(r"리스크 단계 전환:\s*([a-zA-Z0-9_]+)")


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
    return scorecard


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
    return None


def _extract_sell_symbol(message: str) -> Optional[str]:
    match = _SELL_SYMBOL_RE.search(message)
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
    entry_by_reason: Dict[str, int] = defaultdict(int)
    entry_by_regime: Dict[str, int] = defaultdict(int)
    reject_by_reason: Dict[str, int] = defaultdict(int)
    setup_pnl: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"closed_trades": 0, "net_pnl": 0, "wins": 0, "losses": 0}
    )
    regime_pnl: Dict[str, int] = defaultdict(int)
    symbol_pnl: Dict[str, int] = defaultdict(int)
    risk_stage_transitions: Dict[str, int] = defaultdict(int)
    active_entries: Dict[str, Dict[str, str]] = {}
    daily_hard_stop_triggered = False
    daily_profit_target_triggered = False

    for raw_line in lines:
        message = _extract_log_message(raw_line)
        if not message:
            continue

        risk_stage_match = _RISK_STAGE_RE.search(message)
        if risk_stage_match:
            risk_stage_transitions[risk_stage_match.group(1)] += 1

        if "일일 총손익 하드스탑 도달!" in message:
            daily_hard_stop_triggered = True
        if "일일 총손익 목표 달성!" in message:
            daily_profit_target_triggered = True

        reject_match = _REJECT_REASON_RE.search(message)
        if reject_match:
            reject_by_reason[reject_match.group(1)] += 1

        setup_match = _SETUP_NAME_RE.search(message)
        entry_reason_match = _ENTRY_REASON_RE.search(message)
        regime_match = _REGIME_LABEL_RE.search(message)
        if "매수 신호:" in message and setup_match:
            setup_name = setup_match.group(1)
            entry_by_setup[setup_name] += 1
            if entry_reason_match:
                entry_by_reason[entry_reason_match.group(1)] += 1
            regime_label = regime_match.group(1) if regime_match else "unknown"
            entry_by_regime[regime_label] += 1
            symbol = _extract_signal_symbol(message)
            if symbol:
                active_entries[symbol] = {
                    "setup_name": setup_name,
                    "regime_label": regime_label,
                }

        if "매도 체결:" in message:
            symbol = _extract_sell_symbol(message)
            if not symbol:
                continue
            entry_meta = active_entries.pop(symbol, {})
            setup_name = entry_meta.get("setup_name") or _extract_context_token(message, "setup_name", "unknown")
            regime_label = entry_meta.get("regime_label") or _extract_context_token(message, "regime_label", "unknown")
            net_pnl = _extract_sell_net_pnl(message)
            metrics = setup_pnl[setup_name]
            metrics["closed_trades"] += 1
            metrics["net_pnl"] += net_pnl
            regime_pnl[regime_label] += net_pnl
            symbol_pnl[symbol] += net_pnl
            if net_pnl > 0:
                metrics["wins"] += 1
            elif net_pnl < 0:
                metrics["losses"] += 1

    sorted_symbols = sorted(symbol_pnl.items(), key=lambda item: (item[1], item[0]), reverse=True)
    top_winners = [
        {"symbol": symbol, "net_pnl": net_pnl}
        for symbol, net_pnl in sorted_symbols[:5]
    ]
    top_losers = [
        {"symbol": symbol, "net_pnl": net_pnl}
        for symbol, net_pnl in sorted(symbol_pnl.items(), key=lambda item: (item[1], item[0]))[:5]
    ]

    return {
        "log_path": str(selected_path) if selected_path else None,
        "entries": {
            "total": int(sum(entry_by_setup.values())),
            "by_setup": dict(sorted(entry_by_setup.items())),
            "by_entry_reason": dict(sorted(entry_by_reason.items())),
            "by_regime": dict(sorted(entry_by_regime.items())),
        },
        "rejections": {
            "total": int(sum(reject_by_reason.values())),
            "by_reason": dict(sorted(reject_by_reason.items())),
        },
        "setup_pnl": {
            setup: metrics
            for setup, metrics in sorted(setup_pnl.items())
        },
        "regime_pnl": dict(sorted(regime_pnl.items())),
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
    setup_pnl = log_analysis.get("setup_pnl", {})
    regime_pnl = log_analysis.get("regime_pnl", {})
    symbols = log_analysis.get("symbols", {})
    risk_events = log_analysis.get("risk_events", {})
    if entries or rejections or setup_pnl:
        setup_entry_summary = ", ".join(
            f"{key} {_safe_int(value)}건"
            for key, value in entries.get("by_setup", {}).items()
        ) or "-"
        rejection_summary = ", ".join(
            f"{key} {_safe_int(value)}건"
            for key, value in rejections.get("by_reason", {}).items()
        ) or "-"
        setup_pnl_summary = ", ".join(
            f"{key} {_format_currency(value.get('net_pnl'))} / {_safe_int(value.get('closed_trades'))}건"
            for key, value in setup_pnl.items()
        ) or "-"
        regime_pnl_summary = ", ".join(
            f"{key} {_format_currency(value)}"
            for key, value in regime_pnl.items()
        ) or "-"
        top_winners_summary = ", ".join(
            f"{item.get('symbol')} {_format_currency(item.get('net_pnl'))}"
            for item in symbols.get("top_winners", [])
        ) or "-"
        top_losers_summary = ", ".join(
            f"{item.get('symbol')} {_format_currency(item.get('net_pnl'))}"
            for item in symbols.get("top_losers", [])
        ) or "-"
        lines.extend(
            [
                "## 로그 분석",
                f"- 분석 로그: {log_analysis.get('log_path') or '-'}",
                f"- 진입 신호 수: {_safe_int(entries.get('total'))}건",
                f"- 진입 셋업별: {setup_entry_summary}",
                f"- 차단 사유 수: {_safe_int(rejections.get('total'))}건",
                f"- 차단 사유별: {rejection_summary}",
                f"- 셋업별 순손익: {setup_pnl_summary}",
                f"- 레짐별 순손익: {regime_pnl_summary}",
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
            scorecards.append(json.loads(path.read_text(encoding="utf-8")))
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
    readiness = evaluate_real_trading_readiness(merged_scorecards)
    readiness_paths = write_readiness_report(readiness, report_root=report_root)
    return {
        "scorecard": scorecard_paths,
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
