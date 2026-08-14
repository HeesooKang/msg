import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_REPORT_ROOT = Path("reports")
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
        "daily_profit_target": 2_500,
    },
    2: {
        "label": "stage2",
        "capital_scale": 0.50,
        "days_required": 5,
        "daily_loss_limit": -2_500,
        "daily_profit_target": 5_000,
    },
    3: {
        "label": "stage3",
        "capital_scale": 1.00,
        "days_required": 0,
        "daily_loss_limit": -5_000,
        "daily_profit_target": 10_000,
    },
}


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


def _format_currency(value: Any) -> str:
    return f"{_safe_int(value):,}원"


def _format_profit_factor(value: Any) -> str:
    return "무손실" if value is None else f"{_safe_float(value):.2f}"


def _extract_strategy_snapshot(strategy: Any) -> Dict[str, Any]:
    daily_pnl = strategy.daily_pnl
    snapshot = {
        "realized_gross_pnl": _safe_int(daily_pnl.realized_gross_pnl),
        "realized_net_pnl": _safe_int(daily_pnl.realized_net_pnl),
        "fees_paid": _safe_int(daily_pnl.fees_paid),
        "taxes_paid": _safe_int(daily_pnl.taxes_paid),
        "trade_count": _safe_int(daily_pnl.trade_count),
        "win_count": _safe_int(daily_pnl.win_count),
        "loss_count": _safe_int(daily_pnl.loss_count),
        "breakeven_count": _safe_int(daily_pnl.breakeven_count),
        "winning_net_pnl_sum": _safe_int(daily_pnl.winning_net_pnl_sum),
        "losing_net_pnl_sum": _safe_int(daily_pnl.losing_net_pnl_sum),
        "largest_win_net": _safe_int(daily_pnl.largest_win_net),
        "largest_loss_net": _safe_int(daily_pnl.largest_loss_net),
    }
    snapshot.update(
        {
            "halted": bool(strategy._halted),
            "halt_reason": str(strategy._halt_reason or ""),
            "open_positions_count": len(strategy.positions),
        }
    )
    return snapshot


def _compute_trade_metrics(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    trades = _safe_int(snapshot.get("trade_count"))
    wins = _safe_int(snapshot.get("win_count"))
    losses = _safe_int(snapshot.get("loss_count"))
    winning_net = _safe_int(snapshot.get("winning_net_pnl_sum"))
    losing_net = _safe_int(snapshot.get("losing_net_pnl_sum"))
    realized_net = _safe_int(snapshot.get("realized_net_pnl"))
    if losing_net < 0:
        profit_factor: Optional[float] = (
            winning_net / abs(losing_net) if winning_net > 0 else 0.0
        )
    elif winning_net > 0:
        profit_factor = None
    else:
        profit_factor = 0.0
    return {
        "closed_trades": trades,
        "wins": wins,
        "losses": losses,
        "breakeven": _safe_int(snapshot.get("breakeven_count")),
        "win_rate": round(wins / trades, 4) if trades else 0.0,
        "average_net_per_trade": round(realized_net / trades, 2) if trades else 0.0,
        "average_win_net": round(winning_net / wins, 2) if wins else 0.0,
        "average_loss_net": round(losing_net / losses, 2) if losses else 0.0,
        "profit_factor": round(profit_factor, 4) if profit_factor is not None else None,
    }


def build_daily_scorecard(
    strategy: Any,
    balance: Any,
    session_pnl: int,
    trading_mode: str,
    generated_at: Optional[datetime] = None,
) -> Dict[str, Any]:
    now = generated_at or datetime.now()
    snapshot = _extract_strategy_snapshot(strategy)
    return {
        "date": now.date().isoformat(),
        "generated_at": now.isoformat(timespec="seconds"),
        "trading_mode": trading_mode,
        "balance": {
            "total_eval_amount": _safe_int(balance.total_eval_amount),
            "total_deposit": _safe_int(balance.total_deposit),
            "total_profit_loss": _safe_int(balance.total_profit_loss),
        },
        "pnl": {
            "session_pnl": _safe_int(session_pnl),
            "realized_gross_pnl": snapshot["realized_gross_pnl"],
            "realized_net_pnl": snapshot["realized_net_pnl"],
            "fees_paid": snapshot["fees_paid"],
            "taxes_paid": snapshot["taxes_paid"],
            "winning_net_pnl_sum": snapshot["winning_net_pnl_sum"],
            "losing_net_pnl_sum": snapshot["losing_net_pnl_sum"],
            "largest_win_net": snapshot["largest_win_net"],
            "largest_loss_net": snapshot["largest_loss_net"],
            "source": "confirmed_sell_fill_ledger",
        },
        "trades": _compute_trade_metrics(snapshot),
        "strategy": {
            "name": "expected_value_long",
            "halted": snapshot["halted"],
            "halt_reason": snapshot["halt_reason"],
            "open_positions_count": snapshot["open_positions_count"],
            "real_money_stage": _safe_int(strategy._real_money_stage),
            "capital_scale": round(
                _safe_float(strategy._capital_scale, 1.0),
                4,
            ),
        },
    }


def _scorecard_paths(report_root: Path, report_date: str) -> Dict[str, Path]:
    year, month, _day = report_date.split("-")
    directory = report_root / year / month
    return {
        "json": directory / f"{DAILY_REPORT_PREFIX}{report_date}.json",
        "md": directory / f"{DAILY_REPORT_PREFIX}{report_date}.md",
    }


def render_daily_scorecard_markdown(scorecard: Dict[str, Any]) -> str:
    pnl = scorecard.get("pnl", {})
    trades = scorecard.get("trades", {})
    strategy = scorecard.get("strategy", {})
    return "\n".join(
        [
            f"# 일일 성과 {scorecard.get('date', '')}",
            "",
            f"- 생성 시각: {scorecard.get('generated_at', '')}",
            f"- 거래 모드: {scorecard.get('trading_mode', '')}",
            f"- 확정 순실현손익: {_format_currency(pnl.get('realized_net_pnl'))}",
            f"- 수수료/세금: {_format_currency(pnl.get('fees_paid'))} / {_format_currency(pnl.get('taxes_paid'))}",
            f"- 청산 거래: {_safe_int(trades.get('closed_trades'))}건",
            f"- 승/패/보합: {_safe_int(trades.get('wins'))}/{_safe_int(trades.get('losses'))}/{_safe_int(trades.get('breakeven'))}",
            f"- 거래당 평균 순손익: {_safe_float(trades.get('average_net_per_trade')):,.2f}원",
            f"- Profit Factor: {_format_profit_factor(trades.get('profit_factor'))}",
            f"- 종료 사유: {strategy.get('halt_reason') or '-'}",
            "",
        ]
    )


def write_daily_scorecard(
    scorecard: Dict[str, Any],
    report_root: Path = DEFAULT_REPORT_ROOT,
) -> Dict[str, Path]:
    paths = _scorecard_paths(report_root, str(scorecard["date"]))
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


def load_scorecards(
    report_root: Path = DEFAULT_REPORT_ROOT,
    limit: int = 60,
) -> List[Dict[str, Any]]:
    files = sorted(
        report_root.rglob(f"{DAILY_REPORT_PREFIX}*.json"),
        key=lambda path: path.name,
    )
    if limit > 0:
        files = files[-limit:]
    scorecards = []
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            scorecards.append(payload)
    return sorted(scorecards, key=lambda item: str(item.get("date", "")))


def _merge_scorecards(
    scorecards: List[Dict[str, Any]],
    current: Dict[str, Any],
) -> List[Dict[str, Any]]:
    merged = [card for card in scorecards if card.get("date") != current.get("date")]
    merged.append(current)
    return sorted(merged, key=lambda item: str(item.get("date", "")))


def _day_net(card: Dict[str, Any]) -> int:
    return _safe_int(card.get("pnl", {}).get("realized_net_pnl", 0))


def _halt_reason(card: Dict[str, Any]) -> str:
    return str(card.get("strategy", {}).get("halt_reason") or "")


def _count_streak(scorecards: Iterable[Dict[str, Any]], positive: bool) -> int:
    streak = 0
    for card in reversed(list(scorecards)):
        day_net = _day_net(card)
        if (positive and day_net > 0) or (not positive and day_net < 0):
            streak += 1
        else:
            break
    return streak


def evaluate_paper_trading_gate(
    scorecards: List[Dict[str, Any]],
    *,
    window_days: int = PAPER_GATE_WINDOW_DAYS,
    min_positive_days: int = PAPER_GATE_MIN_POSITIVE_DAYS,
    min_total_net_pnl: int = PAPER_GATE_MIN_TOTAL_NET_PNL,
    daily_loss_limit: int = PAPER_GATE_DAILY_LOSS_LIMIT,
    daily_profit_target: int = PAPER_GATE_DAILY_TARGET,
) -> Dict[str, Any]:
    paper_cards = [card for card in scorecards if card.get("trading_mode") == "paper"]
    window = paper_cards[-window_days:]
    nets = [_day_net(card) for card in window]
    hard_stop_days = sum(
        _halt_reason(card) == "daily_total_loss_limit" for card in window
    )
    target_hit_days = sum(
        net >= daily_profit_target or _halt_reason(card) == "daily_profit_target"
        for card, net in zip(window, nets)
    )
    positive_days = sum(net > 0 for net in nets)
    total_net = sum(nets)
    max_day_loss = min([0, *nets])
    criteria = {
        "window_filled": len(window) >= window_days,
        "daily_loss_respected": all(net >= daily_loss_limit for net in nets),
        "hard_stop_zero": hard_stop_days == 0,
        "positive_days": positive_days >= min_positive_days,
        "rolling_net_pnl": total_net >= min_total_net_pnl,
        "target_hit_days": target_hit_days >= 1,
    }
    return {
        "window_days": window_days,
        "recorded_days": len(window),
        "first_date": window[0].get("date") if window else None,
        "last_date": window[-1].get("date") if window else None,
        "positive_days": positive_days,
        "hard_stop_days": hard_stop_days,
        "target_hit_days": target_hit_days,
        "rolling_net_pnl": total_net,
        "max_day_loss": max_day_loss,
        "criteria": criteria,
        "passed": len(window) >= window_days and all(criteria.values()),
    }


def _real_stage_cards(
    scorecards: List[Dict[str, Any]],
    stage: int,
) -> List[Dict[str, Any]]:
    return [
        card
        for card in scorecards
        if card.get("trading_mode") == "real"
        and _safe_int(card.get("strategy", {}).get("real_money_stage")) == stage
    ]


def _evaluate_real_stage_window(
    scorecards: List[Dict[str, Any]],
    stage: int,
) -> Dict[str, Any]:
    rules = REAL_MONEY_STAGE_RULES[stage]
    stage_cards = _real_stage_cards(scorecards, stage)
    required = _safe_int(rules["days_required"])
    window = stage_cards[-required:] if required else stage_cards
    cumulative_net = sum(_day_net(card) for card in window)
    hard_stops = sum(
        _halt_reason(card) == "daily_total_loss_limit" for card in window
    )
    return {
        "stage": stage,
        "label": rules["label"],
        "recorded_days": len(window),
        "required_days": required,
        "cumulative_net_pnl": cumulative_net,
        "hard_stop_days": hard_stops,
        "passed": bool(
            required
            and len(window) >= required
            and cumulative_net > 0
            and hard_stops == 0
        ),
    }


def build_real_money_promotion_status(
    scorecards: List[Dict[str, Any]],
    paper_gate: Dict[str, Any],
) -> Dict[str, Any]:
    stage_results = {
        str(stage): _evaluate_real_stage_window(scorecards, stage)
        for stage in (1, 2)
    }
    allowed = 0
    if paper_gate.get("passed"):
        allowed = 1
        if stage_results["1"]["passed"]:
            allowed = 2
        if stage_results["2"]["passed"]:
            allowed = 3
    return {
        "paper_gate_passed": bool(paper_gate.get("passed")),
        "current_stage_allowed": allowed,
        "eligible_for_real_money_stage1": allowed >= 1,
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
    total_net = sum(_day_net(card) for card in scorecards)
    total_trades = sum(
        _safe_int(card.get("trades", {}).get("closed_trades"))
        for card in scorecards
    )
    winning_net = sum(
        _safe_int(card.get("pnl", {}).get("winning_net_pnl_sum"))
        for card in scorecards
    )
    losing_net = sum(
        _safe_int(card.get("pnl", {}).get("losing_net_pnl_sum"))
        for card in scorecards
    )
    if losing_net < 0:
        profit_factor: Optional[float] = (
            winning_net / abs(losing_net) if winning_net > 0 else 0.0
        )
    elif winning_net > 0:
        profit_factor = None
    else:
        profit_factor = 0.0
    expectancy = total_net / total_trades if total_trades else 0.0
    paper_gate = evaluate_paper_trading_gate(scorecards)
    promotion = build_real_money_promotion_status(scorecards, paper_gate)
    criteria = {
        "sample_size": len(scorecards) >= min_days or total_trades >= min_closed_trades,
        "net_pnl_positive": total_net > 0,
        "profit_factor": (
            winning_net > 0
            if profit_factor is None
            else profit_factor >= min_profit_factor
        ),
        "expectancy_positive": expectancy > 0,
    }
    readiness = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "window": {
            "first_date": scorecards[0].get("date") if scorecards else None,
            "last_date": scorecards[-1].get("date") if scorecards else None,
            "recorded_days": len(scorecards),
            "profitable_days": sum(_day_net(card) > 0 for card in scorecards),
            "losing_days": sum(_day_net(card) < 0 for card in scorecards),
            "flat_days": sum(_day_net(card) == 0 for card in scorecards),
            "current_profit_streak": _count_streak(scorecards, positive=True),
            "current_loss_streak": _count_streak(scorecards, positive=False),
        },
        "aggregate": {
            "total_net_pnl": total_net,
            "total_closed_trades": total_trades,
            "total_winning_net_pnl": winning_net,
            "total_losing_net_pnl": losing_net,
            "profit_factor": round(profit_factor, 4) if profit_factor is not None else None,
            "expectancy_net_per_trade": round(expectancy, 2),
        },
        "criteria": criteria,
        "paper_gate": paper_gate,
        "promotion": promotion,
    }
    readiness["ready_for_real_trading"] = (
        all(criteria.values()) and promotion["current_stage_allowed"] >= 3
    )
    return readiness


def render_readiness_markdown(readiness: Dict[str, Any]) -> str:
    window = readiness.get("window", {})
    aggregate = readiness.get("aggregate", {})
    paper_gate = readiness.get("paper_gate", {})
    promotion = readiness.get("promotion", {})
    return "\n".join(
        [
            "# 실투자 전환 상태",
            "",
            f"- 분석 구간: {window.get('first_date') or '-'} ~ {window.get('last_date') or '-'}",
            f"- 기록 일수: {_safe_int(window.get('recorded_days'))}일",
            f"- 누적 순손익: {_format_currency(aggregate.get('total_net_pnl'))}",
            f"- 누적 청산 거래: {_safe_int(aggregate.get('total_closed_trades'))}건",
            f"- 거래당 기대값: {_safe_float(aggregate.get('expectancy_net_per_trade')):,.2f}원",
            f"- Paper gate: {'PASS' if paper_gate.get('passed') else 'FAIL'}",
            f"- 현재 허용 실계좌 단계: {_safe_int(promotion.get('current_stage_allowed'))}",
            f"- 최종 판정: {'가능' if readiness.get('ready_for_real_trading') else '보류'}",
            "",
        ]
    )


def write_readiness_report(
    readiness: Dict[str, Any],
    report_root: Path = DEFAULT_REPORT_ROOT,
) -> Dict[str, Path]:
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
    **_unused: Any,
) -> Dict[str, Dict[str, Path]]:
    existing = load_scorecards(report_root=report_root)
    scorecard = build_daily_scorecard(
        strategy=strategy,
        balance=balance,
        session_pnl=session_pnl,
        trading_mode=trading_mode,
    )
    merged = _merge_scorecards(existing, scorecard)
    scorecard["paper_gate"] = evaluate_paper_trading_gate(merged)
    scorecard_paths = write_daily_scorecard(scorecard, report_root=report_root)
    readiness_paths = write_readiness_report(
        evaluate_real_trading_readiness(merged),
        report_root=report_root,
    )
    return {"scorecard": scorecard_paths, "readiness": readiness_paths}


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _print_today_scorecard(report_root: Path, report_date: Optional[str]) -> int:
    target = report_date or datetime.now().date().isoformat()
    payload = _load_json(_scorecard_paths(report_root, target)["json"])
    if payload is None:
        print("오늘 성적표가 아직 없습니다.")
        return 1
    pnl = payload.get("pnl", {})
    trades = payload.get("trades", {})
    print(f"날짜: {payload.get('date', '')}")
    print(f"확정 순실현손익: {_format_currency(pnl.get('realized_net_pnl'))}")
    print(f"청산 거래: {_safe_int(trades.get('closed_trades'))}건")
    print(f"종료 사유: {payload.get('strategy', {}).get('halt_reason') or '-'}")
    return 0


def _print_readiness(report_root: Path) -> int:
    payload = _load_json(report_root / READINESS_REPORT_JSON)
    if payload is None:
        scorecards = load_scorecards(report_root=report_root)
        if not scorecards:
            print("실투자 전환 상태를 계산할 성적표가 없습니다.")
            return 1
        payload = evaluate_real_trading_readiness(scorecards)
    print(
        "현재 허용 실계좌 단계: "
        f"{_safe_int(payload.get('promotion', {}).get('current_stage_allowed'))}"
    )
    print(
        "최종 판정: "
        f"{'가능' if payload.get('ready_for_real_trading') else '보류'}"
    )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="성과 리포트 조회")
    parser.add_argument("--report-root", default=str(DEFAULT_REPORT_ROOT))
    subparsers = parser.add_subparsers(dest="command", required=True)
    today_parser = subparsers.add_parser("today")
    today_parser.add_argument("--date")
    subparsers.add_parser("gate")
    args = parser.parse_args(argv)
    report_root = Path(args.report_root).expanduser()
    if args.command == "today":
        return _print_today_scorecard(report_root, args.date)
    return _print_readiness(report_root)


if __name__ == "__main__":
    raise SystemExit(main())
