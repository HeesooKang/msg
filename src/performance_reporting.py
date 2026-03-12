import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_REPORT_ROOT = Path("reports")
DAILY_REPORT_PREFIX = "daily-scorecard."
READINESS_REPORT_JSON = "real-trade-readiness.json"
READINESS_REPORT_MD = "real-trade-readiness.md"


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
) -> Dict[str, Any]:
    now = generated_at or datetime.now()
    snapshot = _extract_strategy_snapshot(strategy)

    return {
        "date": now.date().isoformat(),
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
        },
    }


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


def render_daily_scorecard_markdown(scorecard: Dict[str, Any]) -> str:
    pnl = scorecard.get("pnl", {})
    trades = scorecard.get("trades", {})
    balance = scorecard.get("balance", {})
    strategy = scorecard.get("strategy", {})

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
    }
    readiness["ready_for_real_trading"] = all(
        item.get("ok", False) for item in readiness["criteria"].values()
    )
    return readiness


def render_readiness_markdown(readiness: Dict[str, Any]) -> str:
    window = readiness.get("window", {})
    aggregate = readiness.get("aggregate", {})
    criteria = readiness.get("criteria", {})

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
        "## 판정",
        f"- 최종 판정: {'실투자 전환 가능' if readiness.get('ready_for_real_trading') else '실투자 전환 보류'}",
        f"- 표본 크기: {mark(criteria.get('sample_size', {}).get('ok', False))}",
        f"- 누적 순손익 플러스: {mark(criteria.get('net_pnl_positive', {}).get('ok', False))}",
        f"- Profit Factor 기준 충족: {mark(criteria.get('profit_factor', {}).get('ok', False))}",
        f"- 거래당 기대값 플러스: {mark(criteria.get('expectancy_positive', {}).get('ok', False))}",
        "",
    ]
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
) -> Dict[str, Dict[str, Path]]:
    scorecard = build_daily_scorecard(
        strategy=strategy,
        balance=balance,
        session_pnl=session_pnl,
        trading_mode=trading_mode,
    )
    scorecard_paths = write_daily_scorecard(scorecard, report_root=report_root)
    readiness = evaluate_real_trading_readiness(load_scorecards(report_root=report_root))
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
