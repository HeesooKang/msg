"""백테스트 결과 리포트."""

from collections import defaultdict

from src.backtest.engine import BacktestResult


def _closed_trades(result: BacktestResult):
    return [record for record in result.trade_records if record.side == "sell"]


def _setup_breakdown(result: BacktestResult) -> list[tuple[str, dict]]:
    stats = defaultdict(lambda: {"trades": 0, "net_pnl": 0})
    for trade in _closed_trades(result):
        key = trade.setup_name or ("inverse_unknown" if trade.is_inverse else "unknown")
        stats[key]["trades"] += 1
        stats[key]["net_pnl"] += trade.pnl
    return sorted(stats.items(), key=lambda item: (item[1]["net_pnl"], item[0]), reverse=True)


def _regime_breakdown(result: BacktestResult) -> list[tuple[str, int]]:
    stats = defaultdict(int)
    for trade in _closed_trades(result):
        stats[trade.regime_label or "unknown"] += trade.pnl
    return sorted(stats.items(), key=lambda item: (item[1], item[0]), reverse=True)


def _symbol_breakdown(result: BacktestResult) -> list[tuple[str, int]]:
    stats = defaultdict(int)
    for trade in _closed_trades(result):
        stats[trade.symbol] += trade.pnl
    return sorted(stats.items(), key=lambda item: (item[1], item[0]), reverse=True)


def print_report(result: BacktestResult, daily_target: int = 10_000, daily_limit: int = -5_000):
    """백테스트 결과를 콘솔에 출력한다."""
    print()
    print("=" * 55)
    print("           백테스트 결과 리포트")
    print("=" * 55)

    print(f"\n  초기 자본:       {result.initial_capital:>12,}원")
    print(f"  최종 자본:       {result.final_capital:>12,}원")
    print(f"  총 수익률:       {result.total_return_pct:>11.2f}%")
    print(f"  최대 낙폭(MDD):  {result.max_drawdown_pct:>11.2f}%")

    print(f"\n  총 거래 수:      {result.total_trades:>12d}건")
    print(f"  승리:            {result.winning_trades:>12d}건")
    print(f"  패배:            {result.losing_trades:>12d}건")
    print(f"  승률:            {result.win_rate:>11.1f}%")

    if result.avg_win > 0 or result.avg_loss < 0:
        print(f"\n  평균 수익:       {result.avg_win:>12,.0f}원")
        print(f"  평균 손실:       {result.avg_loss:>12,.0f}원")
        ratio = abs(result.avg_win / result.avg_loss) if result.avg_loss != 0 else 0
        print(f"  손익비:          {ratio:>12.2f}")

    trading_days = len(result.daily_records)
    if trading_days > 0:
        target_days = result.days_target_hit(daily_target)
        loss_days = result.days_loss_limit_hit(daily_limit)
        avg_daily = sum(r.realized_pnl for r in result.daily_records) / trading_days

        print(f"\n  거래일 수:       {trading_days:>12d}일")
        print(f"  목표 달성 일수:  {target_days:>12d}일 ({target_days/trading_days*100:.1f}%)")
        print(f"  한도 도달 일수:  {loss_days:>12d}일 ({loss_days/trading_days*100:.1f}%)")
        print(f"  일평균 손익:     {avg_daily:>12,.0f}원")

    closed_trades = _closed_trades(result)
    if closed_trades:
        print("\n  셋업별 손익:")
        for setup_name, metrics in _setup_breakdown(result)[:5]:
            expectancy = metrics["net_pnl"] / metrics["trades"] if metrics["trades"] else 0
            print(
                f"    - {setup_name}: {metrics['trades']}건, "
                f"순손익 {metrics['net_pnl']:,}원, 기대값 {expectancy:,.0f}원"
            )

        print("\n  레짐별 손익:")
        for regime_label, net_pnl in _regime_breakdown(result):
            print(f"    - {regime_label}: {net_pnl:,}원")

        symbol_breakdown = _symbol_breakdown(result)
        if symbol_breakdown:
            print("\n  종목별 손익 상위:")
            for symbol, net_pnl in symbol_breakdown[:3]:
                print(f"    - {symbol}: {net_pnl:,}원")

            print("\n  종목별 손익 하위:")
            for symbol, net_pnl in list(reversed(symbol_breakdown[-3:])):
                print(f"    - {symbol}: {net_pnl:,}원")

    print()
    print("=" * 55)
