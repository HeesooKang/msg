"""백테스트 결과 리포트."""

from src.backtest.engine import BacktestResult


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

    print()
    print("=" * 55)
