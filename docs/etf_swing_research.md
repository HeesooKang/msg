# Confirmed swing breakout research contract

Trial 4, frozen before PnL on 2026-09-21. This replaces the research entry
and trailing reference with confirmed price swings; it is not an added
condition on an earlier candidate. No production route is changed.

- Same four ETFs and all five recorded dates; no symbol/date selection.
- Use the existing observed one-minute bid bars and boundary-completeness
  rules. They are observed extrema, not a claim of exchange-complete OHLC.
- A swing high is the strict highest high in five consecutive completed
  bars, centered on the third. A swing low is the strict lowest low under
  the mirror definition. Equal extrema do not qualify.
- A swing becomes available only after both following bars have closed.
  Never trade at the center bar using information learned two minutes later.
- Store the latest confirmed high and low. Missing/incomplete bars or an
  observation reset clear candidate swing state; held stop prices survive.
- Once both exist, a fresh bid above the latest confirmed high permits one
  ask-limit submission for that swing. No repeated chase on an unfilled
  signal; a newly confirmed swing high is a new setup.
- Initial stop is the latest confirmed swing low. Raise a held stop only
  when a higher swing low is confirmed. Never lower it.
- Old-stop touches precede later confirmation/rebound in the same batch.
  Stop exits are market orders; otherwise close at 15:20. No new entry from
  15:19 and no 180-second expiry.
- Same one-million-won capital, depth/cost/risk sizing, +10,000 realized
  target, -5,000 total-loss stop, next executable quote and partial fills.
- Same tie order: lower relative spread, larger relative breakout, symbol.
- Same execution gaps are unknown. Stress doubles commission and subtracts
  one sell tick. Report every day, including unresolved and zero-order days.
- No companion indicator, regime, model, minimum gain or probability gate.
  Do not tune the five-bar pattern or add a filter after inspecting PnL.

The structural five-bar definition is documented by
[MetaQuotes](https://www.metatrader5.com/en/mobile-trading/iphone/help/chart/indicators/bw_indicators/fractals).
Only that definition is used, not its full multi-indicator trading system
or any claim that this Korean ETF adaptation is profitable.

All existing days have already been inspected. Another positive reused-data
result would remain exploratory and would not prove independent profitability.
Negative evidence or unknown execution must not be repaired into success.
Runtime, credentials, original ledgers and bot processes remain unchanged.
