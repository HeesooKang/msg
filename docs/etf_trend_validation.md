# ETF trend validation contract

Frozen before candidate replay on 2026-09-21. This is a research hypothesis,
not a profitable strategy or a new runtime route. No bot restart is authorized.

## Candidate

- Existing fixed-universe domestic index ETFs: 152100, 292190, 114800, 123310.
  No selection by future return; regular and inverse use the same rules.
- One-minute observed bid bars, ordered by quote and receipt timestamps.
  Do not interpolate missing minutes or accept future/delayed revisions of a
  closed bar. Require 21 consecutive complete calendar minutes for a signal.
- Entry: the latest completed bar closes strictly above the highest bid in
  the preceding 20 completed bars. Submit at the next fresh observed ask.
- Initial stop: minimum bid of the last five completed bars including the
  signal bar. After entry, raise to each newly completed five-bar low only
  when it exceeds the existing stop. Never lower the stop.
- Evaluate existing stops before incorporating a later closed bar. A stop
  touched on a received quote is not erased by a subsequent rebound.
- Exit at the stop or at 15:20; no 180-second expiry. No new entry from 15:19.
- One submission per symbol/signal minute; no retries of the same signal.
- If multiple signals coincide, lower observed relative spread first, then
  greater relative breakout, then symbol. No fabricated EV or win rate.
- Use the existing won PnL, capital/risk/depth sizing, partial-fill and sell
  ledger implementation. Seed 1,000,000, realized target +10,000, total-loss
  stop -5,000, restart loss bypass disabled for comparison.
- Fees .000142 each side. ETF transaction tax zero. Fund distributions and
  possible holding-period withholding are not in quote files; mark this
  limitation and do not mistake this result for a fully verified tax return.
- Base fills use the next received executable quote, never the signal quote.
  Unobserved/late execution or unknown held-price intervals stop portfolio
  evaluation; do not count these as zero-return trades.
- Cost stress: double commission and an extra adverse bid tick on sells.
  A buy remains capped by the submitted limit.

## Comparisons and decision

- Unchanged current strategy on the same four ETFs, capital, fees and input.
- Candidate and baseline both report full provenance, data coverage,
  confirmed PnL, fees, drawdown, fills, failed submissions, unresolved outcomes.
- A reproducible random-entry control uses candidate readiness, the same
  initial/trailing stop and execution rules. Entry frequency is frozen from
  the earlier diagnostic days only, never fitted to evaluation returns.
  Use all eligible minutes on 09-15 through 09-18 for this frequency and
  SHA-256 draws for seeds 0 through 19 on 09-21. Report every seed, never
  select the best control. A minute requires observed quotes within five
  seconds of both boundaries; missing boundary bars break continuity.
- 09-15 through 09-18: diagnostic period; 09-21: chronologically separate
  comparison, NOT an untouched holdout (previous analyses used all five days).
- A genuinely unseen subsequent date with adequate data is required before
  adoption. No universal observation count proves profitability.
- Do not tune 20/5, dates, instruments or filters after inspecting outcomes.
- Negative adequately observed candidate performance rejects this hypothesis.
  Missing data or no trades is inconclusive, not a success or a rejection
  based on fabricated returns. Keep runtime unchanged in either case.
- Positive historical PnL alone does not authorize adoption. Also require
  independent-date evidence, cost robustness and no single-day concentration.
- If rejected, remove the candidate from runnable code; retain this contract
  and the evidence report. Do not rescue it with another condition.

## Operational boundaries

Runtime trading, alerts, account/state recovery, existing risk bypass setting,
original ledgers and environment files are unchanged during validation.
Tests prove implementation contracts, not profitability. Any eventual runtime
replacement must separately verify state migration and the preserved execution
invariants. Do not silently adopt an offline result as a live strategy.
