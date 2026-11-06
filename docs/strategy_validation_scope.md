# Strategy validation scope

Updated 2026-10-15 after the user's explicit full-watchlist collection approval.

## Current objective

Remove the ETF-only research restriction. Compare multiple simple strategies
over the existing fixed and dynamic watchlists, including ordinary shares,
ETFs and inverse products. Do not add a separate inverse route. This supersedes
the ETF-only direction in older goal text and research contracts; those files
remain historical evidence, not the current universe restriction.

The live bot was never switched to the research ETF universe. It retains its
32-symbol default fixed list, dynamic pool and existing additional products.
Research must use the recorded per-batch universe, not today's winners, not a
retrospectively selected profitable subset, and not pretend to cover every
listed Korean security without data.

## Comparison rules

- Compare the current runtime rule with the eight already specified simple
  hypotheses, without tuning thresholds after expanding the universe.
- Keep one strategy per simulated portfolio. Do not combine losing strategies
  into a new router, rescue rule or prediction model.
- Use the same capital, daily target/loss trigger, depth, limit-order and
  missing-execution assumptions across competitors.
- Apply instrument-specific tick sizes and transaction taxes. Ordinary shares
  must not inherit the zero transaction-tax treatment used for ETF research.
- Show the configured PAPER-cost result and a stricter cost case separately.
- Preserve confirmed fills, alerts, recovery and daily controls if a candidate
  eventually qualifies. No runtime replacement before evidence supports it.
- Distinguish failed strategies, no trades and incomplete executions. Reused
  dates are exploratory, not newly independent validation.
- Do not add an ETF-only collector. On 2026-10-15 the user approved keeping
  the existing full watchlist's quote archive running after trading halts,
  until the same day's 15:30 close. Preserve dynamic discovery and actual
  quote timestamps, books and depth; do not synthesize missing observations.
- Finish settlement and the session-end notification before quote-only
  collection. Collection must not evaluate trades, submit orders, poll the
  account, clear daily halts or change target/loss/bypass rules. This approval
  authorizes code changes, not an agent-run connection or process restart.
- The user performs restarts. No credentials or sensitive configuration files.

## First full-watchlist experiment

Contract and frozen cost/universe definitions:
`reports/2026/09/watchlist-strategy-contract.2026-09-22.md`.

ETF reports remain useful controls but no longer determine the candidate
universe or limit the search to ETFs. The target remains a cost-robust strategy
with separate-date evidence, not merely a positive reused-data backtest.
