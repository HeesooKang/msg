# ETF VWAP crossover research contract

Trial 3, frozen on 2026-09-21 before its PnL run. This is a separate entry
hypothesis under the user's request to examine alternative methods. It is
not an extra confirmation on the rolling-channel or opening-range strategy.
It differs from range-high entry; it is research, not a silent live change.

## Rules

- Universe remains 152100, 292190, 114800, 123310, including inverse ETFs
  without a separate formula. All five existing dates are reported.
- Session VWAP is the recorded cumulative traded won amount divided by
  cumulative shares, not an average of sparsely sampled last prices.
- Require fresh, same-day, valid bid/ask snapshots and positive cumulative
  amount/volume. The reported session summary must be available, and VWAP
  must lie within its low/high. Inconsistent inputs are recorded as missing,
  never rescaled or repaired with future values.
- A cross occurs when the previous bid was at or below its then-current
  VWAP and the new bid is strictly above its current VWAP. Use exact integer
  cross-products, not rounded VWAP. The pair must be ordered, no more than
  five seconds apart, and have nondecreasing volume and amount.
- Missing, stale or reset observations break the pair. Starting above VWAP
  is not a cross. A later reversal within the same batch cancels that signal.
- On a cross, enter once at the latest fresh ask. An unfilled limit is
  canceled; do not chase or keep retrying while price stays above VWAP.
  A genuinely new below-to-above cross can create a new signal.
- Initial stop is the low of the latest five complete one-minute observed
  bid bars, excluding the current unfinished minute. No valid stop means no
  executable plan. Subsequent stops rise only, using the same five-bar low.
- Preserve old-stop touches before later rebounds, direct market exits,
  15:20 closing, and no entry from 15:19. No 180-second expiry.
- Use unchanged sizing, fees, ETF tax classification, depth, partial fills,
  one-million-won capital, +10,000 realized target and -5,000 total-loss stop.
- Ties: lower relative spread, greater bid/VWAP ratio, then symbol. No
  regime, slope, extra threshold, trained model or predicted probability.
- Replay and cost stress use the same executable next quote, five-second
  unknown-gap rule, doubled commissions and one adverse sell tick as before.

## Decision and sources

Do not sweep periods or add a volume/market filter after seeing results.
Failure rejects this specification, not a reason to add an exception.
Unknown execution is not zero PnL. Reusing the same dates remains exploratory;
positive aggregate results alone do not establish independent profitability.

VWAP definition: [IBKR glossary](https://www.interactivebrokers.com/campus/glossary-terms/volume-weighted-average-price-vwap/).
This is an observed execution benchmark, not proof that crossing it forecasts
profitable returns. The crossover is the hypothesis being tested here.

Operational code, credentials, original ledgers, alerts and processes remain
unchanged. The temporary research script is not imported by the bot.
