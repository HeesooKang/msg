# ETF alternative research register

## Scope

On 2026-09-21 the user authorized research into other simple strategies when
new validation data is unavailable. This does not authorize tuning repeatedly
until the same sample becomes profitable, restarting the bot, or claiming
independent evidence from previously inspected dates.

Trial 1 remains the frozen 20/5 rolling bid-channel strategy in
`docs/etf_trend_validation.md`. Its evidence is inconclusive: one resolved
losing round trip and material execution gaps. Do not silently change it.

## Trial 2: opening range breakout

Frozen before reading this candidate's PnL. This is a separate hypothesis,
not a rescue condition added to trial 1.

- Same four ETFs: 152100, 292190, 114800, 123310. Same rule for inverse ETFs.
- At the first fresh snapshot timestamped 09:05:00 through 09:05:05, freeze
  that snapshot's reported session high and low. Require its existing
  `day_summary_available` flag and positive internally consistent prices.
  A later snapshot's cumulative high/low cannot reconstruct this range.
- The range ends at the actual anchor timestamp, which can be up to five
  seconds after 09:05. This is an explicitly observed summary-range
  adaptation, not a reconstruction of exact unobserved opening bid bars.
- A later fresh bid strictly above the frozen high triggers one limit
  submission at that snapshot's ask. No first-bar direction, volume, market
  regime, ATR, prediction, or profitability filter is added.
- One submission per symbol per day; an unfilled attempt is not retried.
- Initial stop is the frozen opening low. Raise it only to higher lows of
  the latest five complete observed one-minute bid bars, using the same
  existing bar evidence rules as trial 1. Never lower it.
- Existing stop touches take precedence over a later rebound. Stop orders
  are market exits. Otherwise flatten at 15:20; no new entry from 15:19.
  No 180-second exit.
- Same 1,000,000 capital, +10,000 realized target, -5,000 total-loss stop,
  plan-risk/depth sizing, actual won fees, tax classification and partial
  fills as the prior experiment. The daily restart bypass is off.
- Lower relative spread, larger relative breakout, then symbol rank ties.
- Same next observed executable-quote fills and five-second missing-book
  policy. Missing execution is unknown, never an invented price or zero PnL.
- Stress uses twice the commission plus one adverse sell tick. Candidate
  and stress both run all five dates; no profitable-date selection.
- Reset history loses an unpersisted opening anchor. An afternoon restart
  cannot acquire an anchor from the afternoon session high/low.

The five-minute choice is a conventional external hypothesis, not a sweep
over 5/10/15/30 minute windows on these returns. Report every attempted
candidate. Negative sufficiently observed evidence rejects the candidate;
no fills or execution gaps remain inconclusive. Any positive result is only
exploratory until untouched-date and execution/cost evidence supports it.
Do not adopt the best reused-data result as proven profitability.

## Sources and limits

- Zarattini, Barbon and Aziz, *A Profitable Day Trading Strategy For The
  U.S. Equity Market*: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4729284
  provides an opening-range research motivation, not evidence for this
  Korean ETF adaptation. Its stock selection, direction rule, ATR stop and
  execution assumptions differ and are not imported here.
- Bailey et al., *The Probability of Backtest Overfitting*:
  https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf
  motivates recording all trials and not treating repeated sample reuse as
  independent confirmation. No claimed statistical significance from five
  already inspected, partly incomplete days.

No runtime or operational configuration changes are part of this experiment.
The research program runs in `/private/tmp`, imports only offline replay
helpers, and never loads credentials or sends orders to a broker.
