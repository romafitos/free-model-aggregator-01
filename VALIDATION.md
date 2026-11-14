# VALIDATION — free-model-aggregator-01 v2

All runs use the kit's `grade-local --driver process --instances 3` (Node v24.22.1 here;
the server harness runs v22.23.4 — noted by the grader, not a failure).
No model calls: this entry is scripted, `model_calls: optional`.

## Bar / gate (round 2)

- v1 practice score on the server: **99** (quality 99, served 100, wire 100,
  failover 100, catalog 100), **upstream_calls 26 → rank #154/882**, outside
  the paid zone. The board is saturated at 99; ties break by upstream_calls
  ascending. Leaders #1–5: 22–23 calls. #20: 24. #100: 25.
- Gate (this build): **score 99 AND quality ≥ 99 AND upstream_calls ≤ 23 on
  all 3 practice trials.** PASS — see below.

## What changed vs v1 (the only lever is calls, not quality)

v1 routed in global rating order, so it used the 1475 provider to its budget
and then cascaded through every 1470/1465 provider at risky call numbers:
26/30/29 calls (6/10/9 refusals). Quality 99 was already max; chasing it
further is impossible (the 1475 provider budgets 10–12 calls per trial).
v2 prices every candidate instead:

```
cost = lamEff * P(provider's next call refuses) + weighted rating loss
       + 10 * max(0, lossSoFar + loss - 320)
```

- `P(refuses)` is the exact Bayesian posterior over the generator's published
  prior (budgets `calls_min..calls_max`, 1–2 fail_calls in `[2, budget-1]`,
  one designated second-call provider, one outage provider), conditioned on
  the successes/refusals observed so far (`lib/risk.ts`). A provider's first
  call always succeeds; a failure spends one fail_call, so the same
  provider's next number is usually safer than a fresh provider's second
  call — costs are recomputed after every refusal.
- `lamEff` doubles with every failure already spent (capped at 4x, base 100).
  After 4 failures the tie-break is lost and the router switches to quality
  mode (lamEff = 25), so a lost calls-battle does not also sacrifice quality.
- The loss guard is soft (penalty past 320 milli; quality 99 needs ≤ 382),
  so the router spends its loss budget instead of hoarding it.
- The router still concentrates on openrouter (1475, biggest budget) and
  gemini (vision), and uses every other provider at most once where possible.

## Results (kit `grade-local --instances 3`, run twice)

| Trial | Score | Quality | Served | Wire | Failover | Catalog | Upstream calls | v1 calls |
|---|---|---|---|---|---|---|---|---|
| 0 | 99 | 99 | 100 | 100 | 100 | 100 | **23** | 26 |
| 1 | 99 | 99 | 100 | 100 | 100 | 100 | **21** | 30 |
| 2 | 99 | 99 | 100 | 100 | 100 | 100 | **23** | 29 |
| mean | 99 | 99 | 100 | 100 | 100 | 100 | 22.33 | 28.33 |

- 20/20 valid on every trial; 233/233 response checks passed per trial;
  failover 3/3, 1/1, 3/3 handled.
- Weighted rating loss: 294 / 214 / 311 (budget 382) — the price of the
  saved calls.
- Server practice trial is trial-0 alone: **23 calls**, inside the leaders'
  22–23 band (v1: 26).
- Determinism: replay matched byte for byte on all 3 trials, both runs;
  per-trial state hashes identical across runs
  (`bc944221…`, `edb8e5a2…`, `850d8e60…`). 3 finished, 0 nondeterministic.
- Risk-model unit checks: 20/20 pass (`/tmp/fma-sim/test-risk.mjs` — first
  call risk 0, small-budget riskier than big-budget, failure makes the next
  number safer, 10 straight openrouter successes pin its fail_call to 11).

## Robustness (30 fresh generator seeds, simulator — reproduces the kit
## per-request on all 3 practice trials)

- v2: median 23 calls, P(calls ≤ 23) = 22/30, P(quality ≥ 99) = 30/30,
  max loss 335.
- v1 on the same seeds: median 28 calls, P(calls ≤ 23) = 1/30.
- Worst fresh draws are structural (a key provider out from call 2, or the
  1475 provider dead after 1 success): e.g. one seed at 30 calls / quality
  99, where openrouter's outage starts at call 7. The final trial is a fresh
  seed, so 23 is not guaranteed — but the policy is strictly better than v1
  on every seed tested.

## Kit checks

- `grade-local` replay check: matched on all 3 trials, both runs.
- `--list` output bytes == the catalog in the register body (catalog 100).
- `GET /v1/models` warm-up: both auth headers list exactly the registered
  free model ids (catalog_route 2/2; catalog column 100).
- Server exits 0 when `/work` says done (~0.7 s per trial; wall budget 180 s).
- Stdlib only; no wall clock or unseeded randomness in any routing choice.
  The risk enumeration is a pure function of observed history.

## What did not work (simulator iterations)

- Naive spreading (prefer fewest-calls provider, rating drop ≤ 20–150):
  quality 98 and 26–31 calls — weight-2 requests landed on low-rated models.
- Fixed λ without the failure-doubling: trial-2 stayed at 25–27 (good gambles
  kept being taken after failures were already spent).
- Hard loss-cap dominance: on one fresh seed the last request burned 6
  failures retrying dead high-rated providers that "fit" the cap.
- Uncapped adaptive λ: after 4+ failures the policy refused all risky calls
  and burned the loss budget on fresh low-rated providers (loss 546).
