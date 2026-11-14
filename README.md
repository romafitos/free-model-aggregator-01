# free-model-aggregator-01 — the best free AI model aggregator

One server (`main.ts`, Node 22, stdlib only) that serves the OpenAI Chat
Completions API and the Anthropic Messages API — JSON and streaming — over
ten simulated free-tier providers, and picks the best free model for each
request.

## Run

```sh
node main.ts            # reads $MAKEFUN_PROMPT_FILE, registers with $MAKEFUN_ENV_URL, exits 0 when /work says done
node main.ts --list     # prints the catalog document and exits 0
```

## How it routes (v2)

- Candidates are every free model meeting the request's requirements,
  ranked by the quality table's `rating_milli` (best first). A paid model is
  never a legal pick.
- Attempts are priced, not rating-ordered: each candidate costs
  `lamEff * P(its provider's next call refuses) + weighted rating loss`.
  The refusal probability is the exact Bayesian posterior over the
  generator's published prior (`lib/risk.ts` — budgets, 1–2 fail_calls in
  `[2, budget-1]`, one designated second-call provider, one outage
  provider), conditioned on the refusals and successes observed so far.
  `lamEff` doubles with every failure already spent (capped at 4x); after
  four failures the tie-break is lost and the router switches to quality
  mode. A soft guard penalises accumulated weighted loss past 320 milli
  (quality 99 needs at most 382). Costs are recomputed after every refusal.
- A provider is dead after two consecutive refusals, or a refusal past its
  sealed budget ceiling (`calls_max`). The generator draws budgets from
  `calls_min..calls_max` and fail_calls from `[2, budget-1]`, so a
  provider's first call always succeeds and a refusal past `calls_max` is
  certainly the budget spent. Dead providers are never retried.
- An OpenAI JSON-mode request never routes to an Anthropic-wire provider:
  that wire cannot carry `response_format`.

## How it translates wires

Same-wire calls go through byte for byte (with `stream_options.include_usage`
added on OpenAI streams — the client's bodies carry none, and without it a
streamed answer carries no usage chunk). Cross-wire calls are buffered,
parsed into one normalized answer, and rendered in the client's wire
(`lib/translate.ts`): system prompts, images, tools, tool_choice, stops,
`max_tokens` and JSON mode all survive, because the provider echoes what it
parsed and the client compares that echo with what it sent.

## Files

- `main.ts` — the server: brief, catalog probe, routes, register, poll, routing
- `lib/providers.ts` — brief parsing, catalog probe, catalog building, candidates
- `lib/risk.ts` — Bayesian per-call refusal risk over the generator's prior
- `lib/wire.ts` — Anthropic ↔ OpenAI translation and SSE writers (starter's, kept)
- `lib/translate.ts` — cross-wire parsing/rendering and request builders
- `LICENSE` — MIT

## Scores (local kit trials)

v2: **99/100** on all 3 practice trials (quality 99, served 100, wire 100,
failover 100, catalog 100) with **23 / 21 / 23 upstream calls** (v1: 26 / 30 /
29). Quality 99 is the structural ceiling: 17 of the 20 requests want the
1475-rated model, but its provider's budget caps at 12 calls. The tie-break
is upstream calls, so v2 trades a little rating (weighted loss 294 / 214 /
311 of the 382 budget) for far fewer refused calls. See `VALIDATION.md`.
