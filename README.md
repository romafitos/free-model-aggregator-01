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

## How it routes

- Candidates are every free model meeting the request's requirements,
  ranked by the quality table's `rating_milli` (best first). A paid model is
  never a legal pick.
- Attempts follow that rating order across every (provider, model) pair.
  A refusal is provider-level, never model-level, and a failed call costs
  only the tie-break (upstream calls), so retrying after a one-off fail is
  always worth it.
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
- `lib/wire.ts` — Anthropic ↔ OpenAI translation and SSE writers (starter's, kept)
- `lib/translate.ts` — cross-wire parsing/rendering and request builders
- `LICENSE` — MIT

## Scores (local kit trials)

Mean **99/100** on all 3 practice trials (quality 99, served 100, wire 100,
failover 100, catalog 100). Quality 99 is the structural ceiling: 17 of the
20 requests want the 1475-rated model, but its provider's budget caps at 12
calls. See `VALIDATION.md` for the per-iteration numbers.
