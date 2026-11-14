# VALIDATION — free-model-aggregator-01

All runs use the kit's `grade-local --driver process` (Node v24.22.1 here;
the server harness runs v22.23.4 — noted by the grader, not a failure).
No model calls: this entry is scripted, `model_calls: optional`.

## Bar / gate

- House baseline (kit starter, trial 0): **38** (quality 38, served 40,
  wire 39, failover 0, catalog 100, 14 calls). The starter's header calls it
  "roughly a quarter of the score".
- No live leaderboard yet (0 submissions at build time).
- Structural ceiling: quality 100 needs all 17 requests that want the
  1475-rated model to get it, but that model's provider budgets 10–12 calls
  per trial. Quality 99 / total 99 is the maximum achievable score.
- Gate (round 1 only with ≥8% margin over the bar): **PASS** — 99 vs
  baseline 38 is +160%.

## Iterations (local kit score, `grade-local --instances 3`)

| Iter | Change | Trial 0 | Trial 1 | Trial 2 | Mean |
|---|---|---|---|---|---|
| 0 | Starter as-is (baseline) | 38 (q38/s40/w39/f0/c100, 14 calls) | — | — | 38 |
| 1 | v2: rating-order routing, all 3 wires, failover, dead-tracking | 68 (q65/s65/w95/f50/c100, 26 calls) | 78 (q72/s75/w96/f85/c100, 30 calls) | 77 (q72/s75/w96/f75/c100, 31 calls) | 74.3 |
| 2 | Fix: Anthropic→OpenAI translation (system/tools/stops were sent raw); global rating order instead of per-provider passes | 99 (q99/s100/w100/f100/c100, 26 calls) | 99 (q99/s100/w100/f100/c100, 30 calls) | 99 (q99/s100/w100/f100/c100, 29 calls) | 99 |
| 3 | Fresh seed `aaaa…` (own harness, generator-drawn instance) | — | — | — | 99 (q99/s100/w100/f100/c100, 28 calls) |
| 4 | Fresh seed `1111…` | — | — | — | 99 (q99/s100/w100/f100/c100, 33 calls) |
| 5 | Fresh seed `2222…` | — | — | — | 99 (q99/s100/w100/f100/c100, 30 calls) |
| 6 | Fresh seed `3333…` | — | — | — | 99 (q99/s100/w100/f100/c100, 23 calls) |
| 7 | Edge cases (see below) + determinism re-run | — | — | — | 19/19 pass; state export byte-identical on re-run |
| 8 | Final confirmation, code unchanged since iter 2 | 99 (26 calls) | 99 (30 calls) | 99 (29 calls) | 99 |

Columns: q=quality, s=served, w=wire, f=failover, c=catalog. Replay
matched byte-for-byte on every trial of every iteration (no
nondeterministic trials, no timeouts). Iter 8 reproduced iter 2's state
hashes exactly (`f1982a6b…`, `de752c4f…`, `2d7ddb78…`).

Diagnosis that drove iter 1 → iter 2 (from the trial state exports):
7 requests invalid, all Anthropic-client → OpenAI-provider, all
`echo_system` / `echo_tools` / `echo_stop` — the client body was forwarded
raw instead of through `anthropicToOpenAI`. The pass structure also cost
quality (r02/trial-2 served at 1432 instead of the same provider's 1452
second model).

## Kit checks

- `grade-local` replay check: matched on all 3 practice trials, all runs.
- `--list` output bytes == the catalog in the register body (verified
  against a stub env that records the register POST).
- `GET /v1/models` warm-up: both auth headers list exactly the registered
  free model ids (catalog_route 2/2; catalog column 100).
- Server exits 0 when `/work` says done (~1 s per trial; wall budget 180 s).
- Stdlib only; no wall clock or unseeded randomness in any routing choice.

## Edge cases (direct, against a live env — 19/19 pass)

- Echo preservation per wire pair: anthropic→openai, openai→anthropic,
  openai→gemini, anthropic→gemini (JSON), plus streaming for
  openai→gemini, anthropic→openai, openai→anthropic — system, user text,
  tools, stops, max_tokens, image sha, JSON mode all match the client body.
- Streaming round-trip: re-rendered SSE re-parses to identical text and
  usage.
- Long context (r10, ~140-line log) passthrough.
- 402 failover: Pollinations refuses 402 once its budget is spent.
- 429 with `retry-after`: Groq's 429 carries the header (the entry does not
  wait on it — the env enforces no cooldown; a retry is simply the next call
  number, and waiting would only burn wall time).
- `--list` exits 0; server exits 0 on done.

## What did not change the score

- Upstream calls (tie-break only): 26/30/29 on practice, 23–33 on fresh
  seeds. The extra calls are forced first-pick refusals (drawn fail_calls /
  spent budgets / outages). Deliberately sacrificing rating for fewer calls
  was rejected: it spends quality slack for a tie-break that only matters
  on exact ties, and quality is the 60% column.
- Quality 99 → 100 is unreachable by construction (see Bar / gate).
