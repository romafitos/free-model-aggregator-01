// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The server reads the brief, probes every provider's catalog route, builds
// one catalog, registers it with the env, and serves the client's 20
// requests, each with the best free model that can serve it.
//
// ROUTING (the score is quality 60 / served 15 / wire 10 / failover 10 /
// catalog 5). Candidates are every free model meeting the request, ranked by
// the quality table's rating_milli. The attempt order is the rating order
// across every (provider, model) pair, a provider's second model included:
// a refusal is provider-level, and after a one-off fail_call the provider's
// next call number usually succeeds, at a higher rating than the next
// provider's best. A provider is dead after two
// consecutive refusals, or a refusal past its sealed budget ceiling (the
// generator draws budgets from calls_min..calls_max and fail_calls from
// [2, budget-1], so a provider's first call always succeeds and a refusal
// past calls_max is certainly the budget spent). Dead providers are never
// retried: the trial's budgets are small and a dead provider stays dead.
// A failed call costs only the tie-break (upstream_calls), never the score,
// so retrying a live provider after a one-off fail_call is always worth it.
//
// WIRES. Same-wire calls go through byte for byte (the OpenAI body with the
// routed model, plus stream_options.include_usage on streams: the client's
// bodies carry none, and without it a streamed answer carries no usage chunk
// and the usage check fails). Cross-wire calls are buffered, parsed into one
// normalized answer, and rendered in the client's wire (lib/translate.ts):
// system prompts, images, tools, tool_choice, stops, max_tokens and JSON
// mode all survive the translation, because the provider echoes what it
// parsed and the client compares that echo with what it sent. An OpenAI
// JSON-mode request never routes to an Anthropic-wire provider: that wire
// cannot carry response_format, so the provider would echo json_mode false.
//
// Determinism: every choice is a function of the brief, the requests and the
// providers' answers so far. No clock, no unseeded randomness; the catalog
// and the model lists use fixed timestamps.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  CALLS_MAX,
  candidatesFor,
  isRecord,
  joinBase,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type Candidate,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  anthropicEventsFromOpenAI,
  type ServedModel,
} from './lib/wire.ts';
import {
  anthropicToGemini,
  openAIToAnthropicRequest,
  openAIToGemini,
  parseAnthropicJson,
  parseAnthropicSse,
  parseGeminiJson,
  parseGeminiSse,
  parseOpenAIJson,
  parseOpenAISse,
  renderAnthropicJson,
  renderAnthropicSse,
  renderOpenAIJson,
  renderOpenAISse,
  type NormalizedAnswer,
} from './lib/translate.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take, in milliseconds. */
const UPSTREAM_TIMEOUT_MS = 10000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

/** Per-provider routing state: calls made, consecutive refusals, death. */
interface ProviderState {
  calls: number;
  consecutiveFailures: number;
  dead: boolean;
}
const providerState = new Map<string, ProviderState>();
function stateFor(providerId: string): ProviderState {
  let state = providerState.get(providerId);
  if (state === undefined) {
    state = { calls: 0, consecutiveFailures: 0, dead: false };
    providerState.set(providerId, state);
  }
  return state;
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** Whether the client's OpenAI body asks for JSON mode. */
function jsonModeOf(body: Json, clientApi: 'openai' | 'anthropic'): boolean {
  if (clientApi !== 'openai') return false;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

// ---- upstream calls ----

type UpstreamResult =
  | { ok: true; counted: true; text: string; contentType: string }
  | { ok: false; counted: boolean; status: number };

/** Parse one provider's answer (JSON or SSE) into the normalized form. */
function parseUpstream(wire: 'openai' | 'anthropic' | 'gemini', stream: boolean, text: string): NormalizedAnswer {
  if (!stream) {
    let body: unknown = null;
    try {
      body = JSON.parse(text);
    } catch {
      body = null;
    }
    if (wire === 'openai') return parseOpenAIJson(body);
    if (wire === 'anthropic') return parseAnthropicJson(body);
    return parseGeminiJson(body);
  }
  if (wire === 'openai') return parseOpenAISse(text);
  if (wire === 'anthropic') return parseAnthropicSse(text);
  return parseGeminiSse(text);
}

/** Render the normalized answer in the client's wire. */
function renderAnswer(ans: NormalizedAnswer, clientApi: 'openai' | 'anthropic', stream: boolean, model: string): { text: string; contentType: string } {
  if (clientApi === 'openai') {
    return stream
      ? { text: renderOpenAISse(ans, model), contentType: 'text/event-stream' }
      : { text: JSON.stringify(renderOpenAIJson(ans, model)), contentType: 'application/json' };
  }
  return stream
    ? { text: renderAnthropicSse(ans, model), contentType: 'text/event-stream' }
    : { text: JSON.stringify(renderAnthropicJson(ans, model)), contentType: 'application/json' };
}

/**
 * One inference call to one provider. Same-wire answers come back byte for
 * byte; cross-wire answers are parsed and re-rendered in the client's wire.
 * A refusal that reached the provider's budget check (200, 402, 429, 5xx)
 * counts toward its budget; a 400/401 or an unreachable provider does not.
 */
async function callUpstream(
  cand: Candidate,
  clientApi: 'openai' | 'anthropic',
  body: Json,
  stream: boolean,
): Promise<UpstreamResult> {
  const provider = cand.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const model = cand.model.id;
  let url: string;
  let upstreamBody: Json;
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    // An Anthropic client body must be rewritten into Chat Completions first:
    // sent raw, the provider parses no system message, no tools and no stops.
    upstreamBody = clientApi === 'anthropic' ? anthropicToOpenAI(body, model) : { ...body, model };
    if (stream) upstreamBody.stream_options = { ...(isRecord(upstreamBody.stream_options) ? upstreamBody.stream_options : {}), include_usage: true };
    if (stream) headers.accept = 'text/event-stream';
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    upstreamBody = clientApi === 'openai' ? openAIToAnthropicRequest(body, model) : { ...body, model };
    if (stream) headers.accept = 'text/event-stream';
  } else {
    const action = stream ? 'streamGenerateContent' : 'generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(model)}:${action}${stream ? '?alt=sse' : ''}`;
    upstreamBody = clientApi === 'openai' ? openAIToGemini(body, model) : anthropicToGemini(body, model);
    if (stream) headers.accept = 'text/event-stream';
  }
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(upstreamBody), signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS) });
  } catch (err) {
    log(`${provider.id}/${model}: unreachable: ${(err as Error).message}`);
    return { ok: false, counted: false, status: 0 };
  }
  const text = await res.text();
  if (!res.ok) {
    const concurrent = text.includes('concurrent_requests');
    return { ok: false, counted: !concurrent && res.status !== 400 && res.status !== 401, status: res.status };
  }
  if (provider.wire === clientApi) {
    // Same wire: the provider's answer is already in the client's shape.
    return { ok: true, counted: true, text, contentType: stream ? 'text/event-stream' : 'application/json' };
  }
  if (provider.wire === 'openai' && clientApi === 'anthropic' && !stream) {
    // The starter's translator covers this pair exactly; keep its bytes.
    const parsed: unknown = JSON.parse(text);
    return { ok: true, counted: true, text: JSON.stringify(openAIToAnthropic(isRecord(parsed) ? parsed : {}, model)), contentType: 'application/json' };
  }
  if (provider.wire === 'openai' && clientApi === 'anthropic' && stream) {
    // Stream the OpenAI SSE through the starter's Anthropic event translator.
    const events: string[] = [];
    const streamBody = new Response(text).body;
    if (streamBody !== null) {
      for await (const event of anthropicEventsFromOpenAI(streamBody, model)) events.push(event);
    }
    return { ok: true, counted: true, text: events.join(''), contentType: 'text/event-stream' };
  }
  const ans = parseUpstream(provider.wire, stream, text);
  const rendered = renderAnswer(ans, clientApi, stream, model);
  return { ok: true, counted: true, text: rendered.text, contentType: rendered.contentType };
}

/**
 * Serve one client request: try the candidates in rating order until one
 * answers, tracking each provider's budget and death. A failed call costs
 * only the tie-break, so the order is simply the best rating first, a
 * provider's second model included: after a one-off fail_call its next call
 * number usually succeeds, and at a higher rating than the next provider.
 * Returns the answer to send, or null when no free model could serve it.
 */
async function serve(
  clientApi: 'openai' | 'anthropic',
  body: Json,
): Promise<{ status: number; contentType: string; text: string } | null> {
  const requires = requirementsFor(body);
  const stream = body.stream === true;
  const candidates = candidatesFor(probes, brief.ratings, requires, clientApi, jsonModeOf(body, clientApi));
  if (candidates.length === 0) return null;
  for (const cand of candidates) {
    const state = stateFor(cand.probe.provider.id);
    if (state.dead) continue;
    const result = await callUpstream(cand, clientApi, body, stream);
    if (result.ok) {
      if (result.counted) state.calls += 1;
      state.consecutiveFailures = 0;
      return { status: 200, contentType: result.contentType, text: result.text };
    }
    if (result.counted) {
      state.calls += 1;
      state.consecutiveFailures += 1;
      const max = CALLS_MAX[cand.probe.provider.id];
      if (state.consecutiveFailures >= 2 || (max !== undefined && state.calls > max)) {
        state.dead = true;
        log(`${cand.probe.provider.id} is dead after ${state.calls} calls`);
      } else {
        log(`${cand.probe.provider.id}/${cand.model.id} refused HTTP ${result.status} (call ${state.calls})`);
      }
    } else {
      log(`${cand.probe.provider.id}/${cand.model.id} failed without spending budget (HTTP ${result.status})`);
    }
  }
  return null;
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
    const answer = await serve('openai', body);
    if (answer === null) return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory meets this request');
    const bytes = Buffer.from(answer.text, 'utf8');
    res.writeHead(answer.status, { 'content-type': answer.contentType, 'content-length': bytes.length });
    res.end(bytes);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
    const answer = await serve('anthropic', body);
    if (answer === null) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory meets this request');
    const bytes = Buffer.from(answer.text, 'utf8');
    res.writeHead(answer.status, { 'content-type': answer.contentType, 'content-length': bytes.length });
    res.end(bytes);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
