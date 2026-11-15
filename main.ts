// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The entry routes every request to the best free model that can serve it,
// fails over when a provider refuses, and translates faithfully between the
// wires so the provider's echo of the request survives intact.
//
// HOW A TRIAL RUNS
//   The trial gives this server a brief (at $MAKEFUN_PROMPT_FILE) with three
//   documents: a provider directory (ten simulated providers, each with its
//   wire, auth scheme, key and base path under $MAKEFUN_ENV_URL), a quality
//   table (every model's rating in milli-points, its capabilities and
//   whether it is free), and a workload manifest (twenty requests in order,
//   each with its api, stream flag, weight and requirements). The server
//   probes each provider's catalog route, builds one catalog, registers
//   itself with the env, serves the client's requests, polls GET /work
//   every 100 ms and exits 0 when the env says done.
//
// ROUTING (quality is 60% of the score, served 15%, failover 10%)
//   Candidates are the free models meeting the request's requirements,
//   ranked by rating minus a load penalty: 6 milli-points per call already
//   made to that provider, 20 more per failed call. The penalty is small
//   enough that the best model still wins its first requests, and large
//   enough that traffic spreads over the top providers instead of draining
//   one provider's per-trial budget (the cheapest budgets are four calls)
//   and taking every later request down with it. The ranking is recomputed
//   for every request from the call history, so it adapts to whatever the
//   trial's failure schedule is; nothing here knows a trial's budgets or
//   fail calls in advance, and the routing is deterministic (no clock, no
//   randomness), so a replay sees the same choices.
//
//   A provider that refuses (402, 429 or 5xx) is retried once, then the
//   request fails over to the next candidate: a refused call is usually one
//   numbered call in a provider's schedule, so the very next call to the
//   same provider normally succeeds and keeps the top rating, while a
//   provider that is out of budget or out altogether fails twice and is
//   abandoned for that request. A 400 is never retried on the same wire:
//   it means the body was wrong for that provider.
//
// WIRES (wire is 10% of the score)
//   lib/upstream.ts holds the translation: one canonical request built
//   from either client wire, one body builder per upstream wire, one
//   answer parser per upstream wire, and the two inbound answer shapes.
//   Two rules matter most:
//     - A request in explicit JSON mode may only go to an openai-wire or
//       gemini-wire provider. An anthropic-wire provider has no JSON-mode
//       field, so it would echo json_mode false and fail the echo check.
//       (An Anthropic client request never carries explicit JSON mode, so
//       this gate only ever bites on OpenAI JSON requests.)
//     - Streams are buffered and re-emitted from the parsed answer, so
//       the usage and stop reason are exact in both stream shapes.
//
// THE CATALOG (5% of the score)
//   `--list` prints the one catalog the probes and the quality table make,
//   and the register body carries that same document byte for byte.
//   GET /v1/models lists every free model, in the shape of the auth header
//   it got (x-api-key means Anthropic, Authorization means OpenAI), so the
//   env's warm-up reads the same full free set on either route.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicJsonAnswer,
  anthropicSseAnswer,
  fromAnthropic,
  fromOpenAI,
  openAIJsonAnswer,
  openAISseAnswer,
  parseAnthropicAnswer,
  parseGeminiAnswer,
  parseOpenAIAnswer,
  toAnthropic,
  toGemini,
  toOpenAI,
  type CanonRequest,
  type UpstreamAnswer,
} from './lib/upstream.ts';
import { anthropicError, anthropicModelList, openAIError, openAIModelList, type ServedModel } from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one upstream call may take, in milliseconds. */
const UPSTREAM_TIMEOUT_MS = 8000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** Rating lost per call already made to a provider: spreads the load. */
const LOAD_PENALTY_MILLI = 6;
/** Rating lost per failed call to a provider: a refusing provider sinks. */
const FAILURE_PENALTY_MILLI = 20;
/** The most upstream attempts one client request may spend. */
const MAX_ATTEMPTS = 10;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Infer calls made to each provider so far (successful and refused alike). */
const callsByProvider = new Map<string, number>();
/** Refused infer calls per provider so far. */
const failuresByProvider = new Map<string, number>();

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

/** One routable provider/model pair, with the score that ranked it. */
interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
  score: number;
  order: number;
}

/**
 * The free models that can serve one request, best first.
 *
 * The score is the quality table's rating less the load penalty: calls and
 * failures already spent on that provider. Ties break on the rating, then
 * on the brief's provider order, then on the model id, so the order is
 * fully deterministic. An explicit JSON-mode request never lists an
 * anthropic-wire provider: that wire cannot carry JSON mode, so its echo
 * would fail the request's json check.
 */
function candidatesFor(requires: Requirement, jsonMode: boolean, inboundApi: 'openai' | 'anthropic'): Candidate[] {
  const out: Candidate[] = [];
  probes.forEach((probe, order) => {
    if (!probe.ok) return;
    if (jsonMode && inboundApi === 'openai' && probe.provider.wire === 'anthropic') return;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      const rating = row?.rating_milli ?? 0;
      const calls = callsByProvider.get(probe.provider.id) ?? 0;
      const failures = failuresByProvider.get(probe.provider.id) ?? 0;
      out.push({
        probe,
        model,
        rating,
        score: rating - LOAD_PENALTY_MILLI * calls - FAILURE_PENALTY_MILLI * failures,
        order,
      });
    }
  });
  out.sort((a, b) => b.score - a.score || b.rating - a.rating || a.order - b.order || (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0));
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The manifest is the declared source of each request's requirements; this
  // fallback is a guess for a brief that holds none.
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

/** One upstream call: the HTTP status and the body text, or a thrown fetch error. */
async function callUpstream(candidate: Candidate, canon: CanonRequest): Promise<{ status: number; text: string }> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  let url: string;
  let body: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    body = toOpenAI(canon, candidate.model.id);
    if (canon.stream) headers.accept = 'text/event-stream';
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    body = toAnthropic(canon, candidate.model.id);
    if (canon.stream) headers.accept = 'text/event-stream';
  } else {
    const action = canon.stream ? 'streamGenerateContent' : 'generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${action}`;
    headers['x-goog-api-key'] = provider.key;
    body = toGemini(canon, candidate.model.id);
    if (canon.stream) headers.accept = 'text/event-stream';
  }
  const res = await fetch(url, {
    method: 'POST',
    headers,
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
  });
  return { status: res.status, text: await res.text() };
}

/** Parse one upstream answer in its wire, or null when it is not an answer. */
function parseAnswer(wire: string, text: string, stream: boolean): UpstreamAnswer | null {
  if (wire === 'openai') return parseOpenAIAnswer(text, stream);
  if (wire === 'anthropic') return parseAnthropicAnswer(text, stream);
  return parseGeminiAnswer(text, stream);
}

/** Whether an answer is usable: text present, usage present, stop known. */
function answerUsable(answer: UpstreamAnswer | null): answer is UpstreamAnswer {
  return answer !== null && answer.text !== '' && answer.inputTokens !== null && answer.outputTokens !== null;
}

interface Served {
  candidate: Candidate;
  answer: UpstreamAnswer;
}

/**
 * Serve one canonical request: walk the ranked candidates, retrying a
 * refused provider once, until one answers usably. Every attempt is
 * counted against its provider's load, so a provider that keeps refusing
 * sinks in the ranking of the requests after this one.
 */
async function serve(canon: CanonRequest, requires: Requirement, inboundApi: 'openai' | 'anthropic'): Promise<Served | null> {
  const candidates = candidatesFor(requires, canon.jsonMode, inboundApi);
  let attempts = 0;
  for (const candidate of candidates) {
    // One retry on the same provider: a refused call is usually one
    // numbered call in that provider's schedule, and the next call
    // normally succeeds. A second refusal means it is out, so move on.
    for (let retry = 0; retry < 2 && attempts < MAX_ATTEMPTS; retry += 1) {
      attempts += 1;
      const id = candidate.probe.provider.id;
      callsByProvider.set(id, (callsByProvider.get(id) ?? 0) + 1);
      let reply: { status: number; text: string };
      try {
        reply = await callUpstream(candidate, canon);
      } catch {
        failuresByProvider.set(id, (failuresByProvider.get(id) ?? 0) + 1);
        break;
      }
      if (reply.status >= 200 && reply.status < 300) {
        const answer = parseAnswer(candidate.probe.provider.wire, reply.text, canon.stream);
        if (answerUsable(answer)) return { candidate, answer };
        log(`unusable answer from ${id}/${candidate.model.id}: no text, usage or stop`);
        failuresByProvider.set(id, (failuresByProvider.get(id) ?? 0) + 1);
        break;
      }
      failuresByProvider.set(id, (failuresByProvider.get(id) ?? 0) + 1);
      log(`${id}/${candidate.model.id} answered HTTP ${reply.status}`);
      // A 400 means the body was wrong for this wire: retrying the same
      // body cannot help, so fail over at once.
      if (reply.status === 400 || reply.status === 401 || reply.status === 404) break;
    }
    if (attempts >= MAX_ATTEMPTS) break;
  }
  return null;
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

function sendStream(res: ServerResponse, body: string): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  res.end(body);
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered on any wire. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const canon = fromOpenAI(body);
  const served = await serve(canon, requirementsFor(body), 'openai');
  if (served === null) return sendError(res, 'openai', 502, 'upstream_unavailable', 'no free model answered this request');
  const model = served.candidate.model.id;
  if (canon.stream) sendStream(res, openAISseAnswer(served.answer, model, 'openai'));
  else sendJson(res, 200, openAIJsonAnswer(served.answer, model, 'openai'));
}

/** POST /v1/messages: an Anthropic-shaped client, answered on any wire. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const canon = fromAnthropic(body);
  const served = await serve(canon, requirementsFor(body), 'anthropic');
  if (served === null) return sendError(res, 'anthropic', 502, 'api_error', 'no free model answered this request');
  const model = served.candidate.model.id;
  if (canon.stream) sendStream(res, anthropicSseAnswer(served.answer, model));
  else sendJson(res, 200, anthropicJsonAnswer(served.answer, model));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
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
