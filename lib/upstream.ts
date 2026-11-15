// The three upstream wires, in one place.
//
// The entry is a translator, not a model: the client sends one OpenAI or
// Anthropic request, the entry forwards it to one simulated provider on that
// provider's own wire (openai / anthropic / gemini), and the provider's answer
// is translated back into the wire the client spoke. Nothing about the
// request may be lost in translation, because the provider echoes the request
// back inside its answer text and the trial checks that echo field by field:
// the system prompt, the user text, the tool names in order, the output cap,
// the stop sequences, the image bytes and the JSON-mode flag must all arrive
// at the provider exactly as the client sent them.
//
// The shape that makes that possible is the canonical request below: one
// normalised form of a chat request, built from either client wire, from
// which each upstream wire's body is built. Building upstream bodies from
// the canonical form (instead of rewriting one wire into another ad hoc)
// is what keeps the echo identical no matter which provider serves it.
//
// Streaming is buffered, deliberately. A streamed answer's usage only
// arrives at the end of the upstream stream (an OpenAI usage chunk, an
// Anthropic message_delta, a Gemini usageMetadata frame), but an Anthropic
// stream must carry the input tokens in its very first event, and an OpenAI
// stream must carry its usage before [DONE]. The only way to emit a
// well-ordered stream with exact usage is to read the whole upstream stream
// first, then emit the client's stream from the parsed answer. The trial
// reads the whole response before checking it, so buffering costs nothing
// and buys exact usage and stop reasons.

import { isRecord, type Json } from './providers.ts';

/** One piece of a message: plain text, or an image carried as base64 or a URL. */
export type ContentPart =
  | { kind: 'text'; text: string }
  | { kind: 'image'; mime: string; data: string; url: string | null };

/** One tool call inside an assistant message. */
export interface ToolCall {
  id: string;
  name: string;
  args: unknown;
}

/** One message in canonical form. System messages are lifted out to `system`. */
export interface CanonMessage {
  role: 'user' | 'assistant' | 'tool';
  parts: ContentPart[];
  toolCalls: ToolCall[];
  toolCallId: string;
}

/** One tool definition in canonical form. */
export interface CanonTool {
  name: string;
  description: string;
  parameters: Json;
}

/** A chat request with the wire stripped off: everything every wire needs. */
export interface CanonRequest {
  system: string;
  messages: CanonMessage[];
  tools: CanonTool[];
  maxTokens: number | null;
  temperature: number | undefined;
  topP: number | undefined;
  stops: string[];
  jsonMode: boolean;
  stream: boolean;
}

/** The text of a string or of a block array's text blocks, joined with '\n'. */
export function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** Split a `data:<mime>;base64,<data>` URL into its mime type and base64 data. */
export function splitDataUrl(url: string): { mime: string; data: string } | null {
  const m = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (m === null) return null;
  return { mime: m[1] === '' ? 'image/png' : m[1], data: m[2] };
}

function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

/** OpenAI content parts as canonical parts: text, or an image_url. */
function openAIParts(content: unknown): ContentPart[] {
  if (typeof content === 'string') return content === '' ? [] : [{ kind: 'text', text: content }];
  if (!Array.isArray(content)) return [];
  const out: ContentPart[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') out.push({ kind: 'text', text: block.text });
    else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
      const split = splitDataUrl(block.image_url.url);
      out.push(split === null ? { kind: 'image', mime: '', data: '', url: block.image_url.url } : { kind: 'image', mime: split.mime, data: split.data, url: null });
    }
  }
  return out;
}

/** Anthropic content blocks as canonical parts, plus tool calls / results. */
function anthropicParts(content: unknown, into: CanonMessage): void {
  if (typeof content === 'string') {
    if (content !== '') into.parts.push({ kind: 'text', text: content });
    return;
  }
  if (!Array.isArray(content)) return;
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') into.parts.push({ kind: 'text', text: block.text });
    else if (block.type === 'image' && isRecord(block.source)) {
      const source = block.source;
      if (source.type === 'base64' && typeof source.data === 'string') {
        into.parts.push({ kind: 'image', mime: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data, url: null });
      } else if (source.type === 'url' && typeof source.url === 'string') {
        into.parts.push({ kind: 'image', mime: '', data: '', url: source.url });
      }
    } else if (block.type === 'tool_use') {
      into.toolCalls.push({ id: typeof block.id === 'string' ? block.id : '', name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} });
    } else if (block.type === 'tool_result') {
      into.role = 'tool';
      into.toolCallId = typeof block.tool_use_id === 'string' ? block.tool_use_id : '';
      const text = textOf(block.content);
      into.parts.push({ kind: 'text', text: text !== '' ? text : typeof block.content === 'string' ? block.content : JSON.stringify(block.content ?? '') });
    }
  }
}

/** An OpenAI Chat Completions body as a canonical request. */
export function fromOpenAI(body: Json): CanonRequest {
  const systemParts: string[] = [];
  const messages: CanonMessage[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({ role: 'tool', parts: openAIParts(raw.content), toolCalls: [], toolCallId: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '' });
      continue;
    }
    const msg: CanonMessage = { role: raw.role === 'assistant' ? 'assistant' : 'user', parts: openAIParts(raw.content), toolCalls: [], toolCallId: '' };
    if (raw.role === 'assistant' && Array.isArray(raw.tool_calls)) {
      for (const call of raw.tool_calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        let args: unknown = {};
        if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
          try {
            args = JSON.parse(call.function.arguments);
          } catch {
            args = {};
          }
        } else if (isRecord(call.function.arguments)) args = call.function.arguments;
        msg.toolCalls.push({ id: typeof call.id === 'string' ? call.id : '', name: typeof call.function.name === 'string' ? call.function.name : '', args });
      }
    }
    messages.push(msg);
  }
  const tools: CanonTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
      tools.push({ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } });
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    system: systemParts.join('\n'),
    messages,
    tools,
    maxTokens: numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens),
    temperature: typeof body.temperature === 'number' ? body.temperature : undefined,
    topP: typeof body.top_p === 'number' ? body.top_p : undefined,
    stops: stringList(body.stop),
    jsonMode: format === 'json_object' || format === 'json_schema',
    stream: body.stream === true,
  };
}

/** An Anthropic Messages body as a canonical request. */
export function fromAnthropic(body: Json): CanonRequest {
  const messages: CanonMessage[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const msg: CanonMessage = { role: raw.role === 'assistant' ? 'assistant' : 'user', parts: [], toolCalls: [], toolCallId: '' };
    anthropicParts(raw.content, msg);
    messages.push(msg);
  }
  const tools: CanonTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || typeof raw.name !== 'string') continue;
      tools.push({ name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } });
    }
  }
  // Anthropic has no explicit JSON-mode field in this workload: a request that
  // wants JSON says so in its text, and the provider echoes json_mode false.
  // Never invent a response_format for it: an upstream that carried one
  // would echo json_mode true and break the echo check.
  return {
    system: textOf(body.system),
    messages,
    tools,
    maxTokens: numberOrNull(body.max_tokens),
    temperature: typeof body.temperature === 'number' ? body.temperature : undefined,
    topP: typeof body.top_p === 'number' ? body.top_p : undefined,
    stops: stringList(body.stop_sequences),
    jsonMode: false,
    stream: body.stream === true,
  };
}

/** Canonical parts as OpenAI content: a bare string for one text part, else parts. */
function openAIContent(parts: ContentPart[]): string | Json[] {
  if (parts.length === 1 && parts[0].kind === 'text') return parts[0].text;
  return parts.map((p) => {
    if (p.kind === 'text') return { type: 'text', text: p.text };
    const url = p.url !== null ? p.url : `data:${p.mime === '' ? 'image/png' : p.mime};base64,${p.data}`;
    return { type: 'image_url', image_url: { url } };
  });
}

/** Canonical parts as Anthropic content blocks. */
function anthropicContent(parts: ContentPart[]): Json[] {
  return parts.map((p) => {
    if (p.kind === 'text') return { type: 'text', text: p.text };
    if (p.url !== null) return { type: 'image', source: { type: 'url', url: p.url } };
    return { type: 'image', source: { type: 'base64', media_type: p.mime === '' ? 'image/png' : p.mime, data: p.data } };
  });
}

/** Canonical parts as Gemini content parts. */
function geminiParts(parts: ContentPart[]): Json[] {
  return parts.map((p) => {
    if (p.kind === 'text') return { text: p.text };
    if (p.url !== null) return { fileData: { fileUri: p.url } };
    return { inlineData: { mimeType: p.mime === '' ? 'image/png' : p.mime, data: p.data } };
  });
}

/** A canonical request as an OpenAI Chat Completions body naming `model`. */
export function toOpenAI(c: CanonRequest, model: string): Json {
  const messages: Json[] = [];
  if (c.system !== '') messages.push({ role: 'system', content: c.system });
  for (const msg of c.messages) {
    if (msg.role === 'tool') {
      messages.push({ role: 'tool', tool_call_id: msg.toolCallId, content: textOfParts(msg.parts) });
      continue;
    }
    if (msg.role === 'assistant') {
      const out: Json = { role: 'assistant', content: msg.parts.length === 0 ? null : openAIContent(msg.parts) };
      if (msg.toolCalls.length > 0) {
        out.tool_calls = msg.toolCalls.map((t) => ({ id: t.id, type: 'function', function: { name: t.name, arguments: JSON.stringify(t.args ?? {}) } }));
      }
      messages.push(out);
      continue;
    }
    messages.push({ role: 'user', content: openAIContent(msg.parts) });
  }
  const out: Json = { model, messages };
  if (c.maxTokens !== null) out.max_tokens = c.maxTokens;
  if (c.temperature !== undefined) out.temperature = c.temperature;
  if (c.topP !== undefined) out.top_p = c.topP;
  if (c.stops.length > 0) out.stop = [...c.stops];
  if (c.tools.length > 0) {
    out.tools = c.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.parameters } }));
    out.tool_choice = 'auto';
  }
  if (c.jsonMode) out.response_format = { type: 'json_object' };
  if (c.stream) {
    out.stream = true;
    // Without this the provider sends no usage chunk and the streamed
    // answer would carry no usage for the client to check.
    out.stream_options = { include_usage: true };
  }
  return out;
}

function textOfParts(parts: ContentPart[]): string {
  return parts.map((p) => (p.kind === 'text' ? p.text : '')).join('\n');
}

/** A canonical request as an Anthropic Messages body naming `model`. */
export function toAnthropic(c: CanonRequest, model: string): Json {
  const messages: Json[] = [];
  for (const msg of c.messages) {
    if (msg.role === 'tool') {
      messages.push({ role: 'user', content: [{ type: 'tool_result', tool_use_id: msg.toolCallId, content: textOfParts(msg.parts) }] });
      continue;
    }
    if (msg.role === 'assistant') {
      const content: Json[] = anthropicContent(msg.parts);
      for (const t of msg.toolCalls) content.push({ type: 'tool_use', id: t.id, name: t.name, input: isRecord(t.args) ? t.args : {} });
      messages.push({ role: 'assistant', content });
      continue;
    }
    const blocks = anthropicContent(msg.parts);
    messages.push({ role: 'user', content: blocks.length === 1 && blocks[0].type === 'text' ? (blocks[0].text as string) : blocks });
  }
  const out: Json = { model, messages, max_tokens: c.maxTokens ?? 1024 };
  if (c.system !== '') out.system = c.system;
  if (c.temperature !== undefined) out.temperature = c.temperature;
  if (c.topP !== undefined) out.top_p = c.topP;
  if (c.stops.length > 0) out.stop_sequences = [...c.stops];
  if (c.tools.length > 0) {
    out.tools = c.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
  }
  if (c.stream) out.stream = true;
  return out;
}

/** A canonical request as a Gemini generateContent body naming `model`. */
export function toGemini(c: CanonRequest, model: string): Json {
  const contents: Json[] = [];
  for (const msg of c.messages) {
    if (msg.role === 'tool') {
      contents.push({ role: 'user', parts: [{ text: `Tool result: ${textOfParts(msg.parts)}` }] });
      continue;
    }
    const parts = geminiParts(msg.parts);
    if (msg.role === 'assistant') {
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  const out: Json = { contents };
  if (c.system !== '') out.systemInstruction = { parts: [{ text: c.system }] };
  if (c.tools.length > 0) {
    out.tools = [{ functionDeclarations: c.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })) }];
  }
  const config: Json = {};
  if (c.maxTokens !== null) config.maxOutputTokens = c.maxTokens;
  if (c.temperature !== undefined) config.temperature = c.temperature;
  if (c.topP !== undefined) config.topP = c.topP;
  if (c.stops.length > 0) config.stopSequences = [...c.stops];
  if (c.jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  void model;
  return out;
}

/** One parsed upstream answer: the text, the token usage, and the stop reason. */
export interface UpstreamAnswer {
  text: string;
  inputTokens: number | null;
  outputTokens: number | null;
  stop: string | null;
}

/** One SSE frame: its event name (null when unnamed) and its data lines joined. */
export interface SseFrame {
  event: string | null;
  data: string;
}

/** Split an SSE body into frames, tolerating CRLF. */
export function parseSse(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const chunk of normalized.split('\n\n')) {
    if (chunk.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of chunk.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function jsonOrNull(text: string): Json | null {
  try {
    const v: unknown = JSON.parse(text);
    return isRecord(v) ? v : null;
  } catch {
    return null;
  }
}

function num(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function openAIUsage(body: Json): { input: number | null; output: number | null } {
  const usage = isRecord(body.usage) ? body.usage : {};
  return { input: num(usage.prompt_tokens), output: num(usage.completion_tokens) };
}

/** Parse an OpenAI-wire answer, JSON or SSE. */
export function parseOpenAIAnswer(text: string, stream: boolean): UpstreamAnswer | null {
  if (!stream) {
    const body = jsonOrNull(text);
    if (body === null) return null;
    const choices = Array.isArray(body.choices) ? body.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : {};
    const message = isRecord(choice.message) ? choice.message : {};
    const content = typeof message.content === 'string' ? message.content : textOf(message.content);
    const usage = openAIUsage(body);
    return { text: content, inputTokens: usage.input, outputTokens: usage.output, stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : null };
  }
  let out = '';
  let input: number | null = null;
  let output: number | null = null;
  let stop: string | null = null;
  for (const frame of parseSse(text)) {
    if (frame.data.trim() === '[DONE]') continue;
    const chunk = jsonOrNull(frame.data);
    if (chunk === null) continue;
    const usage = openAIUsage(chunk);
    if (usage.input !== null) input = usage.input;
    if (usage.output !== null) output = usage.output;
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    if (choices.length === 0) continue;
    const choice = isRecord(choices[0]) ? choices[0] : {};
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') out += delta.content;
    if (typeof choice.finish_reason === 'string') stop = choice.finish_reason;
  }
  return { text: out, inputTokens: input, outputTokens: output, stop };
}

/** Parse an Anthropic-wire answer, JSON or SSE. */
export function parseAnthropicAnswer(text: string, stream: boolean): UpstreamAnswer | null {
  if (!stream) {
    const body = jsonOrNull(text);
    if (body === null) return null;
    const usage = isRecord(body.usage) ? body.usage : {};
    return { text: textOf(body.content), inputTokens: num(usage.input_tokens), outputTokens: num(usage.output_tokens), stop: typeof body.stop_reason === 'string' ? body.stop_reason : null };
  }
  let out = '';
  let input: number | null = null;
  let output: number | null = null;
  let stop: string | null = null;
  for (const frame of parseSse(text)) {
    const data = jsonOrNull(frame.data);
    if (data === null) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      const u = num(data.message.usage.input_tokens);
      if (u !== null) input = u;
    } else if (frame.event === 'content_block_delta' && isRecord(data.delta) && data.delta.type === 'text_delta' && typeof data.delta.text === 'string') {
      out += data.delta.text;
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = data.delta.stop_reason;
      if (isRecord(data.usage)) {
        const u = num(data.usage.output_tokens);
        if (u !== null) output = u;
      }
    }
  }
  return { text: out, inputTokens: input, outputTokens: output, stop };
}

/** Parse a Gemini-wire answer, JSON or SSE. */
export function parseGeminiAnswer(text: string, stream: boolean): UpstreamAnswer | null {
  const read = (body: Json): { text: string; input: number | null; output: number | null; stop: string | null } => {
    const candidates = Array.isArray(body.candidates) ? body.candidates : [];
    const cand = isRecord(candidates[0]) ? candidates[0] : {};
    const content = isRecord(cand.content) ? cand.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    const piece = parts.map((p) => (isRecord(p) && typeof p.text === 'string' ? p.text : '')).join('');
    const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
    return { text: piece, input: num(meta.promptTokenCount), output: num(meta.candidatesTokenCount), stop: typeof cand.finishReason === 'string' ? cand.finishReason : null };
  };
  if (!stream) {
    const body = jsonOrNull(text);
    if (body === null) return null;
    const r = read(body);
    return { text: r.text, inputTokens: r.input, outputTokens: r.output, stop: r.stop };
  }
  let out = '';
  let input: number | null = null;
  let output: number | null = null;
  let stop: string | null = null;
  for (const frame of parseSse(text)) {
    const body = jsonOrNull(frame.data);
    if (body === null) continue;
    const r = read(body);
    out += r.text;
    if (r.input !== null) input = r.input;
    if (r.output !== null) output = r.output;
    if (r.stop !== null) stop = r.stop;
  }
  return { text: out, inputTokens: input, outputTokens: output, stop };
}

/** The stop-reason family: 0 ended, 1 hit the cap, 2 called a tool, 3 was refused. */
export function stopFamily(stop: string | null): number | null {
  if (stop === null) return null;
  switch (stop) {
    case 'stop':
    case 'end_turn':
    case 'STOP':
      return 0;
    case 'length':
    case 'max_tokens':
    case 'MAX_TOKENS':
      return 1;
    case 'tool_calls':
    case 'function_call':
    case 'tool_use':
      return 2;
    case 'content_filter':
    case 'refusal':
    case 'SAFETY':
      return 3;
    default:
      return null;
  }
}

/** One family's stop reason in the wire the client spoke. */
export function stopForApi(family: number, api: 'openai' | 'anthropic'): string {
  const openai = ['stop', 'length', 'tool_calls', 'content_filter'];
  const anthropic = ['end_turn', 'max_tokens', 'tool_use', 'refusal'];
  return api === 'openai' ? openai[family] : anthropic[family];
}

/** Split text into chunks for a stream, mirroring the provider's own chunking. */
export function splitText(text: string, size = 48): string[] {
  const out: string[] = [];
  for (let at = 0; at < text.length; at += size) out.push(text.slice(at, at + size));
  return out.length === 0 ? [''] : out;
}

const FIXED_CREATED = 1767225600;

/** An upstream answer as an OpenAI Chat Completions JSON body. */
export function openAIJsonAnswer(answer: UpstreamAnswer, model: string, api: 'openai' | 'anthropic'): Json {
  const stop = stopForApi(stopFamily(answer.stop) ?? 0, api);
  return {
    id: 'chatcmpl-agg',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: stop, logprobs: null }],
    usage: {
      prompt_tokens: answer.inputTokens ?? 0,
      completion_tokens: answer.outputTokens ?? 0,
      total_tokens: (answer.inputTokens ?? 0) + (answer.outputTokens ?? 0),
    },
  };
}

/** An upstream answer as an OpenAI SSE stream, usage chunk before [DONE]. */
export function openAISseAnswer(answer: UpstreamAnswer, model: string, api: 'openai' | 'anthropic'): string {
  const stop = stopForApi(stopFamily(answer.stop) ?? 0, api);
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-agg',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [`data: ${JSON.stringify(chunk({ role: 'assistant', content: '' }, null))}\n\n`];
  for (const piece of splitText(answer.text)) frames.push(`data: ${JSON.stringify(chunk({ content: piece }, null))}\n\n`);
  frames.push(`data: ${JSON.stringify(chunk({}, stop))}\n\n`);
  frames.push(
    `data: ${JSON.stringify({
      id: 'chatcmpl-agg',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: {
        prompt_tokens: answer.inputTokens ?? 0,
        completion_tokens: answer.outputTokens ?? 0,
        total_tokens: (answer.inputTokens ?? 0) + (answer.outputTokens ?? 0),
      },
    })}\n\n`,
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** An upstream answer as an Anthropic Messages JSON body. */
export function anthropicJsonAnswer(answer: UpstreamAnswer, model: string): Json {
  const stop = stopForApi(stopFamily(answer.stop) ?? 0, 'anthropic');
  return {
    id: 'msg_agg',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: stop,
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens ?? 0, output_tokens: answer.outputTokens ?? 0 },
  };
}

/** An upstream answer as an Anthropic SSE stream, in the required event order. */
export function anthropicSseAnswer(answer: UpstreamAnswer, model: string): string {
  const stop = stopForApi(stopFamily(answer.stop) ?? 0, 'anthropic');
  const event = (name: string, data: unknown): string => `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`;
  const frames: string[] = [
    event('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_agg',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens ?? 0, output_tokens: 0 },
      },
    }),
    event('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const piece of splitText(answer.text)) {
    frames.push(event('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(event('content_block_stop', { type: 'content_block_stop', index: 0 }));
  frames.push(
    event('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: stop, stop_sequence: null },
      usage: { output_tokens: answer.outputTokens ?? 0 },
    }),
  );
  frames.push(event('message_stop', { type: 'message_stop' }));
  return frames.join('');
}
