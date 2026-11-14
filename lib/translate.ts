// Cross-wire translation for free-model-aggregator-01.
//
// The trial's client speaks OpenAI or Anthropic; its providers speak OpenAI,
// Anthropic or Gemini. Same-wire calls are forwarded byte for byte (see
// main.ts). This file is the cross-wire half: parse any provider's answer
// (JSON or SSE) into one normalized answer, render it in either client wire,
// and build the upstream request body for a provider on another wire.
//
// Everything here is a pure function of its arguments: no clock, no
// randomness, so a replay renders the same bytes. The providers' answers are
// short echo texts; every stream is buffered fully, then re-rendered, which
// keeps the usage and stop reason exact in every wire.

import { isRecord, type Json } from './providers.ts';

/** One provider answer, wire-free: the text whole, the usage, the stop. */
export interface NormalizedAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  /** The provider's stop reason in its own spelling (stop/end_turn/STOP/...). */
  stop: string;
  toolCalls: { id: string; name: string; arguments: string }[];
}

const EMPTY: NormalizedAnswer = { text: '', inputTokens: 0, outputTokens: 0, stop: 'stop', toolCalls: [] };

function num(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

function textOfBlocks(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
}

/** One SSE frame's lines: the event name (or null) and its data lines joined. */
export interface SseFrame {
  event: string | null;
  data: string;
}

export function parseSseFrames(text: string): SseFrame[] {
  const out: SseFrame[] = [];
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
    }
    if (data.length > 0) out.push({ event, data: data.join('\n') });
  }
  return out;
}

function jsonOf(data: string): Json | null {
  try {
    const v: unknown = JSON.parse(data);
    return isRecord(v) ? v : null;
  } catch {
    return null;
  }
}

/** Split text into pieces of at most n chars (the providers use 13). */
export function splitText(text: string, n = 13): string[] {
  if (text === '') return [];
  const out: string[] = [];
  for (let at = 0; at < text.length; at += n) out.push(text.slice(at, at + n));
  return out;
}

/** A stop reason mapped into the family the client wire names. */
export function mapStop(stop: string, wire: 'openai' | 'anthropic'): string {
  if (stop === 'length' || stop === 'max_tokens' || stop === 'MAX_TOKENS') return wire === 'openai' ? 'length' : 'max_tokens';
  if (stop === 'tool_calls' || stop === 'function_call' || stop === 'tool_use') return wire === 'openai' ? 'tool_calls' : 'tool_use';
  if (stop === 'content_filter' || stop === 'refusal' || stop === 'SAFETY') return wire === 'openai' ? 'content_filter' : 'refusal';
  return wire === 'openai' ? 'stop' : 'end_turn';
}

// ---- parsing: one wire's answer into the normalized form ----

/** An OpenAI Chat Completions JSON answer. */
export function parseOpenAIJson(body: unknown): NormalizedAnswer {
  if (!isRecord(body)) return { ...EMPTY };
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const usage = isRecord(body.usage) ? body.usage : {};
  const toolCalls: NormalizedAnswer['toolCalls'] = [];
  if (Array.isArray(message.tool_calls)) {
    for (const c of message.tool_calls) {
      if (!isRecord(c) || !isRecord(c.function)) continue;
      toolCalls.push({
        id: typeof c.id === 'string' ? c.id : '',
        name: typeof c.function.name === 'string' ? c.function.name : '',
        arguments: typeof c.function.arguments === 'string' ? c.function.arguments : '',
      });
    }
  }
  return {
    text: typeof message.content === 'string' ? message.content : textOfBlocks(message.content),
    inputTokens: num(usage.prompt_tokens),
    outputTokens: num(usage.completion_tokens),
    stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop',
    toolCalls,
  };
}

/** An OpenAI Chat Completions SSE stream. */
export function parseOpenAISse(text: string): NormalizedAnswer {
  const ans: NormalizedAnswer = { text: '', inputTokens: 0, outputTokens: 0, stop: 'stop', toolCalls: [] };
  const byIndex = new Map<number, { id: string; name: string; arguments: string }>();
  for (const frame of parseSseFrames(text)) {
    if (frame.data.trim() === '[DONE]') continue;
    const chunk = jsonOf(frame.data);
    if (chunk === null) continue;
    if (isRecord(chunk.usage)) {
      ans.inputTokens = num(chunk.usage.prompt_tokens);
      ans.outputTokens = num(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') ans.text += delta.content;
    if (Array.isArray(delta.tool_calls)) {
      for (const c of delta.tool_calls) {
        if (!isRecord(c)) continue;
        const index = typeof c.index === 'number' ? c.index : 0;
        const slot = byIndex.get(index) ?? { id: '', name: '', arguments: '' };
        if (typeof c.id === 'string' && c.id !== '') slot.id = c.id;
        const fn = isRecord(c.function) ? c.function : {};
        if (typeof fn.name === 'string' && fn.name !== '') slot.name = fn.name;
        if (typeof fn.arguments === 'string') slot.arguments += fn.arguments;
        byIndex.set(index, slot);
      }
    }
    if (typeof choice.finish_reason === 'string') ans.stop = choice.finish_reason;
  }
  ans.toolCalls = [...byIndex.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([, v]) => v);
  return ans;
}

/** An Anthropic Messages JSON answer. */
export function parseAnthropicJson(body: unknown): NormalizedAnswer {
  if (!isRecord(body)) return { ...EMPTY };
  const usage = isRecord(body.usage) ? body.usage : {};
  const content = Array.isArray(body.content) ? body.content : [];
  let text = '';
  const toolCalls: NormalizedAnswer['toolCalls'] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') text += block.text;
    else if (block.type === 'tool_use') {
      toolCalls.push({
        id: typeof block.id === 'string' ? block.id : '',
        name: typeof block.name === 'string' ? block.name : '',
        arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
      });
    }
  }
  return {
    text,
    inputTokens: num(usage.input_tokens),
    outputTokens: num(usage.output_tokens),
    stop: typeof body.stop_reason === 'string' ? body.stop_reason : 'end_turn',
    toolCalls,
  };
}

/** An Anthropic Messages SSE stream. */
export function parseAnthropicSse(text: string): NormalizedAnswer {
  const ans: NormalizedAnswer = { text: '', inputTokens: 0, outputTokens: 0, stop: 'end_turn', toolCalls: [] };
  const blocks = new Map<number, { type: string; id: string; name: string; args: string }>();
  for (const frame of parseSseFrames(text)) {
    const data = jsonOf(frame.data);
    if (data === null) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      ans.inputTokens = num(data.message.usage.input_tokens);
    } else if (frame.event === 'content_block_start') {
      const index = num(data.index);
      const block = isRecord(data.content_block) ? data.content_block : {};
      blocks.set(index, {
        type: typeof block.type === 'string' ? block.type : 'text',
        id: typeof block.id === 'string' ? block.id : '',
        name: typeof block.name === 'string' ? block.name : '',
        args: '',
      });
    } else if (frame.event === 'content_block_delta') {
      const index = num(data.index);
      const delta = isRecord(data.delta) ? data.delta : {};
      const slot = blocks.get(index);
      if (delta.type === 'text_delta' && typeof delta.text === 'string') ans.text += delta.text;
      else if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string' && slot !== undefined) slot.args += delta.partial_json;
    } else if (frame.event === 'message_delta') {
      const delta = isRecord(data.delta) ? data.delta : {};
      if (typeof delta.stop_reason === 'string') ans.stop = delta.stop_reason;
      if (isRecord(data.usage)) ans.outputTokens = num(data.usage.output_tokens);
    }
  }
  for (const slot of [...blocks.values()]) {
    if (slot.type === 'tool_use') ans.toolCalls.push({ id: slot.id, name: slot.name, arguments: slot.args });
  }
  return ans;
}

/** A Gemini generateContent JSON answer. */
export function parseGeminiJson(body: unknown): NormalizedAnswer {
  if (!isRecord(body)) return { ...EMPTY };
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const cand = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(cand.content) ? cand.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  let text = '';
  const toolCalls: NormalizedAnswer['toolCalls'] = [];
  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string') text += part.text;
    else if (isRecord(part.functionCall)) {
      toolCalls.push({
        id: '',
        name: typeof part.functionCall.name === 'string' ? part.functionCall.name : '',
        arguments: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}),
      });
    }
  }
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text,
    inputTokens: num(usage.promptTokenCount),
    outputTokens: num(usage.candidatesTokenCount),
    stop: typeof cand.finishReason === 'string' ? cand.finishReason : 'STOP',
    toolCalls,
  };
}

/** A Gemini streamGenerateContent SSE stream. */
export function parseGeminiSse(text: string): NormalizedAnswer {
  const ans: NormalizedAnswer = { text: '', inputTokens: 0, outputTokens: 0, stop: 'STOP', toolCalls: [] };
  for (const frame of parseSseFrames(text)) {
    const chunk = jsonOf(frame.data);
    if (chunk === null) continue;
    const part = parseGeminiJson(chunk);
    ans.text += part.text;
    if (part.inputTokens !== 0 || part.outputTokens !== 0) {
      ans.inputTokens = part.inputTokens;
      ans.outputTokens = part.outputTokens;
    }
    if (part.stop !== 'STOP' || ans.stop === 'STOP') ans.stop = part.stop;
    ans.toolCalls.push(...part.toolCalls);
  }
  return ans;
}

// ---- rendering: the normalized answer in a client wire ----

const FIXED_CREATED = 1767225600;

export function renderOpenAIJson(ans: NormalizedAnswer, model: string): Json {
  const message: Json = { role: 'assistant', content: ans.text };
  if (ans.toolCalls.length > 0) {
    message.tool_calls = ans.toolCalls.map((c) => ({
      id: c.id !== '' ? c.id : 'call_unknown',
      type: 'function',
      function: { name: c.name, arguments: c.arguments },
    }));
  }
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: mapStop(ans.stop, 'openai'), logprobs: null }],
    usage: { prompt_tokens: ans.inputTokens, completion_tokens: ans.outputTokens, total_tokens: ans.inputTokens + ans.outputTokens },
  };
}

function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

export function renderOpenAISse(ans: NormalizedAnswer, model: string): string {
  const frames: string[] = [];
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-sim',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  frames.push(sseEvent(null, chunk({ role: 'assistant', content: '' }, null)));
  for (const piece of splitText(ans.text)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  ans.toolCalls.forEach((c, i) => {
    frames.push(sseEvent(null, chunk({ tool_calls: [{ index: i, id: c.id !== '' ? c.id : `call_${i}`, type: 'function', function: { name: c.name, arguments: '' } }] }, null)));
    for (const piece of splitText(c.arguments)) {
      frames.push(sseEvent(null, chunk({ tool_calls: [{ index: i, function: { arguments: piece } }] }, null)));
    }
  });
  frames.push(sseEvent(null, chunk({}, mapStop(ans.stop, 'openai'))));
  frames.push(
    sseEvent(null, {
      id: 'chatcmpl-sim',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: ans.inputTokens, completion_tokens: ans.outputTokens, total_tokens: ans.inputTokens + ans.outputTokens },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

export function renderAnthropicJson(ans: NormalizedAnswer, model: string): Json {
  const content: Json[] = [];
  if (ans.text !== '') content.push({ type: 'text', text: ans.text });
  for (const c of ans.toolCalls) {
    let input: Json = {};
    try {
      const parsed: unknown = JSON.parse(c.arguments);
      if (isRecord(parsed)) input = parsed;
    } catch {
      // Not JSON: an empty input.
    }
    content.push({ type: 'tool_use', id: c.id !== '' ? c.id : 'toolu_unknown', name: c.name, input });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  return {
    id: 'msg_aggregate',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: mapStop(ans.stop, 'anthropic'),
    stop_sequence: null,
    usage: { input_tokens: ans.inputTokens, output_tokens: ans.outputTokens },
  };
}

export function renderAnthropicSse(ans: NormalizedAnswer, model: string): string {
  const frames: string[] = [];
  frames.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_stream',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: ans.inputTokens, output_tokens: 0 },
      },
    }),
  );
  let index = 0;
  if (ans.text !== '') {
    frames.push(sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } }));
    for (const piece of splitText(ans.text)) {
      frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'text_delta', text: piece } }));
    }
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  for (const c of ans.toolCalls) {
    frames.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: { type: 'tool_use', id: c.id !== '' ? c.id : `toolu_${index}`, name: c.name, input: {} },
      }),
    );
    for (const piece of splitText(c.arguments)) {
      frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: piece } }));
    }
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  frames.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: mapStop(ans.stop, 'anthropic'), stop_sequence: null },
      usage: { output_tokens: ans.outputTokens },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}

// ---- request building: a client body in a provider's wire ----

/** An image part of a client body as Gemini inlineData/fileData, or null. */
function geminiImagePart(url: unknown): Json | null {
  if (typeof url !== 'string' || url === '') return null;
  const data = /^data:([^;,]+);base64,(.*)$/s.exec(url);
  if (data !== null) return { inlineData: { mimeType: data[1], data: data[2] } };
  return { fileData: { fileUri: url } };
}

/** An image part of a client body as an Anthropic image block, or null. */
function anthropicImageBlock(url: unknown): Json | null {
  if (typeof url !== 'string' || url === '') return null;
  const data = /^data:([^;,]+);base64,(.*)$/s.exec(url);
  if (data !== null) return { type: 'image', source: { type: 'base64', media_type: data[1], data: data[2] } };
  return { type: 'image', source: { type: 'url', url } };
}

/** OpenAI `stop` (string or array) as an array of strings. */
function stopList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (Array.isArray(v)) return v.filter((x): x is string => typeof x === 'string');
  return [];
}

function maxTokensOf(body: Json): number | null {
  if (typeof body.max_tokens === 'number') return body.max_tokens;
  if (typeof body.max_completion_tokens === 'number') return body.max_completion_tokens;
  return null;
}

/** OpenAI chat messages as Gemini contents (system handled separately). */
function openAIMessagesToGemini(body: Json): Json[] {
  const contents: Json[] = [];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') continue;
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      const text = textOfBlocks(raw.content);
      if (text !== '') parts.push({ text });
      if (Array.isArray(raw.tool_calls)) {
        for (const c of raw.tool_calls) {
          if (!isRecord(c) || !isRecord(c.function)) continue;
          let args: Json = {};
          try {
            const parsed: unknown = JSON.parse(typeof c.function.arguments === 'string' ? c.function.arguments : '');
            if (isRecord(parsed)) args = parsed;
          } catch {
            // Not JSON: an empty input.
          }
          parts.push({ functionCall: { name: typeof c.function.name === 'string' ? c.function.name : '', args } });
        }
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (raw.role === 'tool') {
      contents.push({
        role: 'user',
        parts: [{ functionResponse: { name: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', response: { result: typeof raw.content === 'string' ? raw.content : '' } } }],
      });
      continue;
    }
    if (raw.role !== 'user') continue;
    const parts: Json[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (block.type === 'image_url' && isRecord(block.image_url)) {
          const part = geminiImagePart(block.image_url.url);
          if (part !== null) parts.push(part);
        }
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  return contents;
}

function geminiToolsAndConfig(body: Json, tools: Json[], toolChoice: unknown): { tools: Json[]; toolConfig: Json | null } {
  void body;
  const out: Json[] = [];
  if (tools.length > 0) {
    out.push({
      functionDeclarations: tools.map((t) => {
        const fn = isRecord(t.function) ? t.function : t;
        return {
          name: typeof fn.name === 'string' ? fn.name : '',
          description: typeof fn.description === 'string' ? fn.description : '',
          parameters: isRecord(fn.parameters) ? fn.parameters : isRecord(fn.input_schema) ? fn.input_schema : { type: 'object', properties: {} },
        };
      }),
    });
  }
  let toolConfig: Json | null = null;
  if (typeof toolChoice === 'string') {
    if (toolChoice === 'auto') toolConfig = { functionCallingConfig: { mode: 'AUTO' } };
    else if (toolChoice === 'required') toolConfig = { functionCallingConfig: { mode: 'ANY' } };
    else if (toolChoice === 'none') toolConfig = { functionCallingConfig: { mode: 'NONE' } };
  } else if (isRecord(toolChoice)) {
    if (toolChoice.type === 'auto') toolConfig = { functionCallingConfig: { mode: 'AUTO' } };
    else if (toolChoice.type === 'any') toolConfig = { functionCallingConfig: { mode: 'ANY' } };
    else if (toolChoice.type === 'none') toolConfig = { functionCallingConfig: { mode: 'NONE' } };
    else if (toolChoice.type === 'tool' && typeof toolChoice.name === 'string') {
      toolConfig = { functionCallingConfig: { mode: 'ANY', allowedFunctionNames: [toolChoice.name] } };
    } else if (toolChoice.type === 'function' && isRecord(toolChoice.function) && typeof toolChoice.function.name === 'string') {
      toolConfig = { functionCallingConfig: { mode: 'ANY', allowedFunctionNames: [toolChoice.function.name] } };
    }
  }
  return { tools: out, toolConfig };
}

/** An OpenAI Chat Completions request as a Gemini generateContent body. */
export function openAIToGemini(body: Json, model: string): Json {
  void model;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const systemParts: string[] = [];
  for (const raw of messages) {
    if (isRecord(raw) && raw.role === 'system') {
      const text = textOfBlocks(raw.content);
      if (text !== '') systemParts.push(text);
    }
  }
  const tools = Array.isArray(body.tools) ? body.tools.filter(isRecord) : [];
  const { tools: geminiTools, toolConfig } = geminiToolsAndConfig(body, tools, body.tool_choice);
  const config: Json = {};
  const maxTokens = maxTokensOf(body);
  if (maxTokens !== null) config.maxOutputTokens = maxTokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const stops = stopList(body.stop);
  if (stops.length > 0) config.stopSequences = stops;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  if (format === 'json_object' || format === 'json_schema') config.responseMimeType = 'application/json';
  const out: Json = { contents: openAIMessagesToGemini(body) };
  if (systemParts.length > 0) out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  if (geminiTools.length > 0) out.tools = geminiTools;
  if (toolConfig !== null) out.toolConfig = toolConfig;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** Anthropic messages as Gemini contents (system handled separately). */
function anthropicMessagesToGemini(body: Json): Json[] {
  const contents: Json[] = [];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
          else if (block.type === 'tool_use') {
            parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
          }
        }
      } else if (typeof raw.content === 'string' && raw.content !== '') parts.push({ text: raw.content });
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (raw.role !== 'user') continue;
    const parts: Json[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (block.type === 'image' && isRecord(block.source)) {
          if (block.source.type === 'base64' && typeof block.source.data === 'string') {
            parts.push({ inlineData: { mimeType: typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png', data: block.source.data } });
          } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
            parts.push({ fileData: { fileUri: block.source.url } });
          }
        } else if (block.type === 'tool_result') {
          const content = block.content;
          parts.push({
            functionResponse: {
              name: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
              response: { result: typeof content === 'string' ? content : textOfBlocks(content) },
            },
          });
        }
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  return contents;
}

/** An Anthropic Messages request as a Gemini generateContent body. */
export function anthropicToGemini(body: Json, model: string): Json {
  void model;
  const systemText = textOfBlocks(body.system);
  const tools = Array.isArray(body.tools)
    ? body.tools.filter(isRecord).map((t) => ({ name: t.name, description: t.description, input_schema: t.input_schema }))
    : [];
  const { tools: geminiTools, toolConfig } = geminiToolsAndConfig(body, tools, body.tool_choice);
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Array.isArray(body.stop_sequences)) {
    const stops = body.stop_sequences.filter((x): x is string => typeof x === 'string');
    if (stops.length > 0) config.stopSequences = stops;
  }
  const out: Json = { contents: anthropicMessagesToGemini(body) };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  if (geminiTools.length > 0) out.tools = geminiTools;
  if (toolConfig !== null) out.toolConfig = toolConfig;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** OpenAI chat messages as Anthropic messages; system text returned beside them. */
function openAIMessagesToAnthropic(body: Json): { system: string; messages: Json[] } {
  const systemParts: string[] = [];
  const messages: Json[] = [];
  const rawMessages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of rawMessages) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOfBlocks(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOfBlocks(raw.content);
      if (text !== '') blocks.push({ type: 'text', text });
      if (Array.isArray(raw.tool_calls)) {
        for (const c of raw.tool_calls) {
          if (!isRecord(c) || !isRecord(c.function)) continue;
          let input: Json = {};
          try {
            const parsed: unknown = JSON.parse(typeof c.function.arguments === 'string' ? c.function.arguments : '');
            if (isRecord(parsed)) input = parsed;
          } catch {
            // Not JSON: an empty input.
          }
          blocks.push({ type: 'tool_use', id: typeof c.id === 'string' ? c.id : '', name: typeof c.function.name === 'string' ? c.function.name : '', input });
        }
      }
      messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: typeof raw.content === 'string' ? raw.content : '' }],
      });
      continue;
    }
    if (raw.role !== 'user') continue;
    if (typeof raw.content === 'string') {
      messages.push({ role: 'user', content: raw.content });
      continue;
    }
    if (!Array.isArray(raw.content)) continue;
    const blocks: Json[] = [];
    for (const block of raw.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') blocks.push({ type: 'text', text: block.text });
      else if (block.type === 'image_url' && isRecord(block.image_url)) {
        const image = anthropicImageBlock(block.image_url.url);
        if (image !== null) blocks.push(image);
      }
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }
  return { system: systemParts.join('\n'), messages };
}

/** An OpenAI Chat Completions request as an Anthropic Messages request. */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const { system, messages } = openAIMessagesToAnthropic(body);
  const out: Json = { model, messages };
  const maxTokens = maxTokensOf(body);
  if (maxTokens !== null) out.max_tokens = maxTokens;
  if (system !== '') out.system = system;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stopList(body.stop);
  if (stops.length > 0) out.stop_sequences = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (typeof choice === 'string') {
    if (choice === 'auto') out.tool_choice = { type: 'auto' };
    else if (choice === 'required') out.tool_choice = { type: 'any' };
    else if (choice === 'none') out.tool_choice = { type: 'none' };
  } else if (isRecord(choice)) {
    if (choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
      out.tool_choice = { type: 'tool', name: choice.function.name };
    }
  }
  if (body.stream === true) out.stream = true;
  return out;
}
