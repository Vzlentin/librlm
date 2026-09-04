import { ABSENT_FINAL } from './types.ts';
import type {
  CodeBlock,
  FinalValue,
  JSONValue,
  LogMetadata,
  REPLResult,
  RLMChatCompletion,
  RLMConfigMetadata,
  RLMIteration,
  RLMLogFile,
} from './types.ts';

function fail(message: string): never {
  throw new Error(message);
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function hasOwn(value: Record<string, unknown>, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(value, key);
}

function isObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function requireObject(value: unknown, path: string): Record<string, unknown> {
  if (!isObject(value)) {
    fail(`${path} must be an object`);
  }
  return value;
}

function requireField(value: Record<string, unknown>, key: string, path: string): unknown {
  if (!hasOwn(value, key)) {
    fail(`${path}.${key} is required`);
  }
  return value[key];
}

function requireString(value: unknown, path: string): string {
  if (typeof value !== 'string') {
    fail(`${path} must be a string`);
  }
  return value;
}

function requireNumber(value: unknown, path: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    fail(`${path} must be a finite number`);
  }
  return value;
}

function requireNullableNumber(value: unknown, path: string): number | null {
  return value === null ? null : requireNumber(value, path);
}

function requireJSONValue(value: unknown, path: string): JSONValue {
  if (
    value === null ||
    typeof value === 'string' ||
    typeof value === 'boolean'
  ) {
    return value;
  }
  if (typeof value === 'number') {
    return requireNumber(value, path);
  }
  if (Array.isArray(value)) {
    return value.map((item, index) => requireJSONValue(item, `${path}[${index}]`));
  }
  if (typeof value === 'object') {
    const normalized: { [key: string]: JSONValue } = {};
    for (const [key, item] of Object.entries(value)) {
      normalized[key] = requireJSONValue(item, `${path}.${key}`);
    }
    return normalized;
  }
  return fail(`${path} must be a JSON value`);
}

function parseCanonicalFinal(value: unknown, path: string): FinalValue {
  const final = requireObject(value, path);
  const keys = Object.keys(final);
  if (
    keys.length !== 2 ||
    !hasOwn(final, 'has_final') ||
    !hasOwn(final, 'final_value')
  ) {
    fail(`${path} must contain exactly has_final and final_value`);
  }

  const hasFinal = final.has_final;
  if (typeof hasFinal !== 'boolean') {
    fail(`${path}.has_final must be a boolean`);
  }
  const finalValue = requireJSONValue(final.final_value, `${path}.final_value`);
  if (!hasFinal && finalValue !== null) {
    fail(`${path}.final_value must be null when has_final is false`);
  }
  return { has_final: hasFinal, final_value: finalValue };
}

function normalizeLegacyFinalValue(value: unknown, path: string): JSONValue {
  const normalized = requireJSONValue(value, path);
  if (
    Array.isArray(normalized) &&
    normalized.length === 2 &&
    typeof normalized[0] === 'string' &&
    typeof normalized[1] === 'string'
  ) {
    return normalized[1];
  }
  return normalized;
}

function normalizeFinal(
  container: Record<string, unknown>,
  path: string,
  required: boolean,
): FinalValue {
  const hasNested = hasOwn(container, 'final');
  const hasFlatPresence = hasOwn(container, 'has_final');
  const hasFlatValue = hasOwn(container, 'final_value');
  const hasLegacyPresence = hasOwn(container, 'has_final_answer');
  const hasLegacyValue = hasOwn(container, 'final_answer');

  if (hasNested) {
    if (hasFlatPresence || hasFlatValue || hasLegacyPresence || hasLegacyValue) {
      fail(`${path} contains conflicting final representations`);
    }
    return parseCanonicalFinal(container.final, `${path}.final`);
  }

  if (hasFlatPresence || hasFlatValue) {
    if (!hasFlatPresence || !hasFlatValue) {
      fail(`${path} requires both has_final and final_value`);
    }
    if (hasLegacyPresence || hasLegacyValue) {
      fail(`${path} contains conflicting final representations`);
    }
    return parseCanonicalFinal(
      { has_final: container.has_final, final_value: container.final_value },
      path,
    );
  }

  if (hasLegacyPresence) {
    if (!hasLegacyValue) {
      fail(`${path}.final_answer is required with has_final_answer`);
    }
    if (typeof container.has_final_answer !== 'boolean') {
      fail(`${path}.has_final_answer must be a boolean`);
    }
    const finalValue = normalizeLegacyFinalValue(
      container.final_answer,
      `${path}.final_answer`,
    );
    if (!container.has_final_answer && finalValue !== null) {
      fail(`${path}.final_answer must be null when has_final_answer is false`);
    }
    return {
      has_final: container.has_final_answer,
      final_value: finalValue,
    };
  }

  if (hasLegacyValue) {
    if (container.final_answer === null) {
      return { ...ABSENT_FINAL };
    }
    return {
      has_final: true,
      final_value: normalizeLegacyFinalValue(
        container.final_answer,
        `${path}.final_answer`,
      ),
    };
  }

  if (required) {
    fail(`${path}.final is required`);
  }
  return { ...ABSENT_FINAL };
}

function parseTokenCounts(
  call: Record<string, unknown>,
  path: string,
): { promptTokens: number; completionTokens: number } {
  const hasPromptTokens = hasOwn(call, 'prompt_tokens');
  const hasCompletionTokens = hasOwn(call, 'completion_tokens');
  if (hasPromptTokens || hasCompletionTokens) {
    if (!hasPromptTokens || !hasCompletionTokens) {
      fail(`${path} requires both prompt_tokens and completion_tokens`);
    }
    return {
      promptTokens: requireNumber(call.prompt_tokens, `${path}.prompt_tokens`),
      completionTokens: requireNumber(
        call.completion_tokens,
        `${path}.completion_tokens`,
      ),
    };
  }

  const usage = requireObject(
    requireField(call, 'usage_summary', path),
    `${path}.usage_summary`,
  );
  const summaries = requireObject(
    requireField(usage, 'model_usage_summaries', `${path}.usage_summary`),
    `${path}.usage_summary.model_usage_summaries`,
  );
  let promptTokens = 0;
  let completionTokens = 0;
  for (const [model, rawSummary] of Object.entries(summaries)) {
    const summaryPath = `${path}.usage_summary.model_usage_summaries.${model}`;
    const summary = requireObject(rawSummary, summaryPath);
    promptTokens += requireNumber(
      requireField(summary, 'total_input_tokens', summaryPath),
      `${summaryPath}.total_input_tokens`,
    );
    completionTokens += requireNumber(
      requireField(summary, 'total_output_tokens', summaryPath),
      `${summaryPath}.total_output_tokens`,
    );
  }
  return { promptTokens, completionTokens };
}

function parseRLMCall(value: unknown, path: string): RLMChatCompletion {
  const call = requireObject(value, path);
  const promptValue = requireField(call, 'prompt', path);
  if (
    typeof promptValue !== 'string' &&
    (promptValue === null || typeof promptValue !== 'object' || Array.isArray(promptValue))
  ) {
    fail(`${path}.prompt must be a string or object`);
  }
  const prompt =
    typeof promptValue === 'string'
      ? promptValue
      : requireObject(promptValue, `${path}.prompt`);
  const { promptTokens, completionTokens } = parseTokenCounts(call, path);

  return {
    prompt,
    response: requireString(requireField(call, 'response', path), `${path}.response`),
    prompt_tokens: promptTokens,
    completion_tokens: completionTokens,
    execution_time: requireNumber(
      requireField(call, 'execution_time', path),
      `${path}.execution_time`,
    ),
    final: normalizeFinal(call, path, false),
  };
}

function parseREPLResult(value: unknown, path: string): REPLResult {
  const result = requireObject(value, path);
  const rawLocals = requireObject(
    requireField(result, 'locals', path),
    `${path}.locals`,
  );
  const rawCalls = requireField(result, 'rlm_calls', path);
  if (!Array.isArray(rawCalls)) {
    fail(`${path}.rlm_calls must be an array`);
  }

  return {
    stdout: requireString(requireField(result, 'stdout', path), `${path}.stdout`),
    stderr: requireString(requireField(result, 'stderr', path), `${path}.stderr`),
    locals: rawLocals,
    execution_time: requireNullableNumber(
      requireField(result, 'execution_time', path),
      `${path}.execution_time`,
    ),
    rlm_calls: rawCalls.map((call, index) =>
      parseRLMCall(call, `${path}.rlm_calls[${index}]`),
    ),
    final: normalizeFinal(result, path, false),
  };
}

function parseCodeBlock(value: unknown, path: string): CodeBlock {
  const block = requireObject(value, path);
  return {
    code: requireString(requireField(block, 'code', path), `${path}.code`),
    result: parseREPLResult(requireField(block, 'result', path), `${path}.result`),
  };
}

function parsePrompt(value: unknown, path: string): Array<{ role: string; content: string }> {
  if (!Array.isArray(value)) {
    fail(`${path} must be an array`);
  }
  return value.map((rawMessage, index) => {
    const messagePath = `${path}[${index}]`;
    const message = requireObject(rawMessage, messagePath);
    return {
      role: requireString(requireField(message, 'role', messagePath), `${messagePath}.role`),
      content: requireString(
        requireField(message, 'content', messagePath),
        `${messagePath}.content`,
      ),
    };
  });
}

function parseIteration(value: Record<string, unknown>): RLMIteration {
  if (hasOwn(value, 'type') && value.type !== 'iteration') {
    fail('iteration.type must be "iteration" when present');
  }

  const rawBlocks = requireField(value, 'code_blocks', 'iteration');
  if (!Array.isArray(rawBlocks)) {
    fail('iteration.code_blocks must be an array');
  }

  return {
    type: hasOwn(value, 'type') ? 'iteration' : undefined,
    iteration: requireNumber(
      requireField(value, 'iteration', 'iteration'),
      'iteration.iteration',
    ),
    timestamp: requireString(
      requireField(value, 'timestamp', 'iteration'),
      'iteration.timestamp',
    ),
    prompt: parsePrompt(requireField(value, 'prompt', 'iteration'), 'iteration.prompt'),
    response: requireString(
      requireField(value, 'response', 'iteration'),
      'iteration.response',
    ),
    code_blocks: rawBlocks.map((block, index) =>
      parseCodeBlock(block, `iteration.code_blocks[${index}]`),
    ),
    final: normalizeFinal(value, 'iteration', true),
    iteration_time: requireNullableNumber(
      requireField(value, 'iteration_time', 'iteration'),
      'iteration.iteration_time',
    ),
  };
}

function optionalNullableString(
  value: Record<string, unknown>,
  key: string,
): string | null {
  if (!hasOwn(value, key) || value[key] === null) {
    return null;
  }
  return requireString(value[key], `metadata.${key}`);
}

function optionalNullableNumber(
  value: Record<string, unknown>,
  key: string,
): number | null {
  if (!hasOwn(value, key) || value[key] === null) {
    return null;
  }
  return requireNumber(value[key], `metadata.${key}`);
}

function optionalNullableObject(
  value: Record<string, unknown>,
  key: string,
): Record<string, unknown> | null {
  if (!hasOwn(value, key) || value[key] === null) {
    return null;
  }
  return requireObject(value[key], `metadata.${key}`);
}

function optionalOtherBackends(value: Record<string, unknown>): string[] | null {
  if (!hasOwn(value, 'other_backends') || value.other_backends === null) {
    return null;
  }
  if (!Array.isArray(value.other_backends)) {
    fail('metadata.other_backends must be an array or null');
  }
  return value.other_backends.map((backend, index) =>
    requireString(backend, `metadata.other_backends[${index}]`),
  );
}

function parseConfig(value: Record<string, unknown>): RLMConfigMetadata {
  return {
    root_model: optionalNullableString(value, 'root_model'),
    max_depth: optionalNullableNumber(value, 'max_depth'),
    max_iterations: optionalNullableNumber(value, 'max_iterations'),
    backend: optionalNullableString(value, 'backend'),
    backend_kwargs: optionalNullableObject(value, 'backend_kwargs'),
    environment_type: optionalNullableString(value, 'environment_type'),
    environment_kwargs: optionalNullableObject(value, 'environment_kwargs'),
    other_backends: optionalOtherBackends(value),
  };
}

function getDefaultConfig(): RLMConfigMetadata {
  return {
    root_model: null,
    max_depth: null,
    max_iterations: null,
    backend: null,
    backend_kwargs: null,
    environment_type: null,
    environment_kwargs: null,
    other_backends: null,
  };
}

export interface ParsedJSONL {
  iterations: RLMIteration[];
  config: RLMConfigMetadata;
}

export function parseJSONL(content: string): ParsedJSONL {
  const iterations: RLMIteration[] = [];
  let config = getDefaultConfig();
  const lines = content.split(/\r?\n/);

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    if (!line.trim()) {
      continue;
    }

    let parsed: unknown;
    try {
      parsed = JSON.parse(line);
    } catch (error) {
      throw new Error(
        `Invalid JSONL at line ${index + 1}: malformed JSON (${errorMessage(error)})`,
      );
    }

    try {
      const entry = requireObject(parsed, 'entry');
      if (entry.type === 'metadata') {
        config = parseConfig(entry);
      } else {
        iterations.push(parseIteration(entry));
      }
    } catch (error) {
      throw new Error(`Invalid JSONL at line ${index + 1}: ${errorMessage(error)}`);
    }
  }

  return { iterations, config };
}

export function extractContextQuestion(iterations: RLMIteration[]): string {
  if (iterations.length === 0) return 'No context found';

  const firstIteration = iterations[0];
  const prompt = firstIteration.prompt;

  // Look for user message that contains the actual question
  for (const msg of prompt) {
    if (msg.role === 'user' && msg.content) {
      // Try to extract quoted query
      const queryMatch = msg.content.match(/original query: "([^"]+)"/);
      if (queryMatch) {
        return queryMatch[1];
      }

      // Check if it contains the actual query pattern
      if (msg.content.includes('answer the prompt')) {
        continue;
      }

      // Take first substantial user message
      if (msg.content.length > 50 && msg.content.length < 500) {
        return msg.content.slice(0, 200) + (msg.content.length > 200 ? '...' : '');
      }
    }
  }

  // Fallback: look in system prompt for context info
  const systemMsg = prompt.find(m => m.role === 'system');
  if (systemMsg?.content) {
    const contextMatch = systemMsg.content.match(/context variable.*?:(.*?)(?:\n|$)/i);
    if (contextMatch) {
      return contextMatch[1].trim().slice(0, 200);
    }
  }

  // Check code block output for actual context
  for (const iter of iterations) {
    for (const block of iter.code_blocks) {
      const ctx = block.result.locals.context;
      if (typeof ctx === 'string' && ctx.length < 500) {
        return ctx;
      }
    }
  }

  return 'Context available in REPL environment';
}

export function computeMetadata(iterations: RLMIteration[]): LogMetadata {
  let totalCodeBlocks = 0;
  let totalSubLMCalls = 0;
  let totalExecutionTime = 0;
  let hasErrors = false;
  let final: FinalValue = { ...ABSENT_FINAL };

  for (const iter of iterations) {
    totalCodeBlocks += iter.code_blocks.length;

    // Use iteration_time if available, otherwise sum code block times
    if (iter.iteration_time !== null) {
      totalExecutionTime += iter.iteration_time;
    } else {
      for (const block of iter.code_blocks) {
        totalExecutionTime += block.result.execution_time ?? 0;
      }
    }

    for (const block of iter.code_blocks) {
      if (block.result.stderr) {
        hasErrors = true;
      }
      totalSubLMCalls += block.result.rlm_calls.length;
    }

    if (iter.final.has_final) {
      final = iter.final;
    }
  }

  return {
    totalIterations: iterations.length,
    totalCodeBlocks,
    totalSubLMCalls,
    contextQuestion: extractContextQuestion(iterations),
    final,
    totalExecutionTime,
    hasErrors,
  };
}

export function parseLogFile(fileName: string, content: string): RLMLogFile {
  const { iterations, config } = parseJSONL(content);
  const metadata = computeMetadata(iterations);

  return {
    fileName,
    filePath: fileName,
    iterations,
    metadata,
    config,
  };
}
