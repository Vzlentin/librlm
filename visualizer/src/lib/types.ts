// Types matching the canonical RLM log format.

export type JSONValue = string | number | boolean | null | JSONValue[] | { [key: string]: JSONValue };

export interface FinalValue {
  has_final: boolean;
  final_value: JSONValue;
}

export const ABSENT_FINAL: FinalValue = { has_final: false, final_value: null };

export interface RLMChatCompletion {
  prompt: string | Record<string, unknown>;
  response: string;
  prompt_tokens: number;
  completion_tokens: number;
  execution_time: number;
  final: FinalValue;
}

export interface REPLResult {
  stdout: string;
  stderr: string;
  locals: Record<string, unknown>;
  execution_time: number | null;
  rlm_calls: RLMChatCompletion[];
  final: FinalValue;
}

export interface CodeBlock {
  code: string;
  result: REPLResult;
}

export interface RLMIteration {
  type?: string;
  iteration: number;
  timestamp: string;
  prompt: Array<{ role: string; content: string }>;
  response: string;
  code_blocks: CodeBlock[];
  final: FinalValue;
  iteration_time: number | null;
}

// Metadata saved at the start of a log file about RLM configuration.
export interface RLMConfigMetadata {
  root_model: string | null;
  max_depth: number | null;
  max_iterations: number | null;
  backend: string | null;
  backend_kwargs: Record<string, unknown> | null;
  environment_type: string | null;
  environment_kwargs: Record<string, unknown> | null;
  other_backends: string[] | null;
}

export interface RLMLogFile {
  fileName: string;
  filePath: string;
  iterations: RLMIteration[];
  metadata: LogMetadata;
  config: RLMConfigMetadata;
}

export interface LogMetadata {
  totalIterations: number;
  totalCodeBlocks: number;
  totalSubLMCalls: number;
  contextQuestion: string;
  final: FinalValue;
  totalExecutionTime: number;
  hasErrors: boolean;
}

/** Render structured values as JSON while leaving string answers readable. */
export function formatFinal(final: FinalValue): string | null {
  if (!final.has_final) return null;
  if (typeof final.final_value === 'string') return final.final_value;
  return JSON.stringify(final.final_value, null, 2);
}
