import assert from 'node:assert/strict';
import test from 'node:test';

import { parseJSONL } from '../src/lib/parse-logs.ts';

function validResult() {
  return {
    stdout: '',
    stderr: '',
    locals: {},
    execution_time: 0,
    rlm_calls: [],
  };
}

function validIteration(overrides: Record<string, unknown> = {}) {
  return {
    type: 'iteration',
    iteration: 1,
    timestamp: '2026-01-01T00:00:00Z',
    prompt: [{ role: 'user', content: 'What is the answer?' }],
    response: 'Working on it.',
    code_blocks: [{ code: 'print(42)', result: validResult() }],
    final: { has_final: false, final_value: null },
    iteration_time: 0.5,
    ...overrides,
  };
}

test('rejects malformed JSON with its source line number', () => {
  const content = [
    JSON.stringify(validIteration()),
    '{"type":"iteration",',
  ].join('\n');

  assert.throws(
    () => parseJSONL(content),
    /Invalid JSONL at line 2: malformed JSON/,
  );
});

const malformedIterations: Array<[string, Record<string, unknown>, RegExp]> = [
  [
    'non-array code_blocks',
    validIteration({ code_blocks: {} }),
    /iteration\.code_blocks must be an array/,
  ],
  [
    'missing code block result',
    validIteration({ code_blocks: [{ code: 'print(42)' }] }),
    /iteration\.code_blocks\[0\]\.result is required/,
  ],
  [
    'malformed result calls',
    validIteration({
      code_blocks: [{ code: 'print(42)', result: { ...validResult(), rlm_calls: {} } }],
    }),
    /iteration\.code_blocks\[0\]\.result\.rlm_calls must be an array/,
  ],
  [
    'malformed prompt message',
    validIteration({ prompt: [{ role: 'user', content: 42 }] }),
    /iteration\.prompt\[0\]\.content must be a string/,
  ],
  [
    'malformed final metadata',
    validIteration({ final: { has_final: 'yes', final_value: 'answer' } }),
    /iteration\.final\.has_final must be a boolean/,
  ],
];

for (const [name, iteration, expected] of malformedIterations) {
  test(`rejects ${name} at the parser seam`, () => {
    assert.throws(
      () => parseJSONL(JSON.stringify(iteration)),
      error => {
        assert.match(String(error), /Invalid JSONL at line 1/);
        assert.match(String(error), expected);
        return true;
      },
    );
  });
}

test('does not return a partial trajectory when a later line is invalid', () => {
  const content = [
    JSON.stringify(validIteration()),
    JSON.stringify(validIteration({ response: 42, iteration: 2 })),
    JSON.stringify(validIteration({ iteration: 3 })),
  ].join('\n');
  let parsed: ReturnType<typeof parseJSONL> | undefined;

  assert.throws(
    () => {
      parsed = parseJSONL(content);
    },
    /Invalid JSONL at line 2: iteration\.response must be a string/,
  );
  assert.equal(parsed, undefined);
});
