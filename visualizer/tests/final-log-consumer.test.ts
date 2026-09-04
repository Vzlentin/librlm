import assert from 'node:assert/strict';
import test from 'node:test';

import { parseLogFile } from '../src/lib/parse-logs.ts';
import { formatFinal, type JSONValue } from '../src/lib/types.ts';

const cases: Array<[string, JSONValue, string]> = [
  ['string', 'answer', 'answer'],
  ['object', { answer: 42 }, '{\n  "answer": 42\n}'],
  ['array', [1, 'two'], '[\n  1,\n  "two"\n]'],
  ['scalar', 7, '7'],
  ['explicit null', null, 'null'],
];

for (const [name, value, display] of cases) {
  test(`consumes a canonical ${name} final`, () => {
    const content = JSON.stringify({
      type: 'iteration',
      iteration: 1,
      timestamp: '2026-01-01T00:00:00Z',
      prompt: [],
      response: '',
      code_blocks: [],
      final: { has_final: true, final_value: value },
      iteration_time: 0,
    });
    const parsed = parseLogFile('test.jsonl', content);

    assert.equal(parsed.metadata.final.has_final, true);
    assert.deepEqual(parsed.metadata.final.final_value, value);
    assert.equal(formatFinal(parsed.metadata.final), display);
  });
}

test('normalizes the supported legacy tuple form only at the parser seam', () => {
  const content = JSON.stringify({
    type: 'iteration',
    iteration: 1,
    timestamp: '2026-01-01T00:00:00Z',
    prompt: [],
    response: '',
    code_blocks: [],
    final_answer: ['legacy-label', 'legacy answer'],
    iteration_time: 0,
  });
  const parsed = parseLogFile('legacy.jsonl', content);

  assert.deepEqual(parsed.metadata.final, {
    has_final: true,
    final_value: 'legacy answer',
  });
});
