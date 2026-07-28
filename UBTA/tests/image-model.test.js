import test from 'node:test';
import assert from 'node:assert/strict';
import { normalizeImageWidth, validateImageInput, validateImageSource } from '../src/state/image-model.js';

test('image sources report a specific validation result', () => {
  assert.deepEqual(validateImageSource(''), { ok: false, error: 'missing-source' });
  assert.deepEqual(validateImageSource('not a URL'), { ok: false, error: 'malformed-source' });
  assert.deepEqual(validateImageSource('http://example.test/a.png'), { ok: false, error: 'unsupported-protocol' });
  assert.deepEqual(validateImageSource('https://example.test/a.svg'), { ok: false, error: 'unsupported-extension' });
  assert.deepEqual(validateImageSource('https://example.test/a.png?version=2'), { ok: true, url: 'https://example.test/a.png?version=2' });
});

test('image input trims text and applies the canonical width bounds', () => {
  assert.deepEqual(validateImageInput({ source: 'https://example.test/a.jpg', altText: '  Receipt  ', width: 45 }), { ok: true, image: { source: 'https://example.test/a.jpg', altText: 'Receipt', width: 45 } });
  assert.deepEqual(validateImageInput({ source: 'https://example.test/a.jpg', altText: '  ' }), { ok: false, error: 'missing-alt-text' });
  assert.equal(normalizeImageWidth(5), 20);
  assert.equal(normalizeImageWidth(500), 100);
  assert.equal(normalizeImageWidth(''), 100);
});
