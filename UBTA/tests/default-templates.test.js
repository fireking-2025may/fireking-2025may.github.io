import assert from 'node:assert/strict';
import test from 'node:test';
import { webcrypto } from 'node:crypto';
import {
  decryptEnvelope,
  encryptEnvelope,
  validateEnvelope,
  DAMAGED_FILE_MESSAGE,
} from '../src/editor/crypto.js';
import {
  EncryptedTemplateLoader,
  generateOpaqueId,
  loadDefaultTemplates,
  migrateTemplatePayload,
  templateBlocks,
  validateTemplatePayload,
} from '../src/editor/default-templates.js';
import { applyPlainTextEdit } from '../../encrypted-files/rich-text.js';
import { insertionIndex } from '../src/editor/insertion-context.js';
import { History } from '../src/state/history.js';

let counter = 1;
const nextBytes = () => new Uint8Array(16).fill(counter++);
const id = () => generateOpaqueId(new Set(), nextBytes);
const valid = () => ({
  type: 'editor-defaults',
  schemaVersion: 2,
  templates: [
    {
      id: id(),
      header: 'Stamp duty',
      blocks: [
        {
          id: id(),
          type: 'paragraph',
          runs: [
            {
              text: 'Body',
              highlight: true,
              link: { href: 'https://example.com' },
            },
          ],
        },
      ],
    },
  ],
});

test('opaque IDs have the fixed alphabet and regenerate collisions', () => {
  const first = generateOpaqueId(new Set(), () => new Uint8Array(16));
  let calls = 0;
  const second = generateOpaqueId(new Set([first]), () =>
    new Uint8Array(16).fill(calls++),
  );
  assert.match(first, /^[1-9A-F]{16}$/);
  assert.equal(first.length, 16);
  assert.doesNotMatch(first, /[0a-z]/);
  assert.notEqual(second, first);
  assert.equal(calls, 2);
});
test('version 2 validates immutable rich records and rejects invalid IDs, links and duplicates', () => {
  const payload = valid(),
    result = validateTemplatePayload(payload);
  assert.equal(result[0].id, payload.templates[0].id);
  assert.equal(result[0].blocks[0].id, payload.templates[0].blocks[0].id);
  assert.equal(result[0].blocks[0].runs[0].highlight, true);
  assert.ok(Object.isFrozen(result[0].blocks[0]));
  for (const bad of [
    structuredClone(payload),
    structuredClone(payload),
    structuredClone(payload),
  ]) {
    if (bad === bad) {
      /* assigned below */
    }
  }
  const invalidId = structuredClone(payload);
  invalidId.templates[0].id = 'legacy-description';
  assert.throws(() => validateTemplatePayload(invalidId), /malformed/);
  const unsafe = structuredClone(payload);
  unsafe.templates[0].blocks[0].runs[0].link.href = 'javascript:alert(1)';
  assert.throws(() => validateTemplatePayload(unsafe), /malformed/);
  const duplicate = structuredClone(payload);
  duplicate.templates[0].blocks[0].id = duplicate.templates[0].id;
  assert.throws(() => validateTemplatePayload(duplicate), /malformed/);
});
test('normalization preserves numeric-leading opaque IDs at every nesting level', () => {
  const payload = valid();
  payload.templates[0].blocks = [
    {
      id: id(),
      type: 'table',
      captionRuns: [{ text: 'Caption' }],
      columns: [
        {
          id: id(),
          headingRuns: [{ text: 'Heading' }],
          width: 100,
          format: 'text',
          totalEnabled: false,
        },
      ],
      rows: [
        {
          id: id(),
          isTotal: false,
          cells: [{ id: id(), runs: [{ text: 'Value' }] }],
        },
      ],
    },
  ];
  const expectedIds = [
    payload.templates[0].blocks[0].id,
    payload.templates[0].blocks[0].columns[0].id,
    payload.templates[0].blocks[0].rows[0].id,
    payload.templates[0].blocks[0].rows[0].cells[0].id,
  ];

  const normalized = validateTemplatePayload(payload)[0].blocks[0];
  assert.deepEqual(
    [
      normalized.id,
      normalized.columns[0].id,
      normalized.rows[0].id,
      normalized.rows[0].cells[0].id,
    ],
    expectedIds,
  );
  validateTemplatePayload({
    type: 'editor-defaults',
    schemaVersion: 2,
    templates: [{ ...payload.templates[0], blocks: [normalized] }],
  });
});
test('plain-text edits retain links and review highlights on rich runs', () => {
  const original = [
    { text: 'Linked', link: { href: 'https://example.com' } },
    { text: ' review', highlight: true },
  ];

  const replaced = applyPlainTextEdit(original, 'LinkXd review');
  const edited = applyPlainTextEdit(replaced, 'LinkXd review!');

  assert.deepEqual(edited, [
    { text: 'LinkXd', link: { href: 'https://example.com' } },
    { text: ' review!', highlight: true },
  ]);
  assert.notEqual(edited[0].link, original[0].link);
});
test('legacy migration preserves Unicode, order and content while replacing descriptive IDs once', () => {
  counter = 1;
  const legacy = {
    type: 'editor-defaults',
    templates: [
      { id: 'first', header: 'Café', body: 'Body 🚀\nline' },
      { id: 'second', header: 'Two', body: '' },
    ],
  };
  const migrated = migrateTemplatePayload(legacy, { randomBytes: nextBytes });
  assert.equal(migrated.summary.templateCount, 2);
  assert.equal(migrated.summary.contentChanged, false);
  assert.deepEqual(
    migrated.payload.templates.map((x) => x.header),
    ['Café', 'Two'],
  );
  assert.equal(
    migrated.payload.templates[0].blocks[0].runs[0].text,
    'Body 🚀\nline',
  );
  validateTemplatePayload(migrated.payload);
});
test('loader accepts legacy during transition and caches only success', async () => {
  let calls = 0;
  const legacy = {
    type: 'editor-defaults',
    templates: [{ id: 'old', header: 'Header', body: 'Body' }],
  };
  const loader = new EncryptedTemplateLoader({
    decrypt: async (_e, password) => {
      calls++;
      if (password !== 'right') throw Error(DAMAGED_FILE_MESSAGE);
      return legacy;
    },
  });
  await assert.rejects(loader.unlock({}, 'wrong'), /Incorrect password/);
  assert.equal(loader.cached, false);
  assert.equal((await loader.unlock({}, 'right'))[0].header, 'Header');
  await loader.unlock({}, 'ignored');
  assert.equal(calls, 2);
});
test('loader decrypts an encrypted database and normalizes document blocks', async () => {
  const payload = valid();
  const envelope = await encryptEnvelope(payload, 'correct horse', {
    subtle: webcrypto.subtle,
    randomBytes: (length) => webcrypto.getRandomValues(new Uint8Array(length)),
  });
  const loader = new EncryptedTemplateLoader({
    decrypt: (value, password) =>
      decryptEnvelope(value, password, webcrypto.subtle),
  });

  const templates = await loadDefaultTemplates({
    loader,
    envelope,
    password: 'correct horse',
  });

  assert.equal(templates[0].header, 'Stamp duty');
  assert.deepEqual(templates[0].blocks[0].runs[0], {
    text: 'Body',
    highlight: true,
    link: { href: 'https://example.com' },
  });
  assert.ok(Object.isFrozen(templates[0].blocks[0]));
});
test('offline loading distinguishes missing data, invalid credentials, and malformed plaintext', async () => {
  const encrypted = await encryptEnvelope(valid(), 'right', {
    subtle: webcrypto.subtle,
    randomBytes: (length) => webcrypto.getRandomValues(new Uint8Array(length)),
  });
  const loader = () =>
    new EncryptedTemplateLoader({
      decrypt: (value, password) =>
        decryptEnvelope(value, password, webcrypto.subtle),
    });

  await assert.rejects(
    loadDefaultTemplates({ loader: loader(), envelope: null, password: 'right' }),
    /unavailable offline/,
  );
  await assert.rejects(
    loadDefaultTemplates({ loader: loader(), envelope: encrypted, password: 'wrong' }),
    /Incorrect password or damaged file/,
  );
  const damaged = { ...encrypted, ciphertext: `${encrypted.ciphertext.slice(0, -4)}AAAA` };
  await assert.rejects(
    loadDefaultTemplates({ loader: loader(), envelope: damaged, password: 'right' }),
    /Incorrect password or damaged file/,
  );
  const malformedLoader = new EncryptedTemplateLoader({
    decrypt: async () => ({ type: 'editor-defaults', schemaVersion: 2, templates: [] }),
  });
  await assert.rejects(
    loadDefaultTemplates({ loader: malformedLoader, envelope: encrypted, password: 'right' }),
    /decrypted template database is malformed/,
  );
  assert.equal(malformedLoader.cached, false);
});
test('insertion deep-clones all blocks with fresh nested IDs and is one history action', () => {
  const template = valid().templates[0];
  template.blocks.push({
    id: id(),
    type: 'bulletList',
    items: [{ id: id(), level: 1, runs: [{ text: 'Item' }] }],
  });
  let n = 0;
  const blocks = templateBlocks(template, (prefix) => `${prefix}-${++n}`);
  assert.equal(blocks[0].runs[0].text, template.header);
  assert.equal(blocks.length, 3);
  assert.notEqual(blocks[1].id, template.blocks[0].id);
  assert.notEqual(blocks[2].items[0].id, template.blocks[1].items[0].id);
  const group = { id: 'g', blocks: [{ id: 'a' }] },
    history = new History(group),
    at = insertionIndex(group, {
      groupId: 'g',
      referenceBlockId: 'a',
      position: 'after',
    });
  group.blocks.splice(at, 0, ...blocks);
  history.commit(group);
  assert.equal(history.undo().blocks.length, 1);
});
test('encryption round-trips and fresh randomness changes salt, IV and ciphertext', async () => {
  const payload = valid(),
    options = {
      subtle: webcrypto.subtle,
      randomBytes: (n) => webcrypto.getRandomValues(new Uint8Array(n)),
    };
  const a = await encryptEnvelope(payload, 'password', options),
    b = await encryptEnvelope(payload, 'password', options);
  assert.notEqual(a.salt, b.salt);
  assert.notEqual(a.iv, b.iv);
  assert.notEqual(a.ciphertext, b.ciphertext);
  assert.deepEqual(
    await decryptEnvelope(a, 'password', webcrypto.subtle),
    payload,
  );
  await assert.rejects(
    decryptEnvelope(a, 'incorrect', webcrypto.subtle),
    /Incorrect password/,
  );
  validateEnvelope(a);
  assert.throws(
    () => validateEnvelope({ ...a, iterations: 100000 }),
    /Incorrect password or damaged file/,
  );
});
