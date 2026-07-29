import { decryptEnvelope } from './crypto.js';
import { normaliseBlock, safeHref, safeImageSrc } from '../state/schema.js';

export const TEMPLATE_SCHEMA_VERSION = 2;
export const OPAQUE_ID_LENGTH = 16;
export const OPAQUE_ID_ALPHABET = '123456789ABCDEF';
export const OPAQUE_ID_PATTERN = /^[1-9A-F]{16}$/;
const malformed = () => Error('The decrypted template database is malformed.');

/** Generate a 16-character, 60-bit opaque ID from cryptographic randomness. */
export function generateOpaqueId(
  existing = new Set(),
  randomBytes = (n) => {
    const bytes = new Uint8Array(n);
    globalThis.crypto.getRandomValues(bytes);
    return bytes;
  },
) {
  for (;;) {
    const bytes = randomBytes(OPAQUE_ID_LENGTH);
    if (!(bytes instanceof Uint8Array) || bytes.length !== OPAQUE_ID_LENGTH)
      throw Error('The random byte source returned invalid data.');
    const id = [...bytes].map((byte) => OPAQUE_ID_ALPHABET[byte % 15]).join('');
    if (!existing.has(id)) return id;
  }
}

const assertRuns = (runs) => {
  if (!Array.isArray(runs)) throw malformed();
  for (const run of runs) {
    if (!run || typeof run.text !== 'string') throw malformed();
    if (run.highlight != null && typeof run.highlight !== 'boolean')
      throw malformed();
    if (
      run.link != null &&
      (!run.link || safeHref(run.link.href) !== run.link.href)
    )
      throw malformed();
    if (
      Object.keys(run).some(
        (key) => !['text', 'highlight', 'link'].includes(key),
      )
    )
      throw malformed();
  }
};
const takeId = (value, ids) => {
  if (!OPAQUE_ID_PATTERN.test(value || '') || ids.has(value)) throw malformed();
  ids.add(value);
};
function assertBlock(block, ids) {
  if (
    !block ||
    ![
      'paragraph',
      'heading',
      'bulletList',
      'numberList',
      'table',
      'image',
    ].includes(block.type)
  )
    throw malformed();
  takeId(block.id, ids);
  if (block.type === 'paragraph') assertRuns(block.runs);
  else if (block.type === 'heading') {
    if (![2, 3, 4].includes(block.level)) throw malformed();
    assertRuns(block.runs);
  } else if (block.type.endsWith('List')) {
    if (!Array.isArray(block.items) || !block.items.length) throw malformed();
    let previous = 1;
    for (const item of block.items) {
      takeId(item?.id, ids);
      if (
        !Number.isInteger(item.level) ||
        item.level < 1 ||
        item.level > 3 ||
        item.level > previous + 1
      )
        throw malformed();
      previous = item.level;
      assertRuns(item.runs);
    }
  } else if (block.type === 'table') {
    assertRuns(block.captionRuns);
    if (
      !Array.isArray(block.columns) ||
      !block.columns.length ||
      block.columns.length > 8 ||
      !Array.isArray(block.rows)
    )
      throw malformed();
    for (const column of block.columns) {
      takeId(column?.id, ids);
      assertRuns(column.headingRuns);
      if (
        !['text', 'number', 'gbp', 'percentage'].includes(column.format) ||
        typeof column.width !== 'number' ||
        typeof column.totalEnabled !== 'boolean'
      )
        throw malformed();
    }
    for (const row of block.rows) {
      takeId(row?.id, ids);
      if (
        !Array.isArray(row.cells) ||
        row.cells.length !== block.columns.length
      )
        throw malformed();
      for (const cell of row.cells) {
        takeId(cell?.id, ids);
        assertRuns(cell.runs);
      }
    }
  } else {
    if (
      safeImageSrc(block.src) !== block.src ||
      typeof block.alt !== 'string' ||
      !block.alt.trim()
    )
      throw malformed();
    assertRuns(block.captionRuns);
  }
  // Shared normalisation is the canonical closed schema; unknown keys are rejected.
  const normalized = normaliseBlock(block);
  if (normalized.type !== block.type) throw malformed();
}

function normaliseTemplateBlock(block) {
  const normalized = normaliseBlock(block);
  normalized.id = block.id;

  if (block.items) {
    normalized.items.forEach((item, index) => {
      item.id = block.items[index].id;
    });
  }
  if (block.columns) {
    normalized.columns.forEach((column, index) => {
      column.id = block.columns[index].id;
    });
  }
  if (block.rows) {
    normalized.rows.forEach((row, rowIndex) => {
      row.id = block.rows[rowIndex].id;
      row.cells.forEach((cell, cellIndex) => {
        cell.id = block.rows[rowIndex].cells[cellIndex].id;
      });
    });
  }

  return normalized;
}
const immutable = (value) => {
  if (value && typeof value === 'object') {
    Object.values(value).forEach(immutable);
    Object.freeze(value);
  }
  return value;
};

export function validateTemplatePayload(payload) {
  if (
    !payload ||
    payload.type !== 'editor-defaults' ||
    payload.schemaVersion !== 2 ||
    !Array.isArray(payload.templates) ||
    !payload.templates.length
  )
    throw malformed();
  const ids = new Set();
  const templates = payload.templates.map((template) => {
    takeId(template?.id, ids);
    if (
      typeof template.header !== 'string' ||
      !template.header.trim() ||
      !Array.isArray(template.blocks) ||
      !template.blocks.length
    )
      throw malformed();
    template.blocks.forEach((block) => assertBlock(block, ids));
    return {
      id: template.id,
      header: template.header.trim(),
      blocks: template.blocks.map(normaliseTemplateBlock),
    };
  });
  return immutable(templates);
}

export function migrateTemplatePayload(payload, { randomBytes } = {}) {
  if (
    !payload ||
    payload.type !== 'editor-defaults' ||
    payload.schemaVersion != null ||
    !Array.isArray(payload.templates) ||
    !payload.templates.length
  )
    throw malformed();
  const ids = new Set();
  const id = () => {
    const value = generateOpaqueId(ids, randomBytes);
    ids.add(value);
    return value;
  };
  const templates = payload.templates.map((template) => {
    if (
      typeof template?.header !== 'string' ||
      !template.header.trim() ||
      typeof template.body !== 'string'
    )
      throw malformed();
    return {
      id: id(),
      header: template.header,
      blocks: [
        { id: id(), type: 'paragraph', runs: [{ text: template.body }] },
      ],
    };
  });
  return {
    payload: { type: 'editor-defaults', schemaVersion: 2, templates },
    summary: {
      templateCount: templates.length,
      idsChanged: true,
      contentChanged: false,
    },
  };
}

export class EncryptedTemplateLoader {
  #templates = null;
  #decrypt;
  constructor({ decrypt = decryptEnvelope } = {}) {
    this.#decrypt = decrypt;
  }
  get cached() {
    return this.#templates !== null;
  }
  async unlock(envelope, password) {
    if (this.#templates) return this.#templates;
    let payload = await this.#decrypt(envelope, password);
    if (payload?.schemaVersion == null)
      payload = migrateTemplatePayload(payload).payload;
    this.#templates = validateTemplatePayload(payload);
    return this.#templates;
  }
}

export async function loadDefaultTemplates({ loader, envelope, password }) {
  if (!envelope) throw Error('Default templates are unavailable offline.');
  return loader.unlock(envelope, password);
}

const freshenBlockIds = (block, newId) => {
  const copy = structuredClone(block);
  copy.id = newId('block');
  for (const item of copy.items || []) item.id = newId('item');
  for (const column of copy.columns || []) column.id = newId('column');
  for (const row of copy.rows || []) {
    row.id = newId('row');
    for (const cell of row.cells || []) cell.id = newId('cell');
  }
  return copy;
};
export const templateBlocks = (template, newId) => [
  {
    id: newId('block'),
    type: 'heading',
    level: 2,
    runs: [{ text: template.header }],
  },
  ...template.blocks.map((block) => freshenBlockIds(block, newId)),
];
