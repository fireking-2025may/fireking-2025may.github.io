import { decryptEnvelope } from './crypto.js';
export function validateTemplatePayload(payload) {
  if (
    !payload ||
    payload.type !== 'editor-defaults' ||
    !Array.isArray(payload.templates) ||
    !payload.templates.length
  )
    throw Error('The decrypted template database is malformed.');
  const ids = new Set();
  return payload.templates.map((t) => {
    if (
      !t ||
      typeof t.id !== 'string' ||
      !t.id.trim() ||
      ids.has(t.id) ||
      typeof t.header !== 'string' ||
      !t.header.trim() ||
      typeof t.body !== 'string' ||
      !t.body.trim()
    )
      throw Error('The decrypted template database is malformed.');
    ids.add(t.id);
    return Object.freeze({ id: t.id, header: t.header.trim(), body: t.body });
  });
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
    const templates = validateTemplatePayload(
      await this.#decrypt(envelope, password),
    );
    this.#templates = Object.freeze(templates);
    return this.#templates;
  }
}
export const templateBlocks = (template, newId) => [
  {
    id: newId('block'),
    type: 'heading',
    level: 2,
    runs: [{ text: template.header }],
  },
  { id: newId('block'), type: 'paragraph', runs: [{ text: template.body }] },
];
