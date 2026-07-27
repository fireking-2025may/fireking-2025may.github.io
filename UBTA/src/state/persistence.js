import { normaliseDocument, validateDocument, newId } from './schema.js';

export const STORAGE_VERSION = 1;
export const URL_FORMAT_VERSION = 1;
export const DEFAULT_URL_LIMIT = 32000;
const prefix = 'ubta:editor';

const utf8ToBase64 = value => {
  const bytes = new TextEncoder().encode(value);
  let binary = '';
  for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
};
const base64ToUtf8 = value => {
  const padded = value.replace(/-/g, '+').replace(/_/g, '/') + '='.repeat((4 - value.length % 4) % 4);
  const binary = atob(padded), bytes = Uint8Array.from(binary, character => character.charCodeAt(0));
  return new TextDecoder().decode(bytes);
};
const parseEnvelope = raw => {
  const envelope = JSON.parse(raw);
  if (envelope?.storageVersion !== STORAGE_VERSION || !envelope.documentId || !envelope.savedAt) throw Error('Unsupported persistence record');
  return { ...envelope, document: validateDocument(envelope.document) };
};

export function encodeDocument(document, { limit = DEFAULT_URL_LIMIT } = {}) {
  const payload = utf8ToBase64(JSON.stringify(validateDocument(document)));
  const fragment = `doc=v${URL_FORMAT_VERSION}.${payload}`;
  if (fragment.length > limit) throw Object.assign(Error('Document is too large for a reliable share URL'), { code: 'URL_TOO_LARGE', length: fragment.length });
  return fragment;
}

export function decodeDocument(fragment) {
  const match = String(fragment || '').replace(/^#/, '').match(/^doc=v(\d+)\.([A-Za-z0-9_-]+)$/);
  if (!match || Number(match[1]) !== URL_FORMAT_VERSION) throw Error('Unsupported or malformed document URL');
  return validateDocument(JSON.parse(base64ToUtf8(match[2])));
}

export class Persistence {
  constructor({ storage = globalThis.localStorage, clock = () => new Date(), id = () => newId('document') } = {}) {
    this.storage = storage; this.clock = clock; this.id = id;
  }
  keys(documentId) { return { draft: `${prefix}:${documentId}:draft:v1`, recovery: `${prefix}:${documentId}:recovery:v1`, temporary: `${prefix}:${documentId}:temporary:v1`, versions: `${prefix}:${documentId}:versions:v1` }; }
  index() { try { return JSON.parse(this.storage.getItem(`${prefix}:index:v1`)) || { storageVersion: STORAGE_VERSION, documents: {} }; } catch { return { storageVersion: STORAGE_VERSION, documents: {} }; } }
  writeIndex(index) { this.storage.setItem(`${prefix}:index:v1`, JSON.stringify(index)); }
  create(document) { return { documentId: this.id(), revision: 0, document: validateDocument(document) }; }
  saveDraft({ documentId, revision = 0, document }, reason = 'autosave') {
    const keys = this.keys(documentId), envelope = { storageVersion: STORAGE_VERSION, documentId, revision: revision + 1, savedAt: this.clock().toISOString(), reason, document: validateDocument(document) };
    const raw = JSON.stringify(envelope), previous = this.storage.getItem(keys.draft);
    this.storage.setItem(keys.temporary, raw);
    parseEnvelope(this.storage.getItem(keys.temporary));
    if (previous) this.storage.setItem(keys.recovery, previous);
    this.storage.setItem(keys.draft, raw); this.storage.removeItem(keys.temporary);
    const index = this.index(); index.lastDocumentId = documentId; index.documents ||= {}; index.documents[documentId] = { title: envelope.document.meta.projectTitle, modifiedAt: envelope.savedAt, status: envelope.document.meta.status }; this.writeIndex(index);
    return envelope;
  }
  load(documentId, kind = 'draft') { const raw = this.storage.getItem(this.keys(documentId)[kind]); return raw ? parseEnvelope(raw) : null; }
  loadLatest() { const id = this.index().lastDocumentId; if (!id) return null; try { return this.load(id); } catch { try { return { ...this.load(id, 'recovery'), reason: 'recovery' }; } catch { const version = this.versions(id)[0]; return version ? { storageVersion: STORAGE_VERSION, documentId: id, revision: 0, savedAt: version.createdAt, reason: 'version-recovery', document: version.document } : null; } } }
  versions(documentId) { try { return (JSON.parse(this.storage.getItem(this.keys(documentId).versions)) || []).map(version => ({ ...version, document: validateDocument(version.document) })); } catch { return []; } }
  saveVersion(record, { label, description = '' } = {}) {
    const versions = this.versions(record.documentId), version = { id: newId('version'), createdAt: this.clock().toISOString(), label: label || record.document.meta.version || `v${versions.length + 1}`, description, status: record.document.meta.status, document: validateDocument(record.document) };
    versions.unshift(version); this.storage.setItem(this.keys(record.documentId).versions, JSON.stringify(versions.slice(0, 20))); return version;
  }
  deleteVersion(documentId, versionId) { const versions = this.versions(documentId).filter(x => x.id !== versionId); this.storage.setItem(this.keys(documentId).versions, JSON.stringify(versions)); return versions; }
  startup(hash, fallback) {
    if (String(hash || '').startsWith('#doc=')) return { ...this.create(decodeDocument(hash)), source: 'url' };
    const latest = this.loadLatest(); return latest ? { ...latest, source: latest.reason === 'recovery' ? 'recovery' : 'draft' } : { ...this.create(normaliseDocument(fallback)), source: 'seed' };
  }
}
