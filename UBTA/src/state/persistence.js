import { validateDocument } from './schema.js';

export const STORAGE_VERSION = 1;
export const STORAGE_KEYS = { draft:'ubta:draft', recovery:'ubta:recovery', versions:'ubta:versions' };
const VERSION_LIMIT = 20;
const parse = value => { try { return value ? JSON.parse(value) : null; } catch { return null; } };
const validRecord = record => record?.storageVersion === STORAGE_VERSION && Number.isInteger(record.revision) && record.revision >= 0 && record.document;
const checkedDocument = value => validateDocument(value);

export function readDraft(storage = localStorage) {
  const record = parse(storage.getItem(STORAGE_KEYS.draft));
  if (!validRecord(record)) return null;
  try { return { ...record, document:checkedDocument(record.document) }; } catch { return null; }
}

/** A save is compare-and-swap: another tab can never have its draft or recovery overwritten. */
export function saveDraft(document, expectedRevision, storage = localStorage, now = Date.now()) {
  const current = readDraft(storage), actualRevision = current?.revision ?? 0;
  if (actualRevision !== expectedRevision) return { ok:false, stale:true, revision:actualRevision };
  const clean = checkedDocument(document), next = { storageVersion:STORAGE_VERSION, revision:actualRevision + 1, savedAt:now, document:clean };
  // Do not rotate recovery until after the stale-tab check above.
  if (current) storage.setItem(STORAGE_KEYS.recovery, JSON.stringify(current));
  storage.setItem(STORAGE_KEYS.draft, JSON.stringify(next));
  return { ok:true, stale:false, revision:next.revision, record:next };
}

export function readRecovery(storage = localStorage) {
  const record = parse(storage.getItem(STORAGE_KEYS.recovery));
  if (!validRecord(record)) return null;
  try { return { ...record, document:checkedDocument(record.document) }; } catch { return null; }
}

export function listVersions(storage = localStorage) {
  const records = parse(storage.getItem(STORAGE_KEYS.versions));
  if (!Array.isArray(records)) return [];
  return records.filter(validRecord).flatMap(record => { try { return [{ ...record, document:checkedDocument(record.document) }]; } catch { return []; } }).slice(0, VERSION_LIMIT);
}

export function saveVersion(document, label, storage = localStorage, now = Date.now()) {
  const record = { storageVersion:STORAGE_VERSION, revision:0, id:`version-${now}-${Math.random().toString(36).slice(2)}`, savedAt:now, label:String(label || '').trim() || 'Saved version', document:checkedDocument(document) };
  const versions = [record, ...listVersions(storage)].slice(0, VERSION_LIMIT);
  storage.setItem(STORAGE_KEYS.versions, JSON.stringify(versions));
  return record;
}

export function deleteVersion(id, storage = localStorage) {
  storage.setItem(STORAGE_KEYS.versions, JSON.stringify(listVersions(storage).filter(version => version.id !== id)));
}

const toBase64 = text => { const bytes = new TextEncoder().encode(text); let binary=''; bytes.forEach(byte => binary += String.fromCharCode(byte)); return btoa(binary).replace(/\+/g,'-').replace(/\//g,'_').replace(/=+$/,''); };
const fromBase64 = value => { const base64=value.replace(/-/g,'+').replace(/_/g,'/'); const binary=atob(base64 + '='.repeat((4-base64.length%4)%4)); return new TextDecoder().decode(Uint8Array.from(binary, char => char.charCodeAt(0))); };
export function encodeSnapshot(document) { return toBase64(JSON.stringify({ snapshotVersion:1, document:checkedDocument(document) })); }
export function decodeSnapshot(value) {
  try { const envelope=JSON.parse(fromBase64(value)); if (envelope?.snapshotVersion !== 1) throw Error('Unsupported snapshot version'); return checkedDocument(envelope.document); }
  catch (error) { throw Error(`Invalid UBTA snapshot: ${error.message}`); }
}
export function snapshotFromLocation(location = globalThis.location) { const match=location.hash.match(/^#snapshot=([A-Za-z0-9_-]+)$/); return match ? decodeSnapshot(match[1]) : null; }
