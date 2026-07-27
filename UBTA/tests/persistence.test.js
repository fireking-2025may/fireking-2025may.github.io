import test from 'node:test';
import assert from 'node:assert/strict';
import { Persistence, decodeDocument, encodeDocument } from '../src/state/persistence.js';
import { seedDocument } from '../src/state/schema.js';

class MemoryStorage {
  constructor() { this.values = new Map(); }
  getItem(key) { return this.values.get(key) ?? null; }
  setItem(key, value) { this.values.set(key, String(value)); }
  removeItem(key) { this.values.delete(key); }
}
const factory = storage => new Persistence({ storage, clock: () => new Date('2026-07-27T12:00:00Z'), id: () => 'document-test' });

test('share URLs round-trip unicode through validation',()=>{const document=structuredClone(seedDocument);document.meta.projectTitle='Café restructure';assert.equal(decodeDocument(`#${encodeDocument(document)}`).meta.projectTitle,'Café restructure')});
test('share URL decoder rejects malformed and unsupported records',()=>{assert.throws(()=>decodeDocument('#doc=v2.abc'),/Unsupported/);assert.throws(()=>decodeDocument('#nope'),/Unsupported/);assert.throws(()=>encodeDocument(seedDocument,{limit:10}),error=>error.code==='URL_TOO_LARGE')});
test('draft writes rotate the prior valid value into recovery',()=>{const storage=new MemoryStorage(),p=factory(storage),record=p.create(seedDocument),first=p.saveDraft(record),second=p.saveDraft({...first,document:{...first.document,meta:{...first.document.meta,projectTitle:'Changed'}}});assert.equal(second.revision,2);assert.equal(p.load(record.documentId,'recovery').revision,1);assert.equal(p.loadLatest().document.meta.projectTitle,'Changed')});
test('corrupt current draft falls back to recovery',()=>{const storage=new MemoryStorage(),p=factory(storage),record=p.create(seedDocument),first=p.saveDraft(record);p.saveDraft({...first,document:{...first.document,meta:{...first.document.meta,projectTitle:'Changed'}}});storage.setItem(p.keys(record.documentId).draft,'bad json');assert.equal(p.loadLatest().revision,1)});
test('saved versions are immutable clones, newest first, and deletable',()=>{const storage=new MemoryStorage(),p=factory(storage),record=p.create(seedDocument),version=p.saveVersion(record,{label:'v1'});record.document.meta.projectTitle='Mutated';assert.notEqual(p.versions(record.documentId)[0].document.meta.projectTitle,'Mutated');p.deleteVersion(record.documentId,version.id);assert.equal(p.versions(record.documentId).length,0)});
test('URL startup creates an independent local identity',()=>{const p=factory(new MemoryStorage()),opened=p.startup(`#${encodeDocument(seedDocument)}`,seedDocument);assert.equal(opened.source,'url');assert.equal(opened.documentId,'document-test')});
test('failed storage writes do not replace a previous valid draft',()=>{const storage=new MemoryStorage(),p=factory(storage),record=p.create(seedDocument);p.saveDraft(record);const before=storage.getItem(p.keys(record.documentId).draft),original=storage.setItem.bind(storage);storage.setItem=(key,value)=>{if(key.endsWith(':temporary:v1'))throw Error('quota');original(key,value)};assert.throws(()=>p.saveDraft({...record,document:{...record.document,meta:{...record.document.meta,projectTitle:'Lost'}}}),/quota/);assert.equal(storage.getItem(p.keys(record.documentId).draft),before)});
