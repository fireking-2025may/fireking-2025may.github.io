import test from 'node:test';
import assert from 'node:assert/strict';
import {
  SCHEMA_VERSION,
  normaliseDocument,
  normaliseBlock,
  safeHref,
  safeImageSource,
  seedDocument,
} from '../src/state/schema.js';

test('seed exercises the version two schema', () => {
  const normalised = normaliseDocument(seedDocument);
  assert.equal(normalised.schemaVersion, 2);
  assert.equal(normalised.sections.length, 2);
  assert.equal(normalised.steps.length, 2);
  assert.ok(normalised.sections.flatMap(section => section.blocks).some(block => block.type === 'table'));
  assert.ok(normalised.sections.flatMap(section => section.blocks).some(block => block.type === 'image'));
  assert.equal(normalised.derivedContent.transactionProposals.sectionId, 'transaction-proposals');
});

test('version one documents migrate before normalisation', () => {
  const migrated = normaliseDocument({schemaVersion: 1, steps: [{id: 'first', title: 'First'}]});
  assert.equal(migrated.schemaVersion, SCHEMA_VERSION);
  assert.equal(migrated.steps[0].summary, '');
  assert.equal(migrated.steps[0].proposal, null);
  assert.deepEqual(migrated.derivedContent.transactionProposals, {automatic: true, sectionId: null});
});

test('unknown formatting and blocks are safely reduced', () => {
  const block = normaliseBlock({type: 'script', font: 'Comic Sans', runs: [{text: 'safe', bold: true, style: 'color:red'}]});
  assert.deepEqual(block.runs, [{text: 'safe', highlight: false, link: null}]);
  assert.equal(block.type, 'paragraph');
  assert.equal('font' in block, false);
});

test('links permit only approved protocols', () => {
  assert.equal(safeHref('javascript:alert(1)'), null);
  assert.equal(safeHref('https://example.test'), 'https://example.test');
  assert.equal(safeHref('#anchor-ok'), '#anchor-ok');
});

test('list nesting is capped at three', () => {
  assert.equal(normaliseBlock({type: 'bulletList', items: [{level: 9, runs: [{text: 'x'}]}]}).items[0].level, 3);
});

test('table widths are bounded percentages with a sum of 100', () => {
  const table = normaliseBlock({type: 'table', columns: [{id: 'a', width: -5}, {id: 'b', width: 30}, {id: 'c', width: 200}], rows: []});
  assert.equal(table.columns.reduce((sum, column) => sum + column.width, 0), 100);
  assert.ok(table.columns.every(column => column.width > 0 && column.width <= 100));
  assert.deepEqual(table.totals, {enabled: false, label: '', columns: []});
});

test('images accept only matching allowlisted local data URLs', () => {
  const png = 'data:image/png;base64,iVBORw0KGgo=';
  assert.equal(safeImageSource(png), png);
  for (const source of ['https://example.test/a.png', 'data:image/svg+xml;base64,PHN2Zz4=', 'javascript:alert(1)']) assert.equal(safeImageSource(source), null);
  assert.equal(normaliseBlock({type: 'image', source: png, mimeType: 'image/jpeg'}).source, null);
});

test('document IDs are unique across anchor and lookup entities', () => {
  const document = normaliseDocument({
    schemaVersion: 2,
    sections: [{id: 'same', blocks: [{id: 'same', type: 'table', columns: [], headerRows: [{id: 'same', cells: [{id: 'same', runs: []}]}]}]}],
    steps: [{id: 'same', blocks: [{id: 'same', type: 'image'}]}],
  });
  const table = document.sections[0].blocks[0];
  const ids = [document.sections[0].id, table.id, table.headerRows[0].id, table.headerRows[0].cells[0].id, document.steps[0].id, document.steps[0].blocks[0].id];
  assert.equal(new Set(ids).size, ids.length);
});

test('unsupported future documents are rejected', () => {
  assert.throws(() => normaliseDocument({schemaVersion: 999}), /Unsupported schema version/);
});
