import test from 'node:test';
import assert from 'node:assert/strict';
import { normaliseDocument, normaliseBlock, safeHref, safeImageSrc, seedDocument, transactionProposals, validateDocument } from '../src/state/schema.js';

test('seed satisfies the closed version-two schema', () => {
  const document = validateDocument(seedDocument);
  assert.equal(document.schemaVersion, 2);
  assert.equal(document.sections.length, 1);
  assert.equal(document.steps.length, 1);
});

test('unknown formatting and blocks are safely reduced', () => {
  const block = normaliseBlock({ type: 'script', font: 'Comic Sans', runs: [{ text: 'safe', bold: true, style: 'color:red' }] });
  assert.deepEqual(block.runs, [{ text: 'safe', highlight: false, link: null }]);
  assert.equal(block.type, 'paragraph');
  assert.equal('font' in block, false);
});

test('links permit only approved protocols', () => {
  assert.equal(safeHref('javascript:alert(1)'), null);
  assert.equal(safeHref('https://example.test'), 'https://example.test');
  assert.equal(safeHref('#anchor-ok'), '#anchor-ok');
});

test('list nesting is capped at three', () => assert.equal(normaliseBlock({ type: 'bulletList', items: [{ level: 9, runs: [{ text: 'x' }] }] }).items[0].level, 3));

test('tables have stable cells, bounded normalised widths and total rows', () => {
  const table = normaliseBlock({ id: 'fees', type: 'table', columns: [{ id: 'name', heading: 'Name', width: 3 }, { id: 'fee', heading: 'Fee', width: 7, numeric: true }], rows: [{ id: 'total', isTotal: true, cells: [{ runs: [{ text: 'Total' }] }, { runs: [{ text: '20' }] }] }] });
  assert.equal(table.columns.reduce((sum, column) => sum + column.width, 0), 100);
  assert.equal(table.rows[0].isTotal, true);
  assert.equal(table.rows[0].cells.length, 2);
  assert.match(table.rows[0].cells[0].id, /^[A-Za-z][\w-]*$/);
});

test('images accept supported data images and HTTPS only', () => {
  assert.equal(safeImageSrc('javascript:alert(1)'), null);
  assert.equal(safeImageSrc('http://example.test/a.png'), null);
  assert.equal(safeImageSrc('data:image/svg+xml,<svg/>'), null);
  assert.equal(safeImageSrc('data:image/png;base64,YQ=='), 'data:image/png;base64,YQ==');
  assert.equal(safeImageSrc('https://example.test/a.png'), 'https://example.test/a.png');
  assert.throws(() => validateDocument({ schemaVersion: 2, steps: [{ blocks: [{ type: 'image', src: 'https://example.test/a.png', alt: '' }] }] }), /alternative text/);
});

test('transaction proposals are derived from stable step IDs and anchors', () => {
  const proposals = transactionProposals(seedDocument);
  assert.deepEqual(proposals.map(({ id, stepId, anchor }) => ({ id, stepId, anchor })), [{ id: 'proposal-share-restructure', stepId: 'share-restructure', anchor: 'anchor-share-restructure' }]);
  assert.match(proposals[0].summary, /^The directors/);
});

test('version one documents migrate without losing their stable IDs', () => {
  const document = normaliseDocument({ schemaVersion: 1, steps: [{ id: 'existing-step', title: 'Existing', blocks: [] }] });
  assert.equal(document.schemaVersion, 2);
  assert.equal(document.steps[0].id, 'existing-step');
});
