import assert from 'node:assert/strict';
import test from 'node:test';
import {
  addStep, moveStep, deleteStep, addAppendix, moveAppendix, deleteAppendix,
  insertBlock, moveBlock, deleteBlock, insertTableRow, moveTableRow, deleteTableRow,
  insertTableColumn, moveTableColumn, deleteTableColumn, setTableColumnFormat, convertBlockStyle,
} from '../src/state/document-operations.js';

const ids = prefix => `${prefix}-new`;
const fixture = () => ({
  schemaVersion: 5,
  meta: {}, sections: [],
  steps: [
    { id: 's1', title: 'One', blocks: [{ id: 'p1', type: 'paragraph', runs: [{ text: 'One' }] }, { id: 'p2', type: 'paragraph', runs: [] }] },
    { id: 's2', title: 'Two', blocks: [{ id: 'table', type: 'table', captionRuns: [], columns: [{ id: 'c1', headingRuns: [], width: 50, format: 'text' }, { id: 'c2', headingRuns: [], width: 50, format: 'number' }], rows: [{ id: 'r1', cells: [{ id: 'a', runs: [] }, { id: 'b', runs: [{ text: '2' }] }] }, { id: 'r2', cells: [{ id: 'c', runs: [] }, { id: 'd', runs: [{ text: '3' }] }] }, { id: 'total', isTotal: true, cells: [{ id: 'e', runs: [] }, { id: 'f', runs: [] }] }] }] },
  ],
  appendices: [{ id: 'a1', title: 'One', blocks: [] }, { id: 'a2', title: 'Two', blocks: [] }],
});

test('step operations succeed and preserve their input', () => {
  const source = fixture(), added = addStep(source, { referenceId: 's1', position: 'after', idFactory: ids });
  assert.equal(added.changed, true); assert.equal(added.document.steps[1].id, 'step-new'); assert.equal(source.steps.length, 2);
  const moved = moveStep(source, 's2', -1); assert.deepEqual(moved.document.steps.map(x => x.id), ['s2', 's1']);
  const removed = deleteStep(source, 's1'); assert.deepEqual(removed.document.steps.map(x => x.id), ['s2']);
});

test('step operations report boundaries and missing IDs', () => {
  const source = fixture();
  assert.equal(moveStep(source, 's1', -1).reason, 'step-cannot-move');
  assert.equal(moveStep(source, 'missing', 1).reason, 'step-not-found');
  assert.equal(deleteStep(source, 'missing').reason, 'step-not-found');
  assert.equal(addStep(source, { referenceId: 'missing' }).reason, 'step-not-found');
});

test('appendix operations cover success, boundaries, and missing IDs', () => {
  const source = fixture();
  assert.equal(addAppendix(source, { idFactory: ids }).document.appendices.at(-1).id, 'appendix-new');
  assert.deepEqual(moveAppendix(source, 'a2', -1).document.appendices.map(x => x.id), ['a2', 'a1']);
  assert.equal(deleteAppendix(source, 'a1').document.appendices.length, 1);
  assert.equal(moveAppendix(source, 'a1', -1).reason, 'appendix-cannot-move');
  assert.equal(moveAppendix(source, 'missing', 1).reason, 'appendix-not-found');
  assert.equal(deleteAppendix(source, 'missing').reason, 'appendix-not-found');
});

test('block operations cover success, boundaries, and missing IDs', () => {
  const source = fixture(), inserted = insertBlock(source, 's1', { id: 'new', type: 'heading', level: 2, runs: [] }, 1);
  assert.equal(inserted.document.steps[0].blocks[1].id, 'new'); assert.equal(source.steps[0].blocks.length, 2);
  assert.deepEqual(moveBlock(source, 'p2', -1).document.steps[0].blocks.map(x => x.id), ['p2', 'p1']);
  assert.equal(moveBlock(source, 'p1', -1).reason, 'block-cannot-move');
  assert.equal(moveBlock(source, 'missing', 1).reason, 'block-not-found');
  assert.equal(deleteBlock(source, 'p1').document.steps[0].blocks.length, 1);
  assert.equal(deleteBlock(source, 'missing').reason, 'block-not-found');
  assert.equal(insertBlock(source, 'missing', { type: 'paragraph' }, 0).reason, 'group-not-found');
});

test('table row operations cover success, total boundaries, and missing IDs', () => {
  const source = fixture(), inserted = insertTableRow(source, 'table', 'r1', { idFactory: ids });
  assert.equal(inserted.document.steps[1].blocks[0].rows[1].id, 'row-new');
  assert.equal(moveTableRow(source, 'table', 'r2', -1).rowIndex, 0);
  assert.equal(moveTableRow(source, 'table', 'r2', 1).reason, 'row-cannot-move');
  assert.equal(moveTableRow(source, 'table', 'missing', 1).reason, 'row-not-found');
  assert.equal(deleteTableRow(source, 'table', 'r1').document.steps[1].blocks[0].rows.length, 2);
  assert.equal(deleteTableRow(source, 'table', 'missing').reason, 'row-not-found');
});

test('table column operations cover success, boundaries, and missing IDs', () => {
  const source = fixture(), inserted = insertTableColumn(source, 'table', 'c1', { idFactory: ids });
  assert.equal(inserted.document.steps[1].blocks[0].columns[1].id, 'column-new');
  assert.equal(moveTableColumn(source, 'table', 'c2', -1).columnIndex, 0);
  assert.equal(moveTableColumn(source, 'table', 'c1', -1).reason, 'column-cannot-move');
  assert.equal(moveTableColumn(source, 'table', 'missing', 1).reason, 'column-not-found');
  assert.equal(deleteTableColumn(source, 'table', 'c1').document.steps[1].blocks[0].columns.length, 1);
  assert.equal(deleteTableColumn(source, 'table', 'missing').reason, 'column-not-found');
});

test('column format and block style conversion return explicit results', () => {
  const source = fixture(), format = setTableColumnFormat(source, 'table', 'c1', 'gbp');
  assert.equal(format.changed, true); assert.equal(format.document.steps[1].blocks[0].columns[0].format, 'gbp');
  assert.equal(setTableColumnFormat(source, 'table', 'missing', 'gbp').reason, 'column-not-found');
  assert.equal(setTableColumnFormat(source, 'table', 'c1', 'date').reason, 'column-format-unsupported');
  const converted = convertBlockStyle(source, 'p1', 'bulletList', { idFactory: ids });
  assert.equal(converted.document.steps[0].blocks[0].type, 'bulletList'); assert.equal(converted.selectedBlockId, 'p1');
  assert.equal(convertBlockStyle(source, 'missing', 'body').reason, 'block-not-found');
  assert.equal(convertBlockStyle(source, 'table', 'body').reason, 'block-type-unsupported');
  assert.equal(convertBlockStyle(source, 'p1', 'quote').reason, 'block-style-unsupported');
});
