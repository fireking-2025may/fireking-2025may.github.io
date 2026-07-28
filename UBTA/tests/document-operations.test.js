import test from 'node:test';
import assert from 'node:assert/strict';
import * as operations from '../src/state/document-operations.js';

const makeId = prefix => `${prefix}-new`;
const documentFixture = () => ({
  schemaVersion: 5,
  meta: {},
  sections: [],
  steps: [
    { id: 'step-a', title: 'A', blocks: [{ id: 'block-a', type: 'paragraph', runs: [{ text: 'A' }] }, { id: 'block-b', type: 'heading', level: 2, runs: [{ text: 'B' }] }] },
    { id: 'step-b', title: 'B', blocks: [{ id: 'table', type: 'table', columns: [{ id: 'column-a', format: 'text', headingRuns: [] }, { id: 'column-b', format: 'number', headingRuns: [] }], rows: [{ id: 'row-a', cells: [{ id: 'a1', runs: [] }, { id: 'a2', runs: [{ text: '2' }] }] }, { id: 'row-b', cells: [{ id: 'b1', runs: [] }, { id: 'b2', runs: [{ text: '3' }] }] }, { id: 'total', isTotal: true, cells: [{ id: 't1', runs: [] }, { id: 't2', runs: [] }] }] }] },
  ],
  appendices: [{ id: 'appendix-a', title: 'A', blocks: [{ id: 'appendix-block', type: 'paragraph', runs: [] }] }, { id: 'appendix-b', title: 'B', blocks: [] }],
});

function expectUnchanged(result, document, error) {
  assert.equal(result.changed, false);
  assert.strictEqual(result.document, document);
  assert.equal(result.error, error);
}

test('step operations return success, boundary, and missing-ID results', () => {
  const source = documentFixture();
  const added = operations.addStep(source, 'step-a', 'after', { stepId: 'step-new', blockId: 'step-block' });
  assert.deepEqual(added.document.steps.map(step => step.id), ['step-a', 'step-new', 'step-b']);
  assert.notStrictEqual(added.document, source);
  expectUnchanged(operations.addStep(source, 'missing', 'after', {}), source, 'step-not-found');
  assert.deepEqual(operations.moveStep(source, 'step-b', -1).document.steps.map(step => step.id), ['step-b', 'step-a']);
  expectUnchanged(operations.moveStep(source, 'step-a', -1), source, 'step-cannot-move');
  expectUnchanged(operations.moveStep(source, 'missing', 1), source, 'step-not-found');
  assert.deepEqual(operations.deleteStep(source, 'step-a').document.steps.map(step => step.id), ['step-b']);
  const singleStep = { ...source, steps: [source.steps[0]] };
  expectUnchanged(operations.deleteStep(singleStep, 'step-a'), singleStep, 'final-step');
  expectUnchanged(operations.deleteStep(source, 'missing'), source, 'step-not-found');
});

test('appendix operations return success, boundary, and missing-ID results', () => {
  const source = documentFixture();
  const added = operations.addAppendix(source, { appendixId: 'appendix-new', blockId: 'new-block' });
  assert.equal(added.document.appendices.at(-1).id, 'appendix-new');
  assert.deepEqual(operations.moveAppendix(source, 'appendix-b', -1).document.appendices.map(item => item.id), ['appendix-b', 'appendix-a']);
  expectUnchanged(operations.moveAppendix(source, 'appendix-a', -1), source, 'appendix-cannot-move');
  expectUnchanged(operations.moveAppendix(source, 'missing', 1), source, 'appendix-not-found');
  assert.deepEqual(operations.deleteAppendix(source, 'appendix-a').document.appendices.map(item => item.id), ['appendix-b']);
  expectUnchanged(operations.deleteAppendix(source, 'missing'), source, 'appendix-not-found');
});

test('block operations return success, boundary, and missing-ID results', () => {
  const source = documentFixture();
  const inserted = operations.insertBlock(source, { groupId: 'step-a', referenceBlockId: 'block-a', position: 'after' }, 'heading', 3, makeId);
  assert.equal(inserted.block.level, 3);
  assert.deepEqual(inserted.document.steps[0].blocks.map(block => block.id), ['block-a', 'block-new', 'block-b']);
  expectUnchanged(operations.insertBlock(source, { groupId: 'missing' }, 'paragraph', 2, makeId), source, 'group-not-found');
  expectUnchanged(operations.insertBlock(source, { groupId: 'step-a', referenceBlockId: 'missing' }, 'paragraph', 2, makeId), source, 'reference-block-not-found');
  assert.equal(operations.insertBlocks(source, { groupId: 'step-a' }, [{ id: 'supplied', type: 'paragraph', runs: [] }]).document.steps[0].blocks.at(-1).id, 'supplied');
  expectUnchanged(operations.insertBlocks(source, { groupId: 'step-a' }, []), source, 'blocks-required');
  expectUnchanged(operations.insertBlocks(source, { groupId: 'missing' }, [{ id: 'supplied' }]), source, 'group-not-found');
  assert.deepEqual(operations.moveBlock(source, 'block-b', -1).document.steps[0].blocks.map(block => block.id), ['block-b', 'block-a']);
  expectUnchanged(operations.moveBlock(source, 'block-a', -1), source, 'block-cannot-move');
  expectUnchanged(operations.moveBlock(source, 'missing', 1), source, 'block-not-found');
  assert.equal(operations.deleteBlock(source, 'block-a', makeId).selectedBlockId, 'block-b');
  expectUnchanged(operations.deleteBlock(source, 'missing', makeId), source, 'block-not-found');
});

test('table row operations return success, boundary, and missing-ID results', () => {
  const source = documentFixture();
  const added = operations.addTableRow(source, 'table', 'row-a', makeId);
  assert.equal(added.document.steps[1].blocks[0].rows[1].id, 'row-new');
  expectUnchanged(operations.addTableRow(source, 'table', 'missing', makeId), source, 'row-not-found');
  expectUnchanged(operations.addTableRow(source, 'missing', null, makeId), source, 'table-not-found');
  assert.deepEqual(operations.moveTableRow(source, 'table', 'row-b', -1).document.steps[1].blocks[0].rows.slice(0, 2).map(row => row.id), ['row-b', 'row-a']);
  expectUnchanged(operations.moveTableRow(source, 'table', 'row-a', -1), source, 'row-cannot-move');
  expectUnchanged(operations.moveTableRow(source, 'table', 'missing', 1), source, 'row-not-found');
  assert.equal(operations.deleteTableRow(source, 'table', 'row-a').document.steps[1].blocks[0].rows.length, 2);
  expectUnchanged(operations.deleteTableRow(source, 'table', 'missing'), source, 'row-not-found');
});

test('table column operations return success, boundary, and missing-ID results', () => {
  const source = documentFixture();
  const added = operations.addTableColumn(source, 'table', 'column-a', makeId);
  assert.equal(added.document.steps[1].blocks[0].columns[1].id, 'column-new');
  expectUnchanged(operations.addTableColumn(source, 'table', 'missing', makeId), source, 'column-not-found');
  expectUnchanged(operations.addTableColumn(source, 'missing', null, makeId), source, 'table-not-found');
  assert.deepEqual(operations.moveTableColumn(source, 'table', 'column-b', -1).document.steps[1].blocks[0].columns.map(column => column.id), ['column-b', 'column-a']);
  expectUnchanged(operations.moveTableColumn(source, 'table', 'column-a', -1), source, 'column-cannot-move');
  expectUnchanged(operations.moveTableColumn(source, 'table', 'missing', 1), source, 'column-not-found');
  assert.equal(operations.deleteTableColumn(source, 'table', 'column-a').document.steps[1].blocks[0].columns.length, 1);
  const oneColumn = operations.deleteTableColumn(source, 'table', 'column-a').document;
  expectUnchanged(operations.deleteTableColumn(oneColumn, 'table', 'column-b'), oneColumn, 'final-column');
  expectUnchanged(operations.deleteTableColumn(source, 'table', 'missing'), source, 'column-not-found');
  assert.equal(operations.setTableColumnFormat(source, 'table', 'column-a', 'gbp').document.steps[1].blocks[0].columns[0].format, 'gbp');
  expectUnchanged(operations.setTableColumnFormat(source, 'table', 'missing', 'gbp'), source, 'column-not-found');
  expectUnchanged(operations.setTableColumnFormat(source, 'table', 'column-a', 'invalid'), source, 'unsupported-column-format');
});

test('block style conversion returns success, boundary, and missing-ID results', () => {
  const source = documentFixture();
  const converted = operations.convertBlockStyle(source, 'block-a', 'numberList', makeId);
  assert.equal(converted.document.steps[0].blocks[0].type, 'numberList');
  assert.equal(converted.document.steps[0].blocks[0].items[0].runs[0].text, 'A');
  expectUnchanged(operations.convertBlockStyle(source, 'block-a', 'unknown', makeId), source, 'unsupported-style');
  expectUnchanged(operations.convertBlockStyle(source, 'missing', 'body', makeId), source, 'block-not-found');
  expectUnchanged(operations.convertBlockStyle(source, 'table', 'body', makeId), source, 'unsupported-block-type');
});
