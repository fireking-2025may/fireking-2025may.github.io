import test from 'node:test';
import assert from 'node:assert/strict';
import { formatTableNumber, moveTableColumn, moveTableRow, parseTableNumber, recalculateTableTotals, tableColumnFormat } from '../src/state/table-model.js';

test('table domain values parse and format without editor dependencies', () => {
  assert.equal(parseTableNumber('£ 1,234.50'), 1234.5);
  assert.equal(parseTableNumber('not a number'), null);
  assert.equal(formatTableNumber(-2, 'gbp'), '-£2.00');
  assert.equal(tableColumnFormat({ numeric: true }), 'number');
});

test('table domain operations calculate totals and preserve movement boundaries', () => {
  const cell = text => ({ runs: [{ text }] });
  const block = { type: 'table', columns: [{ format: 'number' }], rows: [{ id: 'a', cells: [cell('2')] }, { id: 'b', cells: [cell('3')] }, { id: 'total', isTotal: true, cells: [cell('stale')] }] };
  recalculateTableTotals(block);
  assert.equal(block.rows[2].cells[0].runs[0].text, '5');
  assert.equal(moveTableRow(block, 1, 1), false);
  assert.equal(moveTableRow(block, 1, -1), 0);
  assert.equal(moveTableColumn(block, 0, 1), false);
});
