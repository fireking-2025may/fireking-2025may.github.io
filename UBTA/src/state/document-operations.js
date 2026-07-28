import { newId, normaliseBlock, normaliseDocument, normaliseTableWidths } from './schema.js';
import { recalculateTableTotals, tableColumnFormat } from './table-model.js';

const unchanged = (document, reason) => ({ changed: false, document, reason });
const changed = (document, details = {}) => ({ changed: true, document: normaliseDocument(document), ...details });
const copy = document => structuredClone(document);
const directionOffset = direction => direction === 'before' || direction === 'up' || direction === 'left' ? -1 : direction === 'after' || direction === 'down' || direction === 'right' ? 1 : Number(direction);
const groups = document => [...(document.sections || []), ...(document.steps || []), ...(document.appendices || [])];
const locateBlock = (document, blockId) => {
  for (const group of groups(document)) {
    const index = group.blocks?.findIndex(block => block.id === blockId) ?? -1;
    if (index >= 0) return { group, index, block: group.blocks[index] };
  }
  return null;
};
const locateTable = (document, blockId) => {
  const found = locateBlock(document, blockId);
  return found?.block.type === 'table' ? found : null;
};

function addGroup(document, collection, options = {}) {
  const noun = collection === 'appendices' ? 'appendix' : 'step';
  const source = document[collection] || [];
  const referenceIndex = options.referenceId == null ? source.length - 1 : source.findIndex(group => group.id === options.referenceId);
  if (options.referenceId != null && referenceIndex < 0) return unchanged(document, `${noun}-not-found`);
  const makeId = options.idFactory || newId;
  const id = options.id || makeId(noun);
  const blockId = options.blockId || makeId('block');
  const next = copy(document);
  const index = options.referenceId == null ? source.length : referenceIndex + (options.position === 'before' ? 0 : 1);
  next[collection].splice(index, 0, { id, title: options.title || (collection === 'steps' ? 'New transaction step' : 'New appendix'), blocks: [{ id: blockId, type: 'paragraph', runs: [] }] });
  return changed(next, { selectedGroupId: id, selectedBlockId: blockId, index });
}

function moveGroup(document, collection, id, direction) {
  const noun = collection === 'appendices' ? 'appendix' : 'step';
  const currentIndex = (document[collection] || []).findIndex(group => group.id === id);
  if (currentIndex < 0) return unchanged(document, `${noun}-not-found`);
  const nextIndex = currentIndex + directionOffset(direction);
  if (nextIndex < 0 || nextIndex >= document[collection].length) return unchanged(document, `${noun}-cannot-move`);
  const next = copy(document);
  [next[collection][currentIndex], next[collection][nextIndex]] = [next[collection][nextIndex], next[collection][currentIndex]];
  return changed(next, { selectedGroupId: id, index: nextIndex });
}

function deleteGroup(document, collection, id) {
  const noun = collection === 'appendices' ? 'appendix' : 'step';
  const index = (document[collection] || []).findIndex(group => group.id === id);
  if (index < 0) return unchanged(document, `${noun}-not-found`);
  const next = copy(document);
  next[collection].splice(index, 1);
  const selected = next[collection][Math.min(index, next[collection].length - 1)];
  return changed(next, { selectedGroupId: selected?.id || null, index });
}

const addStepOperation = (document, options) => addGroup(document, 'steps', options);
const moveStepOperation = (document, stepId, direction) => moveGroup(document, 'steps', stepId, direction);
const deleteStepOperation = (document, stepId) => deleteGroup(document, 'steps', stepId);
const addAppendixOperation = (document, options) => addGroup(document, 'appendices', options);
const moveAppendixOperation = (document, appendixId, direction) => moveGroup(document, 'appendices', appendixId, direction);
const deleteAppendixOperation = (document, appendixId) => deleteGroup(document, 'appendices', appendixId);

function insertBlockOperation(document, groupId, block, index) {
  const original = groups(document).find(group => group.id === groupId);
  if (!original) return unchanged(document, 'group-not-found');
  if (!Number.isInteger(index) || index < 0 || index > original.blocks.length) return unchanged(document, 'block-index-invalid');
  const next = copy(document), group = groups(next).find(item => item.id === groupId);
  const value = normaliseBlock(block);
  group.blocks.splice(index, 0, value);
  return changed(next, { selectedGroupId: groupId, selectedBlockId: value.id, index });
}

function moveBlockOperation(document, blockId, direction) {
  const found = locateBlock(document, blockId);
  if (!found) return unchanged(document, 'block-not-found');
  const nextIndex = found.index + directionOffset(direction);
  if (nextIndex < 0 || nextIndex >= found.group.blocks.length) return unchanged(document, 'block-cannot-move');
  const next = copy(document), target = locateBlock(next, blockId);
  [target.group.blocks[target.index], target.group.blocks[nextIndex]] = [target.group.blocks[nextIndex], target.group.blocks[target.index]];
  return changed(next, { selectedGroupId: target.group.id, selectedBlockId: blockId, index: nextIndex });
}

function deleteBlockOperation(document, blockId, options = {}) {
  const found = locateBlock(document, blockId);
  if (!found) return unchanged(document, 'block-not-found');
  const next = copy(document), target = locateBlock(next, blockId);
  target.group.blocks.splice(target.index, 1);
  if (!target.group.blocks.length) target.group.blocks.push({ id: (options.idFactory || newId)('block'), type: 'paragraph', runs: [] });
  const selected = target.group.blocks[Math.min(target.index, target.group.blocks.length - 1)];
  return changed(next, { selectedGroupId: target.group.id, selectedBlockId: selected.id, selectedBlockType: selected.type, index: target.index });
}

function insertTableRowOperation(document, blockId, afterRowId = null, options = {}) {
  const found = locateTable(document, blockId);
  if (!found) return unchanged(document, 'table-not-found');
  const rowIndex = afterRowId == null ? found.block.rows.findIndex(row => row.isTotal) - 1 : found.block.rows.findIndex(row => row.id === afterRowId);
  if (afterRowId != null && rowIndex < 0) return unchanged(document, 'row-not-found');
  const next = copy(document), block = locateTable(next, blockId).block, makeId = options.idFactory || newId;
  const totalIndex = block.rows.findIndex(row => row.isTotal), index = afterRowId == null ? (totalIndex < 0 ? block.rows.length : totalIndex) : Math.min(rowIndex + 1, totalIndex < 0 ? block.rows.length : totalIndex);
  block.rows.splice(index, 0, { id: options.id || makeId('row'), cells: block.columns.map(() => ({ id: makeId('cell'), runs: [] })) });
  recalculateTableTotals(block);
  return changed(next, { selectedBlockId: blockId, rowIndex: index });
}

function moveTableRowOperation(document, blockId, rowId, direction) {
  const found = locateTable(document, blockId);
  if (!found) return unchanged(document, 'table-not-found');
  const index = found.block.rows.findIndex(row => row.id === rowId);
  if (index < 0) return unchanged(document, 'row-not-found');
  const target = index + directionOffset(direction);
  if (target < 0 || target >= found.block.rows.length || found.block.rows[target].isTotal !== found.block.rows[index].isTotal) return unchanged(document, 'row-cannot-move');
  const next = copy(document), rows = locateTable(next, blockId).block.rows;
  [rows[index], rows[target]] = [rows[target], rows[index]];
  recalculateTableTotals(locateTable(next, blockId).block);
  return changed(next, { selectedBlockId: blockId, rowIndex: target });
}

function deleteTableRowOperation(document, blockId, rowId) {
  const found = locateTable(document, blockId);
  if (!found) return unchanged(document, 'table-not-found');
  const index = found.block.rows.findIndex(row => row.id === rowId);
  if (index < 0) return unchanged(document, 'row-not-found');
  const next = copy(document), block = locateTable(next, blockId).block;
  block.rows.splice(index, 1); recalculateTableTotals(block);
  return changed(next, { selectedBlockId: blockId, rowIndex: block.rows.length ? Math.min(index, block.rows.length - 1) : null });
}

function insertTableColumnOperation(document, blockId, afterColumnId = null, options = {}) {
  const found = locateTable(document, blockId);
  if (!found) return unchanged(document, 'table-not-found');
  if (found.block.columns.length >= 8) return unchanged(document, 'column-limit-reached');
  const previous = afterColumnId == null ? found.block.columns.length - 1 : found.block.columns.findIndex(column => column.id === afterColumnId);
  if (afterColumnId != null && previous < 0) return unchanged(document, 'column-not-found');
  const next = copy(document), block = locateTable(next, blockId).block, makeId = options.idFactory || newId, index = previous + 1;
  block.columns.splice(index, 0, { id: options.id || makeId('column'), headingRuns: [{ text: `Column ${index + 1}` }], width: 100 / (block.columns.length + 1), format: 'text' });
  block.rows.forEach(row => row.cells.splice(index, 0, { id: makeId('cell'), runs: [] }));
  const widths = normaliseTableWidths(block.columns.map(() => 100 / block.columns.length), block.columns.length);
  block.columns.forEach((column, columnIndex) => { column.width = widths[columnIndex]; });
  return changed(next, { selectedBlockId: blockId, columnIndex: index });
}

function moveTableColumnOperation(document, blockId, columnId, direction) {
  const found = locateTable(document, blockId);
  if (!found) return unchanged(document, 'table-not-found');
  const index = found.block.columns.findIndex(column => column.id === columnId);
  if (index < 0) return unchanged(document, 'column-not-found');
  const target = index + directionOffset(direction);
  if (target < 0 || target >= found.block.columns.length) return unchanged(document, 'column-cannot-move');
  const next = copy(document), block = locateTable(next, blockId).block;
  [block.columns[index], block.columns[target]] = [block.columns[target], block.columns[index]];
  block.rows.forEach(row => { [row.cells[index], row.cells[target]] = [row.cells[target], row.cells[index]]; });
  recalculateTableTotals(block);
  return changed(next, { selectedBlockId: blockId, columnIndex: target });
}

function deleteTableColumnOperation(document, blockId, columnId) {
  const found = locateTable(document, blockId);
  if (!found) return unchanged(document, 'table-not-found');
  const index = found.block.columns.findIndex(column => column.id === columnId);
  if (index < 0) return unchanged(document, 'column-not-found');
  if (found.block.columns.length <= 1) return unchanged(document, 'column-cannot-delete');
  const next = copy(document), block = locateTable(next, blockId).block;
  block.columns.splice(index, 1); block.rows.forEach(row => row.cells.splice(index, 1)); recalculateTableTotals(block);
  return changed(next, { selectedBlockId: blockId, columnIndex: Math.min(index, block.columns.length - 1) });
}

function setTableColumnFormatOperation(document, blockId, columnId, format) {
  const found = locateTable(document, blockId);
  if (!found) return unchanged(document, 'table-not-found');
  const index = found.block.columns.findIndex(column => column.id === columnId);
  if (index < 0) return unchanged(document, 'column-not-found');
  if (!['text', 'number', 'gbp'].includes(format)) return unchanged(document, 'column-format-unsupported');
  if (tableColumnFormat(found.block.columns[index]) === format) return unchanged(document, 'column-format-unchanged');
  const next = copy(document), block = locateTable(next, blockId).block;
  block.columns[index].format = format; delete block.columns[index].numeric; recalculateTableTotals(block);
  return changed(next, { selectedBlockId: blockId, columnIndex: index });
}

function convertBlockStyleOperation(document, blockId, style, options = {}) {
  const found = locateBlock(document, blockId);
  if (!found) return unchanged(document, 'block-not-found');
  if (!['heading', 'paragraph', 'bulletList', 'numberList'].includes(found.block.type)) return unchanged(document, 'block-type-unsupported');
  const heading = /^heading([1-4])$/.exec(style), type = heading ? 'heading' : style === 'body' ? 'paragraph' : ['bulletList', 'numberList'].includes(style) ? style : null;
  if (!type) return unchanged(document, 'block-style-unsupported');
  const next = copy(document), block = locateBlock(next, blockId).block, makeId = options.idFactory || newId;
  if (type === 'heading' || type === 'paragraph') {
    block.runs = block.runs || block.items?.flatMap(item => item.runs) || [];
    delete block.items; block.type = type;
    if (heading) block.level = Number(heading[1]); else delete block.level;
  } else {
    block.items = block.items || [{ id: makeId('item'), level: 1, runs: block.runs || [] }];
    delete block.runs; delete block.level; block.type = type;
  }
  return changed(next, { selectedBlockId: blockId, selectedBlockType: type });
}

export { addStepOperation as addStep, moveStepOperation as moveStep, deleteStepOperation as deleteStep, addAppendixOperation as addAppendix, moveAppendixOperation as moveAppendix, deleteAppendixOperation as deleteAppendix, insertBlockOperation as insertBlock, moveBlockOperation as moveBlock, deleteBlockOperation as deleteBlock, insertTableRowOperation as insertTableRow, moveTableRowOperation as moveTableRow, deleteTableRowOperation as deleteTableRow, insertTableColumnOperation as insertTableColumn, moveTableColumnOperation as moveTableColumn, deleteTableColumnOperation as deleteTableColumn, setTableColumnFormatOperation as setTableColumnFormat, convertBlockStyleOperation as convertBlockStyle };
