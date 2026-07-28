import { recalculateTableTotals } from './table-model.js';

const unchanged = (document, error) => ({ changed: false, document, error });
const changed = (document, details = {}) => ({ changed: true, document, ...details });
const clone = document => structuredClone(document);

function findDocumentGroup(document, groupId) {
  for (const [groupType, groups] of [['step', document.steps || []], ['appendix', document.appendices || []]]) {
    const groupIndex = groups.findIndex(group => group.id === groupId);
    if (groupIndex !== -1) return { groupType, groupIndex, group: groups[groupIndex] };
  }
  return null;
}

function findDocumentBlock(document, blockId) {
  for (const group of [...(document.steps || []), ...(document.appendices || [])]) {
    const blockIndex = group.blocks.findIndex(block => block.id === blockId);
    if (blockIndex !== -1) return { group, blockIndex, block: group.blocks[blockIndex] };
  }
  return null;
}

export function addStep(document, referenceStepId, position, ids) {
  const index = document.steps.findIndex(step => step.id === referenceStepId);
  if (index === -1) return unchanged(document, 'step-not-found');
  const nextDocument = clone(document);
  const step = { id: ids.stepId, title: 'New transaction step', blocks: [{ id: ids.blockId, type: 'paragraph', runs: [] }] };
  nextDocument.steps.splice(index + (position === 'after' ? 1 : 0), 0, step);
  return changed(nextDocument, { selectedStepId: step.id, selectedBlockId: ids.blockId });
}

export function moveStep(document, stepId, direction) {
  const index = document.steps.findIndex(step => step.id === stepId);
  if (index === -1) return unchanged(document, 'step-not-found');
  const targetIndex = index + direction;
  if (targetIndex < 0 || targetIndex >= document.steps.length) return unchanged(document, 'step-cannot-move');
  const nextDocument = clone(document);
  [nextDocument.steps[index], nextDocument.steps[targetIndex]] = [nextDocument.steps[targetIndex], nextDocument.steps[index]];
  return changed(nextDocument, { selectedStepId: stepId, index: targetIndex });
}

export function deleteStep(document, stepId) {
  const index = document.steps.findIndex(step => step.id === stepId);
  if (index === -1) return unchanged(document, 'step-not-found');
  if (document.steps.length <= 1) return unchanged(document, 'final-step');
  const nextDocument = clone(document);
  nextDocument.steps.splice(index, 1);
  return changed(nextDocument, { selectedStepId: nextDocument.steps[Math.min(index, nextDocument.steps.length - 1)].id });
}

export function addAppendix(document, ids) {
  const nextDocument = clone(document);
  const appendix = { id: ids.appendixId, title: 'New appendix', blocks: [{ id: ids.blockId, type: 'paragraph', runs: [] }] };
  nextDocument.appendices.push(appendix);
  return changed(nextDocument, { selectedAppendixId: appendix.id, selectedBlockId: ids.blockId, index: nextDocument.appendices.length - 1 });
}

export function moveAppendix(document, appendixId, direction) {
  const index = document.appendices.findIndex(appendix => appendix.id === appendixId);
  if (index === -1) return unchanged(document, 'appendix-not-found');
  const targetIndex = index + direction;
  if (targetIndex < 0 || targetIndex >= document.appendices.length) return unchanged(document, 'appendix-cannot-move');
  const nextDocument = clone(document);
  [nextDocument.appendices[index], nextDocument.appendices[targetIndex]] = [nextDocument.appendices[targetIndex], nextDocument.appendices[index]];
  return changed(nextDocument, { selectedAppendixId: appendixId, index: targetIndex });
}

export function deleteAppendix(document, appendixId) {
  const index = document.appendices.findIndex(appendix => appendix.id === appendixId);
  if (index === -1) return unchanged(document, 'appendix-not-found');
  const nextDocument = clone(document);
  nextDocument.appendices.splice(index, 1);
  const target = nextDocument.appendices[Math.min(index, nextDocument.appendices.length - 1)];
  return changed(nextDocument, { selectedAppendixId: target?.id || null, selectedStepId: target ? null : nextDocument.steps.at(-1)?.id || null });
}

function createBlock(type, headingLevel, makeId) {
  const id = makeId('block');
  if (type === 'heading') return { id, type, level: Math.min(4, Math.max(1, Number(headingLevel) || 2)), runs: [] };
  if (type.endsWith('List')) return { id, type, items: [{ id: makeId('item'), level: 1, runs: [] }] };
  if (type === 'table') return { id, type, captionRuns: [{ text: 'Table caption' }], columns: [{ id: makeId('column'), headingRuns: [{ text: 'Column 1' }], width: 50 }, { id: makeId('column'), headingRuns: [{ text: 'Column 2' }], width: 50 }], rows: [{ id: makeId('row'), cells: [{ id: makeId('cell'), runs: [] }, { id: makeId('cell'), runs: [] }] }] };
  return { id, type, runs: [] };
}

export function insertBlock(document, context, type, headingLevel, makeId) {
  if (!['paragraph', 'heading', 'bulletList', 'numberList', 'table'].includes(type)) return unchanged(document, 'unsupported-block-type');
  const found = findDocumentGroup(document, context?.groupId);
  if (!found) return unchanged(document, 'group-not-found');
  let index = found.group.blocks.length;
  if (context.referenceBlockId) {
    const referenceIndex = found.group.blocks.findIndex(block => block.id === context.referenceBlockId);
    if (referenceIndex === -1) return unchanged(document, 'reference-block-not-found');
    index = referenceIndex + (context.position === 'before' ? 0 : 1);
  }
  const nextDocument = clone(document);
  const nextGroup = findDocumentGroup(nextDocument, context.groupId).group;
  const block = createBlock(type, headingLevel, makeId);
  nextGroup.blocks.splice(index, 0, block);
  return changed(nextDocument, { selectedGroupId: nextGroup.id, selectedGroupType: found.groupType, selectedBlockId: block.id, block });
}

export function insertBlocks(document, context, blocks) {
  const found = findDocumentGroup(document, context?.groupId);
  if (!found) return unchanged(document, 'group-not-found');
  if (!Array.isArray(blocks) || !blocks.length) return unchanged(document, 'blocks-required');
  let index = found.group.blocks.length;
  if (context.referenceBlockId) {
    const referenceIndex = found.group.blocks.findIndex(block => block.id === context.referenceBlockId);
    if (referenceIndex === -1) return unchanged(document, 'reference-block-not-found');
    index = referenceIndex + (context.position === 'before' ? 0 : 1);
  }
  const nextDocument = clone(document);
  const nextGroup = findDocumentGroup(nextDocument, context.groupId).group;
  const insertedBlocks = structuredClone(blocks);
  nextGroup.blocks.splice(index, 0, ...insertedBlocks);
  return changed(nextDocument, { selectedGroupId: nextGroup.id, selectedGroupType: found.groupType, selectedBlockId: insertedBlocks[0].id, block: insertedBlocks[0] });
}

export function moveBlock(document, blockId, direction) {
  const found = findDocumentBlock(document, blockId);
  if (!found) return unchanged(document, 'block-not-found');
  const targetIndex = found.blockIndex + direction;
  if (targetIndex < 0 || targetIndex >= found.group.blocks.length) return unchanged(document, 'block-cannot-move');
  const nextDocument = clone(document);
  const nextFound = findDocumentBlock(nextDocument, blockId);
  [nextFound.group.blocks[nextFound.blockIndex], nextFound.group.blocks[targetIndex]] = [nextFound.group.blocks[targetIndex], nextFound.group.blocks[nextFound.blockIndex]];
  return changed(nextDocument, { selectedBlockId: blockId, index: targetIndex });
}

export function deleteBlock(document, blockId, makeId) {
  const found = findDocumentBlock(document, blockId);
  if (!found) return unchanged(document, 'block-not-found');
  const nextDocument = clone(document);
  const nextFound = findDocumentBlock(nextDocument, blockId);
  nextFound.group.blocks.splice(nextFound.blockIndex, 1);
  if (!nextFound.group.blocks.length) nextFound.group.blocks.push({ id: makeId('block'), type: 'paragraph', runs: [] });
  const target = nextFound.group.blocks[Math.max(0, nextFound.blockIndex - 1)] || nextFound.group.blocks[0];
  return changed(nextDocument, { deletedBlockType: found.block.type, selectedBlockId: target.id, selectedBlock: target });
}

function tableResult(document, blockId) {
  const found = findDocumentBlock(document, blockId);
  if (!found || found.block.type !== 'table') return null;
  const nextDocument = clone(document);
  return { nextDocument, block: findDocumentBlock(nextDocument, blockId).block };
}

export function addTableRow(document, blockId, afterRowId, makeId) {
  const table = tableResult(document, blockId);
  if (!table) return unchanged(document, 'table-not-found');
  const rowIndex = table.block.rows.findIndex(row => row.id === afterRowId);
  if (afterRowId && rowIndex === -1) return unchanged(document, 'row-not-found');
  const index = rowIndex === -1 ? table.block.rows.length : rowIndex + 1;
  const row = { id: makeId('row'), isTotal: false, cells: table.block.columns.map(() => ({ id: makeId('cell'), runs: [] })) };
  table.block.rows.splice(index, 0, row); recalculateTableTotals(table.block);
  return changed(table.nextDocument, { selectedRowId: row.id, index });
}

export function moveTableRow(document, blockId, rowId, direction) {
  const table = tableResult(document, blockId);
  if (!table) return unchanged(document, 'table-not-found');
  const index = table.block.rows.findIndex(row => row.id === rowId);
  if (index === -1) return unchanged(document, 'row-not-found');
  const target = index + direction;
  if (target < 0 || target >= table.block.rows.length || table.block.rows[target].isTotal !== table.block.rows[index].isTotal) return unchanged(document, 'row-cannot-move');
  [table.block.rows[index], table.block.rows[target]] = [table.block.rows[target], table.block.rows[index]]; recalculateTableTotals(table.block);
  return changed(table.nextDocument, { selectedRowId: rowId, index: target });
}

export function deleteTableRow(document, blockId, rowId) {
  const table = tableResult(document, blockId);
  if (!table) return unchanged(document, 'table-not-found');
  const index = table.block.rows.findIndex(row => row.id === rowId);
  if (index === -1) return unchanged(document, 'row-not-found');
  table.block.rows.splice(index, 1); recalculateTableTotals(table.block);
  return changed(table.nextDocument, { selectedRowId: table.block.rows[Math.min(index, table.block.rows.length - 1)]?.id || null, index: Math.min(index, table.block.rows.length - 1) });
}

export function addTableColumn(document, blockId, afterColumnId, makeId) {
  const table = tableResult(document, blockId);
  if (!table) return unchanged(document, 'table-not-found');
  const columnIndex = table.block.columns.findIndex(column => column.id === afterColumnId);
  if (afterColumnId && columnIndex === -1) return unchanged(document, 'column-not-found');
  const index = columnIndex === -1 ? table.block.columns.length : columnIndex + 1;
  const column = { id: makeId('column'), headingRuns: [{ text: `Column ${index + 1}` }], width: 100 / (table.block.columns.length + 1), format: 'text' };
  table.block.columns.splice(index, 0, column);
  table.block.rows.forEach(row => row.cells.splice(index, 0, { id: makeId('cell'), runs: [] })); recalculateTableTotals(table.block);
  return changed(table.nextDocument, { selectedColumnId: column.id, index });
}

export function moveTableColumn(document, blockId, columnId, direction) {
  const table = tableResult(document, blockId);
  if (!table) return unchanged(document, 'table-not-found');
  const index = table.block.columns.findIndex(column => column.id === columnId);
  if (index === -1) return unchanged(document, 'column-not-found');
  const target = index + direction;
  if (target < 0 || target >= table.block.columns.length) return unchanged(document, 'column-cannot-move');
  [table.block.columns[index], table.block.columns[target]] = [table.block.columns[target], table.block.columns[index]];
  table.block.rows.forEach(row => { [row.cells[index], row.cells[target]] = [row.cells[target], row.cells[index]]; }); recalculateTableTotals(table.block);
  return changed(table.nextDocument, { selectedColumnId: columnId, index: target });
}

export function deleteTableColumn(document, blockId, columnId) {
  const table = tableResult(document, blockId);
  if (!table) return unchanged(document, 'table-not-found');
  const index = table.block.columns.findIndex(column => column.id === columnId);
  if (index === -1) return unchanged(document, 'column-not-found');
  if (table.block.columns.length <= 1) return unchanged(document, 'final-column');
  table.block.columns.splice(index, 1); table.block.rows.forEach(row => row.cells.splice(index, 1)); recalculateTableTotals(table.block);
  return changed(table.nextDocument, { selectedColumnId: table.block.columns[Math.min(index, table.block.columns.length - 1)].id, index: Math.min(index, table.block.columns.length - 1) });
}

export function setTableColumnFormat(document, blockId, columnId, format) {
  const table = tableResult(document, blockId);
  if (!table) return unchanged(document, 'table-not-found');
  const column = table.block.columns.find(candidate => candidate.id === columnId);
  if (!column) return unchanged(document, 'column-not-found');
  if (!['text', 'number', 'gbp'].includes(format)) return unchanged(document, 'unsupported-column-format');
  column.format = format;
  delete column.numeric;
  recalculateTableTotals(table.block);
  return changed(table.nextDocument, { selectedColumnId: columnId, format });
}

export function convertBlockStyle(document, blockId, style, makeId) {
  const found = findDocumentBlock(document, blockId);
  if (!found) return unchanged(document, 'block-not-found');
  if (!['heading', 'paragraph', 'bulletList', 'numberList'].includes(found.block.type)) return unchanged(document, 'unsupported-block-type');
  const nextDocument = clone(document), block = findDocumentBlock(nextDocument, blockId).block;
  if (style.startsWith('heading')) { block.type = 'heading'; block.level = Number(style.at(-1)); delete block.items; block.runs ||= []; }
  else if (style === 'body') { block.type = 'paragraph'; delete block.items; block.runs ||= []; }
  else if (['bulletList', 'numberList'].includes(style)) { block.items ||= [{ id: makeId('item'), level: 1, runs: block.runs || [] }]; block.type = style; delete block.runs; }
  else return unchanged(document, 'unsupported-style');
  return changed(nextDocument, { selectedBlockId: blockId, blockType: block.type });
}
