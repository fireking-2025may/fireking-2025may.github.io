import { normaliseRuns, safeHref } from '../state/schema.js';
export function validatedLink(external, internal) {
  const candidate = String(external || '').trim();
  return safeHref(candidate || internal);
}
export function containedSelectionOffsets(
  element,
  selection,
  documentObject = document,
) {
  if (!element || !selection?.rangeCount) return null;
  const range = selection.getRangeAt(0);
  if (
    !element.contains(range.startContainer) ||
    !element.contains(range.endContainer)
  )
    return null;
  const before = documentObject.createRange();
  before.selectNodeContents(element);
  before.setEnd(range.startContainer, range.startOffset);
  return [
    before.toString().length,
    before.toString().length + range.toString().length,
  ];
}
export function confirmStepDeletion(steps, index, confirmAction) {
  if (steps.length <= 1 || index < 0 || index >= steps.length) return false;
  return confirmAction(`Delete Step ${index + 1}: ${steps[index].title}?`);
}
export function splitListRuns(runs, offset) {
  let position = 0,
    before = [],
    after = [];
  for (const run of runs) {
    const split = Math.max(0, Math.min(run.text.length, offset - position));
    if (split) before.push({ ...run, text: run.text.slice(0, split) });
    if (split < run.text.length)
      after.push({ ...run, text: run.text.slice(split) });
    position += run.text.length;
  }
  return [normaliseRuns(before), normaliseRuns(after)];
}
export function removeEmptyListItem(block, index) {
  const item = block?.items?.[index];
  if (!item || index < 1 || item.runs?.some((run) => run.text)) return null;
  block.items.splice(index, 1);
  const previous = block.items[index - 1];
  return {
    itemId: previous.id,
    offset: (previous.runs || []).reduce(
      (length, run) => length + run.text.length,
      0,
    ),
  };
}
export function insertTableRowBeforeTotals(block, makeId) {
  let index =
    block.rows.reduce((last, row, i) => (row.isTotal ? last : i), -1) + 1;
  block.rows.splice(index, 0, {
    id: makeId('row'),
    isTotal: false,
    cells: block.columns.map(() => ({ id: makeId('cell'), runs: [] })),
  });
  return index;
}
export function insertTableRowAfter(block, rowIndex, makeId) {
  const index =
    Number.isInteger(rowIndex) && rowIndex >= 0 && rowIndex < block.rows.length
      ? rowIndex + 1
      : block.rows.length;
  block.rows.splice(index, 0, {
    id: makeId('row'),
    isTotal: false,
    cells: block.columns.map(() => ({ id: makeId('cell'), runs: [] })),
  });
  return index;
}
export function insertTableColumnAfter(block, columnIndex, makeId) {
  const index =
    Number.isInteger(columnIndex) &&
    columnIndex >= 0 &&
    columnIndex < block.columns.length
      ? columnIndex + 1
      : block.columns.length;
  block.columns.splice(index, 0, {
    id: makeId('column'),
    headingRuns: [{ text: `Column ${index + 1}` }],
    width: 100 / (block.columns.length + 1),
    format: 'text',
    totalEnabled: false,
  });
  block.rows.forEach((row) =>
    row.cells.splice(index, 0, { id: makeId('cell'), runs: [] }),
  );
  return index;
}
export class GenerationGate {
  generation = 0;
  next() {
    return ++this.generation;
  }
  isCurrent(value) {
    return value === this.generation;
  }
  cancel() {
    this.generation++;
  }
}
export function restoreTextSelection(
  element,
  offsets,
  environment = globalThis,
) {
  if (!element || !offsets) return false;
  element.focus({ preventScroll: true });
  const walker = environment.document.createTreeWalker(
    element,
    environment.NodeFilter.SHOW_TEXT,
  );
  let position = 0,
    start,
    end,
    node;
  while ((node = walker.nextNode())) {
    if (!start && offsets[0] <= position + node.length)
      start = [node, offsets[0] - position];
    if (offsets[1] <= position + node.length) {
      end = [node, offsets[1] - position];
      break;
    }
    position += node.length;
  }
  if (!start) return false;
  const range = environment.document.createRange();
  range.setStart(...start);
  range.setEnd(...(end || start));
  const selection = environment.getSelection();
  selection.removeAllRanges();
  selection.addRange(range);
  return true;
}
