import { safeHref } from '../state/schema.js';

export const validatedLink = (external, internal) => safeHref(external.trim() || internal);

export function confirmStepDeletion(steps, index, confirmAction = globalThis.confirm) {
  if (steps.length <= 1 || !steps[index]) return false;
  return confirmAction(`Delete Step ${index + 1}: ${steps[index].title}?`);
}

export function splitListRuns(runs, offset) {
  const before = [], after = [];
  let position = 0;
  for (const run of runs) {
    const cut = Math.max(0, Math.min(run.text.length, offset - position));
    if (cut) before.push({ ...run, text: run.text.slice(0, cut) });
    if (cut < run.text.length) after.push({ ...run, text: run.text.slice(cut) });
    position += run.text.length;
  }
  return [before, after];
}

export function insertTableRowBeforeTotals(block, makeId) {
  const ordinary = block.rows.map((row, index) => !row.isTotal ? index : -1).filter(index => index >= 0);
  const index = (ordinary.at(-1) ?? -1) + 1;
  block.rows.splice(index, 0, { id: makeId('row'), isTotal: false, cells: block.columns.map(() => ({ id: makeId('cell'), runs: [] })) });
  return index;
}

export class GenerationGate {
  value = 0;
  next() { return ++this.value; }
  isCurrent(generation) { return generation === this.value; }
}

export function restoreTextSelection(element, offsets, environment = globalThis) {
  if (!element) return false;
  element.focus({ preventScroll: true });
  const [start, end] = offsets || [0, 0];
  const walker = environment.document.createTreeWalker(element, environment.NodeFilter.SHOW_TEXT);
  let position = 0, anchor, focus, node;
  while ((node = walker.nextNode())) {
    if (!anchor && start <= position + node.length) anchor = [node, start - position];
    if (end <= position + node.length) { focus = [node, end - position]; break; }
    position += node.length;
  }
  if (!anchor) return false;
  const range = environment.document.createRange();
  range.setStart(...anchor); range.setEnd(...(focus || anchor));
  const selection = environment.getSelection();
  selection.removeAllRanges(); selection.addRange(range);
  return true;
}
