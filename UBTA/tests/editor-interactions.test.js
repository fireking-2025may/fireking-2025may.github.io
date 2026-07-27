import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { runsFromElement, transformRuns, insertPlainText } from '../src/editor/dom-runs.js';
import { GenerationGate, confirmStepDeletion, insertTableRowBeforeTotals, restoreTextSelection, splitListRuns, validatedLink } from '../src/editor/interactions.js';
import { History } from '../src/state/history.js';

const fixture = JSON.parse(await readFile(new URL('./fixtures/rich-runs.json', import.meta.url)));
const text = value => ({ nodeType: 3, nodeValue: value, length: value.length, nextSibling: null });
function element(tag, children = [], href) {
  const node = { nodeType: 1, tagName: tag.toUpperCase(), childNodes: children, nextSibling: null, getAttribute: name => name === 'href' ? href : null };
  children.forEach((child, index) => child.nextSibling = children[index + 1] || null);
  return node;
}
const inflate = value => typeof value === 'string' ? text(value) : element(value.tag, value.children.map(inflate), value.href);

test('highlighting a mixed highlighted/link selection retains every link run', () => {
  const result = transformRuns(fixture.mixedRuns, [3, 16], 'highlight');
  assert.equal(result.map(run => run.text).join(''), 'plain both linked');
  assert.deepEqual([...new Set(result.filter(run => run.link).map(run => run.link.href))], ['#anchor-target', 'https://example.test']);
  assert.equal(result.filter(run => run.highlight).map(run => run.text).join(''), 'in both linke');
});

test('linking and unlinking only the selection preserves surrounding highlights', () => {
  const linked = transformRuns(fixture.mixedRuns, [0, 5], 'link', 'https://new.test');
  assert.equal(linked[0].link.href, 'https://new.test');
  assert.equal(linked.filter(run => run.highlight).map(run => run.text).join(''), 'both');
  const unlinked = transformRuns(linked, [6, 10], 'unlink');
  assert.equal(unlinked.filter(run => run.highlight && run.link === null).map(run => run.text).join(''), 'both');
});

test('nested editable markup serialises without discarding highlight or link formatting', () => {
  const root = inflate(fixture.nestedEditable);
  const runs = runsFromElement(root);
  assert.deepEqual(runs, [
    { text: 'nested', highlight: true, link: { href: '#anchor-target' } },
    { text: 'line two', highlight: false, link: null },
  ]);
});

test('paste inserts clipboard plain text rather than clipboard HTML', () => {
  let prevented = false, inserted;
  const event = { preventDefault: () => prevented = true, clipboardData: { getData: type => type === 'text/plain' ? 'safe text' : '<mark>unsafe</mark>' } };
  insertPlainText(event, { execCommand: (...args) => inserted = args });
  assert.equal(prevented, true);
  assert.deepEqual(inserted, ['insertText', false, 'safe text']);
});

test('Enter in a list item splits rich runs at the caret', () => {
  const [before, after] = splitListRuns(fixture.mixedRuns, 8);
  assert.equal(before.map(run => run.text).join(''), 'plain bo');
  assert.equal(after.map(run => run.text).join(''), 'th linked');
  assert.equal(after[0].highlight, true);
  assert.equal(after[0].link.href, '#anchor-target');
});

test('Tab from the final table cell inserts an ordinary row before totals', () => {
  const block = { columns: [{}, {}], rows: [{ id: 'ordinary' }, { id: 'total', isTotal: true }] };
  let id = 0;
  assert.equal(insertTableRowBeforeTotals(block, prefix => `${prefix}-${++id}`), 1);
  assert.deepEqual(block.rows.map(row => row.id), ['ordinary', 'row-1', 'total']);
  assert.equal(block.rows[1].cells.length, 2);
});

test('step deletion requires confirmation and protects the final step', () => {
  const steps = [{ title: 'One' }, { title: 'Two' }];
  let prompt;
  assert.equal(confirmStepDeletion(steps, 1, value => (prompt = value, false)), false);
  assert.equal(prompt, 'Delete Step 2: Two?');
  assert.equal(confirmStepDeletion([steps[0]], 0, () => true), false);
});

test('undo and redo treat a debounced input commit as one operation', async () => {
  const history = new History({ text: 'before' });
  await new Promise(resolve => setTimeout(() => { history.commit({ text: 'after' }); resolve(); }, 5));
  assert.deepEqual(history.undo(), { text: 'before' });
  assert.deepEqual(history.redo(), { text: 'after' });
});

test('link dialog accepts safe destinations and rejects invalid external input', () => {
  assert.equal(validatedLink('https://example.test/path', '#anchor-a'), 'https://example.test/path');
  assert.equal(validatedLink('', '#anchor-a'), '#anchor-a');
  assert.equal(validatedLink('javascript:alert(1)', '#anchor-a'), null);
});

test('stale pagination generations cannot publish their result', () => {
  const gate = new GenerationGate(), first = gate.next(), second = gate.next();
  assert.equal(gate.isCurrent(first), false);
  assert.equal(gate.isCurrent(second), true);
});

test('focus restoration focuses without scrolling and restores the text range', () => {
  const node = text('nested text'); let focusOptions, selected;
  const element = { focus: options => focusOptions = options };
  const range = { setStart: (...args) => range.start = args, setEnd: (...args) => range.end = args };
  const environment = {
    NodeFilter: { SHOW_TEXT: 4 },
    document: { createTreeWalker: () => ({ nextNode: (() => { let done = false; return () => done ? null : (done = true, node); })() }), createRange: () => range },
    getSelection: () => ({ removeAllRanges() {}, addRange: value => selected = value }),
  };
  assert.equal(restoreTextSelection(element, [2, 8], environment), true);
  assert.deepEqual(focusOptions, { preventScroll: true });
  assert.deepEqual(range.start, [node, 2]); assert.deepEqual(range.end, [node, 8]);
  assert.equal(selected, range);
});
