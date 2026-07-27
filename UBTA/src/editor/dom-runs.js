import { normaliseRuns, safeHref } from '../state/schema.js';

export function runsFromElement(element) {
  const output = [];
  const walk = (node, highlighted = false, href = null) => {
    if (node.nodeType === 3) {
      output.push({ text: node.nodeValue || '', highlight: highlighted, link: href ? { href } : null });
      return;
    }
    if (node.nodeType !== 1) return;
    const tag = node.tagName.toLowerCase();
    if (tag === 'br') {
      output.push({ text: '\n', highlight: highlighted, link: href ? { href } : null });
      return;
    }
    const nextHighlight = highlighted || tag === 'mark';
    const nextHref = tag === 'a' ? (safeHref(node.getAttribute('href')) || href) : href;
    [...node.childNodes].forEach(child => walk(child, nextHighlight, nextHref));
    if (['div', 'p'].includes(tag) && node.nextSibling && output.at(-1)?.text?.at(-1) !== '\n') {
      output.push({ text: '\n', highlight: highlighted, link: href ? { href } : null });
    }
  };
  [...element.childNodes].forEach(node => walk(node));
  return normaliseRuns(output);
}

export function transformRuns(runs, [start, end], kind, value) {
  if (!runs || start === end) return runs;
  let position = 0;
  const selected = runs.filter(run => {
    const runStart = position;
    position += run.text.length;
    return runStart < end && position > start;
  });
  const addHighlight = kind === 'highlight' && !selected.every(run => run.highlight);
  position = 0;
  const output = [];
  for (const run of runs) {
    const runEnd = position + run.text.length;
    if (runEnd <= start || position >= end) {
      output.push({ ...run });
      position = runEnd;
      continue;
    }
    const from = Math.max(start, position);
    const to = Math.min(end, runEnd);
    if (position < from) output.push({ ...run, text: run.text.slice(0, from - position) });
    if (to > from) {
      const part = { ...run, text: run.text.slice(from - position, to - position) };
      if (kind === 'highlight') part.highlight = addHighlight;
      if (kind === 'link') part.link = { href: value };
      if (kind === 'unlink') part.link = null;
      output.push(part);
    }
    if (to < runEnd) output.push({ ...run, text: run.text.slice(to - position) });
    position = runEnd;
  }
  return normaliseRuns(output);
}

export function insertPlainText(event, documentObject = document) {
  event.preventDefault();
  const text = event.clipboardData.getData('text/plain');
  documentObject.execCommand('insertText', false, text);
  return text;
}
