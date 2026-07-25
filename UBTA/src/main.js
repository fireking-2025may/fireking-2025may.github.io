import { seedDocument, normaliseDocument, normaliseRuns, safeHref, safeImageSrc, transactionProposals } from './state/schema.js';
import { History } from './state/history.js';

const history = new History(structuredClone(seedDocument));
let state = history.value;
let currentPage = 1;
let timer;
const $ = selector => document.querySelector(selector);
const escapeHTML = value => String(value).replace(/[&<>"']/g, character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[character]);
const textRuns = runs => runs.map(run => run.text).join('');

function hasReview(group) {
  return group.blocks.some(block => {
    const runs = block.runs || block.items?.flatMap(item => item.runs) || block.rows?.flatMap(row => row.cells.flatMap(cell => cell.runs)) || [];
    return runs.some(run => run.highlight);
  });
}

function header() {
  const element = document.createElement('header');
  element.className = 'page-header';
  element.setAttribute('aria-label', 'UBTA Accountants report header');
  element.innerHTML = '<span class="header-brand">UBTA</span><span class="header-details">UBTA Accountants Ltd <b aria-hidden="true">·</b> Practical tax advice</span>';
  return element;
}

function sheet(kind) {
  const element = document.createElement('section');
  element.className = `sheet-source ${kind}`;
  element.append(header());
  return element;
}

function footer(element, page, total) {
  const item = document.createElement('footer');
  item.className = 'page-footer';
  item.textContent = `Page ${page} of ${total}`;
  element.append(item);
}

function runsHTML(runs = []) {
  return runs.map(run => {
    let text = escapeHTML(run.text);
    if (run.link) text = `<a href="${escapeHTML(run.link.href)}" target="_blank" rel="noopener">${text}</a>`;
    return run.highlight ? `<mark>${text}</mark>` : text;
  }).join('');
}

function editableRuns(runs, attributes = '') {
  return `<span class="editable-runs" contenteditable="true" spellcheck="true" ${attributes}>${runsHTML(runs)}</span>`;
}

function editable(block) {
  const element = document.createElement('div');
  element.className = 'editable-block';
  element.dataset.blockId = block.id;
  element.dataset.type = block.type;
  element.dataset.level = block.level || '';
  element.lang = 'en-GB';

  if (['heading', 'paragraph'].includes(block.type)) {
    element.contentEditable = 'true';
    element.spellcheck = true;
    element.innerHTML = runsHTML(block.runs);
  } else if (block.type.endsWith('List')) {
    element.contentEditable = 'true';
    element.classList.add('list-block');
    element.dataset.listType = block.type;
    element.innerHTML = block.items.map(item => `<div class="list-item" data-item-id="${item.id}" data-level="${item.level}">${runsHTML(item.runs)}</div>`).join('');
  } else if (block.type === 'table') {
    element.classList.add('table-block');
    element.innerHTML = `<table><caption>${editableRuns([{ text: block.caption }], 'data-table-caption')}</caption><colgroup>${block.columns.map(column => `<col style="width:${column.width}%">`).join('')}</colgroup><thead><tr>${block.columns.map((column, index) => `<th scope="col">${editableRuns([{ text: column.heading }], `data-column="${index}"`)}${index < block.columns.length - 1 ? `<button class="column-resizer" data-resize="${index}" aria-label="Resize ${escapeHTML(column.heading)} column" title="Drag to resize column"></button>` : ''}</th>`).join('')}</tr></thead><tbody>${block.rows.map((row, rowIndex) => `<tr class="${row.isTotal ? 'total-row' : ''}" data-row="${rowIndex}">${row.cells.map((cell, columnIndex) => `<td class="${block.columns[columnIndex].numeric ? 'numeric' : ''}">${editableRuns(cell.runs, `data-cell="${columnIndex}"`)}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
  } else if (block.type === 'image') {
    element.classList.add('image-block');
    element.innerHTML = block.src
      ? `<figure><img src="${escapeHTML(block.src)}" alt="${escapeHTML(block.alt)}" style="width:${block.width}%"><figcaption>${escapeHTML(block.caption)}</figcaption></figure>`
      : '<p class="image-error" role="status">Image unavailable. Add a valid HTTPS image or upload an image.</p>';
  }
  return element;
}

function cover() {
  const element = sheet('cover');
  const meta = state.meta;
  element.insertAdjacentHTML('beforeend', `<div class="cover-kicker">${escapeHTML(meta.documentType)} · ${escapeHTML(meta.status)}</div><h1>${escapeHTML(meta.clientName)}</h1><h2>${escapeHTML(meta.projectTitle)} — ${escapeHTML(meta.subtitle)}</h2><dl class="meta-grid"><dt>Document</dt><dd>${escapeHTML(meta.documentType)}</dd><dt>Date</dt><dd>${escapeHTML(meta.date)}</dd><dt>Version</dt><dd>${escapeHTML(meta.version)}</dd><dt>Adviser</dt><dd>${escapeHTML(meta.adviser)}</dd><dt>Status</dt><dd>${escapeHTML(meta.status)}</dd></dl>`);
  return element;
}

function contents(pageMap = {}) {
  const element = sheet('contents');
  const entries = state.sections.map(group => ({ group, label: group.title, className: '' }))
    .concat(transactionProposals(state).map(proposal => ({ group: state.steps.find(step => step.id === proposal.stepId), label: proposal.title, className: 'toc-step', anchor: proposal.anchor })));
  element.insertAdjacentHTML('beforeend', `<h1 class="doc-title">Contents</h1><ol class="toc">${entries.map(entry => { const anchor = entry.anchor || `anchor-${entry.group.id}`; return `<li class="${entry.className} ${hasReview(entry.group) ? 'toc-review' : ''}" data-anchor="${anchor}"><a href="#${anchor}">${escapeHTML(entry.label)}</a><span class="dots"></span><span class="toc-page">${pageMap[anchor] || '—'}</span></li>`; }).join('')}</ol>`);
  return element;
}

function groupSheet(group, title, kind) {
  const element = sheet(kind);
  element.id = `anchor-${group.id}`;
  element.tabIndex = -1;
  const label = kind === 'step' ? title.match(/^Step \d+/)?.[0] : '';
  element.insertAdjacentHTML('beforeend', kind === 'step' ? `<div class="step-heading"><div class="step-label">${escapeHTML(label)}</div><input class="step-title" data-step-title="${group.id}" value="${escapeHTML(group.title)}" aria-label="${escapeHTML(label)} title"></div>` : `<h1 class="section-title">${escapeHTML(title)}</h1>`);
  const body = document.createElement('div');
  body.className = 'document-body';
  group.blocks.forEach(block => body.append(editable(block)));
  element.append(body);
  return element;
}

async function render() {
  const source = document.createElement('main');
  source.append(cover(), contents(window.__pageMap || {}));
  state.sections.forEach(group => source.append(groupSheet(group, group.title, 'section')));
  state.steps.forEach((group, index) => source.append(groupSheet(group, `Step ${index + 1}. ${group.title}`, 'step')));
  const previewer = new Paged.Previewer();
  const target = $('#preview');
  await previewer.preview(source, [], target);
  const pages = [...target.children];
  const pageMap = {};
  pages.forEach((page, index) => {
    footer(page.querySelector('.sheet-source'), index + 1, pages.length);
    page.dataset.pageNumber = index + 1;
    page.querySelectorAll('[id^="anchor-"]').forEach(anchor => { pageMap[anchor.id] = index + 1; });
  });
  window.__pageMap = pageMap;
  target.querySelectorAll('.toc li').forEach(item => { item.querySelector('.toc-page').textContent = pageMap[item.dataset.anchor] || '—'; });
  bindEditing();
  $('#total').textContent = pages.length;
  currentPage = Math.min(currentPage, pages.length);
  $('#current').textContent = currentPage;
}

function findBlock(id) {
  for (const group of [...state.sections, ...state.steps]) {
    const block = group.blocks.find(item => item.id === id);
    if (block) return block;
  }
}

function commit(mutator) {
  state = structuredClone(state);
  mutator();
  state = normaliseDocument(state);
  history.commit(state);
  state = history.value;
  render();
}

// Translate the browser's small, approved inline vocabulary back into schema runs.
// Text from all other elements is retained, but unsafe markup and URLs are discarded.
function runsFromElement(element) {
  const runs = [];
  const walk = (node, highlight = false, link = null) => {
    if (node.nodeType === Node.TEXT_NODE) {
      runs.push({ text: node.nodeValue || '', highlight, link: link ? { href: link } : null });
      return;
    }
    if (node.nodeType !== Node.ELEMENT_NODE) return;
    const tag = node.tagName.toLowerCase();
    const nextHighlight = highlight || tag === 'mark';
    const nextLink = tag === 'a' ? safeHref(node.getAttribute('href')) : link;
    node.childNodes.forEach(child => walk(child, nextHighlight, nextLink));
  };
  element.childNodes.forEach(node => walk(node));
  return normaliseRuns(runs);
}

function syncBlock(element) {
  const id = element.dataset.blockId;
  commit(() => {
    const block = findBlock(id);
    if (!block) return;
    if (block.type.endsWith('List')) {
      block.items = [...element.querySelectorAll('.list-item')].map((item, index) => ({ id: item.dataset.itemId || `${id}-item-${index}`, level: +item.dataset.level || 1, runs: runsFromElement(item) }));
    } else if (block.type === 'table') {
      block.caption = element.querySelector('[data-table-caption]').textContent;
      element.querySelectorAll('[data-column]').forEach(item => { block.columns[+item.dataset.column].heading = item.textContent; });
      element.querySelectorAll('[data-row]').forEach(row => row.querySelectorAll('[data-cell]').forEach(cell => { block.rows[+row.dataset.row].cells[+cell.dataset.cell].runs = runsFromElement(cell); }));
    } else {
      block.runs = runsFromElement(element);
    }
  });
}

function bindEditing() {
  document.querySelectorAll('.editable-block').forEach(element => {
    element.addEventListener('focusin', event => { window.activeBlock = element.dataset.blockId; window.activeEditable = event.target.closest('[contenteditable="true"]'); });
    element.addEventListener('input', () => { clearTimeout(timer); timer = setTimeout(() => syncBlock(element), 450); });
    element.addEventListener('paste', event => { if (!event.target.closest('[contenteditable="true"]')) return; event.preventDefault(); document.execCommand('insertText', false, event.clipboardData.getData('text/plain')); });
    element.addEventListener('keydown', listKeys);
  });
  document.querySelectorAll('[data-step-title]').forEach(input => input.addEventListener('change', () => commit(() => { const step = state.steps.find(item => item.id === input.dataset.stepTitle); if (step) step.title = input.value.trim() || 'Untitled step'; })));
  bindResizers();
  document.querySelectorAll('.toc a').forEach(anchor => { anchor.onclick = event => { event.preventDefault(); const destination = document.querySelector(anchor.getAttribute('href')); destination?.closest('.pagedjs_page')?.scrollIntoView({ behavior: 'smooth', block: 'center' }); destination?.focus(); }; });
}

function bindResizers() {
  document.querySelectorAll('.column-resizer').forEach(handle => {
    handle.addEventListener('keydown', event => {
      if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
      event.preventDefault();
      const tableBlock = handle.closest('.table-block');
      const index = +handle.dataset.resize;
      const delta = event.key === 'ArrowLeft' ? -2 : 2;
      commit(() => {
        const block = findBlock(tableBlock.dataset.blockId);
        if (block.columns[index].width + delta < 10 || block.columns[index + 1].width - delta < 10) return;
        block.columns[index].width += delta;
        block.columns[index + 1].width -= delta;
      });
    });
    handle.addEventListener('pointerdown', event => {
    event.preventDefault();
    const tableBlock = handle.closest('.table-block');
    const blockId = tableBlock.dataset.blockId;
    const index = +handle.dataset.resize;
    const startX = event.clientX;
    const block = findBlock(blockId);
    const start = block.columns.map(column => column.width);
    const move = moveEvent => {
      const delta = ((moveEvent.clientX - startX) / tableBlock.offsetWidth) * 100;
      const amount = Math.max(-start[index] + 10, Math.min(start[index + 1] - 10, delta));
      const columns = tableBlock.querySelectorAll('col');
      columns[index].style.width = `${start[index] + amount}%`;
      columns[index + 1].style.width = `${start[index + 1] - amount}%`;
    };
    const up = upEvent => {
      removeEventListener('pointermove', move);
      removeEventListener('pointerup', up);
      const delta = ((upEvent.clientX - startX) / tableBlock.offsetWidth) * 100;
      commit(() => { const current = findBlock(blockId); const amount = Math.max(-start[index] + 10, Math.min(start[index + 1] - 10, delta)); current.columns[index].width = start[index] + amount; current.columns[index + 1].width = start[index + 1] - amount; });
    };
    addEventListener('pointermove', move);
    addEventListener('pointerup', up, { once: true });
    });
  });
}

function listKeys(event) {
  const element = event.currentTarget;
  if (!element.dataset.listType) return;
  const item = event.target.closest('.list-item');
  if (event.key === 'Enter') {
    event.preventDefault();
    if (!item.textContent.trim()) { commit(() => { const block = findBlock(element.dataset.blockId); block.type = 'paragraph'; block.runs = []; delete block.items; }); return; }
    const next = document.createElement('div'); next.className = 'list-item'; next.dataset.itemId = `item-${crypto.randomUUID()}`; next.dataset.level = item.dataset.level; next.contentEditable = 'true'; item.after(next); next.focus();
  }
}

function selectedOffsets(element) {
  const selection = getSelection();
  if (!selection.rangeCount || !element?.contains(selection.anchorNode) || !element.contains(selection.focusNode)) return null;
  const range = selection.getRangeAt(0);
  const before = document.createRange(); before.selectNodeContents(element); before.setEnd(range.startContainer, range.startOffset);
  const start = before.toString().length;
  return [start, start + range.toString().length];
}

function runsAt(runs, position) { let count = 0; return runs.find(run => (count += run.text.length) > position) || runs.at(-1) || {}; }
function transformSelection(kind, value) {
  const element = window.activeEditable || document.querySelector(`[data-block-id="${window.activeBlock}"]`);
  const offsets = selectedOffsets(element);
  if (!offsets || offsets[0] === offsets[1]) return;
  commit(() => {
    const block = findBlock(window.activeBlock); if (!block?.runs) return;
    const text = textRuns(block.runs); const [start, end] = offsets; const current = runsAt(block.runs, start);
    block.runs = normaliseRuns([{ text: text.slice(0, start) }, { text: text.slice(start, end), highlight: kind === 'highlight' ? !current.highlight : current.highlight, link: kind === 'link' ? { href: value } : kind === 'unlink' ? null : current.link }, { text: text.slice(end) }]);
  });
}

function command(action) {
  const id = window.activeBlock;
  if (action === 'undo') { state = history.undo(); return render(); }
  if (action === 'redo') { state = history.redo(); return render(); }
  if (action === 'addStep') return commit(() => state.steps.push({ id: `step-${crypto.randomUUID()}`, title: 'New transaction step', blocks: [{ id: `block-${crypto.randomUUID()}`, type: 'paragraph', runs: [{ text: 'Describe the transaction step.' }] }] }));
  if (action === 'moveStepUp' || action === 'moveStepDown' || action === 'deleteStep') return manageStep(action);
  if (action === 'addImage') return addImage();
  if (action === 'highlight') return transformSelection('highlight');
  if (action === 'link') { const href = prompt('Link destination (https:, http:, mailto: or #anchor)'); if (href && safeHref(href)) transformSelection('link', href); else if (href) alert('That link destination is not permitted.'); return; }
  if (action === 'unlink') return transformSelection('unlink');
  if (!id) return;
  commit(() => { const block = findBlock(id); if (!block) return; if (action.startsWith('heading')) { block.type = 'heading'; block.level = +action.at(-1); block.runs ||= block.items?.flatMap(item => item.runs) || []; } else if (action === 'body') { block.type = 'paragraph'; block.runs ||= block.items?.flatMap(item => item.runs) || []; } else if (['bulletList', 'numberList'].includes(action)) { const runs = block.runs || [{ text: 'List item' }]; block.type = action; block.items = [{ id: `${id}-item`, level: 1, runs }]; delete block.runs; } });
}

function activeStepIndex() { return state.steps.findIndex(step => step.blocks.some(block => block.id === window.activeBlock)); }
function manageStep(action) {
  const index = activeStepIndex();
  if (index < 0) return;
  commit(() => { if (action === 'deleteStep') state.steps.splice(index, 1); else { const target = action === 'moveStepUp' ? index - 1 : index + 1; if (target >= 0 && target < state.steps.length) [state.steps[index], state.steps[target]] = [state.steps[target], state.steps[index]]; } });
}

function addImage() {
  const input = $('#image-upload');
  input.value = '';
  input.onchange = () => {
    const file = input.files[0]; if (!file || !/^image\/(png|jpeg|gif|webp)$/.test(file.type) || file.size > 5_000_000) { alert('Choose a PNG, JPEG, GIF or WebP image smaller than 5 MB.'); return; }
    const alt = prompt('Describe the image for people who cannot see it:')?.trim(); if (!alt) { alert('Alternative text is required.'); return; }
    const reader = new FileReader(); reader.onload = () => { if (!safeImageSrc(reader.result)) return; const index = Math.max(0, activeStepIndex()); commit(() => state.steps[index].blocks.push({ id: `image-${crypto.randomUUID()}`, type: 'image', src: reader.result, alt, caption: '', width: 100 })); }; reader.readAsDataURL(file);
  };
  input.click();
}

document.querySelectorAll('[data-command]').forEach(button => { button.onclick = () => command(button.dataset.command); });
$('#style').onchange = event => command(event.target.value);
function go(page) { const pages = [...document.querySelectorAll('.pagedjs_page')]; currentPage = Math.max(1, Math.min(pages.length, page)); $('#current').textContent = currentPage; pages[currentPage - 1]?.scrollIntoView({ behavior: 'smooth', block: 'center' }); }
$('#prev').onclick = () => go(currentPage - 1); $('#next').onclick = () => go(currentPage + 1);
addEventListener('keydown', event => { if (event.altKey && event.key === 'ArrowLeft') go(currentPage - 1); if (event.altKey && event.key === 'ArrowRight') go(currentPage + 1); });
render();
