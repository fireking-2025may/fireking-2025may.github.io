import {
  seedDocument,
  normaliseDocument,
  normaliseRuns,
  safeHref,
  transactionProposals,
  hasReview,
  newId,
  stableAnchor,
  METADATA_KEYS,
} from './state/schema.js';
import { validateImageInput } from './state/image-model.js';
import {
  parseTableNumber,
  formatTableNumber,
  tableColumnFormat,
  recalculateTableTotals,
} from './state/table-model.js';
import {
  addStep as addStepOperation,
  moveStep as moveStepOperation,
  deleteStep as deleteStepOperation,
  addAppendix as addAppendixOperation,
  moveAppendix as moveAppendixOperation,
  deleteAppendix as deleteAppendixOperation,
  insertBlock as insertBlockOperation,
  moveBlock as moveBlockOperation,
  deleteBlock as deleteBlockOperation,
  insertTableRow as insertTableRowOperation,
  moveTableRow as moveTableRowOperation,
  deleteTableRow as deleteTableRowOperation,
  insertTableColumn as insertTableColumnOperation,
  moveTableColumn as moveTableColumnOperation,
  deleteTableColumn as deleteTableColumnOperation,
  setTableColumnFormat as setTableColumnFormatOperation,
  setTableColumnTotal as setTableColumnTotalOperation,
  convertBlockStyle as convertBlockStyleOperation,
} from './state/document-operations.js';
import { History } from './state/history.js';
import {
  readDraft,
  readRecovery,
  saveDraft,
  saveVersion,
  listVersions,
  deleteVersion,
  nextAvailableVersion,
  encodeSnapshot,
  snapshotFromLocation,
  updateSnapshotUrl,
} from './state/persistence.js';
import {
  runsFromElement,
  transformRuns,
  insertPlainText,
} from './editor/dom-runs.js';
import {
  validatedLink,
  containedSelectionOffsets,
  confirmStepDeletion,
  splitListRuns,
  removeEmptyListItem,
  restoreTextSelection,
} from './editor/interactions.js';
import { RepaginationScheduler } from './editor/repagination.js';
import { PrintLifecycle } from './editor/printing.js';
import {
  routeInsertionCommand,
  canApplyBlockStyle,
  blockStyleChoices,
} from './editor/command-routing.js';
import { renderList } from './editor/list-rendering.js';
import { blockTypeLabel } from './editor/block-deletion.js';
import {
  insertionIndex,
  insertionContextFromPoint,
  canOpenBlankSpaceInsertion,
} from './editor/insertion-context.js';
import {
  EncryptedTemplateLoader,
  loadDefaultTemplates,
  templateBlocks,
} from './editor/default-templates.js';
import { defaultTemplateEnvelope } from './editor/default-template-data.js';
import {
  followEditorLink,
  handleEditableLinkClick,
  createEditableLinkClickHandler,
} from './editor/link-actions.js';
import { bindCallouts } from './editor/callouts.js';
import { NavigationHistory } from './editor/navigation-history.js';
import { ExcelEditor } from './editor/excel-editor.js';
import { stepHasMovements } from './state/excel-model.js';

let initial = seedDocument,
  loadedRevision = 0;
try {
  const draft = readDraft();
  initial =
    snapshotFromLocation() ||
    draft?.document ||
    readRecovery()?.document ||
    seedDocument;
  loadedRevision = draft?.revision || 0;
} catch (error) {
  setTimeout(() => alert(error.message), 0);
}
const history = new History(initial);
let state = history.checkout();
let currentPage = 1;
let dirty = false,
  autosaveTimer = null;
export const editorSelection = {
  activeGroupId: null,
  activeGroupType: null,
  activeStepId: null,
  activeBlockId: null,
  container: null,
  row: null,
  column: null,
  rowId: null,
  columnId: null,
  cellId: null,
  blockType: null,
  offsets: null,
  restoreFocus: false,
};
let activeEditor = 'steps',
  excelController = null;
const pending = new Map(),
  pendingStepTitles = new Map();
const $ = (s) => document.querySelector(s);
const esc = (x) =>
  String(x).replace(
    /[&<>"']/g,
    (c) =>
      ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[
        c
      ],
  );
const runText = (runs) => (runs || []).map((r) => r.text).join('');
const runsHTML = (runs) =>
  (runs || [])
    .map((r) => {
      let text = esc(r.text).replace(/\n/g, '<br>');
      if (r.link) text = `<a href="${esc(r.link.href)}">${text}</a>`;
      return r.highlight ? `<mark>${text}</mark>` : text;
    })
    .join('');
const editableRuns = (runs, attrs = '') =>
  `<span class="editable-runs" contenteditable="true" spellcheck="true" ${attrs}>${runsHTML(runs)}</span>`;
const allGroups = () => [
  ...state.sections,
  ...state.steps,
  ...state.appendices,
];
const findBlock = (id) => {
  for (const group of allGroups()) {
    const block = group.blocks.find((x) => x.id === id);
    if (block) return { group, block };
  }
  return {};
};
const selectedEditableGroup = () =>
  (editorSelection.activeGroupType === 'appendix'
    ? state.appendices
    : state.steps
  ).find((x) => x.id === editorSelection.activeGroupId);
const appendixLabel = (index) => `Appendix ${String.fromCharCode(65 + index)}`;

function brandHTML() {
  return `<span class="brand-mark" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i></span><span class="brand-wordmark">UBTA</span>`;
}
function header() {
  const x = document.createElement('header');
  x.className = 'page-header';
  x.innerHTML = `<span class="header-brand">${brandHTML()}</span>`;
  return x;
}
function sheet(kind) {
  const x = document.createElement('section');
  x.className = `sheet-source ${kind}`;
  x.append(header());
  return x;
}
function footer(x, n, total) {
  const f = document.createElement('footer');
  f.className = 'page-footer';
  f.textContent = `Page ${n} of ${total}`;
  x.append(f);
}
function blockElement(block) {
  const x = document.createElement('div');
  x.className = 'editable-block';
  x.dataset.blockId = block.id;
  x.dataset.type = block.type;
  x.dataset.level = block.level || '';
  x.draggable = false;
  x.tabIndex = -1;
  if (['heading', 'paragraph'].includes(block.type))
    x.innerHTML = editableRuns(block.runs, 'data-container="block"');
  else if (block.type.endsWith('List')) {
    x.classList.add('list-block');
    x.dataset.listType = block.type;
    x.innerHTML = renderList(block, runsHTML, esc);
  } else if (block.type === 'table') {
    x.classList.add('table-block');
    recalculateTableTotals(block);
    const cellHTML = (cell, row, ri, ci) => {
      const format = tableColumnFormat(block.columns[ci]),
        numeric = format !== 'text',
        value = runText(cell.runs),
        parsed = numeric ? parseTableNumber(value) : null,
        display = parsed === null ? value : formatTableNumber(parsed, format),
        attrs = `data-container="cell" data-row="${ri}" data-column="${ci}" data-row-id="${esc(row.id)}" data-column-id="${esc(block.columns[ci].id)}" data-cell-id="${esc(cell.id)}"`;
      return `<td class="format-${format}">${row.isTotal && numeric ? `<output ${attrs}>${esc(display)}</output>` : editableRuns(numeric ? [{ text: display }] : cell.runs, attrs)}</td>`;
    };
    x.innerHTML = `<table><caption>${editableRuns(block.captionRuns, 'data-container="caption"')}</caption><colgroup>${block.columns.map((c) => `<col style="width:${c.width}%">`).join('')}</colgroup><thead><tr>${block.columns.map((c, i) => `<th class="format-${tableColumnFormat(c)}">${editableRuns(c.headingRuns, `data-container="heading" data-column="${i}"`)}</th>`).join('')}</tr></thead><tbody>${block.rows.map((r, ri) => `<tr class="${r.isTotal ? 'total-row' : ''}" data-row="${ri}">${r.cells.map((c, ci) => cellHTML(c, r, ri, ci)).join('')}</tr>`).join('')}</tbody></table>`;
  } else {
    x.classList.add('image-block');
    x.innerHTML = block.src
      ? `<figure><img src="${esc(block.src)}" alt="${esc(block.alt || '')}" style="width:${block.width}%"><figcaption>${editableRuns(block.captionRuns, 'data-container="image-caption"')}</figcaption></figure>`
      : '<p class="image-error">Image unavailable.</p>';
  }
  x.insertAdjacentHTML(
    'afterbegin',
    `<div class="block-controls" contenteditable="false"><button type="button" data-block-move="before" aria-label="Move block before">↑</button><button type="button" data-block-move="after" aria-label="Move block after">↓</button><button type="button" class="delete-block" data-block-delete aria-label="Delete ${blockTypeLabel(block.type)}">Delete</button></div>`,
  );
  return x;
}
function groupSheet(group, title, kind) {
  const x = sheet(kind);
  x.id = stableAnchor(group);
  x.tabIndex = -1;
  if (['step', 'appendix'].includes(kind)) {
    x.dataset.editableGroupId = group.id;
    x.dataset.groupType = kind;
    if (group.id === editorSelection.activeGroupId)
      x.classList.add('is-selected');
    x.innerHTML += `<div class="step-heading"><div class="step-label">${esc(kind === 'step' ? title.match(/^Step \d+/)?.[0] || '' : title.split('. ')[0])}</div><input class="step-title" data-group-title="${group.id}" value="${esc(group.title)}" aria-label="${kind === 'step' ? 'Step' : 'Appendix'} title"><div class="step-title-print" aria-hidden="true">${esc(group.title)}</div></div>`;
  } else x.innerHTML += `<h1 class="section-title">${esc(title)}</h1>`;
  const body = document.createElement('div');
  body.className = 'document-body';
  group.blocks.forEach((b) => body.append(blockElement(b)));
  x.append(body);
  return x;
}
function cover() {
  const x = sheet('cover');
  x.innerHTML += `<h1><span>${esc(state.meta.clientName)}</span><span>${esc(state.meta.projectTitle)}</span></h1><p class="cover-subtitle">${esc(state.meta.subtitle)}</p><p class="cover-date">${esc(state.meta.date)}</p>`;
  return x;
}
function contentsEntries() {
  return [
    ...state.sections.map((group) => ({
      group,
      type: 'section',
      label: group.title,
    })),
    ...state.steps.map((group, index) => ({
      group,
      type: 'step',
      label: `Step ${index + 1}. ${group.title}`,
    })),
    ...state.appendices.map((group, index) => ({
      group,
      type: 'appendix',
      label: `${appendixLabel(index)}. ${group.title}`,
    })),
  ];
}
function contents(map = {}) {
  const x = sheet('contents'),
    entries = contentsEntries();
  x.innerHTML +=
    '<h1 class="doc-title">Contents</h1><ol class="toc">' +
    entries
      .map(
        ({ group, type, label }) =>
          `<li class="toc-${type} ${hasReview(group) ? 'toc-review' : ''}"><a href="#${stableAnchor(group)}">${esc(label)}</a><span class="dots"></span><span class="toc-page">${map[stableAnchor(group)] || '—'}</span></li>`,
      )
      .join('') +
    '</ol>';
  return x;
}
function captureSelection() {
  const offsets = containedSelectionOffsets(
    activeEditable(),
    getSelection(),
    document,
  );
  if (offsets) editorSelection.offsets = offsets;
}
function captureView() {
  captureSelection();
  return {
    currentPage,
    scrollX: window.scrollX,
    scrollY: window.scrollY,
    selection: structuredClone(editorSelection),
  };
}
async function restoreView(view) {
  setCurrentPage(view.currentPage, { scroll: false, focus: false });
  Object.assign(editorSelection, view.selection);
  if (editorSelection.restoreFocus)
    restoreTextSelection(activeEditable(), editorSelection.offsets);
  window.scrollTo(view.scrollX, view.scrollY);
}
async function paginate({ isCurrent }) {
  const source = document.createElement('main'),
    staging = document.createElement('div');
  staging.className = 'pagedjs_pages';
  source.append(cover(), contents(window.__pageMap || {}));
  state.sections.forEach((g) =>
    source.append(groupSheet(g, g.title, 'section')),
  );
  state.steps.forEach((g, i) =>
    source.append(groupSheet(g, `Step ${i + 1}. ${g.title}`, 'step')),
  );
  state.appendices.forEach((g, i) =>
    source.append(groupSheet(g, `${appendixLabel(i)}. ${g.title}`, 'appendix')),
  );
  await new Paged.Previewer().preview(source, [], staging);
  if (!isCurrent()) return false;
  const pages = [...staging.children],
    map = {};
  pages.forEach((p, i) => {
    footer(p.querySelector('.sheet-source'), i + 1, pages.length);
    p.dataset.pageNumber = String(i + 1);
    p.querySelectorAll('[id^=anchor-]').forEach((a) => (map[a.id] ??= i + 1));
    if (p.dataset.anchorId) map[p.dataset.anchorId] ??= i + 1;
  });
  if (!isCurrent()) return false;
  window.__pageMap = map;
  disconnectPageObserver();
  $('#preview').replaceChildren(...pages);
  bindEditing();
  observePages();
  updateControls();
  setCurrentPage(currentPage, { scroll: false, focus: false });
  $('#total').textContent = pages.length;
  rebuildPageJump(pages);
  return true;
}
const repagination = new RepaginationScheduler(paginate, {
  capture: captureView,
  restore: restoreView,
});
const navigationHistory = new NavigationHistory();
const render = ({ immediate = true } = {}) =>
  repagination.request({ immediate });

function activeEditable() {
  const id = editorSelection.activeBlockId;
  if (!id)
    return editorSelection.activeGroupId
      ? document.querySelector(
          `[data-group-title="${editorSelection.activeGroupId}"]`,
        )
      : null;
  const block = document.querySelector(`[data-block-id="${id}"]`);
  if (!block) return null;
  const c = editorSelection.container;
  if (c === 'block') return block.querySelector('[data-container=block]');
  if (c === 'item')
    return block.querySelector(
      `li[data-container="item"][data-item-id="${editorSelection.itemId}"]`,
    );
  if (c === 'caption') return block.querySelector('[data-container=caption]');
  if (c === 'image-caption')
    return block.querySelector('[data-container=image-caption]');
  if (c === 'heading')
    return block.querySelector(
      `[data-container=heading][data-column="${editorSelection.column}"]`,
    );
  if (c === 'cell')
    return (
      block.querySelector(
        `[data-container=cell][data-cell-id="${CSS.escape(editorSelection.cellId || '')}"]`,
      ) ||
      block.querySelector(
        `[data-container=cell][data-row="${editorSelection.row}"][data-column="${editorSelection.column}"]`,
      )
    );
  return block;
}
function setActive(target) {
  const group = target.closest('[data-editable-group-id]');
  if (group) {
    editorSelection.activeGroupId = group.dataset.editableGroupId;
    editorSelection.activeGroupType = group.dataset.groupType;
    editorSelection.activeStepId =
      group.dataset.groupType === 'step' ? group.dataset.editableGroupId : null;
  }
  const block = target.closest('[data-block-id]'),
    item = target.closest('li[data-container="item"]');
  editorSelection.activeBlockId = block?.dataset.blockId || null;
  editorSelection.blockType = block?.dataset.type || null;
  editorSelection.container =
    item?.dataset.container ||
    target.dataset.container ||
    (['heading', 'paragraph'].includes(editorSelection.blockType)
      ? 'block'
      : null);
  editorSelection.itemId = item?.dataset.itemId || null;
  editorSelection.row = target.dataset.row ?? null;
  editorSelection.column = target.dataset.column ?? null;
  editorSelection.rowId = target.dataset.rowId ?? null;
  editorSelection.columnId = target.dataset.columnId ?? null;
  editorSelection.cellId = target.dataset.cellId ?? null;
  editorSelection.restoreFocus = true;
  document
    .querySelectorAll('[data-editable-group-id]')
    .forEach((x) =>
      x.classList.toggle(
        'is-selected',
        x.dataset.editableGroupId === editorSelection.activeGroupId,
      ),
    );
  updateControls();
}
function syncElement(el, record = true) {
  if (!el?.isConnected) return false;
  const blockEl = el.closest('[data-block-id]'),
    found = findBlock(blockEl?.dataset.blockId);
  if (!found.block) return false;
  const before = structuredClone(state);
  const b = found.block,
    c = el.dataset.container;
  if (c === 'item') {
    const i = b.items.find((x) => x.id === el.dataset.itemId);
    if (i) i.runs = runsFromElement(el);
  } else if (c === 'caption') b.captionRuns = runsFromElement(el);
  else if (c === 'image-caption') b.captionRuns = runsFromElement(el);
  else if (c === 'heading')
    b.columns[+el.dataset.column].headingRuns = runsFromElement(el);
  else if (c === 'cell') {
    captureSelection();
    const row =
        b.rows.find((r) => r.id === el.dataset.rowId) ||
        b.rows[+el.dataset.row],
      columnIndex = b.columns.findIndex(
        (column) => column.id === el.dataset.columnId,
      ),
      cell =
        row?.cells.find((cell) => cell.id === el.dataset.cellId) ||
        row?.cells[columnIndex < 0 ? +el.dataset.column : columnIndex];
    if (cell) cell.runs = runsFromElement(el);
    editorSelection.cellId = cell?.id || editorSelection.cellId;
  } else {
    const peers = [
      ...document.querySelectorAll(
        `[data-block-id="${b.id}"] [data-container="block"]`,
      ),
    ];
    b.runs =
      peers.length > 1 ? peers.flatMap(runsFromElement) : runsFromElement(el);
  }
  state = normaliseDocument(state);
  if (record) history.commit(state);
  else history.replace(state);
  const changed = JSON.stringify(before) !== JSON.stringify(state);
  if (changed) markDirty();
  return changed;
}
function schedule(el) {
  const key = targetKey(el),
    old = pending.get(key);
  if (old) clearTimeout(old.timer);
  pending.set(key, {
    el,
    timer: setTimeout(() => {
      const p = pending.get(key);
      if (!p || p.el !== el || !el.isConnected) return;
      syncElement(el);
      pending.delete(key);
      updateControls();
      render({ immediate: false });
    }, 450),
  });
}
const targetKey = (el) =>
  [
    el.closest('[data-block-id]')?.dataset.blockId,
    el.dataset.container,
    el.dataset.itemId,
    el.dataset.row,
    el.dataset.column,
  ].join(':');
function flushTarget(el, record = true) {
  const key = targetKey(el),
    p = pending.get(key);
  if (!p) return;
  clearTimeout(p.timer);
  pending.delete(key);
  if (p.el.isConnected) syncElement(p.el, record);
}
function syncStepTitle(el, { renderAfter = true } = {}) {
  const pendingTitle = pendingStepTitles.get(el.dataset.groupTitle);
  if (pendingTitle) clearTimeout(pendingTitle.timer);
  pendingStepTitles.delete(el.dataset.groupTitle);
  if (!el.isConnected) return false;
  const group = allGroups().find((x) => x.id === el.dataset.groupTitle),
    title = el.value.trim() || 'Untitled';
  if (!group || group.title === title) return false;
  group.title = title;
  history.commit(state);
  markDirty();
  updateControls();
  if (renderAfter) render({ immediate: false });
  return true;
}
function flushAll(record = true) {
  [...pending.values()].forEach((p) => {
    clearTimeout(p.timer);
    if (p.el.isConnected) syncElement(p.el, record);
  });
  pending.clear();
  [...pendingStepTitles.values()].forEach((p) =>
    syncStepTitle(p.el, { renderAfter: false }),
  );
}
const hasPendingInputs = () => pending.size > 0 || pendingStepTitles.size > 0;
function markDirty() {
  dirty = true;
  const indicator = $('#save-state');
  if (indicator) {
    indicator.textContent = 'Unsaved changes';
    indicator.className = 'save-state dirty';
  }
  clearTimeout(autosaveTimer);
  autosaveTimer = setTimeout(() => persistDraft(), 1500);
}
function commit(mutator, { focus = true } = {}) {
  flushAll();
  const before = structuredClone(state);
  mutator();
  state = normaliseDocument(state);
  history.replace(before);
  history.commit(state);
  state = history.checkout();
  markDirty();
  editorSelection.restoreFocus = focus;
  return render({ restore: focus });
}
function applyDocumentOperation(result, { focus = true } = {}) {
  if (!result?.changed) return false;
  const before = state;
  history.replace(before);
  history.commit(result.document);
  state = history.checkout();
  markDirty();
  if (result.selectedGroupId) {
    editorSelection.activeGroupId = result.selectedGroupId;
    editorSelection.activeGroupType = state.steps.some(
      (x) => x.id === result.selectedGroupId,
    )
      ? 'step'
      : 'appendix';
    editorSelection.activeStepId =
      editorSelection.activeGroupType === 'step'
        ? result.selectedGroupId
        : null;
  }
  if (result.selectedBlockId)
    editorSelection.activeBlockId = result.selectedBlockId;
  if ('rowIndex' in result) editorSelection.row = result.rowIndex;
  if ('columnIndex' in result) editorSelection.column = result.columnIndex;
  refreshSelectedTableCell(result);
  editorSelection.restoreFocus = focus;
  render({ restore: focus });
  return true;
}
function refreshSelectedTableCell(result) {
  if (!('rowIndex' in result) && !('columnIndex' in result)) return;
  const { block } = findBlock(editorSelection.activeBlockId);
  if (block?.type !== 'table') return;
  const row = block.rows[Number(editorSelection.row)],
    column = block.columns[Number(editorSelection.column)];
  editorSelection.rowId = row?.id || null;
  editorSelection.columnId = column?.id || null;
  editorSelection.cellId =
    row?.cells?.[Number(editorSelection.column)]?.id || null;
}
function selectOperationResult(result, type) {
  const group =
    state.steps.find((x) => x.id === result.selectedGroupId) ||
    state.appendices.find((x) => x.id === result.selectedGroupId);
  if (!group) return;
  selectGroup(group, state.steps.includes(group) ? 'step' : 'appendix', {
    blockId: result.selectedBlockId,
  });
  Object.assign(editorSelection, {
    activeBlockId: result.selectedBlockId || null,
    blockType: type || result.selectedBlockType || null,
    container:
      type === 'table'
        ? 'cell'
        : type?.endsWith('List')
          ? 'item'
          : type === 'image'
            ? 'image-caption'
            : 'block',
    row: type === 'table' ? 0 : null,
    column: type === 'table' ? 0 : null,
    rowId:
      type === 'table'
        ? group.blocks.find((b) => b.id === result.selectedBlockId)?.rows?.[0]
            ?.id
        : null,
    columnId:
      type === 'table'
        ? group.blocks.find((b) => b.id === result.selectedBlockId)
            ?.columns?.[0]?.id
        : null,
    cellId:
      type === 'table'
        ? group.blocks.find((b) => b.id === result.selectedBlockId)?.rows?.[0]
            ?.cells?.[0]?.id
        : null,
    offsets: [0, 0],
    restoreFocus: true,
  });
}

function bindEditing() {
  document.querySelectorAll('[data-editable-group-id]').forEach((x) => {
    x.addEventListener('pointerdown', (e) => {
      if (
        e.target.closest(
          '.editable-block,.step-heading,button,input,select,textarea,[contenteditable="true"]',
        )
      )
        setActive(e.target);
    });
    x.addEventListener('click', blankStepClick);
  });
  document.querySelectorAll('[contenteditable=true]').forEach((el) => {
    el.addEventListener('focusin', () => setActive(el));
    el.addEventListener('input', () => {
      repagination.cancel();
      schedule(el);
    });
    el.addEventListener('blur', () => {
      flushTarget(el);
      editorSelection.restoreFocus = false;
    });
    el.addEventListener('paste', (e) => insertPlainText(e, document));
  });
  document.querySelectorAll('[data-group-title]').forEach((el) => {
    const sync = () => syncStepTitle(el);
    el.addEventListener('focusin', () => setActive(el));
    el.addEventListener('input', () => {
      repagination.cancel();
      const old = pendingStepTitles.get(el.dataset.groupTitle);
      if (old) clearTimeout(old.timer);
      pendingStepTitles.set(el.dataset.groupTitle, {
        el,
        timer: setTimeout(sync, 450),
      });
    });
    el.addEventListener('blur', () => {
      sync();
      editorSelection.restoreFocus = false;
    });
    el.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        el.blur();
      } else if (e.key === 'Tab') editorSelection.restoreFocus = false;
    });
  });
  document
    .querySelectorAll('.list-block > :is(ol,ul) > li[data-container="item"]')
    .forEach((el) => {
      el.addEventListener('keydown', listKeydown);
      el.addEventListener('blur', (event) => listBlur(event, el));
    });
  document
    .querySelectorAll('[data-list-append]')
    .forEach((el) => el.addEventListener('click', appendListItem));
  document
    .querySelectorAll('[data-type="heading"] > [data-container="block"]')
    .forEach((el) => el.addEventListener('keydown', headingEnter));
  document
    .querySelectorAll('[data-container=cell]')
    .forEach((el) => el.addEventListener('keydown', tableTab));
  document
    .querySelectorAll('[data-block-move]')
    .forEach(
      (b) =>
        (b.onclick = () =>
          moveBlock(
            b.closest('[data-block-id]').dataset.blockId,
            b.dataset.blockMove,
          )),
    );
  document
    .querySelectorAll('[data-block-delete]')
    .forEach(
      (b) =>
        (b.onclick = () =>
          deleteBlock(b.closest('[data-block-id]').dataset.blockId)),
    );
  document.querySelectorAll('.toc a').forEach(
    (a) =>
      (a.onclick = (e) => {
        e.preventDefault();
        navigateToAnchor(a.getAttribute('href').slice(1));
      }),
  );
}
function headingEnter(e) {
  if (e.key !== 'Enter' || e.shiftKey) return;
  e.preventDefault();
  const el = e.currentTarget;
  flushTarget(el);
  const { group, block } = findBlock(
    el.closest('[data-block-id]').dataset.blockId,
  );
  if (!group || block?.type !== 'heading') return;
  const id = newId('block'),
    index = group.blocks.indexOf(block);
  const result = insertBlockOperation(
    state,
    group.id,
    { id, type: 'paragraph', runs: [] },
    index + 1,
  );
  if (applyDocumentOperation(result))
    Object.assign(editorSelection, {
      activeBlockId: id,
      blockType: 'paragraph',
      container: 'block',
      itemId: null,
      row: null,
      column: null,
      offsets: [0, 0],
    });
}
function listKeydown(e) {
  if (e.key === 'Enter') return listEnter(e);
  const el = e.currentTarget,
    { block } = findBlock(el.closest('[data-block-id]').dataset.blockId),
    index = block?.items?.findIndex((item) => item.id === el.dataset.itemId),
    offsets = getOffsets(el),
    atBoundary =
      e.key === 'ArrowUp'
        ? offsets?.[0] === 0
        : offsets?.[1] === runText(block?.items?.[index]?.runs || []).length;
  if (
    (e.key === 'ArrowUp' || e.key === 'ArrowDown') &&
    !e.shiftKey &&
    !e.ctrlKey &&
    !e.metaKey &&
    !e.altKey &&
    offsets?.[0] === offsets?.[1] &&
    atBoundary
  ) {
    const target = index + (e.key === 'ArrowUp' ? -1 : 1);
    if (target < 0 || target >= block.items.length) return;
    e.preventDefault();
    const offset =
      e.key === 'ArrowUp' ? runText(block.items[target].runs).length : 0;
    editorSelection.itemId = block.items[target].id;
    editorSelection.offsets = [offset, offset];
    restoreTextSelection(
      el.parentElement.children[target],
      editorSelection.offsets,
    );
    return;
  }
  if (e.key !== 'Backspace') return;
  flushTarget(el);
  if (!Number.isInteger(index) || index < 1 || runText(block.items[index].runs))
    return;
  e.preventDefault();
  commit(() => {
    const target = removeEmptyListItem(block, index);
    editorSelection.itemId = target.itemId;
    editorSelection.offsets = [target.offset, target.offset];
  });
}
function listBlur(event, el) {
  const blockEl = el.closest('[data-block-id]'),
    next = event.relatedTarget;
  if (
    next?.closest?.(
      `[data-block-id="${blockEl?.dataset.blockId}"] [data-item-id="${el.dataset.itemId}"]`,
    )
  )
    return;
  queueMicrotask(() => {
    if (!el.isConnected) return;
    const { block } = findBlock(blockEl.dataset.blockId),
      index = block?.items?.findIndex((item) => item.id === el.dataset.itemId);
    if (
      block?.items.length > 1 &&
      index >= 0 &&
      !runText(block.items[index].runs)
    ) {
      commit(
        () => {
          block.items.splice(index, 1);
          const target = block.items[Math.max(0, index - 1)];
          editorSelection.itemId = target.id;
          editorSelection.offsets = [
            runText(target.runs).length,
            runText(target.runs).length,
          ];
        },
        { focus: false },
      );
    }
  });
}
function appendListItem(event) {
  event.preventDefault();
  event.stopPropagation();
  flushAll();
  const blockId =
      event.currentTarget.closest('[data-block-id]').dataset.blockId,
    { block } = findBlock(blockId);
  if (!block?.items) return;
  const id = newId('item');
  Object.assign(editorSelection, {
    activeBlockId: blockId,
    blockType: block.type,
    container: 'item',
    itemId: id,
    offsets: [0, 0],
    restoreFocus: true,
  });
  commit(() =>
    block.items.push({ id, level: block.items.at(-1)?.level || 1, runs: [] }),
  );
}
function listEnter(e) {
  if (e.key !== 'Enter') return;
  e.preventDefault();
  const el = e.currentTarget;
  flushTarget(el);
  const { block } = findBlock(el.closest('[data-block-id]').dataset.blockId),
    index = block.items.findIndex((i) => i.id === el.dataset.itemId);
  const offsets = getOffsets(el) || [0, 0],
    text = runText(block.items[index].runs);
  if (!text.trim()) {
    commit(() => {
      block.items.splice(index, 1);
      if (!block.items.length) {
        block.type = 'paragraph';
        block.runs = [];
        delete block.items;
      }
    });
    return;
  }
  const id = newId('item');
  commit(() => {
    const item = block.items[index],
      [before, after] = splitListRuns(item.runs, offsets[0]);
    item.runs = before;
    block.items.splice(index + 1, 0, { id, level: item.level, runs: after });
    editorSelection.itemId = id;
    editorSelection.offsets = [0, 0];
  });
}
function tableTab(e) {
  if (e.key !== 'Tab') return;
  const cells = [
      ...document.querySelectorAll(
        `[data-block-id="${editorSelection.activeBlockId}"] [data-container=cell]`,
      ),
    ],
    i = cells.indexOf(e.currentTarget);
  if (e.shiftKey || i < cells.length - 1) return;
  e.preventDefault();
  flushTarget(e.currentTarget);
  const { block } = findBlock(editorSelection.activeBlockId);
  tableMutation('addRow');
}
const getOffsets = (el) =>
  containedSelectionOffsets(el, getSelection(), document);
function resolveRuns() {
  const { block } = findBlock(editorSelection.activeBlockId);
  if (!block) return null;
  if (editorSelection.container === 'block') return block.runs;
  if (editorSelection.container === 'item')
    return block.items.find((x) => x.id === editorSelection.itemId)?.runs;
  if (
    editorSelection.container === 'caption' ||
    editorSelection.container === 'image-caption'
  )
    return block.captionRuns;
  if (editorSelection.container === 'heading')
    return block.columns[editorSelection.column]?.headingRuns;
  if (editorSelection.container === 'cell')
    return block.rows[editorSelection.row]?.cells[editorSelection.column]?.runs;
  return null;
}
function replaceResolved(runs) {
  const { block } = findBlock(editorSelection.activeBlockId);
  if (editorSelection.container === 'block') block.runs = runs;
  else if (editorSelection.container === 'item')
    block.items.find((x) => x.id === editorSelection.itemId).runs = runs;
  else if (['caption', 'image-caption'].includes(editorSelection.container))
    block.captionRuns = runs;
  else if (editorSelection.container === 'heading')
    block.columns[editorSelection.column].headingRuns = runs;
  else if (editorSelection.container === 'cell')
    block.rows[editorSelection.row].cells[editorSelection.column].runs = runs;
}
function transformSelection(kind, value, { useCaptured = false } = {}) {
  flushAll();
  if (!useCaptured) {
    const offsets = getOffsets(activeEditable());
    if (offsets) editorSelection.offsets = offsets;
  }
  const runs = resolveRuns();
  if (!runs || !editorSelection.offsets) return false;
  if (
    kind === 'highlight' &&
    editorSelection.offsets[0] === editorSelection.offsets[1]
  )
    editorSelection.offsets = [0, runText(runs).length];
  commit(() =>
    replaceResolved(transformRuns(runs, editorSelection.offsets, kind, value)),
  );
  return true;
}
function defaultInsertionContext() {
  const group = selectedEditableGroup();
  return group
    ? {
        groupId: group.id,
        referenceBlockId: editorSelection.activeBlockId,
        position: 'after',
      }
    : null;
}
function insertBlock(
  type,
  context = defaultInsertionContext(),
  headingLevel = 2,
) {
  if (
    typeof type !== 'string' ||
    !['paragraph', 'heading', 'bulletList', 'numberList', 'table'].includes(
      type,
    )
  )
    return false;
  const group = [...state.steps, ...state.appendices].find(
    (x) => x.id === context?.groupId,
  );
  if (!group) return false;
  const index = insertionIndex(group, context);
  if (index < 0) return false;
  const id = newId('block');
  let block = { id, type, runs: [] };
  if (type === 'heading')
    block.level = Math.min(4, Math.max(1, Number(headingLevel) || 2));
  if (type.endsWith('List'))
    block = { id, type, items: [{ id: newId('item'), level: 1, runs: [] }] };
  if (type === 'table')
    block = {
      id,
      type,
      captionRuns: [{ text: 'Table caption' }],
      columns: [
        { id: newId('column'), headingRuns: [{ text: 'Column 1' }], width: 50 },
        { id: newId('column'), headingRuns: [{ text: 'Column 2' }], width: 50 },
      ],
      rows: [
        {
          id: newId('row'),
          cells: [
            { id: newId('cell'), runs: [] },
            { id: newId('cell'), runs: [] },
          ],
        },
      ],
    };
  const result = insertBlockOperation(state, group.id, block, index);
  selectOperationResult(result, type);
  editorSelection.restoreFocus = true;
  if (!applyDocumentOperation(result)) return false;
  return true;
}
function moveBlock(id, direction) {
  flushAll();
  const result = moveBlockOperation(state, id, direction);
  if (!applyDocumentOperation(result)) return false;
  editorSelection.activeBlockId = id;
  announce(`Block moved to position ${result.index + 1}`);
  return true;
}
function deleteBlock(id) {
  flushAll();
  const found = findBlock(id);
  if (!found.block) return false;
  const label = blockTypeLabel(found.block.type),
    result = deleteBlockOperation(state, id);
  if (!applyDocumentOperation(result)) return false;
  selectOperationResult(result, result.selectedBlockType);
  announce(`${label[0].toUpperCase() + label.slice(1)} deleted`);
  return true;
}
function selectGroup(group, type, { blockId = null } = {}) {
  Object.assign(editorSelection, {
    activeGroupId: group?.id || null,
    activeGroupType: group ? type : null,
    activeStepId: type === 'step' ? group?.id || null : null,
    activeBlockId: blockId,
    blockType: blockId ? 'paragraph' : null,
    container: blockId ? 'block' : null,
    offsets: [0, 0],
  });
}
function manageStep(action) {
  flushAll();
  const id = editorSelection.activeStepId,
    index = state.steps.findIndex((x) => x.id === id);
  if (index < 0) return;
  if (action === 'deleteStep') {
    if (
      stepHasMovements(state.excel, id) &&
      !confirm(
        'Deleting this step will delete its nonzero Excel share movements from every company. Continue?',
      )
    )
      return;
    if (!confirmStepDeletion(state.steps, index, confirm)) return;
    const result = deleteStepOperation(state, id);
    if (applyDocumentOperation(result)) {
      const target = state.steps.find((x) => x.id === result.selectedGroupId);
      if (target) selectGroup(target, 'step');
      announce('Step deleted');
    }
    return;
  }
  const result = moveStepOperation(state, id, action === 'moveStepUp' ? -1 : 1);
  if (applyDocumentOperation(result))
    announce(`Step moved to position ${result.index + 1}`);
}
function addStep(where = 'after') {
  flushAll();
  const result = addStepOperation(state, {
    referenceId: editorSelection.activeStepId,
    position: where,
    idFactory: newId,
  });
  if (!applyDocumentOperation(result)) return false;
  selectOperationResult(result, 'paragraph');
  return true;
}
function addAppendix() {
  flushAll();
  const result = addAppendixOperation(state, { idFactory: newId });
  if (!applyDocumentOperation(result)) return false;
  selectOperationResult(result, 'paragraph');
  announce(`${appendixLabel(result.index)} added`);
  return true;
}
function manageAppendix(action) {
  flushAll();
  const id = editorSelection.activeGroupId,
    index = state.appendices.findIndex((x) => x.id === id);
  if (index < 0) return;
  const appendix = state.appendices[index];
  if (action === 'deleteAppendix') {
    if (
      !confirm(
        `Delete ${appendixLabel(index)}: ${appendix.title}? This can be undone.`,
      )
    )
      return;
    const result = deleteAppendixOperation(state, id);
    if (!applyDocumentOperation(result)) return;
    const target =
      state.appendices.find((x) => x.id === result.selectedGroupId) ||
      state.steps.at(-1);
    if (target)
      selectGroup(
        target,
        state.appendices.includes(target) ? 'appendix' : 'step',
      );
    announce('Appendix deleted');
    return;
  }
  const result = moveAppendixOperation(
    state,
    id,
    action === 'moveAppendixUp' ? -1 : 1,
  );
  if (applyDocumentOperation(result))
    announce(`Appendix moved to ${appendixLabel(result.index)}`);
}
function tableMutation(action) {
  flushAll();
  const { block } = findBlock(editorSelection.activeBlockId);
  if (block?.type !== 'table') return false;
  const rowIndex =
      editorSelection.row == null ? NaN : Number(editorSelection.row),
    columnIndex =
      editorSelection.column == null ? NaN : Number(editorSelection.column),
    row = block.rows[rowIndex],
    column = block.columns[columnIndex];
  let result;
  if (action === 'moveRowUp' || action === 'moveRowDown')
    result = moveTableRowOperation(
      state,
      block.id,
      row?.id,
      action === 'moveRowUp' ? -1 : 1,
    );
  else if (action === 'moveColumnLeft' || action === 'moveColumnRight')
    result = moveTableColumnOperation(
      state,
      block.id,
      column?.id,
      action === 'moveColumnLeft' ? -1 : 1,
    );
  else if (action === 'addRow')
    result = insertTableRowOperation(state, block.id, row?.id, {
      idFactory: newId,
    });
  else if (action === 'addColumn')
    result = insertTableColumnOperation(state, block.id, column?.id, {
      idFactory: newId,
    });
  else if (action === 'removeRow')
    result = deleteTableRowOperation(state, block.id, row?.id);
  else if (action === 'removeColumn')
    result = deleteTableColumnOperation(state, block.id, column?.id);
  else return false;
  if (!applyDocumentOperation(result)) return false;
  if ('rowIndex' in result) editorSelection.row = result.rowIndex;
  if ('columnIndex' in result) editorSelection.column = result.columnIndex;
  editorSelection.offsets = [0, 0];
  if (action.startsWith('moveRow'))
    announce(`Row moved to position ${result.rowIndex + 1}`);
  if (action.startsWith('moveColumn'))
    announce(`Column moved to position ${result.columnIndex + 1}`);
  return true;
}
function linkPicker() {
  captureSelection();
  const dialog = $('#link-picker'),
    list = $('#link-destination'),
    destinations = [
      ...state.steps.map((group, index) => ({
        group,
        label: `Step ${index + 1}. ${group.title}`,
      })),
      ...state.appendices.map((group, index) => ({
        group,
        label: `${appendixLabel(index)}. ${group.title}`,
      })),
    ];
  list.innerHTML = destinations
    .map(
      ({ group, label }) =>
        `<option value="#${stableAnchor(group)}">${esc(label)}</option>`,
    )
    .join('');
  dialog.showModal();
}
let insertionContext = null,
  imageInsertionContext = null;
function blankStepClick(event) {
  if (event.button !== 0 || event.defaultPrevented) return;
  const stepElement = event.currentTarget;
  const active = document.activeElement,
    editing = active?.matches?.('[contenteditable=true],[data-group-title]');
  if (
    editing &&
    !event.target.closest(
      '[contenteditable=true],[data-group-title],button,input,select,textarea,a',
    )
  ) {
    flushAll();
    active.blur();
    getSelection()?.removeAllRanges();
    Object.assign(editorSelection, {
      activeBlockId: null,
      blockType: null,
      container: null,
      itemId: null,
      row: null,
      column: null,
      rowId: null,
      columnId: null,
      cellId: null,
      offsets: null,
      restoreFocus: false,
    });
    return;
  }
  if (
    event.target.closest(
      '.editable-block,.step-heading,.page-header,.page-footer,a,button,input,select,textarea,[contenteditable="true"],.block-controls,.toolbar',
    )
  )
    return;
  const body =
    event.target.closest('.document-body') ||
    stepElement.querySelector('.document-body');
  if (!body || !stepElement.contains(event.target)) return;
  const group = [...state.steps, ...state.appendices].find(
    (item) => item.id === stepElement.dataset.editableGroupId,
  );
  if (!group) return;
  const canInsert = canOpenBlankSpaceInsertion(
    editorSelection.activeGroupId,
    group.id,
  );
  selectGroup(group, stepElement.dataset.groupType);
  document
    .querySelectorAll('[data-editable-group-id]')
    .forEach((element) =>
      element.classList.toggle(
        'is-selected',
        element.dataset.editableGroupId === group.id,
      ),
    );
  if (!canInsert) return;
  insertionContext = insertionContextFromPoint(
    group.id,
    [...stepElement.querySelectorAll('[data-block-id]')],
    event.clientY,
  );
  const chooser = $('#insertion-chooser');
  if (!chooser.open) chooser.showModal();
  queueMicrotask(() =>
    chooser.querySelector('[data-insert-choice="paragraph"]').focus(),
  );
}
function addImage(context = defaultInsertionContext()) {
  if (
    ![...state.steps, ...state.appendices].some(
      (group) => group.id === context?.groupId,
    )
  )
    return;
  imageInsertionContext = context;
  const form = $('#image-form'),
    error = $('#image-form-error');
  form.reset();
  error.hidden = true;
  error.textContent = '';
  $('#image-dialog').showModal();
  form.elements.src.focus();
}
function historyChangeTarget(next) {
  if (!next) return null;
  const currentGroups = allGroups(),
    nextGroups = [...next.sections, ...next.steps, ...next.appendices];
  for (const nextGroup of nextGroups) {
    const currentGroup = currentGroups.find(
      (group) => group.id === nextGroup.id,
    );
    if (!currentGroup) return { groupId: nextGroup.id };
    if (currentGroup.title !== nextGroup.title)
      return { groupId: currentGroup.id };
    const ids = new Set([
      ...currentGroup.blocks.map((block) => block.id),
      ...nextGroup.blocks.map((block) => block.id),
    ]);
    for (const id of ids) {
      const before = currentGroup.blocks.find((block) => block.id === id),
        after = nextGroup.blocks.find((block) => block.id === id);
      if (JSON.stringify(before) !== JSON.stringify(after))
        return { groupId: currentGroup.id, blockId: before?.id || null };
    }
  }
  const removed = currentGroups.find(
    (group) => !nextGroups.some((item) => item.id === group.id),
  );
  return removed ? { groupId: removed.id } : null;
}
function revealHistoryChange(next) {
  const target = historyChangeTarget(next);
  if (!target) return;
  const node =
    (target.blockId &&
      document.querySelector(
        `[data-block-id="${CSS.escape(target.blockId)}"]`,
      )) ||
    document.querySelector(
      `[data-editable-group-id="${CSS.escape(target.groupId)}"]`,
    ) ||
    document.querySelector(
      `.pagedjs_page[data-group-id="${CSS.escape(target.groupId)}"]`,
    );
  const page =
    node?.closest('.pagedjs_page') ||
    (node?.matches('.pagedjs_page') ? node : null);
  if (!page) return;
  const pages = [...document.querySelectorAll('.pagedjs_page')];
  setCurrentPage(pages.indexOf(page) + 1, { scroll: false, focus: false });
  page.scrollIntoView({ behavior: 'auto', block: 'center' });
}
function command(action) {
  flushAll();
  if ((action === 'undo' || action === 'redo') && activeEditor === 'excel') {
    excelController[action]();
    markDirty();
    return;
  }
  if (action === 'undo') {
    const previousView = navigationHistory.undo();
    if (previousView) {
      restoreView(previousView);
      announce('Returned to the previous location');
      return;
    }
  }
  if (action === 'undo' || action === 'redo') {
    navigationHistory.clear();
    pending.forEach((x) => clearTimeout(x.timer));
    pending.clear();
    const next = action === 'undo' ? history.peekUndo() : history.peekRedo();
    revealHistoryChange(next);
    const currentExcel = state.excel,
      stepsState = action === 'undo' ? history.undo() : history.redo();
    state = normaliseDocument({ ...stepsState, excel: currentExcel });
    history.replace(state);
    markDirty();
    if (
      ![...state.steps, ...state.appendices].some(
        (x) => x.id === editorSelection.activeGroupId,
      )
    ) {
      const group = state.steps[0] || state.appendices[0];
      selectGroup(group, state.steps.includes(group) ? 'step' : 'appendix');
    }
    return render();
  }
  if (action === 'addStepBefore' || action === 'addStepAfter')
    return addStep(action.endsWith('After') ? 'after' : 'before');
  if (['moveStepUp', 'moveStepDown', 'deleteStep'].includes(action))
    return manageStep(action);
  if (action === 'addAppendix') return addAppendix();
  if (['moveAppendixUp', 'moveAppendixDown', 'deleteAppendix'].includes(action))
    return manageAppendix(action);
  const insertion = routeInsertionCommand(action);
  if (insertion?.kind === 'table') return tableMutation(insertion.action);
  if (insertion?.kind === 'block' && typeof insertion.blockType === 'string')
    return insertBlock(insertion.blockType);
  if (action === 'addImage') return addImage();
  if (action === 'highlight' || action === 'unlink')
    return transformSelection(action === 'unlink' ? 'unlink' : 'highlight');
  if (action === 'link') return linkPicker();
  if (editorSelection.blockType === 'image') return;
  const { block } = findBlock(editorSelection.activeBlockId);
  if (
    !block ||
    !['heading', 'paragraph', 'bulletList', 'numberList'].includes(block.type)
  )
    return;
  if (canApplyBlockStyle(block.type, action)) {
    const result = convertBlockStyleOperation(state, block.id, action, {
      idFactory: newId,
    });
    if (applyDocumentOperation(result))
      editorSelection.blockType = result.selectedBlockType;
  }
}
function announce(message) {
  $('#editor-status').textContent = '';
  requestAnimationFrame(() => ($('#editor-status').textContent = message));
}
function updateControls() {
  const i = state.steps.findIndex((x) => x.id === editorSelection.activeStepId),
    step = i >= 0,
    appendixIndex =
      editorSelection.activeGroupType === 'appendix'
        ? state.appendices.findIndex(
            (x) => x.id === editorSelection.activeGroupId,
          )
        : -1,
    editable = !!selectedEditableGroup();
  document
    .querySelectorAll('[data-command=moveStepUp]')
    .forEach((x) => (x.disabled = !step || i === 0));
  document
    .querySelectorAll('[data-command=moveStepDown]')
    .forEach((x) => (x.disabled = !step || i === state.steps.length - 1));
  document
    .querySelectorAll('[data-command=deleteStep]')
    .forEach((x) => (x.disabled = !step || state.steps.length === 1));
  document
    .querySelectorAll('[data-requires-step]')
    .forEach((x) => (x.disabled = !step));
  document
    .querySelectorAll('[data-requires-group]')
    .forEach((x) => (x.disabled = !editable));
  document
    .querySelectorAll('[data-command=moveAppendixUp]')
    .forEach((x) => (x.disabled = appendixIndex <= 0));
  document
    .querySelectorAll('[data-command=moveAppendixDown]')
    .forEach(
      (x) =>
        (x.disabled =
          appendixIndex < 0 || appendixIndex === state.appendices.length - 1),
    );
  document
    .querySelectorAll('[data-command=deleteAppendix]')
    .forEach((x) => (x.disabled = appendixIndex < 0));
  const runs = resolveRuns(),
    text = !!runs;
  document
    .querySelectorAll('[data-text-command]')
    .forEach((x) => (x.disabled = !text));
  const { block } = findBlock(editorSelection.activeBlockId),
    style = $('#style'),
    choices = blockStyleChoices(block?.type);
  if (style.dataset.choices !== JSON.stringify(choices)) {
    style.replaceChildren(
      ...choices.map(([value, label]) => {
        const option = document.createElement('option');
        option.value = value;
        option.textContent = label;
        return option;
      }),
    );
    style.dataset.choices = JSON.stringify(choices);
  }
  document
    .querySelectorAll('[data-table-command]')
    .forEach((x) => (x.hidden = block?.type !== 'table'));
  document
    .querySelectorAll('[data-table-group]')
    .forEach((x) => (x.hidden = block?.type !== 'table'));
  style.disabled = !text;
  if (block?.type === 'heading') style.value = `heading${block.level}`;
  else if (block?.type === 'paragraph') style.value = 'body';
  else if (['bulletList', 'numberList'].includes(block?.type))
    style.value = block.type;
  const row = editorSelection.row == null ? NaN : Number(editorSelection.row),
    column =
      editorSelection.column == null ? NaN : Number(editorSelection.column),
    selectedRow = block?.rows?.[row];
  document
    .querySelectorAll('[data-command=moveRowUp]')
    .forEach(
      (x) =>
        (x.disabled =
          !selectedRow ||
          row === 0 ||
          block.rows[row - 1]?.isTotal !== selectedRow.isTotal),
    );
  document
    .querySelectorAll('[data-command=moveRowDown]')
    .forEach(
      (x) =>
        (x.disabled =
          !selectedRow ||
          row === block.rows.length - 1 ||
          block.rows[row + 1]?.isTotal !== selectedRow.isTotal),
    );
  document
    .querySelectorAll('[data-command=moveColumnLeft]')
    .forEach((x) => (x.disabled = !block?.columns?.[column] || column === 0));
  document
    .querySelectorAll('[data-command=moveColumnRight]')
    .forEach(
      (x) =>
        (x.disabled =
          !block?.columns?.[column] || column === block.columns.length - 1),
    );
  const format = $('#column-format'),
    selectedColumn = block?.columns?.[column];
  format.disabled = !selectedColumn;
  if (selectedColumn) format.value = tableColumnFormat(selectedColumn);
  const total = $('#column-total-enabled');
  total.disabled =
    !selectedColumn || tableColumnFormat(selectedColumn) === 'text';
  total.checked = !!selectedColumn?.totalEnabled;
}
document.querySelectorAll('[data-command]').forEach((b) => {
  b.addEventListener('pointerdown', (e) => {
    if (b.dataset.textCommand !== undefined) {
      e.preventDefault();
      captureSelection();
    }
  });
  b.addEventListener('click', () => command(b.dataset.command));
});
$('#style').onchange = (e) => command(e.target.value);
$('#column-format').onchange = (e) => {
  const { block } = findBlock(editorSelection.activeBlockId),
    column =
      block?.columns?.[
        editorSelection.column == null ? NaN : Number(editorSelection.column)
      ];
  if (!column) return;
  const result = setTableColumnFormatOperation(
    state,
    block.id,
    column.id,
    e.target.value,
  );
  if (applyDocumentOperation(result))
    announce(
      `Column format changed to ${e.target.selectedOptions[0].textContent}`,
    );
};
$('#column-total-enabled').onchange = (e) => {
  captureSelection();
  const { block } = findBlock(editorSelection.activeBlockId),
    column =
      block?.columns?.find((c) => c.id === editorSelection.columnId) ||
      block?.columns?.[Number(editorSelection.column)];
  if (!column) return;
  applyDocumentOperation(
    setTableColumnTotalOperation(state, block.id, column.id, e.target.checked),
  );
};
const linkDialog = $('#link-picker');
let applyingLink = false;
$('#link-form').onsubmit = (e) => {
  e.preventDefault();
  const external = $('#external-link').value,
    href = validatedLink(external, $('#link-destination').value);
  if (href) {
    applyingLink = true;
    linkDialog.close();
    transformSelection('link', href, { useCaptured: true });
    applyingLink = false;
    editorSelection.offsets = null;
  } else
    announce('Enter a valid HTTPS, HTTP, mailto, or internal destination.');
};
linkDialog.addEventListener('close', () => {
  if (!applyingLink) editorSelection.offsets = null;
  $('#link-form').reset();
});
const insertionChooser = $('#insertion-chooser'),
  templateLoader = new EncryptedTemplateLoader();
function showInsertionPanel(name, focus = true) {
  insertionChooser
    .querySelectorAll('[data-insertion-panel]')
    .forEach((panel) => (panel.hidden = panel.dataset.insertionPanel !== name));
  if (focus)
    queueMicrotask(() =>
      insertionChooser
        .querySelector(
          `[data-insertion-panel="${name}"] button, [data-insertion-panel="${name}"] input`,
        )
        ?.focus(),
    );
}
function finishInsertion(callback) {
  const context = insertionContext;
  insertionContext = null;
  insertionChooser.close();
  callback(context);
}
insertionChooser.querySelectorAll('[data-open-insertion-panel]').forEach(
  (button) =>
    (button.onclick = () => {
      showInsertionPanel(button.dataset.openInsertionPanel);
      if (button.dataset.openInsertionPanel === 'defaults') openDefaults();
    }),
);
insertionChooser
  .querySelectorAll('[data-insertion-back]')
  .forEach((button) => (button.onclick = () => showInsertionPanel('main')));
insertionChooser
  .querySelectorAll('[data-insert-choice]')
  .forEach(
    (button) =>
      (button.onclick = () =>
        finishInsertion((context) =>
          button.dataset.insertChoice === 'image'
            ? addImage(context)
            : insertBlock(button.dataset.insertChoice, context),
        )),
  );
insertionChooser
  .querySelectorAll('[data-insert-heading]')
  .forEach(
    (button) =>
      (button.onclick = () =>
        finishInsertion((context) =>
          insertBlock('heading', context, button.dataset.insertHeading),
        )),
  );
function renderTemplates(templates) {
  const choices = $('#default-template-choices');
  choices.replaceChildren(
    ...templates.map((template) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = template.header;
      button.onclick = () =>
        finishInsertion((context) => insertTemplate(template, context));
      return button;
    }),
  );
  $('#default-unlock').hidden = true;
  choices.querySelector('button')?.focus();
}
async function openDefaults() {
  const error = $('#default-error');
  error.hidden = true;
  if (templateLoader.cached) {
    renderTemplates(await templateLoader.unlock());
    return;
  }
  $('#default-unlock').hidden = false;
  $('#default-password').focus();
}
$('#unlock-defaults').onclick = async () => {
  const input = $('#default-password'),
    button = $('#unlock-defaults'),
    error = $('#default-error'),
    password = input.value;
  input.value = '';
  button.disabled = true;
  error.hidden = true;
  try {
    renderTemplates(
      await loadDefaultTemplates({
        loader: templateLoader,
        envelope: defaultTemplateEnvelope,
        password,
      }),
    );
  } catch (failure) {
    error.textContent = failure.message;
    error.hidden = false;
    input.focus();
  } finally {
    button.disabled = false;
  }
};
$('#default-password').addEventListener('keydown', (event) => {
  if (event.key !== 'Enter') return;
  event.preventDefault();
  event.stopPropagation();
  $('#unlock-defaults').click();
});
function insertTemplate(template, context) {
  const group = [...state.steps, ...state.appendices].find(
      (item) => item.id === context?.groupId,
    ),
    at = insertionIndex(group, context);
  if (!group || at < 0) {
    announce('That insertion location is no longer available.');
    return false;
  }
  const blocks = templateBlocks(template, newId);
  let result;
  for (const [offset, block] of blocks.entries()) {
    result = insertBlockOperation(
      result?.document || state,
      group.id,
      block,
      at + offset,
    );
    if (!result.changed) return false;
  }
  applyDocumentOperation(result);
  Object.assign(editorSelection, {
    activeBlockId: blocks[0].id,
    blockType: 'heading',
    container: 'block',
    itemId: null,
    row: null,
    column: null,
    offsets: [0, template.header.length],
  });
  announce(`${template.header} inserted`);
  return true;
}
insertionChooser.addEventListener('close', () => {
  insertionContext = null;
  $('#default-password').value = '';
  showInsertionPanel('main', false);
});
bindCallouts();
function persistDraft({ quiet = false, updateUrl = false } = {}) {
  flushAll(false);
  clearTimeout(autosaveTimer);
  let result;
  try {
    result = saveDraft(state, loadedRevision);
  } catch (error) {
    const indicator = $('#save-state');
    indicator.textContent = 'Save failed';
    indicator.className = 'save-state error';
    if (!quiet) alert(`Could not save this draft: ${error.message}`);
    return false;
  }
  if (!result.ok) {
    const indicator = $('#save-state');
    indicator.textContent = 'Newer draft in another tab';
    indicator.className = 'save-state error';
    if (!quiet)
      alert(
        'This draft was changed in another tab. Reload before saving so that the newer draft and recovery copy are not overwritten.',
      );
    return false;
  }
  loadedRevision = result.revision;
  dirty = false;
  if (updateUrl)
    updateSnapshotUrl(result.record.document, window.location, window.history);
  const indicator = $('#save-state');
  indicator.textContent = `Saved locally ${new Date(result.record.savedAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
  indicator.className = 'save-state';
  return true;
}
function versionsDialog() {
  const dialog = $('#versions-dialog'),
    list = $('#versions-list'),
    versions = listVersions();
  list.innerHTML = versions.length
    ? versions
        .map(
          (v) =>
            `<div class="version-row"><span><b>${esc(v.label)}</b><small>${new Date(v.savedAt).toLocaleString()}</small></span><button type="button" data-restore-version="${esc(v.id)}">Restore</button><button type="button" data-delete-version="${esc(v.id)}">Delete</button></div>`,
        )
        .join('')
    : '<p>No saved versions yet.</p>';
  list.querySelectorAll('[data-restore-version]').forEach(
    (button) =>
      (button.onclick = async () => {
        const version = listVersions().find(
          (v) => v.id === button.dataset.restoreVersion,
        );
        if (!version) return;
        if (!persistDraft()) return;
        state = normaliseDocument(version.document);
        history.commit(state);
        excelController?.restore(state.excel);
        markDirty();
        dialog.close();
        await render({ restore: false });
        announce('Saved version restored');
      }),
  );
  list.querySelectorAll('[data-delete-version]').forEach(
    (button) =>
      (button.onclick = () => {
        deleteVersion(button.dataset.deleteVersion);
        versionsDialog();
      }),
  );
  if (!dialog.open) dialog.showModal();
}
const propertiesDialog = $('#properties-dialog'),
  propertiesForm = $('#properties-form');
$('#document-properties').onclick = () => {
  for (const key of METADATA_KEYS)
    propertiesForm.elements[key].value = state.meta[key];
  propertiesDialog.showModal();
};
propertiesForm.onsubmit = async (event) => {
  event.preventDefault();
  for (const key of METADATA_KEYS)
    propertiesForm.elements[key].value =
      propertiesForm.elements[key].value.trim();
  if (!propertiesForm.reportValidity()) return;
  await commit(
    () => {
      for (const key of METADATA_KEYS)
        state.meta[key] = propertiesForm.elements[key].value;
    },
    { focus: false },
  );
  propertiesDialog.close();
  announce('Document properties updated');
};
const imageDialog = $('#image-dialog'),
  imageForm = $('#image-form'),
  imageError = $('#image-form-error');
imageDialog.addEventListener('close', () => (imageInsertionContext = null));
imageForm.onsubmit = (event) => {
  event.preventDefault();
  const caption = imageForm.elements.caption.value.trim(),
    widthValue = imageForm.elements.width.value,
    result = validateImageInput({
      source: imageForm.elements.src.value,
      altText: imageForm.elements.alt.value,
      width: widthValue,
    });
  if (!result.ok || !imageForm.checkValidity()) {
    const messages = {
      'missing-source':
        'Enter an image URL and alternative text, and use a width between 20 and 100.',
      'missing-alt-text':
        'Enter an image URL and alternative text, and use a width between 20 and 100.',
      'malformed-source': 'Enter a complete, well-formed image URL.',
      'unsupported-protocol':
        'Unsupported protocol. Image URLs must use HTTPS.',
      'unsupported-extension':
        'Unsupported image extension. Use a PNG, JPEG, GIF or WebP URL.',
    };
    imageError.textContent =
      messages[result.error] ||
      'Enter an image URL and alternative text, and use a width between 20 and 100.';
    imageError.hidden = false;
    return;
  }
  const context = imageInsertionContext,
    group = [...state.steps, ...state.appendices].find(
      (item) => item.id === context?.groupId,
    ),
    at = insertionIndex(group, context),
    id = newId('image'),
    { source, altText, width } = result.image;
  if (!group || at < 0) {
    imageError.textContent = 'That insertion location is no longer available.';
    imageError.hidden = false;
    return;
  }
  const operation = insertBlockOperation(
    state,
    group.id,
    {
      id,
      type: 'image',
      src: source,
      alt: altText,
      captionRuns: caption ? [{ text: caption }] : [],
      width,
    },
    at,
  );
  if (!applyDocumentOperation(operation)) return;
  selectOperationResult(operation, 'image');
  imageDialog.close();
  announce('Published image added');
};
$('#save-draft').onclick = () => persistDraft({ updateUrl: true });
$('#save-version').onclick = () => {
  if (!persistDraft()) return;
  const label = prompt(
    'Name this saved version:',
    `${state.meta.version || 'Version'} — ${state.meta.status || 'Draft'}`,
  );
  if (label === null) return;
  const nextVersion = nextAvailableVersion(state.meta.version, listVersions());
  saveVersion(state, label);
  commit(
    () => {
      state.meta.version = nextVersion;
    },
    { focus: false },
  );
  announce(`Named version saved; document advanced to ${nextVersion}`);
};
$('#show-versions').onclick = versionsDialog;
$('#share-snapshot').onclick = async () => {
  flushAll(false);
  const url = `${location.href.split('#')[0]}#snapshot=${encodeSnapshot(state)}`;
  try {
    await navigator.clipboard.writeText(url);
    announce('Snapshot link copied');
  } catch {
    prompt('Copy this validated snapshot link:', url);
  }
};
$('#open-selected-link').onpointerdown = (event) => event.preventDefault();
$('#open-selected-link').onclick = () =>
  openEditorLink($('#link-context').dataset.href);
document.addEventListener('selectionchange', updateLinkContext);
const printButton = $('#print-document');
const printing = new PrintLifecycle({
  flush: () => flushAll(false),
  cancelPagination: () => repagination.cancel(),
  render: () => render({ immediate: true }),
  prepare: () => {
    const view = captureView();
    document.documentElement.classList.add('preparing-print');
    printButton.disabled = true;
    announce('Preparing complete document for printing');
    return view;
  },
  clearPresentation: () => {
    getSelection()?.removeAllRanges();
    document.activeElement?.blur?.();
  },
  print: () => window.print(),
  onAfterPrint: (cleanup) => {
    window.addEventListener('afterprint', cleanup, { once: true });
    return () => window.removeEventListener('afterprint', cleanup);
  },
  restore: (view) => {
    document.documentElement.classList.remove('preparing-print');
    printButton.disabled = false;
    restoreView(view);
    announce('Print preparation finished');
  },
  onError: (error) => announce(`Print preparation failed: ${error.message}`),
});
const exportDocument = () => printing.run();
printButton.onclick = exportDocument;
$('#export-document').onclick = exportDocument;
document.addEventListener('keydown', (event) => {
  if (!(event.ctrlKey || event.metaKey) || event.altKey) return;
  const key = event.key.toLowerCase();
  let action = null;
  if (key === 'z') action = event.shiftKey ? 'redo' : 'undo';
  else if (key === 'y' && !event.shiftKey) action = 'redo';
  else if (key === 's') action = event.shiftKey ? 'export' : 'save';
  if (!action) return;
  event.preventDefault();
  flushAll();
  if (action === 'save') persistDraft({ updateUrl: true });
  else if (action === 'export') exportDocument();
  else command(action);
});
addEventListener('pagehide', () => {
  if (dirty || hasPendingInputs()) persistDraft({ quiet: true });
});
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'hidden' && (dirty || hasPendingInputs()))
    persistDraft({ quiet: true });
});
addEventListener('storage', (event) => {
  if (event.key === 'ubta:draft' && readDraft()?.revision !== loadedRevision) {
    const indicator = $('#save-state');
    indicator.textContent = 'Newer draft in another tab';
    indicator.className = 'save-state error';
  }
});
let pageObserver = null,
  observedAreas = new Map();
function pageLabel(page, index) {
  const type = page.dataset.groupType,
    groupId = page.dataset.groupId,
    continued = Number(page.dataset.pageFragment) > 0;
  if (type === 'step') {
    const i = state.steps.findIndex((x) => x.id === groupId);
    return `Page ${index + 1} — Step ${i + 1}${continued ? ' (continued)' : ''}`;
  }
  if (type === 'appendix') {
    const i = state.appendices.findIndex((x) => x.id === groupId);
    return `Page ${index + 1} — ${appendixLabel(i)}${continued ? ' (continued)' : ''}`;
  }
  const sheet = page.querySelector('.sheet-source');
  return `Page ${index + 1} — ${sheet?.classList.contains('cover') ? 'Cover' : sheet?.classList.contains('contents') ? 'Contents' : type === 'section' ? 'Section' : 'Document'}`;
}
function rebuildPageJump(
  pages = [...document.querySelectorAll('.pagedjs_page')],
) {
  const jump = $('#page-jump');
  jump.replaceChildren(
    ...pages.map((page, index) => {
      const option = document.createElement('option');
      option.value = String(index + 1);
      option.textContent = pageLabel(page, index);
      return option;
    }),
  );
  jump.value = String(Math.min(currentPage, pages.length || 1));
}
function setCurrentPage(page, { scroll = false, focus = false } = {}) {
  const pages = [...document.querySelectorAll('.pagedjs_page')];
  currentPage = Math.max(1, Math.min(pages.length || 1, Number(page) || 1));
  $('#current').textContent = currentPage;
  const jump = $('#page-jump');
  if (jump) jump.value = String(currentPage);
  $('#prev').disabled = currentPage <= 1;
  $('#next').disabled = currentPage >= pages.length;
  if (scroll)
    pages[currentPage - 1]?.scrollIntoView({
      behavior: 'smooth',
      block: 'center',
    });
  if (focus)
    setTimeout(
      () => focusPageContext(pages[currentPage - 1]),
      scroll ? 350 : 0,
    );
}
function focusPageContext(page) {
  if (!page) return;
  const type = page.dataset.groupType,
    id = page.dataset.groupId;
  if (type === 'step' || type === 'appendix') {
    const group = (type === 'step' ? state.steps : state.appendices).find(
      (x) => x.id === id,
    );
    selectGroup(group, type);
    const target =
      Number(page.dataset.pageFragment) === 0
        ? page.querySelector('[data-group-title]')
        : page.querySelector('[contenteditable=true], [data-group-title]');
    target?.focus({ preventScroll: true });
    document
      .querySelectorAll('[data-editable-group-id]')
      .forEach((x) =>
        x.classList.toggle('is-selected', x.dataset.editableGroupId === id),
      );
    announce(`${pageLabel(page, currentPage - 1)}, ${group?.title || ''}`);
  } else {
    Object.assign(editorSelection, {
      activeGroupId: null,
      activeGroupType: null,
      activeStepId: null,
      activeBlockId: null,
      container: null,
    });
    page.focus({ preventScroll: true });
    announce(pageLabel(page, currentPage - 1));
  }
  updateControls();
}
function navigateToAnchor(anchor) {
  const pages = [...document.querySelectorAll('.pagedjs_page')],
    index = pages.findIndex(
      (page) =>
        page.dataset.anchorId === anchor ||
        page.querySelector(`#${CSS.escape(anchor)}`),
    );
  if (index < 0) return false;
  navigationHistory.push(captureView());
  setCurrentPage(index + 1, { scroll: true, focus: true });
  return true;
}
const openEditorLink = (href) =>
  followEditorLink(href, {
    navigateInternal: navigateToAnchor,
    openExternal: (url) => {
      const opened = window.open(url, '_blank', 'noopener,noreferrer');
      if (opened) opened.opener = null;
    },
    openMailto: (url) => {
      window.location.href = url;
    },
    report: announce,
  });
const editorInitializationDependencies = {
  handleClick: handleEditableLinkClick,
  openLink: openEditorLink,
};
const editableLinkClick = createEditableLinkClickHandler(
  editorInitializationDependencies,
);
$('#preview').addEventListener('click', editableLinkClick);
function selectedEditableLink() {
  const selection = getSelection(),
    node = selection?.anchorNode;
  return (
    (node?.nodeType === Node.ELEMENT_NODE
      ? node
      : node?.parentElement
    )?.closest?.('.editable-runs a') || null
  );
}
function updateLinkContext() {
  const link = selectedEditableLink(),
    control = $('#link-context');
  control.hidden = !link;
  control.dataset.href = link?.getAttribute('href') || '';
}
function disconnectPageObserver() {
  pageObserver?.disconnect();
  pageObserver = null;
  observedAreas.clear();
}
function observePages() {
  if (!('IntersectionObserver' in window)) return;
  const pages = [...document.querySelectorAll('.pagedjs_page')];
  pageObserver = new IntersectionObserver(
    (entries) => {
      for (const entry of entries)
        observedAreas.set(
          entry.target,
          entry.intersectionRect.width * entry.intersectionRect.height,
        );
      let best = null;
      for (const page of pages) {
        const area = observedAreas.get(page) || 0,
          center = Math.abs(
            page.getBoundingClientRect().top +
              page.getBoundingClientRect().height / 2 -
              innerHeight / 2,
          );
        if (
          !best ||
          area > best.area ||
          (area === best.area && center < best.center)
        )
          best = { page, area, center };
      }
      if (best?.area > 0) setCurrentPage(pages.indexOf(best.page) + 1);
    },
    { threshold: [0, 0.1, 0.25, 0.5, 0.75, 1] },
  );
  pages.forEach((page) => pageObserver.observe(page));
}
function go(page) {
  setCurrentPage(page, { scroll: true, focus: true });
}
$('#prev').onclick = () => go(currentPage - 1);
$('#next').onclick = () => go(currentPage + 1);
$('#page-jump').onchange = (e) => go(Number(e.target.value));
excelController = new ExcelEditor({
  root: $('#excel-editor'),
  toolbar: $('#excel-toolbar'),
  getDocument: () => state,
  updateDocument: (excel) => {
    state = { ...state, excel };
    history.replace(state);
    markDirty();
  },
  report: (message) => {
    announce(message);
    alert(message);
  },
});
document.querySelectorAll('[data-editor]').forEach(
  (button) =>
    (button.onclick = () => {
      activeEditor = button.dataset.editor;
      document
        .querySelectorAll('[data-editor]')
        .forEach((item) =>
          item.toggleAttribute('aria-current', item === button),
        );
      const excel = activeEditor === 'excel';
      $('#preview').hidden = excel;
      $('#excel-editor').hidden = !excel;
      $('#excel-toolbar').hidden = !excel;
      if (excel) excelController.synchronize();
    }),
);
render({ restore: false });
