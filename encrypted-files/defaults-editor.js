import { decryptEnvelope, encryptEnvelope, validateEnvelope, validatePasswordConfirmation } from './crypto.js';
import { generateOpaqueId, migrateTemplatePayload, validateTemplatePayload } from '../UBTA/src/editor/default-templates.js';

const $ = (selector) => document.querySelector(selector);
const state = { payload: null, dirty: false, migrationPending: false };
const status = (message, error = false) => { $('#status').textContent = message; $('#status').className = error ? 'error' : ''; };
const allIds = () => {
  const ids = new Set();
  for (const template of state.payload?.templates || []) {
    ids.add(template.id);
    for (const block of template.blocks) {
      ids.add(block.id); for (const item of block.items || []) ids.add(item.id);
      for (const column of block.columns || []) ids.add(column.id);
      for (const row of block.rows || []) { ids.add(row.id); for (const cell of row.cells || []) ids.add(cell.id); }
    }
  }
  return ids;
};
const newId = () => generateOpaqueId(allIds());
const runs = (text = '') => [{ text }];
const newBlock = (type) => {
  if (type === 'heading') return { id: newId(), type, level: 3, runs: runs('Subheading') };
  if (type.endsWith('List')) return { id: newId(), type, items: [{ id: newId(), level: 1, runs: runs('List item') }] };
  if (type === 'table') {
    const columns = [1, 2].map((n) => ({ id: newId(), headingRuns: runs(`Column ${n}`), width: 50, format: 'text', totalEnabled: false }));
    return { id: newId(), type, captionRuns: runs('Table caption'), columns, rows: [{ id: newId(), isTotal: false, cells: columns.map(() => ({ id: newId(), runs: runs('') })) }] };
  }
  return { id: newId(), type: 'paragraph', runs: runs('Paragraph text') };
};
const textOf = (value) => value.map((run) => run.text).join('');
const setText = (target, value) => { target.splice(0, target.length, { text: value.replace(/\r\n?/g, '\n') }); state.dirty = true; };
const button = (label, action, className = '') => { const element = document.createElement('button'); element.type = 'button'; element.textContent = label; element.className = className; element.addEventListener('click', action); return element; };
function renderBlock(block, template) {
  const article = document.createElement('article'); article.className = 'block'; article.dataset.type = block.type;
  const label = document.createElement('strong'); label.textContent = block.type;
  const controls = document.createElement('div'); controls.className = 'block-controls';
  const index = () => template.blocks.indexOf(block);
  controls.append(label, button('↑', () => move(template.blocks, index(), -1)), button('↓', () => move(template.blocks, index(), 1)), button('Delete', () => { if (template.blocks.length < 2) return status('A template must retain at least one block.', true); if (confirm('Delete this block?')) { template.blocks.splice(index(), 1); changed(); } }, 'danger'));
  article.append(controls);
  if (block.runs) article.append(editable(block.runs, block.type === 'heading' ? 'heading' : 'paragraph'));
  if (block.items) for (const item of block.items) {
    const row = document.createElement('div'); row.className = 'list-item';
    row.append(button('Outdent', () => { item.level = Math.max(1, item.level - 1); changed(); }), button('Indent', () => { item.level = Math.min(3, item.level + 1); changed(); }), editable(item.runs)); article.append(row);
  }
  if (block.type === 'table') {
    article.append(editable(block.captionRuns, 'caption'));
    const table = document.createElement('table'), head = table.createTHead().insertRow(), body = table.createTBody();
    for (const column of block.columns) head.insertCell().append(editable(column.headingRuns));
    for (const row of block.rows) { const tr = body.insertRow(); for (const cell of row.cells) tr.insertCell().append(editable(cell.runs)); }
    article.append(table, button('Add row', () => { block.rows.push({ id: newId(), isTotal: false, cells: block.columns.map(() => ({ id: newId(), runs: runs('') })) }); changed(); }), button('Add column', () => { if (block.columns.length === 8) return; const column = { id: newId(), headingRuns: runs('Column'), width: 100 / (block.columns.length + 1), format: 'text', totalEnabled: false }; block.columns.push(column); block.columns.forEach((c) => c.width = 100 / block.columns.length); block.rows.forEach((row) => row.cells.push({ id: newId(), runs: runs('') })); changed(); }));
  }
  return article;
}
function editable(target, className = '') { const element = document.createElement('div'); element.className = `editable ${className}`; element.contentEditable = 'true'; element.role = 'textbox'; element.ariaLabel = `${className || 'content'} text`; element.textContent = textOf(target); element.addEventListener('input', () => setText(target, element.textContent)); element.addEventListener('paste', (event) => { event.preventDefault(); document.execCommand('insertText', false, event.clipboardData.getData('text/plain')); }); return element; }
function move(array, index, direction) { const next = index + direction; if (next < 0 || next >= array.length) return; [array[index], array[next]] = [array[next], array[index]]; changed(); }
function changed() { state.dirty = true; render(); }
function render() {
  const root = $('#templates'); root.replaceChildren(); $('#workspace').hidden = !state.payload;
  for (const template of state.payload?.templates || []) {
    const card = document.createElement('details'); card.className = 'template-card'; card.open = true;
    const summary = document.createElement('summary'); summary.textContent = template.header || 'Untitled template'; card.append(summary);
    const header = document.createElement('input'); header.value = template.header; header.ariaLabel = 'Template header'; header.placeholder = 'Template label'; header.addEventListener('input', () => { template.header = header.value; state.dirty = true; summary.textContent = header.value || 'Untitled template'; });
    const toolbar = document.createElement('div'); toolbar.className = 'toolbar';
    for (const [label, type] of [['Paragraph', 'paragraph'], ['Heading', 'heading'], ['Bullet list', 'bulletList'], ['Numbered list', 'numberList'], ['Table', 'table']]) toolbar.append(button(`Add ${label}`, () => { template.blocks.push(newBlock(type)); changed(); }));
    const ti = () => state.payload.templates.indexOf(template);
    toolbar.append(button('Move template up', () => move(state.payload.templates, ti(), -1)), button('Move template down', () => move(state.payload.templates, ti(), 1)), button('Delete template', () => { if (state.payload.templates.length < 2) return status('The database must retain at least one template.', true); if (confirm('Delete this entire template?')) { state.payload.templates.splice(ti(), 1); changed(); } }, 'danger'));
    card.append(header, toolbar, ...template.blocks.map((block) => renderBlock(block, template)));
    const diagnostic = document.createElement('details'), diagnosticSummary = document.createElement('summary'), code = document.createElement('code'); diagnosticSummary.textContent = 'Diagnostics'; code.textContent = `Read-only template ID: ${template.id}`; diagnostic.append(diagnosticSummary, code); card.append(diagnostic); root.append(card);
  }
}
$('#load-form').addEventListener('submit', async (event) => {
  event.preventDefault(); const password = $('#current-password').value; $('#current-password').value = '';
  try {
    const file = $('#defaults-file').files[0]; if (!file) throw Error('Select the canonical JSON file.');
    const envelope = JSON.parse(await file.text()); validateEnvelope(envelope);
    let payload = await decryptEnvelope(envelope, password);
    if (payload.schemaVersion == null) { const migration = migrateTemplatePayload(payload); payload = migration.payload; state.migrationPending = true; $('#migration').hidden = false; $('#migration-summary').textContent = `${migration.summary.templateCount} template(s) were migrated. Internal IDs changed once; displayed headers and body content did not.`; }
    state.payload = { type: 'editor-defaults', schemaVersion: 2, templates: validateTemplatePayload(payload).map((template) => structuredClone(template)) }; state.dirty = false; render(); status(`Loaded ${state.payload.templates.length} template(s).`);
  } catch (error) { status(error.message, true); }
});
$('#confirm-migration').addEventListener('change', (event) => { if (event.target.checked) { state.migrationPending = false; state.dirty = true; } });
$('#add-template').addEventListener('click', () => { const template = { id: newId(), header: '', blocks: [newBlock('paragraph')] }; state.payload.templates.push(template); changed(); queueMicrotask(() => [...document.querySelectorAll('.template-card input')].at(-1)?.focus()); });
$('#validate').addEventListener('click', () => { try { validateTemplatePayload(state.payload); status('The complete version-2 database is valid.'); } catch (error) { status(error.message, true); } });
$('#preview').addEventListener('click', () => { const output = $('#preview-output'); output.replaceChildren(); for (const template of state.payload.templates) { const h = document.createElement('h2'); h.textContent = template.header; output.append(h); for (const block of template.blocks) { const element = document.createElement(block.type === 'heading' ? `h${block.level}` : block.type.endsWith('List') ? (block.type === 'bulletList' ? 'ul' : 'ol') : 'p'); if (block.items) for (const item of block.items) { const li = document.createElement('li'); li.textContent = textOf(item.runs); element.append(li); } else element.textContent = textOf(block.runs || block.captionRuns || []); output.append(element); } } });
$('#encrypt-form').addEventListener('submit', async (event) => { event.preventDefault(); try { if (state.migrationPending) throw Error('Confirm the migration review before saving.'); const payload = { type: 'editor-defaults', schemaVersion: 2, templates: validateTemplatePayload(state.payload).map((x) => structuredClone(x)) }; const password = validatePasswordConfirmation($('#output-password').value, $('#confirm-password').value); $('#output-password').value = $('#confirm-password').value = ''; const envelope = await encryptEnvelope(payload, password); const link = document.createElement('a'); link.download = '8f3c1a7e4d92b605.json'; link.href = URL.createObjectURL(new Blob([JSON.stringify(envelope, null, 2) + '\n'], { type: 'application/json' })); link.click(); setTimeout(() => URL.revokeObjectURL(link.href), 1000); state.dirty = false; status('Encrypted with fresh salt and IV; download started.'); } catch (error) { $('#output-password').value = $('#confirm-password').value = ''; status(error.message, true); } });
$('#clear').addEventListener('click', () => { state.payload = null; state.dirty = state.migrationPending = false; $('#templates').replaceChildren(); $('#preview-output').replaceChildren(); $('#load-form').reset(); $('#encrypt-form').reset(); $('#workspace').hidden = true; status('Decrypted data and passwords cleared.'); });
addEventListener('beforeunload', (event) => { if (state.dirty && state.payload) event.preventDefault(); });
