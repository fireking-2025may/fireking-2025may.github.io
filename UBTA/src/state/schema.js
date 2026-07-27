export const SCHEMA_VERSION = 3;
export const TABLE_WIDTH_MIN = 8;
export const TABLE_WIDTH_MAX = 92;
const IDENTIFIER = /^[A-Za-z][\w-]*$/;
const IMAGE_DATA = /^data:image\/(png|jpeg|gif|webp);base64,[a-z0-9+/=\s]+$/i;

export const newId = prefix => `${prefix}-${globalThis.crypto?.randomUUID?.() || Math.random().toString(36).slice(2)}`;
const makeId = (value, prefix = 'id') => IDENTIFIER.test(value || '') ? value : newId(prefix);

export const safeHref = value => typeof value === 'string' && /^(https?:\/\/|mailto:|#[A-Za-z][\w:.-]*$)/i.test(value) ? value : null;
export const safeImageSrc = value => {
  if (typeof value !== 'string') return null;
  if (IMAGE_DATA.test(value)) return value.replace(/\s/g, '');
  try { const url = new URL(value); return url.protocol === 'https:' && /\.(png|jpe?g|gif|webp)(?:$|[?#])/i.test(url.pathname + url.search + url.hash) ? url.href : null; }
  catch { return null; }
};

export function normaliseRuns(runs) {
  return (Array.isArray(runs) ? runs : []).map(run => ({
    text: String(run?.text ?? '').replace(/\r\n?/g, '\n'),
    highlight: run?.highlight === true,
    link: safeHref(run?.link?.href) ? { href: safeHref(run.link.href) } : null,
  })).filter(run => run.text).reduce((out, run) => {
    const last = out.at(-1);
    if (last && last.highlight === run.highlight && last.link?.href === run.link?.href) last.text += run.text;
    else out.push(run);
    return out;
  }, []);
}

/** Redistribute widths without ever pushing a bounded column outside 8–92%. */
export function normaliseTableWidths(values, count = values?.length || 1) {
  count = Math.max(1, Math.min(8, Number(count) || 1));
  // More than twelve columns cannot total 100 at the declared minimum. The schema
  // caps tables at eight, but this defines deterministic behaviour for direct use.
  if (count * TABLE_WIDTH_MIN > 100 || count * TABLE_WIDTH_MAX < 100) return Array(count).fill(100 / count);
  const raw = Array.from({ length: count }, (_, i) => Number(values?.[i]));
  let widths = raw.map(value => Number.isFinite(value) && value > 0 ? Math.min(TABLE_WIDTH_MAX, Math.max(TABLE_WIDTH_MIN, value)) : 100 / count);
  for (let pass = 0; pass < 20; pass++) {
    const remainder = 100 - widths.reduce((a, b) => a + b, 0);
    if (Math.abs(remainder) < 1e-9) break;
    const candidates = widths.map((width, i) => ({ i, capacity: remainder > 0 ? TABLE_WIDTH_MAX - width : width - TABLE_WIDTH_MIN })).filter(x => x.capacity > 1e-9);
    if (!candidates.length) break;
    const capacity = candidates.reduce((sum, x) => sum + x.capacity, 0);
    candidates.forEach(x => { widths[x.i] += Math.sign(remainder) * Math.min(x.capacity, Math.abs(remainder) * x.capacity / capacity); });
  }
  widths = widths.map(width => Math.round(width * 1000) / 1000);
  let residue = Math.round((100 - widths.reduce((a, b) => a + b, 0)) * 1000) / 1000;
  for (const width of widths.map((value, i) => ({ value, i }))) {
    const room = residue > 0 ? TABLE_WIDTH_MAX - width.value : width.value - TABLE_WIDTH_MIN;
    const amount = Math.sign(residue) * Math.min(Math.abs(residue), room);
    widths[width.i] += amount; residue -= amount;
    if (Math.abs(residue) < .0005) break;
  }
  return widths;
}

const normaliseCell = (cell, prefix) => ({ id: makeId(cell?.id, prefix), runs: normaliseRuns(cell?.runs) });
export function normaliseBlock(block) {
  const allowed = ['heading', 'paragraph', 'bulletList', 'numberList', 'table', 'image'];
  const type = allowed.includes(block?.type) ? block.type : 'paragraph';
  const output = { id: makeId(block?.id, 'block'), type };
  if (type === 'heading') { output.level = Math.min(4, Math.max(1, Number(block.level) || 1)); output.runs = normaliseRuns(block.runs); }
  else if (type === 'paragraph') output.runs = normaliseRuns(block.runs);
  else if (type.endsWith('List')) output.items = (Array.isArray(block.items) && block.items.length ? block.items : [{ runs: [] }]).map(item => ({ id: makeId(item?.id, 'item'), level: Math.min(3, Math.max(1, Number(item?.level) || 1)), runs: normaliseRuns(item?.runs) }));
  else if (type === 'table') {
    const source = Array.isArray(block.columns) ? block.columns.slice(0, 8) : [];
    const count = Math.max(1, source.length || 2), widths = normaliseTableWidths(source.map(x => x?.width), count);
    output.captionRuns = normaliseRuns(block.captionRuns || [{ text: block.caption ?? '' }]);
    output.columns = Array.from({ length: count }, (_, i) => ({ id: makeId(source[i]?.id, 'column'), headingRuns: normaliseRuns(source[i]?.headingRuns || [{ text: source[i]?.heading ?? `Column ${i + 1}` }]), width: widths[i], numeric: source[i]?.numeric === true }));
    output.rows = (Array.isArray(block.rows) ? block.rows : []).map((row, ri) => ({ id: makeId(row?.id, 'row'), isTotal: row?.isTotal === true, cells: output.columns.map((column, ci) => normaliseCell(row?.cells?.[ci], `${output.id}-${ri}-${column.id}`)) }));
  } else {
    output.src = safeImageSrc(block.src); output.alt = String(block.alt ?? ''); output.captionRuns = normaliseRuns(block.captionRuns || [{ text: block.caption ?? '' }]); output.width = Math.min(100, Math.max(20, Number(block.width) || 100));
  }
  return output;
}

const normaliseGroup = (group, prefix, ensureBlock = false) => ({ id: makeId(group?.id, prefix), title: String(group?.title ?? ''), summary: String(group?.summary ?? ''), blocks: (Array.isArray(group?.blocks) && group.blocks.length ? group.blocks : ensureBlock ? [{ id: newId('block'), type: 'paragraph', runs: [] }] : []).map(normaliseBlock) });
export function normaliseDocument(document) {
  const meta = {}; for (const key of ['clientName','projectTitle','documentType','date','version','subtitle','adviser','status']) meta[key] = String(document?.meta?.[key] ?? '');
  return { schemaVersion: SCHEMA_VERSION, meta, sections: (Array.isArray(document?.sections) ? document.sections : []).map(x => normaliseGroup(x, 'section')), steps: (Array.isArray(document?.steps) ? document.steps : []).map(x => normaliseGroup(x, 'step', true)), appendices: (Array.isArray(document?.appendices) ? document.appendices : []).map(x => normaliseGroup(x, 'appendix')) };
}

export function *runContainers(group) {
  for (const block of group?.blocks || []) {
    if (block.runs) yield { block, kind: 'block', runs: block.runs };
    for (const item of block.items || []) yield { block, item, kind: 'listItem', runs: item.runs };
    if (block.captionRuns) yield { block, kind: block.type === 'image' ? 'imageCaption' : 'tableCaption', runs: block.captionRuns };
    for (const column of block.columns || []) yield { block, column, kind: 'tableHeading', runs: column.headingRuns };
    for (const row of block.rows || []) for (const cell of row.cells || []) yield { block, row, cell, kind: 'tableCell', runs: cell.runs };
  }
}
export const hasReview = group => [...runContainers(group)].some(container => container.runs.some(run => run.highlight));
export const stableAnchor = entity => `anchor-${entity.id}`;
export function validateDocument(document) { if (![1,2,SCHEMA_VERSION].includes(document?.schemaVersion)) throw Error('Unsupported schema version'); const result = normaliseDocument(document); for (const group of [...result.sections,...result.steps,...result.appendices]) for (const block of group.blocks) if (block.type === 'image' && !block.src) throw Error('Images require a safe source'); return result; }

const eligibleText = step => [...runContainers(step)].filter(x => (x.kind === 'block' && x.block.type === 'paragraph') || ['listItem','tableCaption','imageCaption'].includes(x.kind)).map(x => x.runs.map(r => r.text).join('').trim()).find(Boolean) || '';
export function transactionProposals(document) { return normaliseDocument(document).steps.map((step, index) => ({ id:`proposal-${step.id}`, stepId:step.id, title:`Step ${index+1}. ${step.title}`, anchor:stableAnchor(step), summary:step.summary.trim() || eligibleText(step) })); }

export const seedDocument = normaliseDocument({ schemaVersion: SCHEMA_VERSION, meta:{clientName:'Example Client Ltd',projectTitle:'Corporate Restructure',documentType:'Steps Plan',subtitle:'Detailed Steps Plan',date:'25 July 2026',version:'v1',adviser:'UBTA Accountants Ltd',status:'Draft'}, sections:[{id:'scope',title:'Scope of works',blocks:[{id:'scope-h',type:'heading',level:2,runs:[{text:'Purpose and scope'}]},{id:'scope-p1',type:'paragraph',runs:[{text:'This plan outlines the principal implementation steps for a proposed corporate restructure.'}]}]}], steps:[{id:'share-restructure',title:'Implement the corporate share restructure',blocks:[{id:'step-h',type:'heading',level:2,runs:[{text:'Implementation'}]},{id:'step-p1',type:'paragraph',runs:[{text:'The directors will approve the proposed corporate restructure and authorise the required documentation.'}]},{id:'step-list',type:'numberList',items:[{id:'st1',level:1,runs:[{text:'Prepare board minutes and resolutions.'}]}]},{id:'consideration',type:'table',caption:'Illustrative consideration',columns:[{id:'detail',heading:'Detail',width:70},{id:'amount',heading:'Amount (£)',width:30,numeric:true}],rows:[{id:'ordinary-shares',cells:[{runs:[{text:'Ordinary shares'}]},{runs:[{text:'10,000'}]}]},{id:'total',isTotal:true,cells:[{runs:[{text:'Total'}]},{runs:[{text:'10,000'}]}]}]}]}], appendices:[] });
