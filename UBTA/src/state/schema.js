export const SCHEMA_VERSION = 2;
export const IMAGE_MIME_TYPES = Object.freeze(['image/png', 'image/jpeg', 'image/gif', 'image/webp']);

export const safeHref = href => {
  if (typeof href !== 'string') return null;
  return /^(https?:\/\/|mailto:|#[A-Za-z][\w:.-]*)/i.test(href) ? href : null;
};

export const safeImageSource = source => {
  if (typeof source !== 'string') return null;
  const match = /^data:(image\/[a-z0-9.+-]+);base64,([a-z0-9+/]+={0,2})$/i.exec(source);
  if (!match || !IMAGE_MIME_TYPES.includes(match[1].toLowerCase())) return null;
  return source;
};

const newId = prefix => `${prefix}-${globalThis.crypto?.randomUUID?.() || Math.random().toString(36).slice(2)}`;
const validId = value => typeof value === 'string' && /^[A-Za-z][\w-]*$/.test(value);
const makeId = (value, prefix = 'id', used) => {
  let result = validId(value) ? value : newId(prefix);
  if (used) {
    while (used.has(result)) result = newId(prefix);
    used.add(result);
  }
  return result;
};
const text = value => String(value ?? '');
const bounded = (value, minimum, maximum, fallback) => {
  const number = Number(value);
  return Number.isFinite(number) ? Math.min(maximum, Math.max(minimum, number)) : fallback;
};

export function normaliseRuns(runs) {
  return (Array.isArray(runs) ? runs : [])
    .map(run => ({
      text: text(run?.text),
      highlight: run?.highlight === true,
      link: safeHref(run?.link?.href) ? {href: safeHref(run.link.href)} : null,
    }))
    .filter(run => run.text)
    .reduce((result, run) => {
      const previous = result.at(-1);
      if (previous && previous.highlight === run.highlight && previous.link?.href === run.link?.href) previous.text += run.text;
      else result.push(run);
      return result;
    }, []);
}

const normaliseWidths = columns => {
  if (!columns.length) return [];
  const widths = columns.map(column => bounded(column?.width, 1, 100, 100 / columns.length));
  const sum = widths.reduce((total, width) => total + width, 0);
  const normalised = widths.map(width => Number((width * 100 / sum).toFixed(6)));
  normalised[normalised.length - 1] = Number((100 - normalised.slice(0, -1).reduce((total, width) => total + width, 0)).toFixed(6));
  return normalised;
};

const normaliseRow = (row, used) => ({
  id: makeId(row?.id, 'row', used),
  cells: (Array.isArray(row?.cells) ? row.cells : []).map(cell => ({
    id: makeId(cell?.id, 'cell', used),
    runs: normaliseRuns(cell?.runs),
  })),
});

export function normaliseBlock(block, usedIds = new Set()) {
  const allowed = ['heading', 'paragraph', 'bulletList', 'numberList', 'table', 'image'];
  const type = allowed.includes(block?.type) ? block.type : 'paragraph';
  const out = {id: makeId(block?.id, type === 'image' ? 'image' : 'block', usedIds), type};

  if (type === 'heading') {
    out.level = bounded(block.level, 1, 4, 1);
    out.runs = normaliseRuns(block.runs);
  } else if (type === 'paragraph') {
    out.runs = normaliseRuns(block.runs);
  } else if (type === 'bulletList' || type === 'numberList') {
    out.items = (Array.isArray(block.items) ? block.items : []).map(item => ({
      id: makeId(item?.id, 'item'),
      level: bounded(item?.level, 1, 3, 1),
      runs: normaliseRuns(item?.runs),
    }));
  } else if (type === 'table') {
    const sourceColumns = Array.isArray(block.columns) ? block.columns : [];
    const widths = normaliseWidths(sourceColumns);
    out.columns = sourceColumns.map((column, index) => ({
      id: makeId(column?.id, 'column'),
      width: widths[index],
      alignment: ['left', 'centre', 'right'].includes(column?.alignment) ? column.alignment : 'left',
    }));
    out.headerRows = (Array.isArray(block.headerRows) ? block.headerRows : []).map(row => normaliseRow(row, usedIds));
    out.rows = (Array.isArray(block.rows) ? block.rows : []).map(row => normaliseRow(row, usedIds));
    out.footerRows = (Array.isArray(block.footerRows) ? block.footerRows : []).map(row => normaliseRow(row, usedIds));
    const columnIds = new Set(out.columns.map(column => column.id));
    out.totals = {
      enabled: block?.totals?.enabled === true,
      label: text(block?.totals?.label),
      columns: [...new Set(Array.isArray(block?.totals?.columns) ? block.totals.columns.filter(columnId => columnIds.has(columnId)) : [])],
    };
  } else {
    const source = safeImageSource(block.source);
    const sourceMime = source?.slice(5, source.indexOf(';')).toLowerCase() || null;
    const mimeType = IMAGE_MIME_TYPES.includes(String(block.mimeType).toLowerCase()) ? String(block.mimeType).toLowerCase() : sourceMime;
    out.source = source && mimeType === sourceMime ? source : null;
    out.mimeType = out.source ? mimeType : null;
    out.intrinsicWidth = bounded(block.intrinsicWidth, 1, 100000, 1);
    out.intrinsicHeight = bounded(block.intrinsicHeight, 1, 100000, 1);
    out.renderedWidth = bounded(block.renderedWidth, 1, 100, 100);
    out.alignment = ['left', 'centre', 'right'].includes(block.alignment) ? block.alignment : 'centre';
    out.altText = text(block.altText);
    out.captionRuns = normaliseRuns(block.captionRuns);
  }
  return out;
}

export function migrateDocument(document) {
  const input = document && typeof document === 'object' ? structuredClone(document) : {};
  const version = input.schemaVersion ?? 1;
  if (version !== 1 && version !== SCHEMA_VERSION) throw Error('Unsupported schema version');
  if (version === 1) {
    input.steps = (Array.isArray(input.steps) ? input.steps : []).map(step => ({
      ...step,
      summary: step.summary ?? '',
      proposal: step.proposal ?? null,
    }));
    input.derivedContent = input.derivedContent ?? {transactionProposals: {automatic: true, sectionId: null}};
    input.schemaVersion = SCHEMA_VERSION;
  }
  return input;
}

export function normaliseDocument(document) {
  const input = migrateDocument(document);
  const usedIds = new Set();
  const meta = {};
  for (const key of ['clientName', 'projectTitle', 'documentType', 'date', 'version', 'subtitle', 'adviser', 'status']) meta[key] = text(input.meta?.[key]);
  const sections = (Array.isArray(input.sections) ? input.sections : []).map(section => ({
    id: makeId(section?.id, 'section', usedIds),
    title: text(section?.title),
    blocks: (Array.isArray(section?.blocks) ? section.blocks : []).map(block => normaliseBlock(block, usedIds)),
  }));
  const steps = (Array.isArray(input.steps) ? input.steps : []).map(step => ({
    id: makeId(step?.id, 'step', usedIds),
    title: text(step?.title),
    summary: text(step?.summary),
    proposal: step?.proposal == null ? null : text(step.proposal),
    blocks: (Array.isArray(step?.blocks) ? step.blocks : []).map(block => normaliseBlock(block, usedIds)),
  }));
  const sectionIds = new Set(sections.map(section => section.id));
  const configuredSection = input.derivedContent?.transactionProposals?.sectionId;
  return {
    schemaVersion: SCHEMA_VERSION,
    meta,
    derivedContent: {
      transactionProposals: {
        automatic: input.derivedContent?.transactionProposals?.automatic !== false,
        sectionId: sectionIds.has(configuredSection) ? configuredSection : null,
      },
    },
    sections,
    steps,
  };
}

export function validateDocument(document) {
  return normaliseDocument(document);
}

const pixel = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLzWQAAAABJRU5ErkJggg==';
export const seedDocument = normaliseDocument({
  schemaVersion: SCHEMA_VERSION,
  meta: {clientName: 'Example Client Ltd', projectTitle: 'Corporate Restructure', documentType: 'Steps Plan', subtitle: 'Detailed Steps Plan', date: '25 July 2026', version: 'v2', adviser: 'UBTA Accountants Ltd', status: 'Draft'},
  derivedContent: {transactionProposals: {automatic: true, sectionId: 'transaction-proposals'}},
  sections: [
    {id: 'scope', title: 'Scope of works', blocks: [
      {id: 'scope-h', type: 'heading', level: 2, runs: [{text: 'Purpose and scope'}]},
      {id: 'scope-p1', type: 'paragraph', runs: [{text: 'This plan outlines the principal implementation steps for a proposed corporate restructure.'}]},
      {id: 'consideration-table', type: 'table', columns: [{id: 'description', width: 70}, {id: 'amount', width: 30, alignment: 'right'}], headerRows: [{id: 'consideration-head', cells: [{id: 'description-head', runs: [{text: 'Description'}]}, {id: 'amount-head', runs: [{text: 'Amount'}]}]}], rows: [{id: 'cash-row', cells: [{id: 'cash-label', runs: [{text: 'Cash consideration'}]}, {id: 'cash-value', runs: [{text: '£100,000'}]}]}], footerRows: [], totals: {enabled: true, label: 'Total consideration', columns: ['amount']}},
      {id: 'structure-image', type: 'image', source: pixel, mimeType: 'image/png', intrinsicWidth: 1, intrinsicHeight: 1, renderedWidth: 60, alignment: 'centre', altText: 'Illustrative group structure', captionRuns: [{text: 'Illustrative post-transaction structure.'}]},
    ]},
    {id: 'transaction-proposals', title: 'Transaction Proposals', blocks: [{id: 'proposal-intro', type: 'paragraph', runs: [{text: 'Proposals are generated from the explicit step data below.'}]}]},
  ],
  steps: [
    {id: 'share-restructure', title: 'Implement the corporate share restructure', summary: 'Approve and document the share restructure.', proposal: 'The company will reorganise its existing share capital.', blocks: [{id: 'step-p1', type: 'paragraph', runs: [{text: 'The directors will approve the proposed corporate restructure and authorise the required documentation.'}]}]},
    {id: 'statutory-filings', title: 'Complete statutory filings', summary: 'File the approved changes with Companies House.', proposal: null, blocks: [{id: 'step-p2', type: 'paragraph', runs: [{text: 'Submit all required forms after completion.'}]}]},
  ],
});
