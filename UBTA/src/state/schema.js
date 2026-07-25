export const SCHEMA_VERSION = 2;

const IDENTIFIER = /^[A-Za-z][\w-]*$/;
const IMAGE_DATA = /^data:image\/(png|jpeg|gif|webp);base64,[a-z0-9+/=\s]+$/i;

export const safeHref = value => {
  if (typeof value !== 'string') return null;
  return /^(https?:\/\/|mailto:|#[A-Za-z][\w:.-]*$)/i.test(value) ? value : null;
};

export const safeImageSrc = value => {
  if (typeof value !== 'string') return null;
  if (IMAGE_DATA.test(value)) return value.replace(/\s/g, '');
  try {
    const url = new URL(value);
    return url.protocol === 'https:' ? url.href : null;
  } catch {
    return null;
  }
};

const makeId = (value, prefix = 'id') =>
  IDENTIFIER.test(value || '')
    ? value
    : `${prefix}-${globalThis.crypto?.randomUUID?.() || Math.random().toString(36).slice(2)}`;

export function normaliseRuns(runs) {
  return (Array.isArray(runs) ? runs : [])
    .map(run => ({
      text: String(run?.text ?? ''),
      highlight: run?.highlight === true,
      link: safeHref(run?.link?.href) ? { href: safeHref(run.link.href) } : null,
    }))
    .filter(run => run.text)
    .reduce((result, run) => {
      const previous = result.at(-1);
      if (
        previous &&
        previous.highlight === run.highlight &&
        previous.link?.href === run.link?.href
      ) {
        previous.text += run.text;
      } else {
        result.push(run);
      }
      return result;
    }, []);
}

const normaliseCell = (cell, prefix) => ({
  id: makeId(cell?.id, prefix),
  runs: normaliseRuns(cell?.runs),
});

export function normaliseBlock(block) {
  const allowed = ['heading', 'paragraph', 'bulletList', 'numberList', 'table', 'image'];
  const type = allowed.includes(block?.type) ? block.type : 'paragraph';
  const output = { id: makeId(block?.id, 'block'), type };

  if (type === 'heading') {
    output.level = Math.min(4, Math.max(1, Number(block.level) || 1));
    output.runs = normaliseRuns(block.runs);
  } else if (type === 'paragraph') {
    output.runs = normaliseRuns(block.runs);
  } else if (type.endsWith('List')) {
    output.items = (Array.isArray(block.items) ? block.items : []).map(item => ({
      id: makeId(item?.id, 'item'),
      level: Math.min(3, Math.max(1, Number(item?.level) || 1)),
      runs: normaliseRuns(item?.runs),
    }));
  } else if (type === 'table') {
    const sourceColumns = Array.isArray(block.columns) ? block.columns : [];
    const columnCount = Math.min(8, Math.max(1, sourceColumns.length || 2));
    const widthTotal = sourceColumns
      .slice(0, columnCount)
      .reduce((sum, column) => sum + Math.max(5, Number(column?.width) || 0), 0) || columnCount;
    output.caption = String(block.caption ?? '');
    output.columns = Array.from({ length: columnCount }, (_, index) => ({
      id: makeId(sourceColumns[index]?.id, 'column'),
      heading: String(sourceColumns[index]?.heading ?? `Column ${index + 1}`),
      width: Math.round((Math.max(5, Number(sourceColumns[index]?.width) || 1) / widthTotal) * 1000) / 10,
      numeric: sourceColumns[index]?.numeric === true,
    }));
    output.rows = (Array.isArray(block.rows) ? block.rows : []).map((row, rowIndex) => ({
      id: makeId(row?.id, 'row'),
      isTotal: row?.isTotal === true,
      cells: output.columns.map((column, columnIndex) =>
        normaliseCell(row?.cells?.[columnIndex], `${output.id}-${rowIndex}-${column.id}`),
      ),
    }));
  } else {
    output.src = safeImageSrc(block.src);
    output.alt = String(block.alt ?? '').trim();
    output.caption = String(block.caption ?? '');
    output.width = Math.min(100, Math.max(20, Number(block.width) || 100));
  }
  return output;
}

const normaliseGroup = (group, prefix) => ({
  id: makeId(group?.id, prefix),
  title: String(group?.title ?? ''),
  blocks: (Array.isArray(group?.blocks) ? group.blocks : []).map(normaliseBlock),
});

export function normaliseDocument(document) {
  const meta = {};
  for (const key of ['clientName', 'projectTitle', 'documentType', 'date', 'version', 'subtitle', 'adviser', 'status']) {
    meta[key] = String(document?.meta?.[key] ?? '');
  }
  return {
    schemaVersion: SCHEMA_VERSION,
    meta,
    sections: (Array.isArray(document?.sections) ? document.sections : []).map(group => normaliseGroup(group, 'section')),
    steps: (Array.isArray(document?.steps) ? document.steps : []).map(group => normaliseGroup(group, 'step')),
  };
}

export function validateDocument(document) {
  if (![1, SCHEMA_VERSION].includes(document?.schemaVersion)) throw Error('Unsupported schema version');
  const normalised = normaliseDocument(document);
  for (const group of [...normalised.sections, ...normalised.steps]) {
    for (const block of group.blocks) {
      if (block.type === 'image' && (!block.src || !block.alt)) {
        throw Error('Images require a safe source and meaningful alternative text');
      }
    }
  }
  return normalised;
}

export function transactionProposals(document) {
  return normaliseDocument(document).steps.map((step, index) => ({
    id: `proposal-${step.id}`,
    stepId: step.id,
    title: `Step ${index + 1}. ${step.title}`,
    anchor: `anchor-${step.id}`,
    summary: step.blocks
      .filter(block => block.type === 'paragraph')
      .flatMap(block => block.runs)
      .map(run => run.text)
      .join(' ')
      .trim(),
  }));
}

export const seedDocument = normaliseDocument({
  schemaVersion: SCHEMA_VERSION,
  meta: { clientName: 'Example Client Ltd', projectTitle: 'Corporate Restructure', documentType: 'Steps Plan', subtitle: 'Detailed Steps Plan', date: '25 July 2026', version: 'v1', adviser: 'UBTA Accountants Ltd', status: 'Draft' },
  sections: [{ id: 'scope', title: 'Scope of works', blocks: [
    { id: 'scope-h', type: 'heading', level: 2, runs: [{ text: 'Purpose and scope' }] },
    { id: 'scope-p1', type: 'paragraph', runs: [{ text: 'This plan outlines the principal implementation steps for a proposed corporate restructure.' }] },
    { id: 'scope-p2', type: 'paragraph', runs: [{ text: 'Figures and legal mechanics marked for review require completion before finalisation.', highlight: true }] },
    { id: 'scope-list', type: 'bulletList', items: [{ id: 'si1', level: 1, runs: [{ text: 'Confirm commercial objectives' }] }, { id: 'si2', level: 2, runs: [{ text: 'Review supporting records' }] }, { id: 'si3', level: 1, runs: [{ text: 'Consult the ' }, { text: 'HMRC guidance', link: { href: 'https://www.gov.uk/government/organisations/hm-revenue-customs' } }] }] },
  ] }],
  steps: [{ id: 'share-restructure', title: 'Implement the corporate share restructure', blocks: [
    { id: 'step-h', type: 'heading', level: 2, runs: [{ text: 'Implementation' }] },
    { id: 'step-p1', type: 'paragraph', runs: [{ text: 'The directors will approve the proposed corporate restructure and authorise the required documentation.' }] },
    { id: 'step-p2', type: 'paragraph', runs: [{ text: 'Professional advisers should confirm the sequence immediately before implementation.' }] },
    { id: 'step-list', type: 'numberList', items: [{ id: 'st1', level: 1, runs: [{ text: 'Prepare board minutes and resolutions.' }] }, { id: 'st2', level: 2, runs: [{ text: 'Check company records and statutory filings.' }] }, { id: 'st3', level: 1, runs: [{ text: 'Complete and retain the signed documents.' }] }] },
    { id: 'consideration', type: 'table', caption: 'Illustrative consideration', columns: [{ id: 'detail', heading: 'Detail', width: 70 }, { id: 'amount', heading: 'Amount (£)', width: 30, numeric: true }], rows: [{ id: 'ordinary-shares', cells: [{ runs: [{ text: 'Ordinary shares' }] }, { runs: [{ text: '10,000' }] }] }, { id: 'total', isTotal: true, cells: [{ runs: [{ text: 'Total' }] }, { runs: [{ text: '10,000' }] }] }] },
  ] }],
});
