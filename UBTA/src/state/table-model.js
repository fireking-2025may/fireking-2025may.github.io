const FORMATS = new Set(['text', 'number', 'gbp']);
const runsText = (runs) => (runs || []).map((run) => run.text || '').join('');
const plainRuns = (text) =>
  text === '' ? [] : [{ text, highlight: false, link: null }];

export const tableColumnFormat = (column) =>
  FORMATS.has(column?.format)
    ? column.format
    : column?.numeric === true
      ? 'number'
      : 'text';

/** Blank and invalid values return null; arbitrary text is never coerced to zero. */
export function parseTableNumber(value) {
  const compact = String(value ?? '').replace(/\s/g, '');
  if (!compact) return null;
  const match = compact.match(
    /^(?:£)?([+-]?)(?:£)?((?:\d{1,3}(?:,\d{3})+)|\d+)(\.\d+)?$/,
  );
  if (!match) return null;
  const number = Number(
    `${match[1]}${match[2].replaceAll(',', '')}${match[3] || ''}`,
  );
  return Number.isFinite(number) ? number : null;
}

export function formatTableNumber(value, format = 'number') {
  if (!Number.isFinite(value)) return '';
  if (format === 'gbp')
    return `${value < 0 ? '-' : ''}£${Math.abs(value).toLocaleString('en-GB', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  if (format === 'number')
    return value.toLocaleString('en-GB', { maximumFractionDigits: 20 });
  return String(value);
}

export function recalculateTableTotals(block) {
  if (block?.type !== 'table') return block;
  block.rows.forEach((row, rowIndex) => {
    if (!row.isTotal) return;
    block.columns.forEach((column, columnIndex) => {
      const format = tableColumnFormat(column);
      if (format === 'text') return;
      if (column.totalEnabled === false) {
        if (row.cells?.[columnIndex]) row.cells[columnIndex].runs = [];
        return;
      }
      const values = block.rows
        .slice(0, rowIndex)
        .filter((candidate) => !candidate.isTotal)
        .map((candidate) =>
          parseTableNumber(runsText(candidate.cells?.[columnIndex]?.runs)),
        )
        .filter((value) => value !== null);
      const total = values.reduce((sum, value) => sum + value, 0);
      if (row.cells?.[columnIndex])
        row.cells[columnIndex].runs = plainRuns(
          formatTableNumber(total, format),
        );
    });
  });
  return block;
}

export function moveTableRow(block, index, offset) {
  const target = index + offset,
    row = block?.rows?.[index];
  if (
    !row ||
    target < 0 ||
    target >= block.rows.length ||
    block.rows[target].isTotal !== row.isTotal
  )
    return false;
  [block.rows[index], block.rows[target]] = [
    block.rows[target],
    block.rows[index],
  ];
  return target;
}

export function moveTableColumn(block, index, offset) {
  const target = index + offset;
  if (!block?.columns?.[index] || target < 0 || target >= block.columns.length)
    return false;
  [block.columns[index], block.columns[target]] = [
    block.columns[target],
    block.columns[index],
  ];
  block.rows.forEach((row) => {
    [row.cells[index], row.cells[target]] = [
      row.cells[target],
      row.cells[index],
    ];
  });
  return target;
}
