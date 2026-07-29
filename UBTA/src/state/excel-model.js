const id = (prefix) =>
  `${prefix}-${globalThis.crypto?.randomUUID?.() || Math.random().toString(36).slice(2)}`;
export const TYPES = ['Individual', 'Trust', 'Company'];
export const MAX_SHAREHOLDERS = 100;
export const MAX_CLASSES = 100;
const integer = (value) => (Number.isSafeInteger(value) ? value : 0);

export function parseShares(value) {
  if (typeof value === 'number')
    return Number.isSafeInteger(value) ? value : null;
  const text = String(value ?? '')
    .trim()
    .replaceAll(',', '');
  if (!text) return 0;
  const negative = text.match(/^\((\d+)\)$/),
    candidate = negative ? `-${negative[1]}` : text;
  const number = Number(candidate);
  return /^-?\d+$/.test(candidate) && Number.isSafeInteger(number)
    ? number
    : null;
}
export const formatShares = (value) =>
  integer(value) < 0
    ? `(${Math.abs(integer(value)).toLocaleString('en-GB')})`
    : integer(value).toLocaleString('en-GB');
export const cellKey = (holderId, classId) => `${holderId}|${classId}`;
export const valueAt = (values, holderId, classId) =>
  integer(values?.[cellKey(holderId, classId)]);
export const classesOf = (company) =>
  company.groups.flatMap((group) => group.classes);

export function createCompany(name = 'Company 1') {
  const shareholders = TYPES.map((type, index) => ({
    id: id('holder'),
    name: `Shareholder ${index + 1}`,
    type,
  }));
  const groups = [
    'Ordinary Shares',
    'Preference Shares',
    'Deferred Shares',
  ].map((name) => ({
    id: id('group'),
    name,
    classes: [{ id: id('class'), label: '' }],
  }));
  return {
    id: id('company'),
    name,
    shareholders,
    groups,
    opening: {},
    movements: {},
    collapsed: {},
  };
}
export function createExcel() {
  const company = createCompany();
  return {
    schemaVersion: 1,
    companies: [company],
    selectedCompanyId: company.id,
    syncedSteps: [],
  };
}

export function normaliseExcel(source, steps = []) {
  const input =
    source && Array.isArray(source.companies) ? source : createExcel();
  const companies = input.companies.map((company) => {
    const shareholders = (
      Array.isArray(company.shareholders) ? company.shareholders : []
    )
      .slice(0, 100)
      .map((holder) => ({
        id: String(holder.id || id('holder')),
        name: String(holder.name ?? ''),
        type: TYPES.includes(holder.type) ? holder.type : 'Individual',
      }));
    const groups = (Array.isArray(company.groups) ? company.groups : []).map(
      (group) => ({
        id: String(group.id || id('group')),
        name: String(group.name ?? ''),
        classes: (Array.isArray(group.classes) ? group.classes : []).map(
          (item) => ({
            id: String(item.id || id('class')),
            label: String(item.label ?? ''),
          }),
        ),
      }),
    );
    let count = 0;
    for (const group of groups)
      group.classes = group.classes.filter(() => ++count <= 100);
    const classIds = new Set(classesOf({ groups }).map((item) => item.id)),
      holderIds = new Set(shareholders.map((holder) => holder.id));
    const clean = (values) =>
      Object.fromEntries(
        Object.entries(values || {})
          .filter(([key]) => {
            const [holderId, classId] = key.split('|');
            return holderIds.has(holderId) && classIds.has(classId);
          })
          .map(([key, value]) => [key, integer(value)]),
      );
    const movements = {};
    for (const step of steps)
      movements[step.id] = clean(company.movements?.[step.id]);
    return {
      id: String(company.id || id('company')),
      name: String(company.name || 'Untitled company'),
      shareholders,
      groups,
      opening: clean(company.opening),
      movements,
      collapsed: Object.fromEntries(
        steps.map((step) => [step.id, company.collapsed?.[step.id] === true]),
      ),
    };
  });
  const selectedCompanyId = companies.some(
    (item) => item.id === input.selectedCompanyId,
  )
    ? input.selectedCompanyId
    : companies[0]?.id || null;
  return {
    schemaVersion: 1,
    companies,
    selectedCompanyId,
    syncedSteps: steps.map((step) => ({ id: step.id, label: step.title })),
  };
}
export const syncSteps = (excel, steps) => normaliseExcel(excel, steps);
export const stepHasMovements = (excel, stepId) =>
  excel.companies.some((company) =>
    Object.values(company.movements?.[stepId] || {}).some(
      (value) => value !== 0,
    ),
  );
export function resultingValues(company, steps, throughIndex) {
  const result = { ...company.opening };
  for (const step of steps.slice(0, throughIndex + 1))
    for (const holder of company.shareholders)
      for (const shareClass of classesOf(company)) {
        const key = cellKey(holder.id, shareClass.id);
        result[key] =
          integer(result[key]) +
          valueAt(company.movements[step.id], holder.id, shareClass.id);
      }
  return result;
}
export function totals(company, values) {
  const classes = classesOf(company),
    rows = Object.fromEntries(
      company.shareholders.map((holder) => [
        holder.id,
        classes.reduce(
          (sum, item) => sum + valueAt(values, holder.id, item.id),
          0,
        ),
      ]),
    ),
    columns = Object.fromEntries(
      classes.map((item) => [
        item.id,
        company.shareholders.reduce(
          (sum, holder) => sum + valueAt(values, holder.id, item.id),
          0,
        ),
      ]),
    );
  return {
    rows,
    columns,
    grand: Object.values(columns).reduce((sum, value) => sum + value, 0),
  };
}
export function findBlocking(
  company,
  classIds = null,
  holderId = null,
  steps = [],
) {
  const wanted = new Set(classIds || classesOf(company).map((item) => item.id));
  for (const holder of company.shareholders)
    if (!holderId || holder.id === holderId)
      for (const shareClass of classesOf(company))
        if (wanted.has(shareClass.id)) {
          if (valueAt(company.opening, holder.id, shareClass.id))
            return { location: 'Opening Position', holder, shareClass };
          for (const step of steps)
            if (valueAt(company.movements[step.id], holder.id, shareClass.id))
              return { location: step.title, holder, shareClass };
        }
  return null;
}
export function applyPaste(
  company,
  values,
  startRow,
  startColumn,
  range,
  mode = 'replace',
) {
  const classes = classesOf(company);
  if (!range.length || range.some((row) => row.length !== range[0].length))
    throw Error('The copied range must be rectangular.');
  if (
    startRow + range.length > company.shareholders.length ||
    startColumn + range[0].length > classes.length
  )
    throw Error(
      'The complete pasted range does not fit in the editable table.',
    );
  const parsed = range.map((row) => row.map(parseShares));
  if (parsed.flat().some((value) => value === null))
    throw Error('Every pasted value must be a whole integer.');
  const output = { ...values };
  parsed.forEach((row, ri) =>
    row.forEach((value, ci) => {
      const key = cellKey(
          company.shareholders[startRow + ri].id,
          classes[startColumn + ci].id,
        ),
        current = integer(output[key]);
      output[key] =
        mode === 'negative'
          ? -value
          : mode === 'add'
            ? current + value
            : mode === 'subtract'
              ? current - value
              : value;
    }),
  );
  return output;
}
