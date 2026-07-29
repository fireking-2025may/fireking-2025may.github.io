import { History } from '../state/history.js';
import {
  TYPES,
  classesOf,
  createCompany,
  parseShares,
  formatShares,
  cellKey,
  valueAt,
  resultingValues,
  totals,
  findBlocking,
  syncSteps,
} from '../state/excel-model.js';

const escapeHtml = (value) =>
  String(value).replace(
    /[&<>"']/g,
    (character) =>
      ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[
        character
      ],
  );

const gridSelector = '[data-grid-row][data-grid-column]';

export function gridDestination(targets, current, key, shiftKey = false) {
  const row = Number(current.dataset.gridRow);
  const column = Number(current.dataset.gridColumn);
  const index = targets.indexOf(current);

  if (key === 'Tab') {
    return targets[
      (index + (shiftKey ? -1 : 1) + targets.length) % targets.length
    ];
  }

  const rowChange =
    key === 'ArrowUp' || (key === 'Enter' && shiftKey)
      ? -1
      : key === 'ArrowDown' || key === 'Enter'
        ? 1
        : 0;
  const columnChange = key === 'ArrowLeft' ? -1 : key === 'ArrowRight' ? 1 : 0;
  return (
    targets.find(
      (target) =>
        Number(target.dataset.gridRow) === row + rowChange &&
        Number(target.dataset.gridColumn) === column + columnChange,
    ) || current
  );
}

export class ExcelEditor {
  constructor({ root, toolbar, getDocument, updateDocument, report }) {
    this.root = root;
    this.toolbar = toolbar;
    this.getDocument = getDocument;
    this.updateDocument = updateDocument;
    this.report = report;
    this.history = new History(getDocument().excel);
    this.focus = {};
    this.bindToolbar();
  }
  get excel() {
    return this.getDocument().excel;
  }
  get company() {
    return this.excel.companies.find(
      (item) => item.id === this.excel.selectedCompanyId,
    );
  }
  replace(excel, record = true) {
    if (record) this.history.commit(excel);
    else this.history.replace(excel);
    this.updateDocument(excel);
    this.render();
  }
  mutate(change) {
    const excel = structuredClone(this.excel);
    change(excel);
    this.replace(excel);
  }
  synchronize() {
    const next = syncSteps(this.excel, this.getDocument().steps);
    if (JSON.stringify(next) !== JSON.stringify(this.excel)) this.replace(next);
    else this.render();
  }
  undo() {
    const next = syncSteps(this.history.undo(), this.getDocument().steps);
    this.history.replace(next);
    this.updateDocument(next);
    this.render();
  }
  redo() {
    const next = syncSteps(this.history.redo(), this.getDocument().steps);
    this.history.replace(next);
    this.updateDocument(next);
    this.render();
  }
  restore(excel) {
    this.history = new History(excel);
    this.render();
  }
  bindToolbar() {
    this.toolbar.addEventListener('click', (event) => {
      const action = event.target.dataset.excelAction;
      if (!action) return;
      this[action]?.();
    });
  }
  addCompany() {
    const name = prompt('Company name:', 'New company');
    if (!name?.trim()) return;
    this.mutate((excel) => {
      const company = createCompany(name.trim());
      excel.companies.push(company);
      excel.selectedCompanyId = company.id;
    });
  }
  deleteCompany() {
    const company = this.company;
    if (!company) return;
    const typed = prompt(
      `Type “${company.name}” to permanently delete this company. This can be undone.`,
    );
    if (typed !== company.name) {
      if (typed !== null)
        this.report('Company name did not match; nothing was deleted.');
      return;
    }
    this.mutate((excel) => {
      excel.companies = excel.companies.filter(
        (item) => item.id !== company.id,
      );
      excel.selectedCompanyId = excel.companies[0]?.id || null;
    });
  }
  addShareholder() {
    if (!this.company || this.company.shareholders.length >= 100)
      return this.report('The 100-shareholder limit has been reached.');
    this.mutate((excel) => {
      const company = excel.companies.find(
          (item) => item.id === excel.selectedCompanyId,
        ),
        at = this.focus.holderId
          ? company.shareholders.findIndex(
              (item) => item.id === this.focus.holderId,
            ) + 1
          : company.shareholders.length;
      const holder = {
        id: `holder-${crypto.randomUUID()}`,
        name: '',
        type: 'Individual',
      };
      company.shareholders.splice(at, 0, holder);
      this.focus = { holderId: holder.id };
    });
  }
  removeShareholder() {
    const company = this.company,
      holder = company?.shareholders.find(
        (item) => item.id === this.focus.holderId,
      );
    if (!holder) return this.report('Focus a shareholder row first.');
    const blocking = findBlocking(
      company,
      null,
      holder.id,
      this.getDocument().steps,
    );
    if (blocking) return this.blocked(blocking);
    this.mutate((excel) => {
      const current = excel.companies.find((item) => item.id === company.id);
      current.shareholders = current.shareholders.filter(
        (item) => item.id !== holder.id,
      );
    });
  }
  addGroup() {
    if (!this.company) return;
    if (classesOf(this.company).length >= 100)
      return this.report('The 100-share-class limit has been reached.');
    this.mutate((excel) => {
      const company = excel.companies.find(
        (item) => item.id === excel.selectedCompanyId,
      );
      company.groups.push({
        id: `group-${crypto.randomUUID()}`,
        name: 'New group',
        classes: [{ id: `class-${crypto.randomUUID()}`, label: '' }],
      });
    });
  }
  removeGroup() {
    const company = this.company,
      group = company?.groups.find((item) => item.id === this.focus.groupId);
    if (!group) return this.report('Focus a share-group header first.');
    const blocking = findBlocking(
      company,
      group.classes.map((item) => item.id),
      null,
      this.getDocument().steps,
    );
    if (blocking) return this.blocked(blocking);
    this.mutate(
      (excel) =>
        (excel.companies.find((item) => item.id === company.id).groups =
          company.groups.filter((item) => item.id !== group.id)),
    );
  }
  addClass() {
    if (!this.company) return;
    if (classesOf(this.company).length >= 100)
      return this.report('The 100-share-class limit has been reached.');
    this.mutate((excel) => {
      const company = excel.companies.find(
          (item) => item.id === excel.selectedCompanyId,
        ),
        group =
          company.groups.find((item) => item.id === this.focus.groupId) ||
          company.groups.at(-1);
      if (!group) return this.report('Add a share group first.');
      const at = this.focus.classId
        ? group.classes.findIndex((item) => item.id === this.focus.classId) + 1
        : group.classes.length;
      group.classes.splice(at, 0, {
        id: `class-${crypto.randomUUID()}`,
        label: '',
      });
    });
  }
  removeClass() {
    const company = this.company,
      shareClass = classesOf(company || { groups: [] }).find(
        (item) => item.id === this.focus.classId,
      );
    if (!shareClass) return this.report('Focus a share-class header first.');
    const blocking = findBlocking(
      company,
      [shareClass.id],
      null,
      this.getDocument().steps,
    );
    if (blocking) return this.blocked(blocking);
    this.mutate((excel) => {
      for (const group of excel.companies.find((item) => item.id === company.id)
        .groups)
        group.classes = group.classes.filter(
          (item) => item.id !== shareClass.id,
        );
    });
  }
  blocked(item) {
    this.report(
      `${this.company.name} — ${item.location}: ${item.holder.name}, ${item.shareClass.label || 'unnamed share class'} contains data and blocks deletion.`,
    );
  }
  table(title, values, editable, stepId = '') {
    const company = this.company;
    const classes = classesOf(company);
    const summary = totals(company, values);
    let groupHeaders = '';
    let classHeaders = '';

    for (const group of company.groups) {
      groupHeaders += `<th class="group-head" data-group-id="${escapeHtml(group.id)}" colspan="${Math.max(1, group.classes.length)}"><input data-group-name value="${escapeHtml(group.name)}" aria-label="Share group"></th>`;
      for (const item of group.classes) {
        const column =
          classes.findIndex((shareClass) => shareClass.id === item.id) + 1;
        classHeaders += `<th data-group-id="${escapeHtml(group.id)}" data-class-id="${escapeHtml(item.id)}"><input data-class-label data-grid-row="0" data-grid-column="${column}" value="${escapeHtml(item.label)}" aria-label="Share class"></th>`;
      }
    }

    const rows = company.shareholders
      .map((holder, row) => {
        const shareholder = `<th><input data-holder-name data-holder-id="${escapeHtml(holder.id)}" data-grid-row="${row + 1}" data-grid-column="0" value="${escapeHtml(holder.name)}" aria-label="Shareholder"></th>`;
        const type = `<td><select data-holder-type data-holder-id="${escapeHtml(holder.id)}" tabindex="-1">${TYPES.map((item) => `<option${item === holder.type ? ' selected' : ''}>${item}</option>`).join('')}</select></td>`;
        const rowValues = classes
          .map((item, column) => {
            const value = valueAt(values, holder.id, item.id);
            const negative = value < 0 && (editable === 'opening' || !editable);
            return `<td class="number ${negative ? 'negative' : ''}" tabindex="${editable ? '0' : '-1'}" ${editable ? 'contenteditable="true"' : ''} data-grid-row="${row + 1}" data-grid-column="${column + 1}" data-holder-id="${escapeHtml(holder.id)}" data-class-id="${escapeHtml(item.id)}" data-step-id="${escapeHtml(stepId)}" data-editable="${editable ? 'true' : 'false'}">${formatShares(value)}</td>`;
          })
          .join('');
        return `<tr>${shareholder}${type}${rowValues}<td class="number total">${formatShares(summary.rows[holder.id])}</td></tr>`;
      })
      .join('');

    const columnTotals = classes
      .map(
        (item) =>
          `<td class="number">${formatShares(summary.columns[item.id])}</td>`,
      )
      .join('');
    return `<section class="share-table"><h3>${escapeHtml(title)}</h3><div class="table-scroll"><table><thead><tr><th rowspan="2">Shareholder</th><th rowspan="2">Type</th>${groupHeaders}<th rowspan="2">Total</th></tr><tr>${classHeaders}</tr></thead><tbody>${rows}<tr class="total-row"><th colspan="2">Total</th>${columnTotals}<td class="number">${formatShares(summary.grand)}</td></tr></tbody></table></div></section>`;
  }
  render() {
    const excel = this.excel,
      company = this.company;
    this.root.innerHTML = `<div class="company-tabs" role="tablist">${excel.companies.map((item) => `<button role="tab" aria-selected="${item.id === excel.selectedCompanyId}" data-company-id="${escapeHtml(item.id)}">${escapeHtml(item.name)}</button>`).join('')}</div>${
      company
        ? this.table('Opening Position', company.opening, 'opening') +
          this.getDocument()
            .steps.map(
              (step, index) =>
                `<section class="step-share-section"><button class="collapse-step" data-collapse="${escapeHtml(step.id)}" aria-expanded="${!company.collapsed[step.id]}">${company.collapsed[step.id] ? '▸' : '▾'} Step ${index + 1}: ${escapeHtml(step.title)}</button><div ${company.collapsed[step.id] ? 'hidden' : ''}>${this.table(`Step ${index + 1} Movement`, company.movements[step.id], true, step.id)}${this.table(`Step ${index + 1} Resulting Position`, resultingValues(company, this.getDocument().steps, index), false, step.id)}</div></section>`,
            )
            .join('')
        : '<p>Add a company to begin.</p>'
    }`;
    this.bindGrid();
  }
  bindGrid() {
    this.root
      .querySelectorAll('[data-company-id]')
      .forEach(
        (button) =>
          (button.onclick = () =>
            this.mutate(
              (excel) => (excel.selectedCompanyId = button.dataset.companyId),
            )),
      );
    this.root.querySelectorAll('[data-collapse]').forEach(
      (button) =>
        (button.onclick = () =>
          this.mutate((excel) => {
            const company = excel.companies.find(
              (item) => item.id === excel.selectedCompanyId,
            );
            company.collapsed[button.dataset.collapse] =
              !company.collapsed[button.dataset.collapse];
          })),
    );
    this.root.querySelectorAll('[data-holder-id]').forEach(
      (element) =>
        (element.onfocus = () => {
          this.focus.holderId = element.dataset.holderId;
        }),
    );
    this.root.querySelectorAll('[data-group-id]').forEach(
      (element) =>
        (element.onfocus = () => {
          this.focus.groupId = element.dataset.groupId;
          this.focus.classId = element.dataset.classId;
        }),
    );
    this.root
      .querySelectorAll(gridSelector)
      .forEach(
        (target) => (target.onkeydown = (event) => this.keydown(event, target)),
      );
    this.root.querySelectorAll('td[data-editable=true]').forEach((cell) => {
      cell.onfocus = () => {
        const header = [
          ...cell.closest('table').querySelectorAll('[data-class-id]'),
        ].find((item) => item.dataset.classId === cell.dataset.classId);
        this.focus = {
          holderId: cell.dataset.holderId,
          classId: cell.dataset.classId,
          groupId: header?.dataset.groupId,
        };
        if (cell.textContent.trim() === '0') cell.textContent = '';
      };
      cell.onblur = () => {
        if (cell.dataset.skipBlur === 'true') return;
        this.commitCell(cell);
      };
    });
    const companyOf = (excel) =>
      excel.companies.find((item) => item.id === excel.selectedCompanyId);
    this.root.querySelectorAll('[data-holder-name]').forEach(
      (input) =>
        (input.onchange = () =>
          this.mutate((excel) => {
            const holder = companyOf(excel).shareholders.find(
              (item) => item.id === input.dataset.holderId,
            );
            if (holder) holder.name = input.value;
          })),
    );
    this.root.querySelectorAll('[data-holder-type]').forEach(
      (select) =>
        (select.onchange = () =>
          this.mutate((excel) => {
            const holder = companyOf(excel).shareholders.find(
              (item) => item.id === select.dataset.holderId,
            );
            if (holder) holder.type = select.value;
          })),
    );
    this.root.querySelectorAll('[data-group-name]').forEach(
      (input) =>
        (input.onchange = () =>
          this.mutate((excel) => {
            const group = companyOf(excel).groups.find(
              (item) =>
                item.id === input.closest('[data-group-id]').dataset.groupId,
            );
            if (group) group.name = input.value;
          })),
    );
    this.root.querySelectorAll('[data-class-label]').forEach(
      (input) =>
        (input.onchange = () =>
          this.mutate((excel) => {
            const header = input.closest('[data-class-id]');
            const shareClass = classesOf(companyOf(excel)).find(
              (item) => item.id === header.dataset.classId,
            );
            if (shareClass) shareClass.label = input.value;
          })),
    );
  }
  commitCell(cell) {
    if (cell.contentEditable !== 'true') return;
    const value = parseShares(cell.textContent);
    if (value === null) {
      this.report('Only whole integers are permitted.');
      this.render();
      return;
    }
    this.mutate((excel) => {
      const company = excel.companies.find(
          (item) => item.id === excel.selectedCompanyId,
        ),
        values = cell.dataset.stepId
          ? company.movements[cell.dataset.stepId]
          : company.opening;
      values[cellKey(cell.dataset.holderId, cell.dataset.classId)] = value;
    });
  }
  keydown(event, target) {
    if (event.key === 'Escape' && target.matches('td[data-editable=true]')) {
      event.preventDefault();
      this.render();
      return;
    }
    const navigationKeys = [
      'ArrowUp',
      'ArrowDown',
      'ArrowLeft',
      'ArrowRight',
      'Enter',
      'Tab',
    ];
    if (!navigationKeys.includes(event.key)) return;

    const table = target.closest('table');
    const targets = [...table.querySelectorAll(gridSelector)].filter(
      (item) => !item.matches('td[data-editable=false]'),
    );
    const destination = gridDestination(
      targets,
      target,
      event.key,
      event.shiftKey,
    );
    const destinationPosition = {
      row: destination.dataset.gridRow,
      column: destination.dataset.gridColumn,
    };
    const tables = [...this.root.querySelectorAll('.share-table table')];
    const tableIndex = tables.indexOf(table);
    event.preventDefault();

    if (target.matches('td[data-editable=true]')) {
      target.dataset.skipBlur = 'true';
      this.commitCell(target);
    } else if (target.matches('[data-holder-name], [data-class-label]')) {
      target.onchange();
    }
    const currentTable =
      this.root.querySelectorAll('.share-table table')[tableIndex] || table;
    currentTable
      .querySelector(
        `${gridSelector}[data-grid-row="${destinationPosition.row}"][data-grid-column="${destinationPosition.column}"]`,
      )
      ?.focus();
  }
}
