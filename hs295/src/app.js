"use strict";
const C = HS295Core;
let state = {
    company: null,
    people: [],
    transfers: [],
    step: 1,
    reviewed: false,
  },
  noticeTimer;
const $ = (s) => document.querySelector(s),
  openEditor = (content) => {
    const editor = $("#editor");
    editor.replaceChildren(content);
    if (!editor.open) editor.showModal();
  },
  closeEditor = () => {
    const editor = $("#editor");
    if (editor.open) editor.close();
    editor.replaceChildren();
  },
  el = (tag, txt, cls) => {
    const n = document.createElement(tag);
    if (txt != null) n.textContent = txt;
    if (cls) n.className = cls;
    return n;
  };
function dirty() {
  state.reviewed = false;
}
function clearNotice() {
  clearTimeout(noticeTimer);
  noticeTimer = undefined;
  const n = $("#notice");
  n.hidden = true;
  n.textContent = "";
  n.removeAttribute("tabindex");
}
function show(id) {
  clearNotice();
  document.querySelectorAll(".view").forEach((v) => (v.hidden = v.id !== id));
  scrollTo(0, 0);
}
function notice(msg, type = "info") {
  clearTimeout(noticeTimer);
  const n = $("#notice");
  n.textContent = msg;
  n.className = "notice " + type;
  n.hidden = false;
  if (type !== "error") noticeTimer = setTimeout(clearNotice, 6000);
  else {
    n.tabIndex = -1;
    n.focus();
  }
}
function field(form, name, label, type = "text", help = "") {
  const wrap = el("div", null, "field");
  const l = el("label", label);
  l.htmlFor = form.id + "-" + name;
  const i = el(type === "textarea" ? "textarea" : "input");
  i.id = l.htmlFor;
  i.name = name;
  if (type !== "textarea") i.type = type;
  wrap.append(l, i);
  if (help) wrap.append(el("small", help));
  return wrap;
}
function clearFieldErrors(form) {
  form.querySelectorAll(".field-error").forEach((x) => x.remove());
  form.querySelectorAll(".invalid-field").forEach((x) => {
    x.classList.remove("invalid-field");
    x.removeAttribute("aria-invalid");
    x.removeAttribute("aria-describedby");
  });
}
function fieldNameForError(message) {
  const rules = [
    ["Full name", "fullName"],
    ["fullName", "fullName"],
    ["Address line 1", "address1"],
    ["address1", "address1"],
    ["Address line 2", "address2"],
    ["address2", "address2"],
    ["country", "country"],
    ["Country", "country"],
    ["Postcode", "postcode"],
    ["postcode", "postcode"],
    ["UTR", "identifierValue"],
    ["National Insurance", "identifierValue"],
    ["identifier", "identifierValue"],
    ["transferor", "transferorPersonId"],
    ["transferee", "transfereePersonId"],
    ["Number of shares", "numberShares"],
    ["Nominal value", "nominalValue"],
    ["Share class", "shareClass"],
    ["disposal date", "disposalDate"],
    ["acquisition date", "acquisitionDate"],
    ["Whole acquisition cost", "wholeAcquisitionCost"],
    ["Transfer value", "transferValue"],
    ["Reorganisation", "reorganisationDetails"],
  ];
  return rules.find(([text]) => message.includes(text))?.[1];
}
function showNamedFieldErrors(form, errors) {
  clearFieldErrors(form);
  const grouped = new Map();
  for (const { name, message } of errors) {
    const input = name && form.elements[name];
    if (!input) continue;
    input.classList.add("invalid-field");
    input.setAttribute("aria-invalid", "true");
    if (!grouped.has(input)) grouped.set(input, []);
    grouped.get(input).push(message);
  }
  for (const [input, messages] of grouped) {
    const error = el("small", [...new Set(messages)].join(" "), "field-error");
    error.id = `${input.id}-error`;
    error.setAttribute("role", "alert");
    input.setAttribute("aria-describedby", error.id);
    input.closest(".field")?.append(error);
  }
  form.querySelector(".invalid-field")?.focus();
}
function showFieldErrors(form, errors) {
  clearFieldErrors(form);
  const messages = new Map();
  for (const message of errors) {
    const name = fieldNameForError(message),
      input = name && form.elements[name];
    if (!input) continue;
    input.classList.add("invalid-field");
    input.setAttribute("aria-invalid", "true");
    if (!messages.has(name)) messages.set(name, []);
    messages.get(name).push(message);
  }
  for (const [name, items] of messages) {
    const input = form.elements[name],
      message = el("small", items.join(" "), "field-error");
    message.id = input.id + "-error";
    input.setAttribute("aria-describedby", message.id);
    input.closest(".field")?.append(message);
  }
  form.querySelector(".invalid-field")?.focus();
}
function createBatch(e) {
  e.preventDefault();
  const legalName = $("#company-name").value.trim();
  if (!legalName || legalName.length > 70) {
    showNamedFieldErrors(e.currentTarget, [
      {
        name: "legalName",
        message:
          "Enter the company legal name shown on the Companies House register.",
      },
    ]);
    return;
  }
  clearFieldErrors(e.currentTarget);
  state.company = { legalName, companyNumber: "" };
  state.step = 2;
  dirty();
  renderDashboard();
  show("dashboard");
}
function personForm(existing) {
  const f = el("form");
  f.id = "person-form";
  f.noValidate = true;
  f.append(el("h3", existing ? "Edit person" : "Add person"));
  [
    ["fullName", "Full name"],
    ["address1", "Address line 1"],
    ["address2", "Address line 2"],
    ["country", "Country"],
    ["postcode", "Postcode (include the space)"],
  ].forEach((x) => f.append(field(f, x[0], x[1])));
  f.elements.country.value = existing?.country ?? "United Kingdom";
  const idField = field(
    f,
    "identifierValue",
    "Unique Taxpayer Reference (UTR)",
  );
  idField.querySelector("input").autocomplete = "off";
  const toggle = el("button", "I don't have a UTR", "link-button");
  toggle.type = "button";
  idField.append(
    toggle,
    el(
      "small",
      "Optional. You may enter any text supplied as the UTR reference.",
    ),
  );
  const type = el("input");
  type.type = "hidden";
  type.name = "identifierType";
  type.value = existing?.identifierType || "UTR";
  idField.append(type);
  f.append(idField);
  function setIdentifier(kind) {
    type.value = kind;
    const input = idField.querySelector("input:not([type=hidden])"),
      label = idField.querySelector("label");
    if (kind === "NINO") {
      label.textContent = "National Insurance number";
      input.inputMode = "text";
      input.pattern = "[A-Za-z]{2}[0-9 ]{6,11}[A-Da-d]";
      input.maxLength = 13;
      toggle.textContent = "Use a UTR instead";
    } else {
      label.textContent = "Unique Taxpayer Reference (UTR)";
      input.inputMode = "text";
      input.removeAttribute("pattern");
      input.removeAttribute("maxlength");
      toggle.textContent = "I don't have a UTR";
    }
  }
  toggle.onclick = () => setIdentifier(type.value === "UTR" ? "NINO" : "UTR");
  setIdentifier(type.value);
  if (existing)
    for (const [k, v] of Object.entries(existing))
      if (f.elements[k]) f.elements[k].value = v;
  if (existing) setIdentifier(existing.identifierType);
  const actions = el("div", null, "actions");
  const save = el("button", "Save person", "primary");
  save.type = "submit";
  const cancel = el("button", "Cancel");
  cancel.type = "button";
  cancel.onclick = closeEditor;
  actions.append(save, cancel);
  f.append(actions);
  f.onsubmit = (e) => {
    e.preventDefault();
    const d = Object.fromEntries(new FormData(f));
    d.id = existing?.id || C.uuid();
    d.fullName = d.fullName.trim();
    d.address1 = d.address1.trim();
    d.address2 = d.address2.trim();
    d.country = d.country.trim();
    d.postcode = C.normalizePostcode(d.postcode) || d.postcode.trim();
    d.identifierValue =
      (d.identifierType === "UTR"
        ? C.normalizeUTR(d.identifierValue)
        : C.normalizeNINO(d.identifierValue)) ||
      d.identifierValue.trim().toUpperCase();
    const errors = C.personErrors(d);
    if (errors.length) {
      showFieldErrors(f, errors);
      return;
    }
    clearFieldErrors(f);
    const i = state.people.findIndex((x) => x.id === d.id);
    if (i < 0) state.people.push(d);
    else state.people[i] = d;
    dirty();
    clearNotice();
    closeEditor();
    renderDashboard();
  };
  openEditor(f);
  f.querySelector("input").focus();
}

function coupleForm(existingCouple = null) {
  const [existingHusband, existingWife] = existingCouple || [];
  const f = el("form");
  f.id = "couple-form";
  f.noValidate = true;
  f.append(
    el("h3", existingCouple ? "Edit husband and wife" : "Add husband and wife"),
    el(
      "p",
      "Enter their shared surname and home address once. Both person records stay linked.",
      "help",
    ),
  );
  const namesRow = el("div", null, "paired-fields");
  for (const [name, label] of [
    ["husbandGivenNames", "Husband's first and middle names"],
    ["wifeGivenNames", "Wife's first and middle names"],
  ])
    namesRow.append(field(f, name, label));
  f.append(namesRow);
  for (const [name, label] of [
    ["surname", "Shared surname"],
    ["address1", "Home address line 1"],
    ["address2", "Home address line 2"],
    ["country", "Country"],
    ["postcode", "Postcode (include the space)"],
  ])
    f.append(field(f, name, label));

  function addIdentifier(prefix, label, person) {
    const name = `${prefix}IdentifierValue`;
    const wrapper = field(f, name, `${label} Unique Taxpayer Reference (UTR)`);
    const input = wrapper.querySelector("input");
    const type = el("input");
    const toggle = el("button", `${label} does not have a UTR`, "link-button");
    type.type = "hidden";
    type.name = `${prefix}IdentifierType`;
    toggle.type = "button";
    wrapper.append(
      type,
      toggle,
      el(
        "small",
        "UTR is optional. Alternatively enter a syntactically valid National Insurance number.",
      ),
    );
    function setIdentifier(kind) {
      type.value = kind;
      wrapper.querySelector("label").textContent =
        kind === "NINO"
          ? `${label} National Insurance number`
          : `${label} Unique Taxpayer Reference (UTR)`;
      input.toggleAttribute("pattern", kind === "NINO");
      if (kind === "NINO") input.pattern = "[A-Za-z]{2}[0-9 ]{6,11}[A-Da-d]";
      input.maxLength = kind === "NINO" ? 13 : 524288;
      toggle.textContent =
        kind === "NINO"
          ? `Use a UTR for ${label.toLowerCase()}`
          : `${label} does not have a UTR`;
    }
    toggle.onclick = () => setIdentifier(type.value === "UTR" ? "NINO" : "UTR");
    setIdentifier(person?.identifierType || "UTR");
    input.value = person?.identifierValue || "";
    return wrapper;
  }
  const identifiersRow = el("div", null, "paired-fields");
  identifiersRow.append(
    addIdentifier("husband", "Husband", existingHusband),
    addIdentifier("wife", "Wife", existingWife),
  );
  f.append(identifiersRow);

  const splitName = (person) => {
    const parts = person.fullName.split(" ");
    return { givenNames: parts.slice(0, -1).join(" "), surname: parts.at(-1) };
  };
  if (existingCouple) {
    const husbandName = splitName(existingHusband);
    const wifeName = splitName(existingWife);
    f.elements.husbandGivenNames.value = husbandName.givenNames;
    f.elements.wifeGivenNames.value = wifeName.givenNames;
    f.elements.surname.value = husbandName.surname;
    for (const name of ["address1", "address2", "country", "postcode"])
      f.elements[name].value = existingHusband[name];
  } else f.elements.country.value = "United Kingdom";

  const actions = el("div", null, "actions");
  const save = el("button", "Save husband and wife", "primary");
  const cancel = el("button", "Cancel");
  save.type = "submit";
  cancel.type = "button";
  cancel.onclick = closeEditor;
  actions.append(save, cancel);
  f.append(actions);
  f.onsubmit = (event) => {
    event.preventDefault();
    const d = Object.fromEntries(new FormData(f));
    for (const name of [
      "husbandGivenNames",
      "wifeGivenNames",
      "surname",
      "address1",
      "address2",
      "country",
      "postcode",
    ])
      d[name] = d[name].trim();
    const coupleId = existingHusband?.coupleId || C.uuid();
    const makePerson = (prefix, existing) => {
      const kind = d[`${prefix}IdentifierType`];
      const raw = d[`${prefix}IdentifierValue`];
      return {
        id: existing?.id || C.uuid(),
        coupleId,
        fullName: `${d[`${prefix}GivenNames`]} ${d.surname}`,
        address1: d.address1,
        address2: d.address2,
        country: d.country,
        postcode: C.normalizePostcode(d.postcode) || d.postcode,
        identifierType: kind,
        identifierValue:
          (kind === "UTR" ? C.normalizeUTR(raw) : C.normalizeNINO(raw)) ||
          raw.trim().toUpperCase(),
      };
    };
    const husband = makePerson("husband", existingHusband);
    const wife = makePerson("wife", existingWife);
    const errors = [];
    for (const [name, label] of [
      ["husbandGivenNames", "Husband's names"],
      ["wifeGivenNames", "Wife's names"],
      ["surname", "Shared surname"],
    ]) {
      if (!C.validName(d[name]))
        errors.push({
          name,
          message: `${label} must contain letters, single spaces and correctly placed hyphens only.`,
        });
    }
    const mapError = (message, prefix) => {
      if (/Full name|fullName/.test(message)) return `${prefix}GivenNames`;
      if (/UTR|National Insurance|identifier/.test(message))
        return `${prefix}IdentifierValue`;
      return [
        ["Address line 1", "address1"],
        ["Address line 2", "address2"],
        ["Country", "country"],
        ["Postcode", "postcode"],
      ].find(([text]) => message.includes(text))?.[1];
    };
    for (const [person, prefix, label] of [
      [husband, "husband", "Husband"],
      [wife, "wife", "Wife"],
    ]) {
      for (const message of C.personErrors(person)) {
        const name = mapError(message, prefix);
        if (
          prefix === "wife" &&
          ["address1", "address2", "country", "postcode"].includes(name)
        )
          continue;
        errors.push({ name, message: `${label}: ${message}` });
      }
    }
    if (errors.length) {
      showNamedFieldErrors(f, errors);
      return;
    }
    for (const person of [husband, wife]) {
      const index = state.people.findIndex(
        (candidate) => candidate.id === person.id,
      );
      if (index < 0) state.people.push(person);
      else state.people[index] = person;
    }
    dirty();
    clearNotice();
    closeEditor();
    renderDashboard();
  };
  openEditor(f);
  f.elements.husbandGivenNames.focus();
}

const transferDefaults = () => C.transferDefaults();
const TRANSFER_COPY_FIELDS = [
  ["transferorPersonId", "Transferor"],
  ["transfereePersonId", "Transferee"],
  ["numberShares", "Number of shares"],
  ["nominalValue", "Nominal value"],
  ["shareClass", "Share class"],
  ["disposalDate", "Disposal date"],
  ["acquisitionDate", "Acquisition date"],
  ["wholeAcquisitionCostEntry", "Acquisition cost"],
  ["transferValueEntry", "Transfer value"],
];
function transferFields(t, index = 0, batch = false) {
  const box = el(
    batch ? "fieldset" : "div",
    null,
    batch ? "transfer-column" : null,
  );
  box.dataset.column = String(index);
  if (batch) box.append(el("legend", `Transfer ${index + 1}`));
  function add(name, label, type = "text", help = "") {
    let wrap;
    if (name.endsWith("Mode")) {
      wrap = el("div", null, "field");
      const l = el("label", label),
        input = el("select");
      input.name = name;
      wrap.append(l, input);
    } else if (name.endsWith("PersonId")) {
      wrap = el("div", null, "field");
      const l = el("label", label),
        input = el("select");
      input.name = name;
      input.append(new Option("Select…", ""));
      state.people.forEach((p) => input.append(new Option(p.fullName, p.id)));
      wrap.append(l, input);
    } else wrap = field({ id: `transfer-${index}` }, name, label, type, help);
    const input = wrap.querySelector("input,select,textarea");
    input.id = `transfer-${index}-${name}`;
    wrap.querySelector("label").htmlFor = input.id;
    input.value = t[name] ?? "";
    box.append(wrap);
    if (batch && TRANSFER_COPY_FIELDS.some(([key]) => key === name)) {
      const copy = el(
        "button",
        `Copy ${label.toLowerCase()} to all`,
        "copy-all",
      );
      copy.type = "button";
      copy.dataset.copy = name;
      wrap.append(copy);
    }
    return input;
  }
  add("transferorPersonId", "Transferor");
  add("transfereePersonId", "Transferee");
  add("numberShares", "Number of shares", "number");
  add("nominalValue", "Nominal value per share (£)");
  add("shareClass", "Share class");
  add("disposalDate", "Date of disposal", "date");
  add("acquisitionDate", "Date of acquisition", "date");
  for (const [key, label] of [
    ["wholeAcquisitionCost", "Acquisition cost"],
    ["transferValue", "Estimated transfer value"],
  ]) {
    const pair = el("div", null, "money-pair"),
      total = add(key, label + " — total (£)", "text"),
      perShare = add(key + "PerShare", label + " — per share (£)", "text");
    pair.append(total.closest(".field"), perShare.closest(".field"));
    box.append(pair);
    if ((t[key + "Mode"] || "total") === "perShare") {
      perShare.value = t[key + "Entry"] || "";
      total.value = "";
    } else total.value = t[key + "Entry"] || t[key] || "";
    const clearOther = (e) => {
      if (!e.target.value) return;
      const other = e.target === total ? perShare : total;
      other.value = "";
    };
    total.addEventListener("input", clearOther);
    perShare.addEventListener("input", clearOther);
  }
  const reorg = add(
    "reorganisationDetails",
    "Reorganisation or bonus-issue details (optional)",
    "textarea",
    "Leave blank if there have been none.",
  );
  reorg.maxLength = C.LIMITS.reorganisation;
  const calc = el(
    "output",
    "Enter complete values to see the held-over gain.",
    "calculation",
  );
  calc.setAttribute("aria-live", "polite");
  box.append(calc);
  return box;
}
function readTransfer(box, id) {
  const raw = { id };
  for (const input of box.querySelectorAll("[name]"))
    raw[input.name] = input.type === "checkbox" ? input.checked : input.value;
  for (const key of ["wholeAcquisitionCost", "transferValue"]) {
    const perShare = raw[key + "PerShare"];
    raw[key + "Mode"] = perShare ? "perShare" : "total";
    raw[key + "Entry"] = perShare || raw[key] || "";
    delete raw[key + "PerShare"];
  }
  return C.normalizeTransfer(raw);
}
function updateTransferCalculation(box, id) {
  const d = readTransfer(box, id),
    g = C.gain(d),
    out = box.querySelector(".calculation");
  out.textContent =
    g === null
      ? "Enter complete values to see the held-over gain."
      : `Whole acquisition cost £${Number(d.wholeAcquisitionCost).toLocaleString("en-GB")}; transfer value £${Number(d.transferValue).toLocaleString("en-GB")}; held-over gain £${g.toLocaleString("en-GB")}.`;
}
function transferForm(existing) {
  if (state.people.length < 2) {
    notice("Add at least two people first.", "error");
    return;
  }
  const t = C.normalizeTransfer(
      structuredClone(existing || transferDefaults()),
    ),
    f = el("form");
  f.id = "transfer-form";
  f.noValidate = true;
  f.append(el("h3", existing ? "Edit transfer" : "Add transfer"));
  const box = transferFields(t);
  f.append(...box.childNodes);
  f.oninput = (e) => {
    if (e.target.classList.contains("invalid-field")) {
      e.target.classList.remove("invalid-field");
      e.target.removeAttribute("aria-invalid");
      document.getElementById(e.target.id + "-error")?.remove();
    }
    updateTransferCalculation(f, t.id);
  };
  const a = el("div", null, "actions"),
    save = el("button", "Save transfer", "primary"),
    cancel = el("button", "Cancel");
  save.type = "submit";
  cancel.type = "button";
  cancel.onclick = closeEditor;
  a.append(save, cancel);
  f.append(a);
  f.onsubmit = (e) => {
    e.preventDefault();
    const d = readTransfer(f, t.id),
      errs = C.transferErrors(d, state.people, state.company);
    if (errs.length) {
      showFieldErrors(f, errs);
      return;
    }
    clearFieldErrors(f);
    const i = state.transfers.findIndex((x) => x.id === d.id);
    if (i < 0) state.transfers.push(d);
    else state.transfers[i] = d;
    dirty();
    clearNotice();
    closeEditor();
    renderDashboard();
  };
  openEditor(f);
  updateTransferCalculation(f, t.id);
  f.querySelector("select").focus();
}
function batchTransferForm() {
  if (state.people.length < 2) {
    notice("Add at least two people first.", "error");
    return;
  }
  const pending = [transferDefaults()],
    f = el("form");
  f.id = "batch-transfer-form";
  f.noValidate = true;
  f.append(
    el("h3", "Add batch of transfers"),
    el(
      "p",
      "Complete each transfer column. Nothing is added until every column is valid and you select Save all transfers.",
      "help",
    ),
  );
  const region = el("div", null, "batch-transfer-region");
  region.tabIndex = 0;
  region.setAttribute(
    "aria-label",
    "Pending transfer columns; scroll horizontally to review every transfer",
  );
  const addActions = el("div", null, "actions"),
    actions = el("div", null, "actions"),
    add = el("button", "Add transfer column"),
    save = el("button", "Save all transfers", "primary"),
    cancel = el("button", "Cancel");
  for (const b of [add, cancel]) b.type = "button";
  save.type = "submit";
  addActions.append(add);
  f.append(addActions, region);
  actions.append(save, cancel);
  f.append(actions);
  function draw(focus) {
    region.replaceChildren();
    pending.forEach((t, i) => {
      const column = transferFields(t, i, true),
        remove = el("button", "Remove transfer column", "danger");
      remove.type = "button";
      remove.dataset.remove = String(i);
      remove.disabled = pending.length === 1;
      column.append(remove);
      region.append(column);
    });
    if (focus) region.lastElementChild?.querySelector("select")?.focus();
    pending.forEach((t, i) =>
      updateTransferCalculation(region.children[i], t.id),
    );
  }
  function snapshot() {
    pending.splice(
      0,
      pending.length,
      ...[...region.children].map((box, i) => readTransfer(box, pending[i].id)),
    );
  }
  add.onclick = () => {
    snapshot();
    pending.push(transferDefaults());
    draw(true);
  };
  cancel.onclick = closeEditor;
  region.onclick = (e) => {
    const button = e.target.closest("button");
    if (!button) return;
    if (button.dataset.remove != null) {
      snapshot();
      pending.splice(Number(button.dataset.remove), 1);
      draw();
      return;
    }
    if (button.dataset.copy) {
      snapshot();
      const source = readTransfer(
          button.closest(".transfer-column"),
          pending[Number(button.closest(".transfer-column").dataset.column)].id,
        ),
        key = button.dataset.copy,
        value = source[key];
      const differing = pending.some(
        (t, i) =>
          i !== Number(button.closest(".transfer-column").dataset.column) &&
          String(t[key] ?? "") &&
          t[key] !== value,
      );
      if (
        differing &&
        !confirm(
          `Overwrite existing ${button.textContent.replace(/^Copy | to all$/g, "")} values in other columns?`,
        )
      )
        return;
      for (const t of pending) t[key] = value;
      if (key.endsWith("Entry")) {
        const base = key.replace(/Entry$/, "");
        for (const t of pending) t[base + "Mode"] = source[base + "Mode"];
      }
      draw();
    }
  };
  region.oninput = (e) => {
    const box = e.target.closest(".transfer-column");
    if (box)
      updateTransferCalculation(box, pending[Number(box.dataset.column)].id);
  };
  f.onsubmit = (e) => {
    e.preventDefault();
    snapshot();
    const allErrors = pending.map((t) =>
        C.transferErrors(t, state.people, state.company),
      ),
      errors = allErrors.flat();
    if (errors.length) {
      allErrors.forEach((items, i) =>
        showFieldErrors(region.children[i], items),
      );
      region.querySelector(".invalid-field")?.focus();
      return;
    }
    state.transfers.push(...pending.map((t) => structuredClone(t)));
    dirty();
    clearNotice();
    closeEditor();
    renderDashboard();
  };
  openEditor(f);
  draw();
  region.querySelector("select")?.focus();
}
function renderDashboard() {
  $("#company-summary").textContent = state.company.legalName;
  const pl = $("#people-list");
  pl.replaceChildren();
  state.people.forEach((p) => {
    const row = el("li");
    row.append(el("span", `${p.fullName} — ${p.identifierType}`));
    const e = el("button", "Edit");
    e.onclick = () => {
      const couple =
        p.coupleId &&
        state.people.filter((person) => person.coupleId === p.coupleId);
      if (couple?.length === 2) coupleForm(couple);
      else personForm(p);
    };
    const d = el("button", "Delete");
    d.onclick = () => {
      if (
        state.transfers.some(
          (t) => t.transferorPersonId === p.id || t.transfereePersonId === p.id,
        )
      ) {
        notice(
          "This person is referenced by a transfer and cannot be deleted.",
          "error",
        );
        return;
      }
      if (confirm("Delete this person?")) {
        state.people = state.people.filter((x) => x.id !== p.id);
        dirty();
        renderDashboard();
      }
    };
    row.append(e, d);
    pl.append(row);
  });
  const tl = $("#transfer-list");
  tl.replaceChildren();
  state.transfers.forEach((t, i) => {
    const a = state.people.find((p) => p.id === t.transferorPersonId)?.fullName,
      b = state.people.find((p) => p.id === t.transfereePersonId)?.fullName,
      g = C.gain(t),
      errs = C.transferErrors(t, state.people, state.company),
      row = el("article", null, "card");
    row.append(
      el("h3", `${a} to ${b}`),
      el(
        "p",
        `${Number(t.numberShares).toLocaleString("en-GB")} ${t.shareClass} shares · ${t.disposalDate} · value £${Number(t.transferValue).toLocaleString("en-GB")} · gain £${g?.toLocaleString("en-GB")}`,
      ),
      el(
        "p",
        errs.length ? "✕ Needs attention" : "✓ Valid",
        errs.length ? "bad" : "good",
      ),
    );
    const ac = el("div", null, "actions");
    [
      ["Edit", () => transferForm(t)],
      [
        "Duplicate",
        () => {
          const n = structuredClone(t);
          n.id = C.uuid();
          state.transfers.push(n);
          dirty();
          renderDashboard();
        },
      ],
      [
        "Delete",
        () => {
          if (confirm("Delete this transfer?")) {
            state.transfers.splice(i, 1);
            dirty();
            renderDashboard();
          }
        },
      ],
    ].forEach(([x, fn]) => {
      const b = el("button", x);
      b.onclick = fn;
      ac.append(b);
    });
    row.append(ac);
    tl.append(row);
  });
}
function shareFlowReview() {
  const section = el("section", null, "share-flow-review"),
    heading = el("h3", "Share totals by person"),
    table = el("table"),
    caption = el(
      "caption",
      "Shares transferred and received across this batch",
    ),
    head = el("thead"),
    headRow = el("tr"),
    body = el("tbody");
  heading.id = "share-flow-heading";
  section.setAttribute("aria-labelledby", heading.id);
  for (const label of ["Person", "Shares transferred", "Shares received"]) {
    const cell = el("th", label);
    cell.scope = "col";
    headRow.append(cell);
  }
  head.append(headRow);
  for (const total of C.shareFlowTotals(state.people, state.transfers)) {
    const row = el("tr"),
      name = el(
        "th",
        state.people.find((person) => person.id === total.personId).fullName,
      );
    row.dataset.personId = total.personId;
    name.scope = "row";
    row.append(
      name,
      el("td", BigInt(total.transferred).toLocaleString("en-GB")),
      el("td", BigInt(total.received).toLocaleString("en-GB")),
    );
    body.append(row);
  }
  table.append(caption, head, body);
  section.append(heading, table);
  return section;
}
function review() {
  if (!state.transfers.length) {
    notice("Add at least one transfer.", "error");
    return;
  }
  const errors = state.transfers.flatMap((t, i) =>
    C.transferErrors(t, state.people, state.company).map(
      (e) => `Transfer ${i + 1}: ${e}`,
    ),
  );
  if (errors.length) {
    notice(errors.join(" "), "error");
    return;
  }
  const names = C.pdfFilenames(state.transfers, state.people),
    r = $("#review-list");
  r.replaceChildren(shareFlowReview());
  state.transfers.forEach((t, i) => {
    const a = state.people.find((p) => p.id === t.transferorPersonId),
      b = state.people.find((p) => p.id === t.transfereePersonId),
      cost = t.wholeAcquisitionCost,
      card = el("article", null, "card");
    card.append(
      el("h3", names[i]),
      el("p", `${a.fullName} → ${b.fullName}`),
      el("p", C.assetDescription(t, state.company)),
      el(
        "p",
        `Full disposal; whole acquisition cost £${Number(cost).toLocaleString("en-GB")}. Transfer value £${Number(t.transferValue).toLocaleString("en-GB")}. Held-over gain £${C.gain(t).toLocaleString("en-GB")}.`,
      ),
    );
    r.append(card);
  });
  $("#confirm-review").checked = false;
  show("review");
}
function fit(page, font, text, box, label) {
  let size = 9;
  while (size > 7 && font.widthOfTextAtSize(text, size) > box.w) size -= 0.25;
  if (font.widthOfTextAtSize(text, size) > box.w)
    throw Error(label + " does not fit at 7 points.");
  page.drawText(text, { x: box.x, y: box.y, size, font });
}
function postcodeChars(page, font, text, spec) {
  const [outward, inward] = text.split(" ");
  [...outward].forEach((c, i) =>
    page.drawText(c, { x: spec.xs[i] + 3, y: spec.y, size: 9, font }),
  );
  [...inward].forEach((c, i) =>
    page.drawText(c, { x: spec.xs[4 + i] + 3, y: spec.y, size: 9, font }),
  );
}
function chars(page, font, text, spec) {
  [...text.replace(/\s/g, "")].forEach((c, i) =>
    page.drawText(c, { x: spec.xs[i] + 3, y: spec.y, size: 9, font }),
  );
}
function digits(page, font, value, spec) {
  const s = String(Number(value)),
    start = spec.xs.length - s.length;
  if (start < 0) throw Error("Numeric value does not fit.");
  [...s].forEach((c, i) =>
    page.drawText(c, { x: spec.xs[start + i] + 3, y: spec.y, size: 9, font }),
  );
}
function dateChars(page, font, date, spec) {
  chars(page, font, date.split("-").reverse().join(""), spec);
}
function wrap(page, font, text, box, maxLines, label) {
  const words = text.split(/\s+/),
    lines = [];
  let line = "";
  for (const w of words) {
    const n = line ? line + " " + w : w;
    if (font.widthOfTextAtSize(n, 8) <= box.w) line = n;
    else {
      lines.push(line);
      line = w;
    }
  }
  if (line) lines.push(line);
  if (lines.length > maxLines) throw Error(label + " is too long.");
  lines.forEach((x, i) =>
    page.drawText(x, {
      x: box.x,
      y: box.ys?.[i] ?? box.y - i * 11,
      size: 8,
      font,
    }),
  );
}
async function makePdf(t) {
  const bytes = Uint8Array.from(atob(TEMPLATE_PDF_BASE64), (c) =>
      c.charCodeAt(0),
    ),
    doc = await PDFLib.PDFDocument.load(bytes, { ignoreEncryption: true });
  doc.catalog.delete(PDFLib.PDFName.of("AcroForm"));
  doc.catalog.delete(PDFLib.PDFName.of("Names"));
  const pages = doc.getPages();
  pages.forEach((p) => p.node.delete(PDFLib.PDFName.of("Annots")));
  if (pages.length !== 2) throw Error("Template must have exactly two pages.");
  const font = await doc.embedFont(PDFLib.StandardFonts.Helvetica),
    p1 = pages[0],
    p2 = pages[1],
    a = state.people.find((p) => p.id === t.transferorPersonId),
    b = state.people.find((p) => p.id === t.transfereePersonId),
    m = PDF_COORDS;
  for (const [p, c] of [
    [a, m.page1.transferor],
    [b, m.page1.transferee],
  ]) {
    fit(p1, font, p.fullName, c.name, "Name");
    [p.address1, p.address2, p.country].forEach((x, i) =>
      fit(p1, font, x, c.address[i], "Address"),
    );
    postcodeChars(p1, font, p.postcode, c.postcode);
    fit(p1, font, p.identifierValue, c.identifier, "Identifier");
  }
  p1.drawText("X", { ...m.page1.unlistedX, size: 10, font });
  const desc = C.assetDescription(t, state.company);
  wrap(p1, font, desc, m.page1.asset, 2, "Asset description");
  dateChars(p1, font, t.disposalDate, m.page1.disposal);
  digits(p1, font, C.gain(t), m.page1.gain);
  p1.drawText("X", { ...m.page1.defermentX, size: 10, font });
  dateChars(p2, font, t.acquisitionDate, m.page2.acquisitionDate);
  digits(p2, font, t.wholeAcquisitionCost, m.page2.acquisitionCost);
  digits(p2, font, t.transferValue, m.page2.transferValue);
  p2.drawText("E", {
    x: m.page2.transferValue.code.x,
    y: m.page2.transferValue.code.y,
    size: 9,
    font,
  });
  if ((t.reorganisationDetails || "").trim())
    wrap(
      p2,
      font,
      t.reorganisationDetails,
      m.page2.reorganisation,
      7,
      "Reorganisation details",
    );
  const out = await doc.save({
    useObjectStreams: false,
    addDefaultPage: false,
  });
  const check = await PDFLib.PDFDocument.load(out);
  if (check.getPageCount() !== 2 || check.getForm().getFields().length)
    throw Error("Generated PDF structural check failed.");
  return out;
}
function download(blob, name) {
  const u = URL.createObjectURL(blob),
    a = el("a");
  a.href = u;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(u), 1000);
}
async function exportsView() {
  if (!$("#confirm-review").checked) {
    notice("Select the eligibility and SP8/92 confirmation.", "error");
    return;
  }
  state.reviewed = true;
  const names = C.pdfFilenames(state.transfers, state.people),
    list = $("#export-list");
  list.replaceChildren();
  state.transfers.forEach((t, i) => {
    const row = el("article", null, "card"),
      b = el("button", "Download PDF", "primary"),
      status = el("span", "Ready");
    b.onclick = async () => {
      try {
        status.textContent = "Generating…";
        download(
          new Blob([await makePdf(t)], { type: "application/pdf" }),
          names[i],
        );
        status.textContent = "Downloaded";
      } catch (e) {
        status.textContent = "Error: " + e.message;
      }
    };
    row.append(
      el("strong", names[i]),
      el("span", ` Held-over gain £${C.gain(t).toLocaleString("en-GB")} `),
      b,
      status,
    );
    list.append(row);
  });
  show("exports");
}
async function zipAll() {
  try {
    const z = new JSZip(),
      names = C.pdfFilenames(state.transfers, state.people);
    for (let i = 0; i < state.transfers.length; i++)
      z.file(names[i], await makePdf(state.transfers[i]));
    const ref = C.taxpayerReference(state.transfers, state.people),
      name =
        C.sanitizeFilename(
          `${state.company.legalName} - ${ref} - Gift Holdover Relief Forms`,
        ) + ".zip";
    download(await z.generateAsync({ type: "blob" }), name);
    notice("ZIP downloaded.", "success");
  } catch (e) {
    notice(e.message, "error");
  }
}
function exportSession() {
  if (
    !confirm(
      "This unencrypted session file contains personal and tax-reference information. Store and transmit it securely.",
    )
  )
    return;
  download(
    new Blob([JSON.stringify(C.exportSession(state), null, 2)], {
      type: "application/json",
    }),
    C.sanitizeFilename(state.company.legalName + " - HS295 2026 session") +
      ".json",
  );
}
function importFile(file) {
  if (
    !file ||
    (!file.name.toLowerCase().endsWith(".json") &&
      file.type !== "application/json")
  ) {
    notice("Select a local JSON file.", "error");
    return;
  }
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const x = C.importSessionText(reader.result);
      if (
        confirm(
          `Import ${x.company.legalName}, ${x.people.length} people and ${x.transfers.length} transfers? This replaces the in-memory batch.`,
        )
      ) {
        state = { ...x, step: 2, reviewed: false };
        renderDashboard();
        show("dashboard");
      }
    } catch (e) {
      notice(e.message, "error");
    }
  };
  reader.readAsText(file);
}
$("#batch-form").onsubmit = createBatch;
$("#add-person").onclick = () => personForm();
$("#add-couple").onclick = () => coupleForm();
$("#add-transfer").onclick = () => transferForm();
$("#add-transfer-batch").onclick = batchTransferForm;
$("#review-button").onclick = review;
$("#to-exports").onclick = exportsView;
$("#zip-all").onclick = zipAll;
document
  .querySelectorAll(".export-session")
  .forEach((b) => (b.onclick = exportSession));
document.querySelectorAll(".back-edit").forEach(
  (b) =>
    (b.onclick = () => {
      dirty();
      renderDashboard();
      show("dashboard");
    }),
);
$("#new-batch").onclick = () => {
  if (
    confirm(
      "Start a new batch? The current in-memory batch will be lost unless you first downloaded a session export.",
    )
  ) {
    state = {
      company: null,
      people: [],
      transfers: [],
      step: 1,
      reviewed: false,
    };
    $("#batch-form").reset();
    show("setup");
  }
};
$("#import-file").onchange = (e) => importFile(e.target.files[0]);
window.addEventListener("beforeunload", (e) => {
  if (state.company) {
    e.preventDefault();
    e.returnValue = "";
  }
});

$("#editor").addEventListener("cancel", (e) => {
  e.preventDefault();
  closeEditor();
});
