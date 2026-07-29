const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
export class History {
  constructor(value) {
    this.past = [];
    this.future = [];
    this.value = structuredClone(value);
  }
  checkout() {
    return structuredClone(this.value);
  }
  commit(value) {
    const next = structuredClone(value);
    if (same(this.value, next)) return false;
    this.past.push(structuredClone(this.value));
    this.value = next;
    this.future = [];
    return true;
  }
  replace(value) {
    this.value = structuredClone(value);
  }
  peekUndo() {
    return this.past.length ? structuredClone(this.past.at(-1)) : null;
  }
  peekRedo() {
    return this.future.length ? structuredClone(this.future.at(-1)) : null;
  }
  undo() {
    if (!this.past.length) return this.value;
    this.future.push(structuredClone(this.value));
    this.value = this.past.pop();
    return structuredClone(this.value);
  }
  redo() {
    if (!this.future.length) return this.value;
    this.past.push(structuredClone(this.value));
    this.value = this.future.pop();
    return structuredClone(this.value);
  }
}
