export class NavigationHistory {
  constructor() { this.entries = []; }
  push(view) { if (view) this.entries.push(structuredClone(view)); }
  undo() { return this.entries.length ? this.entries.pop() : null; }
  clear() { this.entries.length = 0; }
}
