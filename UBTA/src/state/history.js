const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
export class History {
  constructor(value) { this.past = []; this.future = []; this.pastMeta = []; this.futureMeta = []; this.value = structuredClone(value); }
  checkout() { return structuredClone(this.value); }
  commit(value) { const next = structuredClone(value); if (same(this.value, next)) return false; this.past.push(structuredClone(this.value)); this.pastMeta.push(null); this.value = next; this.future = []; this.futureMeta = []; return true; }
  checkpoint(meta) { this.past.push(structuredClone(this.value)); this.pastMeta.push(structuredClone(meta)); this.future = []; this.futureMeta = []; return true; }
  replace(value) { this.value = structuredClone(value); }
  peekUndo() { return this.past.length ? structuredClone(this.past.at(-1)) : null; }
  peekRedo() { return this.future.length ? structuredClone(this.future.at(-1)) : null; }
  undoEntry() { if (!this.past.length) return {value:structuredClone(this.value),meta:null,changed:false}; const meta=this.pastMeta.pop()??null; this.future.push(structuredClone(this.value)); this.futureMeta.push(structuredClone(meta)); this.value=this.past.pop(); return {value:structuredClone(this.value),meta:structuredClone(meta),changed:true}; }
  redoEntry() { if (!this.future.length) return {value:structuredClone(this.value),meta:null,changed:false}; const meta=this.futureMeta.pop()??null; this.past.push(structuredClone(this.value)); this.pastMeta.push(structuredClone(meta)); this.value=this.future.pop(); return {value:structuredClone(this.value),meta:structuredClone(meta),changed:true}; }
  undo() { return this.undoEntry().value; }
  redo() { return this.redoEntry().value; }
}
