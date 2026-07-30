export function addItem(items, factory) { return [...items, factory()]; }
export function removeItem(items, id) { return items.length === 1 ? items : items.filter(item => item.id !== id); }
export function moveItem(items, id, delta) { const from=items.findIndex(x=>x.id===id); const to=from+delta; if(from<0||to<0||to>=items.length)return items; const result=[...items]; [result[from],result[to]]=[result[to],result[from]]; return result; }
