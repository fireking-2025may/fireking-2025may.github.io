export function renderList(block, renderRuns, escapeAttribute = String) {
  const tag =
    block.type === 'numberList'
      ? 'ol'
      : block.type === 'bulletList'
        ? 'ul'
        : null;
  if (!tag) return '';
  const counters = [0, 0, 0];
  const items = block.items
    .map((item) => {
      const level = Math.min(3, Math.max(1, Number(item.level) || 1));
      counters[level - 1]++;
      counters.fill(0, level);
      const value = tag === 'ol' ? ` value="${counters[level - 1]}"` : '';
      return `<li class="list-row" data-level="${level}"><span class="list-item" contenteditable="true" data-container="item" data-item-id="${escapeAttribute(item.id)}" data-level="${level}" aria-level="${level}"${value}>${renderRuns(item.runs)}</span><button class="delete-list-item" type="button" data-list-delete="${escapeAttribute(item.id)}" aria-label="Delete empty list line"${item.runs?.some((run) => run.text.trim()) ? ' disabled title="Only empty list lines can be deleted"' : ''}>Delete</button></li>`;
    })
    .join('');
  return `<${tag} class="list-items" data-list-type="${block.type}">${items}</${tag}><button class="list-trailing-hit-area" type="button" aria-label="Add another list item" data-list-append></button>`;
}
