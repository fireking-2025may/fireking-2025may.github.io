export function renderList(block,renderRuns,escapeAttribute=String){
  const tag=block.type==='numberList'?'ol':block.type==='bulletList'?'ul':null;
  if(!tag)return '';
  const items=block.items.map(item=>`<li class="list-item" contenteditable="true" data-container="item" data-item-id="${escapeAttribute(item.id)}" data-level="${item.level}">${renderRuns(item.runs)}</li>`).join('');
  return `<${tag} class="list-items" data-list-type="${block.type}">${items}</${tag}>`;
}
