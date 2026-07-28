export function renderList(block,renderRuns,escapeAttribute=String){
  const tag=block.type==='numberList'?'ol':block.type==='bulletList'?'ul':null;
  if(!tag)return '';
  const counters=[0,0,0];
  const items=block.items.map(item=>{
    const level=Math.min(3,Math.max(1,Number(item.level)||1));
    counters[level-1]++;
    counters.fill(0,level);
    const value=tag==='ol'?` value="${counters[level-1]}"`:'';
    return `<li class="list-item" contenteditable="true" data-container="item" data-item-id="${escapeAttribute(item.id)}" data-level="${level}" aria-level="${level}"${value}>${renderRuns(item.runs)}</li>`;
  }).join('');
  return `<${tag} class="list-items" data-list-type="${block.type}">${items}</${tag}><button class="list-trailing-hit-area" type="button" aria-label="Add another list item" data-list-append></button>`;
}
