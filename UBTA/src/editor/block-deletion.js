const textFromRuns = (runs) =>
  (runs || []).map((run) => run.text || '').join('');

export function blockTypeLabel(type) {
  return (
    {
      paragraph: 'paragraph',
      heading: 'heading',
      bulletList: 'bulleted list',
      numberList: 'numbered list',
      table: 'table',
      image: 'image',
    }[type] || 'block'
  );
}

export function blockHasContent(block) {
  if (['table', 'image'].includes(block.type)) return true;
  if (block.type.endsWith('List'))
    return (block.items || []).some((item) => textFromRuns(item.runs).trim());
  return Boolean(textFromRuns(block.runs).trim());
}

export function removeBlockFromGroup(group, id, makeId) {
  const index = group.blocks.findIndex((block) => block.id === id);
  if (index < 0) return null;
  const [deleted] = group.blocks.splice(index, 1);
  let replacement = null;
  if (!group.blocks.length) {
    replacement = { id: makeId('block'), type: 'paragraph', runs: [] };
    group.blocks.push(replacement);
  }
  return {
    deleted,
    target: group.blocks[index - 1] || group.blocks[index] || replacement,
  };
}
