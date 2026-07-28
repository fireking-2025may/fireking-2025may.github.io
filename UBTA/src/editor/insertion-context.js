export function insertionIndex(group, context) {
  if (!group || group.id !== context?.groupId) return -1;
  if (!context.referenceBlockId) return group.blocks.length;
  const referenceIndex = group.blocks.findIndex(block => block.id === context.referenceBlockId);
  if (referenceIndex < 0) return group.blocks.length;
  return referenceIndex + (context.position === 'before' ? 0 : 1);
}

export function insertionContextFromPoint(groupId, blockElements, clientY) {
  const blocks = blockElements
    .map(element => ({ id: element.dataset.blockId, rect: element.getBoundingClientRect() }))
    .filter(item => item.id && item.rect.height > 0)
    .sort((a, b) => a.rect.top - b.rect.top);
  const preceding = blocks.filter(item => item.rect.top <= clientY).at(-1);
  if (preceding) return { groupId, referenceBlockId: preceding.id, position: 'after' };
  const following = blocks.find(item => item.rect.top > clientY);
  if (following) return { groupId, referenceBlockId: following.id, position: 'before' };
  return { groupId, referenceBlockId: null, position: 'after' };
}
