export const TABLE_COMMANDS = new Set(['addRow','removeRow','moveRowUp','moveRowDown','addColumn','removeColumn','moveColumnLeft','moveColumnRight']);

export const canApplyBlockStyle = (blockType, action) => {
  if (/^heading[1-4]$/.test(action)) return ['paragraph', 'heading'].includes(blockType);
  if (action === 'body') return ['paragraph', 'heading'].includes(blockType);
  return ['bulletList', 'numberList'].includes(action) && ['paragraph', 'heading', 'bulletList', 'numberList'].includes(blockType);
};

export const blockStyleChoices = blockType => ['bulletList','numberList'].includes(blockType)
  ? [['bulletList','Bulleted list'],['numberList','Numbered list']]
  : [['body','Body text'],['heading1','Heading 1'],['heading2','Heading 2'],['heading3','Heading 3'],['heading4','Heading 4'],['bulletList','Bulleted list'],['numberList','Numbered list']];

export const BLOCK_COMMANDS = new Map([
  ['addParagraph','paragraph'],
  ['addHeading','heading'],
  ['addBulletList','bulletList'],
  ['addNumberList','numberList'],
  ['addTable','table']
]);

export function routeInsertionCommand(action){
  if(TABLE_COMMANDS.has(action))return {kind:'table',action};
  const blockType=BLOCK_COMMANDS.get(action);
  if(typeof blockType==='string')return {kind:'block',blockType};
  return null;
}
