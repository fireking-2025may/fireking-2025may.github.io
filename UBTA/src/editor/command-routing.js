export const TABLE_COMMANDS = new Set(['addRow','removeRow','addColumn','removeColumn']);

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
