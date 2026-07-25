export const SCHEMA_VERSION=2;
export const TABLE_MIN_WIDTH=8;
export const TABLE_MAX_WIDTH=92;
export const IMAGE_MIN_WIDTH=10;
export const IMAGE_MAX_WIDTH=100;
export const IMAGE_MAX_HEIGHT=170;
export const IMAGE_MIME_TYPES=['image/png','image/jpeg','image/gif','image/webp'];

export const safeHref=value=>{
  if(typeof value!=='string')return null;
  const href=value.trim();
  return /^(https?:\/\/|mailto:|#[A-Za-z][\w:.-]*$)/i.test(href)?href:null;
};

const rawId=(value,prefix)=>/^[A-Za-z][\w-]*$/.test(value||'')?value:`${prefix}-${globalThis.crypto?.randomUUID?.()||Math.random().toString(36).slice(2)}`;
const uniqueId=(value,prefix,seen)=>{
  const base=rawId(value,prefix);let candidate=base,n=2;
  while(seen.has(candidate))candidate=`${base}-${n++}`;
  seen.add(candidate);return candidate;
};
const clamp=(value,min,max,fallback)=>Math.min(max,Math.max(min,Number.isFinite(Number(value))?Number(value):fallback));

export function normaliseRuns(runs){
  return (Array.isArray(runs)?runs:[]).map(run=>({text:String(run?.text??''),highlight:run?.highlight===true,link:safeHref(run?.link?.href)?{href:safeHref(run.link.href)}:null})).filter(run=>run.text).reduce((result,run)=>{
    const previous=result.at(-1);
    if(previous&&previous.highlight===run.highlight&&previous.link?.href===run.link?.href)previous.text+=run.text;
    else result.push(run);
    return result;
  },[]);
}

export function normaliseWidths(widths,columnCount){
  if(!columnCount)return [];
  const supplied=Array.from({length:columnCount},(_,i)=>clamp(widths?.[i],TABLE_MIN_WIDTH,TABLE_MAX_WIDTH,100/columnCount));
  const total=supplied.reduce((a,b)=>a+b,0)||1;
  let result=supplied.map(width=>width*100/total);
  // Rebalance after clamping. This also guarantees the serialised total is 100.
  result=result.map(width=>clamp(width,TABLE_MIN_WIDTH,TABLE_MAX_WIDTH,100/columnCount));
  const difference=100-result.reduce((a,b)=>a+b,0);
  result[result.length-1]+=difference;
  return result.map(width=>Math.round(width*1000)/1000);
}

function normaliseTable(block,seen){
  const sourceRows=Array.isArray(block.rows)?block.rows:[];
  const columnCount=Math.max(1,Math.min(12,Number(block.columnCount)||Math.max(0,...sourceRows.map(row=>Array.isArray(row?.cells)?row.cells.length:0))||1));
  const rows=sourceRows.filter(row=>row&&Array.isArray(row.cells)).map(row=>({
    id:uniqueId(row.id,'row',seen),
    cells:Array.from({length:columnCount},(_,index)=>{
      const cell=row.cells[index];
      return {id:uniqueId(cell?.id,'cell',seen),runs:normaliseRuns(cell?.runs??(cell==null?[]:[{text:String(cell)}]))};
    })
  }));
  const totalsSource=block.totals||{};
  return {columnCount,widths:normaliseWidths(block.widths,columnCount),rows,totals:{enabled:totalsSource.enabled===true,label:String(totalsSource.label??'Total'),columns:[...new Set((Array.isArray(totalsSource.columns)?totalsSource.columns:[]).map(Number).filter(n=>Number.isInteger(n)&&n>=0&&n<columnCount))]}};
}

export function safeImageSource(source,mime){
  if(typeof source!=='string'||!IMAGE_MIME_TYPES.includes(mime))return null;
  if(source.startsWith('data:'))return source.startsWith(`data:${mime};base64,`)&&/^[A-Za-z0-9+/]*={0,2}$/.test(source.split(',')[1]||'')?source:null;
  if(source.startsWith('blob:'))return source;
  return null;
}

export function normaliseBlock(block,context={seen:new Set()}){
  const seen=context.seen||new Set();
  const accepted=['heading','paragraph','bulletList','numberList','table','image'];
  const type=accepted.includes(block?.type)?block.type:'paragraph';
  const out={id:uniqueId(block?.id,'block',seen),type};
  if(type==='heading'){out.level=clamp(block.level,1,4,1);out.runs=normaliseRuns(block.runs)}
  else if(type==='paragraph')out.runs=normaliseRuns(block.runs);
  else if(type.endsWith('List'))out.items=(Array.isArray(block.items)?block.items:[]).map(item=>({id:uniqueId(item?.id,'item',seen),level:clamp(item?.level,1,3,1),runs:normaliseRuns(item?.runs)}));
  else if(type==='table')Object.assign(out,normaliseTable(block,seen));
  else {
    const mime=IMAGE_MIME_TYPES.includes(block.mime)?block.mime:'image/png';
    Object.assign(out,{src:safeImageSource(block.src,mime),mime,alt:String(block.alt??''),caption:String(block.caption??''),width:clamp(block.width,IMAGE_MIN_WIDTH,IMAGE_MAX_WIDTH,60),height:clamp(block.height,1,IMAGE_MAX_HEIGHT,80)});
  }
  return out;
}

export function migrateDocument(input){
  const document=structuredClone(input||{});
  const version=Number(document.schemaVersion??1);
  if(version<1||version>SCHEMA_VERSION)throw Error('Unsupported schema version');
  if(version===1){document.schemaVersion=2;document.settings={...(document.settings||{}),showProposals:true};}
  return document;
}

export function normaliseDocument(input){
  const document=migrateDocument(input);const seen=new Set();const meta={};
  for(const key of ['clientName','projectTitle','documentType','date','version','subtitle','adviser','status'])meta[key]=String(document?.meta?.[key]??'');
  const groups=key=>(Array.isArray(document?.[key])?document[key]:[]).map(group=>({id:uniqueId(group?.id,key.slice(0,-1),seen),title:String(group?.title??''),proposal:typeof group?.proposal==='string'?group.proposal:null,blocks:(Array.isArray(group?.blocks)?group.blocks:[]).map(block=>normaliseBlock(block,{seen}))}));
  return {schemaVersion:SCHEMA_VERSION,meta,settings:{showProposals:document?.settings?.showProposals!==false},sections:groups('sections'),steps:groups('steps')};
}
export function validateDocument(document){return normaliseDocument(document)}

export const seedDocument=normaliseDocument({schemaVersion:2,meta:{clientName:'Example Client Ltd',projectTitle:'Corporate Restructure',documentType:'Steps Plan',subtitle:'Detailed Steps Plan',date:'25 July 2026',version:'v1',adviser:'UBTA Accountants Ltd',status:'Draft'},settings:{showProposals:true},sections:[{id:'scope',title:'Scope of works',blocks:[{id:'scope-h',type:'heading',level:2,runs:[{text:'Purpose and scope'}]},{id:'scope-p1',type:'paragraph',runs:[{text:'This plan outlines the principal implementation steps for a proposed corporate restructure.'}]},{id:'scope-p2',type:'paragraph',runs:[{text:'Figures and legal mechanics marked for review require completion before finalisation.',highlight:true}]}]}],steps:[{id:'share-restructure',title:'Implement the corporate share restructure',blocks:[{id:'step-h',type:'heading',level:2,runs:[{text:'Implementation'}]},{id:'step-p1',type:'paragraph',runs:[{text:'The directors will approve the proposed corporate restructure and authorise the required documentation.'}]},{id:'step-list',type:'numberList',items:[{id:'st1',level:1,runs:[{text:'Prepare board minutes and resolutions.'}]},{id:'st2',level:2,runs:[{text:'Check company records and statutory filings.'}]}]}]}]});
