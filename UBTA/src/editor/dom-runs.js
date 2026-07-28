import { normaliseRuns, safeHref } from '../state/schema.js';

export function runsFromElement(element) {
  const runs=[];
  const visit=(node,highlight=false,link=null)=>{if(node.nodeType===3){runs.push({text:node.nodeValue||'',highlight,link:link?{href:link}:null});return}if(node.nodeType!==1)return;const tag=node.tagName.toLowerCase();if(tag==='br'){runs.push({text:'\n',highlight,link:link?{href:link}:null});return}const nextHighlight=highlight||tag==='mark',nextLink=tag==='a'?(safeHref(node.getAttribute('href'))||link):link;[...node.childNodes].forEach(child=>visit(child,nextHighlight,nextLink));if(['div','p'].includes(tag)&&node.nextSibling&&runs.at(-1)?.text?.at(-1)!=='\n')runs.push({text:'\n',highlight:nextHighlight,link:nextLink?{href:nextLink}:null});};
  [...element.childNodes].forEach(node=>visit(node));return normaliseRuns(runs);
}

export function transformRuns(runs, offsets, kind, value) {
  if(!offsets)return runs;const collapsed=offsets[0]===offsets[1];if(collapsed&&kind!=='highlight')return runs;const [start,end]=collapsed?[0,runs.reduce((length,run)=>length+run.text.length,0)]:offsets;if(start===end)return runs;let position=0;const selected=runs.filter(run=>{const from=position,until=position+=run.text.length;return from<end&&until>start});const highlight=kind==='highlight'?!selected.every(run=>run.highlight):null;position=0;const output=[];
  for(const run of runs){const from=position,until=position+run.text.length,a=Math.max(start,from),b=Math.min(end,until);if(from<a)output.push({...run,text:run.text.slice(0,a-from)});if(b>a){const part={...run,text:run.text.slice(a-from,b-from)};if(kind==='highlight')part.highlight=highlight;if(kind==='link')part.link={href:value};if(kind==='unlink')part.link=null;output.push(part)}if(b<until)output.push({...run,text:run.text.slice(Math.max(0,b-from))});position=until}return normaliseRuns(output);
}

export function insertPlainText(event, documentObject=document) { event.preventDefault();documentObject.execCommand('insertText',false,event.clipboardData.getData('text/plain')); }
