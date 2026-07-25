import {normaliseRuns,safeHref} from './state/schema.js';
const children=node=>Array.from(node?.childNodes||node?.children||[]);
const tag=node=>String(node?.tagName||node?.nodeName||'').toLowerCase();
const nodeText=node=>node?.nodeType===3||tag(node)==='#text'?String(node.nodeValue??node.textContent??''):null;
export function domToRuns(root){
  const runs=[];
  const visit=(node,mark=false,link=null)=>{
    const value=nodeText(node);if(value!==null){runs.push({text:value,highlight:mark,link:link?{href:link}:null});return}
    const name=tag(node);if(['script','style','iframe','object'].includes(name))return;
    const nextMark=mark||name==='mark';const href=name==='a'?safeHref(node.getAttribute?.('href')??node.href):link;
    const nodes=children(node);if(name==='br')runs.push({text:'\n',highlight:nextMark,link:href?{href}:null});else nodes.forEach(child=>visit(child,nextMark,href));
  };
  children(root).forEach(child=>visit(child));return normaliseRuns(runs);
}
