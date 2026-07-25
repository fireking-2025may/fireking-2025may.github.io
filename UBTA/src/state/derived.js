const text=runs=>(runs||[]).map(run=>run.text).join('').trim();
export const anchorFor=entity=>`anchor-${entity.id}`;
export function deriveSteps(steps=[]){return steps.map((step,index)=>({id:step.id,anchor:anchorFor(step),number:index+1,label:`Step ${index+1}. ${step.title}`}))}
export function deriveProposals(steps=[]){return steps.map((step,index)=>({id:`proposal-${step.id}`,stepId:step.id,anchor:`proposal-${step.id}`,stepAnchor:anchorFor(step),number:index+1,title:`Proposal ${index+1}`,text:step.proposal??text(step.blocks?.find(block=>block.type==='paragraph')?.runs)??''}))}
export function deriveContent(document){return {steps:deriveSteps(document?.steps),proposals:document?.settings?.showProposals===false?[]:deriveProposals(document?.steps)}}
