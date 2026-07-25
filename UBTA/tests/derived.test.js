import test from 'node:test';
import assert from 'node:assert/strict';
import {deriveStepDescriptors} from '../src/state/derived.js';
import {normaliseDocument} from '../src/state/schema.js';

const document = normaliseDocument({meta:{},sections:[],steps:[
  {id:'alpha',title:'Alpha',summary:'First summary',blocks:[]},
  {id:'beta',title:'Beta',summary:'Second summary',proposal:'Custom proposal',blocks:[{id:'review',type:'paragraph',runs:[{text:'Check',highlight:true}]}]}
]});

test('derives ordered, stable step and proposal anchors without mutating state',()=>{const before=structuredClone(document);const descriptors=deriveStepDescriptors(document);assert.deepEqual(descriptors.map(d=>[d.stepNumber,d.id,d.anchor,d.proposalAnchor]),[[1,'alpha','anchor-alpha','anchor-proposal-alpha'],[2,'beta','anchor-beta','anchor-proposal-beta']]);assert.deepEqual(document,before)});
test('uses deterministic generated text unless an override is persisted',()=>{const [generated,overridden]=deriveStepDescriptors(document);assert.equal(generated.proposalText,'Proposal: Alpha — First summary');assert.equal(generated.proposalSource,'generated');assert.equal(overridden.proposalText,'Custom proposal');assert.equal(overridden.proposalSource,'override');assert.equal(overridden.reviewState,'review')});
test('reordering renumbers labels but preserves ID-derived links',()=>{const [beta]=deriveStepDescriptors({...document,steps:[...document.steps].reverse()});assert.equal(beta.tocLabel,'Step 1. Beta');assert.equal(beta.anchor,'anchor-beta');assert.equal(beta.proposalAnchor,'anchor-proposal-beta')});
