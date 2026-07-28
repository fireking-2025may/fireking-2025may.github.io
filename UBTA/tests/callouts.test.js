import test from 'node:test';
import assert from 'node:assert/strict';
import { bindCallouts } from '../src/editor/callouts.js';

function harness(){
  const listeners={},cancelListeners={},button={addEventListener:(type,handler)=>cancelListeners[type]=handler};
  const dialog={closeCalls:[],addEventListener:(type,handler)=>listeners[type]=handler,querySelectorAll:()=>[button],close(value){this.closeCalls.push(value)}};
  bindCallouts({querySelectorAll:()=>[dialog]});
  return {dialog,button,listeners,cancelListeners};
}

test('clicking a dialog backdrop closes its callout',()=>{
  const {dialog,listeners}=harness();
  listeners.click({target:dialog});
  assert.deepEqual(dialog.closeCalls,['cancel']);
});

test('clicking within a callout leaves it open',()=>{
  const {dialog,listeners}=harness();
  listeners.click({target:{}});
  assert.deepEqual(dialog.closeCalls,[]);
});

test('cancel controls always prevent submission and close their callout',()=>{
  const {dialog,cancelListeners}=harness();let prevented=false;
  cancelListeners.click({preventDefault:()=>prevented=true});
  assert.equal(prevented,true);
  assert.deepEqual(dialog.closeCalls,['cancel']);
});
