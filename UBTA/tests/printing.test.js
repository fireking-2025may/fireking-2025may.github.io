import assert from 'node:assert/strict';
import test from 'node:test';
import { PrintLifecycle } from '../src/editor/printing.js';

function harness(overrides={}){
  const calls=[];let afterPrint,fallback;
  const lifecycle=new PrintLifecycle({flush:()=>calls.push('flush'),cancelPagination:()=>calls.push('cancel'),render:async()=>{calls.push('render-start');await Promise.resolve();calls.push('render-end');return true},prepare:()=>{calls.push('prepare');return 'view'},clearPresentation:()=>calls.push('clear'),print:()=>calls.push('print'),restore:view=>calls.push(`restore-${view}`),onAfterPrint:callback=>{afterPrint=callback;return()=>calls.push('remove-listener')},setFallback:callback=>(fallback=callback,1),clearFallback:()=>calls.push('clear-fallback'),...overrides});
  return {lifecycle,calls,after:()=>afterPrint(),fallback:()=>fallback()};
}
test('print flushes pending edits and waits for immediate pagination',async()=>{const h=harness();await h.lifecycle.run();assert.deepEqual(h.calls.slice(0,7),['prepare','flush','cancel','render-start','render-end','clear','print'])});
test('failed pagination never prints and restores the UI',async()=>{const h=harness({render:async()=>false});assert.equal(await h.lifecycle.run(),false);assert.ok(!h.calls.includes('print'));assert.ok(h.calls.includes('restore-view'));assert.equal(h.lifecycle.busy,false)});
test('overlapping print requests are ignored',async()=>{let finish;const h=harness({render:()=>new Promise(resolve=>finish=resolve)});const first=h.lifecycle.run();assert.equal(await h.lifecycle.run(),false);finish(true);await first;assert.equal(h.calls.filter(call=>call==='print').length,1)});
test('afterprint performs cleanup and restores the prior view',async()=>{const h=harness();await h.lifecycle.run();h.after();assert.ok(h.calls.includes('clear-fallback'));assert.ok(h.calls.includes('restore-view'));assert.equal(h.lifecycle.busy,false)});
test('fallback performs cleanup when afterprint is unavailable',async()=>{const h=harness();await h.lifecycle.run();h.fallback();assert.ok(h.calls.includes('remove-listener'));assert.ok(h.calls.includes('restore-view'));assert.equal(h.lifecycle.busy,false)});
