import assert from 'node:assert/strict';
import test from 'node:test';
import { PrintLifecycle } from '../src/editor/printing.js';

function harness(overrides={}){
  const calls=[];let afterPrint,fallback;
  const lifecycle=new PrintLifecycle({flush:()=>calls.push('flush'),cancelPagination:()=>calls.push('cancel'),render:async()=>{calls.push('render-start');await Promise.resolve();calls.push('render-end');return true},prepare:()=>{calls.push('prepare');return 'view'},clearPresentation:()=>calls.push('clear'),print:()=>calls.push('print'),restore:view=>calls.push(`restore-${view}`),onAfterPrint:callback=>{afterPrint=callback;return()=>calls.push('remove-listener')},setFallback:callback=>(fallback=callback,1),clearFallback:()=>calls.push('clear-fallback'),...overrides});
  return {lifecycle,calls,after:()=>afterPrint(),fallback:()=>fallback()};
}
test('print flushes pending edits and waits for pagination and TOC reconciliation',async()=>{const h=harness();await h.lifecycle.run();assert.deepEqual(h.calls.slice(0,9),['prepare','flush','cancel','render-start','render-end','render-start','render-end','clear','print'])});
test('failed pagination never prints, restores the UI, and reports the error',async()=>{let reported;const h=harness({render:async()=>false,onError:error=>reported=error});assert.equal(await h.lifecycle.run(),false);assert.ok(!h.calls.includes('print'));assert.ok(h.calls.includes('restore-view'));assert.match(reported.message,/could not be prepared/);assert.equal(h.lifecycle.busy,false)});
test('a failed TOC reconciliation never opens the print dialog',async()=>{let pass=0,reported;const h=harness({render:async()=>++pass===1,onError:error=>reported=error});assert.equal(await h.lifecycle.run(),false);assert.equal(pass,2);assert.ok(!h.calls.includes('print'));assert.match(reported.message,/reconciled/)});
test('overlapping print requests are ignored',async()=>{let finish,pass=0;const h=harness({render:()=>++pass===1?new Promise(resolve=>finish=resolve):true});const first=h.lifecycle.run();assert.equal(await h.lifecycle.run(),false);finish(true);await first;assert.equal(h.calls.filter(call=>call==='print').length,1)});
test('afterprint performs cleanup and restores the prior view',async()=>{const h=harness();await h.lifecycle.run();h.after();assert.ok(h.calls.includes('clear-fallback'));assert.ok(h.calls.includes('restore-view'));assert.equal(h.lifecycle.busy,false)});
test('fallback performs cleanup when afterprint is unavailable',async()=>{const h=harness();await h.lifecycle.run();h.fallback();assert.ok(h.calls.includes('remove-listener'));assert.ok(h.calls.includes('restore-view'));assert.equal(h.lifecycle.busy,false)});
