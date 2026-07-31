export const PROHIBITED_APIS=['fetch','XMLHttpRequest','WebSocket','EventSource','sendBeacon','serviceWorker','RTCPeerConnection','localStorage','sessionStorage','indexedDB','caches','cookies','historyState'];
export function createProhibitedApiDoubles() { const doubles={}; for(const name of PROHIBITED_APIS) Object.defineProperty(doubles,name,{get(){throw new Error(`Prohibited runtime API accessed: ${name}`);},configurable:true}); return doubles; }
export function installRuntimeGuards(target=globalThis) {
  const fail=name=>()=>{throw new Error(`Prohibited runtime API accessed: ${name}`);};
  for(const name of ['fetch','XMLHttpRequest','WebSocket','EventSource','RTCPeerConnection']) if(name in target) Object.defineProperty(target,name,{value:fail(name),configurable:true});
  for(const name of ['localStorage','sessionStorage','indexedDB','caches']) if(name in target) Object.defineProperty(target,name,{get:fail(name),configurable:true});
  if(target.navigator){for(const name of ['sendBeacon','serviceWorker']) try{Object.defineProperty(target.navigator,name,{get:fail(name),configurable:true});}catch{/* browser may make it non-configurable */}}
}
