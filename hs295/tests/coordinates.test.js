const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');

function coordinates(source='hs295/src/coordinates.js'){
  const code=fs.readFileSync(source,'utf8')+'\n;PDF_COORDS';
  return vm.runInNewContext(code);
}

test('gain uses exactly the seven visible numeric boxes after the pound-sign box',()=>{
  const xs=coordinates().page1.gain.xs;
  assert.deepEqual(Array.from(xs),[176,193,210,227,244,261,278]);
  assert.equal(xs.length,7);
});

test('a maximum-length gain is assigned only inside the visible numeric-box range',()=>{
  const xs=coordinates().page1.gain.xs,value='9999999';
  const start=xs.length-value.length;
  const assigned=Array.from(value,(_,i)=>xs[start+i]+3);
  assert.equal(start,0);
  assert.equal(assigned.length,7);
  assert.ok(assigned.every(x=>x>=179&&x<=281));
});
