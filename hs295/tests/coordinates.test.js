const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');

function coordinates(source='hs295/src/coordinates.js'){
  const code=fs.readFileSync(source,'utf8')+'\n;PDF_COORDS';
  return vm.runInNewContext(code);
}

test('gain uses the configured numeric box coordinates',()=>{
  const xs=coordinates().page1.gain.xs;
  assert.deepEqual(Array.from(xs),[150,166,182,198,214,230,246,262]);
  assert.equal(xs.length,8);
});

test('a maximum-length gain is assigned only inside the visible numeric-box range',()=>{
  const xs=coordinates().page1.gain.xs,value='9999999';
  const start=xs.length-value.length;
  const assigned=Array.from(value,(_,i)=>xs[start+i]+3);
  assert.equal(start,1);
  assert.equal(assigned.length,7);
  assert.ok(assigned.every(x=>x>=169&&x<=265));
});
