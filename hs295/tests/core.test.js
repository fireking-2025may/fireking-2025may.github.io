const test=require('node:test'),assert=require('node:assert/strict'),C=require('../src/core.js');
const people=[{id:'a',fullName:'Alex Example',address1:'1 Sample Road',address2:'Exampleton',country:'United Kingdom',postcode:'AB1 2CD',identifierType:'UTR',identifierValue:'1234567890'},{id:'b',fullName:'Blair Sample',address1:'2 Fiction Lane',address2:'Testford',country:'United Kingdom',postcode:'XY9 9ZZ',identifierType:'NINO',identifierValue:'AB123456C'}];
const company={legalName:'Example Holdings Limited',companyNumber:'01234567'};
const transfer={id:'t',transferorPersonId:'a',transfereePersonId:'b',numberShares:'10000',nominalValue:'1.2500',shareClass:'Ordinary',disposalDate:'2026-06-30',acquisitionDate:'2020-01-02',wholeAcquisitionCost:'25000',transferValue:'100000',reorganisationStatus:'none',reorganisationDetails:'',partDisposal:false,apportionedAcquisitionCost:'',retainedDetails:'',retainedValue:''};
test('full and part gain calculations',()=>{assert.equal(C.gain(transfer),75000);assert.equal(C.gain({...transfer,partDisposal:true,apportionedAcquisitionCost:'10000'}),90000)});
test('zero and negative gains block export',()=>{for(const v of ['25000','20000'])assert.match(C.transferErrors({...transfer,transferValue:v},people,company).join(),/positive gain/)});
test('whole pounds and dates',()=>{assert.equal(C.whole('12.5'),false);assert.equal(C.whole('12'),true);assert.equal(C.validDate('2024-02-29'),true);assert.equal(C.validDate('2023-02-29'),false)});
test('normalises identifiers and postcode',()=>{assert.equal(C.normalizeUTR('12345 67890'),'1234567890');assert.equal(C.normalizeNINO('ab 12 34 56 c'),'AB123456C');assert.equal(C.normalizePostcode('SW1A 1AA'),'SW1A 1AA')});
test('UTR is optional but must contain exactly 10 digits when provided',()=>{assert.doesNotMatch(C.personErrors({...people[0],identifierValue:''}).join(),/UTR/);assert.doesNotMatch(C.personErrors({...people[0],identifierValue:'   '}).join(),/UTR/);assert.match(C.personErrors({...people[0],identifierValue:'123456789'}).join(),/empty or contain exactly 10 digits/)});
test('asset description preserves meaningful precision and permits register-verified name only',()=>{assert.equal(C.assetDescription(transfer,company),'10,000 £1.25 Ordinary shares in Example Holdings Limited (company number 01234567)');assert.equal(C.assetDescription(transfer,{legalName:'Example Holdings Limited',companyNumber:''}),'10,000 £1.25 Ordinary shares in Example Holdings Limited')});
test('filename sanitisation and duplicates',()=>{assert.equal(C.sanitizeFilename('A:B?. '),'A-B-');assert.deepEqual(C.pdfFilenames([transfer,{...transfer,id:'t2'}],people),['Alex Example to Blair Sample - 30-06-2026.pdf','Alex Example to Blair Sample - 30-06-2026 (2).pdf'])});
test('ZIP taxpayer reference rule',()=>{assert.equal(C.taxpayerReference([transfer],people),'1234567890');assert.equal(C.taxpayerReference([transfer,{...transfer,transferorPersonId:'b'}],people),'Multiple Taxpayers')});
test('session round trip and rejects malformed versions/properties',()=>{const text=JSON.stringify(C.exportSession({company,people,transfers:[transfer]}));assert.equal(C.importSessionText(text).transfers.length,1);assert.throws(()=>C.importSessionText('{oops'),/Invalid JSON/);const x=JSON.parse(text);x.schemaVersion=9;assert.throws(()=>C.importSessionText(JSON.stringify(x)),/Unsupported/);x.schemaVersion=1;x.extra=true;assert.throws(()=>C.importSessionText(JSON.stringify(x)),/Unsupported/)});
test('text and numeric capacities block',()=>{assert.match(C.transferErrors({...transfer,transferValue:'12345678'},people,company).join(),/7 digits/);assert.ok(C.personErrors({...people[0],fullName:'X'.repeat(53)}).some(x=>/too long/.test(x)))});
test('gain capacity accepts seven digits and rejects eight',()=>{assert.equal(C.LIMITS.gainDigits,7);assert.doesNotMatch(C.transferErrors({...transfer,wholeAcquisitionCost:'0',transferValue:'9999999'},people,company).join(),/Held-over gain.*digits/);assert.match(C.transferErrors({...transfer,wholeAcquisitionCost:'10000000',transferValue:'99999999'},people,company).join(),/Held-over gain must be no more than 7 digits/) });
test('part disposal requires separately apportioned cost and retained fields',()=>{const e=C.transferErrors({...transfer,partDisposal:true,apportionedAcquisitionCost:'',retainedDetails:'',retainedValue:''},people,company).join();assert.match(e,/Apportioned/);assert.match(e,/part retained/)});

test('estimated transfer value is limited to seven digits and leading zeroes do not consume PDF capacity',()=>{assert.match(C.transferErrors({...transfer,transferValue:'12345678'},people,company).join(),/7 digits/);assert.doesNotMatch(C.transferErrors({...transfer,wholeAcquisitionCost:'00025000',transferValue:'0100000'},people,company).join(),/digits/)})
test('reorganisation details are always optional',()=>assert.doesNotMatch(C.transferErrors({...transfer,reorganisationStatus:'',reorganisationDetails:''},people,company).join(),/reorganisation status|reorganisation or bonus/i))

test('session import normalises missing optional reorganisation text',()=>{const x=C.exportSession({company,people,transfers:[{...transfer,reorganisationStatus:'details',reorganisationDetails:null}]});const imported=C.importSessionText(JSON.stringify(x)).transfers[0];assert.equal(imported.reorganisationDetails,'');assert.equal(imported.reorganisationStatus,'none')})

test('names accept letters, single spaces and correctly placed hyphens',()=>{
  for(const name of ['Alex Example','Anne-Marie Example','Jean-Luc Picard','Mary-Jane Smith-Jones'])assert.equal(C.validName(name),true,name);
  for(const name of ['Alex7 Example','-Alex Example','Alex- Example','Alex -Example','Alex--Example','Alex Example-','Alex  Example'])assert.equal(C.validName(name),false,name);
  assert.match(C.personErrors({...people[0],fullName:'Alex--Example'}).join(),/letters, single spaces and correctly placed hyphens only/);
});

test('postcode requires the stated uppercase format and space',()=>{
  assert.equal(C.normalizePostcode('SW1A 1AA'),'SW1A 1AA');
  assert.equal(C.normalizePostcode('sw1a 1aa'),null);
  assert.equal(C.normalizePostcode('SW1A1AA'),null);
  assert.match(C.personErrors({...people[0],postcode:'SW1A1AA'}).join(),/required format/);
});

test('session import migrates legacy version-1 person syntax',()=>{
  const legacyPerson={...people[0],fullName:"Anne-Marie O'Neill",postcode:'sw1a1aa'};
  const text=JSON.stringify(C.exportSession({company,people:[legacyPerson,people[1]],transfers:[transfer]}));
  const imported=C.importSessionText(text);
  assert.equal(imported.people[0].fullName,"Anne-Marie O'Neill");
  assert.equal(imported.people[0].postcode,'SW1A 1AA');
});

test('canonical totals use exact deterministic half-up rounding',()=>{
  assert.equal(C.canonicalTotal('1.005','100','perShare'),'101');
  assert.equal(C.canonicalTotal('0.0049','100','perShare'),'0');
  assert.equal(C.canonicalTotal('00042','10','total'),'42');
  assert.equal(C.canonicalTotal('1.2','bad','perShare'),null);
});

test('shared transfer normalisation supports total and per-share entry modes',()=>{
  const normalized=C.normalizeTransfer({...transfer,numberShares:'3',wholeAcquisitionCostMode:'perShare',wholeAcquisitionCostEntry:'10.50',transferValueMode:'perShare',transferValueEntry:'40.50'});
  assert.equal(normalized.wholeAcquisitionCost,'32');
  assert.equal(normalized.transferValue,'122');
  assert.equal(C.gain(normalized),90);
  assert.deepEqual(C.transferErrors(normalized,people,company),[]);
});

test('session round trip retains batch entry modes and canonical PDF totals',()=>{
  const normalized=C.normalizeTransfer({...transfer,wholeAcquisitionCostMode:'perShare',wholeAcquisitionCostEntry:'2.50',transferValueMode:'total',transferValueEntry:'100000'});
  const imported=C.importSessionText(JSON.stringify(C.exportSession({company,people,transfers:[normalized]}))).transfers[0];
  assert.equal(imported.wholeAcquisitionCostMode,'perShare');
  assert.equal(imported.wholeAcquisitionCostEntry,'2.50');
  assert.equal(imported.wholeAcquisitionCost,'25000');
});

test('share flow totals are exact, ordered by people and do not mutate inputs',()=>{
  const inputs=[transfer,{...transfer,id:'t2',transferorPersonId:'b',transfereePersonId:'a',numberShares:'2500'},{...transfer,id:'t3',numberShares:'10000'}],snapshot=structuredClone(inputs);
  assert.deepEqual(C.shareFlowTotals(people,inputs),[
    {personId:'a',transferred:'20000',received:'2500'},
    {personId:'b',transferred:'2500',received:'20000'}
  ]);
  assert.deepEqual(inputs,snapshot);
});

test('share flow totals include people with no transfers and fail closed on malformed references',()=>{
  assert.deepEqual(C.shareFlowTotals(people,[]).map(x=>[x.transferred,x.received]),[['0','0'],['0','0']]);
  assert.throws(()=>C.shareFlowTotals(people,[{...transfer,numberShares:'1.5'}]),/invalid share count/);
  assert.throws(()=>C.shareFlowTotals(people,[{...transfer,transfereePersonId:'missing'}]),/unknown person/);
});
