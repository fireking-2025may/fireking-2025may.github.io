import {chromium} from 'playwright';
import path from 'node:path';
import fs from 'node:fs';
import {decodePDFRawStream,PDFDocument,PDFName} from 'pdf-lib';

const browser=await chromium.launch({headless:true}),page=await browser.newPage({acceptDownloads:true});
await page.goto('file://'+path.resolve('hs295_holdover_relief_2026.html'));
await page.fill('#company-name','Example Holdings Limited');
await page.click('text=Create batch');

async function addPerson(person){
  await page.click('#add-person');
  for(const [key,value] of Object.entries(person))if(key!=='identifierType')await page.fill(`[name=${key}]`,value);
  if(person.identifierType==='NINO')await page.click("text=I don't have a UTR");
  await page.click('#person-form button[type=submit]');
}
await addPerson({fullName:'Anne-Marie Example',address1:'1 Sample Road',address2:'Exampleton',country:'United Kingdom',postcode:'AB1 2CD',identifierType:'UTR',identifierValue:'1234567890'});
await addPerson({fullName:'Blair Sample',address1:'2 Fiction Lane',address2:'Testford',country:'United Kingdom',postcode:'XY9 9ZZ',identifierType:'NINO',identifierValue:'AB123456C'});

await page.click('#add-transfer');
await page.selectOption('[name=transferorPersonId]',{label:'Anne-Marie Example'});
await page.selectOption('[name=transfereePersonId]',{label:'Blair Sample'});
for(const [key,value] of Object.entries({numberShares:'10000',nominalValue:'1',shareClass:'Ordinary',disposalDate:'2026-06-30',acquisitionDate:'2020-01-02',wholeAcquisitionCost:'12345678',transferValue:'9999999',apportionedAcquisitionCost:'0',retainedDetails:'Remaining shares',retainedValue:'1'}))await page.fill(`[name=${key}]`,value);
await page.check('[name=partDisposal]');
await page.click('#transfer-form button[type=submit]');
await page.click('#review-button');
await page.check('#confirm-review');
await page.click('#to-exports');
const [download]=await Promise.all([page.waitForEvent('download'),page.click('#export-list button')]);
const output=path.join(process.env.TMPDIR||'/tmp','hs295-completed-part-disposal.pdf');
await download.saveAs(output);
await browser.close();

const document=await PDFDocument.load(fs.readFileSync(output));
if(document.getPageCount()!==2)throw Error('page count');
if(document.getForm().getFields().length)throw Error('form fields remain');
for(const pdfPage of document.getPages()){
  const annotations=pdfPage.node.get(PDFName.of('Annots'));
  if(annotations&&annotations.size()>0)throw Error('annotations remain');
}
if(document.catalog.get(PDFName.of('JavaScript'))||document.catalog.get(PDFName.of('Names')))throw Error('JavaScript/name tree remains');
const contents=document.context.lookup(document.getPages()[1].node.get(PDFName.of('Contents')));
let operators='';
for(let index=0;index<contents.size();index++)operators+=Buffer.from(decodePDFRawStream(document.context.lookup(contents.get(index))).decode()).toString('latin1');
const acquisitionCostOperators=['1 0 0 1 327 557 Tm\n<31> Tj','1 0 0 1 343 557 Tm\n<32> Tj','1 0 0 1 359 557 Tm\n<33> Tj','1 0 0 1 375 557 Tm\n<34> Tj','1 0 0 1 391 557 Tm\n<35> Tj','1 0 0 1 407 557 Tm\n<36> Tj','1 0 0 1 423 557 Tm\n<37> Tj','1 0 0 1 439 557 Tm\n<38> Tj'];
if(!acquisitionCostOperators.every(operator=>operators.includes(operator)))throw Error('Eight-digit acquisition cost was not aligned in the calibrated PDF boxes');
console.log(`Generated ${output} with a hyphenated name and eight-digit acquisition cost; PDF is flattened and structurally clean.`);
