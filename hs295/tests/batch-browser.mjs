import {chromium} from 'playwright';
import path from 'node:path';
const browser=await chromium.launch({headless:true});
const page=await browser.newPage({acceptDownloads:true,viewport:{width:1280,height:900}}),requests=[];
page.on('request',r=>{if(/^https?:/.test(r.url()))requests.push(r.url())});
await page.goto('file://'+path.resolve('hs295_holdover_relief_2026.html'));
await page.fill('#company-name','Example Holdings Limited');await page.click('text=Create batch');
async function person(name,utr){await page.click('#add-person');for(const [key,value] of Object.entries({fullName:name,address1:'1 Sample Road',address2:'London',country:'United Kingdom',postcode:'SW1A 1AA',identifierValue:utr}))await page.fill(`[name=${key}]`,value);await page.click('#person-form button[type=submit]')}
await person('Alex Example','1234567890');await person('Blair Example','1234567891');await person('Casey Example','1234567892');
await page.click('#add-transfer-batch');
if(await page.locator('.transfer-column').count()!==1)throw Error('Batch did not start with one column');
await page.click('text=Add transfer column');if(await page.locator('.transfer-column').count()!==2)throw Error('Column was not added');
const columns=page.locator('.transfer-column');
async function fillColumn(i,from,to,shares,cost,value){const c=columns.nth(i);await c.locator('[name=transferorPersonId]').selectOption({label:from});await c.locator('[name=transfereePersonId]').selectOption({label:to});for(const [key,v] of Object.entries({numberShares:shares,nominalValue:'1',shareClass:'Ordinary',disposalDate:'2026-06-30',acquisitionDate:'2020-01-02',wholeAcquisitionCost:cost,transferValue:value}))await c.locator(`[name=${key}]`).fill(v)}
await fillColumn(0,'Alex Example','Blair Example','3','10.50','40.50');await columns.nth(0).locator('[name=wholeAcquisitionCostPerShare]').fill('10.50');await columns.nth(0).locator('[name=transferValuePerShare]').fill('40.50');
await fillColumn(1,'Blair Example','Casey Example','2','5','25');
page.once('dialog',d=>d.dismiss());await columns.nth(0).getByRole('button',{name:'Copy number of shares to all'}).click();if(await columns.nth(1).locator('[name=numberShares]').inputValue()!=='2')throw Error('Cancelled overwrite changed destination');
page.once('dialog',d=>d.accept());await columns.nth(0).getByRole('button',{name:'Copy number of shares to all'}).click();if(await columns.nth(1).locator('[name=numberShares]').inputValue()!=='3')throw Error('Confirmed copy did not overwrite destination');
await columns.nth(1).locator('[name=transferValue]').fill('');await page.getByRole('button',{name:'Save all transfers',exact:true}).click();if(await page.locator('#transfer-list .card').count()!==0)throw Error('Invalid batch was partially saved');if(await page.locator('.batch-column-error').count()<1)throw Error('Invalid column was not identified');await columns.nth(1).locator('[name=transferValue]').fill('25');
await page.screenshot({path:'/tmp/hs295-batch-transfer.png',fullPage:true});await page.setViewportSize({width:390,height:844});const overflow=await page.locator('.batch-transfer-region').evaluate(x=>x.scrollWidth>x.clientWidth);if(!overflow)throw Error('Narrow batch editor did not retain horizontal overflow');await page.locator('.batch-transfer-region').focus();await page.keyboard.press('Tab');if(!await columns.nth(0).locator('select').first().evaluate(x=>x===document.activeElement))throw Error('Batch keyboard order did not enter first column');
await page.locator('#batch-transfer-form').evaluate(f=>f.requestSubmit());if(await page.locator('#transfer-list .card').count()!==2){throw Error('Valid batch was not saved atomically')};
page.once('dialog',d=>d.accept());const [session]=await Promise.all([page.waitForEvent('download'),page.locator('.export-session').first().click()]);const sessionPath=path.join('/tmp','hs295-batch-session.json');await session.saveAs(sessionPath);await page.reload();page.once('dialog',d=>d.accept());await page.setInputFiles('#import-file',sessionPath);await page.waitForTimeout(50);if(await page.locator('#transfer-list .card').count()!==2)throw Error('Batch session did not import');
await page.click('#review-button');await page.check('#confirm-review');await page.click('#to-exports');const [pdf]=await Promise.all([page.waitForEvent('download'),page.locator('#export-list button').first().click()]);if(!(await pdf.createReadStream()))throw Error('Batch PDF was not generated');
if(requests.length)throw Error('Network requests: '+requests.join(','));await browser.close();console.log('Batch columns, copying, atomic validation, modes, accessibility overflow, session round trip and PDF generation passed.');
