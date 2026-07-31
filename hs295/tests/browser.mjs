import {chromium} from 'playwright';
import path from 'node:path';

const browser=await chromium.launch({headless:true});
const page=await browser.newPage({acceptDownloads:true,viewport:{width:1440,height:1000}});
const errors=[],requests=[];
page.on('console',message=>{if(message.type()==='error')errors.push(message.text())});
page.on('request',request=>{if(/^https?:/.test(request.url()))requests.push(request.url())});

await page.goto('file://'+path.resolve('hs295_holdover_relief_2026.html'));
await page.locator('#company-name').evaluate(input=>input.value='X'.repeat(71));
await page.click('text=Create batch');
if(await page.locator('#notice').isHidden())throw Error('Expected validation error');
await page.fill('#company-name','Example Holdings Limited');
await page.click('text=Create batch');
if(!await page.locator('#notice').isHidden())throw Error('Stale validation error remained after successful correction');
if(!await page.locator('#dashboard').isVisible())throw Error('Dashboard did not open');

await page.click('#add-person');
if(await page.inputValue('[name=country]')!=='United Kingdom')throw Error('New person country was not defaulted');
await page.fill('[name=fullName]','Anne-Marie Example');
await page.fill('[name=address1]','1 Sample Road');
await page.fill('[name=address2]','Exampleton');
await page.fill('[name=country]','France');
await page.fill('[name=postcode]','SW1A 1AA');
await page.fill('[name=identifierValue]','1234567890');
await page.click('#person-form button[type=submit]');
if(!await page.getByText('Anne-Marie Example — UTR').isVisible())throw Error('Hyphenated name was not saved');

await page.getByRole('button',{name:'Edit'}).click();
if(await page.inputValue('[name=country]')!=='France')throw Error('Edited person country was not preserved');
await page.fill('[name=country]','Germany');
await page.getByRole('button',{name:'Cancel'}).click();
await page.getByRole('button',{name:'Edit'}).click();
if(await page.inputValue('[name=country]')!=='France')throw Error('Cancelled country change was saved');
await page.getByRole('button',{name:'Cancel'}).click();

await page.click('#add-couple');
if(await page.inputValue('[name=country]')!=='United Kingdom')throw Error('Couple country was not defaulted');
for(const [name,value] of Object.entries({husbandGivenNames:'Jean-Luc',wifeGivenNames:'Mary-Jane',surname:'Smith-Jones',address1:'2 Shared Road',address2:'London',postcode:'EC1A 1BB',husbandIdentifierValue:'1234567890'}))await page.fill(`[name=${name}]`,value);
await page.getByRole('button',{name:'Wife does not have a UTR'}).click();
await page.fill('[name=wifeIdentifierValue]','invalid');
await page.getByRole('button',{name:'Save husband and wife'}).click();
if(await page.locator('#people-list li').count()!==1)throw Error('Invalid couple was partially saved');
await page.fill('[name=wifeIdentifierValue]','AB123456C');
await page.getByRole('button',{name:'Save husband and wife'}).click();
if(await page.locator('#people-list li').count()!==3)throw Error('Valid couple did not create exactly two people');
if(!await page.getByText('Jean-Luc Smith-Jones — UTR').isVisible()||!await page.getByText('Mary-Jane Smith-Jones — NINO').isVisible())throw Error('Couple names or identifiers were not saved');
await page.click('#add-transfer');
const coupleIds=await page.locator('[name=transferorPersonId] option').evaluateAll(options=>options.filter(x=>/Smith-Jones/.test(x.textContent)).map(x=>x.value));
if(coupleIds.length!==2||coupleIds[0]===coupleIds[1])throw Error('Couple records do not have distinct IDs');
await page.getByRole('button',{name:'Cancel'}).click();

await page.locator('#people-list').evaluate(list=>{for(let i=0;i<40;i++){const item=document.createElement('li');item.textContent='Long directory test '+i;list.append(item)}});
await page.evaluate(()=>scrollTo(0,document.body.scrollHeight));
let toolbarBox=await page.locator('#creation-toolbar').boundingBox();
if(!toolbarBox||toolbarBox.y<0||toolbarBox.y+toolbarBox.height>1000)throw Error('Desktop creation toolbar was not sticky');
await page.setViewportSize({width:390,height:844});
await page.evaluate(()=>scrollTo(0,document.body.scrollHeight));
toolbarBox=await page.locator('#creation-toolbar').boundingBox();
if(!toolbarBox||toolbarBox.y<0||toolbarBox.y+toolbarBox.height>844)throw Error('Mobile creation toolbar was not sticky');
await page.click('#add-person');
if(!await page.locator('#person-form').isVisible())throw Error('Sticky toolbar was not clickable after scrolling');
await page.getByRole('button',{name:'Cancel'}).click();

await page.click('#add-person');
if(await page.inputValue('[name=country]')!=='United Kingdom')throw Error('Reopened new-person form did not restore the default');
await page.fill('[name=fullName]','A'.repeat(53));
await page.fill('[name=address1]','B'.repeat(59));
await page.fill('[name=address2]','London');
await page.fill('[name=postcode]','SW1A 1AA');
await page.fill('[name=identifierValue]','1234567890');
await page.click('#person-form button[type=submit]');
if(await page.locator('.invalid-field').count()!==2)throw Error('Expected overlong name and address to be highlighted');
if(await page.evaluate(()=>document.activeElement?.name)!=='fullName')throw Error('Expected first invalid field to retain focus');

await page.reload();
if(!await page.locator('#setup').isVisible())throw Error('Refresh did not clear memory');
if(errors.length)throw Error('Console errors: '+errors.join(';'));
if(requests.length)throw Error('Network requests: '+requests.join(';'));
await browser.close();
console.log('Chrome person-form smoke test passed, including country defaults, edits, cancellation, hyphenated names and offline operation.');
