import fs from 'node:fs';
const templatePath=process.env.HS295_TEMPLATE;
if(!templatePath) throw new Error('Set HS295_TEMPLATE to the local path of the supplied HS295 Claim form 2026 PDF. The binary template is intentionally not stored in this repository.');
let h=fs.readFileSync('hs295/src/shell.html','utf8');
const put=(k,v)=>h=h.replace(k,v.replace(/^\/\/# sourceMappingURL=.*$/gm,'').replace(/^\/\*# sourceMappingURL=.*?\*\/$/gm,''));
put('/*__CSS__*/',fs.readFileSync('hs295/src/style.css','utf8'));put('/*__PDFLIB__*/',fs.readFileSync('node_modules/pdf-lib/dist/pdf-lib.min.js','utf8'));put('/*__JSZIP__*/',fs.readFileSync('node_modules/jszip/dist/jszip.min.js','utf8'));put('/*__CORE__*/',fs.readFileSync('hs295/src/core.js','utf8'));put('/*__COORDS__*/',fs.readFileSync('hs295/src/coordinates.js','utf8'));put('/*__APP__*/',fs.readFileSync('hs295/src/app.js','utf8'));put('/*__PDF__*/',fs.readFileSync(templatePath).toString('base64'));fs.writeFileSync('hs295_holdover_relief_2026.html',h);console.log(h.length+' bytes');
