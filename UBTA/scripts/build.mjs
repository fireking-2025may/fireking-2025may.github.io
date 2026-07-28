import fs from 'node:fs';
import path from 'node:path';

const root = path.resolve(import.meta.dirname, '..');
const source = relativePath => fs.readFileSync(path.join(root, relativePath), 'utf8');
const bundleModule = relativePath => source(relativePath)
  .replace(/^import .*;\n/gm, '')
  .replace(/export /g, '');

// Keep dependency order explicit: editor helpers use schema utilities, and main
// consumes every module listed here. The distribution must remain usable as a
// single file when opened directly from disk, without native module loading.
const application = [
  'src/state/schema.js',
  'src/state/history.js',
  'src/editor/dom-runs.js',
  'src/editor/interactions.js',
  'src/main.js',
].map(bundleModule).join('\n');

const css = ['app.css', 'document.css']
  .map(file => source(`src/styles/${file}`))
  .join('\n');
const vendor = source('src/vendor/paged.js');
const body = source('src/index.html')
  .match(/<body>([\s\S]*)<\/body>/)[1]
  .replace(/<script[\s\S]*?<\/script>/g, '')
  .trim();
const html = `<!doctype html><html lang="en-GB"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>UBTA Steps Plan Editor</title><style>${css}</style></head><body>${body}<script>${vendor}\n${application}<\/script></body></html>`;

fs.mkdirSync(path.join(root, 'dist'), { recursive: true });
fs.writeFileSync(path.join(root, 'dist/index.html'), html);
console.log(`Built dist/index.html (${html.length} bytes)`);
