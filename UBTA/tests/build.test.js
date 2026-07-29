import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import vm from 'node:vm';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

test('production bundle includes the blank-space insertion helper before its caller', () => {
  execFileSync(process.execPath, ['scripts/build.mjs'], { cwd: root });
  const html = fs.readFileSync(path.join(root, 'dist/index.html'), 'utf8');
  const definition = html.indexOf('function insertionContextFromPoint(');
  const caller = html.indexOf('insertionContextFromPoint(group.id');

  assert.notEqual(definition, -1, 'insertionContextFromPoint must be bundled');
  assert.ok(caller > definition, 'the helper must be defined before blankStepClick uses it');
});

test('production bundle includes the editable-link click helper before its caller', () => {
  execFileSync(process.execPath, ['scripts/build.mjs'], { cwd: root });
  const html = fs.readFileSync(path.join(root, 'dist/index.html'), 'utf8');
  const definition = html.indexOf('function handleEditableLinkClick(');
  const caller = html.indexOf('handleEditableLinkClick(event,openEditorLink)');

  assert.notEqual(definition, -1, 'handleEditableLinkClick must be bundled');
  assert.ok(caller > definition, 'the helper must be defined before editableLinkClick uses it');
});

test('production bundle defines navigation history before creating it', () => {
  execFileSync(process.execPath, ['scripts/build.mjs'], { cwd: root });
  const html = fs.readFileSync(path.join(root, 'dist/index.html'), 'utf8');
  const definition = html.indexOf('class NavigationHistory');
  const construction = html.indexOf('new NavigationHistory()');

  assert.notEqual(definition, -1, 'NavigationHistory must be bundled');
  assert.ok(construction > definition, 'NavigationHistory must be defined before it is constructed');
});

test('production bundle defines Excel normalisation before document initialisation', () => {
  execFileSync(process.execPath, ['scripts/build.mjs'], { cwd: root });
  const html = fs.readFileSync(path.join(root, 'dist/index.html'), 'utf8');
  const definition = html.indexOf('function normaliseExcel(');
  const documentNormalisation = html.indexOf('excel:normaliseExcel(');

  assert.notEqual(definition, -1, 'normaliseExcel must be bundled');
  assert.ok(documentNormalisation > definition, 'normaliseExcel must be defined before normaliseDocument uses it');
});

test('production bundle defines ExcelEditor before creating it', () => {
  execFileSync(process.execPath, ['scripts/build.mjs'], { cwd: root });
  const html = fs.readFileSync(path.join(root, 'dist/index.html'), 'utf8');
  const definition = html.indexOf('class ExcelEditor');
  const construction = html.indexOf('new ExcelEditor(');

  assert.notEqual(definition, -1, 'ExcelEditor must be bundled');
  assert.ok(construction > definition, 'ExcelEditor must be defined before it is constructed');
});

test('Excel Shares keeps the main toolbar and hides the Steps Plan preview', () => {
  const source = fs.readFileSync(path.join(root, 'src/main.js'), 'utf8');
  const css = fs.readFileSync(path.join(root, 'src/styles/app.css'), 'utf8');

  assert.match(source, /\$\('#preview'\)\.hidden=excel/);
  assert.doesNotMatch(source, /\$\('#steps-toolbar'\)\.hidden=excel/);
  assert.match(css, /#preview\[hidden\]\{display:none!important\}/);
});

test('production bundle contains valid JavaScript and resolves table re-exports', () => {
  execFileSync(process.execPath, ['scripts/build.mjs'], { cwd: root });
  const html = fs.readFileSync(path.join(root, 'dist/index.html'), 'utf8');
  const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)];
  const source = scripts.at(-1)?.[1];

  assert.ok(source, 'the production bundle must contain an inline script');
  assert.doesNotMatch(source, /\}\s+from\s+['"]\.\/table-values\.js['"]/, 're-export syntax must not be partially stripped');
  assert.ok(source.indexOf('function parseTableNumber(') < source.indexOf('parseTableNumber(value)'), 'table helpers must be bundled before their callers');
  assert.doesNotThrow(() => new vm.Script(source), 'the production bundle must parse as classic JavaScript');
});

test('standalone template unlock has embedded ciphertext and performs no runtime fetch', () => {
  execFileSync(process.execPath, ['scripts/build.mjs'], { cwd: root });
  const html = fs.readFileSync(path.join(root, 'dist/index.html'), 'utf8');
  assert.match(html, /const defaultTemplateEnvelope=/);
  assert.match(html, /ciphertext:'VgziFV2P5dlF/);
  assert.doesNotMatch(html, /fetch\(['"]\.\.\/\.\.\/encrypted-files/);
});

test('standalone build contains the complete print feature', () => {
  execFileSync(process.execPath, ['scripts/build.mjs'], { cwd: root });
  const html = fs.readFileSync(path.join(root, 'dist/index.html'), 'utf8');
  assert.match(html, /id="print-document"/);
  assert.match(html, /new PrintLifecycle\(/);
  assert.match(html, /window\.print\(\)/);
  assert.match(html, /Print preparation failed:/);
  assert.match(html, /@media print/);
  assert.match(html, /width:297mm!important;height:210mm!important/);
  assert.match(html, /\.pagedjs_page:last-child\{break-after:auto/);
  assert.match(html, /\.step-title-print\{display:none/);
});

test('heading titles use exactly one normal-flow representation in screen and print', () => {
  const css = fs.readFileSync(path.join(root, 'src/styles/document.css'), 'utf8');
  assert.match(css, /\.step-heading\{display:block\}/);
  assert.match(css, /\.step-title-print\{display:none;/);
  assert.match(css, /\.step-heading>\.step-title\{display:block;position:static\}/);
  assert.doesNotMatch(css, /\.step-heading>\.step-title\{[^}]*position:absolute/);
  assert.doesNotMatch(css, /\.step-heading>\.step-title\{[^}]*bottom:/);
  assert.match(css, /@media print\{[\s\S]*\.step-title\{display:none!important\}[\s\S]*\.step-title-print\{display:block\}/);
  assert.match(css, /white-space:normal;overflow-wrap:anywhere/);
  assert.match(css, /\.sheet-source:is\(\.step,\.appendix\)\.is-selected \.step-heading\{border-left:0;padding-left:0\}/);
});

test('A4 print wrappers override responsive preview scaling consistently', () => {
  const css = fs.readFileSync(path.join(root, 'src/styles/app.css'), 'utf8');
  const print = css.slice(css.indexOf('@media print'));
  assert.match(print, /@page\{size:A4 landscape;margin:0\}/);
  assert.match(print, /html,body\{width:297mm!important;min-width:297mm!important[^}]*zoom:1!important;transform:none!important/);
  assert.match(print, /#preview \.pagedjs_pagebox,#preview \.sheet-source\{width:297mm!important;height:210mm!important/);
  assert.match(print, /\.pagedjs_pages\{display:block!important;width:297mm!important[^}]*zoom:1!important;transform:none!important/);
  assert.match(print, /#preview \.pagedjs_page:last-child\{break-after:auto;page-break-after:auto\}/);
});

test('source and standalone toolbars omit global insertion controls but retain contextual insertion', () => {
  const source = fs.readFileSync(path.join(root, 'src/index.html'), 'utf8');
  execFileSync(process.execPath, ['scripts/build.mjs'], { cwd: root });
  const built = fs.readFileSync(path.join(root, 'dist/index.html'), 'utf8');

  for (const [name, html] of [['source', source], ['standalone build', built]]) {
    assert.doesNotMatch(html, /class="[^"]*insert-group/, `${name} must not render the Insert toolbar group`);
    for (const command of ['addParagraph', 'addHeading', 'addBulletList', 'addNumberList', 'addTable', 'addImage']) {
      assert.doesNotMatch(html, new RegExp(`<button[^>]+data-command="${command}"[^>]*data-requires-group`), `${name} must not render the ${command} toolbar button`);
    }
    assert.match(html, /<dialog id="insertion-chooser"/, `${name} must retain the contextual insertion chooser`);
    for (const group of ['document', 'text', 'step', 'appendix', 'table']) {
      assert.match(html, new RegExp(`class="toolbar-group ${group}-group"`), `${name} must render the ${group} toolbar group`);
    }
    assert.match(html, /<nav class="nav" aria-label="Page navigation">/, `${name} must render page navigation`);
    assert.doesNotMatch(html, /class="toolbar-group history-group"/, `${name} must place history actions in Document`);
    assert.doesNotMatch(html, /id="page-select"/, `${name} must omit the visible Go to selector`);
  }
});
