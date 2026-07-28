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
