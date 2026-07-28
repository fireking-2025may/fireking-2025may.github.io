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
  assert.match(html, /\.step-title-print\{visibility:hidden/);
});
