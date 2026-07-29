import test from 'node:test';
import assert from 'node:assert/strict';
import { ExcelEditor } from '../src/editor/excel-editor.js';
import { createExcel, syncSteps } from '../src/state/excel-model.js';

const root = () => ({ innerHTML:'', querySelectorAll:() => [] });
const toolbar = () => ({ addEventListener() {} });

test('render escapes persisted identifiers before inserting markup', () => {
  const attack='id\"><img src=x onerror=alert(1)>';
  const excel=createExcel(),company=excel.companies[0];
  company.id=attack;excel.selectedCompanyId=attack;
  company.shareholders[0].id=attack;
  company.groups[0].id=attack;
  company.groups[0].classes[0].id=attack;
  const step={id:attack,title:'Unsafe identifier'};
  const document={steps:[step],excel:syncSteps(excel,[step]),element:root()};
  const editor=new ExcelEditor({root:document.element,toolbar:toolbar(),getDocument:()=>document,updateDocument:value=>document.excel=value,report() {}});

  editor.render();

  assert.doesNotMatch(document.element.innerHTML,/<img src=x/);
  assert.match(document.element.innerHTML,/&lt;img src=x onerror=alert\(1\)&gt;/);
});

test('Excel undo resynchronises an older snapshot with current steps', () => {
  const document={steps:[],excel:createExcel(),element:root()};
  const editor=new ExcelEditor({root:document.element,toolbar:toolbar(),getDocument:()=>document,updateDocument:value=>document.excel=value,report() {}});
  const step={id:'new-step',title:'New step'};
  document.steps=[step];
  document.excel=syncSteps(document.excel,document.steps);
  editor.synchronize();

  editor.undo();

  assert.deepEqual(document.excel.syncedSteps,[{id:'new-step',label:'New step'}]);
  assert.deepEqual(document.excel.companies[0].movements['new-step'],{});
});
