import test from 'node:test';
import assert from 'node:assert/strict';
import { ExcelEditor, gridDestination } from '../src/editor/excel-editor.js';
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

test('opening and movement cells render as directly editable', () => {
  const document={steps:[{id:'step-1',title:'Transfer'}],excel:syncSteps(createExcel(),[{id:'step-1',title:'Transfer'}]),element:root()};
  const editor=new ExcelEditor({root:document.element,toolbar:toolbar(),getDocument:()=>document,updateDocument:value=>document.excel=value,report() {}});

  editor.render();

  assert.match(document.element.innerHTML,/contenteditable="true"[^>]+data-step-id=""[^>]+data-editable="true"/);
  assert.match(document.element.innerHTML,/contenteditable="true"[^>]+data-step-id="step-1"[^>]+data-editable="true"/);
  assert.match(document.element.innerHTML,/data-class-label data-grid-row="0" data-grid-column="1"/);
  assert.match(document.element.innerHTML,/data-holder-name[^>]+data-grid-row="1" data-grid-column="0"/);
});

test('grid navigation includes shareholder and share class labels', () => {
  const target = (row, column) => ({dataset:{gridRow:String(row), gridColumn:String(column)}});
  const classLabel = target(0, 1);
  const holderLabel = target(1, 0);
  const firstValue = target(1, 1);
  const secondValue = target(1, 2);
  const targets = [classLabel, holderLabel, firstValue, secondValue];

  assert.equal(gridDestination(targets, classLabel, 'ArrowDown'), firstValue);
  assert.equal(gridDestination(targets, firstValue, 'ArrowLeft'), holderLabel);
  assert.equal(gridDestination(targets, holderLabel, 'Tab'), firstValue);
  assert.equal(gridDestination(targets, firstValue, 'Tab', true), holderLabel);
  assert.equal(gridDestination(targets, firstValue, 'ArrowRight'), secondValue);
});

test('focusing a zero-valued share cell clears it ready for typing', () => {
  const header = {dataset:{groupId:'group-1'}};
  const table = {querySelectorAll:() => [header]};
  const cell = {
    dataset:{holderId:'holder-1', classId:'class-1'},
    textContent:'0',
    closest:() => table
  };
  const matches = {
    '[data-grid-row][data-grid-column]':[cell],
    'td[data-editable=true]':[cell]
  };
  const element = {querySelectorAll:selector => matches[selector] || []};
  const document = {steps:[], excel:createExcel(), element};
  const editor = new ExcelEditor({root:element, toolbar:toolbar(), getDocument:() => document, updateDocument:value => document.excel=value, report() {}});

  editor.bindGrid();
  cell.onfocus();

  assert.equal(cell.textContent, '');
});

test('shareholder and header edits use stable row and column identities', () => {
  const excel=createExcel(),company=excel.companies[0],holder=company.shareholders[0],group=company.groups[0],shareClass=group.classes[0];
  const holderName={dataset:{holderId:holder.id},value:'Renamed holder'},holderType={dataset:{holderId:holder.id},value:'Trust'};
  const groupHeader={dataset:{groupId:group.id}},classHeader={dataset:{classId:shareClass.id}};
  const groupName={value:'Voting shares',closest:()=>groupHeader},classLabel={value:'A',closest:()=>classHeader};
  const matches={'[data-holder-name]':[holderName],'[data-holder-type]':[holderType],'[data-group-name]':[groupName],'[data-class-label]':[classLabel]};
  const element={innerHTML:'',querySelectorAll:selector=>matches[selector]||[]};
  const document={steps:[],excel,element};
  const editor=new ExcelEditor({root:element,toolbar:toolbar(),getDocument:()=>document,updateDocument:value=>document.excel=value,report() {}});
  editor.bindGrid();

  holderName.onchange();
  holderType.onchange();
  groupName.onchange();
  classLabel.onchange();

  const updated=document.excel.companies[0];
  assert.equal(updated.shareholders[0].name,'Renamed holder');
  assert.equal(updated.shareholders[0].type,'Trust');
  assert.equal(updated.groups[0].name,'Voting shares');
  assert.equal(updated.groups[0].classes[0].label,'A');
});
