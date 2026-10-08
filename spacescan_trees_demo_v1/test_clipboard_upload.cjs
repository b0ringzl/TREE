const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const code = fs.readFileSync(path.join(__dirname, 'static/clipboard-upload.js'), 'utf8');
async function run(options = {}) {
  let handler, prevented = false;
  const calls = [];
  vm.runInNewContext(code, {
    document: {addEventListener: (type, fn) => {assert.equal(type, 'paste'); handler = fn;},
      querySelector: () => options.dialog || null},
    state: {busy: options.busy || false},
    addFiles: async files => calls.push(files),
  });
  await handler({target: options.target || {}, defaultPrevented: options.prevented || false,
    clipboardData: options.clipboard, preventDefault: () => {prevented = true;}});
  return {calls, prevented};
}
(async () => {
  const png = {type: 'image/png', name: 'image.png'};
  const item = {kind: 'file', type: png.type, getAsFile: () => png};
  const clipboard = {items: [item], files: [png]};
  let result = await run({clipboard});
  assert.equal(result.calls.length, 1);
  assert.equal(result.calls[0].length, 1); // No duplicate from files fallback.
  assert.equal(result.calls[0][0], png); // Original file, no re-encoding.
  assert.equal(result.prevented, true);
  result = await run({clipboard: {files: [png]}});
  assert.equal(result.calls[0][0], png);
  for (const extra of [{busy: true}, {dialog: {}}, {prevented: true},
    {target: {isContentEditable: true}}, {target: {closest: () => ({})}}]) {
    result = await run({clipboard, ...extra});
    assert.equal(result.calls.length, 0);
    assert.equal(result.prevented, false);
  }
  for (const data of [null, {items: [{kind:'string', type:'text/plain'}]},
    {items:[{kind:'file', type:'application/pdf'}]},
    {items:[{kind:'file', type:'image/png', getAsFile:()=>null}]}]) {
    result = await run({clipboard: data});
    assert.equal(result.calls.length, 0);
    assert.equal(result.prevented, false);
  }
  result = await run({clipboard:{items: [item, item]}});
  assert.equal(result.calls[0].length, 2); // Shared uploader enforces four-image cap.
  console.log('Clipboard upload: 12 cases passed.');
})().catch(error => {console.error(error); process.exitCode = 1;});
