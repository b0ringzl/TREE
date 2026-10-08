'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const test = require('node:test');

const source = fs.readFileSync(path.join(__dirname, 'dropbox-import.js'), 'utf8');
const context = vm.createContext({
  URL,
  Blob,
  document: { getElementById: () => null },
});
vm.runInContext(source, context, { filename: 'dropbox-import.js' });

test('accepts only the three supported image formats', () => {
  assert.equal(context.dropboxImageMime('tree.JPG'), 'image/jpeg');
  assert.equal(context.dropboxImageMime('tree.png'), 'image/png');
  assert.equal(context.dropboxImageMime('tree.webp'), 'image/webp');
  assert.equal(context.dropboxImageMime('tree.svg'), null);
  assert.equal(context.dropboxImageMime(null), null);
});

test('accepts Dropbox HTTPS direct links but rejects foreign or unsafe URLs', () => {
  assert.equal(context.safeDropboxDirectUrl('https://dl.dropboxusercontent.com/s/a'),
    'https://dl.dropboxusercontent.com/s/a');
  assert.equal(context.safeDropboxDirectUrl('https://www.dropbox.com/s/a?dl=1'),
    'https://www.dropbox.com/s/a?dl=1');
  for (const url of ['http://dl.dropboxusercontent.com/s/a',
    'https://dl.dropboxusercontent.com.evil.test/s/a',
    'https://someone:password@www.dropbox.com/s/a']) {
    assert.throws(() => context.safeDropboxDirectUrl(url));
  }
});

test('rejects an oversized response and a returned HTML page', async () => {
  const response = (length, type) => ({
    ok: true,
    url: 'https://dl.dropboxusercontent.com/s/a',
    headers: { get: (name) => name === 'content-length' ? length : type },
  });
  await assert.rejects(context.readDropboxImage(response(String(21 * 1024 * 1024), 'image/jpeg'), 'image/jpeg'));
  await assert.rejects(context.readDropboxImage(response('100', 'text/html'), 'image/jpeg'));
});

test('chooser imports only the selected image into the existing upload flow', async () => {
  const button = { dataset: { appKey: 'testAppKey123' }, disabled: false };
  const status = { textContent: '', hidden: true };
  const state = { busy: false, items: [] };
  let chooserOptions;
  const sandbox = vm.createContext({
    URL, Blob, File,
    document: { getElementById: (id) => id === 'importDropbox' ? button : status },
    window: { Dropbox: {
      choose: (options) => { chooserOptions = options; },
      isBrowserSupported: () => true,
    } },
    state,
    fetch: async () => new Response(new Uint8Array([137, 80, 78, 71]), {
      status: 200, headers: { 'content-type': 'image/png' },
    }),
    addFiles: async (files) => { state.items.push(...files); },
  });
  vm.runInContext(source, sandbox, { filename: 'dropbox-import.js' });
  button.onclick();
  assert.equal(chooserOptions.linkType, 'direct');
  assert.deepEqual(Array.from(chooserOptions.extensions), ['.jpg', '.jpeg', '.png', '.webp']);
  chooserOptions.success([{ name: 'tree.png', bytes: 4,
    link: 'https://dl.dropboxusercontent.com/s/tree.png', isDir: false }]);
  await new Promise((resolve) => setTimeout(resolve, 20));
  assert.equal(state.items.length, 1);
  assert.equal(state.items[0].name, 'tree.png');
  assert.match(status.textContent, /Imported 1 image/);
});
