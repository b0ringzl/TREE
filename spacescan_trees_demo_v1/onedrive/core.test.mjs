import test from 'node:test';
import assert from 'node:assert/strict';
import {
  MAX_IMAGE_BYTES, imageMime, imageIsImportable, readLimitedBlob,
  safeDownloadUrl, safeFileName, safeGraphUrl,
} from './core.js';

test('only image files within the upload limit are importable', () => {
  assert.equal(imageMime({ name: 'tree.JPG', file: {} }), 'image/jpeg');
  assert.equal(imageMime({ name: 'notes.jpg', folder: {}, file: {} }), null);
  assert.equal(imageMime({ name: 'tree.svg', file: {} }), null);
  assert.equal(imageIsImportable({ name: 'tree.png', file: {}, size: MAX_IMAGE_BYTES }), true);
  assert.equal(imageIsImportable({ name: 'tree.png', file: {}, size: MAX_IMAGE_BYTES + 1 }), false);
  assert.equal(imageIsImportable({ name: 'tree.png', file: {}, size: 0 }), false);
});

test('Graph paging cannot send a bearer token to another origin', () => {
  assert.equal(safeGraphUrl('me/drive/root/children'), 'https://graph.microsoft.com/v1.0/me/drive/root/children');
  assert.throws(() => safeGraphUrl('https://evil.example/v1.0/me/drive'), /Unexpected/);
  assert.throws(() => safeGraphUrl('https://graph.microsoft.com.evil.example/v1.0/me/drive'), /Unexpected/);
  assert.throws(() => safeGraphUrl('https://graph.microsoft.com/beta/me/drive'), /Unexpected/);
});

test('download URLs require HTTPS and filenames cannot contain paths', () => {
  assert.equal(safeDownloadUrl('https://example.files.1drv.com/file'), 'https://example.files.1drv.com/file');
  assert.throws(() => safeDownloadUrl('http://example.files.1drv.com/file'), /Invalid/);
  assert.equal(safeFileName('folder\\tree.jpg'), 'folder_tree.jpg');
  assert.equal(safeFileName('../tree.png'), '.._tree.png');
});

test('download stops when bytes exceed the image limit', async () => {
  const small = new Response(new Uint8Array([1, 2, 3]));
  assert.equal((await readLimitedBlob(small, 'image/png', 4)).size, 3);
  const large = new Response(new Uint8Array([1, 2, 3, 4, 5]));
  await assert.rejects(readLimitedBlob(large, 'image/png', 4), /exceeds/);
});
