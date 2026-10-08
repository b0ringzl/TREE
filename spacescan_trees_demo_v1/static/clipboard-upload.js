'use strict';
// Read only a user-initiated paste event; never poll or request clipboard access.
document.addEventListener('paste', async event => {
  const target = event.target;
  if (event.defaultPrevented || state.busy || document.querySelector('dialog[open]') ||
      target?.isContentEditable || target?.closest?.('input, textarea, select, [contenteditable="true"]')) return;
  const clipboard = event.clipboardData;
  if (!clipboard) return;
  // Some browsers expose the same image in both items and files. Use only one.
  let files = Array.from(clipboard.items || [])
    .filter(item => item.kind === 'file' && item.type.startsWith('image/'))
    .map(item => item.getAsFile()).filter(Boolean);
  if (!files.length) files = Array.from(clipboard.files || []).filter(file => file.type.startsWith('image/'));
  if (!files.length) return; // Leave text and copied links alone; never fetch pasted URLs.
  event.preventDefault();
  // Keep original bytes and reuse upload validation, previews and tree selection.
  await addFiles(files);
});
