'use strict';

const DROPBOX_MAX_BYTES = 20 * 1024 * 1024;
const DROPBOX_IMAGE_TYPES = Object.freeze({
  jpg: 'image/jpeg', jpeg: 'image/jpeg', png: 'image/png', webp: 'image/webp',
});

function dropboxImageMime(name) {
  if (typeof name !== 'string') return null;
  return DROPBOX_IMAGE_TYPES[name.split('.').pop().toLowerCase()] || null;
}

function safeDropboxDirectUrl(raw) {
  const url = new URL(raw);
  const host = url.hostname.toLowerCase();
  const trusted = host === 'dropbox.com' || host === 'www.dropbox.com' ||
    host.endsWith('.dropboxusercontent.com');
  if (url.protocol !== 'https:' || url.username || url.password || !trusted) {
    throw new Error('Unexpected Dropbox download link / Dropbox 下載連結無效');
  }
  return url.href;
}

function safeDropboxFileName(name) {
  return String(name || 'dropbox-image.jpg').replace(/[\\/\u0000-\u001f]/g, '_').slice(0, 180);
}

async function readDropboxImage(response, mime) {
  if (!response.ok) throw new Error('Dropbox download failed / Dropbox 下載失敗');
  if (response.url) safeDropboxDirectUrl(response.url);
  const declared = Number(response.headers.get('content-length'));
  if (Number.isFinite(declared) && declared > DROPBOX_MAX_BYTES) {
    throw new Error('Image exceeds 20 MB / 影像超過 20 MB');
  }
  if ((response.headers.get('content-type') || '').toLowerCase().includes('text/html')) {
    throw new Error('Dropbox returned a page, not an image / Dropbox 未返回影像');
  }
  if (!response.body) {
    const blob = await response.blob();
    if (blob.size > DROPBOX_MAX_BYTES) throw new Error('Image exceeds 20 MB / 影像超過 20 MB');
    return new Blob([blob], { type: mime });
  }
  const reader = response.body.getReader();
  const chunks = [];
  let total = 0;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    total += value.byteLength;
    if (total > DROPBOX_MAX_BYTES) {
      await reader.cancel();
      throw new Error('Image exceeds 20 MB / 影像超過 20 MB');
    }
    chunks.push(value);
  }
  return new Blob(chunks, { type: mime });
}

const dropboxButton = document.getElementById('importDropbox');
if (dropboxButton) {
  const status = document.getElementById('dropboxStatus');
  const appKey = (dropboxButton.dataset.appKey || '').trim();
  let importing = false;

  function message(value) {
    status.textContent = value;
    status.hidden = !value;
  }

  async function importSelection(files) {
    if (importing || state.busy) return;
    importing = true;
    dropboxButton.disabled = true;
    const available = Math.max(0, 4 - state.items.length);
    const selected = Array.isArray(files) ? files.slice(0, available) : [];
    let imported = 0;
    let failed = 0;
    try {
      if (!selected.length) {
        message('The Demo holds up to four images / 最多可加入四張影像');
        return;
      }
      for (const item of selected) {
        try {
          const mime = dropboxImageMime(item?.name);
          if (item?.isDir || !mime || !Number.isSafeInteger(item.bytes) ||
              item.bytes < 1 || item.bytes > DROPBOX_MAX_BYTES) {
            throw new Error('Choose JPG / PNG / WEBP under 20 MB / 請選擇不超過 20 MB 的 JPG／PNG／WEBP');
          }
          const downloadUrl = safeDropboxDirectUrl(item.link);
          message(`Importing ${safeDropboxFileName(item.name)}… / 正在匯入影像…`);
          // Chooser direct links are short-lived bearer links; consume immediately, never persist them.
          const response = await fetch(downloadUrl, {
            credentials: 'omit', cache: 'no-store', mode: 'cors', redirect: 'follow',
          });
          const blob = await readDropboxImage(response, mime);
          const countBefore = state.items.length;
          await addFiles([new File([blob], safeDropboxFileName(item.name), { type: mime })]);
          if (state.items.length === countBefore) {
            throw new Error('Image validation failed (20 MB / 40 MP limit) / 影像驗證未通過');
          }
          imported++;
        } catch (exception) {
          failed++;
          message(exception.message || 'Dropbox import failed / Dropbox 匯入失敗');
        }
      }
      if (imported) {
        message(`Imported ${imported} image${imported === 1 ? '' : 's'}${failed ? `; ${failed} failed` : ''}. / 已匯入 ${imported} 張${failed ? `；${failed} 張失敗` : ''}。`);
      }
    } finally {
      importing = false;
      dropboxButton.disabled = state.busy;
    }
  }

  dropboxButton.onclick = () => {
    if (state.busy || importing) return;
    if (!appKey) {
      message('Dropbox is not configured. Set SPACESCAN_DROPBOX_APP_KEY and register this site in the Dropbox App Console. / 尚未設定 Dropbox App Key。');
      return;
    }
    if (!window.Dropbox || typeof window.Dropbox.choose !== 'function') {
      message('Dropbox could not load. Check the internet connection or browser blocking. / Dropbox 未能載入，請檢查網絡或瀏覽器攔截設定。');
      return;
    }
    if (window.Dropbox.isBrowserSupported && !window.Dropbox.isBrowserSupported()) {
      message('This browser does not support Dropbox Chooser / 此瀏覽器不支援 Dropbox 選取器');
      return;
    }
    if (state.items.length >= 4) {
      message('The Demo holds up to four images / 最多可加入四張影像');
      return;
    }
    message('Choose an image in Dropbox / 請在 Dropbox 選擇影像');
    try {
      window.Dropbox.choose({
        linkType: 'direct',
        multiselect: true,
        extensions: ['.jpg', '.jpeg', '.png', '.webp'],
        sizeLimit: DROPBOX_MAX_BYTES,
        success: (files) => { void importSelection(files); },
        cancel: () => message(''),
      });
    } catch {
      message('Could not open Dropbox. Allow pop-ups for this site and retry. / 無法開啟 Dropbox，請允許此網站開啟彈出視窗後重試。');
    }
  };
}
