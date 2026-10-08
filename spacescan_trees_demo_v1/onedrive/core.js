export const MAX_IMAGE_BYTES = 20 * 1024 * 1024;
export const GRAPH_BASE = 'https://graph.microsoft.com/v1.0/';

const TYPES = Object.freeze({ jpg: 'image/jpeg', jpeg: 'image/jpeg', png: 'image/png', webp: 'image/webp' });

export function imageMime(item) {
  if (!item || item.folder || !item.file || typeof item.name !== 'string') return null;
  const extension = item.name.split('.').pop().toLowerCase();
  return TYPES[extension] || null;
}

export function imageIsImportable(item) {
  return !!imageMime(item) && Number.isFinite(item.size) && item.size > 0 && item.size <= MAX_IMAGE_BYTES;
}

export function safeGraphUrl(url) {
  const parsed = new URL(url, GRAPH_BASE);
  if (parsed.origin !== 'https://graph.microsoft.com' || !parsed.pathname.startsWith('/v1.0/')) {
    throw new Error('Unexpected Microsoft Graph URL / 非預期的 Graph 網址');
  }
  return parsed.href;
}

export function safeDownloadUrl(url) {
  const parsed = new URL(url);
  if (parsed.protocol !== 'https:' || parsed.username || parsed.password || !parsed.hostname) {
    throw new Error('Invalid OneDrive download URL / OneDrive 下載網址無效');
  }
  return parsed.href;
}

export function safeFileName(name) {
  return String(name || 'onedrive-image.jpg').replace(/[\\/\u0000-\u001f]/g, '_').slice(0, 180);
}

export async function readLimitedBlob(response, mime, limit = MAX_IMAGE_BYTES) {
  if (!response.ok) throw new Error('OneDrive download failed / OneDrive 下載失敗');
  const declared = Number(response.headers.get('content-length'));
  if (Number.isFinite(declared) && declared > limit) throw new Error('Image exceeds 20 MB / 影像超過 20 MB');
  if (!response.body) {
    const blob = await response.blob();
    if (blob.size > limit) throw new Error('Image exceeds 20 MB / 影像超過 20 MB');
    return new Blob([blob], { type: mime });
  }
  const reader = response.body.getReader();
  const chunks = [];
  let total = 0;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    total += value.byteLength;
    if (total > limit) {
      await reader.cancel();
      throw new Error('Image exceeds 20 MB / 影像超過 20 MB');
    }
    chunks.push(value);
  }
  return new Blob(chunks, { type: mime });
}
