import { PublicClientApplication, InteractionRequiredAuthError } from '@azure/msal-browser';
import {
  GRAPH_BASE, imageMime, imageIsImportable, readLimitedBlob,
  safeDownloadUrl, safeFileName, safeGraphUrl,
} from './core.js';

const byId = (id) => document.getElementById(id);
const launch = byId('importOneDrive');

if (launch) {
  const dialog = byId('onedriveDialog');
  const status = byId('onedriveStatus');
  const list = byId('onedriveList');
  const path = byId('onedrivePath');
  const back = byId('onedriveBack');
  const more = byId('onedriveMore');
  const clientId = (launch.dataset.clientId || '').trim();
  const redirectUri = new URL('/static/onedrive-redirect.html', window.location.origin).href;
  const scopes = ['Files.Read'];
  let auth;
  let accessToken;
  let trail = [{ id: null, name: 'OneDrive' }];
  let nextPage = null;
  let working = false;

  function message(value) { status.textContent = value; }
  function setWorking(value) {
    working = value;
    back.disabled = value || trail.length < 2;
    more.disabled = value;
    list.querySelectorAll('button').forEach((button) => { button.disabled = value || button.dataset.importable === 'false'; });
  }

  async function authorize() {
    if (!/^[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}$/i.test(clientId)) {
      throw new Error('OneDrive has not been configured. Set SPACESCAN_ONEDRIVE_CLIENT_ID and register the redirect URI shown below. / 尚未設定 OneDrive 用戶端 ID。');
    }
    if (!auth) {
      auth = new PublicClientApplication({
        auth: { clientId, authority: 'https://login.microsoftonline.com/common', redirectUri },
        cache: { cacheLocation: 'memoryStorage' },
      });
      await auth.initialize();
    }
    let account = auth.getActiveAccount() || auth.getAllAccounts()[0];
    if (!account) {
      const signedIn = await auth.loginPopup({ scopes, redirectUri });
      account = signedIn.account;
      accessToken = signedIn.accessToken;
    }
    if (!account) throw new Error('Microsoft sign-in did not return an account / Microsoft 登入未返回帳戶');
    auth.setActiveAccount(account);
    if (!accessToken) {
      try {
        accessToken = (await auth.acquireTokenSilent({ scopes, account })).accessToken;
      } catch (error) {
        if (!(error instanceof InteractionRequiredAuthError)) throw error;
        accessToken = (await auth.acquireTokenPopup({ scopes, account, redirectUri })).accessToken;
      }
    }
    if (!accessToken) throw new Error('Files.Read authorization was not granted / 未取得唯讀授權');
  }

  async function graph(pathOrUrl) {
    const url = safeGraphUrl(pathOrUrl);
    const response = await fetch(url, {
      headers: { Authorization: `Bearer ${accessToken}`, Accept: 'application/json' },
      cache: 'no-store',
    });
    if (!response.ok) {
      if (response.status === 401) accessToken = null;
      throw new Error(`Microsoft Graph returned ${response.status}. Reopen OneDrive to sign in again. / 請重新開啟 OneDrive 登入。`);
    }
    return response.json();
  }

  function renderPath() {
    path.textContent = trail.map((part) => part.name).join(' / ');
    back.disabled = working || trail.length < 2;
  }

  function addRows(items) {
    let shown = 0;
    for (const item of items) {
      if (!item?.id || (!item.folder && !imageMime(item))) continue;
      const isFolder = !!item.folder;
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'onedrive-row';
      button.textContent = `${isFolder ? '📁' : '🖼'} ${item.name}${!isFolder ? ` · ${(item.size / 1048576).toFixed(1)} MB` : ''}`;
      button.dataset.importable = String(isFolder || imageIsImportable(item));
      button.disabled = working || button.dataset.importable === 'false';
      if (!isFolder && !imageIsImportable(item)) button.title = 'JPG / PNG / WEBP, 20 MB maximum / 只接受不超過 20 MB 的影像';
      button.onclick = () => { if (isFolder) openFolder(item); else importImage(item); };
      list.append(button);
      shown++;
    }
    if (!shown && !list.children.length) {
      const empty = document.createElement('p');
      empty.textContent = 'No image files in this folder / 此資料夾沒有可匯入的影像';
      list.append(empty);
    }
  }

  async function loadPage(url, replace = false) {
    setWorking(true);
    message('Loading OneDrive files… / 正在載入 OneDrive 檔案…');
    if (replace) {
      list.replaceChildren();
      nextPage = null;
      more.hidden = true;
    }
    try {
      const data = await graph(url);
      addRows(Array.isArray(data.value) ? data.value : []);
      nextPage = data['@odata.nextLink'] ? safeGraphUrl(data['@odata.nextLink']) : null;
      more.hidden = !nextPage;
      message('Choose an image to import. Files remain in your OneDrive. / 選取影像後匯入；雲端原檔不會移動。');
    } catch (error) {
      message(error.message || 'OneDrive could not be loaded / 無法載入 OneDrive');
    } finally {
      setWorking(false);
    }
  }

  function folderUrl(id) {
    const resource = id ? `me/drive/items/${encodeURIComponent(id)}/children` : 'me/drive/root/children';
    return `${GRAPH_BASE}${resource}?$select=id,name,size,file,folder&$top=100`;
  }

  function openFolder(item) {
    trail.push({ id: item.id, name: item.name });
    renderPath();
    loadPage(folderUrl(item.id), true);
  }

  async function importImage(item) {
    if (working || state.busy || state.items.length >= 4) {
      message('Up to four images can be loaded, and imports pause during recognition. / 最多四張；識別中不可匯入。');
      return;
    }
    setWorking(true);
    message(`Importing ${item.name}… / 正在匯入影像…`);
    try {
      const meta = await graph(`${GRAPH_BASE}me/drive/items/${encodeURIComponent(item.id)}?$select=id,name,size,file,@microsoft.graph.downloadUrl`);
      if (!imageIsImportable(meta)) throw new Error('This file is not a supported image under 20 MB / 檔案格式或大小不符');
      const downloadUrl = safeDownloadUrl(meta['@microsoft.graph.downloadUrl']);
      // The preauthenticated URL is used immediately and never shown, stored, or sent to the Demo server.
      const response = await fetch(downloadUrl, { credentials: 'omit', cache: 'no-store' });
      const blob = await readLimitedBlob(response, imageMime(meta));
      const countBefore = state.items.length;
      await addFiles([new File([blob], safeFileName(meta.name), { type: imageMime(meta) })]);
      if (state.items.length === countBefore) throw new Error('Image validation failed (20 MB / 40 MP limit) / 影像驗證未通過');
      dialog.close();
    } catch (error) {
      message(error.message || 'OneDrive import failed / OneDrive 匯入失敗');
    } finally {
      setWorking(false);
    }
  }

  launch.onclick = async () => {
    if (state.busy) return;
    if (!dialog.open) dialog.showModal();
    byId('onedriveRedirectUri').textContent = redirectUri;
    if (working) return;
    setWorking(true);
    message('Connecting to Microsoft… / 正在連接 Microsoft…');
    try {
      await authorize();
      trail = [{ id: null, name: 'OneDrive' }];
      renderPath();
      await loadPage(folderUrl(null), true);
    } catch (error) {
      message(error.message || 'OneDrive sign-in failed / OneDrive 登入失敗');
    } finally {
      setWorking(false);
    }
  };
  back.onclick = () => {
    if (working || trail.length < 2) return;
    trail.pop();
    renderPath();
    loadPage(folderUrl(trail.at(-1).id), true);
  };
  more.onclick = () => { if (!working && nextPage) loadPage(nextPage); };
  byId('closeOneDrive').onclick = () => dialog.close();
  dialog.addEventListener('close', () => { message(''); });
}
