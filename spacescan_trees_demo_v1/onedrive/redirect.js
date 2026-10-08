import { broadcastResponseToMainFrame } from '@azure/msal-browser/redirect-bridge';

broadcastResponseToMainFrame().catch(() => {
  document.body.textContent = 'OneDrive sign-in could not return to SpaceScan Trees. Close this window and retry. / 登入未能返回應用，請關閉視窗後重試。';
});
