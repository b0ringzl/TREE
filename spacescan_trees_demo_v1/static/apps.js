'use strict';
(async()=>{
  const get=id=>document.getElementById(id);
  try{
    const response=await fetch('/api/release',{cache:'no-store'});
    if(!response.ok)throw Error();
    const info=await response.json();
    get('version').textContent=info.classifier_version;
    get('androidVersion').textContent=info.android_classifier_version;
    get('apkHash').textContent=info.android_apk_sha256;
    get('modelHash').textContent=info.android_model_sha256;
    get('live').textContent=info.ready?'Current SSTv1 is ready / 目前 SSTv1 已就緒':info.web_version_state==='mismatch'?'Version mismatch — do not compare yet / 版本不一致，請先完成核對':'Local model unavailable / 本地模型暫不可用';
  }catch{get('live').textContent='Version check unavailable / 暫時無法核對版本';}
})();
