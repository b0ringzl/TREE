'use strict';
// Two explicit scopes. Suggestions preserve the SST result and target crop.
const recognitionScope=()=>$('floraScope')?.value||'hong_kong';
const promptedScopeResults=new WeakSet();
let scopeDialogResult=null;
function closeScopeDialog(){if($('floraScopeDialog').open)$('floraScopeDialog').close();scopeDialogResult=null;}
function showWorldCandidate(hk){
  const alt=hk?.flora_alternative;
  if(!alt||state.busy||state.result!==hk)return;
  closeScopeDialog();
  // Reuse the same classified target; never blend scores or redetect a crop.
  state.result={...hk,...alt,flora_alternative:null,hk_original:hk,scope_advisory:null,
                selected_tree:hk.selected_tree,crop_preview:hk.crop_preview};
  $('floraScope').value='broad_plants';renderResult();update();
}
function showScopeDialog(d){
  if(state.result!==d||!d.flora_alternative)return;
  scopeDialogResult=d;promptedScopeResults.add(d);
  $('floraScopeDialogMessage').textContent=d.scope_advisory.message;
  if(!$('floraScopeDialog').open)$('floraScopeDialog').showModal();
}
$('keepSstResult').onclick=closeScopeDialog;
$('showWorldResult').onclick=()=>showWorldCandidate(scopeDialogResult);
$('floraScopeDialog').addEventListener('close',()=>{scopeDialogResult=null;});
const originalInvalidateForScope=invalidate;
invalidate=function(){closeScopeDialog();originalInvalidateForScope();};
const originalUpdateForScope=update;
update=function(){
  originalUpdateForScope();
  const selector=$('floraScope');if(!selector)return;
  const dual=source()==='image_point';selector.disabled=state.busy||dual;
  if(dual)selector.value='hong_kong';
  $('floraScopeNote').textContent=dual?'點雲融合保留原七類流程。':recognitionScope()==='broad_plants'
    ?'世界模型候選；分數不能與 SST 直接比較。':'香港範圍以 SST 識別；可能超出範圍時會提示。';
};
$('floraScope').onchange=()=>{invalidate();error('');update();};
const originalRenderForScope=renderResult;
renderResult=function(){
  originalRenderForScope();
  const d=state.result;if(!d)return;
  let panel=$('floraAdvisory');
  if(!panel){panel=el('div','flora-advisory');panel.id='floraAdvisory';panel.setAttribute('role','status');$('cards').before(panel);}
  panel.replaceChildren();panel.hidden=true;
  const advice=d.scope_advisory;
  const canPrompt=d.flora_alternative&&d.recognition_scope!=='broad_plants'&&
    (advice?.action==='suggest_broad'||advice?.action==='uncertain');
  if(advice?.message&&!canPrompt){panel.append(el('p','',advice.message));panel.hidden=false;}
  if(d.flora_alternative&&d.recognition_scope!=='broad_plants'){
    if(advice?.action==='suggest_broad'||advice?.action==='uncertain'){
      const button=el('button','secondary','查看世界模型候選');
      button.onclick=()=>showScopeDialog(d);
      panel.append(button);panel.hidden=false;
    }
  }
  if(d.recognition_scope==='broad_plants'){
    panel.hidden=false;panel.append(el('p','','世界模型 · '+Number(d.display_class_count).toLocaleString()+' 個植物目錄條目'));
    if(d.hk_original){const button=el('button','secondary','返回本次 SST 結果');button.onclick=()=>{state.result=d.hk_original;$('floraScope').value='hong_kong';renderResult();update();};panel.append(button);}
    $('evidenceText').textContent=d.preprocessing+' 同一目標，不使用點雲或跨模型分數融合。';
    $('resultMeta').textContent=d.model+' · '+d.image_model_version+' · '+d.width+' × '+d.height+' px · '+Number(d.inference_seconds||0).toFixed(2)+' s';
    $('cards').querySelectorAll('.score small').forEach(n=>n.textContent='MODEL SCORE / 模型分數');
  }
  // Once per completed result: dismissing or returning from the world model
  // must not immediately reopen the same prompt. A new recognition can prompt.
  if(canPrompt&&!promptedScopeResults.has(d))showScopeDialog(d);
};
fetch('/api/flora-scopes').then(r=>{if(!r.ok)throw Error();return r.json();}).then(d=>{
  const broad=d.scopes.find(s=>s.id==='broad_plants'),option=$('floraScope').querySelector('[value="broad_plants"]');
  option.disabled=!broad?.available;
  option.textContent=broad?.available?'世界模型（'+Number(broad.count).toLocaleString()+' 條目）':'世界模型 · 暫不可用';
}).catch(()=>{$('floraScope').querySelector('[value="broad_plants"]').disabled=true;});
update();
