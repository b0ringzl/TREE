'use strict';
// No fixture images, hard-coded boxes, or simulated scores in the live flow.
let pickerResolve=null, pickedTree=null;
$('projection').onchange=()=>{
  const item=state.items[state.active];if(item)item.projection=$('projection').value;
  invalidate();error('');update();
};
$('testPanorama').onclick=async()=>{
  if(state.busy)return;
  try{const r=await fetch('/api/test-panorama');if(!r.ok)throw Error();
    const count=state.items.length;
    await addFiles([new File([await r.blob()],'panorama-test.jpg',{type:'image/jpeg'})]);
    if(state.items.length===count+1)state.items[state.active].projection='equirectangular';
    update();$('about').close();
  }catch{error('Panorama unavailable / 全景測試圖片暫不可用');}
};
function finishPicking(value){
  const resolve=pickerResolve;pickerResolve=null;pickedTree=null;
  $('treePicker').close();$('treeOptions').replaceChildren();
  if(resolve)resolve(value);
}
$('cancelTree').onclick=()=>finishPicking(null);
$('treePicker').addEventListener('cancel',e=>{e.preventDefault();finishPicking(null);});
$('confirmTree').onclick=()=>{if(pickedTree)finishPicking(pickedTree);};
function chooseTree(d){
  $('treeOptions').replaceChildren();pickedTree=null;$('confirmTree').disabled=true;
  $('treeChoiceNote').textContent=d.candidates.length>1
    ?'Choose the crop containing your tree. / 請選擇包含目標樹木的切片。'
    :'No clear central tree. Is this your target? / 未找到明確的居中樹木，這是您的目標嗎？';
  if(d.additional_candidates)$('treeChoiceNote').textContent+=' Showing 12 candidates; use a closer view for others. / 顯示 12 個候選；其他目標請提供較近視角。';
  d.candidates.forEach((row,i)=>{
    const button=el('button','tree-choice'),img=el('img');img.src=row.thumbnail;
    img.alt='Tree '+row.id+' / 第 '+row.id+' 株';
    button.style.setProperty('--delay',Math.min(i,5)*65+'ms');
    button.setAttribute('aria-pressed','false');
    const label=el('span','tree-choice-label','Tree '+row.id+' / 第 '+row.id+' 株');
    label.append(el('small','', (row.score*100).toFixed(1)+'%'));
    button.append(img,label);
    button.onclick=()=>{pickedTree=row;$('confirmTree').disabled=false;
      $('treeOptions').querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));};
    $('treeOptions').append(button);
  });
  $('treePicker').showModal();
  return new Promise(resolve=>{pickerResolve=resolve;});
}
async function imageRequest(url,item,extra={}){
  const body=new FormData();body.append('image',item.file);body.append('scenario',scenario());body.append('source','image');body.append('projection',item.projection||'photo');
  body.append('recognition_scope',typeof recognitionScope==='function'?recognitionScope():'hong_kong');
  Object.entries(extra).forEach(([k,v])=>body.append(k,v));
  const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),180000);
  try{
    const r=await fetch(url,{method:'POST',body,signal:controller.signal}),d=await readApiResponse(r);
    if(!r.ok)throw new Error(typeof d.detail==='string'?d.detail:'Request failed / 請求失敗');
    return d;
  }finally{clearTimeout(timer);}
}
async function exportTarget(item,target,button){
  if(state.busy)return;
  button.disabled=true;
  const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),30000);
  try{
    const body=new FormData();body.append('image',item.file);body.append('token',target.token);
    const response=await fetch('/api/export-target',{method:'POST',body,signal:controller.signal});
    if(!response.ok){const detail=await readApiResponse(response);throw Error(typeof detail.detail==='string'?detail.detail:'Crop export failed / 切片匯出失敗');}
    const blob=await response.blob(),url=URL.createObjectURL(blob),a=el('a');
    a.href=url;a.download='SpaceScan-selected-target.jpg';a.click();
    setTimeout(()=>URL.revokeObjectURL(url),1000);
  }catch(e){error(e.name==='AbortError'?'Crop export timed out / 切片匯出逾時':e.message);}
  finally{clearTimeout(timer);button.disabled=false;}
}
function renderSamComparison(item,comparison){
  const panel=el('section','sam-comparison');
  panel.append(el('h3','','Before / after SAM selection / SAM 選樹前後對照'));
  if(!comparison?.available){panel.append(el('p','',comparison?.reason||'Original-photo comparison unavailable; the tight crop result remains unchanged. / 原圖對照暫不可用，保留目前的緊裁片結果。'));$('quality').prepend(panel);return;}
  const grid=el('div','sam-comparison-grid');
  for(const [label,url,row] of [
    ['Original photo / 原圖',item.polygonOriginal.url,comparison.original_top1],
    ['SAM tight crop / SAM 緊裁片',item.url,comparison.crop_top1]
  ]){
    const card=el('div','sam-comparison-card'),image=el('img');image.src=url;image.alt=label;
    card.append(el('b','',label),image,el('p','',row.scientific_name+' / '+row.chinese_name+' · '+(row.score*100).toFixed(1)+'%'));
    grid.append(card);
  }
  panel.append(grid);
  if(!comparison.same_top1)panel.append(el('p','sam-disagreement','The two views rank different candidates first; review the target and diagnostic details. / 兩種取景的首位候選不同，請核對目標及辨識細節。'));
  panel.append(el('p','sam-comparison-note','The original photo may contain other trees. These are separate predictions, not accuracy estimates or an automatic higher-score selection. / 原圖可能包含其他樹木；這是兩次獨立預測，並非準確率估計，也不會自動選取較高分。'));
  $('quality').prepend(panel);
}
$('run').onclick=async()=>{
  const item=state.items[state.active];if(!item||state.busy||source()!=='image')return;
  const forcePicker=!!state.forcePicker;state.forcePicker=false;
  const id=++state.requestId;state.busy=true;invalidate();error('');update();
  try{
    let detection=null,target=null,fallback=null;
    // Explicit organ close-ups are not whole-tree detector inputs.
    const isPano=item.projection==='equirectangular'||((item.projection||'auto')==='auto'&&item.width/item.height>=1.96&&item.width/item.height<=2.04);
    if(!item.manualPolygon&&(isPano||['whole','crown'].includes(item.organ))){
      $('run').textContent=isPano?'Correcting panorama… / 全景透視校正…':'Finding trees… / 正在尋找樹木…';
      detection=await imageRequest('/api/detect-trees',item);
      if(id!==state.requestId)return;
      if(detection.action==='no_target'){
        if(isPano||detection.projection?.kind==='equirectangular'){
          error('No target located automatically in the perspective views. Supply a perspective photo of your target. / 未在透視視圖中自動定位到目標，請提供目標樹的透視照片。');
          return;
        }
        fallback=await choosePhotoFallback(item,detection);
        if(!fallback||id!==state.requestId)return;
      }else{
        target=detection.action==='auto'&&!forcePicker?detection.candidates.find(c=>c.id===detection.auto_id):await chooseTree(detection);
        if(!target||id!==state.requestId)return;
      }
    }
    $('run').textContent='Identifying… / 正在識別…';
    const samSingle=item.manualPolygon?.crop_policy==='sam_tight_bbox';
    const d=await imageRequest(target?'/api/identify-tree':'/api/predict',fallback?.item||item,
                               target?{token:target.token}:samSingle?{view_policy:'single'}:{});
    if(id!==state.requestId)return;
    if(samSingle&&item.polygonOriginal){
      $('run').textContent='Comparing original… / 正在對照原圖…';
      try{
        const original=await imageRequest('/api/predict',item.polygonOriginal,{view_policy:'single'});
        if(id!==state.requestId)return;
        if(original.image_model_version!==d.image_model_version)throw Error('Model version changed during comparison / 對照期間模型版本變更');
        d.sam_comparison={available:true,original_top1:original.top3[0],crop_top1:d.top3[0],same_top1:original.top3[0].species===d.top3[0].species,
          original_top3:original.top3.map(r=>({species:r.species,score:r.score})),crop_top3:d.top3.map(r=>({species:r.species,score:r.score})),
          original_score_for_crop_candidate:original.image_display_distribution?.[d.top3[0].species]??null,
          method:'Same-version single-view classification of full photo versus confirmed tight SAM crop; no fusion or automatic score selection.',
          original_may_contain_other_trees:true,accuracy_estimate:false};
        if(!d.sam_comparison.same_top1)d.warnings.push('Original photo and tight SAM crop rank different candidates first. Check which view actually contains the intended tree. / 原圖與 SAM 緊裁片的首位候選不同，請核對哪張影像真正呈現目標樹。');
      }catch(e){d.sam_comparison={available:false,reason:e.message||'Original-photo comparison unavailable / 原圖對照暫不可用'};}
    }
    state.result={...d,input_name:item.file.name,organ_note:item.organ,
      target_selection:fallback?.kind||(target?(detection.action==='auto'&&!forcePicker?'centered_target':'user_selected'):'explicit_organ_detail'),
      manual_crop_box:fallback?.box||null,
      original_image_size:[item.width,item.height],
      ...(fallback?{crop_preview:fallback.preview}:{}),
      detection_summary:detection?{detector:detection.detector,threshold:detection.threshold,
        seconds:detection.seconds,suppressed:detection.suppressed,candidate_count:detection.candidates.length,
        tiled:detection.tiled,projection:detection.projection,perspective_view_count:detection.perspective_view_count,scores_calibrated:false}:null,
      reference_role:'dataset labels; not independent ground truth'};
    renderResult();
    if(samSingle)renderSamComparison(item,state.result.sam_comparison);
    if(fallback){
      const label=fallback.kind==='user_manual_crop'?'Manual target / 手動選擇目標':'User-confirmed whole photo / 使用者確認整張圖片';
      $('resultMeta').textContent+=' · '+label;
      $('evidenceText').textContent+=' '+label+'. No automatic tree box was accepted; classification ran after your confirmation. / 未取得合格的自動樹框；經您確認後執行分類。';
      const line=el('p','target-return',label),preview=el('img','selected-target-preview');preview.src=fallback.preview;preview.alt=label;
      line.append(preview);$('quality').prepend(line);
    }
    if(target){
      $('resultMeta').textContent+=' · Tree '+target.id+' / 第 '+target.id+' 株';
      $('evidenceText').textContent+=' Detector / 偵測器: '+detection.detector+(d.selected_tree?.projection_view?'. Source panorama reprojected before detection and classification; dimensions describe resampled pixels. / 原始全景先重投影再偵測與分類；尺寸為重採樣像素。':'. Native crop with 4% context padding / 原生裁片，四周保留 4% 上下文。');
      if(d.selected_tree?.projection_view)$('resultMeta').textContent+=' · Perspective corrected / 已透視校正';
      const line=el('p','target-return','Selected target / 本次目標：');
      const image=el('img','selected-target-preview');image.src=d.crop_preview;image.alt='Selected crop / 所選切片';
      const again=el('button','secondary','Choose another tree / 重新選樹');
      again.onclick=async()=>{
        if(state.busy)return;
        // Re-run on the original full-resolution upload, never on the thumbnail.
        state.forcePicker=true;await $('run').onclick();
      };
      const download=el('button','secondary','Export target crop / 匯出目標切片');
      download.title='Native crop used for this result, not the preview thumbnail / 匯出本次識別使用的原生切片，並非預覽縮圖';
      download.onclick=()=>exportTarget(item,target,download);
      line.append(image,again,download);$('quality').prepend(line);
    }
  }catch(e){error(e.name==='AbortError'?'Request timed out; retry shortly. / 請求逾時，請稍後重試。':e.message==='Failed to fetch'?'Connection lost. Check the local service. / 連線中斷，請檢查本地服務。':e.message);}
  finally{state.busy=false;update();}
};
