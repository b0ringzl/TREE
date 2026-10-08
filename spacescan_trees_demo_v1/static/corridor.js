'use strict';

const $ = id => document.getElementById(id);
const state = {catalog:null, mode:'catalog', folder:null, job:null, result:null, timer:null, modelReady:false, frameIndex:0};

function showError(message){$('error').textContent=message||'';$('error').hidden=!message;}
function element(tag,className,text){const node=document.createElement(tag);if(className)node.className=className;if(text!==undefined)node.textContent=text;return node;}
function bytes(value){if(value>=1073741824)return `${(value/1073741824).toFixed(1)} GB`;if(value>=1048576)return `${(value/1048576).toFixed(1)} MB`;return `${Math.max(1,Math.round(value/1024))} KB`;}
function canRun(){return state.modelReady&&(state.mode==='catalog'?!!state.catalog:!!state.folder);}
function refreshRun(){if(!$('working').hidden)return;$('run').disabled=!canRun();}

async function health(){
  try{
    const response=await fetch('/api/health'),data=await response.json();
    state.modelReady=!!data.ready;
    $('statusDot').classList.toggle('ready',state.modelReady);
    $('health').textContent=state.modelReady?'SSTv1 ready / 已就緒':'Model offline / 模型離線';
  }catch{state.modelReady=false;$('health').textContent='Service offline / 服務離線';}
  refreshRun();
}

async function loadCatalog(){
  try{
    const response=await fetch('/api/vmms/corridor/catalog',{cache:'no-store'}),data=await response.json();
    if(!response.ok)throw Error(data.detail||'Route unavailable / 軌跡不可用');
    state.catalog=data;
    $('segment').replaceChildren();
    data.segments.filter(row=>row.continuous_length_m>=10).forEach(row=>{
      const option=element('option','',`${row.stream_id} · frames ${row.start_frame}–${row.end_frame} · ${row.continuous_length_m.toFixed(1)} m`);
      option.value=row.id;$('segment').append(option);
    });
    $('segment').value=data.default_segment_id;
    syncCatalog();
  }catch(error){showError(error.message);}
  refreshRun();
}

function selectedSegment(){return state.catalog?.segments.find(row=>row.id===$('segment').value);}
function syncCatalog(){
  const row=selectedSegment();if(!row)return;
  const max=Math.min(200,row.continuous_length_m);
  $('length').max=Math.floor(max);$('length').value=Math.min(Number($('length').value)||200,Math.floor(max));
  const maxOffset=Math.max(0,row.continuous_length_m-Number($('length').value));
  $('offset').max=Math.floor(maxOffset);$('offset').value=Math.min(Number($('offset').value)||0,Math.floor(maxOffset));
  const length=Number($('length').value),spacing=Number($('spacing').value);
  $('sampleSummary').textContent=`${Math.floor(length/spacing)+1} panoramas / 張全景 · ${length} m · ${spacing} m spacing / 間距`;
  if(state.mode==='catalog'){
    $('dataset').textContent=state.catalog.title;
    $('datasetMeta').textContent=`${state.catalog.segments.length} continuous segments · ${state.catalog.coordinate_system}`;
  }
}

function syncFolder(edited){
  if(!state.folder)return;
  const total=state.folder.valid_image_count;
  const range=edited==='end'?CorridorRange.ending(total,$('folderStart').value,$('folderEnd').value,$('folderStep').value):CorridorRange.selection(total,$('folderStart').value,$('folderCount').value,$('folderStep').value);
  state.folderRange=range;
  $('folderStart').max=total-1;$('folderStart').value=range.start;
  $('folderStep').value=range.step;
  $('folderEnd').min=range.start+range.step;$('folderEnd').max=Math.min(total,range.start+299*range.step);$('folderEnd').value=range.end;
  $('folderCount').max=range.maxCount;$('folderCount').value=range.count;
  $('rangePrevious').disabled=range.start===1;$('rangeFirst').disabled=range.start===1;
  $('rangeNext').disabled=range.end>=total;$('rangeLast').disabled=range.end>=total;
  $('sampleSummary').textContent=`#${range.start}–#${range.end} · ${range.count} panoramas / 張全景 · ${range.step===1?'consecutive / 連續照片':`every ${range.step} frames / 每 ${range.step} 幀`}`;
  const names=state.folder.image_names;
  $('folderSelection').textContent=names?`#${range.start}: ${names[range.start-1]} → #${range.end}: ${names[range.end-1]}`:`#${range.start}–#${range.end} / 已選照片範圍 · Read folder to show filenames / 再次讀取文件夾可查看首尾文件名`;
  $('dataset').textContent=state.folder.name;
  $('datasetMeta').textContent=`${total} valid 2:1 panoramas · ${bytes(state.folder.total_bytes)} · ${state.folder.rejected_image_count} rejected / 已剔除`;
  $('folderInspection').textContent=`${state.folder.first_file} → ${state.folder.last_file} · ${Object.entries(state.folder.dimensions).map(([size,n])=>`${size} × ${n}`).join(' · ')} · files remain in place / 原圖保留在原位置`;
}

function acceptFolder(data){
  state.folder=data;$('folderPath').value=data.folder;
  // A fresh folder read starts a fresh consecutive selection, not a historic 136-frame job.
  $('folderStart').value=1;$('folderCount').value=300;$('folderStep').value=1;syncFolder();
}

function setMode(mode){
  state.mode=mode;
  $('useCatalog').classList.toggle('active',mode==='catalog');$('useFolder').classList.toggle('active',mode==='folder');
  $('catalogFields').hidden=mode!=='catalog';$('folderFields').hidden=mode!=='folder';
  showError('');
  if(mode==='catalog')syncCatalog();
  else if(state.folder)syncFolder();
  else{$('dataset').textContent='Choose a VMMS folder / 選擇 VMMS 文件夾';$('datasetMeta').textContent='2:1 equirectangular panoramas · filename order / 按文件名排序';$('sampleSummary').textContent='No folder selected / 尚未選擇文件夾';}
  refreshRun();
}

async function inspectFolder(folder){
  showError('');$('inspectFolder').disabled=true;$('chooseFolder').disabled=true;
  try{
    const response=await fetch('/api/vmms/corridor/folder/inspect',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({folder})}),data=await response.json();
    if(!response.ok)throw Error(data.detail||'Could not read folder / 無法讀取文件夾');
    acceptFolder(data);
  }catch(error){state.folder=null;showError(error.message);refreshRun();}
  finally{$('inspectFolder').disabled=false;$('chooseFolder').disabled=false;refreshRun();}
}

async function chooseFolder(){
  showError('');$('chooseFolder').disabled=true;$('inspectFolder').disabled=true;
  try{
    const response=await fetch('/api/vmms/corridor/folder/select',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({initial_folder:$('folderPath').value})}),data=await response.json();
    if(!response.ok)throw Error(data.detail||'Could not open folder / 無法開啟文件夾');
    if(data.cancelled)return;
    acceptFolder(data);
  }catch(error){showError(error.message);}
  finally{$('chooseFolder').disabled=false;$('inspectFolder').disabled=false;refreshRun();}
}

['segment','offset','spacing','length'].forEach(id=>$(id).addEventListener('change',syncCatalog));
['folderStart','folderStep','folderCount','folderEnd'].forEach(id=>{
  const update=()=>{clearTimeout(state.rangeTimer);state.rangeEdited=id==='folderEnd'?'end':undefined;syncFolder(state.rangeEdited);};
  $(id).addEventListener('change',update);
  $(id).addEventListener('input',()=>{clearTimeout(state.rangeTimer);state.rangeEdited=id==='folderEnd'?'end':undefined;state.rangeTimer=setTimeout(update,350);});
});
[['rangeFirst','first'],['rangePrevious','previous'],['rangeNext','next'],['rangeLast','last']].forEach(([id,action])=>$(id).onclick=()=>{
  if(!state.folderRange)return;
  const range=CorridorRange.move(state.folder.valid_image_count,state.folderRange,action);
  $('folderStart').value=range.start;$('folderCount').value=range.count;$('folderStep').value=1;syncFolder();
});
$('folderPath').addEventListener('input',()=>{if(state.folder&&$('folderPath').value!==state.folder.folder){state.folder=null;refreshRun();}});
$('useCatalog').onclick=()=>setMode('catalog');$('useFolder').onclick=()=>setMode('folder');
$('chooseFolder').onclick=chooseFolder;$('inspectFolder').onclick=()=>inspectFolder($('folderPath').value);

function setProgress(progress){
  const value=Math.max(0,Math.min(100,Number(progress?.percent)||0));$('bar').style.width=value+'%';$('percent').textContent=value+'%';$('phase').textContent=progress?.message||'Processing / 正在處理';
  const bits=[];if(progress?.sampled_frames!==undefined)bits.push(`${progress.sampled_frames}/${progress.total_frames||'?'} panoramas / 全景`);if(progress?.detections!==undefined)bits.push(`${progress.detections} detections / 偵測`);if(progress?.stable_tracks!==undefined)bits.push(`${progress.stable_tracks} stable tracks / 穩定軌跡`);if(progress?.classified_tracks!==undefined)bits.push(`${progress.classified_tracks} recognised / 已識別`);$('progressMeta').textContent=bits.join(' · ');
}

async function run(){
  if(state.mode==='folder'){clearTimeout(state.rangeTimer);syncFolder(state.rangeEdited);}
  showError('');$('run').disabled=true;$('working').hidden=false;$('results').hidden=true;setProgress({percent:0,message:'Starting local analysis / 正在啟動本機分析'});
  const body=state.mode==='folder'?{source_mode:'folder',folder:state.folder.folder,start_index:Number($('folderStart').value)-1,frame_step:Number($('folderStep').value),max_frames:Number($('folderCount').value)}:{source_mode:'catalog',segment_id:$('segment').value,offset_m:Number($('offset').value),length_m:Number($('length').value),spacing_m:Number($('spacing').value)};
  try{
    const response=await fetch('/api/vmms/corridor/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),data=await response.json();
    if(!response.ok)throw Error(data.detail||'Could not start test / 無法啟動分析');
    state.job=data.job_id;history.replaceState(null,'',`/corridor?job=${state.job}`);poll();
  }catch(error){showError(error.message);$('working').hidden=true;refreshRun();}
}
$('run').onclick=run;

async function poll(){
  try{
    const response=await fetch('/api/vmms/corridor/jobs/'+state.job,{cache:'no-store'}),data=await response.json();
    if(!response.ok)throw Error(data.detail||'Job unavailable / 任務不可用');setProgress(data.progress);
    if(data.status==='complete'){
      state.result=data.result;$('working').hidden=true;
      if(data.result.route?.source_mode==='folder'&&data.result.route.folder_inspection){
        state.folder=data.result.route.folder_inspection;
        $('folderPath').value=state.folder.folder;
        $('folderStart').value=Number(data.result.route.start_index||0)+1;
        $('folderStep').value=Number(data.result.route.frame_step||1);
        $('folderCount').value=Number(data.result.route.max_frames||data.result.route.sample_count||200);
        setMode('folder');
      }
      render();refreshRun();return;
    }
    if(data.status==='failed')throw Error(data.error||'Corridor analysis failed / 走廊分析失敗');
    state.timer=setTimeout(poll,1500);
  }catch(error){showError(error.message);$('working').hidden=true;refreshRun();}
}

function metric(value,label){const card=element('div','metric');card.append(element('b','',String(value)),element('span','',label));return card;}
function position(value,route){return route.distance_unit==='frame'?`image #${Math.round(value)+1} / 第 ${Math.round(value)+1} 張`:`${Number(value).toFixed(1)} m`;}
function showFrame(index){
  const frames=state.result?.frames||[],total=frames.length;
  if(!total)return;
  state.frameIndex=Math.max(0,Math.min(total-1,Math.round(Number(index)||0)));
  const frame=frames[state.frameIndex],readout=`${state.frameIndex+1} / ${total}`;
  const meta=`${position(frame.chainage_m,state.result.route)} · source frame ${frame.frame_id} / 原始幀 · ${frame.candidate_count} tree boxes / 個樹木框`;
  $('frameNumber').max=total;$('frameNumber').value=state.frameIndex+1;
  $('frameRange').max=total;$('frameRange').value=state.frameIndex+1;
  $('frameRange').setAttribute('aria-valuetext',`${readout} · ${meta}`);
  ['frameReadout','detailFrameReadout'].forEach(id=>$(id).textContent=readout);
  ['frameMeta','detailFrameMeta'].forEach(id=>$(id).textContent=meta);
  ['frameImage','detailFrameImage'].forEach(id=>{
    $(id).src=frame.preview_url;
    $(id).alt=`Processed frame ${state.frameIndex+1} of ${total}, source ${frame.frame_id} / 第 ${state.frameIndex+1} 幀處理結果`;
  });
  ['previousFrame','detailPreviousFrame'].forEach(id=>$(id).disabled=state.frameIndex===0);
  ['nextFrame','detailNextFrame'].forEach(id=>$(id).disabled=state.frameIndex===total-1);
  Array.from($('filmstrip').children).forEach((button,i)=>{
    button.classList.toggle('selected',i===state.frameIndex);
    if(i===state.frameIndex)button.setAttribute('aria-current','true');else button.removeAttribute('aria-current');
  });
}
['previousFrame','detailPreviousFrame'].forEach(id=>$(id).onclick=()=>showFrame(state.frameIndex-1));
['nextFrame','detailNextFrame'].forEach(id=>$(id).onclick=()=>showFrame(state.frameIndex+1));
$('frameNumber').onchange=()=>showFrame(Number($('frameNumber').value)-1);
$('goToFrame').onclick=()=>showFrame(Number($('frameNumber').value)-1);
$('frameNumber').onkeydown=event=>{if(event.key==='Enter'){event.preventDefault();showFrame(Number($('frameNumber').value)-1);}};
$('frameRange').oninput=()=>showFrame(Number($('frameRange').value)-1);
$('enlargeFrame').onclick=()=>$('frameDialog').showModal();
$('closeFrameDialog').onclick=()=>$('frameDialog').close();
['frameViewer','frameDialog'].forEach(id=>$(id).addEventListener('keydown',event=>{
  if(event.target.tagName==='INPUT')return;
  if(event.key==='ArrowLeft'||event.key==='ArrowRight'){
    event.preventDefault();showFrame(state.frameIndex+(event.key==='ArrowRight'?1:-1));
  }
}));
function render(){
  const data=state.result,tracks=data.tracks||[],frames=data.frames||[],folder=data.route.distance_unit==='frame';
  const extent=folder?`${data.route.sampled_start_position}–${data.route.sampled_end_position}`:`${data.route.actual_covered_m} m`;
  const stableTotal=Number(data.quality.stable_tracks_before_display_cap??tracks.length),omitted=Number(data.quality.stable_tracks_omitted_by_display_cap||0);
  const identityChecked=data.pipeline?.tracking_revision==='corridor-geometry-gallery-v4-20261003';
  $('identityNotice').hidden=identityChecked;
  $('identityNotice').textContent='Historical tracking result: different targets may be mixed. Run analysis again to apply geometry and appearance consistency checks. / 舊版追蹤結果可能混入不同目標；請重新分析以套用位置與外觀一致性檢查。';
  $('metrics').replaceChildren(metric(extent,folder?'Source image range / 原圖範圍':'Covered route / 實際覆蓋'),metric(data.route.sample_count,'Sampled panoramas / 採樣全景'),metric(tracks.length,'Shown tree candidates / 已顯示樹木候選'),metric(data.quality.low_value_or_unstable_detections_rejected,'Removed observations / 已剔除觀察'));
  $('trackCoverage').hidden=omitted===0;
  $('trackCoverage').textContent=`Showing ${tracks.length} of ${stableTotal} stable candidates; ${omitted} have not been classified. Priority: observation count, then image quality. / 顯示 ${stableTotal} 條穩定候選中的 ${tracks.length} 條；另有 ${omitted} 條尚未分類。優先處理觀察次數多、影像清晰的目標。`;
  $('resultSummary').textContent=`${tracks.length} candidates / 個候選`;$('filmstrip').replaceChildren();
  frames.forEach((frame,index)=>{
    const button=element('button','frame'),image=element('img');button.type='button';
    button.setAttribute('aria-label',`Frame ${index+1} / 第 ${index+1} 幀 · source ${frame.frame_id}`);
    image.src=frame.preview_url;image.alt=`Panorama ${frame.frame_id}`;image.loading='lazy';image.decoding='async';
    button.append(image,element('span','frame-caption',`${index+1} / ${frames.length} · ${position(frame.chainage_m,data.route)} · ${frame.candidate_count} boxes`));
    button.onclick=()=>showFrame(index);$('filmstrip').append(button);
  });
  $('frameViewer').hidden=frames.length===0;showFrame(0);
  $('tracks').replaceChildren();
  tracks.forEach(track=>{
    const card=element('article','track-card'+(track.determination==='undetermined'?' undetermined':'')),title=element('div','track-title'),heading=element('div'),winner=element('div','winner'),top=track.top3[0],review=track.human_review?.status==='confirmed'?track.human_review:null;
    heading.append(element('h3','',`Tree ${track.id} / 第 ${track.id} 株`),element('small','',`${position(track.first_chainage_m,data.route)}–${position(track.last_chainage_m,data.route)} · ${track.observations} observations · ${track.view_count} views`));
    if(track.identity_check)heading.append(element('small','',track.identity_check.status==='single_view_only'?'Single suitable view / 單一合適視角':'Selected views passed appearance consistency / 所選視角通過外觀一致性檢查'));
    if(review)winner.append(element('em','',review.scientific_name),element('small','',review.chinese_name+' · corrected / 已校正'));
    else winner.append(element('em','',top.scientific_name),element('small','',top.chinese_name+' · model score / 模型分數 · '),element('strong','',(top.score*100).toFixed(1)+'%'));
    title.append(heading,winner);card.append(title);const views=element('div','views');
    track.views.forEach(view=>{const figure=element('figure'),image=element('img');image.src=view.url;image.alt=`Tree ${track.id}`;figure.append(image,element('figcaption','',position(view.chainage_m,data.route)));views.append(figure);});card.append(views);
    const candidates=element('ol','candidates'),ranked=review?[...track.top3].sort((a,b)=>Number(b.species===review.species)-Number(a.species===review.species)):track.top3;
    ranked.forEach((row,index)=>{const item=element('li'),suffix=review&&index===0?' · 已校正':'';item.append(element('small','',String(index+1).padStart(2,'0')),element('em','',row.scientific_name+' · '+row.chinese_name+suffix),element('b','',(row.score*100).toFixed(1)+'%'));candidates.append(item);});card.append(candidates);
    if(track.warnings.length){const warnings=element('ul','warnings');track.warnings.forEach(text=>warnings.append(element('li','',text)));card.append(warnings);}$('tracks').append(card);
  });
  if(!tracks.length)$('tracks').append(element('div','empty-result','No tree produced a stable two-position track. / 沒有樹木形成至少兩個位置的穩定軌跡。'));
  $('limitations').replaceChildren();data.limitations.forEach(text=>$('limitations').append(element('li','',text)));$('results').hidden=false;$('results').scrollIntoView({behavior:'smooth',block:'start'});
}

function download(name,type,text){const url=URL.createObjectURL(new Blob([text],{type})),link=document.createElement('a');link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$('exportJson').onclick=()=>{if(state.result)download('spacescan-vmms-corridor.json','application/json',JSON.stringify({...state.result,exported_at:new Date().toISOString()},null,2));};
$('exportCsv').onclick=()=>{if(!state.result)return;const quote=value=>'"'+String(value??'').replaceAll('"','""')+'"',rows=[['tree_id','first_position','last_position','position_unit','observations','views','model_scientific_name','model_chinese_name','model_score','determination','human_review_status','human_scientific_name','human_chinese_name']];state.result.tracks.forEach(track=>rows.push([track.id,track.first_chainage_m,track.last_chainage_m,state.result.route.distance_unit,track.observations,track.view_count,track.top3[0].scientific_name,track.top3[0].chinese_name,track.top3[0].score,track.determination,track.human_review?.status||'',track.human_review?.scientific_name||'',track.human_review?.chinese_name||'']));download('spacescan-vmms-corridor.csv','text/csv;charset=utf-8','\ufeff'+rows.map(row=>row.map(quote).join(',')).join('\r\n'));};

const resumeJob=new URLSearchParams(location.search).get('job');
setMode('catalog');loadCatalog();health();setInterval(health,15000);
if(/^[0-9a-f]{32}$/.test(resumeJob||'')){state.job=resumeJob;$('working').hidden=false;$('run').disabled=true;poll();}
