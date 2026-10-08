'use strict';
// Native-resolution, explicit user selection. Never produces a detector score.
(() => {
  const bar=el('div','polygon-tools'),smart=el('button','secondary','AI select (SAM) / 智能選樹'),crop=el('button','secondary','Polygon crop / 多邊形裁剪'),restore=el('button','text-button','Restore original / 還原原圖'),note=el('p');
  smart.id='samSelect';crop.id='polygonCrop';restore.id='restorePolygon';note.id='polygonNote';bar.append(smart,crop,restore,note);
  $('loadedInput').querySelector('.main-image').after(bar);
  let samReady=false;
  const isPanorama=item=>!!item&&(item.projection==='equirectangular'||((item.projection||'auto')==='auto'&&item.width/item.height>=1.96&&item.width/item.height<=2.04));
  const priorUpdate=update;
  update=function(){priorUpdate();const item=state.items[state.active];crop.disabled=state.busy||!item;smart.disabled=state.busy||!item||!samReady||isPanorama(item?.polygonOriginal||item);smart.title=!samReady?'SAM unavailable; use manual polygon crop / SAM 暫不可用，請手動裁剪':isPanorama(item?.polygonOriginal||item)?'Use perspective-corrected tree selection for panoramas / 全景請使用透視校正選樹':'Click the target tree, review the proposed outline, then confirm / 點選目標樹，檢查輪廓後確認';restore.hidden=!item?.polygonOriginal;restore.disabled=state.busy;note.hidden=!item?.manualPolygon;$('projection').disabled=state.busy||!!item?.manualPolygon;note.textContent=item?.manualPolygon?.crop_policy==='sam_tight_bbox'?'SAM located the tree; recognition uses one unmasked tight crop, without a second center-detail vote. / SAM 已定位樹木；識別僅使用未遮灰的緊裁片，不再額外判斷中心細節。':'Target confirmed manually; identify this crop directly. / 已手動確認目標；識別時直接使用此裁片。';};
  fetch('/api/sam/status').then(r=>r.json()).then(d=>{samReady=!!d.ready;update();}).catch(()=>{samReady=false;update();});
  const priorRender=renderResult;
  renderResult=function(){
    const item=state.items[state.active];
    if(state.result&&item?.manualPolygon)Object.assign(state.result,{target_selection:item.manualPolygon.assist_model?'sam_assisted_user_confirmed':'user_polygon_crop',manual_polygon:item.manualPolygon,original_input_name:item.polygonOriginal.file.name,original_image_size:[item.polygonOriginal.width,item.polygonOriginal.height]});
    priorRender();
    if(state.result&&item?.manualPolygon){$('resultMeta').textContent+=item.manualPolygon.assist_model?' · SAM-assisted target / SAM 輔助選樹':' · Polygon target / 多邊形目標';$('evidenceText').textContent+=item.manualPolygon.crop_policy==='sam_tight_bbox'?' SAM supplied the outline; inference used an unmasked, tightly bounded native-pixel crop without a second center-detail vote. / SAM 提供輪廓；識別使用原圖像素的緊裁片，不再額外判斷中心細節。':' Outside the selected polygon is neutral gray; original pixel resolution retained. The target outline was confirmed by the user. / 多邊形外以中性灰遮蔽，保留原圖像素解像度；目標輪廓已由使用者確認。';if(item.manualPolygon.source_panorama)$('evidenceText').textContent+=' This manual panorama crop is not perspective-corrected. / 此手動全景裁片未作透視校正。';}
  };
  restore.onclick=()=>{if(state.busy)return;const item=state.items[state.active];if(!item?.polygonOriginal)return;URL.revokeObjectURL(item.url);state.items[state.active]=item.polygonOriginal;state.requestId++;invalidate();error('');update();};

  function valid(points){
    if(points.length<3)return false;
    const cross=(a,b,c)=>(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
    const on=(a,b,p)=>cross(a,b,p)===0&&p[0]>=Math.min(a[0],b[0])&&p[0]<=Math.max(a[0],b[0])&&p[1]>=Math.min(a[1],b[1])&&p[1]<=Math.max(a[1],b[1]);
    function hit(a,b,c,d){const u=cross(a,b,c),v=cross(a,b,d),w=cross(c,d,a),z=cross(c,d,b);return (u*v<0&&w*z<0)||on(a,b,c)||on(a,b,d)||on(c,d,a)||on(c,d,b);}
    let area=0;
    for(let i=0;i<points.length;i++){const a=points[i],b=points[(i+1)%points.length];if(a[0]===b[0]&&a[1]===b[1])return false;area+=a[0]*b[1]-b[0]*a[1];for(let j=i+1;j<points.length;j++){if(j===i+1||(i===0&&j===points.length-1))continue;if(hit(a,b,points[j],points[(j+1)%points.length]))return false;}}
    return Math.abs(area)>=128;
  }
  async function openEditor(startSam=false){
    if(state.busy||!state.items[state.active])return;
    const active=state.active,current=state.items[active],original=current.polygonOriginal||current;
    const isPano=isPanorama(original);
    const dialog=el('dialog');dialog.id='polygonEditor';dialog.setAttribute('aria-labelledby','polygonTitle');
    const title=el('h2','','Select the target tree / 點選目標樹木');title.id='polygonTitle';
    const manualHelp='Click around the crown and trunk. Close with the first point or “Preview”; drag vertices to adjust. Backspace: undo · Esc: cancel. / 沿樹冠及樹幹逐點點選，點首點或「預覽」封閉；拖曳頂點調整。退格鍵：撤銷；Esc：取消。';
    const samHelp='Click once inside the target tree. SAM proposes an outline; inspect it and drag vertices if needed. Use crop only after confirming. / 在目標樹內點一下；SAM 會建議輪廓，請檢查並按需拖動頂點，確認後才使用裁片。';
    const help=el('p','',startSam?samHelp:manualHelp);
    const layout=el('div','polygon-layout'),viewport=el('div','polygon-viewport'),canvas=el('canvas'),ctx=canvas.getContext('2d');canvas.id='polygonCanvas';canvas.tabIndex=0;canvas.setAttribute('aria-label','Click polygon vertices / 點選多邊形頂點');viewport.append(canvas);
    const side=el('aside'),preview=el('img','polygon-preview');preview.id='polygonPreview';preview.alt='Recognition crop preview / 識別裁片預覽';preview.hidden=true;
    const previewNote=el('p','','SAM: recognition uses the tight outline bounds from the original pixels, with no gray mask or added margin; only one model view is used. Manual polygon still masks outside gray. The original remains available. / SAM：識別使用輪廓緊框內的原圖像素，不遮灰、不擴邊，只作單次推理。手動畫多邊形仍將外部遮灰；原圖可還原。');
    side.append(el('p','','Crop preview / 裁片預覽'),preview,previewNote);layout.append(viewport,side);
    const status=el('p','polygon-status');status.id='polygonStatus';status.setAttribute('aria-live','polite');
    const samMessage=el('p','sam-status',startSam?'SAM is ready: click inside the target tree. / SAM 已就緒：請點選目標樹內部。':'SAM can propose an outline from one click; manual editing remains available. / SAM 可由單點建議輪廓，仍可手動修正。');
    const warning=el('p');warning.setAttribute('role','alert');
    if(isPano)warning.textContent='Panorama: manual cropping does not remove spherical distortion. For corrected views, cancel and use automatic tree selection first. / 全景：手動裁剪不消除球面畸變；如需校正視圖，請取消並先使用自動選樹。';
    const zoomLabel=el('label','polygon-zoom','Zoom / 放大'),zoom=el('input');zoom.type='range';zoom.min=1;zoom.max=4;zoom.step=.25;zoom.value=1;zoom.id='polygonZoom';zoomLabel.append(zoom);
    const actions=el('div','polygon-actions'),samToggle=el('button','secondary',startSam?'SAM on / SAM 已開啟':'Use SAM / 使用 SAM'),undo=el('button','secondary','Undo point / 撤銷頂點'),reset=el('button','secondary','Redraw / 重新繪製'),close=el('button','secondary','Preview / 預覽'),cancel=el('button','secondary','Cancel / 取消'),apply=el('button','primary','Use crop / 使用裁片');
    samToggle.id='samToggle';samToggle.disabled=!samReady||isPano;undo.id='polygonUndo';reset.id='polygonReset';close.id='polygonClose';cancel.id='polygonCancel';apply.id='polygonApply';apply.disabled=true;actions.append(samToggle,undo,reset,close,cancel,apply);
    dialog.append(title,help,zoomLabel,layout,status,samMessage,warning,actions);document.body.append(dialog);
    let points=current.manualPolygon?current.manualPolygon.vertices.map(p=>[...p]):[],closed=points.length>=3,drag=-1,encoding=false,alive=true,out=null,box=null,previewURL=null,samMode=startSam,samBusy=false,samController=null,samProposed=!!current.manualPolygon?.assist_model;
    const image=new Image();image.src=original.url;
    state.busy=true;update();dialog.showModal();
    function finish(){if(!alive)return;alive=false;if(samController)samController.abort();if(previewURL)URL.revokeObjectURL(previewURL);dialog.close();dialog.remove();state.busy=false;update();(startSam?smart:crop).focus();}
    cancel.onclick=()=>{if(!encoding)finish();};dialog.oncancel=e=>{e.preventDefault();if(!encoding)finish();};
    function path(c,sx,sy,ox=0,oy=0){c.beginPath();points.forEach((p,i)=>c[i?'lineTo':'moveTo']((p[0]-ox)*sx,(p[1]-oy)*sy));if(closed)c.closePath();}
    function draw(){
      if(!alive)return;const sx=canvas.width/original.width,sy=canvas.height/original.height;
      ctx.drawImage(image,0,0,canvas.width,canvas.height);path(ctx,sx,sy);ctx.strokeStyle='#ffe766';ctx.lineWidth=2;ctx.stroke();if(closed){ctx.fillStyle='#89ce9944';ctx.fill();}
      points.forEach((p,i)=>{ctx.beginPath();ctx.arc(p[0]*sx,p[1]*sy,5,0,Math.PI*2);ctx.fillStyle=i===0?'#ffffff':'#ffe766';ctx.fill();ctx.fillStyle='#092e20';ctx.font='bold 12px sans-serif';ctx.fillText(i+1,p[0]*sx+8,p[1]*sy-6);});
      const ok=closed&&valid(points);undo.disabled=!points.length;reset.disabled=!points.length;close.disabled=points.length<3||closed;apply.disabled=true;apply.textContent=samProposed?'Use tight crop / 使用緊裁片':'Use crop / 使用裁片';preview.hidden=true;out=null;
      status.textContent=points.length+' vertices / 個頂點'+(closed?(ok?' · Preparing preview / 正在產生預覽':' · Invalid: avoid crossed edges or tiny selections / 無效：請避免交叉邊線或過小選區'):' · Click to add / 點選新增');
      if(!ok)return;
      const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]);
      try{box=samProposed?samContextBox(points,original.width,original.height,0).crop_box:[Math.floor(Math.min(...xs)),Math.floor(Math.min(...ys)),Math.ceil(Math.max(...xs)),Math.ceil(Math.max(...ys))];}
      catch{status.textContent='SAM outline is too narrow; adjust vertices or redraw. / SAM 輪廓過窄，請調整頂點或重新描繪。';return;}
      const [x,y,r,b]=box;if(r-x<16||b-y<16){status.textContent='Select at least 16 × 16 pixels / 請選擇至少 16 × 16 像素';return;}
      out=document.createElement('canvas');out.width=r-x;out.height=b-y;const c=out.getContext('2d');
      if(samProposed)c.drawImage(image,-x,-y);
      else{c.fillStyle='#808080';c.fillRect(0,0,out.width,out.height);c.save();path(c,1,1,x,y);c.clip();c.drawImage(image,-x,-y);c.restore();}
      const snap=out;out.toBlob(blob=>{if(!alive||out!==snap||!blob)return;if(previewURL)URL.revokeObjectURL(previewURL);previewURL=URL.createObjectURL(blob);preview.src=previewURL;preview.hidden=false;apply.disabled=false;status.textContent=`${points.length} vertices / 個頂點 · ${snap.width} × ${snap.height} px`+(samProposed?' · Tight original-pixel crop / 原圖像素緊裁片':' · Gray outside polygon / 輪廓外遮灰')+(samProposed&&Math.min(snap.width,snap.height)<224?' · Small crop; check missing detail / 裁片偏小，請檢查細節':'');},'image/jpeg',.88);
    }
    function resize(){const scale=Math.min((viewport.clientWidth-2)/original.width,viewport.clientHeight/original.height,1)*Number(zoom.value);canvas.width=Math.max(1,Math.round(original.width*scale));canvas.height=Math.max(1,Math.round(original.height*scale));draw();}
    function point(e){const b=canvas.getBoundingClientRect();return [Math.round(Math.max(0,Math.min(original.width,(e.clientX-b.left)/b.width*original.width))),Math.round(Math.max(0,Math.min(original.height,(e.clientY-b.top)/b.height*original.height)))];}
    function near(p){return points.findIndex(q=>Math.hypot((q[0]-p[0])*canvas.width/original.width,(q[1]-p[1])*canvas.height/original.height)<12);}
    async function propose(pointPixels){
      if(samBusy||!samReady||isPano)return;
      samBusy=true;samController=new AbortController();samToggle.disabled=true;apply.disabled=true;samMessage.textContent='SAM is outlining the clicked tree… / SAM 正在描繪點選的樹木…';
      const timer=setTimeout(()=>samController.abort(),195000);
      try{
        const body=new FormData();body.append('image',original.file);body.append('x',String(pointPixels[0]/original.width));body.append('y',String(pointPixels[1]/original.height));body.append('projection',original.projection||'auto');
        const response=await fetch('/api/sam/segment',{method:'POST',body,signal:samController.signal});
        const data=await readApiResponse(response);if(!response.ok)throw Error(typeof data.detail==='string'?data.detail:'SAM could not outline this tree / SAM 無法描繪此樹');
        if(!alive)return;
        points=data.polygon.map(p=>[Math.round(p[0]*original.width),Math.round(p[1]*original.height)]);closed=true;samProposed=true;draw();
        samMessage.textContent='SAM proposal: check the crown and trunk; drag vertices or redraw before using it. / SAM 建議輪廓：請檢查樹冠與樹幹，可拖動頂點或重新描繪。';
      }catch(e){if(alive)samMessage.textContent=e.name==='AbortError'?'SAM timed out; draw manually or retry. / SAM 逾時，請手動描繪或重試。':e.message;}
      finally{clearTimeout(timer);samBusy=false;samController=null;if(alive){samToggle.disabled=!samReady||isPano;draw();}}
    }
    canvas.onpointerdown=e=>{if(encoding||samBusy||e.button!==0)return;e.preventDefault();canvas.focus();const p=point(e),i=near(p);if(closed&&i>=0){drag=i;canvas.setPointerCapture(e.pointerId);return;}if(samMode){propose(p);return;}if(closed)return;if(i===0&&points.length>=3)closed=true;else if(i<0)points.push(p);draw();};
    canvas.onpointermove=e=>{if(drag<0||encoding||samBusy)return;points[drag]=point(e);draw();};canvas.onpointerup=canvas.onpointercancel=()=>drag=-1;
    samToggle.onclick=()=>{if(samBusy)return;samMode=!samMode;samToggle.textContent=samMode?'SAM on / SAM 已開啟':'Use SAM / 使用 SAM';help.textContent=samMode?samHelp:manualHelp;samMessage.textContent=samMode?'Click inside the target tree; review the generated outline. / 點選目標樹內部，並檢查產生的輪廓。':'Manual polygon mode. / 手動多邊形模式。';};
    undo.onclick=()=>{if(encoding||samBusy)return;points.pop();closed=false;samProposed=false;draw();};reset.onclick=()=>{if(encoding||samBusy)return;points=[];closed=false;samProposed=false;draw();};close.onclick=()=>{if(encoding||samBusy)return;closed=true;draw();};zoom.oninput=resize;
    dialog.onkeydown=e=>{if(e.key==='Backspace'&&!encoding&&!['INPUT','TEXTAREA'].includes(e.target.tagName)){e.preventDefault();undo.onclick();}};
    apply.onclick=async()=>{
      if(apply.disabled||encoding||!out)return;encoding=true;actions.querySelectorAll('button').forEach(b=>b.disabled=true);zoom.disabled=true;
      try{const blob=await new Promise(resolve=>out.toBlob(resolve,'image/png'));if(!blob)throw Error('Crop failed / 裁剪失敗');if(blob.size>20*1024*1024)throw Error('Crop exceeds 20 MB; select a smaller region / 裁片超過 20 MB，請縮小選區');
        const cropMeta=samProposed?samContextBox(points,original.width,original.height,0):null;
        const file=new File([blob],original.file.name.replace(/\.[^.]+$/,'')+(samProposed?'-sam-tight.png':'-polygon.png'),{type:'image/png'});
        if(current.polygonOriginal)URL.revokeObjectURL(current.url);
        state.items[active]={...original,file,url:URL.createObjectURL(blob),width:out.width,height:out.height,projection:'photo',polygonOriginal:original,
          manualPolygon:{vertices:points.map(p=>[...p]),coordinate_space:'original_image_pixels',source_size:[original.width,original.height],selection_box:cropMeta?.selection_box||box,crop_box:box,crop_policy:samProposed?'sam_tight_bbox':'polygon_gray_mask',context_padding_fraction:0,mask_background:samProposed?null:[128,128,128],source_panorama:isPano,perspective_corrected:false,assist_model:samProposed?'SAM 2.1 Tiny':null,proposal_user_confirmed:!!samProposed}};
        state.requestId++;state.forcePicker=false;invalidate();error('');finish();
      }catch(e){encoding=false;actions.querySelectorAll('button').forEach(b=>b.disabled=false);zoom.disabled=false;draw();warning.textContent=e.message;}
    };
    try{await image.decode();if(alive){resize();canvas.focus();}}catch{warning.textContent='Image could not be loaded / 無法載入影像';}
  }
  smart.onclick=()=>openEditor(true);
  crop.onclick=()=>openEditor(false);
  update();
})();
