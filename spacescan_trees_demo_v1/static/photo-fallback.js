'use strict';
// Explicit user fallback only. Never fabricates a detection box or confidence.
async function choosePhotoFallback(item,detection=null){
  const dialog=el('dialog','photo-fallback');
  dialog.setAttribute('aria-label','Select a target / 選擇目標');
  const title=el('h2','','Target not located automatically / 未自動定位到目標樹');
  const rejectedDark=Number(detection?.low_visibility_removed||0)>0;
  const note=el('p','',rejectedDark
    ?'The automatic box contained almost no visible tree detail, so it was rejected. Select the complete tree or use the whole photo; the original pixels will be used for identification. / 自動樹框幾乎沒有可見樹木細節，已拒絕該框。請手動選取完整樹木，或使用整張照片；識別仍採用原始像素。'
    :'This does not mean the photo is low quality. Identify the whole photo if it shows your target, or select a region manually. / 這不代表照片品質差。若照片已包含您的目標，可識別整張圖片，或手動選擇區域。');
  const canvas=el('canvas'),ctx=canvas.getContext('2d');
  canvas.setAttribute('aria-label','Drag to select a rectangle / 拖曳選擇矩形區域');
  const preview=el('div','photo-fallback-preview');preview.append(canvas);
  const status=el('p','',''),fields=el('fieldset','photo-fallback-fields');
  fields.append(el('legend','','Selection in original pixels / 原圖像素選區'));
  const inputs={};
  for(const [key,label,max] of [['x','Left / 左',item.width-1],['y','Top / 上',item.height-1],['w','Width / 寬',item.width],['h','Height / 高',item.height]]){
    const wrap=el('label','',label),input=el('input');input.type='number';input.min=key==='w'||key==='h'?1:0;input.max=max;input.step=1;inputs[key]=input;wrap.append(input);fields.append(wrap);
  }
  const actions=el('div','photo-fallback-actions'),whole=el('button','primary','Identify whole photo / 識別整張圖片'),manual=el('button','secondary','Select target manually / 手動選擇目標'),confirm=el('button','primary','Identify selected region / 識別所選區域'),cancel=el('button','secondary','Cancel / 取消');
  confirm.hidden=true;preview.hidden=true;fields.hidden=true;confirm.disabled=true;
  actions.append(whole,manual,confirm,cancel);dialog.append(title,note,preview,fields,status,actions);document.body.append(dialog);
  const im=new Image();im.src=item.url;
  let rect=null,start=null,resolve,encoding=false;
  const answer=new Promise(r=>resolve=r);
  const finish=value=>{dialog.close();dialog.remove();resolve(value);};
  cancel.onclick=()=>{if(!encoding)finish(null);};
  dialog.addEventListener('cancel',e=>{e.preventDefault();if(!encoding)finish(null);});
  whole.onclick=()=>finish({kind:'user_confirmed_whole_photo',item:{...item,projection:'photo'},preview:item.url,box:null});
  function draw(){
    ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(im,0,0,canvas.width,canvas.height);
    if(rect){const [x,y,w,h]=rect,sx=canvas.width/item.width,sy=canvas.height/item.height;ctx.fillStyle='#ffffff22';ctx.fillRect(x*sx,y*sy,w*sx,h*sy);ctx.strokeStyle='#ffd94a';ctx.lineWidth=3;ctx.strokeRect(x*sx,y*sy,w*sx,h*sy);}
    confirm.disabled=!rect||rect[2]<16||rect[3]<16;
    status.textContent=rect?`${rect[2]} × ${rect[3]} px · Selection, not automatic detection / 手動選區，並非自動偵測`:'Drag around the full target, including its crown and trunk. You can also enter pixel coordinates. / 請拖曳框住完整樹冠及樹幹，亦可輸入像素座標。';
  }
  const sync=()=>{if(rect)['x','y','w','h'].forEach((k,i)=>inputs[k].value=rect[i]);draw();};
  manual.onclick=async()=>{
    try{await im.decode();if(!dialog.isConnected)return;const scale=Math.min(1,1000/item.width,550/item.height);canvas.width=Math.round(item.width*scale);canvas.height=Math.round(item.height*scale);preview.hidden=false;fields.hidden=false;confirm.hidden=false;draw();}
    catch{status.textContent='Image could not be loaded / 無法載入影像';}
  };
  function point(e){const b=canvas.getBoundingClientRect();return [Math.round(Math.max(0,Math.min(1,(e.clientX-b.left)/b.width))*item.width),Math.round(Math.max(0,Math.min(1,(e.clientY-b.top)/b.height))*item.height)];}
  canvas.onpointerdown=e=>{if(encoding||e.button!==0)return;start=point(e);rect=null;canvas.setPointerCapture(e.pointerId);draw();};
  canvas.onpointermove=e=>{if(!start)return;const p=point(e);rect=[Math.min(start[0],p[0]),Math.min(start[1],p[1]),Math.abs(p[0]-start[0]),Math.abs(p[1]-start[1])];sync();};
  canvas.onpointerup=canvas.onpointercancel=()=>{start=null;};
  Object.values(inputs).forEach(input=>input.oninput=()=>{
    const [x,y,w,h]=['x','y','w','h'].map(k=>Number(inputs[k].value));
    if([x,y,w,h].every(Number.isInteger)&&x>=0&&y>=0&&w>0&&h>0&&x+w<=item.width&&y+h<=item.height){rect=[x,y,w,h];draw();}
    else{rect=null;confirm.disabled=true;status.textContent='Enter a valid region inside the image / 請輸入影像範圍內的有效選區';}
  });
  confirm.onclick=async()=>{
    if(confirm.disabled||encoding)return;encoding=true;actions.querySelectorAll('button').forEach(b=>b.disabled=true);
    try{
      const [x,y,w,h]=rect,out=document.createElement('canvas');out.width=w;out.height=h;
      out.getContext('2d').drawImage(im,x,y,w,h,0,0,w,h);
      const blob=await new Promise(r=>out.toBlob(r,'image/png'));if(!blob)throw Error();
      if(blob.size>20*1024*1024)throw Error('Selected region exceeds 20 MB; choose a smaller region. / 所選區域超過 20 MB，請縮小選區。');
      finish({kind:'user_manual_crop',item:{...item,file:new File([blob],'manual-target.png',{type:'image/png'}),width:w,height:h,projection:'photo'},box:[x,y,x+w,y+h],preview:out.toDataURL('image/jpeg',.9)});
    }catch(e){encoding=false;actions.querySelectorAll('button').forEach(b=>b.disabled=false);status.textContent=e.message||'Crop failed; try again / 裁切失敗，請重試';}
  };
  dialog.showModal();return answer;
}
