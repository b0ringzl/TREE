'use strict';
window.renderPointEvidence=function(container,d){
  container.replaceChildren();container.hidden=false;
  const node=(tag,text,cls)=>{const n=document.createElement(tag);if(text)n.textContent=text;if(cls)n.className=cls;return n;};
  container.append(node('h3','Point-cloud evidence / 點雲輔助依據'));
  container.append(node('p',d.support_state==='insufficient_evidence'?'Insufficient geometry evidence / 幾何依據不足':'Structure available · verification required / 已有結構量測・仍需核驗','pe-state'));
  const grid=node('div',null,'pe-grid'),canvas=node('canvas');canvas.width=600;canvas.height=380;
  canvas.setAttribute('aria-label','Rotate observed point cloud / 旋轉已觀測點雲');grid.append(canvas);
  const metrics=node('div',null,'pe-metrics');
  const fmt=v=>v===null||v===undefined?'Not available / 暫不可用':Number(v).toFixed(2)+' m';
  const fields=[['Independent points / 獨立點',d.unique_count+' / '+d.point_count],['Observed vertical span / 已觀測垂直跨度',fmt(d.observed_vertical_span_m)],['Observed upper widths / 上部已觀測寬度',d.observed_upper_width_m?d.observed_upper_width_m.map(fmt).join(' × '):'Not available / 暫不可用'],['Verified total height / 已核實整樹高度','Not available / 暫不可用']];
  fields.forEach(([label,value])=>{const p=node('p');p.append(node('small',label),node('strong',String(value)));metrics.append(p);});grid.append(metrics);container.append(grid);
  container.append(node('p','Drag to rotate · Scroll to zoom · ≤2,048 real display points; measurements use the uploaded unique points. / 拖曳旋轉・滾輪縮放・顯示最多 2,048 個真實點；量測使用上傳的獨立點。','quiet'));
  container.append(node('p','Span uses the central 99% of observed Z values. Colour indicates relative height, not stem segmentation. / 跨度採用已觀測 Z 值中央 99% 區間；顏色表示相對高度，並非樹幹分割。','quiet'));
  if(d.preview_sampling){container.append(node('p','Coverage-aware preview: 20 height layers × 8 sectors; observed XYZ extremes retained. No points invented. / 覆蓋優先預覽：20 個高度層 × 8 個方位區，保留已觀測 XYZ 極值，未生成任何點。','pe-state'));container.append(node('p','Full observed XYZ spans / 全部已觀測 XYZ 跨度: '+d.observed_full_extent_m.map(v=>Number(v).toFixed(2)+' m').join(' × ')+' · Not verified whole-tree dimensions / 並非已核實整樹尺寸','quiet'));}
  if(d.height_profile?.length){const chart=node('div',null,'pe-profile');const max=Math.max(...d.height_profile,1);d.height_profile.forEach((v,i)=>{const bar=node('span');bar.style.height=(v/max*60+1)+'px';bar.title='Layer / 層 '+(i+1)+': '+v;chart.append(bar);});container.append(node('p','Low → high point distribution · not stem completeness / 低至高點分佈・並非樹幹完整度'),chart);}
  d.warnings.forEach(w=>container.append(node('p',w,'pe-warning')));
  container.append(node('p',d.fusion_used?'Uploaded points were used by PTv2 and score fusion. Geometry measurements are separate evidence. / 上傳點雲已用於 PTv2 與分數融合；幾何量測另作核驗依據。':'Image scores unchanged; this result contains no point score fusion. / 影像分數保持不變，此結果未融合點雲分數。','pe-policy'));
  const ctx=canvas.getContext('2d');let yaw=.4,pitch=.1,zoom=1,drag=null;
  function draw(){ctx.fillStyle='#eef3eb';ctx.fillRect(0,0,600,380);const points=d.preview_points||[];
    const rotated=points.map(([x,y,z],i)=>{const a=x*Math.cos(yaw)-y*Math.sin(yaw),b=x*Math.sin(yaw)+y*Math.cos(yaw);return [a,z*Math.cos(pitch)-b*Math.sin(pitch),b*Math.cos(pitch)+z*Math.sin(pitch),d.preview_lower_mask?d.preview_lower_mask[i]:z<-.35];}).sort((a,b)=>a[2]-b[2]);
    for(const [x,z,depth,lower] of rotated){ctx.fillStyle=lower?'#946b43':'#30785c';ctx.beginPath();ctx.arc(300+x*170*zoom,190-z*170*zoom,1.8,0,Math.PI*2);ctx.fill();}
    ctx.fillStyle='#426552';ctx.font='12px sans-serif';ctx.fillText('Observed points / 已觀測點 · display normalized / 顯示正規化',12,365);}
  canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);};
  canvas.onpointermove=e=>{if(!drag)return;yaw+=(e.clientX-drag[0])*.009;pitch=Math.max(-1.3,Math.min(1.3,pitch+(e.clientY-drag[1])*.009));drag=[e.clientX,e.clientY];draw();};canvas.onpointerup=()=>drag=null;canvas.onpointercancel=()=>drag=null;
  canvas.onwheel=e=>{e.preventDefault();zoom=Math.max(.3,Math.min(3,zoom*(e.deltaY<0?1.1:.9)));draw();};draw();
};
