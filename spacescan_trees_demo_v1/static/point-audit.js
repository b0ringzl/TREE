'use strict';
const $=id=>document.getElementById(id);let items=[],filtered=[],active=null,serial=0;
const checkNames=['same_tree','isolated_tree','base_visible','crown_complete','species_verified'];
const coverageStatus=document.createElement('p');coverageStatus.id='coverageStatus';$('coverageControls').append(coverageStatus);
$('coverageMode').options[0].textContent='Whole-cloud overview / 全點雲概覽';
function fail(e){$('error').textContent=e.message;$('error').hidden=false;}
async function request(url,options){const r=await fetch(url,options),d=await r.json();if(!r.ok)throw Error(typeof d.detail==='string'?d.detail:'Request failed / 請求失敗');return d;}
function filter(){filtered=items.filter(r=>!$('species').value||r.scientific_name===$('species').value);$('sample').replaceChildren();filtered.forEach((r,i)=>{const o=new Option(String(i+1).padStart(2,'0')+' · '+r.id,r.key);$('sample').add(o);});load();}
async function load(){const ticket=++serial;active=filtered.find(r=>r.key===$('sample').value);$('error').hidden=true;$('review').reset();$('saved').textContent='';$('evidence').replaceChildren();$('save').disabled=true;if(!active)return;
  const row=active;$('image').src=row.image;$('projection').src=row.projection;$('label').textContent=row.name;$('version').textContent='Asset version / 資料版本: '+row.version;
  $('coverageControls').hidden=!row.coverage_views;$('coverageViews').hidden=!row.coverage_views;$('auditImages').classList.toggle('coverage',!!row.coverage_views);$('coverageMode').value='all';$('coverageStatus').textContent='';
  if(row.coverage_views){$('orthoViews').src=row.coverage_views.views;$('coverageDownload').href=row.coverage_views.point;$('version').textContent+=' · Display / 顯示: '+row.display_version;}setProjection();
  $('position').textContent=(filtered.indexOf(row)+1)+' / '+filtered.length;$('useDemo').href='/?pair='+encodeURIComponent(row.key);
  const rev=row.geometry_review;for(const k of checkNames)$('review').elements[k].checked=!!rev?.[k];for(const k of ['corrected_species','note'])$('review').elements[k].value=rev?.[k]||'';
  try{const d=await request('/api/point-assist/record/'+row.key);if(ticket!==serial)return;renderPointEvidence($('evidence'),d.evidence);const {evidence,...meta}=d;$('provenance').textContent=JSON.stringify(meta,null,2);if(d.coverage_audit){const c=d.coverage_audit;$('coverageStatus').textContent='Measured points / 實測點: '+c.full_count.toLocaleString()+' · Upper / 上部 '+c.upper_count.toLocaleString()+' · Lower / 下部 '+c.lower_count.toLocaleString()+'. Projection uses ≤10,000 real display points across all regions; not a completeness score. / 投影以最多 10,000 個真實點呈現各區域；不代表完整度。';}$('save').disabled=false;}
  catch(e){if(ticket===serial)fail(e);}
}
function setProjection(){if(!active)return;const mode=$('coverageMode').value;const src=active.coverage_views?(mode==='archive'?active.archived_projection:active.coverage_views[mode]):active.projection;$('projection').src=src;$('projectionLink').href=src;$('projectionCaption').textContent=active.coverage_views?(mode==='archive'?'Old overlay: repaired lower stems ONLY / 舊投影：僅補回的幹部':'Measured cloud · click to enlarge / 實測點雲・點擊放大'):'Archived projection / 原有投影';}
$('coverageMode').onchange=setProjection;
$('species').onchange=filter;$('sample').onchange=load;
function move(step){const i=filtered.findIndex(r=>r.key===$('sample').value),r=filtered[i+step];if(r){$('sample').value=r.key;load();}}
$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);
$('review').onsubmit=async e=>{e.preventDefault();if(!active)return;const row=active,key=row.key,d={};checkNames.forEach(k=>d[k]=$('review').elements[k].checked);['note','corrected_species'].forEach(k=>d[k]=$('review').elements[k].value);$('save').disabled=true;
  try{const r=await request('/api/point-assist/review/'+key,{method:'POST',headers:{'Content-Type':'application/json','X-Point-Review':'1'},body:JSON.stringify(d)});row.geometry_review=r.review;if(active?.key===key)$('saved').textContent='Saved for this asset version / 已儲存此資料版本核驗';}
  catch(e){fail(e);}finally{if(active?.key===key)$('save').disabled=false;}};
request('/api/point-assist/catalog').then(d=>{items=d.items;$('count').textContent=d.accepted_views;$('species').add(new Option('All species / 全部樹種',''));d.classes.forEach(c=>$('species').add(new Option(c.name+' · '+c.accepted_views,c.scientific_name)));const initial=new URLSearchParams(location.search).get('species');if(d.classes.some(c=>c.scientific_name===initial))$('species').value=initial;filter();}).catch(fail);
