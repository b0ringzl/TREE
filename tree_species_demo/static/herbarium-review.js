'use strict';
const $ = id => document.getElementById(id);
let rows = [], current = null, busy = false, undoStack = [];
const labels = {pending:'Pending / 待驗收',accepted:'Accepted / 已保留',rejected:'Rejected / 已剔除'};
const organs = {tree_or_crown_context:'Tree / crown / 樹形・樹冠',foliage_or_leafy_branch:'Foliage / 枝葉',flower_or_inflorescence:'Flowers / 花序',fruit_seed_or_cone:'Fruit / seed / cone / 果實・種子・球果',bark_trunk_or_roots:'Trunk / roots / 樹幹・根部',macro_or_staged_detail:'Macro / staged / 微距・擺拍'};
const lanes = {tree_context_candidate:'Tree-context candidate / 樹形候選；不保證整樹完整。',trunk_root_auxiliary_candidate:'Trunk / root candidate / 樹幹・根部輔助候選。',organ_auxiliary_candidate:'Organ-detail candidate / 花果葉特寫輔助候選。',macro_reference_only:'Reference only suggested / 建議僅作微距・擺拍參考。',duplicate_reference_only:'Duplicate suggested / 建議剔除重複版本。',cross_dataset_match_quarantine:'Possible existing-data duplicate / 疑似與既有資料重複，請複核。'};
function filtered(){return rows.filter(r=>(!$('species').value||r.source_species===$('species').value)&&($('status').value==='all'||r.human_review.status===$('status').value));}
function feedback(text,error=false){$('feedback').textContent=text;$('feedback').classList.toggle('error',error);}
async function api(url,options){const res=await fetch(url,options);const data=await res.json();if(!res.ok)throw Error(typeof data.detail==='string'?data.detail:'Request failed / 請求失敗');return data;}
function render(){
 const list=filtered();let r=list.find(r=>r.id===current);if(!r){r=list[0];current=r?.id||null;}
 const counts={pending:0,accepted:0,rejected:0};rows.forEach(r=>counts[r.human_review.status]++);
 $('progress').textContent=`Total / 全部 ${rows.length} · Pending / 待驗收 ${counts.pending} · Kept / 保留 ${counts.accepted} · Removed / 剔除 ${counts.rejected}`;
 $('empty').hidden=!!r;$('full').hidden=!r;$('photo').hidden=!r;
 $('empty').textContent='No images in this filter / 此篩選下沒有影像。請切換狀態或樹種。';
 for(const id of ['accept','reject','prev','next'])$(id).disabled=busy||!r;
 $('undo').disabled=busy||!undoStack.length;$('reload').disabled=busy;$('species').disabled=busy;$('status').disabled=busy;
 if(!r){$('name').textContent='Review complete / 此篩選已完成';for(const id of ['chinese','badge','organs','suggestion','position'])$(id).textContent='';$('source').hidden=true;return;}
 $('position').textContent=`${list.indexOf(r)+1} / ${list.length} · ID ${r.id}`;
 $('name').textContent=r.source_species;$('chinese').textContent=(r.source_names||[]).join(' / ');
 $('badge').textContent=labels[r.human_review.status];$('badge').className='badge '+r.human_review.status;
 $('organs').textContent=r.organ_review.organ_tags.map(t=>organs[t]||t).join(' · ');
 $('suggestion').textContent=lanes[r.candidate_lane]||r.candidate_lane;
 $('source').hidden=false;$('source').href=r.source_page;
 if($('photo').dataset.id!==r.id){$('photo').dataset.id=r.id;$('photo').src=r.image_url_local;$('full').href=r.image_url_local;}
 $('photo').alt=r.source_species+' / '+r.id;
 try{localStorage.setItem('herbariumReviewCurrent',r.id);}catch{}
 const nxt=list[list.indexOf(r)+1];if(nxt){const im=new Image();im.src=nxt.image_url_local;}
}
async function load(){if(busy)return;busy=true;render();try{const data=await api('/api/herbarium-review');rows=data.rows;const selected=$('species').value;$('species').replaceChildren(new Option('All / 全部',''));for(const s of [...new Set(rows.map(r=>r.source_species))])$('species').add(new Option(s,s));$('species').value=selected;feedback('Loaded from disk / 已讀取磁碟上的驗收記錄。');}catch(e){feedback(e.message,true);}finally{busy=false;render();}}
function move(delta){if(busy)return;const list=filtered(),idx=list.findIndex(r=>r.id===current);current=list[Math.max(0,Math.min(list.length-1,idx+delta))]?.id||null;render();}
async function save(status){
 if(busy||!current)return;
 const list=filtered(),idx=list.findIndex(r=>r.id===current),r=list[idx];if(!r)return;
 // Snapshot neighbours BEFORE the current row leaves the pending/rejected filter.
 const nextId=list[idx+1]?.id||list[idx-1]?.id||null;
 busy=true;render();feedback('Saving / 儲存中…');
 try{const data=await api('/api/herbarium-review/decision/'+r.id,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({status,expected_revision:r.human_review.revision})});undoStack.push({id:r.id,previous:data.previous_status,revision:data.revision});r.human_review=data;current=nextId;feedback('Saved / 已儲存 · '+labels[status]+' · '+r.id);}
 catch(e){feedback(e.message+'\nNot advanced; reload to confirm state / 未切換圖片，請重新載入核對狀態。',true);}
 finally{busy=false;render();}
}
async function undo(){if(busy||!undoStack.length)return;const action=undoStack.at(-1),r=rows.find(r=>r.id===action.id);busy=true;render();try{const data=await api('/api/herbarium-review/decision/'+action.id,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({status:action.previous,expected_revision:action.revision})});undoStack.pop();r.human_review=data;const older=[...undoStack].reverse().find(a=>a.id===action.id);if(older)older.revision=data.revision;current=r.id;if($('status').value!=='all'&&$('status').value!==data.status)$('status').value='all';if($('species').value&&$('species').value!==r.source_species)$('species').value='';feedback('Undone and saved / 已撤銷並儲存 · '+r.id);}catch(e){feedback(e.message,true);}finally{busy=false;render();}}
$('accept').onclick=()=>save('accepted');$('reject').onclick=()=>save('rejected');$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);$('undo').onclick=undo;$('reload').onclick=load;
for(const id of ['species','status'])$(id).onchange=()=>{current=null;render();};
$('photo').onerror=()=>feedback('Image could not be loaded. Do not judge quality from this error; reload. / 圖片載入失敗，請重新載入，勿據此判斷品質。',true);
document.addEventListener('keydown',e=>{if(e.repeat||busy||e.altKey||e.ctrlKey||e.metaKey||e.target.closest('input,select,textarea,[contenteditable="true"]'))return;const key=e.key.toLowerCase();const action={a:()=>save('accepted'),d:()=>save('rejected'),delete:()=>save('rejected'),z:undo,arrowleft:()=>move(-1),arrowright:()=>move(1)}[key];if(action){e.preventDefault();action();}});
try{current=localStorage.getItem('herbariumReviewCurrent');}catch{}
load();
