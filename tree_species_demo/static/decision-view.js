/* Presentation only: preserve scores, rejection and source selection upstream. */
function decisionView(decision){
  const review=decision.label==='Unknown';
  const candidates=(decision.top3||[]).filter(p=>Number.isFinite(p.score)).slice(0,3).map(p=>({...p}));
  return {review,source:decision.source,title:review?(candidates.length?'Most likely candidates / 最可能的候選樹種':'Prediction unavailable / 暫無可用預測'):decision.display_name,
    candidates,score:decision.score,reasons:[...(decision.reasons||[])],
    confidence_label:'Confidence (uncalibrated model score) / 置信度（未校準模型分數）',
    notice:review?(candidates.length?'Ranked candidates, not a confirmed identification. / 以下為候選排序，並非已確認的鑑定。':'This model has no usable class scores for the input. Add an image or select a compatible model. / 此模型沒有適用於輸入的類別分數；請補充影像或選擇相容模型。'):''};
}
function decisionHTML(view){
  const escape=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const percentage=v=>Number.isFinite(v)?(v*100).toFixed(1)+'%':'—';
  let body='<div class="quiet">'+escape(view.source)+'</div><h3>'+escape(view.title)+'</h3>';
  if(view.review){
    body+='<p class="quiet">'+escape(view.notice)+'</p>';
    if(view.candidates.length)body+='<div class="quiet">'+escape(view.confidence_label)+'</div><ol class="candidate-list">'+view.candidates.map(p=>'<li><div class="candidate-line"><span>'+escape(p.name)+'</span><strong>'+percentage(p.score)+'</strong></div><div class="bar"><i style="width:'+Math.max(0,Math.min(100,p.score*100))+'%"></i></div></li>').join('')+'</ol>';
  }else body+='<div class="toprow"><span class="quiet">'+escape(view.confidence_label)+'</span><span class="score">'+percentage(view.score)+'</span></div>';
  if(view.reasons.length)body+='<p class="quiet">'+view.reasons.map(escape).join('<br>')+'</p>';
  return body;
}
if(typeof module!=='undefined')module.exports={decisionView,decisionHTML};
