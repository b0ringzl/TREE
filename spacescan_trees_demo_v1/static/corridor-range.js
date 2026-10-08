(function(root){
  'use strict';
  // One-based positions in the naturally sorted, valid panorama list.
  function selection(total,start,count,step=1){
    total=Math.floor(Number(total));
    if(!Number.isFinite(total)||total<2)throw Error('At least two panoramas are required');
    start=Math.max(1,Math.min(total-1,Math.floor(Number(start))||1));
    step=Math.max(1,Math.min(50,Math.floor(Number(step))||1));
    step=Math.min(step,total-start);
    const available=Math.floor((total-start)/step)+1;
    count=Math.max(2,Math.min(300,available,Math.floor(Number(count))||300));
    return {start,end:start+(count-1)*step,count,step,maxCount:Math.min(300,available),total};
  }
  function ending(total,start,end,step=1){
    const initial=selection(total,start,300,step);
    end=Math.max(initial.start+initial.step,Math.min(total,Math.floor(Number(end))||initial.end));
    return selection(total,initial.start,Math.floor((end-initial.start)/initial.step)+1,initial.step);
  }
  function move(total,current,action){
    const count=Math.min(300,current.count);
    let start=current.start;
    if(action==='first')start=1;
    if(action==='previous')start=Math.max(1,start-count);
    if(action==='next')start=Math.min(total-1,start+count);
    if(action==='last')start=Math.max(1,total-count+1);
    return selection(total,start,count,1);
  }
  const api={selection,ending,move};
  if(typeof module!=='undefined'&&module.exports)module.exports=api;
  else root.CorridorRange=api;
})(typeof window!=='undefined'?window:globalThis);
