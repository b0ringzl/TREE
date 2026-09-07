/* Experimental HK pairing is a separate route; existing regional weights stay intact. */
(()=>{
  const select=document.getElementById('domain');
  select.add(new Option('香港街景 · 榕树 / 蒲葵 / 狐尾椰子 · 双源实验','hongkong_vmms'));
  const nav=document.querySelector('.tabs');const link=document.createElement('button');
  link.textContent='香港点云验收';link.onclick=()=>location.href='/hk-pairs';nav.append(link);
  document.querySelector('.hero p').textContent='香港通用19类图像模型、香港街景三类双源实验，以及武汉17类配对模型。香港街景实验优先覆盖榕树、蒲葵、狐尾椰子；候选点云需单独验收。';
  const previousScope=scope;
  scope=function(){
    if(select.value!=='hongkong_vmms'){previousScope();if(select.value==='hongkong')document.getElementById('scope').textContent='香港通用19类图像模型未配备统一的点云分类头。榕树、蒲葵、狐尾椰子的点云/融合请切换到“香港街景三类双源实验”。';return}
    document.getElementById('scope').textContent='实验范围：榕树、蒲葵、狐尾椰子 + 启发式未知。使用YOLO与WHU PTv2冻结特征训练香港分类/融合头；点云对应尚未人工验收，分数不代表真实精度。建议输入2048个独立点、Z轴向上的完整单木。';
    document.getElementById('sample').innerHTML='<option value="">请选择三类实验样本</option>'+state.samples.filter(s=>s.domain==='hongkong_vmms').map(s=>`<option value="${esc(s.id)}">${esc(s.truth)} · ${esc(s.pair_id)} · 待验收</option>`).join('');
  };
  const box=document.createElement('div');box.className='notice';box.textContent='香港街景补充模块：榕树（细叶榕/垂叶榕合并）、蒲葵（Livistona chinensis）、狐尾椰子（Wodyetia bifurcata）。三类图像/点云/融合为隔离实验，未取得人工验收精度；请勿把通用19类的网络图像F1套用到这里。';
  document.getElementById('catalogPanel').append(box);
  const downloads=document.createElement('p');downloads.className='links';
  for(const [name,title] of [['香港三类_双源测试包.zip','下载香港三类双源测试包'],['香港三类双源交付说明.md','下载清洗与训练说明']]){const a=document.createElement('a');a.href='/hk-pair-download/'+encodeURIComponent(name);a.textContent=title;a.download=name;downloads.append(a)}
  document.getElementById('catalogPanel').append(downloads);
})();
