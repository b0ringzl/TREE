/* Opt-in pilot; previous Hong Kong and Wuhan routes remain selectable. */
(()=>{
  const domain='hongkong_vmms_five',select=document.getElementById('domain');
  select.add(new Option('香港扩展五类 · 图像 / 点云 / 双源 · 实验',domain));
  document.querySelector('.hero p').textContent='香港通用图像、香港原三类双源、香港扩展五类实验，以及武汉配对模型。请选择对应模型范围，支持单木输入、质量提示、候选对照和结果导出。';
  const previousScope=scope;
  scope=function(){
    if(select.value!==domain){previousScope();return}
    document.getElementById('scope').textContent='扩展五类：凤凰木、石栗、银合欢、大叶合欢、南洋杉（原训练合并类）。已接入真实实验权重，支持三种输入。凤凰木 / 银合欢点云可靠性未验证；图像优先，双源冲突提示复核，融合单列展示。';
    document.getElementById('sample').innerHTML='<option value="">请选择五类演示样本</option>'+state.samples.filter(s=>s.domain===domain).map(s=>`<option value="${esc(s.id)}">${esc(s.truth)} · ${s.split==='training_example_not_test'?'训练示例（不是测试集）':'小留出示例'} · ${esc(s.pair_id)}</option>`).join('');
  };
  const panel=document.createElement('section');panel.className='notice';
  panel.innerHTML='<h3>香港扩展五类 · 已接入实验模式</h3><p>图像、点云和双源均调用真实权重。五类不是与原三类混在一起的八类统一分类器。凤凰木与银合欢暂无独立类别评估；其他三类也仅有小样本留出结果。</p><div id="hkFiveCatalog">读取五类清单…</div><p><a href="/hk-five-report" download>下载五类实验报告</a> · <a href="/api/hk-five-demo-info" target="_blank" rel="noopener">模型与示例来源</a></p>';
  document.getElementById('catalogPanel').append(panel);
  request('/api/hk-five-demo-info').then(d=>{
    document.getElementById('hkFiveCatalog').innerHTML='<table><thead><tr><th>树种</th><th>已接受视角</th><th>输入 / 评估范围</th></tr></thead><tbody>'+d.classes.map(c=>`<tr><td>${esc(c.chinese_name)}<br>${esc(c.scientific_name)}</td><td>${c.accepted_views}</td><td>图像 / 点云 / 双源<br>${c.validation_scope==='no_independent_class_evaluation'?'未独立评估':'仅小留出子集评估'}</td></tr>`).join('')+'</tbody></table>';
  }).catch(e=>document.getElementById('hkFiveCatalog').textContent='读取失败：'+e.message);
  if(new URLSearchParams(location.search).get('domain')===domain)select.value=domain;
})();
