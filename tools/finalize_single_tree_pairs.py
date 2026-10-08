"""Export agent visual screening, never acceptance/training labels."""
import json, csv, sys
from pathlib import Path
import numpy as np, laspy
from PIL import Image, ImageDraw
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'hk-tree-vmms/tools')]
from rebuild_banyan_coverage_preview import sample_ids
from build_hk_lidar_pairs import project
OUT=ROOT/'outputs/single_tree_panorama_selection_v1'
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def main():
 rows=json.loads((OUT/'manifest.json').read_text(encoding='utf8'))
 coords={c['stream_id']+'__'+c['frame_id']:c for c in csv.DictReader((ROOT/'hk-tree-vmms/outputs/coordinates/frame_coordinates.csv').open(encoding='utf-8-sig'))}
 preferred={'01_TreeID_13':1,'01_TreeID_14':1,'01_TreeID_15':0,'01_TreeID_18':3,'04_TreeID_5':1}
 follow={f'01_TreeID_{i}' for i in [11,12,16,17,21,23]}|{'03_TreeID_21','04_TotalOnly_TreeID_13'}
 reasons01={1:'大樹下部與懸垂結構；影像過暗，不能確認整冠覆蓋。',2:'冠幹分離，車輛遮擋；不作高品質整株。',3:'投影主要落在大樹較低區域；暗冠與整冠缺失待區分。',4:'可見幹，但上部大冠未被充分覆蓋；影像暗。',5:'枝冠與鄰樹重疊，單木邊界不可靠。',6:'寬冠與多條下垂結構，暗區及鄰冠重疊。',7:'投影僅對應大樹的部分枝冠，暗區明顯。',8:'細小稀疏株，遠距離像素不足並有車輛遮擋。',9:'大冠有部分覆蓋，但幹基與周邊枝冠歸屬不足。',10:'規則豎條與懸垂結構，缺少正常整株冠幹形態。',11:'幹可追蹤，但冠部遮擋與截斷仍需複核。',12:'冠幹形態較好，近鄰樹重疊及投影幹偏移待複核。',13:'冠幹連續，街旁小樹可定位，多視角可追蹤；細枝邊界及幹投影偏移仍需核准。',14:'獨立幹及主要冠形可辨，多角度可定位；鄰冠相接及投影偏移需核准。',15:'幹至主要冠部連續，影像可見分枝與花葉；右側枝團歸屬及偏移仍需核准。',16:'幹較完整，但投影顯示上部冠形可能缺失，不能只憑三視圖接受。',17:'下部被其他植物遮擋，分離冠團與本株的關係不清楚。',18:'幹與主要冠部保留，兩視角可定位；低部遮擋及鄰冠邊界需核准，歷史石栗標籤不自動重確證。',19:'上方與下方枝冠交疊，無法確認單株整冠。',20:'主要是大冠局部，基部與整株覆蓋不充分。',21:'小樹冠枝清楚，但遠距離細節不足，下幹可能截斷。',22:'一個視角被亭蓋遮擋，另一視角也不能證明完整單株。',23:'冠幹形態有潛力，影像解析度及車輛遮擋限制樹種判讀。'}
 special03={21:'幼樹冠形可定位，但點雲明確包含支撐木架；清除前不作純樹形樣本。',28:'僅 345 點的微小碎片，不是完整單木。',29:'主幹與氣根具有局部研究價值，但整冠明顯缺失。',31:'幹及分叉可見，但大冠未完整覆蓋。',38:'局部灌叢／細幹與背景混雜，無可靠獨立樹冠。',26:'細長枝幹碎片，無完整冠幹。',34:'局部粗枝及細根，非完整單木。',36:'上下部斷離，且對應大樹的部分幹冠。'}
 reasons04={5:'主要樹冠及一段主幹連續，獨立輪廓較清楚；基部遮擋、冠頂投影偏移仍未解決。',7:'稀疏橫向枝段，並非完整冠幹。',8:'主要冠層，與鄰樹重疊，幹基覆蓋不足。',9:'稀疏掃描條帶，無完整冠形。',10:'碎片化冠層，幹基被構築物遮擋。',11:'冠層與鄰木混接，幹基遮擋。',12:'寬冠下多條豎直結構，整株歸屬不清，影像暗。',17:'稀疏掃描條帶及局部枝葉，非整株。',18:'僅 246 點的小碎片，非完整單木。',19:'只對應大冠部分区域，整冠未覆蓋。',20:'小枝葉片段與交通設施重疊，非可靠整株。'}
 for r in rows:
  ident=r['id'];sid=ident[:2];n=int(ident.split('_')[-1]);total='TotalOnly' in ident
  if sid=='01':reason=reasons01[n]
  elif sid=='03':reason=special03.get(n,'投影對應大榕樹的局部枝冠／氣根，並非獨立完整單木；不按 ID 計作一株。')
  elif total:
   reason='立面覆生植被／連片枝葉，不能確認單株根幹冠；總文件有此 ID，但缺少獨立交付文件。'
   if n==13:reason='冠形與主幹較集中，有補充潛力；影像暗、下幹遮擋，且缺少獨立交付文件，需確認省略原因。'
   if n==15:reason='狹長冠形周邊混有其他樹冠，基部遮擋，可能混株；亦缺獨立交付文件。'
  else:reason=reasons04[n]
  status='priority_candidate' if ident in preferred else 'recoverable_candidate' if ident in follow else 'not_whole_tree_priority'
  valid=[v for v in r['views'] if 'image' in v];best=preferred.get(ident,r['display_views'][0]);v=next(v for v in valid if v['index']==best)
  r.update(screening_status=status,visual_reason=reason,best_view=best,reviewer='assistant_visual_screening_2026-09-12',review_method='all 82 contact entries inspected; five priority and four additional candidates examined at full image resolution',human_acceptance='not_requested_or_recorded_by_this_audit',strict_high_quality_confirmed=False,training_eligible=False,calibration_status='empirical yaw=180; zero lever arm; fine extrinsics unavailable; visible offsets not corrected',species_status='prior evidence retained separately; not confirmed by this screening')
  f=laspy.read(r['path']);xyz=np.column_stack([f.x,f.y,f.z]);q=xyz[sample_ids(xyz,2048)];center=np.median(q,axis=0)
  dump(OUT/ident/'points.json',dict(center_local_xyz=center.tolist(),points=np.round(q-center,3).tolist(),units='m',sample_count=len(q),is_measured_points=True))
  uv,_=project(q,coords[v['frame_key']])
  with Image.open(v['panorama']) as src:im=src.convert('RGB');im.thumbnail((2000,1000))
  dr=ImageDraw.Draw(im)
  for a,b in uv:dr.ellipse((a*im.width-1,b*im.height-1,a*im.width+1,b*im.height+1),fill='#ff35ce')
  dr.rectangle((5,5,280,28),fill='white');dr.text((10,10),ident+' - empirical projection',fill='black');im.save(OUT/ident/'panorama_overview.jpg',quality=92)
  r['panorama_overview']=str(OUT/ident/'panorama_overview.jpg')
  r['geometry_preview']=r.get('geometry_preview') or str(ROOT/'outputs/single_tree_delivery_audit_v1'/sid/(ident[3:]+'.jpg'))
  dump(OUT/ident/'screening.json',r)
 dump(OUT/'screened_manifest.json',rows)
 dump(OUT/'priority_pairs.json',[r for r in rows if r['screening_status']=='priority_candidate'])
 dump(OUT/'visual_decisions.json',[{k:r[k] for k in ['id','screening_status','visual_reason','best_view','strict_high_quality_confirmed','training_eligible','human_acceptance']} for r in rows])
 counts={s:sum(r['screening_status']==s for r in rows) for s in ['priority_candidate','recoverable_candidate','not_whole_tree_priority']}
 dump(OUT/'screening_summary.json',dict(instances=len(rows),panorama_views=sum(len([v for v in r['views'] if 'image' in v]) for r in rows),counts=counts,strict_high_quality_confirmed=0,google_imagery_used=False,raw_files_modified=False,training_performed=False))
 lines=['# 三场景点云＋全景配对筛查','', '范围：01、03、04 的 72 个独立 LAS，另检查 04 总文件中未单独导出的 10 个 ID。不是全部香港数据，也不是 82 株独立树。','',f'全部 82 个 ID 已生成多视角 VMMS 投影与追溯记录。优选候选 {counts["priority_candidate"]} 组，待补救候选 {counts["recoverable_candidate"]} 组，暂不列为整株优选 {counts["not_whole_tree_priority"]} 个 ID。','', '**严格通过的高质量整株配对目前不能宣称已确认。** 优选是相对质量筛选，不是人工接受、树种真值或经标定的完整性认证。现有投影沿用经验 yaw=180°、零杆臂；细外参缺失，部分干冠存在可见偏移。不能仅根据偏移断言点云截断，亦不能将投影命中比例充当配准精度。','', '## 优选候选','', '| ID | 原点数 | 最佳帧 | 结论 |','|---|---:|---|---|']
 for r in rows:
  if r['screening_status']=='priority_candidate':
   v=next(v for v in r['views'] if v['index']==r['best_view']);lines.append(f"| {r['id']} | {r['points']:,} | {v['frame_key']} | {r['visual_reason']} |")
 lines+=['','## 定位与使用','', '查看页面 http://127.0.0.1:8046/：默认显示优选候选，可切换全部 ID，查看全景定位、原图透视裁片、点云叠加、旋转 2048 个实测点和源文件路径。每组保留 3–4 个视角；视角按目标方位间隔至少 12°选取，不将邻帧数量当作独立样本数量。','', '每个相机坐标仅代表该 VMMS 全景拍摄位置，不能冒称树木实测经纬度。单木中心是局部坐标估计，不是独立测量的树根。Google 街景入口仅是对应相机附近的人工检索入口；本轮未使用 Google 图像作为核验依据。','', '## 主要问题','', '01 的较小行道树更有整株配对潜力；03 的许多分割 ID 是大榕树部件，不可独立计树；04 遗漏的 10 个独立文件多数对应立面／连续植被，ID 13 有补充潜力但须确认省略原因。','', '下一步应先核准 5 组优选的单木边界和对应关系，并解决可见投影偏移；需要细部重新确认的树种不得从场景标题继承。主干偏移可能来自外参或时序等因素，本轮没有证明其唯一原因。','', '## 数据安全','', '原始 LAS、历史验收和部署权重均未修改；未训练。总文件的 10 个 ID 仅在本输出目录生成诊断副本，不视为自动恢复的已接受样本。所有未明确接受条目 training_eligible=false。优选表包含先前使用过的实例，不能用来宣称新的独立测试精度。','', '## 完整逐实例结论','']
 lines.extend(f"- {r['id']} — {r['screening_status']}：{r['visual_reason']}" for r in rows)
 (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')
 print(counts,flush=True)
if __name__=='__main__':main()
