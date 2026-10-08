"""Inference-only demonstration delivery. Copies originals; never fits/deploys."""
import sys,json,uuid,shutil,re,html
from pathlib import Path
import numpy as np,torch,joblib,laspy
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'tree_species_demo')]
from training_safety import sha256
from tree_species_teacher_demo.engine import Engine,decode,collapse_logits
from spacescan_trees_demo_v1.server import prepare
from hk_three_species import Features,POINT
from rebuild_banyan_coverage_preview import sample_ids
BASE=ROOT/'outputs/confirmed_point_fusion_20260913_v1';PAIR=ROOT/'outputs/point_assist_case_selection_20260913_v1';LEGACY=ROOT/'outputs/bauhinia_ptv3_fewshot_20260912_v1';SURVEY=ROOT/'outputs/vmms20m_teacher_20260909_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(p,v):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def canon(s):return 'Ficus microcarpa' if s.startswith('Ficus microcarpa /') else s

def image_scores(engine,path):
 data,w,h=prepare(Path(path).read_bytes());pixels,quality,_=decode(data)
 with torch.inference_mode():
  emb=engine.model.encode_image(((pixels[None].cuda()-engine.mean)/engine.std).to(engine.dtype),normalize=True).float()
  logits=emb@engine.bank.T*engine.model.logit_scale.exp().float();extra=emb@engine.extra_bank.T*engine.model.logit_scale.exp().float()
  logits=torch.cat([collapse_logits(logits,engine.classes['groups']),extra],-1)/engine.choice['calibration_temperature']
 return logits[0].cpu().numpy(),dict(width=w,height=h,**{'quality':quality})

def point_scores(path,model):
 bundle=joblib.load(model);assert sha256(POINT)==bundle['encoder_sha256']
 path=Path(path)
 if path.suffix.lower()=='.las':
  las=laspy.read(path);x=np.c_[las.x,las.y,las.z]
 else:
  with np.load(path) as f:x=f['xyz']
 x=np.unique(x[np.isfinite(x).all(1)],axis=0);m=x[sample_ids(x,2048)].astype('float32');c=(x.min(0)+x.max(0))/2;scale=max(float(np.linalg.norm(x-c,axis=1).max()),1e-6);p=((m-c)/scale).astype('float32')
 enc=Features('cuda');fs=[]
 for degree in [0,90,180,270]:
  a=np.deg2rad(degree);r=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]],dtype='float32')
  with torch.inference_mode():fs.append(enc.encode_point(p@r.T))
 f=np.mean(fs,0)[None];f=f/np.maximum(np.linalg.norm(f,axis=1,keepdims=True),1e-8)*10
 return bundle['head'].predict_proba(f)[0],bundle,m

def main():
 torch.set_num_threads(4);torch.manual_seed(20260913)
 out=ROOT/'outputs'/('SpaceScan_Trees_Demo_Files_20260913_'+uuid.uuid4().hex[:8]);out.mkdir(exist_ok=False);print('OUTPUT',out,flush=True)
 for name in ['01_image_gt80_verified','01b_image_gt80_unverified','02_cloud_assist_gain','03_evidence','04_source_panoramas']:(out/name).mkdir()
 copies=[];panos={};image_cache={};scored=[]
 def copy(src,dest):
  src=Path(src);dest.parent.mkdir(parents=True,exist_ok=True);digest=sha256(src)
  if not dest.exists():shutil.copy2(src,dest)
  assert sha256(dest)==digest
  copies.append(dict(source=str(src.resolve()),package=dest.relative_to(out).as_posix(),sha256=digest,bytes=dest.stat().st_size));return dest.relative_to(out).as_posix()
 frames=read(SURVEY/'frames.json')
 def panorama(frame_key=None,path=None):
  if not path:
   if frame_key not in frames:return None
   path=ROOT.parent/'vmms'/frames[frame_key]['source_image_relpath']
  path=Path(path);assert path.is_file(),path
  if str(path) not in panos:panos[str(path)]=copy(path,out/'04_source_panoramas'/path.name)
  return panos[str(path)]
 engine=Engine();engine.load();print('TEACHER READY',flush=True)
 save(out/'03_evidence/teacher.json',dict(encoder_sha256=engine.encoder_sha256,labels=engine.labels,choice=engine.choice,preprocessing='Exact main Demo prepare() then teacher decode(); no enhancement',point_fusion_in_main_demo=False))
 catalog=read(ROOT/'outputs/hk_point_assist_20260911_v1/catalog.json');pairs=read(PAIR/'manifest.json');candidates=[]
 for r in catalog:
  if r['current_accepted'] and r['review_decision']=='accept':
   frame=r['id'].rsplit('__',1)[0];candidates.append(dict(id=r['id'],image=r['asset_paths']['image'],truth=r['species'],confirmed=True,frame_key=frame,evidence=dict(key=r['key'],review=r['review'],catalog_source='hk_point_assist_20260911_v1/catalog.json')))
 for r in pairs:
  for v in r['views']:candidates.append(dict(id=r['id']+'_view'+str(v['index']),image=v['image'],truth=r['species'],confirmed=True,frame_key=v.get('frame_key'),panorama=v.get('panorama'),evidence=dict(pair_id=r['id'],label=r['label'],pair_confirmation=str(BASE/'pair_confirmation.json'))))
 gate={r['id']:r for r in read(SURVEY/'object_gate_20260910_reviewed_v4/decisions.json')}
 for f in (SURVEY/'anchors').glob('*.json'):
  anchor=read(f)
  for r in anchor.get('instances',[]):
   if not r.get('original_top3') or r['original_top3'][0]['score']<=.8:continue
   if gate.get(r['id'],{}).get('decision') not in ['retain_unverified','retain_verified']:continue
   candidates.append(dict(id=r['id'],image=r['crop'],truth=None,confirmed=False,frame_key=anchor['frame_key'],evidence=dict(source_anchor=str(f),historical_top3=r['original_top3'],decision=gate.get(r['id'],{}).get('decision'))))
 seen=set();kept=[]
 for n,r in enumerate(candidates):
  path=Path(r['image']);digest=sha256(path)
  if digest in seen:continue
  seen.add(digest);logits,quality=image_scores(engine,path);image_cache[str(path)]=logits
  exp=np.exp(logits-logits.max());probs=exp/exp.sum();order=np.argsort(-probs);top=[dict(species=engine.labels[i],score=float(probs[i])) for i in order[:3]]
  result=dict(**r,sha256=digest,top3=top,quality=quality,prediction_matches_label=bool(r['truth'] and canon(top[0]['species'])==canon(r['truth'])),selection='pending')
  if top[0]['score']>.8 and (not r['confirmed'] or result['prediction_matches_label']):
   bucket='01_image_gt80_verified' if r['confirmed'] else '01b_image_gt80_unverified';folder=out/bucket/r['id'];folder.mkdir();result['package_image']=copy(path,folder/('image'+path.suffix.lower()));result['package_panorama']=panorama(r.get('frame_key'),r.get('panorama'));result['selection']=bucket;save(folder/'record.json',result);kept.append(result)
  else:result['selection']='excluded_below_threshold_or_label_mismatch'
  scored.append(result)
  if n%10==0:print('IMAGE',n+1,len(candidates),'kept',len(kept),flush=True)
 save(out/'03_evidence/all_screened_images.json',scored)
 # Repeat the two historical improvements from actual measured cloud inputs, not stored scores.
 predictions=read(BASE/'predictions.json');oldrows=read(LEGACY/'manifest.json');classes=read(BASE/'protocol.json')['classes'];indices=[next(i for i,n in enumerate(engine.labels) if canon(n)==c) for c in classes];gains=[]
 for old in predictions:
  if old['branches']['image']['correct'] or not old['branches']['fusion']['correct']:continue
  r=next(r for r in pairs if r['id']==old['id']);v=next(v for v in r['views'] if v['index']==old['view']);pr=r['point_results'][0] if r['case']=='bauhinia' else next(p for p in r['point_results'] if p['fold']=='fold_1');model=Path(pr['model'])/'model.joblib';mr=next(t for t in oldrows if t['key']==r['key']);cfg=read(model.parent/'config.json');assert r['key'] not in cfg['training_keys'];assert sha256(mr['source'])==mr['source_sha256']
  pp,bundle,m=point_scores(mr['source'],model);assert bundle['classes']==classes
  expected=np.array([pr['scores'][c] for c in classes]);assert np.max(abs(pp-expected))<1e-3,'Point replay mismatch'
  logits=image_cache.get(v['image'])
  if logits is None:logits,_=image_scores(engine,v['image'])
  lp=logits[indices];ip=np.exp(lp-lp.max());ip/=ip.sum();fp=.5*ip+.5*pp
  record=dict(id=r['id'],view=v['index'],truth=r['species'],classes=classes,branches={k:dict(prediction=classes[int(p.argmax())],scores={c:float(z) for c,z in zip(classes,p)},correct=classes[int(p.argmax())]==r['species']) for k,p in [('image',ip),('point',pp),('fusion',fp)]},point_replay_max_error=float(np.max(abs(pp-expected))),held_out_tree_not_in_training=True,fusion='Fixed 0.5 image + 0.5 point within six classes',main_demo_fusion_enabled=False,limitations=['Selected previously inspected case; not blind evidence','No metric-height feature; cannot attribute gain to height','Not proof fusion beats point-only','Main 148-class Demo does not perform this fusion'],source_record=r,point_model_source=str(model))
  assert not record['branches']['image']['correct'] and record['branches']['fusion']['correct'],'Historical gain did not reproduce'
  folder=out/'02_cloud_assist_gain'/(r['id']+'_view'+str(v['index']));folder.mkdir()
  record['image']=copy(v['image'],folder/'image.jpg');record['cloud_original_single_tree']=copy(r['cloud'],folder/('original_single_tree'+Path(r['cloud']).suffix));record['cloud_exact_model_input_full']=copy(mr['source'],folder/('cloud_for_replay'+Path(mr['source']).suffix));record['point_head']=copy(model,folder/'point_head.joblib');record['model_preprocessed_archive']=copy(LEGACY/mr['derived'],folder/'model_input_2048_archive.npz')
  np.save(folder/'model_input_metric_2048.npy',m);record['metric_sample_derived']='model_input_metric_2048.npy';record['panorama']=panorama(v.get('frame_key') or r['id'].rsplit('__',1)[0],v.get('panorama'))
  if v.get('overlay'):record['overlay_reference_only']=copy(v['overlay'],folder/'projection_reference.jpg')
  save(folder/'comparison.json',record);gains.append(record);print('GAIN VERIFIED',r['id'],v['index'],flush=True)
 for name in ['REPORT.md','protocol.json','metrics.json','predictions.json','tree_results.json','pair_confirmation.json']:copy(BASE/name,out/'03_evidence'/('original_audit_'+name))
 save(out/'03_evidence/accepted_catalog_snapshot.json',catalog);save(out/'03_evidence/confirmed_pair_manifest.json',pairs)
 save(out/'file_provenance.json',copies);save(out/'manifest.json',dict(high_confidence_images=kept,gain_pairs=gains,scope='All passing images within 95 accepted catalog rows, confirmed pair views, and 66 historical high-score survey candidates. Not an exhaustive rescore of all 799 survey crops.',no_training=True,no_deployment=True))
 verified=[r for r in kept if r['confirmed']];unverified=[r for r in kept if not r['confirmed']]
 info=dict(output=str(out),verified_high_confidence=len(verified),unverified_high_confidence=len(unverified),gain_pairs=len(gains),copied_original_files=len(copies),source_panoramas=len(panos),all_copy_hashes_verified=True)
 save(out/'summary.json',info)
 lines=['# SpaceScan Trees 演示文件','',f'主推荐：{len(verified)} 张人工接受标签一致且当前教师分数 >80% 的原切片；另外 {len(unverified)} 张高分但未确认树种的候选分开放置；{len(gains)} 对点云辅助纠错样例。','',
 '单张图的 80% 是未校准模型分数，不是识别准确率。主推荐集合是筛选过的演示材料，不可用其正确比例声称整体模型精度。','',
 '## 目录','',
 '- 01_image_gt80_verified：优先演示；image.jpg 是未增强、未重编码的已有全景切片，record.json 有人工标签与本轮预测。',
 '- 01b_image_gt80_unverified：当前分数 >80% 但没有可靠人工树种真值；请先核对，不作为准确率证据。',
 '- 02_cloud_assist_gain：两例固定六类的真实点云重新推理对照。image.jpg 为输入切片，cloud_for_replay 为重现对照用的原有全量单木文件；original_single_tree 保留验收时的原单木文件；model_input_metric_2048.npy 为派生的米制轻量输入。',
 '- 04_source_panoramas：上述样例对应的来源全景原图，按文件去重保存；各 record 的路径指向这里。',
 '- 03_evidence：完整原回归集的成功、失败和固定方法，以及本次筛选记录，避免只展示成功例却隐去分母。','',
 '## 怎么演示','',
 '单源：在主 Demo 选择 Image only，上传 01 中的 image 文件，图像类型选普通照片、必要时选树冠／细节直接识别以避免再次截错对象。当前 8043 服务未运行，本次未为打包修改服务或权重。','',
 '双源：主 Demo 目前 fusion_used=false，上传双源不会自动得到本报告的融合增益。请明确称为“离线固定六类点云辅助对照”，不能宣称是 148 类在线融合功能。每对 comparison.json 保存了从原有全量点云实际重算的结果和留出检查；点云头文件只用于本机实验，不含完整编码器环境。','',
 '本批既有 7 株、16 视角回归集，单视角图像 12/16 → 融合 14/16；按树平均 5/7 → 7/7。两例纠错为洋紫荆 TreeID_13 view_0 和狐尾椰子 002153。不能外推到新场景，也不能归因于树高（模型使用尺度归一化、无米制高度输入）。','',
 '源全景及已有切片／单木文件均按字节复制并 SHA-256 校验；没有把预处理输出冒充原图。不是整段 LiDAR 采集地图的备份；完整地图路径保留在来源记录中。数据仅为本地演示整理，权利与署名沿用原来源，未授权公开发布。原模型、原文件、人工记录均未修改。']
 (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
 cards=[]
 for r in kept:
  title=html.escape(r['id']);img=html.escape(r['package_image']);pred=html.escape(r['top3'][0]['species']);cards.append(f'<article><a href="{img}"><img src="{img}" loading="lazy"></a><p>{title}</p><b>{pred} · {r["top3"][0]["score"]:.1%}</b><p>{"Accepted-label match / 與接受標籤一致" if r["confirmed"] else "Unverified / 樹種未確認"}</p></article>')
 (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>SpaceScan Trees Demo Files</title><style>body{font:16px system-ui;margin:32px;background:#f5f7f6}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:20px}article{background:white;padding:16px;overflow-wrap:anywhere}img{width:100%;height:240px;object-fit:contain}a{color:#185d4b}</style><h1>SpaceScan Trees · 演示文件</h1><p>Read README.md first / 請先閱讀 README.md。分數不是準確率；雙源增益為離線六類對照。</p><main>'+''.join(cards)+'</main>',encoding='utf8')
 print('SUMMARY',json.dumps(info,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
