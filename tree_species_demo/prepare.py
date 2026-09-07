"""Select classes on validation only; package quality-ranked held-out examples."""
import csv
import hashlib
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
HK = WORK / 'full_image_species_baseline/experiments/full_web_yolo11s_cls_20260906_seed1'
WHU = WORK / 'experiments/whu18_yolo11_ptv2_fusion_20260826'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def canonical(name):
    return '榕树' if name in ('Ficus microcarpa', 'Ficus benjamina') else name

def main():
    out = ROOT / 'test_data'
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((HK/'pretrain/val_predictions.csv').open(encoding='utf-8-sig')))
    actual = Counter(canonical(r['true_scientific_name']) for r in rows)
    predicted = Counter(canonical(r['predicted_scientific_name']) for r in rows)
    correct = Counter(canonical(r['true_scientific_name']) for r in rows if canonical(r['true_scientific_name']) == canonical(r['predicted_scientific_name']))
    ranked = sorted(({'scientific_name':k, 'support':v, 'precision':correct[k]/max(1,predicted[k]), 'recall':correct[k]/v, 'f1':2*correct[k]/max(1,v+predicted[k])} for k,v in actual.items() if v>=10), key=lambda r:(-r['f1'],-r['support'],r['scientific_name']))
    hk_selected = ranked[:19]
    whu_metrics = read(WHU/'fusion/run/final_metrics.json')['metrics']['val']['per_class']
    whu_selected = sorted(whu_metrics, key=lambda r:(-r['f1'],r['class_index']))[:17]
    excluded = [r for r in whu_metrics if r not in whu_selected]
    chinese = {'Ravenala madagascariensis':'旅人蕉','Araucaria columnaris(Araucaria heterophylla)':'南洋杉（原训练合并类）','Livistona chinensis':'蒲葵','Thevetia peruviana':'黄花夹竹桃','Casuarina equisetifolia':'木麻黄','Michelia x alba':'白兰','Macaranga tanarius':'血桐','Eucalyptus urophylla':'尾叶桉','Machilus velutina':'绒毛润楠','Plumeria rubra':'鸡蛋花','Elaeocarpus hainanensis':'水石榕','Podocarpus macrophyllus':'罗汉松','Lagerstroemia speciosa':'大花紫薇','Caryota mitis':'短穗鱼尾葵','Reevesia thyrsoidea':'梭罗树','Terminalia catappa':'榄仁树','Celtis sinensis':'朴树','Diospyros eriantha':'小果柿','Roystonea regia':'王棕','Platanus x acerifolia':'二球悬铃木','Cinnamomum camphora':'樟','Lagerstroemia indica':'紫薇','Osmanthus fragrans':'桂花','Koelreuteria paniculata':'栾树',"Prunus cerasifera 'atropurpurea'":'紫叶李','Zelkova serrata':'榉树','Cedrus deodara':'雪松','Ginkgo biloba':'银杏','Acer palmatum var. atropurpureum':'红枫','Prunus serrulata':'樱花','Ligustrum lucidum':'女贞','Malus x micromalus':'西府海棠','Magnolia grandiflora':'广玉兰','Metasequoia glyptostroboides':'水杉','Sophora japonica':'国槐','Acer pictum subsp. mono':'五角枫','Photinia x fraseri':'红叶石楠'}
    for domain, items in [('hongkong',hk_selected),('wuhan',whu_selected)]:
        for item in items:
            item.update(domain=domain, chinese_name=chinese.get(item['scientific_name'],item['scientific_name']), modalities=['image'] if domain=='hongkong' else ['image','point','fusion'])
    catalog = {'hongkong':hk_selected,'wuhan':whu_selected,'unknown':{'scientific_name':'Unknown','chinese_name':'未知 / 需复核','policy':'类别范围外、质量严重不足、最大分数<0.55或前两名差<0.15；阈值是demo启发式，未做独立开放集校准'},'excluded_wuhan':excluded,'selection':'Hong Kong: top 19 validation F1 among classes with >=10 samples. Wuhan: top 17 fusion validation F1 of 18 trained species. No test predictions used for selection.','overlap':sorted(set(x['scientific_name'] for x in hk_selected)&set(x['scientific_name'] for x in whu_selected))}
    write(ROOT/'catalog.json',catalog)
    samples=[]
    def image_quality(path):
        try:
            with Image.open(path) as im:
                if min(im.size)<160: return None
                grey=np.asarray(im.convert('L').resize((320,224)))
                bright=float(grey.mean()); sharp=float(cv2.Laplacian(grey,cv2.CV_64F).var())
                if not 40<bright<220 or sharp<18: return None
                return min(sharp,1200)/1200 + (1-abs(bright-128)/128)
        except Exception: return None
    def add(domain,truth,image,point=None,**extra):
        key=f'{domain}_{len(samples)+1:03d}'
        folder=out/key; folder.mkdir(exist_ok=True)
        image_dest=folder/'image.jpg'; shutil.copy2(image,image_dest)
        item={'id':key,'domain':domain,'truth':truth,'image':str(image_dest.relative_to(ROOT)).replace('\\','/'),'source_image':str(image),'split':'test','selection':'ranked by image quality, not prediction correctness',**extra}
        if point:
            dest=folder/'point.npz'; shutil.copy2(point,dest); item.update(point=str(dest.relative_to(ROOT)).replace('\\','/'),source_point=str(point))
        samples.append(item)
    test_rows=list(csv.DictReader((HK/'pretrain/test_predictions.csv').open(encoding='utf-8-sig')))
    for sp in hk_selected:
        candidates=[]
        for r in test_rows:
            if canonical(r['true_scientific_name'])!=sp['scientific_name']: continue
            quality=image_quality(r['source_path'])
            if quality is not None: candidates.append((quality,r))
        seen=set()
        for q,r in sorted(candidates,key=lambda v:-v[0]):
            if r['group_id'] in seen: continue
            add('hongkong',sp['scientific_name'],r['source_path'],group_id=r['group_id'],quality_rank_score=q)
            seen.add(r['group_id'])
            if len(seen)>=2: break
    records=read(WORK/'experiments/whu18_resnet50_20260825/dataset/manifest.json')['records']
    for sp in whu_selected:
        candidates=[]
        for r in records:
            if r['split']!='test' or r['scientific_name']!=sp['scientific_name']: continue
            if r.get('crop_visible_fraction',0)<0.97 or r.get('sampled_unique_point_count',0)<8000: continue
            quality=image_quality(r['image_path'])
            if quality is not None: candidates.append((quality,r))
        selected=sorted(candidates,key=lambda v:(-v[0],v[1]['sample_key']))[:2]
        selected_keys={r['sample_key'] for _,r in selected}
        if len(selected)<2:
            fallback=[]
            for r in records:
                if r['split']=='test' and r['scientific_name']==sp['scientific_name'] and r['sample_key'] not in selected_keys:
                    q=image_quality(r['image_path'])
                    fallback.append(((q or 0)+r.get('crop_visible_fraction',0),r))
            selected.extend(sorted(fallback,key=lambda v:-v[0])[:2-len(selected)])
        for q,r in selected:
            add('wuhan',sp['scientific_name'],r['image_path'],r['point_path'],sample_key=r['sample_key'],crop_visible_fraction=r['crop_visible_fraction'],sampled_unique_point_count=r['sampled_unique_point_count'],quality_rank_score=q,paired=True,quality_caution=r['sample_key'] not in selected_keys)
    # Include excluded classes to exercise unknown handling; not guaranteed rejection.
    excluded_names={x['scientific_name'] for x in hk_selected}
    for r in test_rows:
        if canonical(r['true_scientific_name']) not in excluded_names and image_quality(r['source_path']) is not None:
            add('hongkong','Unknown',r['source_path'],source_species=r['true_scientific_name'],purpose='out_of_catalog_probe')
            break
    for r in records:
        if r['split']=='test' and r['scientific_name']==excluded[0]['scientific_name']:
            add('wuhan','Unknown',r['image_path'],r['point_path'],source_species=r['scientific_name'],purpose='out_of_catalog_probe',paired=True); break
    for kind in ['dark','blur']:
        source=samples[0]; folder=out/f'quality_{kind}'; folder.mkdir(exist_ok=True)
        im=Image.open(ROOT/source['image']).convert('RGB')
        im=Image.fromarray((np.asarray(im)*0.025).astype(np.uint8)) if kind=='dark' else im.resize((16,16)).resize(im.size).filter(ImageFilter.GaussianBlur(18))
        im.save(folder/'image.jpg')
        samples.append({'id':f'quality_{kind}','domain':'hongkong','truth':'Unknown','image':f'test_data/quality_{kind}/image.jpg','purpose':'synthetic_quality_probe','split':'derived_test'})
    for sample in samples:
        sample['image_sha256']=hashlib.sha256((ROOT/sample['image']).read_bytes()).hexdigest()
    write(ROOT/'samples.json',samples)
    write(out/'manifest.json',samples)
    lines=['# 树种清单与测试说明','','香港 19 个已知类 + 1 个未知入口；武汉从已有 18 类模型选 17 类。选择仅依据验证集，保留完整输出概率，排除类不会重新归一化成已知类。','','香港为网络图像域验证结果，不能当作香港街景实测精度；南洋杉沿用原模型合并类别。武汉配对模型要求同一棵树、Z轴向上、单木点云。香港没有这19类对应的点云分类权重。','','未知是拒识机制，不是已训练的第37个输出神经元。0.55 / 0.15 阈值尚未经独立开放集验证。','','武汉排除：'+excluded[0]['scientific_name']+'，融合验证 F1 最低；原权重完整保留。','','|区域|中文参考名|原训练学名|验证F1|验证样本|支持输入|','|---|---|---|---:|---:|---|']
    for s in hk_selected+whu_selected:
        lines.append(f"|{s['domain']}|{s['chinese_name']}|{s['scientific_name']}|{s['f1']:.3f}|{s['support']}|{','.join(s['modalities'])}|")
    lines += ['|共用|未知|Unknown|不适用|未独立校准|任意输入拒识|','','测试包共 '+str(len(samples))+' 个样本条目，正常已知类每类最多2例，按清晰度、曝光和配对可见比例筛选，不按预测是否正确挑选。这是演示测试子集，不能据此宣称整体精度。完整来源、原始划分和图像哈希见 samples.json。','', '数据保持本地测试使用；原始数据授权不因本测试包改变。']
    (ROOT/'树种列表与测试说明.md').write_text('\n'.join(lines),encoding='utf-8')
    with zipfile.ZipFile(ROOT/'测试数据集.zip','w',zipfile.ZIP_DEFLATED) as archive:
        archive.write(out/'manifest.json','test_data/manifest.json')
        for sample in samples:
            for modality in ('image','point'):
                if sample.get(modality): archive.write(ROOT/sample[modality],sample[modality])
    print(json.dumps({'hk':len(hk_selected),'whu':len(whu_selected),'excluded_whu':excluded[0]['scientific_name'],'samples':len(samples)},ensure_ascii=False))

if __name__=='__main__': main()
