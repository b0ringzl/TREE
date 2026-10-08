"""Post-fit safety, review, and inference reproduction checks; no fitting."""
import io
import json
import os
import sqlite3
from pathlib import Path
import numpy as np
import requests
from PIL import Image,ImageOps
from training_safety import verify_backup,sha256

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'outputs/SSTv1_incremental_20260914_v1'
DATA=ROOT/'external_datasets/sstv1_incremental_20260914_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    done=read(RUN/'COMPLETE.json');contract=read(RUN/'base_contract.json');accepted=read(RUN/'accepted_new_manifest.json')
    with sqlite3.connect((DATA/'human_review.sqlite3').as_uri()+'?mode=ro',uri=True) as db:
        current={r[0]:(r[1],r[2]) for r in db.execute('SELECT id,sha256,status FROM decisions')}
    assert all(current[r['id']]==(r['sha256'],'accepted') for r in accepted),'Accepted data changed during fitting'
    backup_bytes=0
    for member,base in zip(done['members'],contract['initial_heads']):
        directory=Path(member['backup']);manifest=verify_backup(directory)
        assert sha256(member['path'])==member['sha256'] and member['sha256']!=base['sha256']
        for f in manifest['files']:
            assert not os.path.samefile(f['source'],directory/f['backup']),'Backup aliases original'
            backup_bytes+=f['bytes']
    for phase in ('old_test','new_test','field_regression'):
        with np.load(RUN/f'{phase}_predictions.npz') as data:
            assert data['baseline'].shape==data['candidate'].shape
            assert data['baseline'].shape[1]==148
            for key in ('baseline','candidate'):
                assert np.isfinite(data[key]).all()
                np.testing.assert_allclose(data[key].sum(1),1,atol=1e-5)
    smoke=read(RUN/'inference_smoke.json');image=Path(smoke['image'])
    rows=read(RUN/'new_test_evaluation_rows.json');index=next(i for i,r in enumerate(rows) if Path(r['source_image'])==image)
    with np.load(RUN/'new_test_predictions.npz') as pred:
        errors={name:float(np.max(np.abs(np.array(smoke[key])-pred[column][index]))) for name,key,column in
            [('baseline_cache_reproduction','baseline_distribution','baseline'),('candidate_cache_reproduction','SSTv1_distribution','candidate')]}
        assert np.argmax(smoke['SSTv1_distribution'])==pred['targets'][index]
    # BF16 image encoding is batch-shape sensitive. Verify deployment-shaped
    # single-image execution separately instead of relaxing a cache tolerance.
    single=read(RUN/'single_image_holdout.json')
    assert single['batch_vs_single']['baseline_top1_changes']==0
    assert single['batch_vs_single']['candidate_top1_changes']==0
    with np.load(RUN/'single_image_holdout_predictions.npz') as pred:
        np.testing.assert_allclose(smoke['baseline_distribution'],pred['baseline'][index],atol=1e-5)
        np.testing.assert_allclose(smoke['SSTv1_distribution'],pred['candidate'][index],atol=1e-5)
    # Same request path as the image Demo, without updating its model.
    with Image.open(image) as im:
        im=ImageOps.exif_transpose(im).convert('RGB');edge=min(1024,max(im.size))
        im=ImageOps.pad(im,(edge,edge),method=Image.Resampling.BICUBIC,color=(128,128,128))
        buf=io.BytesIO();im.save(buf,format='JPEG',quality=95)
    response=requests.post('http://127.0.0.1:8039/api/predict',files={'image':('smoke.jpg',buf.getvalue(),'image/jpeg')},data={'mode':'adapted'},timeout=60)
    response.raise_for_status();live=response.json();save(RUN/'live_baseline_smoke.json',live)
    livep=np.array([live['adapted_distribution'][s] for s in smoke['labels']])
    errors['live_baseline_reproduction']=float(np.max(np.abs(livep-smoke['baseline_distribution'])))
    assert errors['live_baseline_reproduction']<.005
    assert requests.get('http://127.0.0.1:8043/',timeout=15).status_code==200
    result={'all_checks_passed':True,'review_still_accepted':len(accepted),'physical_backup_bytes':backup_bytes,
        'selected_members_changed':3,'probability_reproduction_errors':errors,'production_baseline_unchanged':True,
        'single_image_audit':single['batch_vs_single'],'deployed':False,'training_steps':done['training_steps']}
    save(RUN/'VERIFIED.json',result)
    old=read(RUN/'old_test_results.json');new=read(RUN/'new_test_results.json');field=read(RUN/'field_regression_results.json')
    val=read(RUN/'validation_results.json');labels=read(RUN/'labels.json')
    gains=[];losses=[]
    for i,n in enumerate(old['baseline']['support']):
        if not n:continue
        b=old['baseline']['per_class_recall'][i];c=old['candidate']['per_class_recall'][i]
        delta=round((c-b)*n)
        if delta:
            (gains if delta>0 else losses).append({'label':labels[i],'n':n,'baseline_recall':b,'candidate_recall':c,'net_correct_change':delta})
    gains.sort(key=lambda r:-r['net_correct_change']);losses.sort(key=lambda r:r['net_correct_change'])
    save(RUN/'per_class_changes.json',{'improved':gains,'regressed':losses})
    lines=['# SSTv1 訓練與驗證結果','',
        '**結論：有小幅增益，但目前不足以證明街景識別明顯改善。保留為候選模型，未切換 Demo。**','',
        '以目前 BioCLIP 2.5 + MLP 適配教師的三個既有分類頭繼續微調，各 60 輪；編碼器凍結，沒有重新從零訓練。',
        '359 張可對應舊標籤的已驗收圖片：231 訓練、64 驗證、64 留出測試；另回放 23,582 張既有訓練圖。15 張 Tabebuia rosea 暫緩，不改成近似樹種。沒有使用康文署圖片。','',
        '| 固定比較資料 | 張數 | 原適配教師 | SSTv1 | 變化 |','|---|---:|---:|---:|---:|']
    for title,b,c in [('新增留出測試',new['baseline'],new['candidate']),('舊測試集中本輪重點類別',old['focus_baseline'],old['focus_candidate']),('舊網絡測試全部類別',old['baseline'],old['candidate']),('既有街景診斷',field['baseline'],field['candidate'])]:
        lines.append(f"| {title} | {b['n']} | {b['top1']:.2%} | {c['top1']:.2%} | {(c['top1']-b['top1'])*100:+.2f} 個百分點 |")
    lines+=['',
        '新增測試由 59/64 變為 60/64：修正 1 張細葉榕來源圖片，沒有新增錯例。這仍是在舊模型「榕樹合併類別」層級上的正確，不代表能分清細葉榕與垂葉榕。',
        '舊網絡測試修正 25 張、退步 19 張，淨增加 6 張。既有街景仍為 6/13，沒有提升。',
        f"完整舊驗證集準確率也有退步：{val['old_all']['baseline']['top1']:.2%} → {val['old_all']['candidate']['top1']:.2%}；並非所有資料都提升。",
        '舊測試總體準確率差的 95% 配對群組 bootstrap 區間為 −0.17 至 +0.42 個百分點，包含零；新增測試增益也只有一張，不能稱為已證實穩定顯著提升。',
        '', '## 改善／退步樹種（舊測試）','', '| 標籤 | 樣本數 | 原召回率 | SSTv1 召回率 | 淨正確數變化 |','|---|---:|---:|---:|---:|']
    for row in gains+losses:
        lines.append(f"| {row['label']} | {row['n']} | {row['baseline_recall']:.2%} | {row['candidate_recall']:.2%} | {row['net_correct_change']:+d} |")
    lines+=['','## 安全、限制與重現','',
        '- 三次微調都有獨立實體備份，備份與來源 SHA-256 一致，不是硬連結；原圖像／點雲／融合部署檔案未改動。',
        '- 固定 148 個輸出候選，沒有靠縮減分類範圍提高數字。本輪 21 個來源樹種對應 20 個舊輸出類別；不代表官方 27 條目全面覆蓋。',
        '- 標籤延續舊 Ficus／Araucaria 合併類別，保留來源學名；白千層亞種映射到既有種級輸出。',
        '- 白千層和假檳榔各只有 1 張新增保留圖，只進訓練，不能聲稱其新增測試精度提升。',
        '- 舊網絡／街景資料已多次查看，僅作回歸診斷；新增留出圖未參與參數與輪次選擇，但不能排除預訓練重疊、同一植株的跨觀察重複。',
        '- 新圖按 Demo 灰邊保比例處理；舊特徵沿用原中央裁切。各資料集內是同圖同預處理配對比較，不把跨資料集準確率直接等同。',
        '- 64 張留出圖已全部按 Demo 單張輸入方式重跑，兩個模型的 Top-1 均與批量測試一致，仍為 59/64 → 60/64。批量與單張置信度最多相差約 2.1 個百分點，因此不主張置信度逐位一致。改善案例亦與仍在運行的原教師 API 比對；詳見 VERIFIED.json。',
        '', '### 備份目錄','']+[f"- [{r['backup']}]({Path(r['backup']).as_posix()})" for r in done['members']]
    lines+=['','### 候選推理','',
        '使用 tools/infer_sstv1.py，傳入 --run outputs/SSTv1_incremental_20260914_v1 和 --image 原圖路徑，可同時返回原適配教師及 SSTv1 的結果；不會替換 Demo。',
        '',f"[改善案例原圖]({image.as_posix()}) · [詳細英文報告]({(RUN/'REPORT.md').as_posix()})"]
    (RUN/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
