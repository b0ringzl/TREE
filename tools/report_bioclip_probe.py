"""Post-lock reporting for the frozen BioCLIP adaptation experiment."""
import argparse
import csv
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from distillation_core import read_json, save_json, metrics
from training_safety import sha256, verify_backup
from bioclip_probe import ResidualHead, ROOT


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def cluster_macro_ci(y, a, b, group_ids, repeats=1000):
    """Paired source-group bootstrap stratified by GT class, fixed class support."""
    c = a.shape[1]
    groups = list(dict.fromkeys(group_ids)); group_index = {g: i for i, g in enumerate(groups)}
    inverse = np.array([group_index[g] for g in group_ids])
    strata = {}
    for group, index in group_index.items():
        labels = set(y[inverse == index])
        if len(labels) != 1: raise ValueError('Mixed-label source group')
        strata.setdefault(int(labels.pop()), []).append(index)
    pred_a, pred_b = a.argmax(1), b.argmax(1)
    def f1(pred, weights):
        cm = np.bincount(y*c+pred, weights=weights, minlength=c*c).reshape(c,c)
        denom = cm.sum(0) + cm.sum(1)
        scores = np.divide(2*cm.diagonal(), denom, out=np.zeros(c), where=denom>0)
        return scores[cm.sum(1)>0].mean()
    rng = np.random.default_rng(20260912)
    deltas = []
    for _ in range(repeats):
        counts = np.zeros(len(groups), dtype=np.int64)
        for ids in strata.values():
            drawn = rng.choice(ids, len(ids), replace=True)
            np.add.at(counts, drawn, 1)
        weights = counts[inverse]
        deltas.append(f1(pred_b, weights)-f1(pred_a, weights))
    return {'metric': 'paired_MacroF1_difference', 'point': metrics(y,b)['macro_f1']-metrics(y,a)['macro_f1'],
        'ci95': np.quantile(deltas,[.025,.975]).tolist(), 'repeats': repeats,
        'method': 'Source group bootstrap stratified by true class. Source groups are not verified street tree IDs.'}


def main(out):
    protocol = read_json(out/'protocol.json')
    lock = read_json(out/'selection_lock.json')
    final = read_json(out/'final_results.json')
    labels = read_json(out/'classes.json')['labels']
    rows = read_json(out/'test_rows.json')
    y = np.array([r['target'] for r in rows])
    pred = np.load(out/'test_predictions.npz')
    ci = cluster_macro_ci(y, pred['baseline'], pred['ensemble'], [r['group_id'] for r in rows])
    save_json(out/'test_macro_f1_difference_ci.json', ci)
    fits = read_json(out/'fit_results.json')
    summary = []
    baseline = read_json(out/'baseline_selection.json')
    summary.append({'model':'zero_shot','seed':'NA','phase':'selection',
        'top1':baseline['top1'],'macro_f1':baseline['macro_f1'],'best_epoch':0})
    for r in fits:
        summary.append({'model':r['recipe'],'seed':r['seed'],'phase':'selection',
            'top1':r['selection_top1'],'macro_f1':r['selection_macro_f1'],'best_epoch':r['best_epoch']})
    for phase, result in final.items():
        for kind in ['baseline','ensemble']:
            summary.append({'model':kind,'seed':'NA','phase':phase,'top1':result[kind]['top1'],
                'macro_f1':result[kind]['macro_f1'],'best_epoch':'NA'})
        for seed in result['seeds']:
            summary.append({'model':lock['selected_recipe'],'seed':seed['seed'],'phase':phase,
                'top1':seed['top1'],'macro_f1':seed['macro_f1'],'best_epoch':'see selection_lock.json'})
    write_csv(out/'experiment_summary.csv',summary)
    gains = np.array(final['test']['ensemble']['per_class_f1'])-np.array(final['test']['baseline']['per_class_f1'])
    perclass = [{'species':name,'support':final['test']['baseline']['support'][i],
        'baseline_f1':final['test']['baseline']['per_class_f1'][i],
        'candidate_f1':final['test']['ensemble']['per_class_f1'][i],'gain':float(gains[i])}
        for i,name in enumerate(labels)]
    write_csv(out/'per_class_comparison.csv',perclass)
    for phase in ['test','field','case']:
        rr = read_json(out/f'{phase}_rows.json'); pp = np.load(out/f'{phase}_predictions.npz')
        write_csv(out/f'{phase}_predictions.csv',[{'id':r['sample_id'],'truth':labels[r['target']],
            'baseline':labels[int(pp['baseline'][i].argmax())], 'candidate':labels[int(pp['ensemble'][i].argmax())],
            'baseline_top1_score':float(pp['baseline'][i].max()),'candidate_top1_score':float(pp['ensemble'][i].max())}
            for i,r in enumerate(rr)])
    backups = []
    for run in sorted((out/'runs').iterdir()):
        manifest = verify_backup(run/'pre_training_backup')
        state = torch.load(run/'last.pt', map_location='cpu', weights_only=False)
        steps = max(int(v['step'].item()) for v in state['optimizer']['state'].values())
        backups.append({'run':run.name, 'epochs':state['epoch'], 'optimizer_steps':steps,
            'backup_verified':True,'backup_bytes':sum(r['bytes'] for r in manifest['files'])})
    save_json(out/'backup_verification.json',backups)
    protected = all(sha256(p)==h for p,h in protocol['protected_sources'].items())
    teacher_unchanged = sha256(protocol['teacher_weight'])==protocol['teacher_sha256']
    save_json(out/'production_verification.json',{'unchanged':protected,'teacher_unchanged':teacher_unchanged})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=out/'figures'; figures.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(12,4))
    for recipe in protocol['recipes']:
        histories=[read_json(Path(r['run'])/'history.json') for r in fits if r['recipe']==recipe['name']]
        curves=np.array([[h['planning_macro_f1'] for h in history] for history in histories])
        mean,std=curves.mean(0),curves.std(0); epochs=np.arange(1,len(mean)+1)
        ax.plot(epochs,mean,label=recipe['name']); ax.fill_between(epochs,mean-std,mean+std,alpha=.15)
    ax.set(xlabel='Epoch',ylabel='Planning Macro-F1',title='Frozen BioCLIP heads: three-seed mean and standard deviation')
    ax.legend(); fig.tight_layout(); fig.savefig(figures/'learning_curves.png',dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,4)); ax.bar(np.arange(len(gains)),gains,color=np.where(gains>=0,'#27805d','#b04c4c'))
    ax.set(xlabel='Class index (classes.json)',ylabel='F1 difference',title='Selected three-head ensemble minus unchanged zero-shot: historical test')
    fig.tight_layout(); fig.savefig(figures/'per_class_gain.png',dpi=150); plt.close(fig)
    test=final['test']; field=final['field']; architecture=lock['selected_runs'][0]['architecture']
    params=sum(p.numel() for p in ResidualHead(architecture).parameters())
    lines=['# BioCLIP 凍結表徵＋輕量分類頭實驗','',
        '## 結論','',
        f'本輪完成 **3 種配置 × 3 個隨機種子，共 9 次正式分類頭訓練**，另有 1 次小樣本檢查。選中 `{lock["selected_recipe"]}`，所有選擇均在本輪測試推理前鎖定。正式 Demo 未更換。','',
        '| 同一批 5,146 張歷史測試圖 | Top-1 | Macro-F1 |', '|---|---:|---:|',
        f'| 原 BioCLIP 零樣本 | {test["baseline"]["top1"]:.2%} | {test["baseline"]["macro_f1"]:.4f} |',
        f'| 所選配置三種子概率平均 | {test["ensemble"]["top1"]:.2%} | {test["ensemble"]["macro_f1"]:.4f} |','',
        f'- 單種子 Macro-F1 均值／標準差：{test["macro_f1_mean"]:.4f} ± {test["macro_f1_std"]:.4f}。概率平均結果不是「最好的一個 seed」。',
        f'- 相對零樣本的 Macro-F1 變化：{ci["point"]*100:+.2f} 個百分點；按真實類別分層、來源群組 bootstrap 的 95% 區間 [{ci["ci95"][0]*100:+.2f}, {ci["ci95"][1]*100:+.2f}] 個百分點。',
        '- 這是歷史網路圖片回歸測試，不是全新街景盲測；基礎教師預訓練可能與網路資料重疊，尚未排除。','',
        '## 街景診斷：單獨看待','',
        f'- 同一批 13 張人工接受留出視角：零樣本對 {round(field["baseline"]["top1"]*13)}/13，適配後對 {round(field["ensemble"]["top1"]*13)}/13。涉及 4 個有支持的標籤，樣本極少。',
        '- 未將這批已看過的街景診斷資料放入訓練、選配置或早停；不得把 13 個視角当作 13 棵已確認獨立的樹。',
        '- 這是通用 147 標籤影像分支，不是原三類／五類雙源分類頭；不代表整個 Demo 的準確率。','',
        '## 本轮具体训练了什么','',
        '- 保留 BioCLIP 2.5 Huge 凍結的圖像特徵和原提示詞／校準版本，讀取前輪具備樣本及類別雜湊的特徵快取。',
        '- 線性配置是原零樣本線性打分＋可訓練線性修正，兩者合起來仍為線性分類；MLP 配置是固定零樣本打分＋小型非線性修正。修正層初始輸出零，不隨機破壞教師初始預測。',
        '- 兩種線性學習率 0.001／0.01，MLP 學習率 0.003；每次 80 個全批次更新，只以人工 GT 交叉熵訓練，按類別平衡、已接受標本室資料 ×3。',
        '- 23,582 張訓練圖，保留 148 原輸出、兩種榕樹合併成 147 監督／評估標籤。狐尾椰子和銀合歡仍未納入該分支。',
        '- 每個 seed 依 planning 驗證集 Macro-F1 選 epoch，允許第 0 輪勝出；配置依 selection 驗證集三種子中位數選擇，固定採三種子概率平均。',
        '- 線性配置使用零初始化、全批次、無隨機增強，因此三個 seed 重複出相同解，不能把它們當作三份獨立穩定性證據。MLP 隱藏層初始化和 dropout 才引入本輪的種子差異。',
        f'- 每個選中頭有 {params:,} 個參數。**頭很小不等於整個模型很小：部署仍需 BioCLIP Huge 圖像編碼器**，未把它聲稱為 YOLO 成本。',
        '- 適配後分數不是已證實的開放世界置信度；本輪沒有任意未知物種拒識評估。','',
        '## 配置比較：方案選擇驗證集','',
        '| 配置 | 三種子 Macro-F1 中位數 |', '|---|---:|',
        f'| 零樣本 | {baseline["macro_f1"]:.4f} |']
    for r in lock['ranking']: lines.append(f'| {r["recipe"]} | {r["median_selection_macro_f1"]:.4f} |')
    lines += ['', '## 類別升降','',
        '以下為全部類別中的最大變化，不是新的選模依據：']
    for r in sorted(perclass,key=lambda r:r['gain'])[:5]: lines.append(f'- 下降：{r["species"]}，F1 {r["gain"]:+.4f}，測試支持 {r["support"]} 張。')
    for r in sorted(perclass,key=lambda r:r['gain'],reverse=True)[:5]: lines.append(f'- 上升：{r["species"]}，F1 {r["gain"]:+.4f}，測試支持 {r["support"]} 張。')
    lines += ['', '## 安全與產物','',
        f'- 原部署不變：{protected}；凍結教師原權重不變：{teacher_unchanged}。',
        f'- 驗證了 {len(backups)} 份訓練前實體備份；每份包含初始分類頭、教師原權重、相關部署權重與資料／設定快照。',
        '- 沒有修改原始影像、標籤、點雲、舊模型或舊實驗結果；沒有 Git 提交或推送。',
        '- `selection_lock.json`：選模時間、配置、所有種子與權重雜湊。',
        '- `experiment_summary.csv`：所有正式配置與分 split 數值。',
        '- `per_class_comparison.csv`、`field_predictions.csv`：完整類別升降與街景逐圖結果。',
        '- `backup_verification.json`、`production_verification.json`：備份與來源校驗。',
        '- `figures/learning_curves.png`、`figures/per_class_gain.png`：學習曲線與全部類別變化。','',
        '## 下一步','',
        '優先建立更多具備來源與人工標籤的街景訓練／獨立留出資料，處理單木裁剪品質與跨域差異。不能用本輪網路圖增益代替現場提升；正式接入 Demo 仍需另行批准。']
    raw_path=out/'raw_image_inference.json'
    if raw_path.exists():
        raw=read_json(raw_path)
        field_rows=read_json(out/'field_rows.json')
        matching=[i for i,r in enumerate(field_rows) if r['source_sha256']==raw['image_sha256']]
        if len(matching)!=1: raise ValueError('Raw-image verification sample identity mismatch')
        stored=np.load(out/'field_predictions.npz')['ensemble'][matching[0]]
        direct=np.array(raw['probabilities'])
        verification={'sample_sha256':raw['image_sha256'],'top1_agreement':bool(stored.argmax()==direct.argmax()),
            'maximum_probability_difference':float(np.max(np.abs(stored-direct))),
            'scope':'One direct raw-image inference compared with prior batched cached inference. Not a full deployment equivalence test.',
            'no_GT_input':True}
        save_json(out/'raw_image_verification.json',verification)
        lines += ['', '## 原始影像推理檢查','',
            '- 新增獨立命令列推理工具 tools/infer_bioclip_probe.py，可直接讀取影像，載入凍結教師與所選分類頭；不讀取該圖真值來決定預測，也不改 Demo。',
            f'- 實測一張既有街景圖，直接推理與快取評估的 Top-1 一致：{verification["top1_agreement"]}；最大候選概率差 {verification["maximum_probability_difference"]:.6f}。',
            '- 單張直接推理與批次 BF16 快取並非逐位一致；本檢查不代表所有部署輸入完全等價。這張圖本身仍識別錯誤，沒有用技術連通測試冒充識別正確。',
            '- 結果見 raw_image_inference.json、raw_image_verification.json。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save_json(out/'status.json',{'stage':'COMPLETE','production_unchanged':protected,
        'teacher_unchanged':teacher_unchanged,'deployed':False,'report':str(out/'REPORT.md')})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out.resolve())
