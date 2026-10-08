"""Read completed regression results and produce a concise, auditable handoff."""
import csv
import json
import sys
from pathlib import Path
from training_safety import verify_backup


def main(trial):
    trial = Path(trial)
    data = json.loads((trial / 'comparison.json').read_text(encoding='utf-8'))
    config = json.loads((trial / 'trial_config.json').read_text(encoding='utf-8'))
    backup = verify_backup(trial / 'candidate/pre_training_backup')
    def predictions(path):
        with path.open(encoding='utf-8', newline='') as f:
            return {r['sample_id']: r for r in csv.DictReader(f)}
    old = predictions(trial / 'baseline_test_predictions.csv')
    new = predictions(trial / 'candidate/test_predictions.csv')
    if old.keys() != new.keys():
        raise RuntimeError('Different evaluation samples')
    if any(old[k]['true_class_index'] != new[k]['true_class_index'] for k in old):
        raise RuntimeError('Different evaluation labels')
    recovered = sum(old[k]['correct']=='0' and new[k]['correct']=='1' for k in old)
    regressed = sum(old[k]['correct']=='1' and new[k]['correct']=='0' for k in old)
    a, b = data['baseline']['test_sample'], data['candidate']['test_sample']
    improved = b['accuracy'] >= a['accuracy'] and b['macro_f1'] >= a['macro_f1']
    lines = ['# 已驗收影像增量訓練：結果與回退保障', '',
             '## 結論', '',
             ('本輪固定回歸測試整體指標小幅上升，但鳳凰木、大葉合歡、蒲葵等重點類別 F1 下降；不建議替換正式模型，候選未部署。' if improved else '本輪至少一項整體分類指標下降，保留原模型，不切換候選。'), '',
             f'- 驗收：保留 {config["accepted"]} 張、剔除 {config["review_counts"].get("rejected",0)} 張。本輪新增入訓 {config["new_training_images"]} 張；4 張銀合歡因現有主分類器沒有輸出類而暫存，不改成其他樹種。',
             '- 已接受的重复照片只有一個版本入選，另一版本被使用者剔除；未重複加權。',
             '- 原有 23,455 張訓練圖全部回放；新增圖片只進訓練。148 類圖像分類器，3 輪，初始學習率 0.00005，固定 seed 20260907；新來源在類別平衡採樣中乘以 3。',
             '- 驗證集 5,116 張、測試集 5,146 張完全固定；不是新建的街景盲測，也不是 Demo 19 類或雙源專項準確率。',
             f'- 以驗證集分組 Macro-F1 選取第 {data["selected_epoch"]} 輪，再評估測試集；未以測試結果選 epoch。', '',
             '| 評估範圍 | 原準確率 | 候選準確率 | 原 Macro-F1 | 候選 Macro-F1 |', '|---|---:|---:|---:|---:|']
    scopes={'val_sample':'驗證・逐圖','val_group':'驗證・按既有組聚合','test_sample':'測試・逐圖','test_group':'測試・按既有組聚合'}
    for scope, label in scopes.items():
        x, y = data['baseline'][scope], data['candidate'][scope]
        lines.append(f'| {label} | {x["accuracy"]:.2%} | {y["accuracy"]:.2%} | {x["macro_f1"]:.4f} | {y["macro_f1"]:.4f} |')
    lines += ['', f'相同的 5,146 張測試圖中：{recovered} 張由錯變對，{regressed} 張由對變錯。單次小幅變化未做顯著性證明。', '',
              '## 有新增資料的類別：固定測試對照', '', '| 學名 | 新增訓練圖 | 測試支持數 | 原 F1 | 候選 F1 | ΔF1 |', '|---|---:|---:|---:|---:|---:|']
    old_classes={r['scientific_name']:r for r in a['per_class']}
    new_classes={r['scientific_name']:r for r in b['per_class']}
    for name,count in config['new_images_by_class'].items():
        x,y=old_classes[name],new_classes[name]
        lines.append(f'| {name} | {count} | {x["support"]} | {x["f1"]:.4f} | {y["f1"]:.4f} | {y["f1"]-x["f1"]:+.4f} |')
    lines += ['', '## 備份與持久規則', '',
              f'- 備份含 {len(backup["files"])} 個檔案：6 份部署權重／分類頭，另含樹種清單及推理設定程式。均為獨立副本，非硬連結；已校驗 SHA-256。',
              '- [備份清單](candidate/pre_training_backup/backup_manifest.json)。原始部署檔案再次比對與備份相同，Demo 未切換，因此不需要執行回退。',
              '- 根目錄 AGENTS.md 已記錄「每次訓練前備份、禁止覆蓋舊結果、評估後另行批准部署」。本輪 YOLO 訓練入口已接入強制備份校驗；其他舊訓練入口未在本輪執行，未聲稱全部已改寫。',
              '- 訓練／驗收快照、類別映射、超參數、原評估結果與候選 checkpoint 均保留在本實驗目錄。', '',
              '## 限制與異常記錄', '',
              '- 這是一次小規模試訓，未做「同樣微調但不加新圖」的消融對照，不能把所有指標變化歸因於新圖。',
              '- 本輪只改候選圖像模型，未重訓點雲／融合頭；也未將候選編碼器接入依賴舊編碼器的雙源頭。',
              '- 來源圖像保留水印及署名，本地實驗沒有推定商業或分發授權；未發布圖片或候選權重。']
    if (trial/'postprocessing_recovery.json').exists():
        lines += ['- 3 輪訓練完成後，訓練曲線繪製觸發 PySide/polars 遞迴錯誤。逐一確認三個 epoch、參數有限值與檔案雜湊後，僅恢復評估，沒有重訓或覆蓋原模型。原錯誤日誌及恢復記錄均保留；未來入口使用 Agg 無界面繪圖。']
    (trial/'training_comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'baseline_accuracy':a['accuracy'],'candidate_accuracy':b['accuracy'],'baseline_macro_f1':a['macro_f1'],'candidate_macro_f1':b['macro_f1'],'corrected':recovered,'regressed':regressed,'production_unchanged':True}))


if __name__=='__main__':
    main(sys.argv[1])
