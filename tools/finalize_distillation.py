"""Build a complete evidence-backed report after the isolated experiment finishes."""
import argparse
import json
import subprocess
from pathlib import Path
import numpy as np
import torch
from distillation_core import read_json, save_json
from training_safety import sha256, verify_backup
from run_distillation import report, write_csv, ROOT


def finalize(out):
    report(out)
    baseline = read_json(out / 'metrics/baseline_selection.json')
    teacher = read_json(out / 'metrics/teacher_selection.json')
    screen = read_json(out / 'screen_results.json')
    lock = read_json(out / 'final_lock.json')
    final = read_json(out / 'final_test.json')
    groups = []
    for run in sorted((out / 'runs').iterdir()):
        if not run.is_dir() or not (run / 'config.json').exists():
            continue
        config = read_json(run / 'config.json')
        backup = verify_backup(run / 'pre_training_backup', verify_sources=False)
        details = {'run': run.name, 'config': config, 'backup_files_verified': len(backup['files']),
                   'backup_manifest_sha256': sha256(run / 'pre_training_backup/backup_manifest.json')}
        if (run / 'last.pt').exists():
            state = torch.load(run / 'last.pt', map_location='cpu', weights_only=False)
            steps = max([float(s['step']) for s in state['optimizer']['state'].values() if 'step' in s] or [0])
            details.update(completed_epochs=state['epoch']+1, optimizer_steps=int(steps),
                           attempted_steps=state['global_step'], amp_skips=state['global_step']-int(steps))
            details['status'] = ('SMOKE_NO_OPTIMIZER_UPDATE' if steps == 0 else
                'PARTIAL_INTERRUPTED' if state['epoch']+1 < config['epochs'] else 'COMPLETED')
            del state
        elif (run / 'best.pt').exists():
            details['status'] = 'COMPLETED_HEAD_SCREEN' if config['stage'] == 'frozen_head_screen' else 'INCOMPLETE'
        else:
            details['status'] = 'NO_CHECKPOINT'
        groups.append(details)
    save_json(out / 'training_run_audit.json', groups)
    write_csv(out / 'training_run_audit.csv', [{k: v for k, v in r.items() if k != 'config'} for r in groups])
    git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    save_json(out / 'code_provenance.json', {'git_base_commit': git_commit,
        'files': {str(ROOT / p): sha256(ROOT / p) for p in
         ['tools/run_distillation.py', 'tools/distillation_core.py', 'tools/distillation_field_diagnostic.py',
          'tools/test_distillation_core.py', 'tools/finalize_distillation.py']},
        'note': 'Code is uncommitted. Per-fit source snapshots are stored in pre_training_backup. The first FP16 full fit stopped on code-source change; final locked arms ran the same stable BF16 code.'})
    summary = []
    def row(name, split, m, status='COMPLETED'):
        summary.append({'experiment': name, 'split': split, 'status': status,
            **{k: m.get(k, 'NA') for k in ['n', 'top1', 'macro_f1', 'balanced_accuracy', 'top5']}})
    row('E0_original_YOLO', 'selection_validation', baseline)
    row('E1_BioCLIP_zero_shot', 'selection_validation', teacher)
    for r in screen:
        row(f'head_T{r["temperature"]}_lambda{r["lambda"]}', 'selection_validation',
            read_json(Path(r['run']) / 'selection_metrics.json'))
    for key in ['gt_control', 'kd_candidate']:
        row(key, 'selection_validation', read_json(Path(lock[key]['run']) / 'selection_metrics.json'))
    for key in ['baseline', 'gt_control', 'kd_candidate']:
        row(key, 'historical_regression_test', final[key])
    teacher_test = out / 'metrics/teacher_test.json'
    if teacher_test.exists(): row('E1_BioCLIP_zero_shot', 'historical_regression_test', read_json(teacher_test))
    head_final = read_json(out / 'frozen_head_final_test.json') if (out / 'frozen_head_final_test.json').exists() else None
    if head_final:
        for key in ['gt', 'kd']:
            row('frozen_head_' + key, 'historical_regression_test', head_final[key])
    field = read_json(out / 'metrics/field_diagnostic.json') if (out / 'metrics/field_diagnostic.json').exists() else None
    if field:
        for key in ['baseline', 'teacher']: row(key, 'small_reviewed_VMMS_diagnostic', field[key])
    row('E2_multiview_teacher', 'NA', {}, 'ONE_TREE_CASE_ONLY_NOT_BENCHMARK')
    row('E4_multiview_KD', 'NA', {}, 'BLOCKED_INSUFFICIENT_VERIFIED_TREE_IDS')
    row('E5_DINO_feature_KD', 'NA', {}, 'BLOCKED_GATED_OFFICIAL_WEIGHTS_401')
    write_csv(out / 'experiment_summary.csv', summary)
    gtval = read_json(Path(lock['gt_control']['run']) / 'selection_metrics.json')
    kdval = read_json(Path(lock['kd_candidate']['run']) / 'selection_metrics.json')
    gain = final['kd_candidate']['macro_f1'] - final['gt_control']['macro_f1']
    orig_gain = final['kd_candidate']['macro_f1'] - final['baseline']['macro_f1']
    ci = final['kd_vs_matched_gt_accuracy_ci']
    complement = read_json(out / 'analysis/selection_complementarity.json')
    verified = read_json(out / 'production_verification.json')['unchanged']
    lines = ['# BioCLIP → YOLO 蒸餾實驗報告', '', '## Executive Summary', '',
        '**正式 Demo 未替換。** 本輪完成 BioCLIP 教師評估、7 組分類頭篩選，以及同初始化、同資料、同預算的 GT-only／KD 全骨幹訓練。', '',
        f'- 方案比較驗證集：教師 Top-1 {teacher["top1"]:.2%}、Macro-F1 {teacher["macro_f1"]:.4f}；原 YOLO Top-1 {baseline["top1"]:.2%}、Macro-F1 {baseline["macro_f1"]:.4f}。',
        f'- 最終回歸測試：KD 相對匹配 GT 續訓的 Macro-F1 差值 {gain:+.4f}（{gain*100:+.2f} 個百分點）；相對原模型 {orig_gain:+.4f}。',
        f'- KD 相對匹配 GT 的 Top-1 差值：{ci["point"]*100:+.2f} 個百分點；按來源分組成對 bootstrap 的 95% 區間：[{ci["ci95"][0]*100:+.2f}, {ci["ci95"][1]*100:+.2f}] 個百分點。這不是 Macro-F1 的區間。',
        f'- 原部署檔案 SHA256 未變：{verified}。每次訓練／嘗試均有獨立實體備份。',
        '- 建議：保留原部署；本輪只有單種子、短週期，需結合街景診斷再決定下一輪。', '',
        '## 1. 實驗範圍與資料', '',
        '- 主模型 YOLO11s-cls，保留 148 輸出結構；兩種榕樹經 logsumexp 合併為使用者要求的榕樹，监督與評估共 147 標籤。',
        '- 23,582 張訓練圖＝23,455 張原圖＋127 張目前人工接受的植物標本室影像；拒絕／待定圖不進入训练。沒有用教師改寫人工標籤。',
        '- 調參驗證 2,640 張；方案比較驗證 2,476 張；歷史回歸測試 5,146 張。逐張 SHA 與來源 group 檢查無跨 split 重疊。',
        '- 網路 observation group 不是已驗證的單木 ID。舊編碼器歷史訓練／選模暴露不能靠重新切分消除。BioCLIP 網路預訓練與本測試集重疊未知。',
        '- 狐尾椰子、銀合歡不在這個原分類器的輸出中；其既有獨立 Demo 分支沒有改動，沒有冒充本輪完成了這兩類蒸餾。', '',
        '## 2. 方法與選模', '',
        '- BioCLIP 2.5 Huge：官方 safetensors，固定 revision，eval／凍結／inference_mode；共比較 4 種提示。選擇 a photo of {species}，校準溫度只由調參驗證 NLL 決定。',
        '- 教師快取保存原始相似度、logit_scale、嵌入、類別順序、模型與資料雜湊；KD 溫度與校準溫度分開處理。',
        '- 教師與學生使用同一 EXIF 修正、雙三次縮放、224 中心裁剪；只有通道正規化不同，避免快取與隨機裁剪內容錯位。',
        '- 類別平衡 GT 損失為主監督；KD 使用熵加權 KL(teacher || student)。7 組凍結頭各 40 epochs；完整 GT 和 KD 各 3 epochs、相同初始權重和 seed、BF16、AdamW lr=1e-5。',
        f'- 所選 KD：T={lock["kd_candidate"]["temperature"]}、lambda={lock["kd_candidate"]["lambda"]}。完整 epoch 依方案比較驗證 Macro-F1 選擇；最終配置鎖定後才執行測試。',
        '- 校準／篩選重用歷史驗證資料，存在適應該驗證集的風險；最終結果不是全新盲測。', '',
        '## 3. 相同任務與資料的結果', '',
        '| 模型 | 驗證 Top-1 | 驗證 Macro-F1 | 測試 Top-1 | 測試 Macro-F1 |',
        '|---|---:|---:|---:|---:|']
    for key, display, val in [('baseline', '原 YOLO', baseline), ('gt_control', 'GT-only 續訓', gtval), ('kd_candidate', 'BioCLIP KD', kdval)]:
        lines.append(f'| {display} | {val["top1"]:.2%} | {val["macro_f1"]:.4f} | {final[key]["top1"]:.2%} | {final[key]["macro_f1"]:.4f} |')
    if teacher_test.exists():
        tm = read_json(teacher_test)
        lines.append(f'| BioCLIP 教師（不是輕量學生） | {teacher["top1"]:.2%} | {teacher["macro_f1"]:.4f} | {tm["top1"]:.2%} | {tm["macro_f1"]:.4f} |')
    if head_final:
        head_lock = read_json(out / 'head_final_lock.json')
        for kind in ['gt', 'kd']:
            val = read_json(Path(head_lock[kind]['run']) / 'selection_metrics.json')
            m = head_final[kind]
            lines.append(f'| 凍結特徵分類頭 {kind.upper()} | {val["top1"]:.2%} | {val["macro_f1"]:.4f} | {m["top1"]:.2%} | {m["macro_f1"]:.4f} |')
        lines += ['', '補充對照：驗證集顯示凍結 KD 頭優於完整微調候選，因此在任何新測試推理前另行鎖定最佳 KD 頭及匹配 GT 頭（head_final_lock.json 有時間與權重雜湊）。這是事前選定的兩種訓練方式對照，不是測試後挑出最好頭。',
                  f'凍結 KD 相對凍結 GT 的測試 Macro-F1 差值：{head_final["kd"]["macro_f1"]-head_final["gt"]["macro_f1"]:+.4f}。']
    lines += ['', '不能與先前 20 類封閉集準確率直接相減；本輪是 147 標籤的不同任務。所有細項見 experiment_summary.csv。', '',
        '## 4. 教師是否互補', '',
        f'- 同一方案比較驗證集，YOLO 錯／教師對 {complement.get("student_wrong_teacher_correct",0)} 張；YOLO 對／教師錯 {complement.get("student_correct_teacher_wrong",0)} 張。教師不是人工真值。',
        '- 錯誤個案：analysis/selection_hard_samples.csv；全部類別升降：analysis/test_per_class_comparison.csv；測試逐圖輸出：analysis/test_predictions.csv。', '',
        '## 5. 香港街景與多視角', '']
    if field:
        fm = field['teacher']; bm = field['baseline']
        lines += [f'- 可追溯人工接受的 VMMS 留出診斷：{fm["n"]} 個視角、{fm["classes_with_support"]} 個有支持的標籤。原 YOLO Top-1 {bm["top1"]:.2%}，BioCLIP {fm["top1"]:.2%}。',
                  '- 這裡測的是通用 147 標籤圖像分支，不是既有三類／五類雙源實驗頭；不能把此結果當成整個 Demo 的準確率。',
                  '- 樣本非常少，而且為歷史驗收資料，不能當成部署準確率。這批影像沒有用於調參或訓練。']
    case_path = out / 'metrics/one_verified_tree_case.json'
    if case_path.exists():
        case = read_json(case_path)
        lines += ['- 對人工確認同樹的 4276／4287 幀，完成 2 視角教師均值、對數概率、熵權重融合個案：']
        for r in case['fusion']:
            lines.append(f'  - {r["method"]}：{r["prediction"]}；人工標籤 {r["truth"]} 的候選概率 {r["GT_probability"]:.2%}。')
    lines += ['- 只有一棵明確確認同樹的個案，不足以獨立建立多視角 train／val／test；E4 未執行，沒有猜測 Tree-ID。', '',
        '## 6. DINOv3', '',
        '- 官方 gated 檔案存取返回 HTTP 401，未取得已授權權重；未進行特徵蒸餾。見 audit/DINO_STAGE_BLOCKED_BY_WEIGHTS.md。沒有繞過限制。', '',
        '## 7. 失敗、恢復與安全', '',
        '- 初次稽核遇到模型 names 是帶數字前綴的資料夾名稱，已用現有 class_folders 一一驗證映射；沒有猜測或重排標籤。舊失敗目錄保留。',
        '- 首個 FP16 smoke 的兩步被 AMP 跳過，不能算成功訓練；後續改 BF16，加入有效優化器步數檢查。',
        '- 第一輪 FP16 GT 在兩個 epoch 後因執行中程式來源變動觸發安全停止；檢查點和備份保留。正式比較使用重跑的穩定 BF16 兩組，不混用該中斷結果。',
        '- 所有 run 的實際 epoch／optimizer steps／AMP skips／備份檢查見 training_run_audit.json；不得把失敗 smoke 計入成功試驗。',
        '- 原始圖像、人工標籤、點雲／融合部署權重均未覆蓋；沒有 commit／push。模型與影像快取不進 Git。',
        '- 每個 run 的 pre_training_backup 保存初始權重、相關部署權重、資料與設定快照；恢復必須使用新目錄。', '',
        '## 8. 下一步', '',
        '1. 先看人工接受街景中的錯誤類型和裁剪品質，不以網路圖教師分數代替現場表現。',
        '2. 固定更多獨立單木留出資料，再確認蒸餾增益能否跨來源保持。',
        '3. 若學生落後教師很多，測試較長但匹配預算的 KD、逐層解凍或教師特徵頭；不能直接推定大教師知識已被學生吸收。',
        '4. 有授權 DINOv3 權重後，將特徵蒸餾與視角一致性分開消融。',
        '5. 是否替換 Demo 必須另行獲得使用者批准。', '',
        '## 來源', '',
        '- [BioCLIP 2.5 官方模型卡](https://huggingface.co/imageomics/bioclip-2.5-vith14)',
        '- [DINOv3 官方實作](https://github.com/facebookresearch/dinov3)']
    (out / 'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    completed_resumes = [r for r in groups if r['run'].startswith('resume_') and r['status'] == 'COMPLETED']
    if completed_resumes:
        with (out / 'REPORT.md').open('a', encoding='utf-8') as f:
            f.write('\n## 恢復功能實測\n\n已將先前中斷的 GT 檢查點在新目錄恢復，實際完成剩餘 epoch，'
                    '並重新驗證備份／模型載入。這只驗證恢復功能，未改變已鎖定的 BF16 對照／KD 選擇，'
                    '沒有對恢復版再次使用測試集選模。\n')
    (out / 'NEXT_STEPS.md').write_text('# Next steps based on this run\n\n'
        f'KD minus matched GT test Macro-F1: {gain:+.6f}. KD minus original: {orig_gain:+.6f}.\n\n'
        'KEEP existing production. The teacher/student gap warrants better-controlled KD only if field diagnostics support it. '
        'Prioritize independent reviewed street trees, verified identities and input-crop quality. '
        'Do not report one-tree fusion as a multiview benchmark. DINO requires authorized official weights.\n', encoding='utf-8')
    save_json(out / 'status.json', {'stage': 'COMPLETED_SINGLE_VIEW_PILOT', 'production_deployed': False,
        'production_unchanged': verified, 'multiview_KD': 'BLOCKED_INSUFFICIENT_VERIFIED_IDENTITIES',
        'dino': 'BLOCKED_GATED_WEIGHTS', 'report': str(out / 'REPORT.md')})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True)
    finalize(p.parse_args().out.resolve())
