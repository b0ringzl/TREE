"""Report locked experiments and verified live deployment, without further tuning."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from distillation_core import read_json,save_json
from training_safety import verify_backup,sha256
from spacescan_trees_demo_v1.species_names import bilingual_names

OUT=ROOT/'outputs/SSTv1_Dualview_Deployment_20260920_v1'
EXP=ROOT/'outputs/SSTv1_Inference_Optimization_20260920_v2'
def link(p,label):return f'[{label}](<{Path(p).resolve().as_posix()}>)'
def main():
    assert not (OUT/'REPORT.md').exists()
    reg=read_json(EXP/'regression_results.json');dev=read_json(EXP/'development_results.json')
    live=read_json(OUT/'LIVE_VERIFIED.json');browser=read_json(OUT/'browser_acceptance/verification.json')
    assert reg['deployment_eligible'] and live['passed'] and browser['passed']
    selected=next(c['metrics'] for c in dev['candidates'] if c['recipe']==reg['recipe'])
    before,after=reg['baseline'],reg['candidate']
    backup=verify_backup(OUT/'pre_training_backup',verify_sources=False)
    expbackup=verify_backup(EXP/'pre_training_backup',verify_sources=False)
    changed=[]
    for r in backup['files']:
        now=sha256(r['source'])
        if now!=r['sha256']:changed.append(dict(path=r['source'],before=r['sha256'],after=now))
    assert all(Path(r['path']).suffix not in {'.pt','.safetensors','.joblib'} for r in changed)
    regressions=[]
    for sp,m in after['per_class'].items():
        b=before['per_class'][sp]
        if m['tp']<b['tp']:
            regressions.append(dict(species=sp,**bilingual_names(sp),before=b,after=m,delta=m['tp']-b['tp']))
    regressions.sort(key=lambda r:(r['delta'],r['species']))
    save_json(OUT/'per_class_regressions.json',regressions)
    old=read_json(ROOT/'outputs/windows_portable_acceptance_20260920_v1/formal.json')
    current=read_json(OUT/'live_capture.json')
    user_regressions=[dict(file=a['file'],baseline=a['top1'],candidate=b['top1']) for a,b in zip(old['images'],current['images']) if a['top1']=='Aquilaria sinensis' and b['top1']!='Aquilaria sinensis']
    save_json(OUT/'user_training_replay_regressions.json',user_regressions)
    lines=['# SSTv1-H120-DV：Demo 精度优化与上线验收', '', '日期：2026-09-20', '',
      '## 已落地的改进', '',
      '正式网页两种入口的单源影像识别已接通“完整目标图＋中心细节”各 50% 的概率综合。仍是当前 H120 三个 MLP 与同一 BioCLIP 2.5 编码器，优化器步数为 0；没有新训练、换标签、降低温度或人为抬高正确标签分数。', '',
      '先保留整株轮廓，再利用原始目标图的中心细节。自动选树、透视校正或人工多边形裁剪完成后才执行；未经选树的完整全景仍拒绝直接分类。中心输入直接来自原始目标字节，不能从已经补边的正方形图再次取中心。', '',
      '- 正式入口：[单源影像版](http://127.0.0.1:8043/image-only)、[影像＋点云完整版](http://127.0.0.1:8043/multimodal)。两版图像识别共用新流程。',
      '- UI 版本：SSTv1-H120-DV · 2026-09-20；权重版本仍为 sstv1-aquilaria-user-crops-20260919-h120。',
      '- 结果显示两张实际模型取景；判断不一致时提示检查目标。它们是同一照片的两种裁法，不是多角度观测，也不是多张上传照片融合。', '',
      '## 同一固定数据上的结果', '',
      '| 集合／指标 | 当前 H120 单取景基线 | H120-DV |', '|---|---:|---:|',
      f"| 开发集 Top-1 | {dev['baseline']['all']['correct']}/2650（{dev['baseline']['all']['top1']:.2%}） | {selected['all']['correct']}/2650（{selected['all']['top1']:.2%}） |",
      f"| 历史回归 Top-1 | {before['all']['correct']}/5146（{before['all']['top1']:.2%}） | {after['all']['correct']}/5146（{after['all']['top1']:.2%}） |",
      f"| 历史回归宏 F1 | {before['all']['macro_f1']:.2%} | {after['all']['macro_f1']:.2%} |"]
    for key,label in [('aquilaria','土沉香'),('white','白桂木')]:
        b,a=before[key],after[key]
        lines += [f"| {label}识对／真实数 | {b['tp']}/{b['tp']+b['fn']}（{b['recall']:.1%}） | {a['tp']}/{a['tp']+a['fn']}（{a['recall']:.1%}） |",
                  f"| {label}误报数 | {b['fp']} | {a['fp']} |",
                  f"| {label}精确率 | {b['precision']:.1%} | {a['precision']:.1%} |"]
    lo,hi=reg['bootstrap']['ci95']
    lines += ['',f"历史回归纠正 {reg['corrected']} 张，同时退步 {reg['regressed']} 张；净增 {reg['corrected']-reg['regressed']} 张，Top-1 +{reg['bootstrap']['point']*100:.2f} 个百分点。按既有来源分组 bootstrap 的差值区间为 +{lo*100:.2f}～+{hi*100:.2f} 个百分点。这不能消除反复查看历史数据、未知预训练重叠、未知同株等局限。", '',
      '开发集额外 5 张目录外图片只参与误报检查，不计入 2650 张分类分母；当前输出目录有 148 类，历史回归真值仅覆盖其中 147 类。网页图库回归不是香港街景端到端盲测，不保证任一用户图片都改善。土沉香仍有 14 次误报，不能作为已确认的保育身份。', '',
      '## 选择方法与回退情况', '',
      '备份后预先冻结 11 个全局方案：中心占比 25%／50%／75%，旧分类头占比 0／25%／50%，另两个高分保留整图对照。只在固定开发集选择宏 F1 最好的达标方案，土沉香／白桂木 TP 不降、FP 不增（新旧开发子集分别保护），总体 Top-1 至少 +1 个百分点。锁定后只评估该候选和当前正式基线的历史回归，未据回归结果重选方案。最终选中中心 50%、旧头 0%，没有使用旧分类头。', '',
      '原 H120 是用户指定的训练裁片加强版；它先前没有通过所有原始训练替换门槛。本次对照真实部署中的 H120 单取景，不能把其前身的结果当成本轮基线。', '',
      '退步类别（按识对数降幅列出；完整 148 类指标与逐图变动见链接）：', '',
      '| 树种 | 基线识对 | 新版识对 | 真实数 |', '|---|---:|---:|---:|']
    for r in regressions[:10]:
        lines.append(f"| {r['scientific_name']}／{r['chinese_name']} | {r['before']['tp']} | {r['after']['tp']} | {r['before']['tp']+r['before']['fn']} |")
    lines += ['',f"用户此前 8 张已训练裁片回放为 {live['user_crop_replay']['baseline']}/8 → {live['user_crop_replay']['candidate']}/8，并非独立验证。新增退步："+
      '；'.join(f"{r['file']}：{r['baseline']} → {r['candidate']}" for r in user_regressions)+'。不能只呈现总体增益而隐藏这一取舍。', '',
      '## 真实 Demo 验收', '',
      f"- 20 张固定分布抽样逐一上传，148 类完整概率与锁定实现最大差值 {live['max_probability_delta']:.3g}；旧单取景 API 保持原值。",
      '- 自动检测仍找到同样 3 株树；选择切片与导出原生切片重新上传的概率完全一致。',
      '- 两个界面版本、真实图像上传、原生多边形裁剪后识别、两张模型输入预览、导出版本证据均通过；390 像素移动布局无横向溢出，页面 JavaScript 错误为 0。',
      f"- 点云／图像点云融合结果最大变化 {live['point_fusion_max_delta']:.3g}，属于浮点误差；相关权重未改。坏图返回 JSON 400，未经选树的全景返回 409。",
      f"- 本机中位模型推理约 {live['median_legacy_model_seconds']:.3f} 秒 → {live['median_model_seconds']:.3f} 秒，两次串行 batch=1 推理；请求中位耗时 {live['median_request_seconds']:.3f} 秒。没有声称其它电脑或手机具有同样速度。",
      '- 用户浏览器控制连接异常，最终界面验收使用独立的本机 headless Chrome 测试，不冒称已操控用户当前标签页。', '',
      '## 备份、兼容与剩余缺口', '',
      f"执行前备份 {sum(x['bytes'] for x in expbackup['files'])/2**30:.2f} GiB，上线前独立备份 {sum(x['bytes'] for x in backup['files'])/2**30:.2f} GiB；物理复制、SHA-256 校验通过。上线前备份已包含原代码、权重、发布配置、验证结果与标签。全部 {live['weight_files']} 个受保护权重文件哈希不变，修改的是推理与界面代码。", '',
      '- '+link(EXP/'pre_training_backup','执行前备份'),
      '- '+link(OUT/'pre_training_backup','上线前备份（回滚来源）'),
      '- 旧 Windows 便携 ZIP 和安卓 APK／离线模型文件未改；它们不会自动获得本次提升。版本页已注明区别。不得用本次分数声称这些旧包已同步。',
      '- 正式 148 类目录不包含 Rhodoleia championii／紅花荷、Keteleeria fortunei／油杉。后续需要单独进行 150 类目录扩展及来源隔离评估；这不是继续微调现有 148 类头即可解决的覆盖问题。',
      '- 后续提升优先级：可靠且同株隔离的新评估图；对 49 个退步案例做目标／背景诊断；正式 150 类覆盖验证；再评估骨干解冻或多照片融合。未知目标的整张全景不能直接套用中心细节。', '',
      '## 可复核文件', '',
      '- '+link(EXP/'protocol.json','冻结协议'),
      '- '+link(EXP/'selection_lock.json','开发集选择锁'),
      '- '+link(EXP/'development_results.json','11 方案开发指标'),
      '- '+link(EXP/'regression_results.json','全类别回归指标'),
      '- '+link(EXP/'regression_changed_cases.json','241 个纠正／49 个退步的原图路径'),
      '- '+link(OUT/'LIVE_VERIFIED.json','真实接口验收记录'),
      '- '+link(OUT/'browser_acceptance/verification.json','浏览器回归记录'),
      '- '+link(OUT/'browser_acceptance/desktop-evidence.png','桌面取景预览截图'),
      '- '+link(OUT/'browser_acceptance/mobile-evidence.png','移动取景预览截图')]
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save_json(OUT/'HANDOFF.json',dict(deployed=True,version='SSTv1-H120-DV',date='2026-09-20',
        classifier_weights_unchanged=True,optimizer_steps=0,changed_existing_files=changed,
        all_backup_copies_verified=True,experiment=str(EXP),user_replay_regressions=user_regressions,
        formal_url=BASE if (BASE:='http://127.0.0.1:8043/') else '',
        windows_portable_updated=False,android_updated=False,report=str(OUT/'REPORT.md')))
    print('REPORT',OUT/'REPORT.md',flush=True)

if __name__=='__main__':main()
