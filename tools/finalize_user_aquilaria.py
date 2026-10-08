"""Training-side replay is explicitly separated from fixed regression evidence."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from bioclip_probe import ResidualHead
from train_sstv1_incremental import logits
from distillation_core import read_json,save_json,digest
from training_safety import sha256
ROOT=Path(__file__).resolve().parents[1]
def main(run):
    done=read_json(run/'COMPLETE.json');contract=read_json(run/'base_contract.json');cfg=read_json(run/'config.json');classes=read_json(run/'classes.json');names=read_json(run/'labels.json')
    rows=read_json(run/'feature_rows.json');ids=[i for i,r in enumerate(rows) if r.get('review_authority')=='user_confirmed_species_current_request' and r['brightness']==1.]
    with np.load(run/'features.npz') as f:
        assert str(f['rows_hash'])==digest(rows)
        x=torch.tensor(f['x'][ids].astype(np.float32));z=torch.tensor(f['z'][ids]);extra=torch.tensor(f['extra_bank']);scale=float(f['logit_scale'])
    def ensemble(members):
        pp=[]
        for member in members:
            assert sha256(member['path'])==member['sha256']
            state=torch.load(member['path'],map_location='cpu',weights_only=False);assert state['classes']==classes
            h=ResidualHead('mlp').eval();h.load_state_dict(state['model_state'])
            with torch.inference_mode():pp.append(logits(h,x,z,classes['groups'],cfg['temperature'],extra,scale).softmax(-1).numpy())
        return np.mean(pp,axis=0)
    baseline=ensemble(contract['initial_heads']);previous=ensemble(read_json(ROOT/'outputs/SSTv1_Aquilaria_20260914_v3/COMPLETE.json')['members']);candidate=ensemble(done['members'])
    stats={k:int((p.argmax(1)==11).sum()) for k,p in [('baseline',baseline),('previous_candidate',previous),('candidate',candidate)]}
    rec=[]
    for i,j in enumerate(ids):
        row=rows[j];r={'filename':row['filename'],'original_source':row['original_source'],'image_used_in_training':True}
        for name,arr in [('baseline',baseline),('previous_candidate',previous),('candidate',candidate)]:
            r[name]={'top1':names[int(arr[i].argmax())],'aquilaria_score':float(arr[i,11]),'correct':int(arr[i].argmax())==11}
        rec.append(r)
    save_json(run/'user_training_replay.json',{'n':len(ids),'correct_counts':stats,'warning':'TRAINING INPUT REPLAY ONLY. Not validation/test/generalization accuracy. No selection based on this replay.','records':rec})
    # Source originals must still match before reporting completion.
    imported=read_json(run/'user_import_manifest.json')
    for r in imported:assert sha256(r['original_source'])==r['sha256']==sha256(r['source_image'])
    verification=read_json(run/'VERIFIED.json');verification.update(user_original_files_verified_unchanged=len(imported),user_replay_separated_from_generalization=True,user_training_photos=len(ids));save_json(run/'VERIFIED.json',verification)
    old=read_json(run/'old_test_original_labels.json');audited=read_json(run/'old_test_source_audited.json');new=read_json(run/'new_author_holdout.json');demo=read_json(run/'demo_preprocessing_replay.json');prior=read_json(ROOT/'outputs/SSTv1_Aquilaria_20260914_v3/old_test_original_labels.json')
    def nfrac(a):return f"{a['correct']}/{a['positive_count']}"
    lines=['# 加入用户鉴定土沉香图片后的训练报告','',
        '已完成三个分类头各 60 轮增量训练，初始化来自当前部署 SSTv1，而非覆盖上一轮候选；新旧模型均保持 148 类与相同 BioCLIP 2.5 编码器。Demo 未自动切换。','',
        '## 用户提供数据','',
        f"原目录共 {len(imported)} 张，实际新增训练 {len(ids)} 张。6 张与已有训练数据重复，不重复计数；large (14).jpg 与旧测试图重复，留在测试侧；webphoto.jpg 为拼接展示图，暂不用于训练。84 张均已复制到版本化数据目录并校验 SHA-256，原目录未改动。",
        '这批图片的土沉香标签依据用户本轮明确鉴定。未将来源不清的图片伪称为开放许可；保留原图与水印、不公开图片、不写入 Demo 参考图库、不修改旧 human_review.sqlite3。',
        '新增图片只进入训练。相近编号、同一场景与明显同图视角合并训练权重组，不把旋转、亮度增强或截图当作独立实测树。已知重叠做排除；网络图像的未知同株、异裁切或预训练重叠仍不能保证完全排除。','',
        '## 训练配置','',
        '回放 23,581 张旧训练图＋231 张上一轮人工接受的训练图，保留上一轮自动核验的 18 张训练／5 张验证／3 张留出图片。额外加入本次 76 张用户确认图；仅训练图进行 0.7／1.0／1.2 亮度视图增强，共 653 个专项特征视图。',
        f"60 轮／分类头，学习率 0.00003，冻结编码器；原标签反例与原模型分布约束保持不变。验证选定轮数：{[r['epoch'] for r in done['members']]}。独立验证门槛建议候选：{done['validation_recommends_candidate']}，不等于已部署或已证明街景普适提升。",'',
        '## 固定数据回归对比','',
        '| 指标 | 当前 SSTv1 | 上轮候选 | 加入用户图片后 |','|---|---:|---:|---:|',
        f"| 原固定测试土沉香正确数（25 张） | {nfrac(old['aq_baseline'])} | {nfrac(prior['aq_candidate'])} | {nfrac(old['aq_candidate'])} |",
        f"| 完整旧测试中误报土沉香的其他树 | {old['aq_baseline']['false_positive']} | {prior['aq_candidate']['false_positive']} | {old['aq_candidate']['false_positive']} |",
        f"| 旧 5,146 张全部类别 Top-1 | {old['baseline']['top1']:.2%} | {prior['candidate']['top1']:.2%} | {old['candidate']['top1']:.2%} |",
        f"| 新增来源留出 3 张正确数 | {nfrac(new['aq_baseline'])} | 3/3 | {nfrac(new['aq_candidate'])} |",
        f"| Demo 实际预处理 26 张土沉香 | {nfrac(demo['baseline_aquilaria'])} | 21/26 | {nfrac(demo['candidate_aquilaria'])} |",'',
        f"来源核验版旧土沉香为 23 张：{nfrac(audited['aq_baseline'])} → {nfrac(audited['aq_candidate'])}。从 25 张剔除 2 张失去明确种级来源仅是标签审核，不是训练增益；原始固定 25 张结果始终保留。",
        '原旧测试使用历史中心裁切特征，Demo 重放使用当前灰边保比例补边与单张推理，每一行均为同图同预处理配对比较。Demo 重放只含土沉香，不能衡量其他树的误报。此前已反复查看的测试与 3 张新增留出均属于本轮回归诊断，不是新的独立盲测。','',
        '## 用户图训练后回放（不能作为精度证据）','',
        f"同一批已进入训练的 {len(ids)} 张原图，Top-1 土沉香数量：当前 SSTv1 {stats['baseline']}，上轮候选 {stats['previous_candidate']}，本轮 {stats['candidate']}。这是拟合情况，不是独立测试准确率。逐张结果见 user_training_replay.json。",'',
        '## 备份、验证与候选使用','',
        '三个分类头每次拟合前均有独立物理备份，权重／标签／划分／数据清单／配置／种子／原验证结果均已快照及校验。当前部署图像、点云和融合文件经复核未改变。备份位置：',
        *[f"- {r['backup']}" for r in done['members']],
        '',f"候选权重：{run} 下三个 member 目录的 best.pt。",f"独立推理：tools/predict_aquilaria_image.py --run {run} --image <照片路径>。不会切换 Demo。",
        'VERIFIED.json 保存备份、指标重算、当前 Demo 预处理与原模型接口一致性检查。用户原图的 84 份哈希均再次核对通过。']
    text='\n'.join(lines)+'\n';(run/'REPORT.md').write_text(text,encoding='utf8');(run/'FINAL_REPORT_ZH.md').write_text(text,encoding='utf8')
    print({'user_replay_only':stats,'n':len(ids),'fixed_test_aq':old['aq_candidate'],'demo_positive_replay':demo['candidate_aquilaria'],'validation_recommends':done['validation_recommends_candidate']})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args().run.resolve())
