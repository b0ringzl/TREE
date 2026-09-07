"""Local multimodal street-tree recognition acceptance application.

The application is deliberately read-only. It visualizes frozen three-seed
ensemble predictions for the D1 four-class validation/test samples and combines
them with image-risk metadata and the completed D2f point-quality review.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import threading
import webbrowser
from collections import Counter
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = (
    PROJECT_ROOT
    / "lidar data"
    / "whu"
    / "derived"
    / "tscmdl"
)
PACKAGE_ROOT = DATA_ROOT / "d1_four_class_training_package" / "20260817_165127"
POINT_SUITE = DATA_ROOT / "d1_ptv2_clean_repeats" / "20260818_protocol_v1"
IMAGE_SUITE = DATA_ROOT / "d1_resnet50_clean_repeats" / "20260817_protocol_v1"
FUSION_SUITE = DATA_ROOT / "d1_fusion_clean_repeats" / "20260818_protocol_v1"
SHENYANG_ROOT = DATA_ROOT / "e2_shenyang_four_class_inference" / "20260823_v1"
POINT_REVIEW = (
    DATA_ROOT
    / "d2_exposure_stratified_evaluation"
    / "20260821_real_point_quality_review_v1"
    / "review_state.json"
)
SEEDS = (20260728, 20260729, 20260730)


HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="icon" href="data:,"><title>行道树多模态识别应用</title>
<style>
:root{color-scheme:light;--bg:#edf1f3;--panel:#fff;--line:#d5dde1;--text:#172128;--muted:#66747c;--navy:#173746;--teal:#087f76;--blue:#1769aa;--green:#277b4e;--amber:#9a6500;--red:#b5332c;--soft:#f5f8f9}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px "Segoe UI","Microsoft YaHei",sans-serif}button,select,input{font:inherit;border:1px solid var(--line);border-radius:7px;background:#fff;color:var(--text)}button{cursor:pointer;padding:7px 12px}button:disabled{opacity:.45;cursor:not-allowed}
header{height:70px;background:linear-gradient(118deg,#152f3c,#21576a);color:#fff;padding:0 22px;display:flex;align-items:center;justify-content:space-between}h1{font-size:21px;margin:0 0 3px}.subtitle{font-size:12px;color:#d6e5ea}.badge{border:1px solid rgba(255,255,255,.35);padding:6px 10px;border-radius:999px;font-size:12px}
.toolbar{background:#fff;border-bottom:1px solid var(--line);padding:10px 16px;display:grid;grid-template-columns:190px 130px 190px 170px 170px minmax(210px,1fr) 100px;gap:8px}.toolbar select,.toolbar input{height:38px;width:100%;padding:6px 9px}.toolbar button{height:38px;background:var(--navy);color:#fff;border-color:var(--navy)}
main{height:calc(100vh - 129px);display:grid;grid-template-columns:minmax(640px,1.38fr) minmax(450px,.82fr);gap:10px;padding:10px}.left,.right{min-height:0}.left{display:grid;grid-template-rows:minmax(360px,1fr) auto;gap:10px}.visuals{background:#273138;border-radius:10px;overflow:hidden;display:grid;grid-template-columns:1.28fr 1fr;gap:1px}.visual{position:relative;background:#fff;min-width:0;display:flex;align-items:center;justify-content:center}.visual img{width:100%;height:100%;object-fit:contain}.visual-label{position:absolute;left:10px;top:10px;background:rgba(17,29,35,.78);color:#fff;padding:5px 8px;border-radius:6px;font-size:12px}.sample-head{position:absolute;right:10px;top:10px;background:rgba(255,255,255,.9);padding:5px 8px;border-radius:6px;font-weight:650}.overlay-toggle{position:absolute;left:10px;top:44px;z-index:2;background:rgba(255,255,255,.92);padding:5px 8px;font-size:12px}
.model-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.model-card{background:#fff;border:1px solid var(--line);border-radius:10px;padding:13px}.model-card h2{font-size:14px;margin:0 0 9px;color:var(--muted)}.prediction{font-size:17px;font-weight:700;min-height:43px}.confidence{font-variant-numeric:tabular-nums;color:var(--muted);margin-bottom:8px}.bar{height:7px;background:#e7ecef;border-radius:5px;overflow:hidden}.bar span{display:block;height:100%;background:var(--blue)}.model-card.point .bar span{background:var(--teal)}.model-card.fusion .bar span{background:var(--green)}.correct{color:var(--green)}.wrong{color:var(--red)}
.right{background:#fff;border:1px solid var(--line);border-radius:10px;overflow:auto;padding:15px}.right h2{font-size:16px;margin:0 0 10px}.section{padding-bottom:14px;margin-bottom:14px;border-bottom:1px solid #e5eaec}.section:last-child{border:0;margin-bottom:0}.decision{padding:12px;border-radius:9px;background:#eaf5f3;border-left:4px solid var(--teal)}.decision.review{background:#fff3d8;border-left-color:var(--amber)}.decision.warn{background:#f9e5e2;border-left-color:var(--red)}.decision-label{font-size:12px;color:var(--muted);margin-bottom:4px}.decision-main{font-size:19px;font-weight:750}.decision-note{margin-top:6px;line-height:1.5}
.meta{display:grid;grid-template-columns:125px 1fr;gap:7px 10px}.meta dt{color:var(--muted)}.meta dd{margin:0;overflow-wrap:anywhere}.chips{display:flex;flex-wrap:wrap;gap:6px}.chip{background:#eef3f5;border:1px solid #d6e0e4;padding:4px 7px;border-radius:999px;font-size:12px}.chip.risk{background:#fff0de;border-color:#efc594;color:#8d4e00}.chip.good{background:#e7f4eb;border-color:#b9d9c3;color:#27623c}.nav{display:flex;align-items:center;justify-content:space-between;gap:8px}.nav span{font-variant-numeric:tabular-nums;color:var(--muted)}.nav button{min-width:88px}.truth-toggle{display:flex;align-items:center;gap:7px}.truth-toggle input{width:16px;height:16px}
.prob-list{margin-top:9px;display:grid;gap:4px;font-size:12px}.prob-row{display:grid;grid-template-columns:minmax(110px,1fr) 55px;gap:6px}.prob-row span:last-child{text-align:right;font-variant-numeric:tabular-nums}
.empty{height:100%;display:flex;align-items:center;justify-content:center;color:var(--muted)}
@media(max-width:1150px){.toolbar{grid-template-columns:repeat(3,1fr)}main{height:auto;grid-template-columns:1fr}.left{min-height:780px}.right{overflow:visible}.visuals{min-height:520px}}
@media(max-width:700px){header{height:auto;min-height:88px;padding:13px 12px;gap:10px}h1{font-size:19px}.badge{padding:5px 8px;white-space:nowrap}.toolbar{grid-template-columns:1fr 1fr;padding:8px;gap:7px}.toolbar input{min-width:0}main{display:block;padding:7px}.left{min-height:0}.visuals{grid-template-columns:1fr;grid-template-rows:340px 340px;min-height:680px}.model-grid{grid-template-columns:1fr;margin-top:8px}.model-card{min-width:0}.prediction{overflow-wrap:anywhere}.right{margin-top:9px}.prob-row{grid-template-columns:minmax(0,1fr) 55px}}
</style></head><body>
<header><div><h1>行道树多模态识别应用</h1><div id="subtitle" class="subtitle">三随机种子概率集成 · 图像 / 点云 / 融合对照</div></div><div id="summaryBadge" class="badge">加载中</div></header>
<section class="toolbar">
  <select id="dataset"><option value="shenyang">沈阳外部推理（无树种真值）</option><option value="d1">D1 验证 / 测试集</option></select>
  <select id="split"><option value="">验证集 + 测试集</option><option value="val">验证集</option><option value="test">测试集</option></select>
  <select id="truth"><option value="">全部参考树种</option></select>
  <select id="case"><option value="">全部预测关系</option><option value="all_agree">三模态一致</option><option value="point_rescue">点云挽回图像错误</option><option value="fusion_rescue">融合挽回图像错误</option><option value="negative_transfer">点云正确但融合错误</option><option value="disagreement">三模态有分歧</option></select>
  <select id="quality"><option value="">全部影像质量</option><option value="risk">有自动风险</option><option value="normal">无自动风险</option></select>
  <input id="search" type="search" placeholder="查找 sample key / tree ID / 道路">
  <button id="apply">筛选</button>
</section>
<main>
 <section class="left">
  <div class="visuals">
   <div class="visual"><span class="visual-label">配对影像</span><button id="overlayToggle" class="overlay-toggle" hidden>显示点云叠加</button><span id="sampleKey" class="sample-head"></span><img id="image" alt="配对影像"></div>
   <div class="visual"><span class="visual-label">单树点云四视图</span><img id="point" alt="点云视图"></div>
  </div>
  <div class="model-grid">
   <article class="model-card image"><h2>单影像 · ResNet50</h2><div id="imagePred" class="prediction"></div><div id="imageConf" class="confidence"></div><div class="bar"><span id="imageBar"></span></div><div id="imageProbs" class="prob-list"></div></article>
   <article class="model-card point"><h2>单点云 · PTv2</h2><div id="pointPred" class="prediction"></div><div id="pointConf" class="confidence"></div><div class="bar"><span id="pointBar"></span></div><div id="pointProbs" class="prob-list"></div></article>
   <article class="model-card fusion"><h2>图像 + 点云 · 融合</h2><div id="fusionPred" class="prediction"></div><div id="fusionConf" class="confidence"></div><div class="bar"><span id="fusionBar"></span></div><div id="fusionProbs" class="prob-list"></div></article>
  </div>
 </section>
 <aside class="right">
  <section class="section"><h2>应用输出</h2><div id="decision" class="decision"><div class="decision-label">质量感知候选输出（待独立校准）</div><div id="decisionMain" class="decision-main"></div><div id="decisionSource" style="margin-top:4px;color:var(--muted)"></div><div id="decisionNote" class="decision-note"></div></div></section>
  <section class="section"><div class="nav"><button id="prev">上一棵</button><span id="position">0 / 0</span><button id="next">下一棵</button></div></section>
  <section class="section"><h2>样本与质量信息</h2><dl id="meta" class="meta"></dl></section>
  <section class="section"><h2>影像自动风险</h2><div id="imageRisk" class="chips"></div></section>
  <section class="section"><h2 id="pointQualityTitle">点云人工质检</h2><div id="pointQuality" class="chips"></div></section>
  <section class="section"><label class="truth-toggle"><input id="showTruth" type="checkbox" checked>显示参考标签与对错（验收模式）</label><p id="modeNote" style="color:var(--muted);line-height:1.55;margin:9px 0 0">关闭后可用于盲看预测。</p></section>
 </aside>
</main>
<script>
const $=id=>document.getElementById(id); const state={keys:[],index:0,summary:null,sample:null,overlay:false};
const labelMap={complete:'完整',slight_loss:'轻微缺失',moderate_loss:'中度缺失',severe_loss:'严重缺失',clean:'干净',slight_contamination:'轻微混入',moderate_contamination:'中度混入',severe_contamination:'严重混入',pass:'通过',caution:'谨慎可用',fail:'不通过',high:'高',medium:'中',low:'低'};
function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
async function getJson(url){const r=await fetch(url,{cache:'no-store'});if(!r.ok)throw new Error((await r.json()).error||r.statusText);return r.json();}
function params(){const p=new URLSearchParams();for(const id of ['dataset','split','truth','case','quality'])if($(id).value)p.set(id,$(id).value);if($('search').value.trim())p.set('search',$('search').value.trim());return p;}
async function loadCatalog(){const d=await getJson('/api/catalog?'+params());state.keys=d.sample_keys;state.index=0;const total=state.summary.datasets[$('dataset').value].sample_count;$('summaryBadge').textContent=`${d.count} / ${total} 株`;await loadSample();}
function resultClass(pred){if(!$('showTruth').checked||pred.correct===null||pred.correct===undefined)return '';return pred.correct?'correct':'wrong';}
function renderModel(prefix,pred){$(prefix+'Pred').textContent=pred.class_name;$(prefix+'Pred').className='prediction '+resultClass(pred);$(prefix+'Conf').textContent=`集成置信度 ${(pred.confidence*100).toFixed(1)}%`;$(prefix+'Bar').style.width=`${Math.max(1,pred.confidence*100)}%`;$(prefix+'Probs').innerHTML=pred.probabilities.map(x=>`<div class="prob-row"><span>${esc(x.class_name)}</span><span>${(x.probability*100).toFixed(1)}%</span></div>`).join('');}
function renderChips(id,items,kind=''){const node=$(id);node.innerHTML=(items&&items.length)?items.map(x=>`<span class="chip ${kind}">${esc(labelMap[x]||x)}</span>`).join(''):'<span class="chip good">无</span>';}
function renderDecision(d){const box=$('decision');box.className='decision '+(d.level||'');$('decisionMain').textContent=d.label;$('decisionSource').textContent='输出来源：'+d.source;$('decisionNote').textContent=d.note;}
function setImageMode(){if(!state.sample)return;const mode=state.overlay&&state.sample.overlay_available?'overlay':'image';$('image').src='/media/'+encodeURIComponent(state.sample.sample_key)+'/'+mode;$('overlayToggle').textContent=state.overlay?'显示原始裁剪':'显示点云叠加';}
function renderSample(s){state.sample=s;state.overlay=false;$('sampleKey').textContent=s.sample_key;$('overlayToggle').hidden=!s.overlay_available;setImageMode();$('point').src='/media/'+encodeURIComponent(s.sample_key)+'/point';renderModel('image',s.predictions.image);renderModel('point',s.predictions.point);renderModel('fusion',s.predictions.fusion);renderDecision(s.decision);const truth=s.ground_truth_available?esc(s.scientific_name):'无树种真值（不可计算准确率）';const split=s.ground_truth_available?esc(s.split):'沈阳外部域推理';const sourcePoints=s.source_point_count==null?'—':Number(s.source_point_count).toLocaleString();$('meta').innerHTML=`<dt>参考树种</dt><dd id="truthValue">${truth}</dd><dt>数据划分</dt><dd>${split}</dd><dt>道路 / 轨迹</dt><dd>${esc(s.road_id)} / ${esc(s.trajectory_id)}</dd><dt>tree ID</dt><dd>${esc(s.tree_id)}</dd><dt>三模态关系</dt><dd>${esc(s.relationship_label)}</dd><dt>源点数量</dt><dd>${sourcePoints}</dd><dt>相机距离</dt><dd>${s.camera_distance_m==null?'—':Number(s.camera_distance_m).toFixed(2)+' m'}</dd><dt>集成方式</dt><dd>3个随机种子概率平均</dd>`;renderChips('imageRisk',s.image_quality.automatic_risk_reasons,s.image_quality.automatic_risk_reasons.length?'risk':'');const q=s.point_quality;if(q){renderChips('pointQuality',[q.completeness,q.purity,q.overall_usability,'置信度 '+(labelMap[q.confidence]||q.confidence)]);}else if(!s.ground_truth_available){const pointInputs=['官方实例分割真值','固定抽样 8192 点'];if(Number(s.source_point_count)<8192)pointInputs.push('源点不足，含重复上采样');renderChips('pointQuality',pointInputs,Number(s.source_point_count)<8192?'risk':'');}else{renderChips('pointQuality',[]);}$('position').textContent=`${state.index+1} / ${state.keys.length}`;$('prev').disabled=state.index<=0;$('next').disabled=state.index>=state.keys.length-1;applyTruthMode();}
function applyTruthMode(){const show=$('showTruth').checked;if(state.sample){const tv=$('truthValue');if(tv)tv.textContent=state.sample.ground_truth_available?(show?state.sample.scientific_name:'已隐藏'):'无树种真值（不可计算准确率）';for(const p of ['image','point','fusion'])$(p+'Pred').className='prediction '+(show?resultClass(state.sample.predictions[p]):'');}}
async function loadSample(){if(!state.keys.length){state.sample=null;$('sampleKey').textContent='没有符合筛选条件的样本';$('image').removeAttribute('src');$('point').removeAttribute('src');$('position').textContent='0 / 0';return;}const key=state.keys[state.index];renderSample(await getJson('/api/sample/'+encodeURIComponent(key)));}
async function datasetChanged(){const external=$('dataset').value==='shenyang';$('split').disabled=external;$('truth').disabled=external;$('showTruth').disabled=external;$('showTruth').checked=!external;$('subtitle').textContent=external?'沈阳外部域 · 无树种真值 · 四分类候选预测 / 模态一致性':'D1 四分类验证版 · 三随机种子概率集成 · 图像 / 点云 / 融合对照';$('pointQualityTitle').textContent=external?'点云输入状态':'点云人工质检';$('modeNote').textContent=external?'沈阳数据仅含实例分割真值，不含树种标签；页面展示的是闭集候选预测、置信度和模态一致性，不能据此计算准确率。':'关闭后可用于盲看预测。D1 页面只浏览已冻结的验证/测试样本。';if(external){$('split').value='';$('truth').value='';for(const value of ['point_rescue','fusion_rescue','negative_transfer'])document.querySelector(`#case option[value="${value}"]`).disabled=true;if(['point_rescue','fusion_rescue','negative_transfer'].includes($('case').value))$('case').value='';}else{for(const value of ['point_rescue','fusion_rescue','negative_transfer'])document.querySelector(`#case option[value="${value}"]`).disabled=false;}await loadCatalog();}
$('apply').onclick=loadCatalog;$('dataset').onchange=datasetChanged;$('search').onkeydown=e=>{if(e.key==='Enter')loadCatalog()};$('prev').onclick=async()=>{if(state.index>0){state.index--;await loadSample()}};$('next').onclick=async()=>{if(state.index<state.keys.length-1){state.index++;await loadSample()}};$('showTruth').onchange=applyTruthMode;$('overlayToggle').onclick=()=>{state.overlay=!state.overlay;setImageMode();};
(async()=>{state.summary=await getJson('/api/summary');$('truth').innerHTML+=[...state.summary.classes].map(x=>`<option value="${x.class_index}">${esc(x.scientific_name)} (${x.count})</option>`).join('');await datasetChanged();})().catch(e=>{document.body.innerHTML='<pre style="padding:20px">'+esc(e.stack||e)+'</pre>'});
</script></body></html>"""


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


class RecognitionApplication:
    def __init__(self) -> None:
        package = load_json(PACKAGE_ROOT / "manifest.json")
        self.classes = [x["scientific_name"] for x in package["classes"]]
        self.records = {
            row["sample_key"]: row
            for row in package["records"]
            if row["split"] in {"val", "test"}
        }
        self.predictions = {
            "image": self._load_ensemble(IMAGE_SUITE),
            "point": self._load_ensemble(POINT_SUITE),
            "fusion": self._load_ensemble(FUSION_SUITE),
        }
        common = set(self.records)
        for values in self.predictions.values():
            common &= set(values)
        self.records = {key: self.records[key] for key in sorted(common)}
        review_payload = load_json(POINT_REVIEW)
        self.point_reviews = review_payload.get("reviews", {})
        shenyang_manifest = load_json(SHENYANG_ROOT / "manifest.json")
        self.shenyang_records = {
            row["sample_key"]: row for row in shenyang_manifest["records"]
        }
        self.shenyang_predictions = self._load_shenyang_predictions(
            SHENYANG_ROOT / "predictions.csv"
        )
        common_shenyang = set(self.shenyang_records) & set(self.shenyang_predictions)
        self.shenyang_records = {
            key: self.shenyang_records[key] for key in sorted(common_shenyang)
        }
        self.point_cache: dict[str, bytes] = {}
        self.cache_lock = threading.Lock()

    def _load_shenyang_predictions(self, path: Path) -> dict[str, dict[str, dict]]:
        result = {}
        for row in load_csv(path):
            predictions = {}
            for modality in ("image", "point", "fusion"):
                predicted = int(row[f"{modality}_predicted_class"])
                probabilities = [
                    float(row[f"{modality}_probability_{index}"])
                    for index in range(len(self.classes))
                ]
                predictions[modality] = {
                    "predicted_class": predicted,
                    "class_name": self.classes[predicted],
                    "confidence": float(probabilities[predicted]),
                    "correct": None,
                    "probabilities": [
                        {
                            "class_index": index,
                            "class_name": name,
                            "probability": probability,
                        }
                        for index, (name, probability) in enumerate(
                            zip(self.classes, probabilities)
                        )
                    ],
                }
            result[row["sample_key"]] = predictions
        return result

    def _load_ensemble(self, suite: Path) -> dict[str, dict]:
        by_key: dict[str, list[dict[str, str]]] = {}
        for seed in SEEDS:
            path = suite / "runs" / f"seed_{seed}" / "predictions.csv"
            for row in load_csv(path):
                if row["split"] not in {"val", "test"}:
                    continue
                by_key.setdefault(row["sample_key"], []).append(row)
        result = {}
        for key, rows in by_key.items():
            if len(rows) != len(SEEDS):
                continue
            probabilities = np.asarray(
                [[float(row[f"probability_{index}"]) for index in range(len(self.classes))] for row in rows],
                dtype=np.float64,
            ).mean(axis=0)
            predicted = int(np.argmax(probabilities))
            true_class = int(rows[0]["true_class"])
            result[key] = {
                "predicted_class": predicted,
                "class_name": self.classes[predicted],
                "confidence": float(probabilities[predicted]),
                "correct": predicted == true_class,
                "probabilities": [
                    {"class_index": index, "class_name": name, "probability": float(probabilities[index])}
                    for index, name in enumerate(self.classes)
                ],
            }
        return result

    def _relationship(self, key: str) -> tuple[str, str]:
        image = self.predictions["image"][key]
        point = self.predictions["point"][key]
        fusion = self.predictions["fusion"][key]
        predicted = {image["predicted_class"], point["predicted_class"], fusion["predicted_class"]}
        if len(predicted) == 1:
            return "all_agree", "三模态一致"
        if point["correct"] and (not fusion["correct"]):
            return "negative_transfer", "点云正确但融合错误"
        if (not image["correct"]) and point["correct"]:
            return "point_rescue", "点云挽回图像错误"
        if (not image["correct"]) and fusion["correct"]:
            return "fusion_rescue", "融合挽回图像错误"
        return "disagreement", "三模态有分歧"

    def _decision(self, key: str) -> dict:
        row = self.records[key]
        point_review = self.point_reviews.get(key)
        fusion = self.predictions["fusion"][key]
        point = self.predictions["point"][key]
        risks = row.get("automatic_risk_reasons") or []
        if point_review and point_review.get("overall_usability") == "fail":
            return {
                "label": fusion["class_name"],
                "class_index": fusion["predicted_class"],
                "source": "融合模型（点云质检阻断）",
                "level": "warn",
                "note": "点云人工质检不通过；当前融合结果只能作为提示，建议重新分割点云后再识别。",
            }
        if risks and point_review and point_review.get("overall_usability") == "pass" and fusion["predicted_class"] != point["predicted_class"]:
            return {
                "label": point["class_name"],
                "class_index": point["predicted_class"],
                "source": "单点云质量回退",
                "level": "review",
                "note": "影像存在自动质量风险、点云质检通过且两者预测分歧；候选规则暂回退到点云，同时保留人工复核标记。",
            }
        if fusion["predicted_class"] != point["predicted_class"]:
            return {
                "label": fusion["class_name"],
                "class_index": fusion["predicted_class"],
                "source": "融合模型",
                "level": "review",
                "note": "融合与点云预测不一致，已标记人工复核；不要只依据单一置信度自动放行。",
            }
        if risks:
            return {
                "label": fusion["class_name"],
                "class_index": fusion["predicted_class"],
                "source": "融合模型（点云一致）",
                "level": "review",
                "note": "影像存在自动质量风险，但点云与融合结果一致；建议保留质量标记并优先参考结构证据。",
            }
        return {
            "label": fusion["class_name"],
            "class_index": fusion["predicted_class"],
            "source": "融合模型",
            "level": "",
            "note": "点云与融合预测一致，且未发现已记录的严重输入质量阻断。",
        }

    def summary(self) -> dict:
        counts = Counter(int(row["class_index"]) for row in self.records.values())
        metrics = {}
        for modality, predictions in self.predictions.items():
            correct = sum(int(predictions[key]["correct"]) for key in self.records)
            metrics[modality] = correct / len(self.records)
        routing_correct = sum(
            int(self._decision(key)["class_index"] == int(self.records[key]["class_index"]))
            for key in self.records
        )
        metrics["quality_aware_candidate"] = routing_correct / len(self.records)
        return {
            "stage": "application-e1-four-class-acceptance",
            "mode": "frozen_three_seed_ensemble",
            "sample_count": len(self.records),
            "datasets": {
                "d1": {
                    "name": "D1 验证 / 测试集",
                    "sample_count": len(self.records),
                    "ground_truth_available": True,
                    "accuracy": metrics,
                },
                "shenyang": {
                    "name": "WHU-STree 沈阳外部推理",
                    "sample_count": len(self.shenyang_records),
                    "ground_truth_available": False,
                    "accuracy": None,
                    "accuracy_note": "本地沈阳数据只有实例分割标签，没有树种标签。",
                },
            },
            "seeds": list(SEEDS),
            "classes": [
                {"class_index": index, "scientific_name": name, "count": counts[index]}
                for index, name in enumerate(self.classes)
            ],
            "accuracy": metrics,
            "training_started": False,
        }

    def catalog(self, query: dict[str, list[str]]) -> dict:
        dataset = query.get("dataset", ["d1"])[0]
        if dataset == "shenyang":
            return self._catalog_shenyang(query)
        if dataset != "d1":
            raise ValueError(f"unknown dataset: {dataset}")
        split = query.get("split", [""])[0]
        truth = query.get("truth", [""])[0]
        case = query.get("case", [""])[0]
        quality = query.get("quality", [""])[0]
        search = query.get("search", [""])[0].strip().lower()
        keys = []
        for key, row in self.records.items():
            relation, _ = self._relationship(key)
            has_risk = bool(row.get("automatic_risk_reasons"))
            if split and row["split"] != split:
                continue
            if truth and int(row["class_index"]) != int(truth):
                continue
            if case and relation != case:
                continue
            if quality == "risk" and not has_risk:
                continue
            if quality == "normal" and has_risk:
                continue
            haystack = " ".join([key, str(row.get("tree_id", "")), str(row.get("road_id", "")), row["scientific_name"]]).lower()
            if search and search not in haystack:
                continue
            keys.append(key)
        return {"count": len(keys), "sample_keys": keys}

    def _catalog_shenyang(self, query: dict[str, list[str]]) -> dict:
        case = query.get("case", [""])[0]
        quality = query.get("quality", [""])[0]
        search = query.get("search", [""])[0].strip().lower()
        keys = []
        for key, row in self.shenyang_records.items():
            relation, _ = self._shenyang_relationship(key)
            has_risk = bool(row.get("automatic_risk_reasons"))
            if case and relation != case:
                continue
            if quality == "risk" and not has_risk:
                continue
            if quality == "normal" and has_risk:
                continue
            haystack = " ".join(
                [key, str(row.get("tree_id", "")), str(row.get("road_id", ""))]
            ).lower()
            if search and search not in haystack:
                continue
            keys.append(key)
        return {"count": len(keys), "sample_keys": keys}

    def _shenyang_relationship(self, key: str) -> tuple[str, str]:
        values = self.shenyang_predictions[key]
        predicted = {
            values[name]["predicted_class"] for name in ("image", "point", "fusion")
        }
        if len(predicted) == 1:
            return "all_agree", "三模态一致（无真值）"
        return "disagreement", "三模态有分歧（无真值）"

    def _shenyang_decision(self, key: str) -> dict:
        row = self.shenyang_records[key]
        predictions = self.shenyang_predictions[key]
        fusion = predictions["fusion"]
        point = predictions["point"]
        relation, _ = self._shenyang_relationship(key)
        risks = row.get("automatic_risk_reasons") or []
        if relation == "all_agree":
            return {
                "label": fusion["class_name"],
                "class_index": fusion["predicted_class"],
                "source": "三模态一致候选",
                "level": "",
                "note": "影像、点云和融合模型预测一致；但沈阳集没有树种真值，本结果只能作为待标注候选，不能换算为准确率。",
            }
        if risks and fusion["predicted_class"] == point["predicted_class"]:
            return {
                "label": fusion["class_name"],
                "class_index": fusion["predicted_class"],
                "source": "点云与融合一致候选",
                "level": "review",
                "note": "影像存在自动质量风险，点云与融合结果一致；已保留模态分歧标记，建议人工核验后再赋树种标签。",
            }
        return {
            "label": fusion["class_name"],
            "class_index": fusion["predicted_class"],
            "source": "融合模型候选",
            "level": "review",
            "note": "三模态预测不一致，且本数据没有树种真值；该闭集输出必须人工复核，不能视为正确分类。",
        }

    def sample(self, key: str) -> dict:
        if key in self.shenyang_records:
            row = self.shenyang_records[key]
            relation, relation_label = self._shenyang_relationship(key)
            return {
                "sample_key": key,
                "dataset": "shenyang",
                "ground_truth_available": False,
                "split": "external",
                "class_index": None,
                "scientific_name": "无树种真值（不可计算准确率）",
                "road_id": row.get("road_id"),
                "trajectory_id": row.get("trajectory_id"),
                "tree_id": row.get("tree_id"),
                "source_point_count": row.get("source_point_count"),
                "camera_distance_m": row.get("camera_distance_m"),
                "overlay_available": bool(row.get("overlay_path")),
                "relationship": relation,
                "relationship_label": relation_label,
                "image_quality": {
                    "automatic_quality_tier": row.get("automatic_quality_tier"),
                    "automatic_risk_score": row.get("automatic_risk_score"),
                    "automatic_risk_reasons": row.get("automatic_risk_reasons") or [],
                },
                "point_quality": None,
                "predictions": self.shenyang_predictions[key],
                "decision": self._shenyang_decision(key),
            }
        row = self.records[key]
        relation, relation_label = self._relationship(key)
        return {
            "sample_key": key,
            "dataset": "d1",
            "ground_truth_available": True,
            "split": row["split"],
            "class_index": int(row["class_index"]),
            "scientific_name": row["scientific_name"],
            "road_id": row.get("road_id"),
            "trajectory_id": row.get("trajectory_id"),
            "tree_id": row.get("tree_id"),
            "source_point_count": None,
            "camera_distance_m": row.get("camera_distance_m"),
            "overlay_available": False,
            "relationship": relation,
            "relationship_label": relation_label,
            "image_quality": {
                "automatic_quality_tier": row.get("automatic_quality_tier"),
                "automatic_risk_score": row.get("automatic_risk_score"),
                "automatic_risk_reasons": row.get("automatic_risk_reasons") or [],
            },
            "point_quality": self.point_reviews.get(key),
            "predictions": {name: values[key] for name, values in self.predictions.items()},
            "decision": self._decision(key),
        }

    def image_bytes(self, key: str) -> bytes:
        if key in self.shenyang_records:
            row = self.shenyang_records[key]
            return (SHENYANG_ROOT / row["image_path"]).read_bytes()
        row = self.records[key]
        return (PACKAGE_ROOT / row["image_path"]).read_bytes()

    def overlay_bytes(self, key: str) -> bytes:
        row = self.shenyang_records[key]
        overlay_path = row.get("overlay_path")
        if not overlay_path:
            raise FileNotFoundError(f"No overlay for {key}")
        return (SHENYANG_ROOT / overlay_path).read_bytes()

    def point_bytes(self, key: str) -> bytes:
        with self.cache_lock:
            cached = self.point_cache.get(key)
        if cached is not None:
            return cached
        if key in self.shenyang_records:
            row = self.shenyang_records[key]
            point_path = SHENYANG_ROOT / row["point_path"]
        else:
            row = self.records[key]
            point_path = PACKAGE_ROOT / row["point_path"]
        with np.load(point_path) as payload:
            points = np.asarray(payload["points_xyz"], dtype=np.float32)
        unique = np.unique(np.round(points, 6), axis=0)
        if len(unique) > 6500:
            rng = np.random.default_rng(20260823)
            unique = unique[rng.choice(len(unique), 6500, replace=False)]
        figure = plt.figure(figsize=(8.4, 6.1), dpi=135, facecolor="#ffffff")
        axes = [
            figure.add_subplot(2, 2, 1),
            figure.add_subplot(2, 2, 2),
            figure.add_subplot(2, 2, 3),
            figure.add_subplot(2, 2, 4, projection="3d"),
        ]
        views = ((0, 2, "X-Z"), (1, 2, "Y-Z"), (0, 1, "X-Y"))
        colors = unique[:, 2]
        for axis, (a, b, title) in zip(axes[:3], views):
            axis.scatter(unique[:, a], unique[:, b], c=colors, cmap="viridis", s=0.55, alpha=0.78, linewidths=0)
            axis.set_title(title, fontsize=9)
            axis.set_aspect("equal", adjustable="box")
            axis.axis("off")
        axes[3].scatter(unique[:, 0], unique[:, 1], unique[:, 2], c=colors, cmap="viridis", s=0.45, alpha=0.7, linewidths=0)
        axes[3].view_init(elev=20, azim=-58)
        axes[3].set_title("3D", fontsize=9)
        axes[3].set_axis_off()
        figure.suptitle(f"{key} · {len(unique):,} unique points", fontsize=10, color="#26353d")
        figure.tight_layout(pad=0.6)
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", bbox_inches="tight", facecolor="#ffffff")
        plt.close(figure)
        data = buffer.getvalue()
        with self.cache_lock:
            if len(self.point_cache) >= 40:
                self.point_cache.pop(next(iter(self.point_cache)))
            self.point_cache[key] = data
        return data


class Handler(BaseHTTPRequestHandler):
    application: RecognitionApplication

    def log_message(self, fmt: str, *args) -> None:
        return

    def bytes_response(self, data: bytes, content_type: str, status=HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def json_response(self, payload, status=HTTPStatus.OK) -> None:
        self.bytes_response(json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self.bytes_response(HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if parsed.path == "/api/summary":
                self.json_response(self.application.summary())
                return
            if parsed.path == "/api/catalog":
                self.json_response(self.application.catalog(parse_qs(parsed.query)))
                return
            if parsed.path.startswith("/api/sample/"):
                self.json_response(self.application.sample(unquote(parsed.path.removeprefix("/api/sample/"))))
                return
            if parsed.path.startswith("/media/"):
                _, _, encoded_key, mode = parsed.path.split("/", 3)
                key = unquote(encoded_key)
                if mode == "image":
                    self.bytes_response(self.application.image_bytes(key), "image/jpeg")
                elif mode == "overlay":
                    self.bytes_response(self.application.overlay_bytes(key), "image/jpeg")
                elif mode == "point":
                    self.bytes_response(self.application.point_bytes(key), "image/png")
                else:
                    raise ValueError("unknown media mode")
                return
            self.json_response({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except (KeyError, ValueError, FileNotFoundError) as error:
            self.json_response({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:
            self.json_response({"error": f"internal error: {error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8771)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    application = RecognitionApplication()
    Handler.application = application
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Tree species application: {url}", flush=True)
    print(json.dumps(application.summary(), ensure_ascii=False), flush=True)
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
