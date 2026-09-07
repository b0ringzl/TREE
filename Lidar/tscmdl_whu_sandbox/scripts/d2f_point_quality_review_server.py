"""Serve the D2f real point-cloud segmentation quality review UI."""

from __future__ import annotations

import argparse
import io
import json
import threading
import webbrowser
from collections import Counter
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = (
    PROJECT_ROOT
    / "lidar data"
    / "whu"
    / "derived"
    / "tscmdl"
    / "d2_exposure_stratified_evaluation"
    / "20260821_real_point_quality_review_v1"
)


HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,">
<title>D2f 真实点云分割质量复核</title><style>
:root{color-scheme:light;--bg:#edf0f1;--panel:#fff;--line:#ccd3d6;--text:#172027;--muted:#647078;--blue:#126aa5;--green:#257744;--yellow:#9a6500;--red:#b12b24}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px "Segoe UI","Microsoft YaHei",sans-serif}button,select,input,textarea{font:inherit;border:1px solid var(--line);border-radius:5px;background:#fff;color:var(--text)}button{min-height:34px;padding:6px 11px;cursor:pointer}button:hover{border-color:#819097}button:disabled{opacity:.45;cursor:not-allowed}
header{height:58px;padding:0 18px;display:flex;align-items:center;justify-content:space-between;background:#20282d;color:#fff}header h1{font-size:19px;margin:0}#progress{color:#dce4e8;font-variant-numeric:tabular-nums}.toolbar{padding:9px 14px;display:grid;grid-template-columns:120px 190px 110px 130px 130px minmax(190px,1fr);gap:8px;background:#fff;border-bottom:1px solid var(--line)}.toolbar select,.toolbar input{height:34px;padding:5px 8px;width:100%}
main{height:calc(100vh - 112px);display:grid;grid-template-columns:minmax(650px,1.45fr) minmax(430px,.75fr)}.viewer{min-width:0;display:flex;flex-direction:column;border-right:1px solid var(--line)}.viewer-head{height:48px;padding:7px 14px;background:#e4e9eb;display:flex;align-items:center;justify-content:space-between}.sample-title{font-weight:650;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.hint{color:var(--muted);font-size:12px}.visuals{flex:1;min-height:0;display:grid;grid-template-columns:1.35fr 1fr;gap:8px;padding:8px;background:#30383d}.visual-box{min-width:0;min-height:0;display:flex;align-items:center;justify-content:center;background:#fff;position:relative}.visual-box img{max-width:100%;max-height:100%;object-fit:contain}.visual-label{position:absolute;left:7px;top:7px;padding:3px 6px;border-radius:4px;background:rgba(0,0,0,.68);color:#fff}.nav{height:52px;padding:8px 14px;background:#fff;display:flex;justify-content:space-between;align-items:center}.review{overflow:auto;padding:13px 15px 22px;background:#fff}.section{padding-bottom:13px;margin-bottom:13px;border-bottom:1px solid #e2e6e8}.section:last-child{border:0}h2{font-size:15px;margin:0 0 9px}.protocol{font-size:12px;color:#475159;line-height:1.55;background:#f5f7f8;padding:8px;border-radius:5px}.meta{display:grid;grid-template-columns:125px 1fr;gap:5px 9px}.meta dt{color:var(--muted)}.meta dd{margin:0;overflow-wrap:anywhere}.grade{display:grid;grid-template-columns:repeat(5,1fr);gap:5px}.grade button{padding:5px 3px;font-size:12px}.grade button.selected{background:#d9eaf5;border-color:var(--blue);color:#0d4d78;font-weight:650}.three{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}.three button.selected.pass{background:#dcefe3;border-color:var(--green)}.three button.selected.caution{background:#fff0cc;border-color:var(--yellow)}.three button.selected.fail{background:#f7ddda;border-color:var(--red)}.checks{display:grid;grid-template-columns:1fr 1fr;gap:7px}.checks label{display:flex;gap:6px;align-items:center}.checks input{width:16px;height:16px}.field{display:block;margin:0 0 5px;color:var(--muted)}#confidence{width:100%;height:34px;padding:5px}#note{width:100%;min-height:62px;padding:7px;resize:vertical}.actions{display:grid;grid-template-columns:1fr 1fr;gap:7px}.primary{background:var(--blue);border-color:var(--blue);color:#fff}.status{margin-top:8px;min-height:35px;padding:7px;background:#f6f8f9;border:1px solid #dae0e2;border-radius:5px;color:var(--muted)}
@media(max-width:1050px){.toolbar{grid-template-columns:1fr 1fr 1fr}main{height:auto;grid-template-columns:1fr}.viewer{min-height:70vh;border-right:0}.review{overflow:visible}.visuals{min-height:600px}}
</style></head><body>
<header><h1>D2f 真实点云分割质量复核</h1><div id="progress">读取中</div></header>
<section class="toolbar"><select id="splitFilter"><option value="">全部划分</option><option value="val">验证集</option><option value="test">测试集</option></select><select id="classFilter"><option value="">全部四分类</option></select><select id="roadFilter"><option value="">全部道路</option></select><select id="riskFilter"><option value="">全部自动风险</option><option value="high">高风险</option><option value="middle">中风险</option><option value="low">低风险</option></select><select id="statusFilter"><option value="unreviewed">未复核</option><option value="">全部状态</option><option value="reviewed">已复核</option></select><input id="search" type="search" placeholder="查找 sample key / tree ID / 树种"></section>
<main><section class="viewer"><div class="viewer-head"><div id="sampleTitle" class="sample-title">尚未选择样本</div><div class="hint">模型预测在质检阶段隐藏</div></div><div class="visuals"><div class="visual-box"><span class="visual-label">点云三视图 + 立体图</span><img id="pointImage" alt="点云视图"></div><div class="visual-box"><span class="visual-label">配对影像（仅辅助定位）</span><img id="pairedImage" alt="配对影像"></div></div><div class="nav"><button id="prev">上一棵</button><span id="position">0 / 0</span><button id="next">下一棵</button></div></section>
<aside class="review"><section class="section"><h2>判断原则</h2><div class="protocol">完整度：树冠/树干结构是否连续、是否明显截断或单侧缺失。<br>纯度：是否出现明显邻树、地面、背景物或离群簇。<br>不要依据配对图像亮暗打分；不确定时选择“无法判断”，不要强行归类。</div></section><section class="section"><h2>样本信息</h2><dl id="meta" class="meta"></dl></section>
<section class="section"><h2>空间完整度</h2><div class="grade" data-field="completeness"><button data-value="complete">完整</button><button data-value="slight_loss">轻微缺失</button><button data-value="moderate_loss">中度缺失</button><button data-value="severe_loss">严重缺失</button><button data-value="unjudgeable">无法判断</button></div></section>
<section class="section"><h2>分割纯度</h2><div class="grade" data-field="purity"><button data-value="clean">干净</button><button data-value="slight_contamination">轻微混入</button><button data-value="moderate_contamination">中度混入</button><button data-value="severe_contamination">严重混入</button><button data-value="unjudgeable">无法判断</button></div></section>
<section class="section"><h2>总体可用性</h2><div class="three" data-field="overall_usability"><button class="pass" data-value="pass">通过</button><button class="caution" data-value="caution">谨慎可用</button><button class="fail" data-value="fail">不通过</button></div></section>
<section class="section"><h2>问题类型（可多选）</h2><div id="issues" class="checks"></div></section>
<section class="section"><label class="field" for="confidence">判断置信度</label><select id="confidence"><option value="">请选择</option><option value="high">高</option><option value="medium">中</option><option value="low">低</option></select></section>
<section class="section"><label class="field" for="note">备注</label><textarea id="note" placeholder="记录截断位置、疑似混入来源或无法判断原因"></textarea></section>
<section class="section"><div class="actions"><button id="save">保存</button><button id="saveNext" class="primary">保存并下一棵</button></div><div id="status" class="status">尚未提交复核</div></section></aside></main>
<script>
const state={summary:null,catalog:[],index:0,current:null,grades:{completeness:'',purity:'',overall_usability:''}};const $=id=>document.getElementById(id);const issueLabels={crown_truncated:'树冠截断',trunk_missing:'树干缺失',one_side_missing:'单侧缺失',sparse:'整体稀疏',fragmented:'结构破碎',neighbor_tree:'疑似邻树混入',ground_or_background:'地面/背景混入',outliers:'离群点/离群簇',other:'其他'};
async function api(url,options={}){const response=await fetch(url,options);const payload=await response.json();if(!response.ok)throw new Error(payload.error||response.statusText);return payload;}function esc(v){return String(v??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));}function fmt(v,n=3){return Number(v).toFixed(n);}
function query(){const p=new URLSearchParams();for(const [id,name] of [['splitFilter','split'],['classFilter','class_index'],['roadFilter','road'],['riskFilter','risk'],['statusFilter','status']])if($(id).value)p.set(name,$(id).value);if($('search').value.trim())p.set('search',$('search').value.trim());return p;}
async function loadSummary(){state.summary=await api('/api/summary');for(const item of state.summary.classes){const o=document.createElement('option');o.value=item.class_index;o.textContent=`${item.class_index} · ${item.name} (${item.count})`;$('classFilter').appendChild(o);}for(const item of state.summary.roads){const o=document.createElement('option');o.value=item.id;o.textContent=`道路 ${item.id} (${item.count})`;$('roadFilter').appendChild(o);}for(const [value,label] of Object.entries(issueLabels)){$('issues').insertAdjacentHTML('beforeend',`<label><input type="checkbox" value="${value}">${label}</label>`);}updateProgress();}
function updateProgress(payload=null){$('progress').textContent=`已复核 ${payload?.reviewed??state.summary?.reviewed_count??0} / ${payload?.total??state.summary?.sample_count??0}`;}
async function loadCatalog(keep=''){const p=await api('/api/catalog?'+query());state.catalog=p.items;const found=keep?state.catalog.findIndex(x=>x.sample_key===keep):-1;state.index=found>=0?found:0;updateProgress(p);await loadCurrent();}
async function loadCurrent(){if(!state.catalog.length){state.current=null;$('sampleTitle').textContent='当前筛选没有样本';$('pointImage').removeAttribute('src');$('pairedImage').removeAttribute('src');$('position').textContent='0 / 0';return;}state.index=Math.max(0,Math.min(state.index,state.catalog.length-1));const key=state.catalog[state.index].sample_key;state.current=await api('/api/sample/'+encodeURIComponent(key));const r=state.current.record;$('sampleTitle').textContent=`${r.sample_key} · ${r.scientific_name}`;$('position').textContent=`${state.index+1} / ${state.catalog.length}`;$('prev').disabled=state.index===0;$('next').disabled=state.index===state.catalog.length-1;$('pointImage').src=`/media/${encodeURIComponent(key)}/point?t=${r.point_sha256.slice(0,12)}`;$('pairedImage').src=`/media/${encodeURIComponent(key)}/image`;renderMeta();loadReview();}
function renderMeta(){const r=state.current.record,m=r.point_metrics;const rows=[['类别',r.scientific_name],['数据划分',r.split],['道路 / 轨迹',`${r.road_id} / ${r.trajectory_id}`],['tree ID',r.tree_id],['原始点数',m.source_point_count.toLocaleString()],['独立采样点',m.actual_sampled_unique_point_count.toLocaleString()],['重复采样比例',`${(m.repeat_fraction*100).toFixed(1)}%`],['空间跨度 X/Y/Z',`${fmt(m.robust_extent_x,2)} / ${fmt(m.robust_extent_y,2)} / ${fmt(m.robust_extent_z,2)}`],['自动风险分位',`第 ${r.automatic_point_quality_risk_decile+1} / 10 档`]];$('meta').innerHTML=rows.map(([k,v])=>`<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('');}
function setGrade(field,value){state.grades[field]=value;document.querySelectorAll(`[data-field="${field}"] button`).forEach(b=>b.classList.toggle('selected',b.dataset.value===value));}
function loadReview(){const r=state.current.review||{};for(const f of Object.keys(state.grades))setGrade(f,r[f]||'');document.querySelectorAll('#issues input').forEach(x=>x.checked=(r.issue_types||[]).includes(x.value));$('confidence').value=r.confidence||'';$('note').value=r.note||'';$('status').textContent=r.reviewed_at?`已保存：${r.reviewed_at}`:'尚未提交复核';}
async function save(goNext){if(!state.current)return;for(const [f,label] of [['completeness','空间完整度'],['purity','分割纯度'],['overall_usability','总体可用性']])if(!state.grades[f]){$('status').textContent=`请选择${label}`;return;}if(!$('confidence').value){$('status').textContent='请选择判断置信度';return;}const payload={sample_key:state.current.record.sample_key,...state.grades,issue_types:[...document.querySelectorAll('#issues input:checked')].map(x=>x.value),confidence:$('confidence').value,note:$('note').value};try{const out=await api('/api/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});state.summary.reviewed_count=out.reviewed;updateProgress(out);$('status').textContent=`已保存：${out.review.reviewed_at}`;if(goNext){if($('statusFilter').value==='unreviewed')await loadCatalog();else if(state.index<state.catalog.length-1){state.index++;await loadCurrent();}}}catch(error){$('status').textContent=error.message;}}
document.querySelectorAll('[data-field] button').forEach(b=>b.onclick=()=>setGrade(b.parentElement.dataset.field,b.dataset.value));for(const id of ['splitFilter','classFilter','roadFilter','riskFilter','statusFilter'])$(id).onchange=()=>loadCatalog();let timer;$('search').oninput=()=>{clearTimeout(timer);timer=setTimeout(()=>loadCatalog(),250);};$('prev').onclick=async()=>{if(state.index>0){state.index--;await loadCurrent();}};$('next').onclick=async()=>{if(state.index<state.catalog.length-1){state.index++;await loadCurrent();}};$('save').onclick=()=>save(false);$('saveNext').onclick=()=>save(true);(async()=>{try{await loadSummary();await loadCatalog();}catch(error){$('status').textContent='启动失败：'+error.message;}})();
</script></body></html>"""


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def render_point_views(path: Path) -> bytes:
    with np.load(path) as payload:
        points = np.asarray(payload["points_xyz"], dtype=np.float32)
    unique = np.unique(points, axis=0)
    if len(unique) > 5000:
        indices = np.linspace(0, len(unique) - 1, 5000, dtype=np.int64)
        unique = unique[indices]
    color = unique[:, 2]
    fig = plt.figure(figsize=(10.8, 7.5), facecolor="white")
    panels = [
        ("正视图 X-Z", 0, 2),
        ("侧视图 Y-Z", 1, 2),
        ("俯视图 X-Y", 0, 1),
    ]
    for index, (title, x_index, y_index) in enumerate(panels, start=1):
        axis = fig.add_subplot(2, 2, index)
        axis.scatter(unique[:, x_index], unique[:, y_index], c=color, s=1.0, cmap="viridis", alpha=0.75, linewidths=0)
        axis.set_title(title, fontsize=11)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#c8d0d3")
    axis3d = fig.add_subplot(2, 2, 4, projection="3d")
    axis3d.scatter(unique[:, 0], unique[:, 1], unique[:, 2], c=color, s=1.0, cmap="viridis", alpha=0.75, linewidths=0)
    axis3d.view_init(elev=18, azim=-65)
    axis3d.set_title("立体视图", fontsize=11)
    axis3d.set_axis_off()
    fig.suptitle(f"独立点 {len(unique):,} · 颜色表示相对高度", fontsize=12)
    fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.96))
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=135, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


class ReviewApplication:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.manifest_path = self.root / "manifest.json"
        self.state_path = self.root / "review_state.json"
        self.preview_root = self.root / "preview_cache"
        self.preview_root.mkdir(parents=True, exist_ok=True)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8-sig"))
        self.records = list(self.manifest["records"])
        self.record_by_key = {str(row["sample_key"]): row for row in self.records}
        self.state = json.loads(self.state_path.read_text(encoding="utf-8-sig"))
        self.lock = threading.Lock()

    @property
    def reviews(self) -> dict[str, object]:
        value = self.state.setdefault("reviews", {})
        if not isinstance(value, dict):
            raise ValueError("invalid reviews object")
        return value

    def summary(self) -> dict[str, object]:
        classes = Counter((int(r["class_index"]), str(r["scientific_name"])) for r in self.records)
        roads = Counter(str(r["road_id"]) for r in self.records)
        return {
            "sample_count": len(self.records),
            "reviewed_count": len(self.reviews),
            "classes": [{"class_index": key[0], "name": key[1], "count": count} for key, count in sorted(classes.items())],
            "roads": [{"id": key, "count": count} for key, count in sorted(roads.items())],
            "interpretation_limit": self.manifest.get("interpretation_limit"),
        }

    def catalog(self, query: dict[str, list[str]]) -> dict[str, object]:
        split_filter = query.get("split", [""])[0]
        class_filter = query.get("class_index", [""])[0]
        road_filter = query.get("road", [""])[0]
        risk_filter = query.get("risk", [""])[0]
        status_filter = query.get("status", [""])[0]
        search = query.get("search", [""])[0].strip().casefold()
        items = []
        for record in self.records:
            key = str(record["sample_key"])
            reviewed = key in self.reviews
            decile = int(record["automatic_point_quality_risk_decile"])
            risk_group = "high" if decile >= 7 else "middle" if decile >= 3 else "low"
            if split_filter and str(record["split"]) != split_filter:
                continue
            if class_filter and int(record["class_index"]) != int(class_filter):
                continue
            if road_filter and str(record["road_id"]) != road_filter:
                continue
            if risk_filter and risk_group != risk_filter:
                continue
            if status_filter == "unreviewed" and reviewed:
                continue
            if status_filter == "reviewed" and not reviewed:
                continue
            haystack = f"{key} {record['tree_id']} {record['scientific_name']} {record['road_id']}".casefold()
            if search and search not in haystack:
                continue
            items.append({"sample_key": key, "reviewed": reviewed})
        return {"items": items, "filtered_count": len(items), "total": len(self.records), "reviewed": len(self.reviews)}

    def sample(self, key: str) -> dict[str, object]:
        source = self.record_by_key[key]
        allowed = (
            "sample_key", "split", "class_index", "scientific_name", "road_id", "trajectory_id", "tree_id",
            "point_sha256", "automatic_point_quality_risk_score", "automatic_point_quality_risk_decile", "point_metrics"
        )
        return {"record": {field: source[field] for field in allowed}, "review": self.reviews.get(key)}

    def save(self, payload: dict[str, object]) -> dict[str, object]:
        key = str(payload.get("sample_key", ""))
        if key not in self.record_by_key:
            raise ValueError("unknown sample_key")
        protocol = self.manifest["review_protocol"]
        for field in ("completeness", "purity", "overall_usability", "confidence"):
            if str(payload.get(field, "")) not in set(protocol[field]):
                raise ValueError(f"invalid {field}")
        issues = payload.get("issue_types", [])
        if not isinstance(issues, list) or any(str(value) not in set(protocol["issue_types"]) for value in issues):
            raise ValueError("invalid issue_types")
        review = {
            "sample_key": key,
            "completeness": str(payload["completeness"]),
            "purity": str(payload["purity"]),
            "overall_usability": str(payload["overall_usability"]),
            "issue_types": sorted(set(str(value) for value in issues)),
            "confidence": str(payload["confidence"]),
            "note": str(payload.get("note", "")).strip(),
            "reviewed_at": timestamp(),
        }
        with self.lock:
            self.reviews[key] = review
            self.state["updated_at"] = timestamp()
            atomic_json(self.state_path, self.state)
        return {"review": review, "reviewed": len(self.reviews), "total": len(self.records)}

    def image_bytes(self, key: str) -> tuple[bytes, str]:
        path = Path(str(self.record_by_key[key]["image_path"]))
        return path.read_bytes(), "image/jpeg"

    def point_bytes(self, key: str) -> tuple[bytes, str]:
        record = self.record_by_key[key]
        cache_path = self.preview_root / f"{key}_{str(record['point_sha256'])[:12]}.png"
        if not cache_path.exists():
            data = render_point_views(Path(str(record["point_path"])))
            cache_path.write_bytes(data)
        return cache_path.read_bytes(), "image/png"


class Handler(BaseHTTPRequestHandler):
    application: ReviewApplication

    def log_message(self, format: str, *args: object) -> None:
        return

    def json_response(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def bytes_response(self, data: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "private, max-age=86400")
        self.end_headers()
        self.wfile.write(data)

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
                key = unquote(parsed.path.removeprefix("/api/sample/"))
                self.json_response(self.application.sample(key))
                return
            if parsed.path.startswith("/media/"):
                _, _, encoded_key, mode = parsed.path.split("/", 3)
                key = unquote(encoded_key)
                if mode == "point":
                    data, content_type = self.application.point_bytes(key)
                elif mode == "image":
                    data, content_type = self.application.image_bytes(key)
                else:
                    raise ValueError("unknown media mode")
                self.bytes_response(data, content_type)
                return
            self.json_response({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except (KeyError, ValueError, FileNotFoundError) as error:
            self.json_response({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:
            self.json_response({"error": f"internal error: {error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/review":
            self.json_response({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            self.json_response(self.application.save(payload))
        except (ValueError, KeyError, json.JSONDecodeError) as error:
            self.json_response({"error": str(error)}, HTTPStatus.BAD_REQUEST)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    application = ReviewApplication(args.root)
    Handler.application = application
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"D2f point-quality review: {url}", flush=True)
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
