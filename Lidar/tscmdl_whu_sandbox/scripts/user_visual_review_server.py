"""Launch a localhost UI for user-owned WHU-STree visual review."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SANDBOX_ROOT / "src"))

from tscmdl_whu.user_review import (  # noqa: E402
    REVIEW_SCOPE_ALL,
    REVIEW_SCOPE_QUALITY,
    complete_scope,
    discover_manifest,
    export_action_lists,
    inspect_user_review_state,
    load_manifest,
    load_or_create_review_state,
    render_record_image,
    refresh_validation_review_status,
    save_user_review,
    scope_keys,
)


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="icon" href="data:,">
<title>WHU-STree 用户视觉质检</title>
<style>
:root{color-scheme:light;--bg:#f3f5f6;--panel:#fff;--line:#cfd5d8;--text:#172026;--muted:#5d6970;--blue:#1769aa;--green:#237a45;--yellow:#a66a00;--red:#b3261e}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:"Segoe UI","Microsoft YaHei",sans-serif;font-size:14px;letter-spacing:0}
button,select,input,textarea{font:inherit;letter-spacing:0;border:1px solid var(--line);border-radius:5px;background:#fff;color:var(--text)}button{min-height:34px;padding:6px 12px;cursor:pointer}button:hover{border-color:#87939a}button:disabled{opacity:.45;cursor:not-allowed}
header{height:58px;padding:0 18px;display:flex;align-items:center;justify-content:space-between;background:#20282d;color:#fff}header h1{margin:0;font-size:19px;font-weight:650}#progress{color:#d8e1e5;font-variant-numeric:tabular-nums}
.toolbar{min-height:54px;padding:9px 18px;display:grid;grid-template-columns:180px 210px 120px 150px minmax(180px,1fr);gap:8px;align-items:center;border-bottom:1px solid var(--line);background:#fff}.toolbar select,.toolbar input{width:100%;height:34px;padding:5px 8px}
main{height:calc(100vh - 112px);display:grid;grid-template-columns:minmax(520px,1.45fr) minmax(380px,.75fr)}.viewer{min-width:0;display:flex;flex-direction:column;border-right:1px solid var(--line)}
.viewer-head{height:48px;padding:7px 14px;display:flex;align-items:center;justify-content:space-between;background:#e9edef;border-bottom:1px solid var(--line)}.sample-title{min-width:0;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;font-weight:650}
.segmented{display:flex}.segmented button{border-radius:0;min-width:76px}.segmented button:first-child{border-radius:5px 0 0 5px}.segmented button:last-child{border-radius:0 5px 5px 0;margin-left:-1px}.segmented button.active{background:#d8e8f5;border-color:var(--blue);color:#0d4f84}
.image-stage{flex:1;min-height:0;padding:12px;display:flex;align-items:center;justify-content:center;background:#30383d;position:relative}#sampleImage{max-width:100%;max-height:100%;object-fit:contain;background:#fff}#imageMessage{position:absolute;color:#fff;background:rgba(0,0,0,.66);padding:8px 12px;border-radius:4px}
.viewer-nav{height:54px;padding:9px 14px;display:flex;align-items:center;justify-content:space-between;background:#fff;border-top:1px solid var(--line)}.viewer-nav .position{color:var(--muted);font-variant-numeric:tabular-nums}
.review{overflow:auto;padding:14px 16px 24px;background:#fff}.section{padding:0 0 14px;margin:0 0 14px;border-bottom:1px solid #e2e6e8}.section:last-child{border-bottom:0}h2{margin:0 0 10px;font-size:15px}.meta{display:grid;grid-template-columns:108px 1fr;gap:6px 10px}.meta dt{color:var(--muted)}.meta dd{margin:0;overflow-wrap:anywhere}
.rating{display:grid;grid-template-columns:repeat(3,1fr);gap:7px}.rating button.selected.pass{background:#dcefe3;border-color:var(--green);color:#155a31}.rating button.selected.borderline{background:#fff0cc;border-color:var(--yellow);color:#704600}.rating button.selected.reject{background:#f7ddda;border-color:var(--red);color:#7d1b16}
.reasons{display:grid;grid-template-columns:1fr 1fr;gap:7px 10px}.reasons label{display:flex;gap:7px;align-items:flex-start}.reasons input{margin-top:3px}.field-label{display:block;color:var(--muted);margin:0 0 5px}#action{width:100%;height:36px;padding:5px 8px}#note{width:100%;min-height:78px;padding:8px;resize:vertical}
.actions,.scope-actions{display:grid;grid-template-columns:1fr 1fr;gap:8px}.primary{background:#1769aa;border-color:#1769aa;color:#fff}.danger{color:#8c211c}.status{min-height:38px;margin-top:10px;padding:8px;border:1px solid #d8dde0;border-radius:5px;background:#f7f9fa;color:var(--muted);overflow-wrap:anywhere}
@media(max-width:980px){.toolbar{grid-template-columns:1fr 1fr}main{height:auto;min-height:calc(100vh - 112px);grid-template-columns:1fr}.viewer{min-height:68vh;border-right:0;border-bottom:1px solid var(--line)}.review{overflow:visible}}
</style>
</head>
<body>
<header><h1>WHU-STree 用户视觉质检</h1><div id="progress">读取中</div></header>
<section class="toolbar">
  <select id="scope"><option value="quality_sample">分层抽检样本</option><option value="all_samples">全部 17,134 样本</option></select>
  <select id="classFilter"><option value="">全部类别</option></select>
  <select id="splitFilter"><option value="">全部划分</option><option value="train">train</option><option value="val">val</option><option value="test">test</option></select>
  <select id="statusFilter"><option value="">全部状态</option><option value="unreviewed">未评价</option><option value="pass">通过</option><option value="borderline">勉强可用</option><option value="reject">不通过</option></select>
  <input id="search" type="search" placeholder="查找 sample key / tree ID">
</section>
<main>
  <section class="viewer">
    <div class="viewer-head"><div id="sampleTitle" class="sample-title">尚未选择样本</div><div class="segmented"><button id="overlayMode" class="active">点云叠加</button><button id="cropMode">原始裁剪</button></div></div>
    <div class="image-stage"><img id="sampleImage" alt="当前质检样本"><div id="imageMessage">正在载入</div></div>
    <div class="viewer-nav"><button id="prevButton">上一棵</button><div id="position" class="position">0 / 0</div><button id="nextButton">下一棵</button></div>
  </section>
  <aside class="review">
    <section class="section"><h2>样本信息</h2><dl id="meta" class="meta"></dl></section>
    <section class="section"><h2>你的评价</h2><div class="rating"><button data-rating="pass" class="pass">通过</button><button data-rating="borderline" class="borderline">勉强可用</button><button data-rating="reject" class="reject">不通过</button></div></section>
    <section class="section"><h2>问题原因</h2><div class="reasons">
      <label><input type="checkbox" value="projection_offset">投影错位</label><label><input type="checkbox" value="target_missing">目标缺失</label>
      <label><input type="checkbox" value="severe_occlusion">严重遮挡</label><label><input type="checkbox" value="neighbor_tree">邻树干扰</label>
      <label><input type="checkbox" value="blur_or_exposure">模糊或曝光</label><label><input type="checkbox" value="background_clutter">背景干扰</label>
      <label><input type="checkbox" value="wrong_species">疑似树种错误</label><label><input type="checkbox" value="other">其他</label>
    </div></section>
    <section class="section"><label class="field-label" for="action">处理动作</label><select id="action"><option value="">请选择</option><option value="keep">保留点云与图像</option><option value="exclude_image">仅排除图像，保留点云</option><option value="retry_alternate_view">改用其他全景视角</option><option value="manual_recrop">加入重新裁剪队列</option><option value="exclude_sample">整棵从派生训练清单排除</option></select></section>
    <section class="section"><label class="field-label" for="note">你的备注</label><textarea id="note" placeholder="记录你看到的问题或处理要求"></textarea></section>
    <section class="section"><div class="actions"><button id="saveButton">保存评价</button><button id="saveNextButton" class="primary">保存并下一棵</button></div><div id="status" class="status">评价由你提交后才计入正式质检。</div></section>
    <section class="section"><h2>处理与收口</h2><div class="scope-actions"><button id="exportButton">生成处理清单</button><button id="completeButton" class="danger">完成当前范围质检</button></div></section>
  </aside>
</main>
<script>
const state={catalog:[],index:0,current:null,mode:'overlay',classes:[],summary:null,rating:''};
const $=id=>document.getElementById(id);
async function api(url,options={}){const response=await fetch(url,options);const payload=await response.json();if(!response.ok)throw new Error(payload.error||response.statusText);return payload;}
function escapeHtml(value){return String(value??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));}
async function loadSummary(){state.summary=await api('/api/summary');state.classes=state.summary.classes;const select=$('classFilter');for(const item of state.classes){const option=document.createElement('option');option.value=item.class_index;option.textContent=`${item.class_index} · ${item.scientific_name}`;select.appendChild(option);}updateProgress();}
function query(){const p=new URLSearchParams({scope:$('scope').value});if($('classFilter').value)p.set('class_index',$('classFilter').value);if($('splitFilter').value)p.set('split',$('splitFilter').value);if($('statusFilter').value)p.set('status',$('statusFilter').value);if($('search').value.trim())p.set('search',$('search').value.trim());return p;}
async function loadCatalog(keepKey=''){const payload=await api('/api/catalog?'+query());state.catalog=payload.items;const idx=keepKey?state.catalog.findIndex(x=>x.sample_key===keepKey):-1;state.index=idx>=0?idx:0;updateProgress(payload);await loadCurrent();}
function updateProgress(payload=null){const scope=$('scope').value;const total=payload?.scope_total??state.summary?.scope_counts?.[scope]??0;const reviewed=payload?.scope_reviewed??state.summary?.scope_reviewed?.[scope]??0;$('progress').textContent=`用户已评价 ${reviewed} / ${total}`;}
async function loadCurrent(){if(!state.catalog.length){state.current=null;$('sampleTitle').textContent='当前筛选无样本';$('sampleImage').removeAttribute('src');$('imageMessage').textContent='没有可显示的样本';$('imageMessage').hidden=false;$('position').textContent='0 / 0';renderMeta();return;}state.index=Math.max(0,Math.min(state.index,state.catalog.length-1));const key=state.catalog[state.index].sample_key;state.current=await api('/api/sample/'+encodeURIComponent(key));$('sampleTitle').textContent=`${key} · ${state.current.record.scientific_name}`;$('position').textContent=`${state.index+1} / ${state.catalog.length}`;$('prevButton').disabled=state.index===0;$('nextButton').disabled=state.index>=state.catalog.length-1;renderMeta();loadReview();loadImage();}
function renderMeta(){const record=state.current?.record;if(!record){$('meta').innerHTML='';return;}const rows=[['sample key',record.sample_key],['树种',record.scientific_name],['class',record.class_index],['split',record.split],['道路 / 轨迹',`${record.road_id} / ${record.trajectory_id}`],['tree ID',record.tree_id],['相机距离',`${Number(record.camera_distance_m).toFixed(2)} m`],['候选视角',record.view_candidate_count],['源点数',record.source_point_count]];$('meta').innerHTML=rows.map(([k,v])=>`<dt>${escapeHtml(k)}</dt><dd>${escapeHtml(v)}</dd>`).join('');}
function loadReview(){const review=state.current.review||{};state.rating=review.rating||'';document.querySelectorAll('[data-rating]').forEach(b=>b.classList.toggle('selected',b.dataset.rating===state.rating));document.querySelectorAll('.reasons input').forEach(input=>input.checked=(review.reasons||[]).includes(input.value));$('action').value=review.action||'';$('note').value=review.note||'';$('status').textContent=review.reviewed_at?`已保存：${review.reviewed_at}`:'尚未提交用户评价。';}
function loadImage(){if(!state.current)return;$('imageMessage').hidden=false;$('imageMessage').textContent='正在载入';$('sampleImage').onload=()=>{$('imageMessage').hidden=true;};$('sampleImage').onerror=()=>{$('imageMessage').hidden=false;$('imageMessage').textContent='图像载入失败';};$('sampleImage').src=`/media/${encodeURIComponent(state.current.record.sample_key)}/${state.mode}?t=${Date.now()}`;$('overlayMode').classList.toggle('active',state.mode==='overlay');$('cropMode').classList.toggle('active',state.mode==='crop');}
function chooseRating(rating){state.rating=rating;document.querySelectorAll('[data-rating]').forEach(b=>b.classList.toggle('selected',b.dataset.rating===rating));if(rating==='pass')$('action').value='keep';else if(!$('action').value||$('action').value==='keep')$('action').value='retry_alternate_view';}
async function save(goNext){if(!state.current)return;if(!state.rating){$('status').textContent='请先选择评价等级。';return;}if(!$('action').value){$('status').textContent='请选择处理动作。';return;}const payload={sample_key:state.current.record.sample_key,scope:$('scope').value,rating:state.rating,action:$('action').value,reasons:[...document.querySelectorAll('.reasons input:checked')].map(x=>x.value),note:$('note').value};try{const result=await api('/api/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});state.current.review=result.review;const item=state.catalog.find(x=>x.sample_key===payload.sample_key);if(item){item.rating=payload.rating;item.action=payload.action;}updateProgress(result);$('status').textContent=`已保存：${result.review.reviewed_at}`;if(goNext&&state.index<state.catalog.length-1){state.index++;await loadCurrent();}}catch(error){$('status').textContent=error.message;}}
async function complete(){if(!confirm('确认完成当前质检范围？只有该范围内每个样本都有你的评价时才会成功。'))return;try{const result=await api('/api/complete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({scope:$('scope').value})});$('status').textContent=result.scope.status==='complete'?'当前范围已由用户完成质检。':`仍有 ${result.scope.remaining} 个样本未评价。`;updateProgress(result);}catch(error){$('status').textContent=error.message;}}
async function exportLists(){try{const result=await api('/api/export',{method:'POST'});$('status').textContent=`处理清单已生成：${result.output_dir}`;}catch(error){$('status').textContent=error.message;}}
for(const id of ['scope','classFilter','splitFilter','statusFilter'])$(id).addEventListener('change',()=>loadCatalog());let timer;$('search').addEventListener('input',()=>{clearTimeout(timer);timer=setTimeout(()=>loadCatalog(),250);});document.querySelectorAll('[data-rating]').forEach(b=>b.addEventListener('click',()=>chooseRating(b.dataset.rating)));$('overlayMode').onclick=()=>{state.mode='overlay';loadImage();};$('cropMode').onclick=()=>{state.mode='crop';loadImage();};$('prevButton').onclick=async()=>{if(state.index>0){state.index--;await loadCurrent();}};$('nextButton').onclick=async()=>{if(state.index<state.catalog.length-1){state.index++;await loadCurrent();}};$('saveButton').onclick=()=>save(false);$('saveNextButton').onclick=()=>save(true);$('completeButton').onclick=complete;$('exportButton').onclick=exportLists;
(async()=>{try{await loadSummary();await loadCatalog();}catch(error){$('status').textContent=error.message;$('imageMessage').textContent='启动失败';}})();
</script>
</body>
</html>"""


class ReviewApplication:
    def __init__(self, dataset_root: Path, manifest_path: Path, review_path: Path):
        self.dataset_root = dataset_root.resolve()
        self.manifest_path = manifest_path.resolve()
        self.review_path = review_path.resolve()
        self.manifest, self.records = load_manifest(self.manifest_path)
        self.record_by_key = {
            str(record["sample_key"]): record for record in self.records
        }
        self.valid_keys = set(self.record_by_key)
        self.state = load_or_create_review_state(
            self.review_path, self.dataset_root, self.manifest_path
        )
        self.lock = threading.Lock()
        classes_path = self.dataset_root / "classes_19.json"
        if not classes_path.is_file():
            classes_path = self.dataset_root / "classes.json"
        self.classes = json.loads(classes_path.read_text(encoding="utf-8"))
        if isinstance(self.classes, dict):
            self.classes = [
                {"class_index": int(index), "scientific_name": name}
                for name, index in self.classes.items()
            ]

    def reviews(self) -> dict[str, object]:
        reviews = self.state.get("reviews", {})
        if not isinstance(reviews, dict):
            raise ValueError("Invalid reviews object")
        return reviews

    def scope_summary(self) -> tuple[dict[str, int], dict[str, int]]:
        counts: dict[str, int] = {}
        reviewed: dict[str, int] = {}
        reviews = self.reviews()
        for scope in (REVIEW_SCOPE_QUALITY, REVIEW_SCOPE_ALL):
            keys = scope_keys(self.records, scope)
            counts[scope] = len(keys)
            reviewed[scope] = sum(key in reviews for key in keys)
        return counts, reviewed

    def refresh_validation(self) -> dict[str, object]:
        status = inspect_user_review_state(
            self.review_path,
            self.manifest_path,
            self.records,
            REVIEW_SCOPE_QUALITY,
        )
        refresh_validation_review_status(
            self.dataset_root / "validation.json",
            self.review_path,
            status,
        )
        return status

    def catalog(self, query: dict[str, list[str]]) -> dict[str, object]:
        scope = query.get("scope", [REVIEW_SCOPE_QUALITY])[0]
        keys = scope_keys(self.records, scope)
        selected = [self.record_by_key[key] for key in keys]
        class_filter = query.get("class_index", [""])[0]
        split_filter = query.get("split", [""])[0]
        status_filter = query.get("status", [""])[0]
        search = query.get("search", [""])[0].strip().casefold()
        reviews = self.reviews()
        items = []
        for record in selected:
            key = str(record["sample_key"])
            review = reviews.get(key, {})
            split = str(record.get("benchmark_split", record.get("split", "")))
            if class_filter and int(record["class_index"]) != int(class_filter):
                continue
            if split_filter and split != split_filter:
                continue
            rating = str(review.get("rating", "")) if isinstance(review, dict) else ""
            if status_filter == "unreviewed" and rating:
                continue
            if status_filter in {"pass", "borderline", "reject"} and rating != status_filter:
                continue
            haystack = f"{key} {record.get('tree_id', '')}".casefold()
            if search and search not in haystack:
                continue
            items.append(
                {
                    "sample_key": key,
                    "class_index": int(record["class_index"]),
                    "scientific_name": str(record["scientific_name"]),
                    "split": split,
                    "tree_id": int(record["tree_id"]),
                    "rating": rating,
                    "action": str(review.get("action", "")) if isinstance(review, dict) else "",
                }
            )
        scope_counts, scope_reviewed = self.scope_summary()
        return {
            "items": items,
            "filtered_count": len(items),
            "scope_total": scope_counts[scope],
            "scope_reviewed": scope_reviewed[scope],
        }

    def sample(self, key: str) -> dict[str, object]:
        record = self.record_by_key[key]
        return {
            "record": {
                "sample_key": key,
                "class_index": int(record["class_index"]),
                "scientific_name": str(record["scientific_name"]),
                "split": str(record.get("benchmark_split", record.get("split", ""))),
                "road_id": str(record["road_id"]),
                "trajectory_id": str(record["trajectory_id"]),
                "tree_id": int(record["tree_id"]),
                "camera_distance_m": float(record["camera_distance_m"]),
                "view_candidate_count": int(record["view_candidate_count"]),
                "source_point_count": int(record["source_point_count"]),
            },
            "review": self.reviews().get(key),
        }


class ReviewHandler(BaseHTTPRequestHandler):
    server: "ReviewServer"

    def log_message(self, format: str, *args: object) -> None:
        return

    def json_response(self, value: object, status: int = 200) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def html_response(self) -> None:
        payload = HTML.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def read_json(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        value = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object")
        return value

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        app = self.server.app
        try:
            if parsed.path == "/":
                self.html_response()
                return
            if parsed.path == "/api/summary":
                counts, reviewed = app.scope_summary()
                self.json_response({"classes": app.classes, "scope_counts": counts, "scope_reviewed": reviewed, "review_path": str(app.review_path)})
                return
            if parsed.path == "/api/catalog":
                self.json_response(app.catalog(parse_qs(parsed.query)))
                return
            if parsed.path.startswith("/api/sample/"):
                self.json_response(app.sample(unquote(parsed.path.removeprefix("/api/sample/"))))
                return
            if parsed.path.startswith("/media/"):
                parts = parsed.path.split("/")
                if len(parts) != 4:
                    raise ValueError("Invalid media path")
                key, mode = unquote(parts[2]), parts[3]
                if mode not in {"overlay", "crop"}:
                    raise ValueError("Invalid image mode")
                payload, mime = render_record_image(app.dataset_root, app.record_by_key[key], mode)
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "private, max-age=3600")
                self.end_headers()
                self.wfile.write(payload)
                return
            self.json_response({"error": "Not found"}, 404)
        except Exception as error:  # noqa: BLE001
            self.json_response({"error": str(error)}, 400)

    def do_POST(self) -> None:  # noqa: N802
        app = self.server.app
        try:
            if self.path == "/api/review":
                payload = self.read_json()
                with app.lock:
                    review = save_user_review(app.review_path, app.state, payload, app.valid_keys)
                    app.refresh_validation()
                    counts, reviewed = app.scope_summary()
                scope = str(payload.get("scope", REVIEW_SCOPE_QUALITY))
                self.json_response({"review": review, "scope_total": counts[scope], "scope_reviewed": reviewed[scope]})
                return
            if self.path == "/api/complete":
                payload = self.read_json()
                scope = str(payload.get("scope", REVIEW_SCOPE_QUALITY))
                with app.lock:
                    scope_state = complete_scope(app.review_path, app.state, app.records, scope)
                    app.refresh_validation()
                    counts, reviewed = app.scope_summary()
                self.json_response({"scope": scope_state, "scope_total": counts[scope], "scope_reviewed": reviewed[scope]})
                return
            if self.path == "/api/export":
                with app.lock:
                    output_dir = app.dataset_root / "user_review_outputs"
                    summary = export_action_lists(output_dir, app.records, app.state)
                    app.refresh_validation()
                self.json_response({"summary": summary, "output_dir": str(output_dir)})
                return
            self.json_response({"error": "Not found"}, 404)
        except Exception as error:  # noqa: BLE001
            self.json_response({"error": str(error)}, 400)


class ReviewServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], app: ReviewApplication):
        super().__init__(address, ReviewHandler)
        self.app = app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "lidar data" / "whu" / "derived" / "tscmdl" / "c1_full_shared_dataset",
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    manifest_path = args.manifest.resolve() if args.manifest is not None else discover_manifest(dataset_root)
    review_path = args.review.resolve() if args.review is not None else dataset_root / "user_visual_review.json"
    app = ReviewApplication(dataset_root, manifest_path, review_path)
    counts, reviewed = app.scope_summary()
    if args.check:
        keys = scope_keys(app.records, REVIEW_SCOPE_QUALITY) or scope_keys(app.records, REVIEW_SCOPE_ALL)
        payload, mime = render_record_image(dataset_root, app.record_by_key[keys[0]], "overlay")
        print(json.dumps({"status": "passed", "sample_count": len(app.records), "quality_scope_count": counts[REVIEW_SCOPE_QUALITY], "quality_scope_reviewed": reviewed[REVIEW_SCOPE_QUALITY], "overlay_bytes": len(payload), "overlay_mime": mime, "review_path": str(review_path)}, ensure_ascii=False, indent=2))
        return
    server = ReviewServer(("127.0.0.1", args.port), app)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print("WHU-STree user visual review is ready.", flush=True)
    print(f"URL: {url}", flush=True)
    print(f"Review file: {review_path}", flush=True)
    print("Press Ctrl+C in this window to stop the review server.", flush=True)
    if not args.no_browser:
        threading.Timer(0.7, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
