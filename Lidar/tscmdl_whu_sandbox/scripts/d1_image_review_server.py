"""Serve the D1 four-class image quality review UI on localhost."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import webbrowser
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DERIVED_ROOT = PROJECT_ROOT / "lidar data" / "whu" / "derived" / "tscmdl"
sys.path.insert(0, str(SANDBOX_ROOT / "src"))

from tscmdl_whu.user_review import (  # noqa: E402
    atomic_json,
    load_manifest,
    load_or_create_review_state,
    render_record_image,
    save_user_review,
)


SCOPES = {
    "evaluation": "正式验证与核心测试",
    "training_risk": "训练高风险样本",
    "all_review": "全部待审样本",
}


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="icon" href="data:,">
<title>D1 四分类图像质检</title>
<style>
:root{color-scheme:light;--bg:#eef1f2;--panel:#fff;--line:#cdd3d6;--text:#172027;--muted:#5e6970;--blue:#1769aa;--green:#237a45;--yellow:#9a6400;--red:#b3261e}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:"Segoe UI","Microsoft YaHei",sans-serif;font-size:14px;letter-spacing:0}button,select,input,textarea{font:inherit;letter-spacing:0;border:1px solid var(--line);border-radius:5px;background:#fff;color:var(--text)}button{min-height:34px;padding:6px 12px;cursor:pointer}button:hover{border-color:#87939a}button:disabled{opacity:.45;cursor:not-allowed}
header{height:58px;padding:0 18px;display:flex;align-items:center;justify-content:space-between;background:#20282d;color:#fff}header h1{margin:0;font-size:19px;font-weight:650}#progress{color:#d8e1e5;font-variant-numeric:tabular-nums}.toolbar{min-height:54px;padding:9px 18px;display:grid;grid-template-columns:220px 220px 145px 145px minmax(180px,1fr);gap:8px;align-items:center;border-bottom:1px solid var(--line);background:#fff}.toolbar select,.toolbar input{width:100%;height:34px;padding:5px 8px}
main{height:calc(100vh - 112px);display:grid;grid-template-columns:minmax(560px,1.45fr) minmax(390px,.72fr)}.viewer{min-width:0;display:flex;flex-direction:column;border-right:1px solid var(--line)}.viewer-head{height:48px;padding:7px 14px;display:flex;align-items:center;justify-content:space-between;background:#e5e9eb;border-bottom:1px solid var(--line)}.sample-title{min-width:0;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;font-weight:650}.segmented{display:flex}.segmented button{border-radius:0;min-width:82px}.segmented button:first-child{border-radius:5px 0 0 5px}.segmented button:last-child{border-radius:0 5px 5px 0;margin-left:-1px}.segmented button.active{background:#d8e8f5;border-color:var(--blue);color:#0d4f84}.image-stage{flex:1;min-height:0;padding:12px;display:flex;align-items:center;justify-content:center;background:#30383d;position:relative}#sampleImage{max-width:100%;max-height:100%;object-fit:contain;background:#fff}#imageMessage{position:absolute;color:#fff;background:rgba(0,0,0,.68);padding:8px 12px;border-radius:4px}.viewer-nav{height:54px;padding:9px 14px;display:flex;align-items:center;justify-content:space-between;background:#fff;border-top:1px solid var(--line)}.position{color:var(--muted);font-variant-numeric:tabular-nums}
.review{overflow:auto;padding:14px 16px 24px;background:#fff}.section{padding:0 0 14px;margin:0 0 14px;border-bottom:1px solid #e2e6e8}.section:last-child{border-bottom:0}h2{margin:0 0 10px;font-size:15px}.meta{display:grid;grid-template-columns:115px 1fr;gap:6px 10px}.meta dt{color:var(--muted)}.meta dd{margin:0;overflow-wrap:anywhere}.rating{display:grid;grid-template-columns:repeat(3,1fr);gap:7px}.rating button.selected.pass{background:#dcefe3;border-color:var(--green);color:#155a31}.rating button.selected.borderline{background:#fff0cc;border-color:var(--yellow);color:#704600}.rating button.selected.reject{background:#f7ddda;border-color:var(--red);color:#7d1b16}.reasons{display:grid;grid-template-columns:1fr 1fr;gap:7px 10px}.reasons label{display:flex;gap:7px;align-items:flex-start}.reasons input{margin-top:3px}.field-label{display:block;color:var(--muted);margin:0 0 5px}#action{width:100%;height:36px;padding:5px 8px}#note{width:100%;min-height:70px;padding:8px;resize:vertical}.actions{display:grid;grid-template-columns:1fr 1fr;gap:8px}.primary{background:#1769aa;border-color:#1769aa;color:#fff}.status{min-height:38px;margin-top:10px;padding:8px;border:1px solid #d8dde0;border-radius:5px;background:#f7f9fa;color:var(--muted);overflow-wrap:anywhere}.risk{color:#8a3f00}
@media(max-width:1000px){.toolbar{grid-template-columns:1fr 1fr}main{height:auto;min-height:calc(100vh - 112px);grid-template-columns:1fr}.viewer{min-height:68vh;border-right:0;border-bottom:1px solid var(--line)}.review{overflow:visible}}
</style>
</head>
<body>
<header><h1>D1 四分类图像质检</h1><div id="progress">读取中</div></header>
<section class="toolbar">
  <select id="scope"><option value="evaluation">正式验证与核心测试</option><option value="training_risk">训练高风险样本</option><option value="all_review">全部待审样本</option></select>
  <select id="classFilter"><option value="">全部四分类</option></select>
  <select id="splitFilter"><option value="">全部划分</option><option value="train">train</option><option value="val">val</option><option value="test">test</option></select>
  <select id="statusFilter"><option value="unreviewed">未评价</option><option value="">全部状态</option><option value="pass">通过</option><option value="borderline">勉强可用</option><option value="reject">不通过</option></select>
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
      <label><input type="checkbox" value="projection_offset">点云投影错位</label><label><input type="checkbox" value="target_missing">目标树缺失</label>
      <label><input type="checkbox" value="severe_occlusion">目标严重遮挡</label><label><input type="checkbox" value="neighbor_tree">邻树干扰</label>
      <label><input type="checkbox" value="blur_or_exposure">模糊或曝光异常</label><label><input type="checkbox" value="background_clutter">背景干扰</label>
      <label><input type="checkbox" value="wrong_species">疑似树种错误</label><label><input type="checkbox" value="other">其他</label>
    </div></section>
    <section class="section"><label class="field-label" for="action">处理动作</label><select id="action"><option value="">请选择</option><option value="keep">保留图像</option><option value="exclude_image">排除该图像</option><option value="retry_alternate_view">改用其他全景视角</option><option value="manual_recrop">加入重新裁剪队列</option><option value="exclude_sample">整棵样本排除</option></select></section>
    <section class="section"><label class="field-label" for="note">备注</label><textarea id="note" placeholder="记录问题或处理要求"></textarea></section>
    <section class="section"><div class="actions"><button id="saveButton">保存评价</button><button id="saveNextButton" class="primary">保存并下一棵</button></div><div id="status" class="status">尚未提交评价</div></section>
  </aside>
</main>
<script>
const state={catalog:[],index:0,current:null,mode:'overlay',summary:null,rating:''};const $=id=>document.getElementById(id);
async function api(url,options={}){const response=await fetch(url,options);const payload=await response.json();if(!response.ok)throw new Error(payload.error||response.statusText);return payload;}
function escapeHtml(value){return String(value??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));}
function query(){const p=new URLSearchParams({scope:$('scope').value});if($('classFilter').value)p.set('class_index',$('classFilter').value);if($('splitFilter').value)p.set('split',$('splitFilter').value);if($('statusFilter').value)p.set('status',$('statusFilter').value);if($('search').value.trim())p.set('search',$('search').value.trim());return p;}
async function loadSummary(){state.summary=await api('/api/summary');for(const item of state.summary.classes){const option=document.createElement('option');option.value=item.class_index;option.textContent=`${item.class_index} · ${item.scientific_name}`;$('classFilter').appendChild(option);}updateProgress();}
function updateProgress(payload=null){const scope=$('scope').value;const total=payload?.scope_total??state.summary?.scope_counts?.[scope]??0;const reviewed=payload?.scope_reviewed??state.summary?.scope_reviewed?.[scope]??0;$('progress').textContent=`已评价 ${reviewed} / ${total}`;}
async function loadCatalog(keepKey=''){const payload=await api('/api/catalog?'+query());state.catalog=payload.items;const found=keepKey?state.catalog.findIndex(x=>x.sample_key===keepKey):-1;state.index=found>=0?found:0;updateProgress(payload);await loadCurrent();}
async function loadCurrent(){if(!state.catalog.length){state.current=null;$('sampleTitle').textContent='当前筛选没有样本';$('sampleImage').removeAttribute('src');$('imageMessage').textContent='没有可显示的样本';$('imageMessage').hidden=false;$('position').textContent='0 / 0';renderMeta();return;}state.index=Math.max(0,Math.min(state.index,state.catalog.length-1));const key=state.catalog[state.index].sample_key;state.current=await api('/api/sample/'+encodeURIComponent(key));$('sampleTitle').textContent=`${key} · ${state.current.record.model_class_name}`;$('position').textContent=`${state.index+1} / ${state.catalog.length}`;$('prevButton').disabled=state.index===0;$('nextButton').disabled=state.index>=state.catalog.length-1;renderMeta();loadReview();loadImage();}
function renderMeta(){const r=state.current?.record;if(!r){$('meta').innerHTML='';return;}const reasons=(r.automatic_risk_reasons||[]).join('、')||'无';const rows=[['四分类标签',r.model_class_name],['原始树种',r.source_scientific_name],['数据划分',r.model_split],['质检范围',r.review_scope==='evaluation'?'正式评估':'训练高风险'],['道路 / 轨迹',`${r.road_id} / ${r.trajectory_id}`],['tree ID',r.tree_id],['相机距离',`${Number(r.camera_distance_m).toFixed(2)} m`],['自动风险',reasons]];$('meta').innerHTML=rows.map(([k,v],i)=>`<dt>${escapeHtml(k)}</dt><dd class="${i===7&&reasons!=='无'?'risk':''}">${escapeHtml(v)}</dd>`).join('');}
function loadReview(){const review=state.current.review||{};state.rating=review.rating||'';document.querySelectorAll('[data-rating]').forEach(b=>b.classList.toggle('selected',b.dataset.rating===state.rating));document.querySelectorAll('.reasons input').forEach(input=>input.checked=(review.reasons||[]).includes(input.value));$('action').value=review.action||'';$('note').value=review.note||'';$('status').textContent=review.reviewed_at?`已保存：${review.reviewed_at}`:'尚未提交评价';}
function loadImage(){if(!state.current)return;$('imageMessage').hidden=false;$('imageMessage').textContent='正在载入';$('sampleImage').onload=()=>{$('imageMessage').hidden=true;};$('sampleImage').onerror=()=>{$('imageMessage').hidden=false;$('imageMessage').textContent='图像载入失败';};$('sampleImage').src=`/media/${encodeURIComponent(state.current.record.sample_key)}/${state.mode}?t=${Date.now()}`;$('overlayMode').classList.toggle('active',state.mode==='overlay');$('cropMode').classList.toggle('active',state.mode==='crop');}
function chooseRating(rating){state.rating=rating;document.querySelectorAll('[data-rating]').forEach(b=>b.classList.toggle('selected',b.dataset.rating===rating));if(rating==='pass')$('action').value='keep';else if(!$('action').value||$('action').value==='keep')$('action').value='exclude_image';}
async function save(goNext){if(!state.current)return;if(!state.rating){$('status').textContent='请先选择评价等级';return;}if(!$('action').value){$('status').textContent='请选择处理动作';return;}const key=state.current.record.sample_key;const payload={sample_key:key,scope:$('scope').value,rating:state.rating,action:$('action').value,reasons:[...document.querySelectorAll('.reasons input:checked')].map(x=>x.value),note:$('note').value};try{const result=await api('/api/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});state.summary.scope_reviewed=result.scope_reviewed_all;updateProgress(result);$('status').textContent=`已保存：${result.review.reviewed_at}`;if(goNext){if($('statusFilter').value==='unreviewed')await loadCatalog();else if(state.index<state.catalog.length-1){state.index++;await loadCurrent();}}}catch(error){$('status').textContent=error.message;}}
for(const id of ['scope','classFilter','splitFilter','statusFilter'])$(id).addEventListener('change',()=>loadCatalog());let timer;$('search').addEventListener('input',()=>{clearTimeout(timer);timer=setTimeout(()=>loadCatalog(),250);});document.querySelectorAll('[data-rating]').forEach(b=>b.addEventListener('click',()=>chooseRating(b.dataset.rating)));$('overlayMode').onclick=()=>{state.mode='overlay';loadImage();};$('cropMode').onclick=()=>{state.mode='crop';loadImage();};$('prevButton').onclick=async()=>{if(state.index>0){state.index--;await loadCurrent();}};$('nextButton').onclick=async()=>{if(state.index<state.catalog.length-1){state.index++;await loadCurrent();}};$('saveButton').onclick=()=>save(false);$('saveNextButton').onclick=()=>save(true);(async()=>{try{await loadSummary();await loadCatalog();}catch(error){$('status').textContent=error.message;$('imageMessage').textContent='启动失败';}})();
</script>
</body>
</html>"""


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


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
        self.classes = list(self.manifest.get("classes", []))

    def reviews(self) -> dict[str, object]:
        reviews = self.state.get("reviews", {})
        if not isinstance(reviews, dict):
            raise ValueError("Invalid reviews object")
        return reviews

    def scope_keys(self, scope: str) -> list[str]:
        if scope == "all_review":
            return list(self.record_by_key)
        if scope not in SCOPES:
            raise ValueError(f"Unsupported review scope: {scope}")
        return [
            str(record["sample_key"])
            for record in self.records
            if str(record.get("review_scope", "")) == scope
        ]

    def scope_summary(self) -> tuple[dict[str, int], dict[str, int]]:
        reviews = self.reviews()
        counts = {scope: len(self.scope_keys(scope)) for scope in SCOPES}
        reviewed = {
            scope: sum(key in reviews for key in self.scope_keys(scope))
            for scope in SCOPES
        }
        return counts, reviewed

    def catalog(self, query: dict[str, list[str]]) -> dict[str, object]:
        scope = query.get("scope", ["evaluation"])[0]
        records = [self.record_by_key[key] for key in self.scope_keys(scope)]
        class_filter = query.get("class_index", [""])[0]
        split_filter = query.get("split", [""])[0]
        status_filter = query.get("status", [""])[0]
        search = query.get("search", [""])[0].strip().casefold()
        reviews = self.reviews()
        items = []
        for record in records:
            key = str(record["sample_key"])
            review = reviews.get(key, {})
            if class_filter and int(record["model_class_index"]) != int(class_filter):
                continue
            if split_filter and str(record["model_split"]) != split_filter:
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
                    "model_class_index": int(record["model_class_index"]),
                    "model_class_name": str(record["model_class_name"]),
                    "split": str(record["model_split"]),
                    "rating": rating,
                }
            )
        counts, reviewed = self.scope_summary()
        return {
            "items": items,
            "filtered_count": len(items),
            "scope_total": counts[scope],
            "scope_reviewed": reviewed[scope],
        }

    def sample(self, key: str) -> dict[str, object]:
        record = self.record_by_key[key]
        fields = (
            "sample_key",
            "model_class_index",
            "model_class_name",
            "source_scientific_name",
            "model_split",
            "review_scope",
            "road_id",
            "trajectory_id",
            "tree_id",
            "camera_distance_m",
            "automatic_quality_tier",
            "automatic_risk_score",
            "automatic_risk_reasons",
        )
        return {
            "record": {field: record.get(field) for field in fields},
            "review": self.reviews().get(key),
        }

    def save(self, payload: dict[str, object]) -> dict[str, object]:
        with self.lock:
            review = save_user_review(
                self.review_path, self.state, payload, self.valid_keys
            )
            counts, reviewed = self.scope_summary()
            self.state["scope_status"] = {
                scope: {
                    "status": "complete" if reviewed[scope] == counts[scope] else "in_progress",
                    "total": counts[scope],
                    "reviewed": reviewed[scope],
                    "remaining": counts[scope] - reviewed[scope],
                }
                for scope in SCOPES
            }
            self.state["updated_at"] = timestamp()
            atomic_json(self.review_path, self.state)
        scope = str(payload.get("scope", "evaluation"))
        return {
            "review": review,
            "scope_total": counts[scope],
            "scope_reviewed": reviewed[scope],
            "scope_reviewed_all": reviewed,
        }


class ReviewHandler(BaseHTTPRequestHandler):
    server: "ReviewServer"

    def log_message(self, format: str, *args: object) -> None:
        return

    def send_json(self, value: object, status: int = 200) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
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
                payload = HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if parsed.path == "/api/summary":
                counts, reviewed = app.scope_summary()
                self.send_json(
                    {
                        "classes": app.classes,
                        "scope_names": SCOPES,
                        "scope_counts": counts,
                        "scope_reviewed": reviewed,
                        "review_path": str(app.review_path),
                    }
                )
                return
            if parsed.path == "/api/catalog":
                self.send_json(app.catalog(parse_qs(parsed.query)))
                return
            if parsed.path.startswith("/api/sample/"):
                self.send_json(
                    app.sample(unquote(parsed.path.removeprefix("/api/sample/")))
                )
                return
            if parsed.path.startswith("/media/"):
                parts = parsed.path.split("/")
                if len(parts) != 4:
                    raise ValueError("Invalid media path")
                key, mode = unquote(parts[2]), parts[3]
                if mode not in {"overlay", "crop"}:
                    raise ValueError("Invalid image mode")
                payload, mime = render_record_image(
                    app.dataset_root, app.record_by_key[key], mode
                )
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "private, max-age=3600")
                self.end_headers()
                self.wfile.write(payload)
                return
            self.send_json({"error": "Not found"}, 404)
        except Exception as error:  # noqa: BLE001
            self.send_json({"error": str(error)}, 400)

    def do_POST(self) -> None:  # noqa: N802
        try:
            if self.path == "/api/review":
                self.send_json(self.server.app.save(self.read_json()))
                return
            self.send_json({"error": "Not found"}, 404)
        except Exception as error:  # noqa: BLE001
            self.send_json({"error": str(error)}, 400)


class ReviewServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], app: ReviewApplication):
        super().__init__(address, ReviewHandler)
        self.app = app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DERIVED_ROOT / "c1_full_shared_dataset",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DERIVED_ROOT / "d1_four_class_image_quality" / "review_manifest.json",
    )
    parser.add_argument(
        "--review",
        type=Path,
        default=DERIVED_ROOT / "d1_four_class_image_quality" / "user_visual_review.json",
    )
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument(
        "--pid-file",
        type=Path,
        default=DERIVED_ROOT / "d1_four_class_image_quality" / "review_server.pid",
    )
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = ReviewApplication(
        args.dataset_root.resolve(), args.manifest.resolve(), args.review.resolve()
    )
    if args.check:
        counts, reviewed = app.scope_summary()
        first_key = next(iter(app.record_by_key))
        payload, mime = render_record_image(
            app.dataset_root, app.record_by_key[first_key], "overlay"
        )
        print(
            json.dumps(
                {
                    "status": "passed",
                    "sample_count": len(app.records),
                    "scope_counts": counts,
                    "scope_reviewed": reviewed,
                    "overlay_bytes": len(payload),
                    "overlay_mime": mime,
                    "review_path": str(app.review_path),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    server = ReviewServer(("127.0.0.1", args.port), app)
    pid_file = args.pid_file.resolve()
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()) + "\n", encoding="ascii")
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print("D1 image quality review is ready.", flush=True)
    print(f"URL: {url}", flush=True)
    print(f"Review file: {app.review_path}", flush=True)
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if pid_file.is_file() and pid_file.read_text(encoding="ascii").strip() == str(
            os.getpid()
        ):
            pid_file.unlink()


if __name__ == "__main__":
    main()
