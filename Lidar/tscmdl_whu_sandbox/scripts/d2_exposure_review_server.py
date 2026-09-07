"""Serve the D2 automatic-exposure visual confirmation UI on localhost."""

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
DEFAULT_ROOT = (
    DERIVED_ROOT
    / "d2_exposure_stratified_evaluation"
    / "20260819_target_aware_visual_confirmation_v2"
)
sys.path.insert(0, str(SANDBOX_ROOT / "src"))

from tscmdl_whu.user_review import (  # noqa: E402
    atomic_json,
    load_manifest,
    render_record_image,
    sha256_file,
)


HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,">
<title>D2 目标树曝光检测确认</title>
<style>
:root{color-scheme:light;--bg:#eef1f2;--panel:#fff;--line:#cdd3d6;--text:#172027;--muted:#5e6970;--blue:#1769aa;--green:#237a45;--yellow:#9a6400;--red:#b3261e}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:"Segoe UI","Microsoft YaHei",sans-serif;font-size:14px}button,select,input,textarea{font:inherit;border:1px solid var(--line);border-radius:5px;background:#fff;color:var(--text)}button{min-height:34px;padding:6px 12px;cursor:pointer}button:hover{border-color:#87939a}button:disabled{opacity:.45;cursor:not-allowed}
header{height:58px;padding:0 18px;display:flex;align-items:center;justify-content:space-between;background:#20282d;color:#fff}header h1{margin:0;font-size:19px;font-weight:650}#progress{color:#d8e1e5;font-variant-numeric:tabular-nums}.toolbar{min-height:54px;padding:9px 18px;display:grid;grid-template-columns:170px 190px minmax(180px,1fr) 90px 150px minmax(180px,1.2fr);gap:8px;align-items:center;border-bottom:1px solid var(--line);background:#fff}.toolbar select,.toolbar input{width:100%;height:34px;padding:5px 8px}
main{height:calc(100vh - 112px);display:grid;grid-template-columns:minmax(560px,1.45fr) minmax(400px,.72fr)}.viewer{min-width:0;display:flex;flex-direction:column;border-right:1px solid var(--line)}.viewer-head{height:48px;padding:7px 14px;display:flex;align-items:center;justify-content:space-between;background:#e5e9eb;border-bottom:1px solid var(--line)}.sample-title{min-width:0;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;font-weight:650}.segmented{display:flex}.segmented button{border-radius:0;min-width:82px}.segmented button:first-child{border-radius:5px 0 0 5px}.segmented button:last-child{border-radius:0 5px 5px 0;margin-left:-1px}.segmented button.active{background:#d8e8f5;border-color:var(--blue);color:#0d4f84}.image-stage{flex:1;min-height:0;padding:12px;display:flex;align-items:center;justify-content:center;background:#30383d;position:relative}#sampleImage{max-width:100%;max-height:100%;object-fit:contain;background:#fff}#imageMessage{position:absolute;color:#fff;background:rgba(0,0,0,.68);padding:8px 12px;border-radius:4px}.viewer-nav{height:54px;padding:9px 14px;display:flex;align-items:center;justify-content:space-between;background:#fff;border-top:1px solid var(--line)}.position{color:var(--muted);font-variant-numeric:tabular-nums}
.review{overflow:auto;padding:14px 16px 24px;background:#fff}.section{padding:0 0 14px;margin:0 0 14px;border-bottom:1px solid #e2e6e8}.section:last-child{border-bottom:0}h2{margin:0 0 10px;font-size:15px}.meta{display:grid;grid-template-columns:125px 1fr;gap:6px 10px}.meta dt{color:var(--muted)}.meta dd{margin:0;overflow-wrap:anywhere}.confirm{display:grid;grid-template-columns:repeat(3,1fr);gap:7px}.confirm button.selected.accurate{background:#dcefe3;border-color:var(--green);color:#155a31}.confirm button.selected.uncertain{background:#fff0cc;border-color:var(--yellow);color:#704600}.confirm button.selected.false_positive{background:#f7ddda;border-color:var(--red);color:#7d1b16}.field-label{display:block;color:var(--muted);margin:0 0 5px}#actualGroup{width:100%;height:36px;padding:5px 8px}#note{width:100%;min-height:70px;padding:8px;resize:vertical}.actions{display:grid;grid-template-columns:1fr 1fr;gap:8px}.primary{background:#1769aa;border-color:#1769aa;color:#fff}.status{min-height:38px;margin-top:10px;padding:8px;border:1px solid #d8dde0;border-radius:5px;background:#f7f9fa;color:var(--muted);overflow-wrap:anywhere}.risk{color:#8a3f00;font-weight:650}
@media(max-width:1000px){.toolbar{grid-template-columns:1fr 1fr}main{height:auto;min-height:calc(100vh - 112px);grid-template-columns:1fr}.viewer{min-height:68vh;border-right:0;border-bottom:1px solid var(--line)}.review{overflow:visible}}
</style></head><body>
<header><h1>D2 目标树区域曝光检测可视化确认 v2</h1><div id="progress">读取中</div></header>
<section class="toolbar"><select id="groupFilter"><option value="">全部曝光异常</option><option value="dark">自动判定：过暗</option><option value="bright">自动判定：过亮</option></select><select id="classFilter"><option value="">全部四分类</option></select><select id="speciesFilter"><option value="">全部原始树种</option></select><select id="roadFilter"><option value="">全部道路</option></select><select id="statusFilter"><option value="unconfirmed">未确认</option><option value="">全部状态</option><option value="accurate">检测准确</option><option value="uncertain">不确定</option><option value="false_positive">检测错误</option></select><input id="search" type="search" placeholder="查找 sample key / tree ID / 树种"></section>
<main><section class="viewer"><div class="viewer-head"><div id="sampleTitle" class="sample-title">尚未选择样本</div><div class="segmented"><button id="overlayMode" class="active">点云叠加</button><button id="cropMode">原始裁剪</button></div></div><div class="image-stage"><img id="sampleImage" alt="当前曝光样本"><div id="imageMessage">正在载入</div></div><div class="viewer-nav"><button id="prevButton">上一棵</button><div id="position" class="position">0 / 0</div><button id="nextButton">下一棵</button></div></section>
<aside class="review"><section class="section"><h2>自动检测信息</h2><dl id="meta" class="meta"></dl></section><section class="section"><h2>检测结果是否准确</h2><div class="confirm"><button data-status="accurate" class="accurate">检测准确</button><button data-status="uncertain" class="uncertain">不确定</button><button data-status="false_positive" class="false_positive">检测错误</button></div></section><section class="section"><label class="field-label" for="actualGroup">你的实际判断</label><select id="actualGroup"><option value="">请选择</option><option value="dark">过暗</option><option value="bright">过亮</option><option value="normal">亮度正常</option></select></section><section class="section"><label class="field-label" for="note">备注</label><textarea id="note" placeholder="记录误报原因或其他观察"></textarea></section><section class="section"><div class="actions"><button id="saveButton">保存确认</button><button id="saveNextButton" class="primary">保存并下一棵</button></div><div id="status" class="status">尚未提交确认</div></section></aside></main>
<script>
const state={catalog:[],index:0,current:null,mode:'overlay',summary:null,confirmationStatus:''};const $=id=>document.getElementById(id);const labels={dark:'过暗',bright:'过亮',mixed:'混合异常',normal:'亮度正常'};
async function api(url,options={}){const response=await fetch(url,options);const payload=await response.json();if(!response.ok)throw new Error(payload.error||response.statusText);return payload;}function escapeHtml(value){return String(value??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));}
function query(){const p=new URLSearchParams();if($('groupFilter').value)p.set('group',$('groupFilter').value);if($('classFilter').value)p.set('class_index',$('classFilter').value);if($('speciesFilter').value)p.set('source_species',$('speciesFilter').value);if($('roadFilter').value)p.set('road',$('roadFilter').value);if($('statusFilter').value)p.set('status',$('statusFilter').value);if($('search').value.trim())p.set('search',$('search').value.trim());return p;}
async function loadSummary(){state.summary=await api('/api/summary');for(const item of state.summary.classes){const option=document.createElement('option');option.value=item.class_index;option.textContent=`${item.class_index} · ${item.scientific_name}`;$('classFilter').appendChild(option);}for(const item of state.summary.source_species){const option=document.createElement('option');option.value=item.name;option.textContent=`${item.name} (${item.count})`;$('speciesFilter').appendChild(option);}for(const item of state.summary.roads){const option=document.createElement('option');option.value=item.id;option.textContent=`道路 ${item.id} (${item.count})`;$('roadFilter').appendChild(option);}updateProgress();}
function updateProgress(payload=null){const total=payload?.total??state.summary?.sample_count??0;const confirmed=payload?.confirmed??state.summary?.confirmed_count??0;$('progress').textContent=`已确认 ${confirmed} / ${total}`;}
async function loadCatalog(keepKey=''){const payload=await api('/api/catalog?'+query());state.catalog=payload.items;const found=keepKey?state.catalog.findIndex(x=>x.sample_key===keepKey):-1;state.index=found>=0?found:0;updateProgress(payload);await loadCurrent();}
async function loadCurrent(){if(!state.catalog.length){state.current=null;$('sampleTitle').textContent='当前筛选没有样本';$('sampleImage').removeAttribute('src');$('imageMessage').textContent='没有可显示的样本';$('imageMessage').hidden=false;$('position').textContent='0 / 0';renderMeta();return;}state.index=Math.max(0,Math.min(state.index,state.catalog.length-1));const key=state.catalog[state.index].sample_key;state.current=await api('/api/sample/'+encodeURIComponent(key));const r=state.current.record;$('sampleTitle').textContent=`${key} · ${r.model_class_name} · 自动${labels[r.d2_exposure_group]}`;$('position').textContent=`${state.index+1} / ${state.catalog.length}`;$('prevButton').disabled=state.index===0;$('nextButton').disabled=state.index>=state.catalog.length-1;renderMeta();loadConfirmation();loadImage();}
function pct(v){return `${(Number(v||0)*100).toFixed(2)}%`;}function renderMeta(){const r=state.current?.record;if(!r){$('meta').innerHTML='';return;}const m=r.automatic_metrics||{};const t=r.d2_target_metrics||{};const rows=[['v2 自动判定',labels[r.d2_exposure_group]||r.d2_exposure_group],['四分类标签',r.model_class_name],['原始树种',r.source_scientific_name],['数据划分',r.model_split],['道路 / 轨迹',`${r.road_id} / ${r.trajectory_id}`],['tree ID',r.tree_id],['相机距离',`${Number(r.camera_distance_m).toFixed(2)} m`],['整图平均亮度',Number(m.mean_luminance).toFixed(2)],['整图暗像素',pct(m.dark_ratio)],['整图亮像素',pct(m.bright_ratio)],['目标区平均亮度',Number(t.target_r5_mean_luminance).toFixed(2)],['目标区暗像素',pct(t.target_r5_dark_ratio)],['目标区亮像素',pct(t.target_r5_bright_ratio)],['目标区边缘能量',Number(t.target_r5_edge_energy).toFixed(2)],['检测规则',r.d2_exposure_rule||'']];$('meta').innerHTML=rows.map(([k,v],i)=>`<dt>${escapeHtml(k)}</dt><dd class="${i===0||i>=10?'risk':''}">${escapeHtml(v)}</dd>`).join('');}
function loadConfirmation(){const c=state.current.confirmation||{};state.confirmationStatus=c.status||'';document.querySelectorAll('[data-status]').forEach(b=>b.classList.toggle('selected',b.dataset.status===state.confirmationStatus));$('actualGroup').value=c.actual_group||state.current.record.d2_exposure_group||'';$('note').value=c.note||'';$('status').textContent=c.confirmed_at?`已保存：${c.confirmed_at}`:'尚未提交确认';}
function loadImage(){if(!state.current)return;$('imageMessage').hidden=false;$('imageMessage').textContent='正在载入';$('sampleImage').onload=()=>{$('imageMessage').hidden=true;};$('sampleImage').onerror=()=>{$('imageMessage').hidden=false;$('imageMessage').textContent='图像载入失败';};$('sampleImage').src=`/media/${encodeURIComponent(state.current.record.sample_key)}/${state.mode}?t=${Date.now()}`;$('overlayMode').classList.toggle('active',state.mode==='overlay');$('cropMode').classList.toggle('active',state.mode==='crop');}
function chooseStatus(status){state.confirmationStatus=status;document.querySelectorAll('[data-status]').forEach(b=>b.classList.toggle('selected',b.dataset.status===status));if(status==='accurate')$('actualGroup').value=state.current.record.d2_exposure_group;else if(status==='false_positive')$('actualGroup').value='normal';}
async function save(goNext){if(!state.current)return;if(!state.confirmationStatus){$('status').textContent='请先选择检测结果是否准确';return;}if(!$('actualGroup').value){$('status').textContent='请选择你的实际判断';return;}const payload={sample_key:state.current.record.sample_key,status:state.confirmationStatus,actual_group:$('actualGroup').value,note:$('note').value};try{const result=await api('/api/confirm',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});state.summary.confirmed_count=result.confirmed;updateProgress(result);$('status').textContent=`已保存：${result.confirmation.confirmed_at}`;if(goNext){if($('statusFilter').value==='unconfirmed')await loadCatalog();else if(state.index<state.catalog.length-1){state.index++;await loadCurrent();}}}catch(error){$('status').textContent=error.message;}}
for(const id of ['groupFilter','classFilter','speciesFilter','roadFilter','statusFilter'])$(id).addEventListener('change',()=>loadCatalog());let timer;$('search').addEventListener('input',()=>{clearTimeout(timer);timer=setTimeout(()=>loadCatalog(),250);});document.querySelectorAll('[data-status]').forEach(b=>b.addEventListener('click',()=>chooseStatus(b.dataset.status)));$('overlayMode').onclick=()=>{state.mode='overlay';loadImage();};$('cropMode').onclick=()=>{state.mode='crop';loadImage();};$('prevButton').onclick=async()=>{if(state.index>0){state.index--;await loadCurrent();}};$('nextButton').onclick=async()=>{if(state.index<state.catalog.length-1){state.index++;await loadCurrent();}};$('saveButton').onclick=()=>save(false);$('saveNextButton').onclick=()=>save(true);(async()=>{try{await loadSummary();await loadCatalog();}catch(error){$('status').textContent=error.message;$('imageMessage').textContent='启动失败';}})();
</script></body></html>"""


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class ExposureApplication:
    def __init__(self, dataset_root: Path, manifest_path: Path, confirmation_path: Path):
        self.dataset_root = dataset_root.resolve()
        self.manifest_path = manifest_path.resolve()
        self.confirmation_path = confirmation_path.resolve()
        self.manifest, self.records = load_manifest(self.manifest_path)
        self.record_by_key = {str(record["sample_key"]): record for record in self.records}
        self.valid_keys = set(self.record_by_key)
        self.classes = list(self.manifest.get("classes", []))
        self.lock = threading.Lock()
        expected_hash = sha256_file(self.manifest_path)
        if self.confirmation_path.is_file():
            self.state = json.loads(self.confirmation_path.read_text(encoding="utf-8-sig"))
            if self.state.get("source_manifest_sha256") != expected_hash:
                raise ValueError("visualization manifest changed after confirmation started")
        else:
            now = timestamp()
            self.state = {
                "schema_version": 1,
                "stage": "D2a-automatic-exposure-visual-confirmation",
                "reviewer": "user",
                "dataset_root": str(self.dataset_root),
                "source_manifest": str(self.manifest_path),
                "source_manifest_sha256": expected_hash,
                "created_at": now,
                "updated_at": now,
                "confirmations": {},
            }
            atomic_json(self.confirmation_path, self.state)

    def confirmations(self) -> dict[str, object]:
        value = self.state.setdefault("confirmations", {})
        if not isinstance(value, dict):
            raise ValueError("invalid confirmations object")
        return value

    def catalog(self, query: dict[str, list[str]]) -> dict[str, object]:
        group_filter = query.get("group", [""])[0]
        class_filter = query.get("class_index", [""])[0]
        species_filter = query.get("source_species", [""])[0]
        road_filter = query.get("road", [""])[0]
        status_filter = query.get("status", [""])[0]
        search = query.get("search", [""])[0].strip().casefold()
        confirmations = self.confirmations()
        items = []
        for record in self.records:
            key = str(record["sample_key"])
            confirmation = confirmations.get(key, {})
            status = str(confirmation.get("status", "")) if isinstance(confirmation, dict) else ""
            if group_filter and str(record.get("d2_exposure_group")) != group_filter:
                continue
            if class_filter and int(record["model_class_index"]) != int(class_filter):
                continue
            if species_filter and str(record.get("source_scientific_name")) != species_filter:
                continue
            if road_filter and str(record.get("road_id")) != road_filter:
                continue
            if status_filter == "unconfirmed" and status:
                continue
            if status_filter in {"accurate", "uncertain", "false_positive"} and status != status_filter:
                continue
            if search and search not in (
                f"{key} {record.get('tree_id', '')} "
                f"{record.get('source_scientific_name', '')} {record.get('road_id', '')}"
            ).casefold():
                continue
            items.append(
                {
                    "sample_key": key,
                    "model_class_index": int(record["model_class_index"]),
                    "model_class_name": str(record["model_class_name"]),
                    "exposure_group": str(record["d2_exposure_group"]),
                    "status": status,
                }
            )
        return {
            "items": items,
            "filtered_count": len(items),
            "total": len(self.records),
            "confirmed": len(confirmations),
        }

    def sample(self, key: str) -> dict[str, object]:
        record = self.record_by_key[key]
        fields = (
            "sample_key",
            "model_class_index",
            "model_class_name",
            "source_scientific_name",
            "model_split",
            "road_id",
            "trajectory_id",
            "tree_id",
            "camera_distance_m",
            "automatic_risk_score",
            "automatic_risk_reasons",
            "automatic_metrics",
            "d2_exposure_group",
            "d2_target_metrics",
            "d2_detector_version",
            "d2_exposure_rule",
        )
        return {
            "record": {field: record.get(field) for field in fields},
            "confirmation": self.confirmations().get(key),
        }

    def save(self, payload: dict[str, object]) -> dict[str, object]:
        key = str(payload.get("sample_key", "")).strip()
        status = str(payload.get("status", "")).strip()
        actual_group = str(payload.get("actual_group", "")).strip()
        if key not in self.valid_keys:
            raise KeyError(f"unknown sample key: {key}")
        if status not in {"accurate", "uncertain", "false_positive"}:
            raise ValueError(f"unsupported confirmation status: {status}")
        if actual_group not in {"dark", "bright", "normal"}:
            raise ValueError(f"unsupported actual group: {actual_group}")
        confirmation = {
            "sample_key": key,
            "automatic_group": str(self.record_by_key[key]["d2_exposure_group"]),
            "status": status,
            "actual_group": actual_group,
            "note": str(payload.get("note", "")).strip(),
            "confirmed_at": timestamp(),
            "reviewer": "user",
        }
        with self.lock:
            self.confirmations()[key] = confirmation
            self.state["updated_at"] = timestamp()
            atomic_json(self.confirmation_path, self.state)
        return {
            "confirmation": confirmation,
            "total": len(self.records),
            "confirmed": len(self.confirmations()),
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
            raise ValueError("expected a JSON object")
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
                summary = app.manifest.get("summary", {})
                self.send_json(
                    {
                        "classes": app.classes,
                        "sample_count": len(app.records),
                        "confirmed_count": len(app.confirmations()),
                        "manifest_summary": summary,
                        "source_species": [
                            {"name": name, "count": count}
                            for name, count in sorted(
                                summary.get("source_species_counts", {}).items()
                            )
                        ],
                        "roads": [
                            {"id": road_id, "count": count}
                            for road_id, count in sorted(
                                summary.get("road_counts", {}).items()
                            )
                        ],
                        "confirmation_path": str(app.confirmation_path),
                    }
                )
                return
            if parsed.path == "/api/catalog":
                self.send_json(app.catalog(parse_qs(parsed.query)))
                return
            if parsed.path.startswith("/api/sample/"):
                self.send_json(app.sample(unquote(parsed.path.removeprefix("/api/sample/"))))
                return
            if parsed.path.startswith("/media/"):
                parts = parsed.path.split("/")
                if len(parts) != 4:
                    raise ValueError("invalid media path")
                key, mode = unquote(parts[2]), parts[3]
                if mode not in {"overlay", "crop"}:
                    raise ValueError("invalid image mode")
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
            self.send_json({"error": "not found"}, 404)
        except Exception as error:  # noqa: BLE001
            self.send_json({"error": str(error)}, 400)

    def do_POST(self) -> None:  # noqa: N802
        try:
            if self.path == "/api/confirm":
                self.send_json(self.server.app.save(self.read_json()))
                return
            self.send_json({"error": "not found"}, 404)
        except Exception as error:  # noqa: BLE001
            self.send_json({"error": str(error)}, 400)


class ReviewServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], app: ExposureApplication):
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
        default=DEFAULT_ROOT / "visualization_manifest.json",
    )
    parser.add_argument(
        "--confirmation",
        type=Path,
        default=DEFAULT_ROOT / "user_exposure_confirmation.json",
    )
    parser.add_argument("--port", type=int, default=8769)
    parser.add_argument(
        "--pid-file",
        type=Path,
        default=DEFAULT_ROOT / "review_server.pid",
    )
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = ExposureApplication(
        args.dataset_root.resolve(),
        args.manifest.resolve(),
        args.confirmation.resolve(),
    )
    if args.check:
        first_key = next(iter(app.record_by_key))
        overlay, overlay_mime = render_record_image(
            app.dataset_root, app.record_by_key[first_key], "overlay"
        )
        crop, crop_mime = render_record_image(
            app.dataset_root, app.record_by_key[first_key], "crop"
        )
        print(
            json.dumps(
                {
                    "status": "passed",
                    "sample_count": len(app.records),
                    "group_counts": app.manifest.get("summary", {}).get("group_counts", {}),
                    "first_sample_key": first_key,
                    "overlay_bytes": len(overlay),
                    "overlay_mime": overlay_mime,
                    "crop_bytes": len(crop),
                    "crop_mime": crop_mime,
                    "confirmation_path": str(app.confirmation_path),
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
    print("D2 automatic exposure confirmation is ready.", flush=True)
    print(f"URL: {url}", flush=True)
    print(f"Confirmation file: {app.confirmation_path}", flush=True)
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
