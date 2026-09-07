import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  AlertTriangle,
  ArrowLeft,
  ArrowRight,
  BarChart3,
  Check,
  ChevronDown,
  ChevronUp,
  Eraser,
  Eye,
  KeyRound,
  Loader2,
  Maximize2,
  Minimize2,
  Play,
  RotateCcw,
  Save,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Target,
  X,
} from "lucide-react";
import {
  appendCandidatePolygon,
  buildFreehandPolygon,
  contextMenuHitTest,
  pixelsToPoint,
  polygonHitTest,
  polygonToPixels,
  vertexHitTest,
} from "./annotationGeometry.js";
import "./styles.css";

const API = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8021";

function imageLabel(path) {
  const name = path.split("/").pop() || path;
  return name.replace(".jpg", "").replaceAll("_", " ");
}

function shotSummary(shot) {
  if (!shot) return "No camera metadata";
  return `date ${formatStreetViewDate(shot.date)} | heading ${Number(shot.heading).toFixed(1)} deg | pitch ${Number(shot.pitch).toFixed(1)} deg | fov ${shot.fov} | distance ${Number(shot.distance_m).toFixed(1)} m`;
}

function formatNumber(value, digits = 6) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "-";
}

function formatStreetViewDate(value) {
  if (!value) return "Unknown";
  const text = String(value).trim();
  if (!text) return "Unknown";
  const parts = text.split("-");
  if (parts.length >= 2) return `${parts[0]}-${parts[1].padStart(2, "0")}`;
  return text;
}

function defaultPreprocess() {
  return {
    recipe_version: "photo_v1",
    exposure_ev: 0,
    contrast: 1,
    reason: "normal",
    auto_suggested: false,
    human_accepted: false,
    quality_before: {},
    quality_after: {},
  };
}

function defaultVisibility() {
  return { status: "clear", reason: "none", note: "" };
}

function normalizePreprocess(value) {
  return {
    ...defaultPreprocess(),
    ...(value || {}),
    quality_before: value?.quality_before || {},
    quality_after: value?.quality_after || {},
  };
}

function preprocessReason(exposureEv, contrast) {
  if (Number(exposureEv) > 0.05) return "underexposed";
  if (Number(exposureEv) < -0.05) return "overexposed";
  if (Math.abs(Number(contrast) - 1) > 0.05) return "low_contrast";
  return "normal";
}

function percent(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${(number * 100).toFixed(1)}%` : "-";
}

function cameraPoint(shot, maxDistance, size = 220) {
  const center = size / 2;
  const radius = Math.max(24, (Number(shot.distance_m) / maxDistance) * 82);
  const bearing = ((Number(shot.heading) + 180) % 360) * (Math.PI / 180);
  return {
    x: center + Math.sin(bearing) * radius,
    y: center - Math.cos(bearing) * radius,
    radius,
  };
}

function ViewGeometryPanel({ shots }) {
  const validShots = (shots || []).filter(Boolean);
  if (!validShots.length) return null;
  const maxDistance = Math.max(10, ...validShots.map((shot) => Number(shot.distance_m) || 0));

  return (
    <section className="geometry-panel" title="Animated map of the three Street View sampling positions around the target tree">
      <div className="geometry-copy">
        <strong>Sampling geometry</strong>
        <span>Animated rays show each camera position, distance, and viewing angle toward the same target tree.</span>
      </div>
      <svg viewBox="0 0 220 220" role="img" aria-label="Street View sampling geometry">
        <circle className="geo-ring" cx="110" cy="110" r="32" />
        <circle className="geo-ring" cx="110" cy="110" r="64" />
        <circle className="geo-ring" cx="110" cy="110" r="92" />
        <line className="geo-north-line" x1="110" y1="20" x2="110" y2="200" />
        <line className="geo-north-line" x1="20" y1="110" x2="200" y2="110" />
        <circle className="geo-tree" cx="110" cy="110" r="8" />
        <text className="geo-tree-label" x="110" y="102" textAnchor="middle">TREE</text>
        {validShots.map((shot, index) => {
          const point = cameraPoint(shot, maxDistance);
          return (
            <g className="geo-shot" key={`${shot.filename || index}-${shot.pano_id || ""}`} style={{ "--delay": `${index * 0.18}s` }}>
              <line className="geo-ray" x1={point.x} y1={point.y} x2="110" y2="110" />
              <circle className="geo-camera" cx={point.x} cy={point.y} r="7" />
              <text className="geo-label" x={point.x} y={point.y - 12} textAnchor="middle">
                {index + 1}: {Number(shot.distance_m).toFixed(1)}m
              </text>
              <text className="geo-angle" x={(point.x + 110) / 2} y={(point.y + 110) / 2 - 5} textAnchor="middle">
                {Number(shot.heading).toFixed(0)} deg
              </text>
            </g>
          );
        })}
      </svg>
    </section>
  );
}


function CanvasAnnotator({
  image,
  polygons,
  draftPoints,
  preprocess,
  showOriginal = false,
  selectedPolygon,
  selectedVertex,
  samAssistEnabled = false,
  samAssistBusy = false,
  onPolygonsChange,
  onDraftChange,
  onSelect,
  onSamAssistPoint,
}) {
  const canvasRef = useRef(null);
  const imageRef = useRef(null);
  const dragRef = useRef(null);
  const pressRef = useRef(null);
  const [freehandPoints, setFreehandPoints] = useState([]);
  const [contextMenu, setContextMenu] = useState(null);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imageRef.current;
    if (!canvas || !img || !img.complete) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const exposureEv = showOriginal ? 0 : Number(preprocess?.exposure_ev || 0);
    const contrast = showOriginal ? 1 : Number(preprocess?.contrast || 1);
    ctx.save();
    ctx.filter = `brightness(${2 ** exposureEv}) contrast(${contrast})`;
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    ctx.restore();
    ctx.filter = "none";

    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    ctx.save();
    ctx.fillStyle = "rgba(239, 68, 68, 0.08)";
    ctx.fillRect(cx - 100, cy - 100, 200, 200);
    ctx.setLineDash([7, 6]);
    ctx.strokeStyle = "rgba(239, 68, 68, 0.78)";
    ctx.lineWidth = 2;
    ctx.strokeRect(cx - 100, cy - 100, 200, 200);
    ctx.setLineDash([]);
    ctx.strokeStyle = "rgba(239, 68, 68, 0.95)";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(cx - 36, cy);
    ctx.lineTo(cx - 8, cy);
    ctx.moveTo(cx + 8, cy);
    ctx.lineTo(cx + 36, cy);
    ctx.moveTo(cx, cy - 36);
    ctx.lineTo(cx, cy - 8);
    ctx.moveTo(cx, cy + 8);
    ctx.lineTo(cx, cy + 36);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(cx, cy, 5, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();

    polygons.forEach((polygon, index) => {
      const points = polygonToPixels(polygon.points, canvas.width, canvas.height);
      const isActive = selectedPolygon === index;
      ctx.save();
      ctx.strokeStyle = isActive ? "#16a34a" : "#facc15";
      ctx.fillStyle = isActive ? "rgba(22, 163, 74, 0.14)" : "rgba(250, 204, 21, 0.12)";
      ctx.lineWidth = isActive ? 3 : 2;
      ctx.beginPath();
      points.forEach(([x, y], pointIndex) => {
        if (pointIndex === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = isActive ? "#16a34a" : "#facc15";
      points.forEach(([x, y], pointIndex) => {
        ctx.beginPath();
        ctx.arc(x, y, isActive && selectedVertex === pointIndex ? 7 : 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      });
      ctx.restore();
    });

    const draft = polygonToPixels(draftPoints, canvas.width, canvas.height);
    if (draft.length) {
      ctx.save();
      ctx.strokeStyle = "#0f766e";
      ctx.fillStyle = "#0f766e";
      ctx.lineWidth = 2.5;
      ctx.setLineDash([6, 5]);
      ctx.beginPath();
      draft.forEach(([x, y], index) => {
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.setLineDash([]);
      draft.forEach(([x, y], index) => {
        ctx.beginPath();
        ctx.arc(x, y, index === 0 && draft.length >= 3 ? 8 : 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "rgba(255, 255, 255, 0.95)";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      });
      ctx.restore();
    }

    const freehand = polygonToPixels(freehandPoints, canvas.width, canvas.height);
    if (freehand.length) {
      ctx.save();
      ctx.strokeStyle = "#0f766e";
      ctx.fillStyle = "rgba(15, 118, 110, 0.12)";
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      freehand.forEach(([x, y], index) => {
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.restore();
    }
  }, [draftPoints, freehandPoints, polygons, preprocess, selectedPolygon, selectedVertex, showOriginal]);

  useEffect(() => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src = `${API}${image}`;
    imageRef.current = img;
    img.onload = draw;
  }, [image, draw]);

  useEffect(() => {
    draw();
  }, [draw]);

  function point(event) {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    return {
      x: ((event.clientX - rect.left) / rect.width) * canvas.width,
      y: ((event.clientY - rect.top) / rect.height) * canvas.height,
    };
  }

  function finishDraft() {
    if (draftPoints.length < 3) return;
    const next = [...polygons, { class_id: 0, points: draftPoints }];
    onPolygonsChange(next);
    onDraftChange([]);
    onSelect(next.length - 1, -1);
  }

  function clearPendingPress() {
    const press = pressRef.current;
    if (press?.timer) window.clearTimeout(press.timer);
    pressRef.current = null;
    setFreehandPoints([]);
  }

  function finishFreehand(pointOverride = null) {
    const press = pressRef.current;
    if (!press?.active) return false;
    const canvas = canvasRef.current;
    const points = pointOverride ? [...press.points, pixelsToPoint(pointOverride, canvas.width, canvas.height)] : press.points;
    const polygon = buildFreehandPolygon(points, canvas.width, canvas.height);
    clearPendingPress();
    if (!polygon) return true;
    const next = [...polygons, polygon];
    onPolygonsChange(next);
    onDraftChange([]);
    onSelect(next.length - 1, -1);
    return true;
  }

  function onMouseDown(event) {
    if (event.button !== 0) return;
    setContextMenu(null);
    const canvas = canvasRef.current;
    const p = point(event);
    if (event.detail > 1) {
      finishDraft();
      return;
    }

    const vertexHit = vertexHitTest(polygons, p.x, p.y, canvas.width, canvas.height);
    if (vertexHit) {
      onSelect(vertexHit.polygonIndex, vertexHit.vertexIndex);
      dragRef.current = vertexHit;
      return;
    }

    if (draftPoints.length >= 3) {
      const [firstX, firstY] = polygonToPixels([draftPoints[0]], canvas.width, canvas.height)[0];
      if (Math.hypot(p.x - firstX, p.y - firstY) <= 13) {
        finishDraft();
        return;
      }
    }

    const polygonIndex = polygonHitTest(polygons, p.x, p.y, canvas.width, canvas.height);
    if (polygonIndex >= 0) {
      onSelect(polygonIndex, -1);
      return;
    }

    const firstPoint = pixelsToPoint(p, canvas.width, canvas.height);
    if (samAssistEnabled && draftPoints.length === 0 && onSamAssistPoint) {
      onSelect(-1, -1);
      onSamAssistPoint(firstPoint);
      return;
    }

    pressRef.current = {
      active: false,
      start: p,
      points: [firstPoint],
      timer: window.setTimeout(() => {
        if (!pressRef.current) return;
        pressRef.current.active = true;
        setFreehandPoints([...pressRef.current.points]);
      }, 260),
    };
  }

  function onMouseMove(event) {
    const drag = dragRef.current;
    const canvas = canvasRef.current;
    const p = point(event);
    if (drag) {
      const next = polygons.map((polygon, polygonIndex) => {
        if (polygonIndex !== drag.polygonIndex) return polygon;
        return {
          ...polygon,
          points: polygon.points.map((point, vertexIndex) => (vertexIndex === drag.vertexIndex ? pixelsToPoint(p, canvas.width, canvas.height) : point)),
        };
      });
      onPolygonsChange(next);
      return;
    }

    const press = pressRef.current;
    if (!press?.active) return;
    const nextPoint = pixelsToPoint(p, canvas.width, canvas.height);
    press.points = [...press.points, nextPoint];
    setFreehandPoints(press.points);
  }

  function onMouseUp(event) {
    const press = pressRef.current;
    dragRef.current = null;
    if (!press) return;
    const canvas = canvasRef.current;
    const p = point(event);
    if (finishFreehand(p)) return;
    if (press.timer) window.clearTimeout(press.timer);
    pressRef.current = null;
    onDraftChange([...draftPoints, pixelsToPoint(p, canvas.width, canvas.height)]);
    onSelect(-1, -1);
  }

  function onContextMenu(event) {
    event.preventDefault();
    clearPendingPress();
    dragRef.current = null;
    const canvas = canvasRef.current;
    const p = point(event);
    const hit = contextMenuHitTest({ polygons, candidates: [], x: p.x, y: p.y, width: canvas.width, height: canvas.height });
    if (!hit) {
      setContextMenu(null);
      return;
    }
    const rect = canvas.getBoundingClientRect();
    setContextMenu({
      target: hit,
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    });
    if (hit.type === "polygon") onSelect(hit.index, -1);
  }

  function deleteContextTarget() {
    if (!contextMenu) return;
    const { target } = contextMenu;
    const confirmed = window.confirm("Delete this polygon label?");
    if (!confirmed) return;
    if (target.type === "polygon") {
      onPolygonsChange(polygons.filter((_, index) => index !== target.index));
      onSelect(-1, -1);
    }
    setContextMenu(null);
  }

  function onCanvasLeave(event) {
    if (pressRef.current?.active) {
      finishFreehand(point(event));
      return;
    }
    clearPendingPress();
    dragRef.current = null;
  }

  return (
    <div className="annotator-wrap">
      <canvas
        ref={canvasRef}
        width="640"
        height="640"
        className="annotator-canvas"
        title={samAssistEnabled ? "AI assist is enabled. Click blank crown area to request a SAM polygon." : "Click to add polygon vertices. Long-press empty space to draw a closed freehand curve. Right-click labels to delete."}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onCanvasLeave}
        onDoubleClick={finishDraft}
        onContextMenu={onContextMenu}
      />
      {samAssistBusy && (
        <div className="assist-overlay">
          <Loader2 className="spin" size={18} />
          SAM
        </div>
      )}
      {contextMenu && (
        <div className="context-menu" style={{ left: contextMenu.x, top: contextMenu.y }}>
          <button type="button" className="context-danger" onClick={deleteContextTarget}>
            Delete
          </button>
        </div>
      )}
    </div>
  );
}

const QUALITY_LABELS = {
  normal: "亮度正常",
  underexposed: "偏暗",
  overexposed: "偏亮",
  low_contrast: "低对比度",
  mixed: "明暗混合",
};

const VISIBILITY_REASONS = [
  ["none", "无"],
  ["occluded", "树木遮挡"],
  ["exposure_unrecoverable", "曝光无法恢复"],
  ["blur", "运动模糊/失焦"],
  ["target_missing", "目标树缺失"],
  ["neighboring_tree", "邻树干扰"],
  ["other", "其他"],
];

function ImageEnhancementPanel({
  item,
  analyzing,
  onPreprocessChange,
  onAutoAdjust,
  onReset,
  onCompareStart,
  onCompareEnd,
  onVisibilityChange,
}) {
  const preprocess = normalizePreprocess(item.preprocess);
  const quality = preprocess.quality_before || {};
  const changed = Math.abs(Number(preprocess.exposure_ev)) > 0.01 || Math.abs(Number(preprocess.contrast) - 1) > 0.01;
  const setExposure = (value) => {
    const exposure_ev = Number(value);
    onPreprocessChange({
      exposure_ev,
      reason: preprocessReason(exposure_ev, preprocess.contrast),
      auto_suggested: false,
      human_accepted: true,
    });
  };
  const setContrast = (value) => {
    const contrast = Number(value);
    onPreprocessChange({
      contrast,
      reason: preprocessReason(preprocess.exposure_ev, contrast),
      auto_suggested: false,
      human_accepted: true,
    });
  };

  return (
    <section className="enhancement-panel" aria-label="影像增强与可标注性">
      <div className="enhancement-head">
        <span><SlidersHorizontal size={15} /> 影像增强</span>
        <span className={changed ? "recipe-badge adjusted" : "recipe-badge"}>
          {changed ? "已调整并记录" : "原始图像"}
        </span>
      </div>
      <div className="adjustment-grid">
        <label className="adjustment-control">
          <span>曝光 <output>{Number(preprocess.exposure_ev) >= 0 ? "+" : ""}{Number(preprocess.exposure_ev).toFixed(1)} EV</output></span>
          <input
            type="range"
            min="-2"
            max="2"
            step="0.1"
            value={preprocess.exposure_ev}
            onChange={(event) => setExposure(event.target.value)}
          />
        </label>
        <label className="adjustment-control">
          <span>对比度 <output>{Number(preprocess.contrast).toFixed(2)}×</output></span>
          <input
            type="range"
            min="0.5"
            max="1.5"
            step="0.05"
            value={preprocess.contrast}
            onChange={(event) => setContrast(event.target.value)}
          />
        </label>
      </div>
      <div className="enhancement-actions">
        <button type="button" onClick={onAutoAdjust} disabled={analyzing} title="分析目标树中央区域并应用建议值">
          {analyzing ? <Loader2 className="spin" size={15} /> : <Sparkles size={15} />} 自动建议
        </button>
        <button
          type="button"
          onPointerDown={onCompareStart}
          onPointerUp={onCompareEnd}
          onPointerCancel={onCompareEnd}
          onPointerLeave={onCompareEnd}
          title="按住时临时显示未经增强的原始图像"
        >
          <Eye size={15} /> 按住看原图
        </button>
        <button type="button" onClick={onReset} disabled={!changed} title="恢复 0 EV 与 1.00× 对比度">
          <RotateCcw size={15} /> 重置
        </button>
      </div>
      {Object.keys(quality).length > 0 && (
        <div className="quality-summary" title="统计区域避开大部分天空和车辆，以目标树所在中央区域为主">
          <span className={`quality-status ${quality.quality_status || "normal"}`}>{QUALITY_LABELS[quality.quality_status] || quality.quality_status}</span>
          <span>中位亮度 {formatNumber(quality.p50_luminance, 0)}/255</span>
          <span>暗像素 {percent(quality.dark_pixel_ratio)}</span>
          <span>亮像素 {percent(quality.bright_pixel_ratio)}</span>
        </div>
      )}
      <div className="visibility-controls">
        <label>
          <span>可标注性</span>
          <select
            value={item.visibility?.status || "clear"}
            onChange={(event) => onVisibilityChange({ status: event.target.value })}
          >
            <option value="clear">清晰可标注</option>
            <option value="partial">部分可见/勉强可标注</option>
            <option value="unannotatable">无法标注并剔除</option>
          </select>
        </label>
        <label>
          <span>原因</span>
          <select
            value={item.visibility?.reason || "none"}
            onChange={(event) => onVisibilityChange({ reason: event.target.value })}
          >
            {VISIBILITY_REASONS.map(([value, label]) => <option value={value} key={value}>{label}</option>)}
          </select>
        </label>
      </div>
      {(item.visibility?.status !== "clear" || item.visibility?.reason !== "none") && (
        <input
          className="visibility-note"
          value={item.visibility?.note || ""}
          maxLength={500}
          placeholder="可选：记录遮挡位置、可见树冠范围或其他情况"
          onChange={(event) => onVisibilityChange({ note: event.target.value })}
        />
      )}
      {item.visibility?.status === "unannotatable" && (
        <p className="unannotatable-warning">该视角已自动设为 Drop；质量原因和增强参数仍会保存，供后续预处理策略学习。</p>
      )}
    </section>
  );
}

function ApiKeyPanel({ onReady }) {
  const [apiKey, setApiKey] = useState("");
  const [status, setStatus] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch(`${API}/api/config/status`)
      .then((res) => res.json())
      .then((data) => setStatus(data))
      .catch((err) => setError(err.message));
  }, []);

  async function submitKey() {
    setBusy(true);
    setError("");
    try {
      const endpoint = apiKey.trim() ? "/api/config/api-key" : "/api/config/validate";
      const options = apiKey.trim()
        ? {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ api_key: apiKey.trim() }),
          }
        : { method: "POST" };
      const res = await fetch(`${API}${endpoint}`, options);
      if (!res.ok) throw new Error((await res.json()).detail || "API key validation failed");
      onReady();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="setup-shell">
      <section className="setup-panel api-panel">
        <div className="title-row">
          <div>
            <h1>Google Maps API</h1>
            <p>Validate the Street View key before choosing a species and collecting images.</p>
          </div>
          <KeyRound size={26} />
        </div>
        {status?.google_maps_api_key_available && (
          <div className="success-line" title="A key was passed to the backend for this session">
            <ShieldCheck size={18} />
            Active session key detected ({status.google_maps_api_key_source}, length {status.google_maps_api_key_length}).
          </div>
        )}
        <label className="field">
          <span>API key</span>
          <div className="search-box">
            <KeyRound size={18} />
            <input
              type="password"
              value={apiKey}
              onChange={(event) => setApiKey(event.target.value)}
              placeholder={status?.google_maps_api_key_available ? "Leave blank to validate the current session key" : "Enter Google Maps API Key"}
              title="Google Maps API Key used only by the current backend session"
            />
          </div>
        </label>
        {error && <div className="error-line">{error}</div>}
        <button className="primary" onClick={submitKey} disabled={busy || (!apiKey.trim() && !status?.google_maps_api_key_available)} title="Validate API key and continue to species setup">
          {busy ? <Loader2 className="spin" size={18} /> : <ShieldCheck size={18} />}
          Validate & Continue
        </button>
      </section>
    </main>
  );
}

function SetupPanel({ onStart }) {
  const [species, setSpecies] = useState([]);
  const [query, setQuery] = useState("");
  const [pickerOpen, setPickerOpen] = useState(false);
  const [traits, setTraits] = useState("");
  const [targetCount, setTargetCount] = useState(50);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch(`${API}/api/species/list`)
      .then((res) => res.json())
      .then((data) => {
        setSpecies(data.species || []);
        setQuery((data.species || [])[0] || "");
      })
      .catch((err) => setError(err.message));
  }, []);

  const selectedSpecies = useMemo(() => species.find((item) => item.toLowerCase() === query.toLowerCase()) || "", [species, query]);

  useEffect(() => {
    if (!selectedSpecies) {
      setTraits("");
      return;
    }
    const controller = new AbortController();
    fetch(`${API}/api/species/traits?species=${encodeURIComponent(selectedSpecies)}`, { signal: controller.signal })
      .then((res) => (res.ok ? res.json() : { traits: "" }))
      .then((data) => setTraits(data.traits || ""))
      .catch((err) => {
        if (err.name !== "AbortError") setTraits("");
    });
    return () => controller.abort();
  }, [selectedSpecies]);

  const filtered = useMemo(() => {
    const needle = pickerOpen && selectedSpecies === query ? "" : query.toLowerCase();
    return species.filter((item) => item.toLowerCase().includes(needle));
  }, [pickerOpen, query, selectedSpecies, species]);

  function chooseSpecies(item) {
    setQuery(item);
    setPickerOpen(false);
  }

  async function start() {
    setBusy(true);
    setError("");
    try {
      if (!selectedSpecies) throw new Error("Choose an exact species from the list before starting.");
      const res = await fetch(`${API}/api/task/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ species: selectedSpecies, target_count: Number(targetCount) }),
      });
      if (!res.ok) throw new Error((await res.json()).detail || "Failed to start task");
      onStart(selectedSpecies);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="setup-shell">
      <section className="setup-panel">
        <div className="title-row">
          <div>
            <h1>何文田行道树人工标注</h1>
            <p>从实采全景影像与附近树点中选取候选视图，人工确认目标树并绘制轮廓。</p>
          </div>
        </div>
        <label className="field">
          <span>Species</span>
          <div className="search-box">
            <Search size={18} />
            <input
              value={query}
              onChange={(event) => {
                setQuery(event.target.value);
                setPickerOpen(true);
              }}
              placeholder="筛选树种"
              title="输入中英文或学名筛选附近树种"
            />
          </div>
        </label>
        <div className="species-picker">
          <button
            type="button"
            className="picker-toggle"
            onClick={() => setPickerOpen((value) => !value)}
            title="Open species selection table"
            aria-expanded={pickerOpen}
          >
            {pickerOpen ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
            <span>{selectedSpecies || "选择树种"}</span>
            <span className="picker-count">{filtered.length} / {species.length}</span>
          </button>
          {pickerOpen && (
            <div className="species-table-wrap">
              <table className="species-table">
                <thead>
                  <tr>
                    <th>Species</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {filtered.map((item) => (
                    <tr key={item} className={item === selectedSpecies ? "selected" : ""}>
                      <td>
                        <button type="button" className="species-row-button" onClick={() => chooseSpecies(item)} title={`Use ${item}`}>
                          {item}
                        </button>
                      </td>
                      <td>{item === selectedSpecies ? "Selected" : ""}</td>
                    </tr>
                  ))}
                  {filtered.length === 0 && (
                    <tr>
                      <td colSpan="2" className="empty-cell">No matching species</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          )}
        </div>
        <label className="field compact">
          <span>本轮目标树数量</span>
          <input type="number" min="1" max="10000" value={targetCount} onChange={(event) => setTargetCount(event.target.value)} title="Number of tree samples to cache for this session" />
        </label>
        {error && <div className="error-line">{error}</div>}
        <button className="primary" onClick={start} disabled={busy || !selectedSpecies} title="开始从何文田实采数据中加载候选目标树">
          {busy ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
          Start
        </button>
        <section className="traits-panel" title="Reference text loaded from traits.txt">
          <div className="traits-title">树种与数据状态</div>
          <pre>{traits || "选择树种后显示何文田附近树点数量与已标注进度。"}</pre>
        </section>
      </section>
    </main>
  );
}

function ResultsPanel({ onBack }) {
  const [status, setStatus] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    setBusy(true);
    setError("");
    try {
      const res = await fetch(`${API}/api/task/status`);
      if (!res.ok) throw new Error("Failed to load session status");
      setStatus(await res.json());
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const apiCounts = status?.api_counts ? Object.entries(status.api_counts).sort(([a], [b]) => a.localeCompare(b)) : [];
  const apiErrors = status?.api_error_counts ? Object.entries(status.api_error_counts).sort(([a], [b]) => a.localeCompare(b)) : [];

  return (
    <main className="workspace">
      <header className="toolbar">
        <div>
          <h1>Session Report</h1>
          <p>{status?.species || "No active species"}</p>
        </div>
        <div className="actions">
          <button onClick={onBack} title="Return to the labeling workspace">
            <ArrowLeft size={18} /> Labeling
          </button>
          <button onClick={refresh} disabled={busy} title="Refresh current backend counters">
            {busy ? <Loader2 className="spin" size={18} /> : <BarChart3 size={18} />} Refresh
          </button>
        </div>
      </header>
      {error && <div className="error-line">{error}</div>}
      <section className="report-grid">
        <div className="report-card">
          <span>Task status</span>
          <strong>{status?.status || "-"}</strong>
        </div>
        <div className="report-card">
          <span>Target</span>
          <strong>{status?.target_count ?? "-"}</strong>
        </div>
        <div className="report-card">
          <span>Queued</span>
          <strong>{status?.queued_count ?? "-"}</strong>
        </div>
        <div className="report-card">
          <span>Saved</span>
          <strong>{status?.reviewed_count ?? "-"}</strong>
        </div>
        <div className="report-card">
          <span>Rejected</span>
          <strong>{status?.rejected_count ?? "-"}</strong>
        </div>
        <div className="report-card">
          <span>Failed</span>
          <strong>{status?.failed_count ?? "-"}</strong>
        </div>
      </section>
      {status?.message && <div className="status-line neutral">{status.message}</div>}
      <section className="report-columns">
        <div className="report-panel">
          <h2>API calls</h2>
          {apiCounts.length ? (
            apiCounts.map(([name, value]) => (
              <div className="report-row" key={name}>
                <span>{name}</span>
                <strong>{value}</strong>
              </div>
            ))
          ) : (
            <p>No API calls recorded.</p>
          )}
        </div>
        <div className="report-panel">
          <h2>API errors / retries</h2>
          {apiErrors.length ? (
            apiErrors.map(([name, value]) => (
              <div className="report-row" key={name}>
                <span>{name}</span>
                <strong>{value}</strong>
              </div>
            ))
          ) : (
            <p>No API errors recorded.</p>
          )}
        </div>
      </section>
    </main>
  );
}

function Workspace({ species, onShowResults, onChangeSpecies }) {
  const [sample, setSample] = useState(null);
  const [annotations, setAnnotations] = useState([]);
  const [history, setHistory] = useState([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [selected, setSelected] = useState({ image: 0, polygon: 0, vertex: -1 });
  const [expandedIndex, setExpandedIndex] = useState(-1);
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState(false);
  const [samModels, setSamModels] = useState([]);
  const [samEnabled, setSamEnabled] = useState(false);
  const [samModelKey, setSamModelKey] = useState("");
  const [samMessage, setSamMessage] = useState("");
  const [samBusyImage, setSamBusyImage] = useState(-1);
  const [analyzingImage, setAnalyzingImage] = useState(-1);
  const [compareOriginal, setCompareOriginal] = useState(-1);

  const applyHistoryEntry = useCallback((entry, nextIndex) => {
    setSample(entry.sample);
    setAnnotations(entry.annotations);
    setHistoryIndex(nextIndex);
    setSelected({ image: 0, polygon: entry.annotations?.[0]?.polygons?.length ? 0 : -1, vertex: -1 });
    setExpandedIndex(-1);
    setStatus("");
  }, []);

  useEffect(() => {
    if (!sample || historyIndex < 0) return;
    setHistory((current) => {
      const entry = current[historyIndex];
      if (!entry || entry.sample.tree_id !== sample.tree_id) return current;
      const next = [...current];
      next[historyIndex] = { ...entry, sample, annotations };
      return next;
    });
  }, [annotations, historyIndex, sample]);

  useEffect(() => {
    if (!sample || !annotations.length || busy) return undefined;
    const controller = new AbortController();
    const timer = window.setTimeout(async () => {
      try {
        const res = await fetch(`${API}/api/task/checkpoint`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ tree_id: sample.tree_id, species: sample.species, annotations }),
          signal: controller.signal,
        });
        if (!res.ok) {
          const data = await res.json();
          throw new Error(data.detail || "自动保存失败");
        }
      } catch (err) {
        if (err.name !== "AbortError") setStatus(`自动保存失败：${err.message}`);
      }
    }, 900);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [annotations, busy, sample]);

  const loadNext = useCallback(async () => {
    setBusy(true);
    try {
      const res = await fetch(`${API}/api/task/next`);
      const data = await res.json();
      if (data.empty) {
        const taskStatus = data.task?.status;
        setStatus(data.task?.message || "当前任务没有更多样本。");
        if (taskStatus !== "completed") window.setTimeout(loadNext, 2500);
        return;
      }
      const draftByImage = new Map(
        (data.draft_annotations || []).map((annotation) => [annotation.image, annotation]),
      );
      const seeded = data.images.map((image) => {
        const draft = draftByImage.get(image);
        return {
          image,
          polygons: draft?.polygons || [],
          boxes: draft?.boxes || [],
          draftPoints: [],
          keep: draft?.keep ?? true,
          preprocess: normalizePreprocess(draft?.preprocess),
          visibility: { ...defaultVisibility(), ...(draft?.visibility || {}) },
        };
      });
      const entry = { sample: data, annotations: seeded };
      setHistory((current) => {
        const base = historyIndex >= 0 ? current.slice(0, historyIndex + 1) : current;
        return [...base, entry];
      });
      applyHistoryEntry(entry, historyIndex + 1);
      if (data.draft_annotations?.length) setStatus("已恢复该树上次保存的标注草稿。");
    } finally {
      setBusy(false);
    }
  }, [applyHistoryEntry, historyIndex]);

  useEffect(() => {
    loadNext();
  }, []);

  useEffect(() => {
    fetch(`${API}/api/sam/models`)
      .then((res) => res.json())
      .then((data) => {
        const models = data.models || [];
        setSamModels(models);
        const preferred = models.find((model) => model.type === "sam2") || models[0];
        if (preferred) setSamModelKey(preferred.key);
      })
      .catch((err) => setSamMessage(`SAM models unavailable: ${err.message}`));
  }, []);

  function goBack() {
    if (historyIndex <= 0) return;
    applyHistoryEntry(history[historyIndex - 1], historyIndex - 1);
  }

  function goForward() {
    if (historyIndex >= 0 && historyIndex < history.length - 1) {
      applyHistoryEntry(history[historyIndex + 1], historyIndex + 1);
      return;
    }
    loadNext();
  }

  async function advanceAfterAction() {
    if (historyIndex >= 0 && historyIndex < history.length - 1) {
      applyHistoryEntry(history[historyIndex + 1], historyIndex + 1);
      return;
    }
    await loadNext();
  }

  const submit = useCallback(async () => {
    if (!sample) return;
    setBusy(true);
    try {
      await fetch(`${API}/api/task/submit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tree_id: sample.tree_id, species: sample.species, annotations }),
      });
      await advanceAfterAction();
    } finally {
      setBusy(false);
    }
  }, [annotations, history, historyIndex, loadNext, sample]);

  const saveProgressAndChangeSpecies = useCallback(async () => {
    setBusy(true);
    try {
      if (sample) {
        const res = await fetch(`${API}/api/task/checkpoint`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ tree_id: sample.tree_id, species: sample.species, annotations }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "保存进度失败");
      }
      onChangeSpecies();
    } catch (err) {
      setStatus(`保存进度失败：${err.message}`);
    } finally {
      setBusy(false);
    }
  }, [annotations, onChangeSpecies, sample]);

  const reject = useCallback(async () => {
    if (!sample) return;
    setBusy(true);
    try {
      await fetch(`${API}/api/task/reject`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tree_id: sample.tree_id, species: sample.species, annotations }),
      });
      await advanceAfterAction();
    } finally {
      setBusy(false);
    }
  }, [annotations, history, historyIndex, loadNext, sample]);

  useEffect(() => {
    function onKeyDown(event) {
      if (event.target instanceof Element && event.target.closest("input, select, textarea, button")) return;
      if (event.code === "Space") {
        event.preventDefault();
        submit();
      }
      if (event.key.toLowerCase() === "r") reject();
      if (event.key === "Delete") {
        setAnnotations((current) =>
          current.map((item, imageIndex) =>
            imageIndex === selected.image
              ? { ...item, polygons: item.polygons.filter((_, polygonIndex) => polygonIndex !== selected.polygon) }
              : item,
          ),
        );
        setSelected((current) => ({ ...current, polygon: -1, vertex: -1 }));
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [reject, selected, submit]);

  function updatePolygons(index, polygons) {
    setAnnotations((current) => current.map((item, i) => (i === index ? { ...item, polygons } : item)));
  }

  function updateDraft(index, draftPoints) {
    setAnnotations((current) => current.map((item, i) => (i === index ? { ...item, draftPoints } : item)));
  }

  function toggleKeep(index) {
    setAnnotations((current) => current.map((item, i) => (
      i === index
        ? {
            ...item,
            keep: !item.keep,
            visibility: !item.keep && item.visibility?.status === "unannotatable"
              ? { ...item.visibility, status: "partial" }
              : item.visibility,
          }
        : item
    )));
  }

  function updatePreprocess(index, patch) {
    setAnnotations((current) => current.map((item, i) => (
      i === index
        ? { ...item, preprocess: { ...normalizePreprocess(item.preprocess), ...patch } }
        : item
    )));
  }

  function resetPreprocess(index) {
    setAnnotations((current) => current.map((item, i) => {
      if (i !== index) return item;
      const previous = normalizePreprocess(item.preprocess);
      return {
        ...item,
        preprocess: {
          ...defaultPreprocess(),
          quality_before: previous.quality_before,
          human_accepted: true,
        },
      };
    }));
  }

  async function autoAdjust(index) {
    const item = annotations[index];
    if (!item || analyzingImage >= 0) return;
    setAnalyzingImage(index);
    try {
      const res = await fetch(`${API}/api/image/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: item.image }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "图像分析失败");
      updatePreprocess(index, {
        ...data.suggested_preprocess,
        quality_before: data.metrics || {},
        quality_after: {},
        auto_suggested: true,
        human_accepted: true,
      });
      const label = QUALITY_LABELS[data.metrics?.quality_status] || data.metrics?.quality_status || "已分析";
      setStatus(`视角 ${index + 1}：${label}，已应用自动建议；可继续手动微调。`);
    } catch (err) {
      setStatus(`自动分析失败：${err.message}`);
    } finally {
      setAnalyzingImage(-1);
    }
  }

  function updateVisibility(index, patch) {
    setAnnotations((current) => current.map((item, i) => {
      if (i !== index) return item;
      const visibility = { ...defaultVisibility(), ...(item.visibility || {}), ...patch };
      if (visibility.status === "clear") visibility.reason = "none";
      if (visibility.status !== "clear" && visibility.reason === "none") visibility.reason = "occluded";
      return {
        ...item,
        visibility,
        keep: visibility.status === "unannotatable" ? false : item.keep,
      };
    }));
  }

  function clearPolygons(index) {
    setAnnotations((current) => current.map((item, i) => (i === index ? { ...item, polygons: [], draftPoints: [] } : item)));
    setSelected({ image: index, polygon: -1, vertex: -1 });
  }

  function clearAllPolygons() {
    setAnnotations((current) => current.map((item) => ({ ...item, polygons: [], draftPoints: [] })));
    setSelected({ image: 0, polygon: -1, vertex: -1 });
  }

  async function requestSamAssist(index, point) {
    if (!samEnabled || !samModelKey || samBusyImage >= 0) return;
    const item = annotations[index];
    if (!item) return;
    setSamBusyImage(index);
    setSamMessage("SAM is segmenting the clicked crown area...");
    try {
      const res = await fetch(`${API}/api/task/sam-segment`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: item.image, model_key: samModelKey, point }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "SAM request failed");
      if (data.error) throw new Error(data.error);
      const candidate = data.candidates?.[0];
      if (!candidate) throw new Error("SAM did not return a usable polygon for this point.");

      const nextPolygonIndex = item.polygons.length;
      setAnnotations((current) =>
        current.map((entry, imageIndex) =>
          imageIndex === index
            ? {
                ...entry,
                polygons: appendCandidatePolygon(entry.polygons, candidate),
                draftPoints: [],
              }
            : entry,
        ),
      );
      setSelected({ image: index, polygon: nextPolygonIndex, vertex: -1 });
      setSamMessage(`Added SAM polygon from ${candidate.model_key || samModelKey}.`);
    } catch (err) {
      setSamMessage(err.message);
    } finally {
      setSamBusyImage(-1);
    }
  }

  const keptCount = annotations.filter((item) => item.keep).length;

  return (
    <main className="workspace">
      <header className="toolbar">
        <div>
          <h1>{sample ? sample.tree_id : species}</h1>
          <p>{sample?.species || species}</p>
        </div>
        <div className="actions">
          <div className="sam-controls" title="Optional SAM2/SAM3-assisted polygon generation">
            <button
              type="button"
              className={samEnabled ? "assist-toggle active" : "assist-toggle"}
              onClick={() => setSamEnabled((value) => !value)}
              disabled={!samModels.length || busy}
              title={samModels.length ? "Toggle SAM-assisted click segmentation" : "No SAM weights were found"}
            >
              <Sparkles size={18} />
              AI Assist
            </button>
            <select
              value={samModelKey}
              onChange={(event) => setSamModelKey(event.target.value)}
              disabled={!samModels.length || samBusyImage >= 0}
              title="Choose SAM model"
            >
              {samModels.length ? (
                samModels.map((model) => (
                  <option key={model.key} value={model.key}>
                    {model.display_name} {model.size_label ? `(${model.size_label})` : ""}
                  </option>
                ))
              ) : (
                <option value="">No SAM models</option>
              )}
            </select>
          </div>
          <button onClick={goBack} disabled={busy || historyIndex <= 0} title="Return to the previous tree for editing">
            <ArrowLeft size={18} /> Back
          </button>
          <button onClick={goForward} disabled={busy || !sample} title="Go to the next reviewed tree, or fetch a new tree when no forward history exists">
            <ArrowRight size={18} /> Next
          </button>
          <button onClick={onShowResults} disabled={busy} title="Open current session report and API call counters">
            <BarChart3 size={18} /> Report
          </button>
          <button onClick={saveProgressAndChangeSpecies} disabled={busy} title="保存当前树的标注草稿并返回树种选择界面">
            <Save size={18} /> 保存进度并换树种
          </button>
          <button onClick={clearAllPolygons} disabled={busy || !sample} title="Clear all predicted and manual polygons">
            <Eraser size={18} /> Clear polygons
          </button>
          <button onClick={reject} disabled={busy || !sample} title="Reject this tree now; files are kept for one-step back and removed when you start the tree after next (R)">
            <X size={18} /> Reject
          </button>
          <button className="primary" onClick={submit} disabled={busy || !sample || keptCount === 0} title="Save kept views and load next tree (Space)">
            {busy ? <Loader2 className="spin" size={18} /> : <Check size={18} />} 保存 {keptCount}/{annotations.length || 3}
          </button>
        </div>
      </header>

      {sample?.warnings?.length > 0 && (
        <section className="warning-panel" title="Camera continuity warnings from heading, pitch, FOV, and distance">
          <AlertTriangle size={18} />
          <div>
            {sample.warnings.map((warning) => (
              <p key={warning}>{warning}</p>
            ))}
          </div>
        </section>
      )}

      {status && <div className="status-line">{status}</div>}
      {samMessage && <div className={`status-line ${samMessage.startsWith("Added") ? "neutral" : ""}`}>{samMessage}</div>}

      <ViewGeometryPanel shots={sample?.candidates || []} />

      <section className="canvas-grid">
        {annotations.map((item, index) => (
          <div className={`canvas-pane ${item.keep ? "" : "dropped"} ${expandedIndex === index ? "expanded" : ""}`} key={item.image}>
            {(() => {
              const shot = sample?.candidates?.[index];
              return (
                <>
            <div className="pane-head">
              <span className="pane-title" title={shotSummary(shot)}>
                <Target size={14} />
                {imageLabel(item.image)}
              </span>
              <div className="pane-tools">
                <button
                  type="button"
                  className={item.keep ? "keep-button kept" : "keep-button"}
                  title={item.keep ? "This view will be saved. Click to drop it." : "This view will be dropped. Click to keep it."}
                  aria-label={item.keep ? "Drop this view" : "Keep this view"}
                  onClick={() => toggleKeep(index)}
                >
                  {item.keep ? "Keep" : "Drop"}
                </button>
                <button
                  type="button"
                  className="zoom-button"
                  title={expandedIndex === index ? "Restore this view to the three-column layout" : "Enlarge this view for detailed annotation"}
                  aria-label={expandedIndex === index ? "Restore this view" : "Enlarge this view"}
                  onClick={() => setExpandedIndex((current) => (current === index ? -1 : index))}
                >
                  {expandedIndex === index ? <Minimize2 size={15} /> : <Maximize2 size={15} />}
                  {expandedIndex === index ? "Restore" : "Zoom"}
                </button>
                <span title="Current polygon count">{item.polygons.length} polygons</span>
                <button type="button" className="icon-button" title="Clear polygons in this view" aria-label="Clear polygons in this view" onClick={() => clearPolygons(index)}>
                  <Eraser size={15} />
                </button>
              </div>
            </div>
            <CanvasAnnotator
              image={item.image}
              polygons={item.polygons}
              draftPoints={item.draftPoints}
              preprocess={item.preprocess}
              showOriginal={compareOriginal === index}
              selectedPolygon={selected.image === index ? selected.polygon : -1}
              selectedVertex={selected.image === index ? selected.vertex : -1}
              samAssistEnabled={samEnabled && Boolean(samModelKey)}
              samAssistBusy={samBusyImage === index}
              onPolygonsChange={(polygons) => updatePolygons(index, polygons)}
              onDraftChange={(draftPoints) => updateDraft(index, draftPoints)}
              onSelect={(polygon, vertex = -1) => setSelected({ image: index, polygon, vertex })}
              onSamAssistPoint={(point) => requestSamAssist(index, point)}
            />
            {!item.keep && <div className="drop-overlay">Dropped from dataset</div>}
            <ImageEnhancementPanel
              item={item}
              analyzing={analyzingImage === index}
              onPreprocessChange={(patch) => updatePreprocess(index, patch)}
              onAutoAdjust={() => autoAdjust(index)}
              onReset={() => resetPreprocess(index)}
              onCompareStart={() => setCompareOriginal(index)}
              onCompareEnd={() => setCompareOriginal(-1)}
              onVisibilityChange={(patch) => updateVisibility(index, patch)}
            />
            <div className="view-meta" title="Street View sampling metadata for this image">
              <div><strong>ID</strong><span>{sample?.tree_id || "-"}</span></div>
              <div><strong>经度</strong><span>{formatNumber(shot?.lon)}</span></div>
              <div><strong>纬度</strong><span>{formatNumber(shot?.lat)}</span></div>
              <div><strong>距离</strong><span>{formatNumber(shot?.distance_m, 2)} m</span></div>
              <div><strong>采集时间</strong><span>{formatStreetViewDate(shot?.date)}</span></div>
            </div>
                </>
              );
            })()}
          </div>
        ))}
      </section>
    </main>
  );
}

function App() {
  const [activeSpecies, setActiveSpecies] = useState("");
  const [page, setPage] = useState("setup");
  if (page === "report") return <ResultsPanel onBack={() => setPage(activeSpecies ? "workspace" : "setup")} />;
  if (activeSpecies) return (
    <Workspace
      species={activeSpecies}
      onShowResults={() => setPage("report")}
      onChangeSpecies={() => {
        setActiveSpecies("");
        setPage("setup");
      }}
    />
  );
  return <SetupPanel onStart={(species) => {
    setActiveSpecies(species);
    setPage("workspace");
  }} />;
}

createRoot(document.getElementById("root")).render(<App />);
