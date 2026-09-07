import React, { useCallback, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  AlertTriangle,
  ArrowLeft,
  ArrowRight,
  Check,
  Eye,
  ExternalLink,
  ImageOff,
  Loader2,
  LocateFixed,
  Plus,
  RotateCcw,
  Save,
  Sparkles,
  Trash2,
  TreePine,
} from "lucide-react";
import { polygonToPixels, vertexHitTest } from "./annotationGeometry.js";
import "./frameStyles.css";

const API = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8021";

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

function labelColor(species) {
  let hash = 0;
  for (const character of species || "tree") hash = ((hash << 5) - hash + character.charCodeAt(0)) | 0;
  return `hsl(${Math.abs(hash) % 360} 72% 42%)`;
}

function normalizeSpeciesForUi(value) {
  const name = String(value || "").trim();
  const folded = name.toLocaleLowerCase();
  return folded.startsWith("ficus microcarpa")
    || folded.startsWith("ficus benjamina")
    || ["細葉榕", "细叶榕", "垂葉榕", "垂叶榕", "榕樹"].some((token) => name.includes(token))
    ? "榕树"
    : name;
}

function newLabelId() {
  return globalThis.crypto?.randomUUID?.() || `tree_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

function orientation(a, b, c) {
  const value = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
  if (value > 1e-12) return 1;
  if (value < -1e-12) return -1;
  return 0;
}

function properSegmentsIntersect(a, b, c, d) {
  return orientation(a, b, c) * orientation(a, b, d) < 0
    && orientation(c, d, a) * orientation(c, d, b) < 0;
}

function untanglePolygon(points) {
  const repaired = points.map((point) => [...point]);
  const operationLimit = repaired.length * repaired.length;
  for (let operation = 0; operation <= operationLimit; operation += 1) {
    let crossing = null;
    for (let first = 0; first < repaired.length && !crossing; first += 1) {
      for (let second = first + 1; second < repaired.length; second += 1) {
        if (second === (first + 1) % repaired.length || (second + 1) % repaired.length === first) continue;
        if (properSegmentsIntersect(
          repaired[first],
          repaired[(first + 1) % repaired.length],
          repaired[second],
          repaired[(second + 1) % repaired.length],
        )) {
          crossing = [first, second];
          break;
        }
      }
    }
    if (!crossing) return repaired;
    const [first, second] = crossing;
    repaired.splice(first + 1, second - first, ...repaired.slice(first + 1, second + 1).reverse());
  }
  return repaired;
}

function percent(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${(number * 100).toFixed(1)}%` : "-";
}

function labelConfidence(label) {
  if (label?.confidence !== null && label?.confidence !== undefined) {
    const explicit = Number(label.confidence);
    if (Number.isFinite(explicit) && explicit >= 0 && explicit <= 1) return explicit;
  }

  const assistant = label?.assistant_prelabel || {};
  const domainConfidence = Number(assistant.domain_confidence);
  if (
    Number.isFinite(domainConfidence)
    && domainConfidence >= 0
    && domainConfidence <= 1
    && [assistant.species, assistant.domain_species].includes(label?.species)
  ) return domainConfidence;

  const note = String(label?.note || "");
  for (const pattern of [/域内YOLO候选[:：]\s*([^；|]+?\(([01](?:\.\d+)?)\))(?=；|\s*\||$)/, /域内=([^；|]+?\(([01](?:\.\d+)?)\))(?=；|\s*\||$)/]) {
    const match = note.match(pattern);
    if (!match) continue;
    const candidate = match[1].replace(/\s*\([01](?:\.\d+)?\)\s*$/, "").trim();
    const confidence = Number(match[2]);
    if (candidate === label?.species && Number.isFinite(confidence)) return confidence;
  }
  return null;
}

function pixelBounds(points) {
  if (!points.length) return null;
  const xs = points.map(([x]) => x);
  const ys = points.map(([, y]) => y);
  const left = Math.min(...xs);
  const right = Math.max(...xs);
  const top = Math.min(...ys);
  const bottom = Math.max(...ys);
  return { left, right, top, bottom, width: right - left, height: bottom - top };
}

function boxHitTest(labels, pixelX, pixelY, width, height) {
  return labels
    .map((label, index) => ({
      index,
      bounds: pixelBounds(polygonToPixels(label.points, width, height)),
    }))
    .filter(({ bounds }) => bounds
      && pixelX >= bounds.left
      && pixelX <= bounds.right
      && pixelY >= bounds.top
      && pixelY <= bounds.bottom)
    .sort((a, b) => (a.bounds.width * a.bounds.height) - (b.bounds.width * b.bounds.height))[0]?.index ?? -1;
}

function drawBadge(context, text, anchorX, anchorY, align, background, canvasWidth) {
  context.font = "700 18px system-ui";
  context.textBaseline = "middle";
  const paddingX = 8;
  const badgeHeight = 28;
  const badgeWidth = context.measureText(text).width + paddingX * 2;
  const requestedLeft = align === "right" ? anchorX - badgeWidth : anchorX;
  const left = Math.max(0, Math.min(canvasWidth - badgeWidth, requestedLeft));
  const top = Math.max(0, anchorY);
  context.fillStyle = background;
  context.fillRect(left, top, badgeWidth, badgeHeight);
  context.fillStyle = "white";
  context.fillText(text, left + paddingX, top + badgeHeight / 2 + 1);
}

function badgeWidth(context, text) {
  context.font = "700 18px system-ui";
  return context.measureText(text).width + 16;
}

function hasMeaningfulDraft(labels, draftPoints, preprocess, note) {
  return labels.length > 0
    || draftPoints.length > 0
    || Boolean(note.trim())
    || Number(preprocess?.exposure_ev || 0) !== 0
    || Number(preprocess?.contrast || 1) !== 1
    || Boolean(preprocess?.auto_suggested)
    || Boolean(preprocess?.human_accepted);
}

function googleStreetViewUrl(frame) {
  const parameters = new URLSearchParams({
    api: "1",
    map_action: "pano",
    viewpoint: `${frame.latitude},${frame.longitude}`,
    heading: String(frame.heading_deg),
    pitch: "0",
    fov: "90",
  });
  return `https://www.google.com/maps/@?${parameters.toString()}`;
}

function PanoramaCanvas({
  image,
  labels,
  draftPoints,
  activeSpecies,
  selectedId,
  preprocess,
  showOriginal,
  zoom,
  drawMode,
  onLabelsChange,
  onDraftChange,
  onSelect,
}) {
  const canvasRef = useRef(null);
  const imageRef = useRef(null);
  const dragRef = useRef(null);
  const freehandRef = useRef(null);

  const commitPolygon = useCallback((points) => {
    if (points.length < 3 || !activeSpecies) return;
    const label = {
      label_id: newLabelId(),
      species: activeSpecies,
      points: untanglePolygon(points),
      visibility: "clear",
      note: "",
    };
    onLabelsChange([...labels, label]);
    onDraftChange([]);
    onSelect(label.label_id);
  }, [activeSpecies, labels, onDraftChange, onLabelsChange, onSelect]);

  const finishDraft = useCallback(() => {
    commitPolygon(draftPoints);
  }, [commitPolygon, draftPoints]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const imageElement = imageRef.current;
    if (!canvas || !imageElement || !imageElement.complete) return;
    const context = canvas.getContext("2d");
    context.clearRect(0, 0, canvas.width, canvas.height);
    const exposureEv = showOriginal ? 0 : Number(preprocess?.exposure_ev || 0);
    const contrast = showOriginal ? 1 : Number(preprocess?.contrast || 1);
    context.save();
    context.filter = `brightness(${2 ** exposureEv}) contrast(${contrast})`;
    context.drawImage(imageElement, 0, 0, canvas.width, canvas.height);
    context.restore();
    context.filter = "none";

    labels.forEach((label) => {
      const points = polygonToPixels(label.points, canvas.width, canvas.height);
      const bounds = pixelBounds(points);
      if (!bounds) return;
      const color = labelColor(label.species);
      const selected = label.label_id === selectedId;
      context.save();
      context.strokeStyle = color;
      context.lineWidth = selected ? 5 : 3;
      context.strokeRect(bounds.left, bounds.top, bounds.width, bounds.height);

      if (selected) {
        context.beginPath();
        points.forEach(([x, y], index) => (index === 0 ? context.moveTo(x, y) : context.lineTo(x, y)));
        context.closePath();
        context.setLineDash([7, 5]);
        context.strokeStyle = "rgba(255,255,255,0.92)";
        context.lineWidth = 2;
        context.stroke();
        context.setLineDash([]);
        points.forEach(([x, y]) => {
          context.beginPath();
          context.arc(x, y, 5, 0, Math.PI * 2);
          context.fillStyle = color;
          context.fill();
          context.strokeStyle = "white";
          context.lineWidth = 1.5;
          context.stroke();
        });
      }

      const confidence = labelConfidence(label);
      const confidenceText = confidence === null ? "人工" : `${(confidence * 100).toFixed(1)}%`;
      const badgesFitOneRow = badgeWidth(context, label.species) + badgeWidth(context, confidenceText) + 6 <= bounds.width;
      drawBadge(
        context,
        label.species,
        bounds.left,
        badgesFitOneRow ? bounds.top : bounds.top + 30,
        "left",
        "rgba(15, 23, 20, 0.88)",
        canvas.width,
      );
      drawBadge(
        context,
        confidenceText,
        bounds.right,
        bounds.top,
        "right",
        confidence === null ? "rgba(71, 85, 105, 0.92)" : "rgba(180, 83, 9, 0.94)",
        canvas.width,
      );
      context.restore();
    });

    if (draftPoints.length) {
      const points = polygonToPixels(draftPoints, canvas.width, canvas.height);
      context.save();
      context.beginPath();
      points.forEach(([x, y], index) => (index === 0 ? context.moveTo(x, y) : context.lineTo(x, y)));
      context.strokeStyle = "#ffcf33";
      context.fillStyle = "#ffcf33";
      context.lineWidth = 3;
      context.setLineDash([8, 6]);
      context.stroke();
      context.setLineDash([]);
      points.forEach(([x, y], index) => {
        context.beginPath();
        context.arc(x, y, index === 0 && points.length >= 3 ? 9 : 6, 0, Math.PI * 2);
        context.fill();
      });
      context.restore();
    }
  }, [draftPoints, labels, preprocess, selectedId, showOriginal]);

  useEffect(() => {
    const nextImage = new Image();
    nextImage.crossOrigin = "anonymous";
    nextImage.src = `${API}${image}`;
    imageRef.current = nextImage;
    nextImage.onload = draw;
  }, [draw, image]);

  useEffect(draw, [draw]);

  function canvasPoint(event) {
    const canvas = canvasRef.current;
    const rectangle = canvas.getBoundingClientRect();
    return {
      pixelX: ((event.clientX - rectangle.left) / rectangle.width) * canvas.width,
      pixelY: ((event.clientY - rectangle.top) / rectangle.height) * canvas.height,
      normalized: [
        Math.max(0, Math.min(1, (event.clientX - rectangle.left) / rectangle.width)),
        Math.max(0, Math.min(1, (event.clientY - rectangle.top) / rectangle.height)),
      ],
    };
  }

  function onPointerDown(event) {
    if (event.button !== 0) return;
    event.preventDefault();
    const canvas = canvasRef.current;
    const point = canvasPoint(event);
    const editable = labels.map((label) => ({ points: label.points }));
    const vertex = vertexHitTest(editable, point.pixelX, point.pixelY, canvas.width, canvas.height, 11);
    if (vertex) {
      dragRef.current = vertex;
      event.currentTarget.setPointerCapture?.(event.pointerId);
      onSelect(labels[vertex.polygonIndex].label_id);
      return;
    }
    if (drawMode === "click" && draftPoints.length >= 3) {
      const [first] = polygonToPixels([draftPoints[0]], canvas.width, canvas.height);
      if (Math.hypot(first[0] - point.pixelX, first[1] - point.pixelY) <= 15) {
        finishDraft();
        return;
      }
    }
    const polygonIndex = boxHitTest(labels, point.pixelX, point.pixelY, canvas.width, canvas.height);
    if (polygonIndex >= 0) {
      onSelect(labels[polygonIndex].label_id);
      return;
    }
    if (drawMode === "drag") {
      freehandRef.current = {
        points: [point.normalized],
        lastPixelX: point.pixelX,
        lastPixelY: point.pixelY,
        totalDistance: 0,
      };
      event.currentTarget.setPointerCapture?.(event.pointerId);
      onDraftChange([point.normalized]);
      onSelect("");
      return;
    }
    onDraftChange([...draftPoints, point.normalized]);
    onSelect("");
  }

  function onPointerMove(event) {
    const point = canvasPoint(event);
    if (dragRef.current) {
      const { polygonIndex, vertexIndex } = dragRef.current;
      onLabelsChange(labels.map((label, labelIndex) => (
        labelIndex !== polygonIndex
          ? label
          : {
              ...label,
              points: label.points.map((vertex, index) => (index === vertexIndex ? point.normalized : vertex)),
            }
      )));
      return;
    }
    if (!freehandRef.current) return;
    const trace = freehandRef.current;
    const distance = Math.hypot(point.pixelX - trace.lastPixelX, point.pixelY - trace.lastPixelY);
    if (distance < 6) return;
    trace.points.push(point.normalized);
    trace.lastPixelX = point.pixelX;
    trace.lastPixelY = point.pixelY;
    trace.totalDistance += distance;
    onDraftChange([...trace.points]);
  }

  function finishPointer(event, cancelled = false) {
    if (freehandRef.current) {
      const trace = freehandRef.current;
      freehandRef.current = null;
      if (!cancelled && trace.points.length >= 3 && trace.totalDistance >= 24) {
        commitPolygon(trace.points);
      } else {
        onDraftChange([]);
      }
    }
    if (dragRef.current && !cancelled) {
      const { polygonIndex } = dragRef.current;
      onLabelsChange(labels.map((label, index) => (
        index === polygonIndex ? { ...label, points: untanglePolygon(label.points) } : label
      )));
    }
    dragRef.current = null;
    if (event.currentTarget.hasPointerCapture?.(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

  function onContextMenu(event) {
    event.preventDefault();
    const canvas = canvasRef.current;
    const point = canvasPoint(event);
    const polygonIndex = boxHitTest(labels, point.pixelX, point.pixelY, canvas.width, canvas.height);
    if (polygonIndex < 0) return;
    const removedId = labels[polygonIndex].label_id;
    onLabelsChange(labels.filter((_, index) => index !== polygonIndex));
    if (removedId === selectedId) onSelect("");
  }

  return (
    <div className="panorama-scroll">
      <canvas
        ref={canvasRef}
        width="1600"
        height="800"
        className="panorama-canvas"
        style={{ width: `${zoom * 100}%`, touchAction: "none" }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={(event) => finishPointer(event)}
        onPointerCancel={(event) => finishPointer(event, true)}
        onContextMenu={onContextMenu}
        aria-label="全景影像树木标注画布"
      />
    </div>
  );
}

function App() {
  const [item, setItem] = useState(null);
  const [task, setTask] = useState(null);
  const [species, setSpecies] = useState([]);
  const [activeSpecies, setActiveSpecies] = useState("");
  const [customSpecies, setCustomSpecies] = useState("");
  const [labels, setLabels] = useState([]);
  const [draftPoints, setDraftPoints] = useState([]);
  const [selectedId, setSelectedId] = useState("");
  const [preprocess, setPreprocess] = useState(defaultPreprocess());
  const [note, setNote] = useState("");
  const [spacing, setSpacing] = useState(15);
  const [zoom, setZoom] = useState(1);
  const [drawMode, setDrawMode] = useState("drag");
  const [jumpValue, setJumpValue] = useState(1);
  const [showOriginal, setShowOriginal] = useState(false);
  const [busy, setBusy] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [message, setMessage] = useState("");
  const hydratedRef = useRef(false);

  const hydrate = useCallback((data) => {
    const saved = data.draft || data.review || {};
    setItem(data);
    setLabels(saved.labels || []);
    setDraftPoints(saved.draft_points || []);
    setPreprocess({ ...defaultPreprocess(), ...(saved.preprocess || {}) });
    setNote(saved.note || "");
    setSelectedId("");
    setJumpValue(data.index + 1);
    hydratedRef.current = true;
  }, []);

  const loadItem = useCallback(async (index) => {
    setBusy(true);
    hydratedRef.current = false;
    try {
      const response = await fetch(`${API}/api/frame/item?index=${index}`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "影像加载失败");
      hydrate(data);
      setMessage(data.review ? `该帧已有正式记录：${data.review.frame_status}` : data.draft ? "已恢复草稿。" : "");
    } catch (error) {
      setMessage(error.message);
    } finally {
      setBusy(false);
    }
  }, [hydrate]);

  const start = useCallback(async (spacingValue = 15) => {
    setBusy(true);
    try {
      const [speciesResponse, startResponse] = await Promise.all([
        fetch(`${API}/api/frame/species`),
        fetch(`${API}/api/frame/start`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ spacing_m: Number(spacingValue) }),
        }),
      ]);
      const speciesData = await speciesResponse.json();
      const taskData = await startResponse.json();
      if (!startResponse.ok) throw new Error(taskData.detail || "任务启动失败");
      setSpecies(speciesData.species || []);
      setActiveSpecies((current) => current || speciesData.species?.[0] || "Unknown / 待定");
      setTask(taskData);
      setSpacing(Number(spacingValue));
      await loadItem(taskData.current_index || 0);
    } catch (error) {
      setMessage(error.message);
    } finally {
      setBusy(false);
    }
  }, [loadItem]);

  useEffect(() => { start(15); }, [start]);

  const payload = useCallback((frameStatus = "annotated") => ({
    frame_id: item?.frame?.frame_key || item?.frame?.frame_id,
    frame_status: frameStatus,
    labels: frameStatus === "annotated" ? labels : [],
    draft_points: draftPoints,
    preprocess,
    note,
  }), [draftPoints, item, labels, note, preprocess]);

  useEffect(() => {
    if (!item || !hydratedRef.current || busy) return undefined;
    if (!hasMeaningfulDraft(labels, draftPoints, preprocess, note)) return undefined;
    const controller = new AbortController();
    const timer = window.setTimeout(async () => {
      try {
        await fetch(`${API}/api/frame/draft`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload("annotated")),
          signal: controller.signal,
        });
      } catch (error) {
        if (error.name !== "AbortError") setMessage(`自动保存失败：${error.message}`);
      }
    }, 900);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [busy, draftPoints, item, labels, note, payload, preprocess]);

  async function saveDraftNow() {
    if (!item) return;
    if (!hasMeaningfulDraft(labels, draftPoints, preprocess, note)) return;
    await fetch(`${API}/api/frame/draft`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload("annotated")),
    });
  }

  async function goTo(index) {
    if (!item || busy) return;
    try {
      await saveDraftNow();
    } catch (error) {
      setMessage(`切换前保存失败：${error.message}`);
      return;
    }
    await loadItem(Math.max(0, Math.min(item.total - 1, index)));
  }

  async function submit(frameStatus) {
    if (!item) return;
    if (draftPoints.length) {
      setMessage("还有未闭合轮廓：点击黄色起点闭合，或撤销当前轮廓。");
      return;
    }
    if (frameStatus === "annotated" && labels.length === 0) {
      setMessage("请至少标注一棵树；若画面无目标树，请选择“无树”。");
      return;
    }
    if (frameStatus !== "annotated" && labels.length > 0) {
      setMessage("画面已有树木标注，请先清空标注，再设为无树或不可用。");
      return;
    }
    setBusy(true);
    try {
      const response = await fetch(`${API}/api/frame/submit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload(frameStatus)),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "保存失败");
      const statusResponse = await fetch(`${API}/api/frame/status`);
      setTask(await statusResponse.json());
      setMessage(`帧 ${data.frame_id} 已保存：${data.frame_status}。`);
      await loadItem(data.next_unreviewed_index ?? Math.min(item.total - 1, item.index + 1));
    } catch (error) {
      setMessage(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function autoAnalyze() {
    if (!item || analyzing) return;
    setAnalyzing(true);
    try {
      const response = await fetch(`${API}/api/frame/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ frame_id: item.frame.frame_key || item.frame.frame_id }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "影像分析失败");
      setPreprocess({
        ...data.suggested_preprocess,
        quality_before: data.metrics,
        human_accepted: true,
      });
      setMessage(`质量分析：${data.metrics.quality_status}，建议已应用。`);
    } catch (error) {
      setMessage(error.message);
    } finally {
      setAnalyzing(false);
    }
  }

  function updateSelected(patch) {
    setLabels((current) => current.map((label) => (label.label_id === selectedId ? { ...label, ...patch } : label)));
  }

  function deleteLabel(labelId) {
    setLabels((current) => current.filter((label) => label.label_id !== labelId));
    if (labelId === selectedId) setSelectedId("");
  }

  function addCustomSpecies() {
    const value = normalizeSpeciesForUi(customSpecies);
    if (!value) return;
    setSpecies((current) => (current.includes(value) ? current : [...current, value]));
    setActiveSpecies(value);
    setCustomSpecies("");
    setMessage(`已加入本次标注树种：${value}。提交标注后会写入类别表。`);
  }

  if (!item) {
    return <main className="frame-loading"><Loader2 className="spin" /> 正在载入全景序列… {message}</main>;
  }

  const selectedLabel = labels.find((label) => label.label_id === selectedId);
  const quality = preprocess.quality_before || {};
  return (
    <main className="frame-app">
      <header className="frame-toolbar">
        <div className="title-block">
          <h1><TreePine size={22} /> {task?.labeler_title || "全景影像逐帧标注"}</h1>
          <p>影像驱动模式 · 多目标、多树种 · 不依赖现有树点坐标</p>
        </div>
        <div className="navigation-controls">
          <label>采样间距
            <select value={spacing} onChange={(event) => start(Number(event.target.value))} disabled={busy}>
              <option value="0">全部原始帧</option>
              <option value="2">每 2 m 取一帧</option>
              <option value="5">每 5 m 取一帧</option>
              <option value="10">每 10 m 取一帧</option>
              <option value="15">每 15 m 取一帧（当前验收）</option>
            </select>
          </label>
          <button onClick={() => goTo(item.index - 1)} disabled={busy || item.index <= 0}><ArrowLeft size={17} /> 上一帧</button>
          <label className="jump-control">跳转
            <input type="number" min="1" max={item.total} value={jumpValue} onChange={(event) => setJumpValue(event.target.value)} />
          </label>
          <button onClick={() => goTo(Number(jumpValue) - 1)} disabled={busy}>前往</button>
          <button onClick={() => goTo(item.index + 1)} disabled={busy || item.index >= item.total - 1}>下一帧 <ArrowRight size={17} /></button>
        </div>
      </header>

      <section className="progress-strip">
        <strong>{item.index + 1} / {item.total}</strong>
        <span>有效有标签 {task?.valuable_frame_count ?? 0} 帧</span>
        <span>树轮廓 {task?.tree_label_count ?? 0} 个</span>
        <span>无标签 / 不可用 {task?.unlabeled_frame_count ?? item.total} 帧</span>
        <span>{item.frame.stream_id} · 当前流 {Number(item.frame.route_distance_m).toFixed(1)} m</span>
        <span>全路段合计 {Number(task?.route_length_m || 0).toFixed(1)} m</span>
        <span className="autosave-state"><Save size={14} /> 编辑停止 0.9 秒后自动保存草稿</span>
      </section>

      {message && <div className="frame-message">{message}</div>}

      <section className="frame-workspace">
        <div className="image-column">
          <div className="canvas-tools">
            <label>新轮廓树种
              <select value={activeSpecies} onChange={(event) => setActiveSpecies(event.target.value)}>
                {species.map((name) => <option value={name} key={name}>{name}</option>)}
              </select>
            </label>
            <span className="species-count">{species.length} 个树种候选</span>
            <div className="custom-species-control">
              <input value={customSpecies} onChange={(event) => setCustomSpecies(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter") addCustomSpecies(); }} placeholder="列表没有？输入树种" />
              <button onClick={addCustomSpecies} disabled={!customSpecies.trim()}><Plus size={15} /> 添加</button>
            </div>
            <span className="draw-help">
              {drawMode === "drag"
                ? "按住鼠标沿树冠边界拖动，松开后自动闭合；拖动顶点可修正，右键删除轮廓。"
                : "依次点击树冠边界，点击黄色起点闭合；拖动顶点修正；右键删除轮廓。"}
            </span>
            <label className="draw-mode-control">描边方式
              <select value={drawMode} onChange={(event) => setDrawMode(event.target.value)} disabled={draftPoints.length > 0}>
                <option value="drag">拖动描边（松开闭合）</option>
                <option value="click">逐点点击</option>
              </select>
            </label>
            <button onClick={() => setDraftPoints([])} disabled={!draftPoints.length}><RotateCcw size={15} /> 撤销当前轮廓</button>
            <label className="zoom-control">画面缩放
              <select value={zoom} onChange={(event) => setZoom(Number(event.target.value))}>
                <option value="1">100%</option>
                <option value="1.5">150%</option>
                <option value="2">200%</option>
              </select>
            </label>
            <a
              className="streetview-link"
              href={googleStreetViewUrl(item.frame)}
              target="_blank"
              rel="noreferrer"
              title="把当前帧的经纬度与车辆航向传给 Google Street View 进行外部核对"
            >
              <ExternalLink size={15} /> 谷歌街景核对
            </a>
          </div>
          <PanoramaCanvas
            image={item.image}
            labels={labels}
            draftPoints={draftPoints}
            activeSpecies={activeSpecies}
            selectedId={selectedId}
            preprocess={preprocess}
            showOriginal={showOriginal}
            zoom={zoom}
            drawMode={drawMode}
            onLabelsChange={setLabels}
            onDraftChange={setDraftPoints}
            onSelect={setSelectedId}
          />
          <div className="frame-metadata">
            <span><strong>Frame</strong>{item.frame.frame_id}</span>
            <span><strong>影像流</strong>{item.frame.stream_id}</span>
            <span><strong>Seq</strong>{item.frame.seq_id}</span>
            <span><strong>时间</strong>{item.frame.hong_kong_datetime}</span>
            <span><strong>经纬度</strong>{item.frame.latitude.toFixed(6)}, {item.frame.longitude.toFixed(6)}</span>
            <span><strong>航向</strong>{item.frame.heading_deg.toFixed(1)}°</span>
            <span><strong>分组</strong>{item.route_block_id}</span>
          </div>
        </div>

        <aside className="label-sidebar">
          <section className="side-section">
            <div className="section-title"><strong>本帧树木</strong><span>{labels.length} 棵</span></div>
            {labels.length === 0 ? <p className="empty-copy">尚未圈选树木。</p> : labels.map((label, index) => (
              <button className={label.label_id === selectedId ? "label-card selected" : "label-card"} key={label.label_id} onClick={() => setSelectedId(label.label_id)}>
                <span className="label-dot" style={{ background: labelColor(label.species) }} />
                <span><strong>树 {index + 1}</strong><small>{label.species} · {labelConfidence(label) === null ? "人工" : `YOLO ${(labelConfidence(label) * 100).toFixed(1)}%`}</small></span>
                <Trash2 size={15} onClick={(event) => { event.stopPropagation(); deleteLabel(label.label_id); }} />
              </button>
            ))}
          </section>

          {selectedLabel && (
            <section className="side-section selected-editor">
              <strong>编辑所选树木</strong>
              <label>树种
                <select value={selectedLabel.species} onChange={(event) => {
                  const nextSpecies = normalizeSpeciesForUi(event.target.value);
                  updateSelected({
                    species: nextSpecies,
                    confidence: nextSpecies === selectedLabel.species ? selectedLabel.confidence : null,
                  });
                }}>
                  {species.map((name) => <option value={name} key={name}>{name}</option>)}
                  {!species.includes(selectedLabel.species) && <option value={selectedLabel.species}>{selectedLabel.species}</option>}
                </select>
              </label>
              <label>可见性
                <select value={selectedLabel.visibility} onChange={(event) => updateSelected({ visibility: event.target.value })}>
                  <option value="clear">清晰</option>
                  <option value="partial">部分遮挡</option>
                  <option value="uncertain">树种不确定</option>
                </select>
              </label>
              <label>识别置信度
                <output>{labelConfidence(selectedLabel) === null ? "人工标注 / 无模型分数" : `YOLO ${(labelConfidence(selectedLabel) * 100).toFixed(1)}%`}</output>
              </label>
              <label>备注
                <input value={selectedLabel.note || ""} onChange={(event) => updateSelected({ note: event.target.value })} placeholder="可选" />
              </label>
            </section>
          )}

          <section className="side-section photo-panel">
            <div className="section-title"><strong>影像辅助</strong><span>{preprocess.recipe_version}</span></div>
            <label>曝光 <output>{Number(preprocess.exposure_ev) >= 0 ? "+" : ""}{Number(preprocess.exposure_ev).toFixed(1)} EV</output>
              <input type="range" min="-2" max="2" step="0.1" value={preprocess.exposure_ev} onChange={(event) => setPreprocess({ ...preprocess, exposure_ev: Number(event.target.value), auto_suggested: false, human_accepted: true })} />
            </label>
            <label>对比度 <output>{Number(preprocess.contrast).toFixed(2)}×</output>
              <input type="range" min="0.5" max="1.5" step="0.05" value={preprocess.contrast} onChange={(event) => setPreprocess({ ...preprocess, contrast: Number(event.target.value), auto_suggested: false, human_accepted: true })} />
            </label>
            <div className="photo-actions">
              <button onClick={autoAnalyze} disabled={analyzing}>{analyzing ? <Loader2 className="spin" size={15} /> : <Sparkles size={15} />} 自动建议</button>
              <button onPointerDown={() => setShowOriginal(true)} onPointerUp={() => setShowOriginal(false)} onPointerLeave={() => setShowOriginal(false)}><Eye size={15} /> 按住看原图</button>
              <button onClick={() => setPreprocess({ ...defaultPreprocess(), quality_before: preprocess.quality_before, human_accepted: true })}><RotateCcw size={15} /> 重置</button>
            </div>
            {Object.keys(quality).length > 0 && <div className="quality-readout">
              <span>{quality.quality_status}</span><span>中位 {quality.p50_luminance}/255</span><span>暗 {percent(quality.dark_pixel_ratio)}</span><span>亮 {percent(quality.bright_pixel_ratio)}</span>
            </div>}
          </section>

          <section className="side-section">
            <label>本帧备注
              <textarea value={note} onChange={(event) => setNote(event.target.value)} placeholder="场景、遮挡、重复帧或其他说明" />
            </label>
          </section>

          <section className="submission-actions">
            <button className="save-frame" onClick={() => submit("annotated")} disabled={busy}><Check size={17} /> 保存有树标注并下一帧</button>
            <button onClick={() => submit("no_tree")} disabled={busy}><LocateFixed size={17} /> 本帧无树</button>
            <button className="unusable" onClick={() => submit("unusable")} disabled={busy}><ImageOff size={17} /> 影像不可用</button>
          </section>
          <p className="split-note"><AlertTriangle size={14} /> 相邻帧高度相似，训练/验证/测试必须按 100 m 路段分组，不能随机按图拆分。</p>
        </aside>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
