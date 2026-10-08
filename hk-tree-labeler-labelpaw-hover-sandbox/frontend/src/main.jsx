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
  ExternalLink,
  KeyRound,
  Loader2,
  Maximize2,
  Minimize2,
  Play,
  Search,
  ShieldCheck,
  Sparkles,
  Target,
  X,
} from "lucide-react";
import {
  appendCandidatePolygon,
  buildFreehandPolygon,
  candidateToPolygon,
  contextMenuHitTest,
  pixelsToPoint,
  polygonHitTest,
  polygonToPixels,
  shouldRefreshHoverPoint,
  vertexHitTest,
} from "./annotationGeometry.js";
import "./styles.css";

const API = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8010";

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

function googleMapsPanoUrl(shot) {
  if (!shot?.lat || !shot?.lon) return "";
  const params = new URLSearchParams({
    api: "1",
    map_action: "pano",
    viewpoint: `${shot.lat},${shot.lon}`,
  });
  if (Number.isFinite(Number(shot.heading))) params.set("heading", Number(shot.heading).toFixed(2));
  if (Number.isFinite(Number(shot.pitch))) params.set("pitch", Number(shot.pitch).toFixed(2));
  if (Number.isFinite(Number(shot.fov))) params.set("fov", String(Math.round(Number(shot.fov))));
  return `https://www.google.com/maps/@?${params.toString()}`;
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
  selectedPolygon,
  selectedVertex,
  samAssistEnabled = false,
  samAssistBusy = false,
  samPreviewCandidate = null,
  onPolygonsChange,
  onDraftChange,
  onSelect,
  onSamAssistPoint,
  onSamHoverPoint,
  onSamConfirmPreview,
  onSamPreviewClear,
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
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

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

    const previewPolygon = candidateToPolygon(samPreviewCandidate);
    if (previewPolygon) {
      const points = polygonToPixels(previewPolygon.points, canvas.width, canvas.height);
      ctx.save();
      ctx.strokeStyle = "#22c55e";
      ctx.fillStyle = "rgba(34, 197, 94, 0.18)";
      ctx.lineWidth = 2.5;
      ctx.setLineDash([7, 5]);
      ctx.beginPath();
      points.forEach(([x, y], pointIndex) => {
        if (pointIndex === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    }

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
  }, [draftPoints, freehandPoints, polygons, samPreviewCandidate, selectedPolygon, selectedVertex]);

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
    if (samAssistEnabled && draftPoints.length === 0 && onSamConfirmPreview?.()) {
      onSelect(-1, -1);
      return;
    }
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

    if (samAssistEnabled && draftPoints.length === 0 && !pressRef.current?.active && onSamHoverPoint) {
      const vertexHit = vertexHitTest(polygons, p.x, p.y, canvas.width, canvas.height);
      const polygonIndex = polygonHitTest(polygons, p.x, p.y, canvas.width, canvas.height);
      if (vertexHit || polygonIndex >= 0) {
        onSamPreviewClear?.();
      } else {
        onSamHoverPoint(pixelsToPoint(p, canvas.width, canvas.height));
      }
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
        title={samAssistEnabled ? "AI assist is enabled. Hover over blank crown area for a SAM preview, then click to confirm." : "Click to add polygon vertices. Long-press empty space to draw a closed freehand curve. Right-click labels to delete."}
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

function SetupPanel({ onStart, onOpenLookup }) {
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
            <h1>HK Tree Street View Labeler</h1>
            <p>Choose one local species folder and start an on-demand review session.</p>
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
              placeholder="Filter species"
              title="Type to filter local species folders"
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
            <span>{selectedSpecies || "Select species"}</span>
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
          <span>Target count</span>
          <input type="number" min="1" max="10000" value={targetCount} onChange={(event) => setTargetCount(event.target.value)} title="Number of tree samples to cache for this session" />
        </label>
        {error && <div className="error-line">{error}</div>}
        <button className="primary" onClick={start} disabled={busy || !selectedSpecies} title="Start Street View collection and labeling">
          {busy ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
          Start
        </button>
        <button className="secondary-wide" onClick={onOpenLookup} disabled={busy} title="Look up one tree by global tree ID without choosing species">
          <Search size={18} />
          根据ID查询该树街景图像
        </button>
        <section className="traits-panel" title="Reference text loaded from traits.txt">
          <div className="traits-title">Species traits</div>
          <pre>{traits || "Select an exact species name to view traits.txt."}</pre>
        </section>
      </section>
    </main>
  );
}

function LookupPanel({ onBack, onOpenSample }) {
  const [treeId, setTreeId] = useState("");
  const [matches, setMatches] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [status, setStatus] = useState("");

  async function lookup(registryId = null) {
    const trimmed = treeId.trim();
    if (!trimmed) {
      setError("请输入树木编号。");
      return;
    }
    setBusy(true);
    setError("");
    setStatus(registryId ? "正在抓取所选树木附近的街景图像..." : "正在查询全局树木数据库...");
    try {
      const res = await fetch(`${API}/api/task/lookup-id`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tree_id: trimmed, registry_id: registryId }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "ID 查询失败");
      if (data.multiple) {
        setMatches(data.matches || []);
        setStatus("该编号对应多棵树，请选择具体记录。");
        return;
      }
      if (!data.sample) throw new Error("没有返回可标注的街景样本。");
      setMatches([]);
      onOpenSample(data.sample);
    } catch (err) {
      setError(err.message);
      setStatus("");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="setup-shell">
      <section className="setup-panel lookup-panel">
        <div className="title-row">
          <div>
            <h1>根据ID查询该树街景图像</h1>
            <p>输入树木编号后，系统会在全局树木数据库中查找树种和坐标，再抓取附近 Google Street View 图像。</p>
          </div>
          <button type="button" onClick={onBack} disabled={busy} title="返回树种选择页面">
            <ArrowLeft size={18} /> 返回
          </button>
        </div>
        <label className="field">
          <span>树木编号 ID</span>
          <div className="search-box">
            <Search size={18} />
            <input
              value={treeId}
              onChange={(event) => {
                setTreeId(event.target.value);
                setMatches([]);
                setError("");
                setStatus("");
              }}
              placeholder="例如 TREE-001 或坐标表中的 tree_id"
              title="输入坐标表中的树木编号"
              onKeyDown={(event) => {
                if (event.key === "Enter") lookup();
              }}
            />
          </div>
        </label>
        <button className="primary" onClick={() => lookup()} disabled={busy || !treeId.trim()} title="查询并抓取该树附近街景图像">
          {busy ? <Loader2 className="spin" size={18} /> : <Search size={18} />}
          查询街景图像
        </button>
        {status && <div className="status-line neutral">{status}</div>}
        {error && <div className="error-line">{error}</div>}
        {matches.length > 0 && (
          <section className="match-list" title="Multiple registry rows matched this tree ID">
            {matches.map((match) => (
              <button
                type="button"
                key={match.registry_id}
                className="match-row"
                onClick={() => lookup(match.registry_id)}
                disabled={busy}
                title="使用这条记录抓取街景图像"
              >
                <span>
                  <strong>{match.tree_id}</strong>
                  <em>{match.species}</em>
                </span>
                <span>{formatNumber(match.lon)} / {formatNumber(match.lat)}</span>
              </button>
            ))}
          </section>
        )}
      </section>
    </main>
  );
}

function ResultsPanel({ onBack, onChooseSpecies }) {
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
          <button onClick={onChooseSpecies} title="Return to species selection">
            <Search size={18} /> 选择树种
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

function Workspace({ species, initialSample = null, onInitialSampleConsumed, onShowResults, onChooseSpeciesRequest }) {
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
  const [samPreview, setSamPreview] = useState({ image: -1, candidate: null });
  const samHoverRef = useRef({ timer: null, requestId: 0, lastImage: -1, lastPoint: null });

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

  const seedSample = useCallback((data) => {
    const seeded = data.images.map((image) => ({
      image,
      polygons: [],
      draftPoints: [],
      keep: true,
    }));
    const entry = { sample: data, annotations: seeded };
    setHistory((current) => {
      const base = historyIndex >= 0 ? current.slice(0, historyIndex + 1) : current;
      return [...base, entry];
    });
    applyHistoryEntry(entry, historyIndex + 1);
  }, [applyHistoryEntry, historyIndex]);

  const loadNext = useCallback(async () => {
    setBusy(true);
    try {
      const res = await fetch(`${API}/api/task/next`);
      const data = await res.json();
      if (data.empty) {
        setStatus(data.task?.message || "Waiting for cached samples...");
        window.setTimeout(loadNext, 2500);
        return;
      }
      seedSample(data);
    } finally {
      setBusy(false);
    }
  }, [seedSample]);

  useEffect(() => {
    if (initialSample) {
      seedSample(initialSample);
      onInitialSampleConsumed?.();
      return;
    }
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

  useEffect(() => {
    setSamPreview({ image: -1, candidate: null });
    samHoverRef.current.lastPoint = null;
    samHoverRef.current.lastImage = -1;
  }, [samModelKey, samEnabled]);

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

  const reject = useCallback(async () => {
    if (!sample) return;
    setBusy(true);
    try {
      await fetch(`${API}/api/task/reject`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tree_id: sample.tree_id, species: sample.species }),
      });
      await advanceAfterAction();
    } finally {
      setBusy(false);
    }
  }, [history, historyIndex, loadNext, sample]);

  useEffect(() => {
    function onKeyDown(event) {
      if (event.target instanceof HTMLInputElement) return;
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
    setAnnotations((current) => {
      const next = current.map((item, i) => (i === index ? { ...item, keep: !item.keep } : item));
      if (sample && next.length > 0 && next.every((item) => !item.keep)) {
        window.setTimeout(() => reject(), 0);
      }
      return next;
    });
  }

  function clearPolygons(index) {
    setAnnotations((current) => current.map((item, i) => (i === index ? { ...item, polygons: [], draftPoints: [] } : item)));
    setSelected({ image: index, polygon: -1, vertex: -1 });
  }

  function clearAllPolygons() {
    setAnnotations((current) => current.map((item) => ({ ...item, polygons: [], draftPoints: [] })));
    setSelected({ image: 0, polygon: -1, vertex: -1 });
    setSamPreview({ image: -1, candidate: null });
  }

  async function requestSamAssist(index, point) {
    if (!samEnabled || !samModelKey || samBusyImage >= 0) return;
    const item = annotations[index];
    if (!item) return;
    setSamBusyImage(index);
    setSamMessage("SAM is segmenting the clicked crown area...");
    try {
      const res = await fetch(`${API}/api/task/sam-hover`, {
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

  function requestSamHover(index, point) {
    if (!samEnabled || !samModelKey || samBusyImage >= 0) return;
    const hover = samHoverRef.current;
    if (hover.lastImage === index && !shouldRefreshHoverPoint(hover.lastPoint, point)) return;
    hover.lastImage = index;
    hover.lastPoint = point;
    if (hover.timer) window.clearTimeout(hover.timer);
    hover.timer = window.setTimeout(async () => {
      const item = annotations[index];
      if (!item) return;
      const requestId = hover.requestId + 1;
      hover.requestId = requestId;
      setSamBusyImage(index);
      try {
        const res = await fetch(`${API}/api/task/sam-hover`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image: item.image, model_key: samModelKey, point }),
        });
        const data = await res.json();
        if (requestId !== samHoverRef.current.requestId) return;
        if (!res.ok) throw new Error(data.detail || "SAM hover request failed");
        if (data.error) throw new Error(data.error);
        const candidate = data.candidates?.[0] || null;
        setSamPreview({ image: index, candidate });
        setSamMessage(candidate ? "SAM hover preview ready. Click to confirm." : "SAM did not find a polygon under the cursor.");
      } catch (err) {
        if (requestId === samHoverRef.current.requestId) setSamMessage(err.message);
      } finally {
        if (requestId === samHoverRef.current.requestId) setSamBusyImage(-1);
      }
    }, 260);
  }

  function confirmSamPreview(index) {
    if (samPreview.image !== index || !samPreview.candidate) return false;
    const nextPolygonIndex = annotations[index]?.polygons.length || 0;
    setAnnotations((current) =>
      current.map((entry, imageIndex) =>
        imageIndex === index
          ? {
              ...entry,
              polygons: appendCandidatePolygon(entry.polygons, samPreview.candidate),
              draftPoints: [],
            }
          : entry,
      ),
    );
    setSelected({ image: index, polygon: nextPolygonIndex, vertex: -1 });
    setSamMessage(`Added hover SAM polygon from ${samPreview.candidate.model_key || samModelKey}.`);
    setSamPreview({ image: -1, candidate: null });
    return true;
  }

  function clearSamPreview(index) {
    if (samPreview.image === index) setSamPreview({ image: -1, candidate: null });
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
          <button onClick={onChooseSpeciesRequest} disabled={busy} title="Open the current report before returning to species selection">
            <ArrowLeft size={18} /> 返回树种选择
          </button>
          <button onClick={clearAllPolygons} disabled={busy || !sample} title="Clear all predicted and manual polygons">
            <Eraser size={18} /> Clear polygons
          </button>
          <button onClick={reject} disabled={busy || !sample} title="Reject this tree now; files are kept for one-step back and removed when you start the tree after next (R)">
            <X size={18} /> Reject
          </button>
          <button className="primary" onClick={submit} disabled={busy || !sample || keptCount === 0} title="Save kept views and load next tree (Space)">
            {busy ? <Loader2 className="spin" size={18} /> : <Check size={18} />} Save {keptCount}/3
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
              const mapsUrl = googleMapsPanoUrl(shot);
              return (
                <>
            <div className="pane-head">
              <span className="pane-title" title={shotSummary(shot)}>
                <Target size={14} />
                {imageLabel(item.image)}
              </span>
              <div className="pane-tools">
                {mapsUrl ? (
                  <a
                    className="maps-link"
                    href={mapsUrl}
                    target="_blank"
                    rel="noreferrer"
                    title="Open this exact sampling coordinate, heading, pitch, and FOV in Google Maps Street View"
                    aria-label="Open this view in Google Maps Street View"
                  >
                    <ExternalLink size={15} />
                    Maps
                  </a>
                ) : (
                  <button type="button" disabled title="Street View coordinate metadata is unavailable">
                    <ExternalLink size={15} />
                    Maps
                  </button>
                )}
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
              selectedPolygon={selected.image === index ? selected.polygon : -1}
              selectedVertex={selected.image === index ? selected.vertex : -1}
              samAssistEnabled={samEnabled && Boolean(samModelKey)}
              samAssistBusy={samBusyImage === index}
              samPreviewCandidate={samPreview.image === index ? samPreview.candidate : null}
              onPolygonsChange={(polygons) => updatePolygons(index, polygons)}
              onDraftChange={(draftPoints) => updateDraft(index, draftPoints)}
              onSelect={(polygon, vertex = -1) => setSelected({ image: index, polygon, vertex })}
              onSamAssistPoint={(point) => requestSamAssist(index, point)}
              onSamHoverPoint={(point) => requestSamHover(index, point)}
              onSamConfirmPreview={() => confirmSamPreview(index)}
              onSamPreviewClear={() => clearSamPreview(index)}
            />
            {!item.keep && <div className="drop-overlay">Dropped from dataset</div>}
            <div className="view-meta" title="Street View sampling metadata for this image">
              <div><strong>ID</strong><span>{sample?.tree_id || "-"}</span></div>
              <div><strong>X</strong><span>{formatNumber(shot?.lon)}</span></div>
              <div><strong>Y</strong><span>{formatNumber(shot?.lat)}</span></div>
              <div><strong>Distance</strong><span>{formatNumber(shot?.distance_m, 2)} m</span></div>
              <div><strong>Captured</strong><span>{formatStreetViewDate(shot?.date)}</span></div>
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
  const [apiReady, setApiReady] = useState(false);
  const [activeSpecies, setActiveSpecies] = useState("");
  const [page, setPage] = useState("setup");
  const [initialSample, setInitialSample] = useState(null);
  const [reportReturnTarget, setReportReturnTarget] = useState("workspace");

  function openReport(returnTarget = "workspace") {
    setReportReturnTarget(returnTarget);
    setPage("report");
  }

  function chooseSpecies() {
    setActiveSpecies("");
    setInitialSample(null);
    setPage("setup");
  }

  if (!apiReady) return <ApiKeyPanel onReady={() => setApiReady(true)} />;
  if (page === "lookup") {
    return (
      <LookupPanel
        onBack={chooseSpecies}
        onOpenSample={(sample) => {
          setActiveSpecies(sample.species);
          setInitialSample(sample);
          setPage("workspace");
        }}
      />
    );
  }
  if (page === "report") {
    return (
      <ResultsPanel
        onBack={() => (reportReturnTarget === "setup" ? chooseSpecies() : setPage(activeSpecies ? "workspace" : "setup"))}
        onChooseSpecies={chooseSpecies}
      />
    );
  }
  if (page === "workspace" && activeSpecies) {
    return (
      <Workspace
        species={activeSpecies}
        initialSample={initialSample}
        onInitialSampleConsumed={() => setInitialSample(null)}
        onShowResults={() => openReport("workspace")}
        onChooseSpeciesRequest={() => openReport("setup")}
      />
    );
  }
  return (
    <SetupPanel
      onOpenLookup={() => setPage("lookup")}
      onStart={(species) => {
        setActiveSpecies(species);
        setInitialSample(null);
        setPage("workspace");
      }}
    />
  );
}

createRoot(document.getElementById("root")).render(<App />);
