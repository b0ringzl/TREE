import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  AlertTriangle,
  Check,
  Eraser,
  Loader2,
  Maximize2,
  Minimize2,
  Play,
  Search,
  Target,
  X,
} from "lucide-react";
import "./styles.css";

const API = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

function clamp(value, min = 0, max = 1) {
  return Math.min(max, Math.max(min, value));
}

function imageLabel(path) {
  const name = path.split("/").pop() || path;
  return name.replace(".jpg", "").replaceAll("_", " ");
}

function shotSummary(shot) {
  if (!shot) return "No camera metadata";
  return `heading ${Number(shot.heading).toFixed(1)} deg | pitch ${Number(shot.pitch).toFixed(1)} deg | fov ${shot.fov} | distance ${Number(shot.distance_m).toFixed(1)} m`;
}

async function predictBoxes(image) {
  const res = await fetch(`${API}/api/task/predict?image=${encodeURIComponent(image)}`);
  if (!res.ok) return [];
  const data = await res.json();
  return data.boxes?.length ? data.boxes.slice(0, 1) : [];
}

function boxToPixels(box, width, height) {
  return {
    x: (box.x_center - box.width / 2) * width,
    y: (box.y_center - box.height / 2) * height,
    w: box.width * width,
    h: box.height * height,
  };
}

function pixelsToBox(rect, width, height) {
  const x1 = clamp(Math.min(rect.x, rect.x + rect.w) / width);
  const y1 = clamp(Math.min(rect.y, rect.y + rect.h) / height);
  const x2 = clamp(Math.max(rect.x, rect.x + rect.w) / width);
  const y2 = clamp(Math.max(rect.y, rect.y + rect.h) / height);
  return {
    class_id: 0,
    x_center: (x1 + x2) / 2,
    y_center: (y1 + y2) / 2,
    width: Math.max(0.002, x2 - x1),
    height: Math.max(0.002, y2 - y1),
  };
}

function hitTest(boxes, x, y, width, height) {
  for (let i = boxes.length - 1; i >= 0; i -= 1) {
    const rect = boxToPixels(boxes[i], width, height);
    const handles = [
      ["nw", rect.x, rect.y],
      ["ne", rect.x + rect.w, rect.y],
      ["sw", rect.x, rect.y + rect.h],
      ["se", rect.x + rect.w, rect.y + rect.h],
    ];
    for (const [handle, hx, hy] of handles) {
      if (Math.abs(x - hx) <= 10 && Math.abs(y - hy) <= 10) return { index: i, mode: handle };
    }
    if (x >= rect.x && x <= rect.x + rect.w && y >= rect.y && y <= rect.y + rect.h) {
      return { index: i, mode: "move" };
    }
  }
  return null;
}

function CanvasAnnotator({ image, boxes, selected, onChange, onSelect }) {
  const canvasRef = useRef(null);
  const imageRef = useRef(null);
  const dragRef = useRef(null);

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

    boxes.forEach((box, index) => {
      const rect = boxToPixels(box, canvas.width, canvas.height);
      const isActive = selected === index;
      ctx.save();
      ctx.strokeStyle = isActive ? "#16a34a" : "#facc15";
      ctx.fillStyle = isActive ? "rgba(22, 163, 74, 0.14)" : "rgba(250, 204, 21, 0.12)";
      ctx.lineWidth = isActive ? 3 : 2;
      ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
      ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);
      ctx.fillStyle = isActive ? "#16a34a" : "#facc15";
      [
        [rect.x, rect.y],
        [rect.x + rect.w, rect.y],
        [rect.x, rect.y + rect.h],
        [rect.x + rect.w, rect.y + rect.h],
      ].forEach(([x, y]) => ctx.fillRect(x - 4, y - 4, 8, 8));
      ctx.restore();
    });
  }, [boxes, selected]);

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

  function onMouseDown(event) {
    const canvas = canvasRef.current;
    const p = point(event);
    const hit = hitTest(boxes, p.x, p.y, canvas.width, canvas.height);
    if (hit) {
      onSelect(hit.index);
      dragRef.current = { ...hit, start: p, original: boxes[hit.index] };
      return;
    }
    const next = [pixelsToBox({ x: p.x, y: p.y, w: 1, h: 1 }, canvas.width, canvas.height)];
    onChange(next);
    onSelect(0);
    dragRef.current = { index: 0, mode: "create", start: p, original: next[0] };
  }

  function onMouseMove(event) {
    const drag = dragRef.current;
    if (!drag) return;
    const canvas = canvasRef.current;
    const p = point(event);
    const original = boxToPixels(drag.original, canvas.width, canvas.height);
    let rect = { ...original };
    const dx = p.x - drag.start.x;
    const dy = p.y - drag.start.y;
    if (drag.mode === "move") rect = { ...rect, x: original.x + dx, y: original.y + dy };
    if (drag.mode === "create") rect = { x: drag.start.x, y: drag.start.y, w: dx, h: dy };
    if (drag.mode.includes("n")) {
      rect.y = original.y + dy;
      rect.h = original.h - dy;
    }
    if (drag.mode.includes("s")) rect.h = original.h + dy;
    if (drag.mode.includes("w")) {
      rect.x = original.x + dx;
      rect.w = original.w - dx;
    }
    if (drag.mode.includes("e")) rect.w = original.w + dx;
    const next = [...boxes];
    next[drag.index] = pixelsToBox(rect, canvas.width, canvas.height);
    onChange(next);
  }

  return (
    <canvas
      ref={canvasRef}
      width="640"
      height="640"
      className="annotator-canvas"
      title="Drag to create a box. Drag a box to move it. Drag corners to resize."
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseUp={() => (dragRef.current = null)}
      onMouseLeave={() => (dragRef.current = null)}
    />
  );
}

function SetupPanel({ onStart }) {
  const [species, setSpecies] = useState([]);
  const [query, setQuery] = useState("");
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

  useEffect(() => {
    const exactMatch = species.find((item) => item.toLowerCase() === query.toLowerCase());
    if (!exactMatch) {
      setTraits("");
      return;
    }
    const controller = new AbortController();
    fetch(`${API}/api/species/traits?species=${encodeURIComponent(exactMatch)}`, { signal: controller.signal })
      .then((res) => (res.ok ? res.json() : { traits: "" }))
      .then((data) => setTraits(data.traits || ""))
      .catch((err) => {
        if (err.name !== "AbortError") setTraits("");
      });
    return () => controller.abort();
  }, [query, species]);

  const filtered = useMemo(() => {
    const needle = query.toLowerCase();
    return species.filter((item) => item.toLowerCase().includes(needle)).slice(0, 12);
  }, [species, query]);

  async function start() {
    setBusy(true);
    setError("");
    try {
      const res = await fetch(`${API}/api/task/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ species: query, target_count: Number(targetCount) }),
      });
      if (!res.ok) throw new Error((await res.json()).detail || "Failed to start task");
      onStart(query);
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
            <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Filter species" title="Type to filter local species folders" />
          </div>
        </label>
        <div className="suggestions">
          {filtered.map((item) => (
            <button key={item} type="button" onClick={() => setQuery(item)} title={`Use ${item}`}>
              {item}
            </button>
          ))}
        </div>
        <label className="field compact">
          <span>Target count</span>
          <input type="number" min="1" max="10000" value={targetCount} onChange={(event) => setTargetCount(event.target.value)} title="Number of tree samples to cache for this session" />
        </label>
        {error && <div className="error-line">{error}</div>}
        <button className="primary" onClick={start} disabled={busy || !query} title="Start Street View collection and labeling">
          {busy ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
          Start
        </button>
        <section className="traits-panel" title="Reference text loaded from traits.txt">
          <div className="traits-title">Species traits</div>
          <pre>{traits || "Select an exact species name to view traits.txt."}</pre>
        </section>
      </section>
    </main>
  );
}

function Workspace({ species }) {
  const [sample, setSample] = useState(null);
  const [annotations, setAnnotations] = useState([]);
  const [selected, setSelected] = useState({ image: 0, box: 0 });
  const [expandedIndex, setExpandedIndex] = useState(-1);
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState(false);

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
      setSample(data);
      const seeded = await Promise.all(data.images.map(async (image) => ({ image, boxes: await predictBoxes(image), keep: true })));
      setAnnotations(seeded);
      setSelected({ image: 0, box: 0 });
      setExpandedIndex(-1);
      setStatus("");
    } finally {
      setBusy(false);
    }
  }, []);

  useEffect(() => {
    loadNext();
  }, [loadNext]);

  const submit = useCallback(async () => {
    if (!sample) return;
    setBusy(true);
    await fetch(`${API}/api/task/submit`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tree_id: sample.tree_id, species: sample.species, annotations }),
    });
    await loadNext();
  }, [annotations, loadNext, sample]);

  const reject = useCallback(async () => {
    if (!sample) return;
    setBusy(true);
    await fetch(`${API}/api/task/reject`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tree_id: sample.tree_id, species: sample.species }),
    });
    await loadNext();
  }, [loadNext, sample]);

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
              ? { ...item, boxes: item.boxes.filter((_, boxIndex) => boxIndex !== selected.box) }
              : item,
          ),
        );
        setSelected((current) => ({ ...current, box: -1 }));
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [reject, selected, submit]);

  function updateBoxes(index, boxes) {
    setAnnotations((current) => current.map((item, i) => (i === index ? { ...item, boxes: boxes.slice(0, 1) } : item)));
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

  function clearBoxes(index) {
    setAnnotations((current) => current.map((item, i) => (i === index ? { ...item, boxes: [] } : item)));
    setSelected({ image: index, box: -1 });
  }

  function clearAllBoxes() {
    setAnnotations((current) => current.map((item) => ({ ...item, boxes: [] })));
    setSelected({ image: 0, box: -1 });
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
          <button onClick={clearAllBoxes} disabled={busy || !sample} title="Clear all predicted and manual boxes">
            <Eraser size={18} /> Clear boxes
          </button>
          <button onClick={reject} disabled={busy || !sample} title="Reject this tree and delete temporary files (R)">
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

      <section className="canvas-grid">
        {annotations.map((item, index) => (
          <div className={`canvas-pane ${item.keep ? "" : "dropped"} ${expandedIndex === index ? "expanded" : ""}`} key={item.image}>
            <div className="pane-head">
              <span className="pane-title" title={shotSummary(sample?.candidates?.[index])}>
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
                <span title="Current box count">{item.boxes.length} boxes</span>
                <button type="button" className="icon-button" title="Clear boxes in this view" aria-label="Clear boxes in this view" onClick={() => clearBoxes(index)}>
                  <Eraser size={15} />
                </button>
              </div>
            </div>
            <CanvasAnnotator
              image={item.image}
              boxes={item.boxes}
              selected={selected.image === index ? selected.box : -1}
              onChange={(boxes) => updateBoxes(index, boxes)}
              onSelect={(box) => setSelected({ image: index, box })}
            />
            {!item.keep && <div className="drop-overlay">Dropped from dataset</div>}
          </div>
        ))}
      </section>
    </main>
  );
}

function App() {
  const [activeSpecies, setActiveSpecies] = useState("");
  return activeSpecies ? <Workspace species={activeSpecies} /> : <SetupPanel onStart={setActiveSpecies} />;
}

createRoot(document.getElementById("root")).render(<App />);
