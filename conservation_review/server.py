"""Local-only, source-preserving binary species review. No model predictions are shown."""
from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import secrets
import sqlite3
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, urlparse

HERE = Path(__file__).resolve().parent
CHOICES = {"aquilaria", "not_aquilaria", "uncertain", "unusable"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class ReviewError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def safe_link(value):
    value = str(value or "")
    return value if urlparse(value).scheme in {"https", "http"} else ""


class ReviewStore:
    def __init__(self, manifest, database, image_roots=()):
        self.manifest = Path(manifest).resolve(strict=True)
        self.manifest_sha256 = sha256(self.manifest)
        self.roots = [self.manifest.parent] + [Path(p).resolve(strict=True) for p in image_roots]
        raw = json.loads(self.manifest.read_text(encoding="utf-8-sig"))
        if isinstance(raw, dict):
            raw = next((raw[k] for k in ("items", "rows", "records", "images") if isinstance(raw.get(k), list)), None)
        if not isinstance(raw, list):
            raise ValueError("Manifest must be a list, or contain an items/rows/records/images list.")
        self.items = {}
        for i, row in enumerate(raw):
            if not isinstance(row, dict):
                raise ValueError(f"Manifest row {i} is not an object")
            item_id = str(row.get("id") or row.get("sample_id") or f"item_{i:05d}")
            if not item_id or len(item_id) > 200 or item_id in self.items:
                raise ValueError(f"Invalid or duplicate item id: {item_id}")
            filename = next((row.get(k) for k in ("source_image", "local_path", "image_path", "path", "image") if row.get(k)), None)
            if not isinstance(filename, str):
                raise ValueError(f"No local image path for {item_id}")
            p = Path(filename)
            if not p.is_absolute():
                p = self.manifest.parent / p
            p = p.resolve()
            permitted = p.suffix.lower() in IMAGE_EXTENSIONS and any(p.is_relative_to(r) for r in self.roots)
            supplied_hash = str(row.get("sha256") or row.get("image_sha256") or "").lower()
            actual_hash = sha256(p) if permitted and p.is_file() else ""
            integrity = bool(permitted and actual_hash and (not supplied_hash or supplied_hash == actual_hash))
            label = row.get("source_species") or row.get("scientific_name") or "Not supplied / 未提供"
            priority = str(row.get("review_priority") or "normal")
            review_note = row.get("review_note", "")
            if not isinstance(review_note, str):
                review_note = ""
            item = {
                "id": item_id, "sha256": supplied_hash or actual_hash,
                "source_species": str(label), "source_url": safe_link(row.get("source_url") or row.get("url")),
                "source": str(row.get("source") or "Not supplied / 未提供"),
                "license": str(row.get("license") or "Not supplied / 未提供"),
                "attribution": str(row.get("attribution") or "Not supplied / 未提供"),
                "observation_id": str(row.get("observation_id") or ""),
                "review_priority": priority, "filename": p.name,
                "review_note": review_note[:3000],
                "image_url": f"/api/image/{quote(item_id, safe='')}",
                "integrity_ok": integrity,
                "integrity_error": "" if integrity else "Missing image, unapproved path, or SHA-256 mismatch / 影像缺失、路徑未授權或雜湊不符",
                "_path": p,
            }
            self.items[item_id] = item
        self.database = Path(database).resolve()
        if self.database == self.manifest:
            raise ValueError("Database cannot overwrite manifest")
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        with self.connect() as con:
            con.executescript("""
              CREATE TABLE IF NOT EXISTS decisions (
                item_id TEXT PRIMARY KEY, sha256 TEXT NOT NULL, decision TEXT NOT NULL,
                note TEXT NOT NULL, reviewer_identity TEXT NOT NULL, updated_at TEXT NOT NULL,
                revision INTEGER NOT NULL, session_id TEXT NOT NULL
              );
              CREATE TABLE IF NOT EXISTS history (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT, item_id TEXT NOT NULL,
                sha256 TEXT NOT NULL, session_id TEXT NOT NULL, action TEXT NOT NULL,
                previous_json TEXT, current_json TEXT, created_at TEXT NOT NULL,
                undone_at TEXT, manifest_sha256 TEXT NOT NULL
              );
              CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """)
            con.execute("INSERT OR IGNORE INTO metadata VALUES ('created_at', ?)", (utc_now(),))
            con.execute("INSERT OR IGNORE INTO metadata VALUES ('purpose', 'human binary review; not a license grant')")

    def connect(self):
        con = sqlite3.connect(str(self.database), timeout=10)
        con.row_factory = sqlite3.Row
        return con

    def get_item(self, item_id):
        if item_id not in self.items:
            raise ReviewError("Unknown image / 找不到影像", 404)
        return self.items[item_id]

    def verified_image(self, item_id):
        item = self.get_item(item_id)
        p = item["_path"].resolve()
        if not item["integrity_ok"] or not any(p.is_relative_to(r) for r in self.roots) or not p.is_file() or sha256(p) != item["sha256"]:
            raise ReviewError("Image integrity check failed; source not changed by this tool / 影像校驗失敗；本工具未修改原圖", 409)
        return p

    def listing(self):
        with self.lock, self.connect() as con:
            decisions = {r["item_id"]: dict(r) for r in con.execute("SELECT * FROM decisions")}
        result = []
        for item in self.items.values():
            clean = {k: v for k, v in item.items() if not k.startswith("_")}
            decision = decisions.get(item["id"])
            clean["decision"] = decision if decision and decision["sha256"] == item["sha256"] else None
            clean["stale_review"] = bool(decision and decision["sha256"] != item["sha256"])
            result.append(clean)
        return result

    def save(self, payload):
        if not isinstance(payload, dict):
            raise ReviewError("Expected a JSON object")
        item = self.get_item(str(payload.get("id", "")))
        self.verified_image(item["id"])
        decision = str(payload.get("decision", ""))
        if decision not in CHOICES:
            raise ReviewError("Unsupported decision")
        if payload.get("sha256") != item["sha256"]:
            raise ReviewError("Stale image hash; refresh before reviewing", 409)
        session_id = str(payload.get("session_id", ""))
        if not 8 <= len(session_id) <= 128:
            raise ReviewError("A review session id is required")
        note = str(payload.get("note", "")).strip()
        if len(note) > 3000:
            raise ReviewError("Note too long (maximum 3000 characters)")
        expected = payload.get("expected_revision", 0)
        if not isinstance(expected, int) or isinstance(expected, bool):
            raise ReviewError("Invalid revision")
        with self.lock, self.connect() as con:
            old = con.execute("SELECT * FROM decisions WHERE item_id=?", (item["id"],)).fetchone()
            previous = dict(old) if old else None
            revision = previous["revision"] if previous else 0
            if revision != expected:
                raise ReviewError("Review changed in another tab; refresh / 其他分頁已更新，請重新整理", 409)
            new = dict(item_id=item["id"], sha256=item["sha256"], decision=decision, note=note,
                       reviewer_identity="human", updated_at=utc_now(), revision=revision + 1, session_id=session_id)
            con.execute("INSERT OR REPLACE INTO decisions VALUES (:item_id,:sha256,:decision,:note,:reviewer_identity,:updated_at,:revision,:session_id)", new)
            event = con.execute("INSERT INTO history (item_id,sha256,session_id,action,previous_json,current_json,created_at,manifest_sha256) VALUES (?,?,?,'save',?,?,?,?)",
                                (item["id"], item["sha256"], session_id, json.dumps(previous), json.dumps(new), new["updated_at"], self.manifest_sha256))
            return {"decision": new, "event_id": event.lastrowid, "saved": True}

    def undo(self, payload):
        session_id = str(payload.get("session_id", ""))
        if not 8 <= len(session_id) <= 128:
            raise ReviewError("A review session id is required")
        with self.lock, self.connect() as con:
            event = con.execute("SELECT * FROM history WHERE session_id=? AND action='save' AND undone_at IS NULL ORDER BY event_id DESC LIMIT 1", (session_id,)).fetchone()
            if not event:
                raise ReviewError("Nothing to undo in this tab / 此分頁沒有可撤銷的決定", 404)
            now = con.execute("SELECT * FROM decisions WHERE item_id=?", (event["item_id"],)).fetchone()
            prior_saved = json.loads(event["current_json"])
            if not now or dict(now) != prior_saved:
                raise ReviewError("A newer decision exists; cannot undo safely / 已有較新的決定，不能安全撤銷", 409)
            previous = json.loads(event["previous_json"])
            if previous:
                con.execute("INSERT OR REPLACE INTO decisions VALUES (:item_id,:sha256,:decision,:note,:reviewer_identity,:updated_at,:revision,:session_id)", previous)
            else:
                con.execute("DELETE FROM decisions WHERE item_id=?", (event["item_id"],))
            stamp = utc_now()
            con.execute("UPDATE history SET undone_at=? WHERE event_id=?", (stamp, event["event_id"]))
            con.execute("INSERT INTO history (item_id,sha256,session_id,action,previous_json,current_json,created_at,manifest_sha256) VALUES (?,?,?,'undo',?,?,?,?)",
                        (event["item_id"], event["sha256"], session_id, json.dumps(prior_saved), json.dumps(previous), stamp, self.manifest_sha256))
            return {"id": event["item_id"], "decision": previous, "saved": True}

    def export(self):
        items = self.listing()
        with self.connect() as con:
            history = [dict(r) for r in con.execute("SELECT * FROM history ORDER BY event_id")]
        return {"schema_version": 1, "exported_at": utc_now(), "manifest": str(self.manifest),
                "manifest_sha256": self.manifest_sha256,
                "meaning": "Binary human identification/quality decisions only. Source species remain unchanged. No permission grant or automatic training approval.",
                "decisions": [{"id": i["id"], "source_species": i["source_species"], "source_url": i["source_url"],
                               "sha256": i["sha256"], "human_review": i["decision"]} for i in items if i["decision"]],
                "history": history}


class ReviewServer(ThreadingHTTPServer):
    def __init__(self, address, store):
        self.store = store
        self.csrf = secrets.token_urlsafe(32)
        super().__init__(address, ReviewHandler)


class ReviewHandler(BaseHTTPRequestHandler):
    server_version = "ConservationReview/1.0"

    def send_bytes(self, data, content_type, status=200, download=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{download}"')
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, body, status=200, download=None):
        self.send_bytes(json.dumps(body, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status, download)

    def local_host(self):
        host = self.headers.get("Host", "")
        return host in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}

    def handle_error(self, exc):
        status = exc.status if isinstance(exc, ReviewError) else 500
        message = str(exc) if isinstance(exc, ReviewError) else "Internal review error / 復核服務內部錯誤"
        if status == 500:
            self.log_error("%s: %s", type(exc).__name__, exc)
        self.send_json({"error": message}, status)

    def do_GET(self):
        try:
            if not self.local_host():
                raise ReviewError("Local host only", 403)
            path = urlparse(self.path).path
            if path == "/api/items":
                self.send_json({"items": self.server.store.listing(), "csrf_token": self.server.csrf,
                                "manifest_sha256": self.server.store.manifest_sha256,
                                "reviewer_identity": "human", "source_manifest": self.server.store.manifest.name})
            elif path == "/api/health":
                self.send_json({"ok": True, "total": len(self.server.store.items), "model_scores_shown": False})
            elif path == "/api/export":
                self.send_json(self.server.store.export(), download="conservation-human-review.json")
            elif path.startswith("/api/image/"):
                from urllib.parse import unquote
                image_path = self.server.store.verified_image(unquote(path.removeprefix("/api/image/")))
                self.send_bytes(image_path.read_bytes(), mimetypes.guess_type(image_path.name)[0] or "application/octet-stream")
            elif path in {"/", "/index.html", "/app.js", "/style.css"}:
                file = HERE / "static" / ("index.html" if path == "/" else path[1:])
                mime = {".html": "text/html", ".js": "text/javascript", ".css": "text/css"}[file.suffix]
                self.send_bytes(file.read_bytes(), mime + "; charset=utf-8")
            else:
                raise ReviewError("Not found", 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            self.handle_error(exc)

    def do_POST(self):
        try:
            if not self.local_host() or self.headers.get("X-Review-Token") != self.server.csrf:
                raise ReviewError("Invalid review session / 復核工作階段無效", 403)
            origin = self.headers.get("Origin")
            if origin and origin not in {f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"}:
                raise ReviewError("Cross-origin writes are not allowed", 403)
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                raise ReviewError("Expected application/json", 415)
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 16000:
                    raise ReviewError("Invalid request size", 413)
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict):
                    raise ReviewError("Expected a JSON object")
            except (ValueError, UnicodeDecodeError):
                raise ReviewError("Invalid JSON", 400)
            path = urlparse(self.path).path
            if path == "/api/decision":
                self.send_json(self.server.store.save(payload))
            elif path == "/api/undo":
                self.send_json(self.server.store.undo(payload))
            else:
                raise ReviewError("Not found", 404)
        except Exception as exc:
            self.handle_error(exc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=os.environ.get("CONSERVATION_REVIEW_MANIFEST"))
    parser.add_argument("--database", default=os.environ.get("CONSERVATION_REVIEW_DATABASE"))
    parser.add_argument("--image-root", action="append", default=[], help="Explicit additional read-only image directory; repeatable")
    parser.add_argument("--port", type=int, default=8051)
    args = parser.parse_args()
    if not args.manifest:
        parser.error("--manifest or CONSERVATION_REVIEW_MANIFEST is required")
    manifest = Path(args.manifest).resolve()
    db = args.database or HERE / "data" / (manifest.parent.name + "_" + hashlib.sha256(str(manifest).encode()).hexdigest()[:8]) / "human_review.sqlite3"
    store = ReviewStore(manifest, db, args.image_root)
    server = ReviewServer(("127.0.0.1", args.port), store)
    print(json.dumps({"url": f"http://127.0.0.1:{args.port}/?filter=pending", "items": len(store.items), "database": str(store.database)}, ensure_ascii=False), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
