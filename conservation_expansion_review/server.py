"""Multi-species human review adapter. Shares image/HTTP guards, never old choices/DB."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import sqlite3
import sys
from http.server import ThreadingHTTPServer
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from conservation_review import server as common

CHOICES = {"confirm_source", "reject_source", "uncertain", "unusable"}
SPECIES_ZH = {"Artocarpus hypargyreus": "白桂木", "Rhodoleia championii": "紅花荷", "Keteleeria fortunei": "油杉"}
OWNER = "conservation_expansion_review_v1"
COLUMNS = "item_id,sha256,decision,note,reviewer_identity,updated_at,revision,session_id,source_species,source_species_zh_hant"
PARAMETERS = ",".join(":" + column for column in COLUMNS.split(","))


def text(value, limit=3000):
    return value[:limit] if isinstance(value, str) else ""


def load_rows(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(raw, dict):
        raw = next((raw[k] for k in ("items", "rows", "records", "images") if isinstance(raw.get(k), list)), [])
    return raw


def plantnet_reference(value):
    """Explicit, bounded external-model reference; no arbitrary model fields pass through."""
    if not isinstance(value, dict):
        return None
    name = text(value.get("scientific_name"), 200)
    url = common.safe_link(value.get("source_url"))
    if not name or not url:
        return None
    score = value.get("score")
    if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score) or not 0 <= score <= 1:
        score = None
    return {"scientific_name": name, "score": score, "source_url": url,
            "checked_at": text(value.get("checked_at"), 100),
            "meaning": "Automatic model reference, not ground truth / 自動模型參考，並非真值"}


class ReviewStore(common.ReviewStore):
    def __init__(self, manifest, database, image_roots=()):
        database = Path(database).resolve()
        # Reject an old/shared DB before calling the base constructor (which creates tables).
        if database.exists():
            with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as con:
                try:
                    owner = con.execute("SELECT value FROM metadata WHERE key='review_tool_owner'").fetchone()
                except sqlite3.Error:
                    owner = None
            if not owner or owner[0] != OWNER:
                raise ValueError("Refusing non-expansion review DB; use a new dedicated path")
        super().__init__(manifest, database, image_roots)
        with self.connect() as con:
            columns = {r[1] for r in con.execute("PRAGMA table_info(decisions)")}
            for name in ("source_species", "source_species_zh_hant"):
                if name not in columns:
                    con.execute(f"ALTER TABLE decisions ADD COLUMN {name} TEXT NOT NULL DEFAULT ''")
            con.execute("INSERT OR IGNORE INTO metadata VALUES ('review_tool_owner', ?)", (OWNER,))
        for index, row in enumerate(load_rows(self.manifest)):
            item_id = str(row.get("id") or row.get("sample_id") or f"item_{index:05d}")
            item = self.items[item_id]
            names = row.get("source_names")
            source_zh = names[0] if isinstance(names, list) and names and isinstance(names[0], str) else ""
            item["source_species_zh_hant"] = text(row.get("source_species_zh_hant"), 200) or SPECIES_ZH.get(item["source_species"], source_zh)
            item["agent_question"] = text(row.get("agent_question")) or item.get("review_note", "")
            item["plantnet_reference"] = plantnet_reference(row.get("plantnet_reference"))

    def listing(self):
        result = super().listing()
        for item in result:
            decision = item["decision"]
            if decision and decision["source_species"] != item["source_species"]:
                item["decision"] = None
                item["stale_review"] = True
        return result

    def save(self, payload):
        if not isinstance(payload, dict):
            raise common.ReviewError("Expected a JSON object")
        item = self.get_item(str(payload.get("id", "")))
        self.verified_image(item["id"])
        choice = payload.get("decision")
        if not isinstance(choice, str) or choice not in CHOICES:
            raise common.ReviewError("Unsupported source-species review decision")
        if payload.get("sha256") != item["sha256"] or payload.get("source_species") != item["source_species"]:
            raise common.ReviewError("Image/species changed; refresh / 影像或核對樹種已變更，請重新整理", 409)
        session = str(payload.get("session_id", ""))
        note = str(payload.get("note", "")).strip()
        expected = payload.get("expected_revision", 0)
        if not 8 <= len(session) <= 128 or len(note) > 3000 or not isinstance(expected, int) or isinstance(expected, bool):
            raise common.ReviewError("Invalid session, note length or revision")
        with self.lock, self.connect() as con:
            old = con.execute("SELECT * FROM decisions WHERE item_id=?", (item["id"],)).fetchone()
            previous = dict(old) if old else None
            revision = previous["revision"] if previous else 0
            if revision != expected:
                raise common.ReviewError("Review changed in another tab; refresh / 其他分頁已更新，請重新整理", 409)
            new = dict(item_id=item["id"], sha256=item["sha256"], decision=choice, note=note,
                       reviewer_identity="human", updated_at=common.utc_now(), revision=revision + 1, session_id=session,
                       source_species=item["source_species"], source_species_zh_hant=item["source_species_zh_hant"])
            con.execute(f"INSERT OR REPLACE INTO decisions ({COLUMNS}) VALUES ({PARAMETERS})", new)
            event = con.execute("INSERT INTO history (item_id,sha256,session_id,action,previous_json,current_json,created_at,manifest_sha256) VALUES (?,?,?,'save',?,?,?,?)",
                                (item["id"], item["sha256"], session, json.dumps(previous), json.dumps(new), new["updated_at"], self.manifest_sha256))
            return {"saved": True, "decision": new, "event_id": event.lastrowid}

    def undo(self, payload):
        session = str(payload.get("session_id", ""))
        if not 8 <= len(session) <= 128:
            raise common.ReviewError("A review session id is required")
        with self.lock, self.connect() as con:
            event = con.execute("SELECT * FROM history WHERE session_id=? AND action='save' AND undone_at IS NULL ORDER BY event_id DESC LIMIT 1", (session,)).fetchone()
            if not event:
                raise common.ReviewError("Nothing to undo in this tab / 此分頁沒有可撤銷的決定", 404)
            current = con.execute("SELECT * FROM decisions WHERE item_id=?", (event["item_id"],)).fetchone()
            recorded = json.loads(event["current_json"])
            if not current or dict(current) != recorded:
                raise common.ReviewError("A newer decision exists; cannot undo safely / 已有較新的決定，不能安全撤銷", 409)
            previous = json.loads(event["previous_json"])
            if previous:
                con.execute(f"INSERT OR REPLACE INTO decisions ({COLUMNS}) VALUES ({PARAMETERS})", previous)
            else:
                con.execute("DELETE FROM decisions WHERE item_id=?", (event["item_id"],))
            stamp = common.utc_now()
            con.execute("UPDATE history SET undone_at=? WHERE event_id=?", (stamp, event["event_id"]))
            con.execute("INSERT INTO history (item_id,sha256,session_id,action,previous_json,current_json,created_at,manifest_sha256) VALUES (?,?,?,'undo',?,?,?,?)",
                        (event["item_id"], event["sha256"], session, json.dumps(recorded), json.dumps(previous), stamp, self.manifest_sha256))
            return {"id": event["item_id"], "decision": previous, "saved": True}

    def export(self):
        output = super().export()
        output["schema_version"] = "expansion-1"
        output["meaning"] = "Human confirmation/rejection of the recorded source species only; rejection does not assign another species. No license grant or automatic training approval. / 僅確認或否定來源樹種；否定不會指定其他物種。"
        return output


class Handler(common.ReviewHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        if path not in {"/", "/index.html", "/style.css", "/app.js", "/api/health"}:
            return super().do_GET()
        try:
            if not self.local_host():
                raise common.ReviewError("Local host only", 403)
            if path == "/api/health":
                return self.send_json({"ok": True, "total": len(self.server.store.items), "review_type": OWNER,
                                       "external_model_reference_is_not_ground_truth": True})
            if path == "/style.css":
                data = (common.HERE / "static" / "style.css").read_bytes() + (HERE / "static" / "style.css").read_bytes()
                return self.send_bytes(data, "text/css; charset=utf-8")
            name = "index.html" if path == "/" else path[1:]
            mime = "text/javascript" if name.endswith(".js") else "text/html"
            self.send_bytes((HERE / "static" / name).read_bytes(), mime + "; charset=utf-8")
        except Exception as error:
            self.handle_error(error)


class ReviewServer(ThreadingHTTPServer):
    def __init__(self, address, store):
        self.store = store
        self.csrf = secrets.token_urlsafe(32)
        super().__init__(address, Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=os.environ.get("EXPANSION_REVIEW_MANIFEST"))
    parser.add_argument("--database", default=os.environ.get("EXPANSION_REVIEW_DATABASE"))
    parser.add_argument("--image-root", action="append", default=[])
    parser.add_argument("--port", type=int, default=8052)
    args = parser.parse_args()
    if not args.manifest:
        parser.error("--manifest or EXPANSION_REVIEW_MANIFEST is required")
    manifest = Path(args.manifest).resolve()
    database = args.database or HERE / "data" / (manifest.parent.name + "_" + hashlib.sha256(str(manifest).encode()).hexdigest()[:8]) / "human_review.sqlite3"
    store = ReviewStore(manifest, database, args.image_root)
    service = ReviewServer(("127.0.0.1", args.port), store)
    print(json.dumps({"url": f"http://127.0.0.1:{args.port}/?filter=pending", "items": len(store.items), "database": str(store.database)}, ensure_ascii=False), flush=True)
    try:
        service.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        service.server_close()


if __name__ == "__main__":
    main()
