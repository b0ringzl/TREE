from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import DATASET_DIR, TEMP_DIR, ensure_runtime_dirs, google_maps_api_key_source
from .data_access import list_species, read_traits_text
from .prebox_predictor import predict_preboxes
from .sam_assistant import discover_sam_models, predict_sam_polygon
from .schemas import ApiKeyRequest, RejectRequest, SamSegmentRequest, StartTaskRequest, SubmitRequest
from .task_manager import manager
from .tree_segmenter import predict_tree_segments

ensure_runtime_dirs()

app = FastAPI(title="HK Tree Street View Labeler")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/temp", StaticFiles(directory=str(TEMP_DIR)), name="temp")
app.mount("/dataset", StaticFiles(directory=str(DATASET_DIR)), name="dataset")


@app.middleware("http")
async def count_api_calls(request, call_next):
    if request.url.path.startswith("/api/"):
        manager.record_api_call(f"{request.method} {request.url.path}")
    return await call_next(request)


@app.get("/api/species/list")
def get_species_list() -> dict:
    return {"species": list_species()}


@app.get("/api/species/traits")
def get_species_traits(species: str) -> dict:
    try:
        return {"species": species, "traits": read_traits_text(species)}
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/config/status")
def get_config_status() -> dict:
    return {
        "google_maps_api_key_available": bool(manager.client.api_key),
        "google_maps_api_key_source": "session" if manager.client.api_key else google_maps_api_key_source(),
        "google_maps_api_key_length": len(manager.client.api_key or ""),
    }


@app.post("/api/config/api-key")
async def set_api_key(payload: ApiKeyRequest) -> dict:
    try:
        return await manager.configure_api_key(payload.api_key)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/config/validate")
async def validate_api_key() -> dict:
    try:
        await manager.client.validate_api_key()
        return {"status": "ok", "google_maps_api_key_available": True}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/task/start")
async def start_task(payload: StartTaskRequest) -> dict:
    try:
        return await manager.start(payload.species, payload.target_count)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/task/status")
def task_status() -> dict:
    return manager.status()


@app.get("/api/task/next")
async def next_task() -> dict:
    sample = await manager.next_sample()
    if not sample:
        return {"empty": True, "task": manager.status()}
    return {
        "tree_id": sample.tree_id,
        "species": sample.species,
        "images": sample.images,
        "candidates": sample.candidates,
        "tree": sample.tree,
        "warnings": sample.warnings,
    }


@app.get("/api/task/predict")
def predict_task(image: str) -> dict:
    try:
        return predict_preboxes(image)
    except Exception as exc:
        return {"image": image, "boxes": [], "error": str(exc)}


@app.get("/api/task/segment")
def segment_task(image: str) -> dict:
    try:
        return predict_tree_segments(image)
    except Exception as exc:
        return {"image": image, "candidates": [], "error": str(exc)}


@app.get("/api/sam/models")
def sam_models() -> dict:
    return {"models": discover_sam_models()}


@app.post("/api/task/sam-segment")
def sam_segment_task(payload: SamSegmentRequest) -> dict:
    try:
        return predict_sam_polygon(payload.image, payload.model_key, payload.point)
    except Exception as exc:
        return {"image": payload.image, "candidates": [], "error": str(exc)}


@app.post("/api/task/submit")
def submit_task(payload: SubmitRequest) -> dict:
    try:
        return manager.submit(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/task/reject")
def reject_task(payload: RejectRequest) -> dict:
    try:
        return manager.reject(payload.tree_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
