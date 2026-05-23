from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import TEMP_DIR, ensure_runtime_dirs
from .data_access import list_species, read_traits_text
from .schemas import RejectRequest, StartTaskRequest, SubmitRequest
from .task_manager import manager

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
        "warnings": sample.warnings,
    }


@app.get("/api/task/predict")
def predict_task(image: str) -> dict:
    # Replace this fallback with an Ultralytics/ONNX model call when weights are available.
    return {
        "image": image,
        "boxes": [],
    }


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
