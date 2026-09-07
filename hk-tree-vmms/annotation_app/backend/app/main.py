from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import (
    ANNOTATION_DIR,
    FRAME_LABELER_DATASET,
    FRAME_LABELER_TITLE,
    FRAME_STREAM_IDS,
    VIEW_CACHE_DIR,
    ensure_dirs,
)
from .schemas import (
    CheckpointRequest,
    ImageAnalyzeRequest,
    FrameAnalyzeRequest,
    FrameAnnotationRequest,
    FrameStartRequest,
    RejectRequest,
    SamSegmentRequest,
    StartTaskRequest,
    SubmitRequest,
)
from .frame_manager import frame_manager
from .task_manager import manager


ensure_dirs()
app = FastAPI(title=f"HK Tree VMMS {FRAME_LABELER_TITLE}")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5177",
        "http://127.0.0.1:5177",
        "http://localhost:5178",
        "http://127.0.0.1:5178",
        "http://localhost:5179",
        "http://127.0.0.1:5179",
        "http://localhost:5180",
        "http://127.0.0.1:5180",
        "http://localhost:5181",
        "http://127.0.0.1:5181",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/views", StaticFiles(directory=str(VIEW_CACHE_DIR)), name="views")
app.mount("/annotations", StaticFiles(directory=str(ANNOTATION_DIR)), name="annotations")


@app.middleware("http")
async def count_api_calls(request, call_next):
    if request.url.path.startswith("/api/"):
        manager.record_api_call(f"{request.method} {request.url.path}")
    return await call_next(request)


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "mode": f"local_vmms_{FRAME_LABELER_DATASET}",
        "labeler_title": FRAME_LABELER_TITLE,
        "frame_stream_ids": list(FRAME_STREAM_IDS),
        "frame_count": len(frame_manager.frames),
        "frames": len(manager.frames),
        "trees": len(manager.trees),
        "species": len(manager.list_species()),
    }


@app.get("/api/config/status")
def config_status() -> dict:
    return {
        "local_vmms_ready": True,
        "google_maps_api_key_available": False,
        "google_maps_api_key_source": "not_required",
        "google_maps_api_key_length": 0,
    }


@app.get("/api/species/list")
def species_list() -> dict:
    return {"species": manager.list_species()}


@app.get("/api/species/traits")
def species_traits(species: str) -> dict:
    return {"species": species, "traits": manager.species_info(species)}


@app.post("/api/task/start")
def start_task(payload: StartTaskRequest) -> dict:
    try:
        return manager.start(payload.species, payload.target_count)
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/task/status")
def task_status() -> dict:
    return manager.status()


@app.get("/api/task/next")
def next_task() -> dict:
    sample = manager.next_sample()
    if sample is None:
        return {"empty": True, "task": manager.status()}
    return {
        "tree_id": sample.tree_id,
        "species": sample.species,
        "images": sample.images,
        "candidates": sample.candidates,
        "tree": sample.tree,
        "warnings": sample.warnings,
        "draft_annotations": sample.draft_annotations,
    }


@app.post("/api/task/submit")
def submit_task(payload: SubmitRequest) -> dict:
    try:
        return manager.submit(payload)
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/task/checkpoint")
def checkpoint_task(payload: CheckpointRequest) -> dict:
    try:
        return manager.checkpoint(payload)
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/task/reject")
def reject_task(payload: RejectRequest) -> dict:
    try:
        return manager.reject(payload.tree_id, payload.annotations)
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/sam/models")
def sam_models() -> dict:
    return {"models": []}


@app.post("/api/image/analyze")
def analyze_image(payload: ImageAnalyzeRequest) -> dict:
    try:
        return manager.analyze_image(payload.image)
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/frame/status")
def frame_status() -> dict:
    return frame_manager.status()


@app.get("/api/frame/species")
def frame_species() -> dict:
    return {"species": frame_manager.species}


@app.post("/api/frame/start")
def frame_start(payload: FrameStartRequest) -> dict:
    try:
        return frame_manager.start(payload.spacing_m)
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/frame/item")
def frame_item(index: int = 0) -> dict:
    try:
        return frame_manager.item(index)
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/frame/image/{frame_id}")
def frame_image(frame_id: str) -> FileResponse:
    try:
        return FileResponse(frame_manager.preview_file(frame_id), media_type="image/jpeg")
    except Exception as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post("/api/frame/analyze")
def frame_analyze(payload: FrameAnalyzeRequest) -> dict:
    try:
        return frame_manager.analyze(payload.frame_id)
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/frame/draft")
def frame_draft(payload: FrameAnnotationRequest) -> dict:
    try:
        return frame_manager.save_draft(payload)
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/frame/submit")
def frame_submit(payload: FrameAnnotationRequest) -> dict:
    try:
        return frame_manager.submit(payload)
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/task/sam-segment")
def sam_segment(_: SamSegmentRequest) -> dict:
    return {"candidates": [], "error": "SAM will be connected after panorama crop calibration."}


@app.get("/api/task/predict")
def predict_task(image: str) -> dict:
    return {"image": image, "boxes": [], "source": "manual_only"}


@app.get("/api/task/segment")
def segment_task(image: str) -> dict:
    return {"image": image, "candidates": [], "source": "manual_only"}
