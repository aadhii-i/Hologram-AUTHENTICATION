"""FastAPI service exposing the hologram detection pipeline.

Endpoints
---------
POST /upload            - upload a video file, get back a `video_id`
POST /detect             - run detection on a previously uploaded `video_id`
GET  /result/{video_id}  - fetch the last stored detection result
GET  /download/{video_id}/{artifact} - download a generated output file

Run with: `uvicorn api.main:app --reload`
"""

from __future__ import annotations

import shutil
import sys
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.detector import DetectionResult, HologramDetector  # noqa: E402
from app.utils import ensure_dir, get_logger, load_settings  # noqa: E402

logger = get_logger("api")

app = FastAPI(
    title="Hologram Genuineness Detection API",
    description="Classical computer-vision pipeline that classifies holograms "
    "as GENUINE or COUNTERFEIT from their G->N->V tilt color transition.",
    version="1.0.0",
)

_settings = load_settings()
_upload_dir = ensure_dir(_settings.get("api", {}).get("upload_dir", "outputs/uploads"))
_detector = HologramDetector()
_results: dict[str, DetectionResult] = {}
_uploads: dict[str, Path] = {}


class DetectRequest(BaseModel):
    video_id: str
    roi_mode: Optional[str] = None  # "auto" | "aruco" | "manual"


class UploadResponse(BaseModel):
    video_id: str
    filename: str
    size_bytes: int


@app.get("/")
def root():
    return {"service": "hologram-detector", "status": "ok"}


@app.post("/upload", response_model=UploadResponse)
async def upload_video(file: UploadFile = File(...)):
    supported = _settings.get("video", {}).get(
        "supported_extensions", [".mp4", ".mov", ".avi", ".mkv"]
    )
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Supported: {supported}",
        )

    video_id = uuid.uuid4().hex[:12]
    dest = _upload_dir / f"{video_id}{suffix}"
    with open(dest, "wb") as out_file:
        shutil.copyfileobj(file.file, out_file)

    _uploads[video_id] = dest
    logger.info("Uploaded video %s -> %s", file.filename, dest)
    return UploadResponse(video_id=video_id, filename=file.filename or dest.name, size_bytes=dest.stat().st_size)


@app.post("/detect")
async def detect(request: DetectRequest):
    video_path = _uploads.get(request.video_id)
    if video_path is None or not video_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No uploaded video found for video_id '{request.video_id}'. Call /upload first.",
        )

    try:
        result = _detector.process_video(
            str(video_path), roi_mode=request.roi_mode, video_id=request.video_id
        )
    except Exception as exc:  # surfaced as a 422 rather than an opaque 500
        logger.exception("Detection failed for %s", request.video_id)
        raise HTTPException(status_code=422, detail=f"Detection failed: {exc}") from exc

    _results[request.video_id] = result
    return result.to_json_dict()


@app.get("/result/{video_id}")
async def get_result(video_id: str):
    result = _results.get(video_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"No result for video_id '{video_id}'. Run /detect first.",
        )
    return result.to_json_dict()


@app.get("/download/{video_id}/{artifact}")
async def download(video_id: str, artifact: str):
    result = _results.get(video_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No result for video_id '{video_id}'.")

    path_str = result.output_paths.get(artifact)
    if not path_str:
        raise HTTPException(
            status_code=404,
            detail=f"No artifact named '{artifact}'. Available: {list(result.output_paths)}",
        )

    path = Path(path_str)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Artifact file missing on disk: {path}")

    return FileResponse(path, filename=path.name)
