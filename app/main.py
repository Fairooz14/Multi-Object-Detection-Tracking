"""
Endpoints:
  POST /detect    — single image detection
  POST /track     — video multi-object tracking
  GET  /health    — health check
  GET  /benchmark — latency & FPS benchmark
"""

import io
import os
import uuid
import time
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse

from app.tracker import load_model, run_detection, run_tracking_on_video, benchmark
from app.schemas import DetectionResponse, TrackingResponse, BenchmarkResponse


# ── Lifespan: load model once at startup ──────────────────────────────────────
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print("Loading YOLOv11n model...")
    model = load_model()
    print("Model loaded.")
    yield
    print("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="YOLOv11 Detection & Tracking API",
    description=(
        "Real-time multi-object detection and tracking using YOLOv11 + ByteTrack. "
        "Accepts images for single-frame detection and videos for full tracking."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/avi", "video/quicktime", "video/x-msvideo"}
OUTPUT_DIR = Path(tempfile.gettempdir()) / "tracker_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["Utility"])
def health():
    return {"status": "ok", "model_loaded": model is not None}


# ── Single Image Detection ────────────────────────────────────────────────────
@app.post("/detect", response_model=DetectionResponse, tags=["Detection"])
async def detect_image(file: UploadFile = File(...)):
    """
    Upload an image (JPEG/PNG/WebP) and receive bounding boxes,
    class names, confidence scores, and inference time.
    """
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Use JPEG, PNG, or WebP."
        )

    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    frame = np.array(img)[:, :, ::-1]  # RGB → BGR for OpenCV

    detections, elapsed_ms = run_detection(model, frame)

    return DetectionResponse(
        filename=file.filename,
        image_width=img.width,
        image_height=img.height,
        detections=detections,
        inference_time_ms=elapsed_ms,
        model="yolo11n",
    )


# ── Video Tracking ────────────────────────────────────────────────────────────
@app.post("/track", response_model=TrackingResponse, tags=["Tracking"])
async def track_video(file: UploadFile = File(...), max_frames: int = 300):
    """
    Upload a video (MP4/AVI/MOV) and receive per-frame tracking results
    with persistent track IDs. Returns JSON results + path to annotated video.

    - max_frames: limit frames processed (default 300, ~10s at 30fps)
    """
    if file.content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Use MP4, AVI, or MOV."
        )

    # Save upload to temp file
    suffix = Path(file.filename).suffix or ".mp4"
    tmp_input = OUTPUT_DIR / f"{uuid.uuid4()}_input{suffix}"
    tmp_output = OUTPUT_DIR / f"{uuid.uuid4()}_tracked.mp4"

    contents = await file.read()
    tmp_input.write_bytes(contents)

    try:
        frame_results, avg_fps = run_tracking_on_video(
            model,
            str(tmp_input),
            output_path=str(tmp_output),
            max_frames=max_frames,
        )
    finally:
        tmp_input.unlink(missing_ok=True)

    avg_inf = (
        sum(f.inference_time_ms for f in frame_results) / len(frame_results)
        if frame_results else 0.0
    )

    return TrackingResponse(
        filename=file.filename,
        total_frames=max_frames,
        frames_processed=len(frame_results),
        fps_avg=avg_fps,
        avg_inference_time_ms=round(avg_inf, 2),
        output_video_path=str(tmp_output) if tmp_output.exists() else None,
        frame_results=frame_results,
        model="yolo11n",
    )


# ── Download annotated video ──────────────────────────────────────────────────
@app.get("/download/{filename}", tags=["Utility"])
def download_video(filename: str):
    """Download an annotated output video by filename."""
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(str(path), media_type="video/mp4", filename=filename)


# ── Benchmark ─────────────────────────────────────────────────────────────────
@app.get("/benchmark", response_model=BenchmarkResponse, tags=["Utility"])
def run_benchmark(n_frames: int = 100):
    """
    Run inference benchmark on random frames.
    Returns average latency, p95 latency, FPS, and RAM usage.
    """
    stats = benchmark(model, n_frames=n_frames)
    return BenchmarkResponse(**stats)