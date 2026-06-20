from pydantic import BaseModel
from typing import List, Optional


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    track_id: Optional[int] = None
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox


class FrameResult(BaseModel):
    frame_index: int
    detections: List[Detection]
    inference_time_ms: float


class DetectionResponse(BaseModel):
    filename: str
    image_width: int
    image_height: int
    detections: List[Detection]
    inference_time_ms: float
    model: str


class TrackingResponse(BaseModel):
    filename: str
    total_frames: int
    frames_processed: int
    fps_avg: float
    avg_inference_time_ms: float
    output_video_path: Optional[str]
    frame_results: List[FrameResult]
    model: str


class BenchmarkResponse(BaseModel):
    model: str
    device: str
    total_frames: int
    avg_inference_time_ms: float
    p95_inference_time_ms: float
    avg_fps: float
    ram_used_mb: float