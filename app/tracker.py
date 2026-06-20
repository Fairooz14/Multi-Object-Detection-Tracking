import time
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from ultralytics import YOLO

from app.schemas import Detection, BoundingBox, FrameResult


MODEL_PATH = Path("models/yolov11n.pt")  
TRACKER_CONFIG = "bytetrack.yaml"          


def load_model(model_path: Optional[str] = None) -> YOLO:
    """Load YOLOv11 model. Auto-downloads yolov11n if no custom path given."""
    path = model_path or str(MODEL_PATH) if MODEL_PATH.exists() else "yolo11n.pt"
    model = YOLO(path)
    return model


def run_detection(model: YOLO, frame: np.ndarray) -> Tuple[List[Detection], float]:
    """
    Run single-frame detection (no tracking).
    Returns list of Detection objects and inference time in ms.
    """
    start = time.perf_counter()
    results = model.predict(frame, verbose=False, conf=0.35)
    elapsed_ms = (time.perf_counter() - start) * 1000

    detections: List[Detection] = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        names = result.names
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            detections.append(Detection(
                class_id=cls_id,
                class_name=names[cls_id],
                confidence=round(conf, 4),
                bbox=BoundingBox(x1=round(x1, 1), y1=round(y1, 1),
                                 x2=round(x2, 1), y2=round(y2, 1))
            ))

    return detections, round(elapsed_ms, 2)


def run_tracking_on_video(
    model: YOLO,
    video_path: str,
    output_path: Optional[str] = None,
    max_frames: int = 300,
) -> Tuple[List[FrameResult], float]:
    """
    Run ByteTrack multi-object tracking on a video file.
    Annotates frames with track IDs and bounding boxes.
    Returns list of FrameResult and average FPS.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps_src, (width, height))

    frame_results: List[FrameResult] = []
    inference_times: List[float] = []
    frame_idx = 0

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.perf_counter()
        results = model.track(frame, persist=True, tracker=TRACKER_CONFIG,
                              verbose=False, conf=0.35)
        elapsed_ms = (time.perf_counter() - start) * 1000
        inference_times.append(elapsed_ms)

        detections: List[Detection] = []
        annotated = frame.copy()

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            names = result.names
            for box in boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                track_id = int(box.id[0]) if box.id is not None else None

                detections.append(Detection(
                    track_id=track_id,
                    class_id=cls_id,
                    class_name=names[cls_id],
                    confidence=round(conf, 4),
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
                ))

                # Draw on frame
                color = _id_to_color(track_id or cls_id)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"#{track_id} {names[cls_id]} {conf:.2f}" if track_id else \
                        f"{names[cls_id]} {conf:.2f}"
                cv2.putText(annotated, label, (x1, max(y1 - 8, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Overlay FPS
        fps_display = 1000 / elapsed_ms if elapsed_ms > 0 else 0
        cv2.putText(annotated, f"FPS: {fps_display:.1f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if writer:
            writer.write(annotated)

        frame_results.append(FrameResult(
            frame_index=frame_idx,
            detections=detections,
            inference_time_ms=round(elapsed_ms, 2)
        ))
        frame_idx += 1

    cap.release()
    if writer:
        writer.release()

    avg_fps = 1000 / np.mean(inference_times) if inference_times else 0.0
    return frame_results, round(avg_fps, 2)


def benchmark(model: YOLO, n_frames: int = 100) -> dict:
    """
    Benchmark inference on random frames to measure latency and FPS.
    Returns dict of stats compatible with BenchmarkResponse.
    """
    import psutil, torch
    dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    times: List[float] = []

    # Warm up
    for _ in range(5):
        model.predict(dummy, verbose=False)

    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024

    for _ in range(n_frames):
        start = time.perf_counter()
        model.predict(dummy, verbose=False)
        times.append((time.perf_counter() - start) * 1000)

    mem_after = process.memory_info().rss / 1024 / 1024
    device = "cuda" if torch.cuda.is_available() else "cpu"

    return {
        "model": model.model_name if hasattr(model, "model_name") else "yolo11n",
        "device": device,
        "total_frames": n_frames,
        "avg_inference_time_ms": round(float(np.mean(times)), 2),
        "p95_inference_time_ms": round(float(np.percentile(times, 95)), 2),
        "avg_fps": round(1000 / float(np.mean(times)), 2),
        "ram_used_mb": round(mem_after - mem_before, 2),
    }


def _id_to_color(track_id: int) -> Tuple[int, int, int]:
    """Map track ID to a consistent BGR color."""
    np.random.seed(track_id % 256)
    return tuple(int(c) for c in np.random.randint(80, 255, 3))