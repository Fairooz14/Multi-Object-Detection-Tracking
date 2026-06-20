# Real-Time Multi-Object Detection & Tracking with FastAPI

A production-ready computer vision pipeline that performs **real-time multi-object detection and tracking** on video streams using YOLOv11 + ByteTrack, served via a **FastAPI REST inference API**.

Built to demonstrate enterprise-grade CV engineering: model training, tracking pipeline, API deployment, and benchmarking.

---

## What This Does

- Detects and tracks multiple objects across video frames with persistent IDs
- Runs YOLOv11n for fast, accurate detection
- Uses ByteTrack for robust multi-object tracking (handles occlusion well)
- Exposes a FastAPI endpoint that accepts a video or image and returns annotated output + JSON results
- Benchmarks FPS and latency per frame

---

## Project Structure

```
project1_tracking/
├── app/
│   ├── main.py            # FastAPI app
│   ├── tracker.py         # ByteTrack wrapper + YOLO inference
│   └── schemas.py         # Pydantic response models
├── notebooks/
│   └── train_and_export.ipynb   # Colab notebook: fine-tune + export
├── scripts/
│   ├── benchmark.py       # FPS & latency benchmarking script
│   └── visualize.py       # Draw bounding boxes + track IDs on video
├── models/                # Place yolov11n.pt or custom weights here
├── sample_data/           # Sample images/video for testing
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place model weights
Download YOLOv11n weights (auto-downloads on first run) or place custom `.pt` file in `models/`.

### 3. Run the API
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Test with an image
```bash
curl -X POST "http://localhost:8000/detect" \
  -F "file=@sample_data/sample.jpg"
```

### 5. Test with a video
```bash
curl -X POST "http://localhost:8000/track" \
  -F "file=@sample_data/sample.mp4"
```

### 6. Open interactive docs
Visit: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/detect` | Single image detection — returns boxes, classes, confidence |
| POST | `/track` | Video tracking — returns per-frame track IDs + annotated video |
| GET | `/health` | Health check |
| GET | `/benchmark` | Run benchmark on sample and return FPS / latency stats |

---

## Benchmark Results (YOLOv11n, CPU)

| Metric | Value |
|--------|-------|
| Avg inference time | ~38ms/frame |
| Throughput | ~26 FPS |
| mAP@0.5 (COCO) | 0.529 |
| Model size | 5.4 MB |

> GPU results are significantly faster. See `scripts/benchmark.py` for reproduction.

---

## Training on Custom Data (Colab)

Open `notebooks/train_and_export.ipynb` in Google Colab. It covers:
1. Dataset setup (supports COCO format or custom YOLO format)
2. Fine-tuning YOLOv11n on your data
3. Evaluating mAP, precision, recall
4. Exporting to ONNX (feeds into Project 2)

---

## Docker

```bash
docker build -t yolo-tracker .
docker run -p 8000:8000 yolo-tracker
```

---

## Tech Stack

- **Detection**: YOLOv11 (Ultralytics)
- **Tracking**: ByteTrack (via Ultralytics built-in)
- **API**: FastAPI + Uvicorn
- **Validation**: Pydantic
- **Containerization**: Docker
- **Benchmarking**: time, psutil, OpenCV

---

## Relevance to Enterprise CV

This project demonstrates:
- End-to-end CV pipeline from raw video to structured JSON output
- REST API design for AI model serving
- Multi-object tracking with persistent IDs (surveillance / FMCG use cases)
- Inference benchmarking for production readiness
