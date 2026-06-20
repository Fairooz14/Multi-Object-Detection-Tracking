import argparse
import time
import numpy as np
import psutil
import torch
from ultralytics import YOLO


def run_benchmark(model_path: str, n_frames: int, imgsz: int = 640):
    model = YOLO(model_path)
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # Warm-up
    print("Warming up...")
    for _ in range(10):
        model.predict(dummy, verbose=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    process = psutil.Process()
    times = []

    print(f"Benchmarking {n_frames} frames on {device}...")
    for i in range(n_frames):
        start = time.perf_counter()
        model.predict(dummy, verbose=False)
        times.append((time.perf_counter() - start) * 1000)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{n_frames}] avg so far: {np.mean(times):.1f} ms")

    times = np.array(times)
    print("\n── Benchmark Results ───────────────────────────────")
    print(f"  Model         : {model_path}")
    print(f"  Device        : {device}")
    print(f"  Frames        : {n_frames}")
    print(f"  Image size    : {imgsz}x{imgsz}")
    print(f"  Avg latency   : {np.mean(times):.2f} ms")
    print(f"  P50 latency   : {np.percentile(times, 50):.2f} ms")
    print(f"  P95 latency   : {np.percentile(times, 95):.2f} ms")
    print(f"  P99 latency   : {np.percentile(times, 99):.2f} ms")
    print(f"  Avg FPS       : {1000/np.mean(times):.1f}")
    print(f"  Min/Max (ms)  : {times.min():.2f} / {times.max():.2f}")
    print("────────────────────────────────────────────────────")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv11 Benchmark")
    parser.add_argument("--model", default="yolo11n.pt", help="Model path or name")
    parser.add_argument("--frames", type=int, default=100, help="Number of frames to benchmark")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    args = parser.parse_args()
    run_benchmark(args.model, args.frames, args.imgsz)