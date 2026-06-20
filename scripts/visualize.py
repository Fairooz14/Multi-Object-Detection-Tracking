import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO
from app.tracker import run_tracking_on_video


def main():
    parser = argparse.ArgumentParser(description="Visualize tracking on a video")
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output", default="output_tracked.mp4", help="Path to save annotated video")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO model path")
    parser.add_argument("--max-frames", type=int, default=500, help="Max frames to process")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    print(f"Processing: {args.input}")
    frame_results, avg_fps = run_tracking_on_video(
        model,
        args.input,
        output_path=args.output,
        max_frames=args.max_frames,
    )

    total_detections = sum(len(f.detections) for f in frame_results)
    print(f"\nDone.")
    print(f"  Frames processed : {len(frame_results)}")
    print(f"  Total detections : {total_detections}")
    print(f"  Average FPS      : {avg_fps:.1f}")
    print(f"  Output saved to  : {args.output}")


if __name__ == "__main__":
    main()