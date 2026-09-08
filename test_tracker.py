"""
Verification script for the soccer analytics CV layer.

Analyzes the first N minutes of a video: samples frames at a fixed interval,
runs the combined detector (player/goalkeeper/referee + specialist ball model)
on each sample, and prints per-sample and summary detection stats.

Usage:
    python test_tracker.py path/to/video.mp4 [--minutes 5] [--sample-seconds 1]
"""

import argparse
import time
from collections import Counter
from pathlib import Path

import cv2

from detector import DEFAULT_BALL_MODEL, DEFAULT_PLAYERS_MODEL, CombinedDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Path to the video file")
    parser.add_argument(
        "--minutes", type=float, default=5.0,
        help="Analyze at most this many minutes from the start (default: 5)",
    )
    parser.add_argument(
        "--sample-seconds", type=float, default=1.0,
        help="Seconds between sampled frames (default: 1.0)",
    )
    parser.add_argument(
        "--players-model", default=str(DEFAULT_PLAYERS_MODEL),
        help="Path to the player/goalkeeper/referee/ball model",
    )
    parser.add_argument(
        "--ball-model", default=str(DEFAULT_BALL_MODEL),
        help="Path to the ball-specialist model",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold for both models (default: 0.25)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.video.exists():
        print(f"Video file not found: {args.video}")
        return 1

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"Failed to open video: {args.video}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_min = (total_frames / fps / 60) if total_frames else 0
    print(f"Video loaded OK: {args.video.name} ({total_frames} frames @ {fps:.1f} fps, {duration_min:.1f} min)")

    frame_step = max(int(round(fps * args.sample_seconds)), 1)
    window_frames = int(fps * args.minutes * 60)
    max_frame = min(total_frames, window_frames) if total_frames else window_frames

    print(f"Loading models: {args.players_model}, {args.ball_model}")
    try:
        detector = CombinedDetector(args.players_model, args.ball_model)
    except FileNotFoundError as exc:
        print(f"Could not load models: {exc}")
        return 1

    class_counts: Counter = Counter()
    samples_analyzed = 0
    total_detections = 0
    start_time = time.time()

    frame_idx = 0
    while frame_idx < max_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break

        detections = detector.detect(frame, conf=args.conf)
        detected_classes = [d.class_name for d in detections]

        samples_analyzed += 1
        total_detections += len(detected_classes)
        class_counts.update(detected_classes)

        timestamp = frame_idx / fps
        preview = ", ".join(detected_classes[:5]) or "none"
        print(f"[{timestamp:6.1f}s] frame {frame_idx}: {len(detected_classes)} object(s) - {preview}")

        frame_idx += frame_step

    cap.release()
    elapsed = time.time() - start_time

    print("\n--- Summary ---")
    print(f"Samples analyzed: {samples_analyzed}")
    print(f"Total detections: {total_detections}")
    if samples_analyzed:
        print(f"Avg detections/sample: {total_detections / samples_analyzed:.2f}")
    if class_counts:
        print("Detections by class:")
        for cls_name, count in class_counts.most_common():
            print(f"  {cls_name}: {count}")
    print(f"Processing time: {elapsed:.1f}s")

    if samples_analyzed == 0:
        print("FAILED: no frames analyzed.")
        return 1

    print("SUCCESS: video loading and combined detection pipeline are working.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
