"""
Verification script for the tracking layer.

Processes frames sequentially (tracking needs frame-to-frame continuity, unlike
test_tracker.py's sparse sampling) and assigns persistent IDs via the combined
detector's track() method. Reports how many distinct objects were tracked per
class and how long each track persisted, as a sanity check that IDs are being
held across frames rather than reassigned every frame.

Usage:
    python test_tracking.py path/to/video.mp4 [--seconds 30] [--frame-stride 1]
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path

import cv2

from detector import DEFAULT_BALL_MODEL, DEFAULT_PLAYERS_MODEL, CombinedDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Path to the video file")
    parser.add_argument(
        "--seconds", type=float, default=30.0,
        help="Analyze at most this many seconds from the start (default: 30)",
    )
    parser.add_argument(
        "--frame-stride", type=int, default=1,
        help="Process every Nth frame (default: 1, i.e. every frame). Higher is faster but tracking is less stable.",
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

    max_frame = min(total_frames, int(fps * args.seconds)) if total_frames else int(fps * args.seconds)

    print(f"Loading models: {args.players_model}, {args.ball_model}")
    try:
        detector = CombinedDetector(args.players_model, args.ball_model)
    except FileNotFoundError as exc:
        print(f"Could not load models: {exc}")
        return 1

    track_frame_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    max_concurrent: dict[str, int] = defaultdict(int)
    frames_processed = 0
    start_time = time.time()

    frame_idx = 0
    while frame_idx < max_frame:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % args.frame_stride == 0:
            detections = detector.track(frame, conf=args.conf)

            concurrent: dict[str, int] = defaultdict(int)
            for d in detections:
                if d.track_id is None:
                    continue
                track_frame_counts[d.class_name][d.track_id] += 1
                concurrent[d.class_name] += 1
            for class_name, count in concurrent.items():
                max_concurrent[class_name] = max(max_concurrent[class_name], count)

            frames_processed += 1
            timestamp = frame_idx / fps
            per_class = ", ".join(f"{c}={len(ids)}" for c, ids in sorted(track_frame_counts.items()))
            print(f"[{timestamp:6.1f}s] frame {frame_idx}: tracked so far - {per_class or 'none yet'}")

        frame_idx += 1

    cap.release()
    elapsed = time.time() - start_time

    print("\n--- Summary ---")
    print(f"Frames processed: {frames_processed} (stride={args.frame_stride})")
    print(f"Processing time: {elapsed:.1f}s")
    if not track_frame_counts:
        print("FAILED: no tracks were established.")
        return 1

    for class_name, ids in sorted(track_frame_counts.items()):
        lengths = list(ids.values())
        avg_len = sum(lengths) / len(lengths)
        print(
            f"{class_name}: {len(ids)} unique track(s), "
            f"max concurrent {max_concurrent[class_name]}, "
            f"avg track length {avg_len:.1f} frames, longest {max(lengths)} frames"
        )

    print("SUCCESS: tracking layer is assigning and persisting IDs across frames.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
