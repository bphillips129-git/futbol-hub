"""
Verification script for jersey number OCR combined with tracking.

Tracks players frame-by-frame (as in test_tracking.py) and, every --ocr-every
frames, runs jersey OCR on each currently-tracked player's box. Reads are
accumulated per track_id and majority-voted at the end, so a single blurry
misread doesn't dominate - this also doubles as a practical check on how
fragmented tracking IDs are, since a real player should keep one track_id
with a stable number long enough to accumulate several matching reads.

Usage:
    python test_jersey_ocr.py path/to/video.mp4 [--seconds 20] [--ocr-every 10]
"""

import argparse
import time
from pathlib import Path

import cv2

from detector import DEFAULT_BALL_MODEL, DEFAULT_PLAYERS_MODEL, CombinedDetector
from jersey_ocr import JerseyOCR, JerseyVoteTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Path to the video file")
    parser.add_argument(
        "--seconds", type=float, default=20.0,
        help="Analyze at most this many seconds from the start (default: 20)",
    )
    parser.add_argument(
        "--ocr-every", type=int, default=10,
        help="Run OCR every Nth tracked frame (default: 10). OCR is expensive; tracking still runs every frame.",
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
        help="Confidence threshold for both detection models (default: 0.25)",
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

    print(f"Loading detection models: {args.players_model}, {args.ball_model}")
    try:
        detector = CombinedDetector(args.players_model, args.ball_model)
    except FileNotFoundError as exc:
        print(f"Could not load models: {exc}")
        return 1

    print("Loading jersey OCR model (first run downloads EasyOCR weights)...")
    ocr = JerseyOCR()
    votes = JerseyVoteTracker()

    frames_processed = 0
    ocr_calls = 0
    ocr_reads = 0
    start_time = time.time()

    frame_idx = 0
    while frame_idx < max_frame:
        ok, frame = cap.read()
        if not ok:
            break

        detections = detector.track(frame, conf=args.conf)
        frames_processed += 1

        if frame_idx % args.ocr_every == 0:
            for d in detections:
                if d.class_name != "player" or d.track_id is None:
                    continue
                ocr_calls += 1
                number, confidence = ocr.read_number(frame, d.box)
                if number is not None:
                    ocr_reads += 1
                    votes.add(d.track_id, number)
            timestamp = frame_idx / fps
            print(f"[{timestamp:6.1f}s] frame {frame_idx}: OCR pass - {ocr_reads}/{ocr_calls} reads so far")

        frame_idx += 1

    cap.release()
    elapsed = time.time() - start_time

    print("\n--- Summary ---")
    print(f"Frames tracked: {frames_processed}, OCR attempts: {ocr_calls}, successful reads: {ocr_reads}")
    print(f"Processing time: {elapsed:.1f}s")

    track_ids = sorted(votes.all_track_ids(), key=lambda t: -votes.best_guess(t)[2])
    if not track_ids:
        print("No jersey numbers were read. This can happen on short/distant clips - try more seconds or a closer camera angle.")
        return 0

    print("\nTrack ID -> best-guess jersey number (votes/total reads for that track):")
    for track_id in track_ids:
        number, count, total = votes.best_guess(track_id)
        print(f"  {track_id}: #{number}  ({count}/{total})")

    print("\nSUCCESS: jersey OCR is reading digits and accumulating votes per track.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
