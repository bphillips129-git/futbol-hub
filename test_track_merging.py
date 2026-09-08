"""
Verification script for fragment merging.

Runs tracking + jersey OCR exactly like test_jersey_ocr.py, but also records
each track_id's first/last frame and box, then merges fragments that share a
confident jersey number and are temporally/spatially plausible as the same
player (see track_merger.py). Reports the raw track_id count vs the merged
persistent-identity count, and shows exactly which track_ids got merged into
each identity, so the merge decisions are auditable.

Usage:
    python test_track_merging.py path/to/video.mp4 [--seconds 20] [--ocr-every 10]
"""

import argparse
import time
from pathlib import Path

import cv2

from detector import DEFAULT_BALL_MODEL, DEFAULT_PLAYERS_MODEL, CombinedDetector
from jersey_ocr import JerseyOCR, JerseyVoteTracker
from track_merger import TrackSummary, merge_fragments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Path to the video file")
    parser.add_argument("--seconds", type=float, default=20.0, help="Analyze at most this many seconds (default: 20)")
    parser.add_argument("--ocr-every", type=int, default=10, help="Run OCR every Nth tracked frame (default: 10)")
    parser.add_argument("--players-model", default=str(DEFAULT_PLAYERS_MODEL))
    parser.add_argument("--ball-model", default=str(DEFAULT_BALL_MODEL))
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold (default: 0.25)")
    parser.add_argument("--max-gap-frames", type=int, default=90, help="Max frame gap to link two fragments (default: 90)")
    parser.add_argument("--speed-factor", type=float, default=1.5, help="Motion budget in box-diagonals/frame (default: 1.5)")
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
    print(f"Video loaded OK: {args.video.name} ({total_frames} frames @ {fps:.1f} fps)")

    max_frame = min(total_frames, int(fps * args.seconds)) if total_frames else int(fps * args.seconds)

    print(f"Loading detection models: {args.players_model}, {args.ball_model}")
    try:
        detector = CombinedDetector(args.players_model, args.ball_model)
    except FileNotFoundError as exc:
        print(f"Could not load models: {exc}")
        return 1

    print("Loading jersey OCR model...")
    ocr = JerseyOCR()
    votes = JerseyVoteTracker()
    track_extent: dict[str, TrackSummary] = {}

    start_time = time.time()
    frame_idx = 0
    while frame_idx < max_frame:
        ok, frame = cap.read()
        if not ok:
            break

        detections = detector.track(frame, conf=args.conf)

        for d in detections:
            if d.class_name != "player" or d.track_id is None:
                continue
            if d.track_id not in track_extent:
                track_extent[d.track_id] = TrackSummary(
                    track_id=d.track_id, class_name=d.class_name,
                    first_frame=frame_idx, last_frame=frame_idx,
                    first_box=d.box, last_box=d.box,
                )
            else:
                summary = track_extent[d.track_id]
                summary.last_frame = frame_idx
                summary.last_box = d.box

        if frame_idx % args.ocr_every == 0:
            for d in detections:
                if d.class_name != "player" or d.track_id is None:
                    continue
                number, confidence = ocr.read_number(frame, d.box)
                if number is not None:
                    votes.add(d.track_id, number)

        frame_idx += 1

    cap.release()
    elapsed = time.time() - start_time

    for track_id, summary in track_extent.items():
        number, vote_count, total = votes.best_guess(track_id)
        summary.number = number
        summary.number_votes = vote_count
        summary.number_total = total

    summaries = list(track_extent.values())
    merged = merge_fragments(
        summaries,
        max_gap_frames=args.max_gap_frames,
        speed_factor=args.speed_factor,
    )

    confident = [
        s for s in summaries
        if s.number is not None and s.number_total >= 2 and s.number_votes / s.number_total >= 0.5
    ]
    print(f"\nConfident tracks considered for merging ({len(confident)}):")
    for s in sorted(confident, key=lambda s: s.first_frame):
        print(
            f"  {s.track_id}: #{s.number} ({s.number_votes}/{s.number_total}), "
            f"frames {s.first_frame}-{s.last_frame}, "
            f"last_box={tuple(round(v) for v in s.last_box)}"
        )

    print(f"\nProcessing time: {elapsed:.1f}s")
    print(f"\nRaw track_ids: {len(summaries)}")
    print(f"Merged identities: {len(merged)}")

    real_merges = {pid: m for pid, m in merged.items() if len(m["track_ids"]) > 1}
    print(f"Fragments actually merged: {len(real_merges)} identity(ies) formed from multiple track_ids\n")

    for persistent_id, m in sorted(real_merges.items(), key=lambda kv: -len(kv[1]["track_ids"])):
        span_frames = m["last_frame"] - m["first_frame"]
        print(
            f"  {persistent_id}: jersey #{m['number']}, "
            f"{len(m['track_ids'])} fragments merged -> {m['track_ids']}, "
            f"spans frames {m['first_frame']}-{m['last_frame']} ({span_frames} frames)"
        )

    if not real_merges:
        print("  (none - try a longer --seconds window so fragments have time to reappear)")

    print("\nSUCCESS: fragment merging pipeline ran end to end." if summaries else "FAILED: no player tracks were found.")
    return 0 if summaries else 1


if __name__ == "__main__":
    raise SystemExit(main())
