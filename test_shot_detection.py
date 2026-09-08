"""
Verification script for shot-on-goal detection, outcome classification, and
player attribution.

Tracks the ball, players, and goalkeeper over a window of real footage, reads
jersey numbers along the way, converts positions into the calibration's
coordinate space, and runs detect_shots(). Defaults to the same window used
to validate goal_calibration.py, which is known to contain a real
ball-reaches-goal-mouth moment around 293s - so a working detector should
surface that as a shot event here.

Usage:
    python test_shot_detection.py path/to/video.mp4 [--center-seconds 300] [--window-seconds 30] [--ocr-every 10]
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2

from detector import CombinedDetector
from goal_calibration import GoalCalibration
from jersey_ocr import JerseyOCR, JerseyVoteTracker
from shot_detector import detect_shots

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_PATH = SCRIPT_DIR / "shot_log.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Path to the video file")
    parser.add_argument("--center-seconds", type=float, default=300.0, help="Midpoint of the scan window (default: 300, the calibration timestamp)")
    parser.add_argument("--window-seconds", type=float, default=30.0, help="Total seconds to scan (default: 30)")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--ocr-every", type=int, default=10, help="Run jersey OCR every Nth frame (default: 10)")
    parser.add_argument("--show-raw-speeds", action="store_true", help="Print every inter-frame ball speed for threshold tuning")
    parser.add_argument("--output", type=Path, default=DEFAULT_LOG_PATH, help=f"Where to write the shot event log (default: {DEFAULT_LOG_PATH.name}, overwritten each run)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.video.exists():
        print(f"Video file not found: {args.video}")
        return 1

    calibration = GoalCalibration.load()
    print(f"Loaded calibration for {calibration.video} @ {calibration.frame_timestamp_seconds}s")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"Failed to open video: {args.video}")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    start_frame = max(0, int((args.center_seconds - args.window_seconds / 2) * fps))
    end_frame = start_frame + int(args.window_seconds * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    detector = CombinedDetector()
    ocr = JerseyOCR()
    votes = JerseyVoteTracker()

    ball_positions = []
    goalkeeper_positions = []
    player_positions = []
    frame_idx = start_frame
    while frame_idx < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        sx = calibration.image_width / frame.shape[1]
        sy = calibration.image_height / frame.shape[0]

        detections = detector.track(frame, conf=args.conf)
        for d in detections:
            x1, y1, x2, y2 = d.box
            center = ((x1 + x2) / 2 * sx, (y1 + y2) / 2 * sy)
            if d.class_name == "ball":
                ball_positions.append((frame_idx, center))
            elif d.class_name == "goalkeeper" and d.track_id is not None:
                goalkeeper_positions.append((frame_idx, d.track_id, center))
            elif d.class_name == "player" and d.track_id is not None:
                player_positions.append((frame_idx, d.track_id, center))

        if frame_idx % args.ocr_every == 0:
            for d in detections:
                if d.class_name not in ("player", "goalkeeper") or d.track_id is None:
                    continue
                number, _ = ocr.read_number(frame, d.box)
                if number is not None:
                    votes.add(d.track_id, number)

        frame_idx += 1
    cap.release()

    jersey_numbers = {}
    for track_id in votes.all_track_ids():
        number, vote_count, total = votes.best_guess(track_id)
        if number is not None and total >= 2 and vote_count / total >= 0.5:
            jersey_numbers[track_id] = number

    print(f"Ball detected in {len(ball_positions)} frames; goalkeeper in {len(goalkeeper_positions)}; player in {len(player_positions)}.")
    print(f"Confidently read {len(jersey_numbers)} jersey number(s): {jersey_numbers}")

    if args.show_raw_speeds and len(ball_positions) > 1:
        print("\nRaw inter-detection ball speeds (px/frame in calibration space):")
        for i in range(1, len(ball_positions)):
            (f0, p0), (f1, p1) = ball_positions[i - 1], ball_positions[i]
            gap = f1 - f0
            speed = ((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2) ** 0.5 / max(gap, 1)
            print(f"  frame {f0}->{f1} (gap {gap}): speed {speed:.1f} px/frame")

    events = detect_shots(
        ball_positions, calibration,
        goalkeeper_positions=goalkeeper_positions,
        player_positions=player_positions,
        jersey_numbers=jersey_numbers,
    )

    print(f"\n--- Shot events detected: {len(events)} ---")
    for e in events:
        if e.on_target:
            entry_sec = e.entry_frame / fps
            print(
                f"  ON TARGET: frames {e.start_frame}-{e.end_frame}, entered goal mouth at frame {e.entry_frame} ({entry_sec:.1f}s), "
                f"peak speed {e.peak_speed:.1f} px/frame"
            )
            print(f"    outcome: {e.outcome} (confidence: {e.outcome_confidence})")
            taker = f"#{e.shot_taker_jersey_number}" if e.shot_taker_jersey_number else e.shot_taker_track_id or "unidentified"
            keeper = f"#{e.goalkeeper_jersey_number}" if e.goalkeeper_jersey_number else e.goalkeeper_track_id or "unidentified"
            print(f"    shot taker: {taker}, goalkeeper faced: {keeper}")
        else:
            print(
                f"  OFF TARGET: frames {e.start_frame}-{e.end_frame}, peak speed {e.peak_speed:.1f} px/frame, "
                f"closest approach {e.closest_approach_px:.1f}px from goal line"
            )

    if not events:
        print("  (none - try --show-raw-speeds to see the actual ball speeds and tune thresholds)")

    log = {
        "video": str(args.video),
        "calibration_video": calibration.video,
        "calibration_frame_timestamp_seconds": calibration.frame_timestamp_seconds,
        "scanned_window": {"start_frame": start_frame, "end_frame": end_frame, "fps": fps},
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Heuristic candidate events from a single 2D camera calibration - "
            "not a certain shot/goal/save determination. See shot_detector.py and goal_calibration.py."
        ),
        "events": [
            {
                "on_target": e.on_target,
                "start_frame": e.start_frame,
                "end_frame": e.end_frame,
                "peak_speed_px_per_frame": round(e.peak_speed, 2),
                "entry_frame": e.entry_frame,
                "entry_time_seconds": round(e.entry_frame / fps, 2) if e.entry_frame is not None else None,
                "closest_approach_px": round(e.closest_approach_px, 1) if e.closest_approach_px is not None else None,
                "outcome": e.outcome,
                "outcome_confidence": e.outcome_confidence,
                "shot_taker_track_id": e.shot_taker_track_id,
                "shot_taker_jersey_number": e.shot_taker_jersey_number,
                "goalkeeper_track_id": e.goalkeeper_track_id,
                "goalkeeper_jersey_number": e.goalkeeper_jersey_number,
                "trajectory": [[f, round(x, 1), round(y, 1)] for f, (x, y) in e.trajectory],
            }
            for e in events
        ],
    }
    args.output.write_text(json.dumps(log, indent=2))
    print(f"\nWrote event log: {args.output} ({len(events)} event(s))")

    print("\nSUCCESS: shot detection pipeline ran end to end." if ball_positions else "FAILED: no ball detections in this window.")
    return 0 if ball_positions else 1


if __name__ == "__main__":
    raise SystemExit(main())
