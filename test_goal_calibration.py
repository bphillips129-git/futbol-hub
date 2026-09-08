"""
Verification script for goal-line calibration.

1. Draws the calibrated goal-mouth polygon onto the exact frame it was
   marked on, saved as an image, so alignment can be checked by eye.
2. Runs synthetic point-in-polygon sanity checks.
3. Runs the ball-specialist model on a window of real footage around the
   calibrated timestamp and reports whether any detected ball position
   falls inside the goal mouth (a real end-to-end check, though a 30-60s
   window may simply not contain a shot near the goal - that's a valid
   "no event" outcome, not a failure).

Usage:
    python test_goal_calibration.py [--window-seconds 30]
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from detector import DEFAULT_BALL_MODEL, CombinedDetector
from goal_calibration import GoalCalibration, find_mouth_entry_frame, is_in_goal_mouth

SCRIPT_DIR = Path(__file__).resolve().parent
ANNOTATED_OUTPUT = SCRIPT_DIR / "calibration_frames" / "goal_calibration_check.jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "video", type=Path, nargs="?", default=None,
        help="Path to the source video (defaults to the calibration's recorded filename, "
             "which may not resolve from every environment - pass it explicitly if so)",
    )
    parser.add_argument(
        "--window-seconds", type=float, default=30.0,
        help="Seconds of real footage to scan around the calibrated timestamp (default: 30)",
    )
    parser.add_argument("--ball-model", default=str(DEFAULT_BALL_MODEL))
    parser.add_argument("--conf", type=float, default=0.25)
    return parser.parse_args()


def draw_check_image(calibration: GoalCalibration, source_video: Path) -> None:
    cap = cv2.VideoCapture(str(source_video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(calibration.frame_timestamp_seconds * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"Could not re-read the calibration frame from {source_video}; skipping visual check image.")
        return

    frame = cv2.resize(frame, (calibration.image_width, calibration.image_height))
    polygon = np.array([(int(x), int(y)) for x, y in calibration.mouth_polygon], dtype=int)
    cv2.polylines(frame, [polygon], isClosed=True, color=(69, 122, 255), thickness=2)
    lp1, lp2 = calibration.goal_line
    cv2.line(frame, (int(lp1[0]), int(lp1[1])), (int(lp2[0]), int(lp2[1])), (69, 122, 255), 4)

    ANNOTATED_OUTPUT.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(ANNOTATED_OUTPUT), frame)
    print(f"Wrote visual check image: {ANNOTATED_OUTPUT}")


def run_synthetic_checks(calibration: GoalCalibration) -> None:
    center_x = (calibration.left_post_base[0] + calibration.right_post_base[0]) / 2
    center_y = (calibration.left_post_top[1] + calibration.left_post_base[1]) / 2
    cases = [
        ("center of goal mouth", (center_x, center_y), True),
        ("far outside (image corner)", (5, 5), False),
        ("below the goal line (in front of goal)", (center_x, calibration.left_post_base[1] + 40), False),
    ]
    print("Synthetic point-in-polygon checks:")
    all_ok = True
    for name, point, expected in cases:
        actual = is_in_goal_mouth(point, calibration)
        ok = actual == expected
        all_ok &= ok
        print(f"  [{'OK' if ok else 'FAIL'}] {name}: expected {expected}, got {actual}")
    return all_ok


def main() -> int:
    args = parse_args()

    calibration = GoalCalibration.load()
    print(f"Loaded calibration for {calibration.video} @ {calibration.frame_timestamp_seconds}s")

    video_path = args.video if args.video is not None else Path(calibration.video)
    if video_path.exists():
        draw_check_image(calibration, video_path)
    else:
        print(f"Source video not found at {video_path}; skipping visual check image.")

    synthetic_ok = run_synthetic_checks(calibration)

    if not video_path.exists():
        print("\nSkipping live ball-detection check (source video unavailable).")
        return 0 if synthetic_ok else 1

    print(f"\nScanning {args.window_seconds:.0f}s of real footage around the calibrated timestamp for ball positions...")
    detector = CombinedDetector(ball_model_path=args.ball_model)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = max(0, int((calibration.frame_timestamp_seconds - args.window_seconds / 2) * fps))
    end_frame = start_frame + int(args.window_seconds * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ball_positions = []
    frame_idx = start_frame
    while frame_idx < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        detections = detector.detect(frame, conf=args.conf)
        for d in detections:
            if d.class_name == "ball":
                x1, y1, x2, y2 = d.box
                center = ((x1 + x2) / 2 * calibration.image_width / frame.shape[1],
                          (y1 + y2) / 2 * calibration.image_height / frame.shape[0])
                ball_positions.append((frame_idx, center))
        frame_idx += 1
    cap.release()

    print(f"Ball detected in {len(ball_positions)} of {end_frame - start_frame} frames scanned.")
    entry_frame = find_mouth_entry_frame(ball_positions, calibration)
    if entry_frame is not None:
        print(f"Ball entered the goal-mouth region at frame {entry_frame} ({entry_frame / fps:.1f}s).")
    else:
        print("Ball never entered the goal-mouth region in this window (a valid outcome, not a failure).")

    print("\nSUCCESS: goal calibration geometry ran end to end." if synthetic_ok else "\nFAILED: synthetic checks did not all pass.")
    return 0 if synthetic_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
