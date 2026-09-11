"""
Extracts a short, padded video clip around each event's frame range from the
source footage, for review in event_log_tool.html - a raw timestamp plus a
detector call ("OFF TARGET, peak speed 97px/frame") isn't enough to tell what
actually happened when the underlying ball-tracking segment is a handful of
frames (often well under a second); a clip you can watch is.

Usage:
    python extract_event_clips.py --video "/path/to/video.mp4" \\
        --event-log shot_log_full_game.json \\
        --events events_to_extract.json \\
        --out-dir clips/ \\
        --pad-seconds 2.5

--events: a JSON list of {"start_frame": N} (subset of the fields
train_action_classifier.py's label export uses - only start_frame is read
here). Omit to extract every event in --event-log.

Requires ffmpeg (already an apt dependency per SETUP.md).
"""

import argparse
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--event-log", required=True, type=Path)
    parser.add_argument("--events", type=Path, help="JSON list of {start_frame: N} to limit extraction to; omit for all events in --event-log")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--pad-seconds", type=float, default=2.5, help="Seconds of context to include before/after the detected segment (default: 2.5)")
    parser.add_argument("--scale-width", type=int, default=960, help="Downscale clip to this width, preserving aspect ratio (default: 960)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = json.loads(args.event_log.read_text())
    fps = data["scanned_window"]["fps"]
    events = data["events"]

    wanted_frames = None
    if args.events:
        wanted_frames = {w["start_frame"] for w in json.loads(args.events.read_text())}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    extracted = []
    for event in events:
        start_frame = event["start_frame"]
        if wanted_frames is not None and start_frame not in wanted_frames:
            continue

        end_frame = event["end_frame"]
        start_time = max(0.0, start_frame / fps - args.pad_seconds)
        duration = (end_frame - start_frame) / fps + 2 * args.pad_seconds
        out_path = args.out_dir / f"clip_f{start_frame}.mp4"

        cmd = [
            "ffmpeg", "-y", "-ss", f"{start_time:.3f}", "-i", str(args.video),
            "-t", f"{duration:.3f}",
            "-vf", f"scale={args.scale_width}:-2",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-an", str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FAILED start_frame={start_frame}: {result.stderr[-500:]}")
            continue

        size_kb = out_path.stat().st_size / 1024
        extracted.append({"start_frame": start_frame, "path": str(out_path), "size_kb": round(size_kb, 1)})
        print(f"extracted start_frame={start_frame} -> {out_path} ({size_kb:.0f} KB, {duration:.1f}s)")

    print(f"\nExtracted {len(extracted)} clip(s) into {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
