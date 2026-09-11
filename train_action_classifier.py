"""
Trains the action-type classifier (action_classifier.py) from human-reviewed
labels plus the matching ball trajectories.

Labels come from event_log_tool.html's review data (pipeline-sourced events
someone has reviewed and picked an ACTION_CATEGORIES value for). This script
has no way to reach that Artifact's database directly, so labels are
supplied as a local JSON export - ask whoever has Artifact tool access to
pull one from the "events" collection.

Trajectories come from one or more shot_log*.json files (the --output of
test_shot_detection.py), matched to a label by video filename + start_frame.

Usage:
    python train_action_classifier.py --labels labeled_events_export.json \\
        --event-logs shot_log_full_game.json shot_log.json \\
        --model-out models/action_classifier.joblib

Labels export format (a JSON list):
    [{"video": "WIVY3619.MP4", "start_frame": 8778, "category": "shot_missed"}, ...]
"video" is matched case-insensitively against just the filename (the log's
own "video" field is often a full path). "start_frame" must match an event's
start_frame in one of --event-logs exactly - it's how a label gets joined
back to the trajectory it describes.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from action_classifier import ActionClassifier, extract_features, features_to_vector
from goal_calibration import GoalCalibration

SCRIPT_DIR = Path(__file__).resolve().parent
MIN_SAMPLES_TO_TRAIN = 10
MIN_CLASSES_TO_TRAIN = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, required=True, help="JSON export of reviewed labels (see module docstring for format)")
    parser.add_argument("--event-logs", type=Path, nargs="+", required=True, help="One or more shot_log*.json files to pull trajectories from")
    parser.add_argument("--model-out", type=Path, default=SCRIPT_DIR / "models" / "action_classifier.joblib")
    return parser.parse_args()


def _video_key(video_field: str) -> str:
    return Path(video_field).name.lower()


def load_events_by_key(event_log_paths: list[Path]) -> dict[tuple[str, int], tuple[dict, float]]:
    """Maps (video_key, start_frame) -> (event, fps) - fps travels with the
    event since it's needed to featurize it and different logs could in
    principle come from videos with different frame rates."""
    events_by_key = {}
    for path in event_log_paths:
        data = json.loads(path.read_text())
        video_key = _video_key(data["video"])
        fps = data["scanned_window"]["fps"]
        for event in data["events"]:
            events_by_key[(video_key, event["start_frame"])] = (event, fps)
    return events_by_key


def main() -> int:
    args = parse_args()

    labels = json.loads(args.labels.read_text())
    events_by_key = load_events_by_key(args.event_logs)
    calibration = GoalCalibration.load()

    X, y, skipped = [], [], []
    for label in labels:
        key = (_video_key(label["video"]), label["start_frame"])
        found = events_by_key.get(key)
        if found is None:
            skipped.append(label)
            continue
        event, fps = found
        trajectory = [(f, (px, py)) for f, px, py in event["trajectory"]]
        try:
            features = extract_features(trajectory, calibration, fps)
        except ValueError:
            skipped.append(label)
            continue
        X.append(features_to_vector(features))
        y.append(label["category"])

    print(f"Matched {len(X)} labeled example(s) to trajectories; skipped {len(skipped)} (no matching event, or too short to featurize).")
    if skipped:
        for s in skipped[:10]:
            print(f"  skipped: video={s.get('video')} start_frame={s.get('start_frame')} category={s.get('category')}")

    class_counts = Counter(y)
    print(f"Class distribution: {dict(class_counts)}")

    if len(X) < MIN_SAMPLES_TO_TRAIN or len(class_counts) < MIN_CLASSES_TO_TRAIN:
        print(
            f"\nNot enough labeled data to train yet (have {len(X)} example(s) across "
            f"{len(class_counts)} class(es); want at least {MIN_SAMPLES_TO_TRAIN} examples "
            f"across {MIN_CLASSES_TO_TRAIN}+ classes). Label more events in the review tool "
            "(aim for ~15-20 per action type - shot, pass, cross, header, throw_in, "
            "tackle_or_block, dribble) and re-export, then run this again."
        )
        return 1

    classifier = ActionClassifier.new()
    classifier.fit(X, y)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    classifier.save(args.model_out)

    print(f"\nTrained on {len(X)} example(s) across {len(class_counts)} class(es).")
    print("Feature importances:")
    for name, importance in sorted(classifier.feature_importances().items(), key=lambda kv: -kv[1]):
        print(f"  {name}: {importance:.3f}")
    print(f"\nSaved model: {args.model_out}")
    print(
        "\nNote: this trained on whatever was available, which may still be too little "
        "to trust - treat this run as confirmation the label -> train -> predict pipeline "
        "works, not as a verdict on classifier quality. More labels, spread across all "
        "action types, is what actually improves it from here."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
