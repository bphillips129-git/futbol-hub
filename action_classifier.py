"""
Action-type classifier prototype: given a ball-trajectory segment (the same
shape shot_detector.py already produces), predicts which SoccerNet-style
ball-action category it actually is (shot_* / pass / cross / header /
throw_in / tackle_or_block / dribble / not_a_ball_action) rather than
assuming anything fast near the goal is a shot attempt.

This is deliberately a small, interpretable model (RandomForest over
hand-engineered trajectory features) rather than a deep model trained on raw
pixels - there isn't remotely enough labeled data for that yet, and the goal
right now is a working label -> train -> predict loop the labeled set can
grow into, not a finished classifier. See train_action_classifier.py.

Feature set is limited to what shot_detector.py already logs: the ball's own
trajectory plus goal calibration geometry. Player-proximity features (was a
player stationary vs sprinting at the start point - one of the more useful
signals for pass vs. shot vs. dribble) would meaningfully improve this but
require test_shot_detection.py to persist player positions alongside each
event, which it doesn't yet.

Distance and speed features are expressed in real-world units (feet,
feet/second) via GoalCalibration.feet_per_pixel, not raw pixels/frames -
this project isn't staying on one camera setup or one match format (7v7/9v9/
11v11 goals and fields are different physical sizes, not just different
pixel counts), so a feature like "40px from goal" would mean something
different in every video. "6 feet from goal" doesn't. Frame-count-based
figures (duration, speed) are similarly converted using each video's own fps
so they're not silently comparing a 30fps game to a 60fps one frame-for-frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
from sklearn.ensemble import RandomForestClassifier

from goal_calibration import GoalCalibration

Point = tuple[float, float]

FEATURE_NAMES = [
    "duration_seconds",
    "num_points",
    "peak_speed_ft_per_sec",
    "avg_speed_ft_per_sec",
    "speed_variability_ft_per_sec",
    "path_length_ft",
    "net_displacement_ft",
    "straightness",
    "start_dist_to_goal_ft",
    "end_dist_to_goal_ft",
    "approach_delta_ft",
    "direction_alignment",
    "start_x_frac",
    "start_y_frac",
    "vertical_extent_ft",
]


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _goal_center(calibration: GoalCalibration) -> Point:
    xs = [calibration.left_post_base[0], calibration.right_post_base[0],
          calibration.left_post_top[0], calibration.right_post_top[0]]
    ys = [calibration.left_post_base[1], calibration.right_post_base[1],
          calibration.left_post_top[1], calibration.right_post_top[1]]
    return (sum(xs) / 4, sum(ys) / 4)


def extract_features(trajectory: Sequence[tuple[int, Point]], calibration: GoalCalibration, fps: float) -> dict[str, float]:
    """trajectory: [(frame_idx, (x, y)), ...] in calibration image space,
    chronological - the same shape ShotEvent.trajectory already is. fps is
    the source video's frame rate, needed to turn frame-gaps into real
    seconds (a "40px/frame" ball looks twice as fast on a 60fps video as the
    same real motion on 30fps footage unless this is accounted for)."""
    if len(trajectory) < 2:
        raise ValueError("need at least 2 trajectory points to extract features")

    ftpp = calibration.feet_per_pixel
    frames = [f for f, _ in trajectory]
    points = [p for _, p in trajectory]
    goal_center = _goal_center(calibration)

    step_distances_px = [_distance(points[i], points[i - 1]) for i in range(1, len(points))]
    step_gaps_frames = [max(frames[i] - frames[i - 1], 1) for i in range(1, len(points))]
    # feet/frame * frames/second = feet/second
    speeds_ft_per_sec = [(d * ftpp) / g * fps for d, g in zip(step_distances_px, step_gaps_frames)]

    path_length_ft = sum(step_distances_px) * ftpp
    net_displacement_ft = _distance(points[0], points[-1]) * ftpp
    duration_seconds = (frames[-1] - frames[0]) / fps

    start_dist_to_goal_ft = _distance(points[0], goal_center) * ftpp
    end_dist_to_goal_ft = _distance(points[-1], goal_center) * ftpp

    dx, dy = points[-1][0] - points[0][0], points[-1][1] - points[0][1]
    gx, gy = goal_center[0] - points[0][0], goal_center[1] - points[0][1]
    move_mag = math.hypot(dx, dy)
    goal_mag = math.hypot(gx, gy)
    direction_alignment = (dx * gx + dy * gy) / (move_mag * goal_mag) if move_mag > 0 and goal_mag > 0 else 0.0

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    mean_speed_ft_per_sec = sum(speeds_ft_per_sec) / len(speeds_ft_per_sec)
    variance = sum((s - mean_speed_ft_per_sec) ** 2 for s in speeds_ft_per_sec) / len(speeds_ft_per_sec)
    path_length_px = sum(step_distances_px)

    return {
        "duration_seconds": duration_seconds,
        "num_points": float(len(points)),
        "peak_speed_ft_per_sec": max(speeds_ft_per_sec),
        "avg_speed_ft_per_sec": mean_speed_ft_per_sec,
        "speed_variability_ft_per_sec": math.sqrt(variance),
        "path_length_ft": path_length_ft,
        "net_displacement_ft": net_displacement_ft,
        "straightness": (path_length_px and _distance(points[0], points[-1]) / path_length_px) or 0.0,
        "start_dist_to_goal_ft": start_dist_to_goal_ft,
        "end_dist_to_goal_ft": end_dist_to_goal_ft,
        "approach_delta_ft": start_dist_to_goal_ft - end_dist_to_goal_ft,
        "direction_alignment": direction_alignment,
        "start_x_frac": xs[0] / calibration.image_width,
        "start_y_frac": ys[0] / calibration.image_height,
        "vertical_extent_ft": (max(ys) - min(ys)) * ftpp,
    }


def features_to_vector(features: dict[str, float]) -> list[float]:
    return [features[name] for name in FEATURE_NAMES]


@dataclass
class ActionClassifier:
    model: RandomForestClassifier

    @classmethod
    def new(cls, n_estimators: int = 200, random_state: int = 0) -> "ActionClassifier":
        return cls(model=RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=6,
            min_samples_leaf=1,
            random_state=random_state,
            class_weight="balanced",
        ))

    def fit(self, X: list[list[float]], y: list[str]) -> None:
        self.model.fit(X, y)

    def predict(self, X: list[list[float]]) -> list[str]:
        return list(self.model.predict(X))

    def predict_proba(self, X: list[list[float]]) -> list[dict[str, float]]:
        proba = self.model.predict_proba(X)
        classes = self.model.classes_
        return [dict(zip(classes, row)) for row in proba]

    def feature_importances(self) -> dict[str, float]:
        return dict(zip(FEATURE_NAMES, self.model.feature_importances_))

    def save(self, path: Path) -> None:
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: Path) -> "ActionClassifier":
        return cls(model=joblib.load(path))
