"""
Goal-line calibration geometry.

Loads the 4 image-space points marked via the calibration Artifact (left/right
post base and top) and provides 2D heuristics for goal-mouth proximity.

This is a monocular, single-frame calibration: it has no notion of ball
height or camera depth, so it can only reason about where the ball sits in
the 2D image, not whether it is truly between the posts and under the bar in
3D. Real goal-line technology (e.g. Hawk-Eye) uses several synchronized,
calibrated cameras per goal specifically because a single camera can't
resolve that - treat everything here as a candidate-event heuristic for
human review, not a certain goal/no-goal determination.

Also: this calibration is only valid for camera framings matching the frame
it was marked on. The source camera (XbootGo) auto-pans/zooms to follow
play, so a real deployment would need to recalibrate per shot or build a
homography from pitch markings - neither is done here.
"""

import json
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CALIBRATION_PATH = SCRIPT_DIR / "goal_calibration.json"

Point = tuple[float, float]


@dataclass
class GoalCalibration:
    video: str
    frame_timestamp_seconds: float
    image_width: int
    image_height: int
    left_post_base: Point
    right_post_base: Point
    left_post_top: Point
    right_post_top: Point

    @classmethod
    def load(cls, path: Path = DEFAULT_CALIBRATION_PATH) -> "GoalCalibration":
        data = json.loads(Path(path).read_text())
        points = data["points"]
        return cls(
            video=data["video"],
            frame_timestamp_seconds=data["frame_timestamp_seconds"],
            image_width=data["image_width"],
            image_height=data["image_height"],
            left_post_base=tuple(points["left_post_base"]),
            right_post_base=tuple(points["right_post_base"]),
            left_post_top=tuple(points["left_post_top"]),
            right_post_top=tuple(points["right_post_top"]),
        )

    @property
    def mouth_polygon(self) -> list[Point]:
        """Goal-mouth quadrilateral in image space, wound for point-in-polygon tests."""
        return [self.left_post_base, self.left_post_top, self.right_post_top, self.right_post_base]

    @property
    def goal_line(self) -> tuple[Point, Point]:
        """The ground-level line between the two posts."""
        return self.left_post_base, self.right_post_base


def point_in_polygon(point: Point, polygon: list[Point]) -> bool:
    """Standard ray-casting point-in-polygon test."""
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def is_in_goal_mouth(point: Point, calibration: GoalCalibration) -> bool:
    """Whether a 2D point (e.g. ball center) falls within the goal-mouth
    quadrilateral as seen by this camera. A 2D proxy only - it cannot tell
    the ball being IN the goal from it merely appearing in front of or
    behind the goal mouth from this camera's viewpoint."""
    return point_in_polygon(point, calibration.mouth_polygon)


def find_mouth_entry_frame(
    ball_positions: list[tuple[int, Point]],
    calibration: GoalCalibration,
) -> int | None:
    """Given [(frame_idx, (x, y)), ...] in chronological order, returns the
    frame_idx of the first transition from outside to inside the goal-mouth
    polygon, or None if the ball never enters. A candidate "shot reached the
    goal mouth" event for human review, not a scored-goal determination."""
    was_inside = False
    for frame_idx, pos in ball_positions:
        inside = is_in_goal_mouth(pos, calibration)
        if inside and not was_inside:
            return frame_idx
        was_inside = inside
    return None
