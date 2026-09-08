"""
Shot-on-goal detection and outcome classification: chains sparse ball
detections into trajectory segments, flags the ones that move with real
speed toward the calibrated goal, splits them into on-target (entered the
goal mouth) vs off-target (came close but missed), and for on-target shots,
heuristically classifies the outcome as a goal, a save, or blocked/missed -
with an explicit confidence level, since this is a genuinely hard call from
a single 2D camera.

"Near the goal" alone isn't a shot - the ball sits in that area during a
goal kick or when the keeper is holding it, both essentially stationary.
What distinguishes a shot is directed motion at real speed toward the goal.

OUTCOME CLASSIFICATION is a heuristic, not a determination. There is no
ball-height or depth information, so "the ball is in the net" and "the ball
is in front of the net" can look identical in this camera's 2D projection.
The logic here:
  - After the ball is on-target, look at what it does next. If a goalkeeper
    was detected near the ball at/around the on-target moment AND the ball
    subsequently moves back out toward the pitch, call it a SAVE (high
    confidence with a clear retreat, medium if the retreat is weak).
  - If the ball retreats but no goalkeeper was detected nearby, it was
    stopped by something else (a post, a defender on the line) - call it
    BLOCKED_OR_MISSED rather than crediting a save that wasn't observed.
  - If the ball is not seen again for a while and doesn't reappear moving
    back onto the pitch, call it a GOAL (medium confidence - a long gap in
    ball detection is also just a normal detection gap, so this is the
    weakest of the three signals).
  - Anything that doesn't cleanly match one of these patterns is UNCERTAIN.
Real goal-line technology uses several synchronized, calibrated cameras per
goal specifically because one camera can't resolve this cleanly - treat
every classification here as a candidate for human review, not a stat to
publish uncritically.

Ball tracking fragments like player tracking does (see detector.py), so
segments are built directly from the chronological position sequence rather
than trusting one BoT-SORT track_id to span a whole shot: consecutive
detections chain into one segment when close enough in time with a
physically sane implied speed; a big gap or an impossible jump starts a new
segment instead (the same idea as track_merger.py's plausibility gate,
simplified since the ball has no jersey number to key on).
"""

from dataclasses import dataclass, field

from goal_calibration import GoalCalibration, is_in_goal_mouth

Point = tuple[float, float]


@dataclass
class ShotEvent:
    start_frame: int
    end_frame: int
    peak_speed: float  # px/frame, in calibration image space
    on_target: bool
    entry_frame: int | None = None
    closest_approach_px: float | None = None  # off-target only: distance from the goal line at closest point
    outcome: str | None = None  # "goal" | "save" | "blocked_or_missed" | "uncertain"; on-target only
    outcome_confidence: str | None = None  # "high" | "medium" | "low"
    trajectory: list[tuple[int, Point]] = field(default_factory=list)
    shot_taker_track_id: str | None = None
    shot_taker_jersey_number: str | None = None
    goalkeeper_track_id: str | None = None
    goalkeeper_distance_px: float | None = None
    goalkeeper_jersey_number: str | None = None


def _distance(a: Point, b: Point) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _distance_to_segment(point: Point, a: Point, b: Point) -> float:
    ax, ay = a
    bx, by = b
    px, py = point
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return _distance(point, a)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    return _distance(point, (ax + t * dx, ay + t * dy))


def _build_segments(
    ball_positions: list[tuple[int, Point]],
    max_gap_frames: int,
    max_speed_px_per_frame: float,
) -> list[list[tuple[int, Point]]]:
    """Chains chronological ball detections into segments, splitting on a
    gap that's too large or a jump too fast to be the same ball in flight."""
    if not ball_positions:
        return []

    segments = [[ball_positions[0]]]
    for frame_idx, pos in ball_positions[1:]:
        prev_frame, prev_pos = segments[-1][-1]
        gap = frame_idx - prev_frame
        implied_speed = _distance(pos, prev_pos) / max(gap, 1)
        if gap <= max_gap_frames and implied_speed <= max_speed_px_per_frame:
            segments[-1].append((frame_idx, pos))
        else:
            segments.append([(frame_idx, pos)])
    return segments


def _classify_outcome(
    entry_frame: int,
    entry_pos: Point,
    ball_positions: list[tuple[int, Point]],
    goalkeeper_positions: list[tuple[int, str, Point]],
    calibration: GoalCalibration,
    outcome_window_frames: int,
    keeper_proximity_px: float,
) -> tuple[str, str]:
    keeper_nearby = [
        (tid, pos) for f, tid, pos in goalkeeper_positions
        if abs(f - entry_frame) <= outcome_window_frames and _distance(pos, entry_pos) <= keeper_proximity_px
    ]

    after = [(f, p) for f, p in ball_positions if entry_frame < f <= entry_frame + outcome_window_frames]

    if not after:
        # ball not seen again in the follow-up window - plausibly settled in the net
        return ("goal", "medium") if not keeper_nearby else ("uncertain", "low")

    retreated = any(not is_in_goal_mouth(p, calibration) for _, p in after)
    if retreated and keeper_nearby:
        # did it clearly move back toward the pitch, or just wobble near the line?
        max_exit_distance = max(
            _distance(p, entry_pos) for _, p in after if not is_in_goal_mouth(p, calibration)
        )
        confidence = "high" if max_exit_distance > keeper_proximity_px else "medium"
        return "save", confidence
    if retreated and not keeper_nearby:
        return "blocked_or_missed", "medium"

    # stayed inside the goal mouth for the whole follow-up window
    return ("goal", "medium") if not keeper_nearby else ("uncertain", "low")


def _find_shot_taker(
    start_frame: int,
    ball_start_pos: Point,
    player_positions: list[tuple[int, str, Point]],
    lookback_frames: int,
) -> tuple[str | None, float | None]:
    candidates = [
        (tid, _distance(pos, ball_start_pos))
        for f, tid, pos in player_positions
        if 0 <= start_frame - f <= lookback_frames
    ]
    if not candidates:
        return None, None
    return min(candidates, key=lambda c: c[1])


def detect_shots(
    ball_positions: list[tuple[int, Point]],
    calibration: GoalCalibration,
    goalkeeper_positions: list[tuple[int, str, Point]] | None = None,
    player_positions: list[tuple[int, str, Point]] | None = None,
    jersey_numbers: dict[str, str] | None = None,
    max_gap_frames: int = 20,
    max_speed_px_per_frame: float = 250.0,
    min_shot_speed: float = 8.0,
    min_segment_points: int = 2,
    outcome_window_frames: int = 60,
    keeper_proximity_px: float | None = None,
    near_miss_radius_px: float | None = None,
    shot_taker_lookback_frames: int = 15,
) -> list[ShotEvent]:
    """goalkeeper_positions/player_positions: [(frame_idx, track_id, (x, y)), ...].
    jersey_numbers: {track_id: number} from JerseyVoteTracker.best_guess() winners,
    used only to label events - a missing entry just leaves the field unset.

    keeper_proximity_px and near_miss_radius_px default to multiples of the
    goal width so they scale with camera zoom instead of being fixed pixel
    counts tuned to one framing.
    """
    goalkeeper_positions = goalkeeper_positions or []
    player_positions = player_positions or []
    jersey_numbers = jersey_numbers or {}

    goal_width = _distance(calibration.left_post_base, calibration.right_post_base)
    if keeper_proximity_px is None:
        keeper_proximity_px = goal_width * 0.6
    if near_miss_radius_px is None:
        near_miss_radius_px = goal_width * 2.5

    segments = _build_segments(ball_positions, max_gap_frames, max_speed_px_per_frame)

    events = []
    for segment in segments:
        if len(segment) < min_segment_points:
            continue

        speeds = [
            _distance(segment[i][1], segment[i - 1][1]) / max(segment[i][0] - segment[i - 1][0], 1)
            for i in range(1, len(segment))
        ]
        peak_speed = max(speeds)
        if peak_speed < min_shot_speed:
            continue  # too slow to be a shot - a goal kick setup, keeper holding the ball, etc.

        entry_frame = None
        entry_pos = None
        for frame_idx, pos in segment:
            if is_in_goal_mouth(pos, calibration):
                entry_frame, entry_pos = frame_idx, pos
                break

        if entry_frame is None:
            distances = [(_distance_to_segment(pos, *calibration.goal_line), pos) for _, pos in segment]
            start_distance = distances[0][0]
            closest, closest_pos = min(distances, key=lambda d: d[0])
            approached_by = start_distance - closest

            # a single spurious point can look "close" by raw distance alone - require a
            # few points near it too, so one bad detection can't manufacture a near-miss
            support_count = sum(1 for d, _ in distances if d <= closest * 1.5)

            # distance-to-segment treats a point far below/above an endpoint as "close"
            # purely from horizontal proximity (see Event 4 in dev notes: y~600 near a
            # goal at y~140-200 scored 392px "close" this way) - bound the vertical range
            # explicitly rather than trusting raw distance alone
            goal_top = min(calibration.left_post_top[1], calibration.right_post_top[1])
            goal_bottom = max(calibration.left_post_base[1], calibration.right_post_base[1])
            goal_height = goal_bottom - goal_top
            y_ok = (goal_top - goal_height * 2) <= closest_pos[1] <= (goal_bottom + goal_height * 2)

            # require it to actually be near the goal (in both distance and plausible
            # height) AND to have net-approached it during this segment with more than
            # one supporting point - "came within range once" also matches a ball that
            # started close and wandered away, or noise jumping between unrelated objects.
            # A 2-point segment can't demonstrate a real supporting cluster (both points
            # trivially "support" each other), so off-target specifically needs a third.
            if (
                closest > near_miss_radius_px
                or approached_by < goal_width * 0.5
                or support_count < 2
                or not y_ok
                or len(segment) < 3
            ):
                continue
            events.append(ShotEvent(
                start_frame=segment[0][0], end_frame=segment[-1][0],
                peak_speed=peak_speed, on_target=False, closest_approach_px=closest,
                trajectory=segment,
            ))
            continue

        outcome, confidence = _classify_outcome(
            entry_frame, entry_pos, ball_positions, goalkeeper_positions,
            calibration, outcome_window_frames, keeper_proximity_px,
        )

        event = ShotEvent(
            start_frame=segment[0][0], end_frame=segment[-1][0], entry_frame=entry_frame,
            peak_speed=peak_speed, on_target=True, outcome=outcome, outcome_confidence=confidence,
            trajectory=segment,
        )

        nearby_keepers = [
            (tid, _distance(pos, entry_pos)) for f, tid, pos in goalkeeper_positions
            if abs(f - entry_frame) <= outcome_window_frames
        ]
        if nearby_keepers:
            event.goalkeeper_track_id, event.goalkeeper_distance_px = min(nearby_keepers, key=lambda x: x[1])
            event.goalkeeper_jersey_number = jersey_numbers.get(event.goalkeeper_track_id)

        shot_taker_id, _ = _find_shot_taker(segment[0][0], segment[0][1], player_positions, shot_taker_lookback_frames)
        if shot_taker_id:
            event.shot_taker_track_id = shot_taker_id
            event.shot_taker_jersey_number = jersey_numbers.get(shot_taker_id)

        events.append(event)

    return events
