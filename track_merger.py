"""
Merges fragmented tracking IDs back into persistent player identities using
jersey-number agreement plus a temporal/spatial plausibility gate.

Tracking (BoT-SORT, see detector.py) frequently drops and reassigns a new
track_id for the same physical player after an occlusion or a crowded cluster
of similarly-dressed players. Jersey OCR (jersey_ocr.py) often reads the same
correct number across those fragments even though the tracker itself lost
continuity - observed directly: one player's number showed up as the top
guess for 9 different track_ids in a single 20-second clip.

Jersey number alone is not a safe merge key: two different players (most
commonly on opposing teams) can wear the same number and be on screen at the
same time. So two fragments are only linked when, in addition to sharing a
confident number: fragment A ends before fragment B starts (no time overlap -
this alone rules out the same-number-different-player case, since both would
be visible over overlapping spans of the match), the gap between them is
short, and the position is physically plausible for the same player to have
moved between the two spots in that time. The distance budget scales with the
box's own size rather than a fixed pixel count, so it adapts to camera
zoom/distance without needing real-world calibration (that would come from
the goal-line/homography work, still not built).
"""

from dataclasses import dataclass


@dataclass
class TrackSummary:
    track_id: str
    class_name: str
    first_frame: int
    last_frame: int
    first_box: tuple[float, float, float, float]
    last_box: tuple[float, float, float, float]
    number: str | None = None
    number_votes: int = 0
    number_total: int = 0


def _box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def _box_diagonal(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def _plausible_link(a: TrackSummary, b: TrackSummary, max_gap_frames: int, speed_factor: float) -> bool:
    gap = b.first_frame - a.last_frame
    if gap <= 0 or gap > max_gap_frames:
        return False  # b must start strictly after a ends, within the allowed gap

    ax, ay = _box_center(a.last_box)
    bx, by = _box_center(b.first_box)
    distance = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5

    scale = max(_box_diagonal(a.last_box), _box_diagonal(b.first_box))
    return distance <= scale * speed_factor * gap


class _UnionFind:
    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def merge_fragments(
    summaries: list[TrackSummary],
    min_votes: int = 2,
    min_vote_fraction: float = 0.5,
    max_gap_frames: int = 90,
    speed_factor: float = 1.5,
) -> dict[str, dict]:
    """Returns {persistent_id: {number, track_ids, first_frame, last_frame}}.

    Tracks with too few/inconsistent jersey reads to trust (below min_votes or
    min_vote_fraction) pass through unmerged as "Unidentified-<track_id>".
    """
    confident = [
        s for s in summaries
        if s.number is not None
        and s.number_total >= min_votes
        and s.number_votes / s.number_total >= min_vote_fraction
    ]

    by_number: dict[str, list[TrackSummary]] = {}
    for s in confident:
        by_number.setdefault(s.number, []).append(s)

    uf = _UnionFind(s.track_id for s in confident)
    for group in by_number.values():
        group.sort(key=lambda s: s.first_frame)
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                if _plausible_link(a, b, max_gap_frames, speed_factor):
                    uf.union(a.track_id, b.track_id)

    clusters: dict[str, list[TrackSummary]] = {}
    for s in confident:
        clusters.setdefault(uf.find(s.track_id), []).append(s)

    result = {}
    for i, members in enumerate(clusters.values(), start=1):
        members.sort(key=lambda s: s.first_frame)
        numbers = [m.number for m in members]
        number = max(set(numbers), key=numbers.count)
        result[f"Player-{number}-{i}"] = {
            "number": number,
            "track_ids": [m.track_id for m in members],
            "first_frame": members[0].first_frame,
            "last_frame": members[-1].last_frame,
        }

    merged_track_ids = {tid for m in result.values() for tid in m["track_ids"]}
    for s in summaries:
        if s.track_id in merged_track_ids:
            continue
        result[f"Unidentified-{s.track_id}"] = {
            "number": s.number,
            "track_ids": [s.track_id],
            "first_frame": s.first_frame,
            "last_frame": s.last_frame,
        }

    return result
