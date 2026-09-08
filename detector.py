"""
Combined detector: merges two fine-tuned YOLO models into one detection call.

soccer_v1 (player/goalkeeper/referee/ball) is strong on player, goalkeeper, and
referee but weak on ball (too few ball examples in its training set). ball_v1
is a specialist trained only on ball, with much better ball recall. This class
runs both per frame and merges results: player/goalkeeper/referee come from
soccer_v1 (its own ball predictions are discarded), ball comes from ball_v1.

track() additionally assigns persistent IDs across frames using BoT-SORT
(appearance + motion; measurably fewer fragmented IDs than ByteTrack on this
footage, since same-team players in identical kits confuse motion-only
tracking). Each underlying model keeps its own independent tracker, so IDs
from the players model and the ball model can numerically collide; track_id
is prefixed ("P"/"B") so it stays unambiguous even without checking
class_name. The library's own default tracker config silently drops all IDs
once a frame has ~40+ boxes (observed directly on this dataset), so
botsort.yaml is used explicitly rather than relying on the unnamed default.

Even with BoT-SORT, expect real ID fragmentation in dense soccer scenes
(observed ~265 unique player IDs over 20s of real footage with ~22 players
on screen) - this is a known hard problem without jersey-number-based
re-identification, not a bug in this code.
"""

from dataclasses import dataclass
from pathlib import Path

from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PLAYERS_MODEL = SCRIPT_DIR / "models" / "soccer_v1_best.pt"
DEFAULT_BALL_MODEL = SCRIPT_DIR / "models" / "ball_v1_best.pt"
DEFAULT_TRACKER = "botsort.yaml"


@dataclass
class Detection:
    class_name: str
    confidence: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2
    track_id: str | None = None


class CombinedDetector:
    def __init__(
        self,
        players_model_path=DEFAULT_PLAYERS_MODEL,
        ball_model_path=DEFAULT_BALL_MODEL,
        tracker: str = DEFAULT_TRACKER,
    ):
        self._players_model_path = Path(players_model_path)
        self._ball_model_path = Path(ball_model_path)
        if not self._players_model_path.exists():
            raise FileNotFoundError(f"Players model not found: {self._players_model_path}")
        if not self._ball_model_path.exists():
            raise FileNotFoundError(f"Ball model not found: {self._ball_model_path}")

        self.tracker = tracker
        self.players_model = YOLO(str(self._players_model_path))
        self.ball_model = YOLO(str(self._ball_model_path))

    def detect(self, frame, conf: float = 0.25) -> list[Detection]:
        detections = []

        players_result = self.players_model(frame, verbose=False, conf=conf)[0]
        names = players_result.names
        for box in players_result.boxes:
            class_name = names[int(box.cls)]
            if class_name == "ball":
                continue  # weak class on this model; ball_v1 covers it instead
            detections.append(Detection(class_name, float(box.conf), tuple(box.xyxy[0].tolist())))

        ball_result = self.ball_model(frame, verbose=False, conf=conf)[0]
        for box in ball_result.boxes:
            detections.append(Detection("ball", float(box.conf), tuple(box.xyxy[0].tolist())))

        return detections

    def track(self, frame, conf: float = 0.25) -> list[Detection]:
        """Call once per frame, in order, for a single continuous video.
        Use reset_tracking() before starting a different video."""
        detections = []

        players_result = self.players_model.track(
            frame, persist=True, tracker=self.tracker, verbose=False, conf=conf
        )[0]
        names = players_result.names
        ids = players_result.boxes.id
        for i, box in enumerate(players_result.boxes):
            class_name = names[int(box.cls)]
            if class_name == "ball":
                continue
            track_id = f"P{int(ids[i])}" if ids is not None else None
            detections.append(Detection(class_name, float(box.conf), tuple(box.xyxy[0].tolist()), track_id))

        ball_result = self.ball_model.track(
            frame, persist=True, tracker=self.tracker, verbose=False, conf=conf
        )[0]
        ball_ids = ball_result.boxes.id
        for i, box in enumerate(ball_result.boxes):
            track_id = f"B{int(ball_ids[i])}" if ball_ids is not None else None
            detections.append(Detection("ball", float(box.conf), tuple(box.xyxy[0].tolist()), track_id))

        return detections

    def reset_tracking(self) -> None:
        """Clear tracker state (e.g. before switching to a new video)."""
        self.players_model = YOLO(str(self._players_model_path))
        self.ball_model = YOLO(str(self._ball_model_path))
