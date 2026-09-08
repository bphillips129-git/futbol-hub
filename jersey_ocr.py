"""
Jersey number OCR: reads digits from a player bounding box crop.

Crops the torso region of the box (skips head and legs, where numbers never
appear), upscales small crops for better recognition, and restricts EasyOCR's
character set to digits only (0-9) - jersey numbers are always numeric, and
this cuts out a lot of false reads from sponsor logos/text on the kit.

Single-frame reads are noisy (motion blur, partial occlusion, player facing
away from the number). JerseyVoteTracker accumulates reads per track_id
across frames and takes a majority vote, meant to be used alongside
CombinedDetector.track() from detector.py.
"""

from collections import Counter, defaultdict

import cv2
import easyocr

TORSO_TOP_FRACTION = 0.15  # skip the head
TORSO_BOTTOM_FRACTION = 0.60  # skip the legs
MIN_CROP_HEIGHT_PX = 200  # upscale smaller crops to roughly this height


class JerseyOCR:
    def __init__(self, gpu: bool = False):
        self.reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)

    def read_number(self, frame, box: tuple[float, float, float, float]) -> tuple[str | None, float]:
        x1, y1, x2, y2 = (int(v) for v in box)
        height = y2 - y1
        torso_y1 = y1 + int(height * TORSO_TOP_FRACTION)
        torso_y2 = y1 + int(height * TORSO_BOTTOM_FRACTION)

        crop = frame[max(torso_y1, 0):max(torso_y2, 0), max(x1, 0):max(x2, 0)]
        if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
            return None, 0.0

        scale = max(1, MIN_CROP_HEIGHT_PX // crop.shape[0])
        if scale > 1:
            crop = cv2.resize(crop, (crop.shape[1] * scale, crop.shape[0] * scale), interpolation=cv2.INTER_CUBIC)

        results = self.reader.readtext(crop, allowlist="0123456789")
        if not results:
            return None, 0.0

        _, text, confidence = max(results, key=lambda r: r[2])
        if not text.isdigit() or not (1 <= len(text) <= 2):
            return None, 0.0  # real jersey numbers are 1-2 digits; longer reads are OCR noise
        return text, confidence


class JerseyVoteTracker:
    def __init__(self):
        self._votes: dict[str, Counter] = defaultdict(Counter)

    def add(self, track_id: str, number: str) -> None:
        self._votes[track_id][number] += 1

    def best_guess(self, track_id: str) -> tuple[str | None, int, int]:
        """Returns (number, votes_for_it, total_votes) or (None, 0, 0)."""
        counter = self._votes.get(track_id)
        if not counter:
            return None, 0, 0
        number, count = counter.most_common(1)[0]
        return number, count, sum(counter.values())

    def all_track_ids(self):
        return self._votes.keys()
