"""
Fine-tunes a YOLO model on the downloaded football players detection dataset
(ball, goalkeeper, player, referee). Run download_dataset.py first.

CPU-only environment: start with a low --epochs value (e.g. 1-2) to measure
per-epoch time before committing to a full training run.

Usage:
    python train.py [--epochs 30] [--imgsz 640] [--model yolov8n.pt]
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

DEFAULT_DATA = Path("datasets/football-players-detection/data.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=None, help="Path to data.yaml (default: datasets/football-players-detection/data.yaml)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--model", default="yolov8n.pt", help="Base checkpoint to fine-tune from")
    parser.add_argument("--name", default="soccer_players", help="Run name under runs/detect/")
    return parser.parse_args()


def resolve_data_path(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    if DEFAULT_DATA.exists():
        return DEFAULT_DATA
    candidates = sorted(Path("datasets").glob("*/data.yaml"))
    if not candidates:
        raise FileNotFoundError("No data.yaml found under datasets/. Run download_dataset.py first.")
    return candidates[0]


def main() -> None:
    args = parse_args()
    data_path = resolve_data_path(args.data)
    print(f"Using dataset: {data_path}")
    print(f"Base model: {args.model}, epochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch}, device=cpu")

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device="cpu",
        name=args.name,
    )


if __name__ == "__main__":
    main()
