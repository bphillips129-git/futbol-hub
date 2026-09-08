"""
Downloads the football players detection dataset (ball, goalkeeper, player, referee)
from Roboflow Universe in YOLO format.

Source: https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc
License: CC BY 4.0

Requires ROBOFLOW_API_KEY set in a .env file in this directory. Get a free key at
https://app.roboflow.com -> Settings -> API Keys, then create a .env file here with:
    ROBOFLOW_API_KEY=<your key>

Usage:
    python download_dataset.py
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

WORKSPACE = "roboflow-jvuqo"
PROJECT = "football-players-detection-3zvbc"
VERSION = 1
FORMAT = "yolov8"
OUTPUT_DIR = "datasets/football-players-detection"


def main() -> int:
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("ROBOFLOW_API_KEY not set.")
        print("Create a free account at https://roboflow.com, generate an API key")
        print("(Settings -> API Keys), and put it in a .env file here as:")
        print("  ROBOFLOW_API_KEY=<your key>")
        return 1

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(WORKSPACE).project(PROJECT)

    try:
        version = project.version(VERSION)
    except Exception as exc:
        available = [v.version for v in project.versions()]
        print(f"Could not load version {VERSION}: {exc}")
        print(f"Available versions for this project: {available}")
        return 1

    dataset = version.download(FORMAT, location=OUTPUT_DIR)
    print(f"Dataset downloaded to: {dataset.location}")
    print(f"data.yaml: {dataset.location}/data.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
