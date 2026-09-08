# Setup

This project targets Windows + WSL2. Native Windows Python **does not work** here — see below.

## Why WSL2 is required

Windows **Smart App Control** (a system-level code-integrity feature) blocks numpy/opencv/torch's
native DLLs when run via native Windows Python — this is not a package problem, it's the OS
refusing to load unsigned compiled binaries. It showed up as:

```
ImportError: DLL load failed while importing _multiarray_umath: An Application Control policy has blocked this file.
```

Do not try to work around this by disabling Smart App Control — it's a one-way, system-wide
security downgrade (once off, it can only be turned back on via a clean Windows reinstall). The
fix is running everything inside WSL2 instead, which isn't subject to it.

## 1. Install WSL2 Ubuntu

```powershell
wsl --install -d Ubuntu --no-launch
```

First launch normally prompts interactively for a username/password, which doesn't work
non-interactively — run as root once to sidestep that, then create a real user:

```powershell
wsl -d Ubuntu -u root -- bash -c "
  adduser --disabled-password --gecos '' YOURUSERNAME &&
  usermod -aG sudo YOURUSERNAME &&
  echo 'YOURUSERNAME ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/YOURUSERNAME &&
  chmod 440 /etc/sudoers.d/YOURUSERNAME
"
wsl -d Ubuntu -u root -- bash -c "printf '[user]\ndefault=YOURUSERNAME\n' > /etc/wsl.conf"
wsl --terminate Ubuntu
```

`wsl -d Ubuntu` now logs in as that user by default.

## 2. System packages

```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip libgl1 libglib2.0-0 ffmpeg
```

`libgl1`/`libglib2.0-0` are OpenCV's runtime deps; `ffmpeg` is only needed if you extract/compress
video clips (not for the core pipeline).

## 3. Clone the repo and set up the venv

**Put the venv on WSL's native filesystem, not under `/mnt/c/...`.** Ultralytics explicitly warns
Windows-mounted paths are much slower for training, and it caused a `PermissionError` writing
training logs when tried. Keep the repo itself on the Windows side (so normal Windows tools/git
work on it) and just point the WSL-native venv at it:

```bash
mkdir -p ~/soccer-analytics && cd ~/soccer-analytics
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
```

Then clone the repo on the **Windows side** (e.g. `C:\Users\<you>\soccer-analytics`) and install
from there:

```bash
cd ~/soccer-analytics
.venv/bin/python -m pip install --only-binary=:all: -r /mnt/c/Users/<you>/soccer-analytics/requirements.txt
```

## 4. Known dependency gotchas

- **`opencv-python` vs `opencv-python-headless`**: only one may be installed at a time — they ship
  the same `cv2` files and having both is unsupported/fragile. `requirements.txt` already pins
  `opencv-python-headless` for this reason (roboflow pulls in headless as a dependency anyway, and
  headless is the right choice for a no-GUI WSL environment). If you ever see both listed in
  `pip show`, `pip uninstall` both and reinstall just `opencv-python-headless`.
- **`lap`** (a tracker dependency) is pinned in `requirements.txt` deliberately — if it's missing,
  Ultralytics silently auto-installs it mid-run the first time `.track()` is called, and since the
  newly installed module doesn't load into the already-running process, **tracking silently returns
  no IDs for the rest of that run** with no error. Always let `pip install -r requirements.txt`
  install it upfront rather than relying on the auto-install.
- **Tracker config**: `detector.py` explicitly passes `tracker="botsort.yaml"` rather than using
  Ultralytics' own unnamed default — the default silently drops all tracking IDs once a frame has
  ~40+ boxes (a real bug in this Ultralytics version, confirmed directly against this project's
  data). BoT-SORT also measurably outperformed ByteTrack on this footage (fewer fragmented IDs).

## 5. Running scripts

Everything runs through the WSL venv, pointed at the Windows-side repo path:

```bash
wsl -d Ubuntu -- ~/soccer-analytics/.venv/bin/python /mnt/c/Users/<you>/soccer-analytics/test_tracker.py <video-path> [options]
```

From Git Bash on Windows specifically, prefix with `MSYS_NO_PATHCONV=1` or `/mnt/c/...`-style
paths get mangled by MSYS's automatic path translation:

```bash
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -- ~/soccer-analytics/.venv/bin/python /mnt/c/Users/<you>/soccer-analytics/test_shot_detection.py "..."
```

## 6. Files not in this repo — copy these separately

- **`models/soccer_v1_best.pt`** and **`models/ball_v1_best.pt`** — trained model weights,
  gitignored (small, ~6MB each, but represent real training time — carry them over directly rather
  than retraining unless you actually want to).
- **`.env`** — needs `ROBOFLOW_API_KEY=<your key>` (get one free at roboflow.com), only required if
  you run `download_dataset.py` / `train.py` to rebuild or retrain the models. Not needed just to
  run the existing detection/tracking/OCR/shot-detection pipeline.

Video footage lives on OneDrive (`Videos/Soccer Videos/`) and is reachable from any machine signed
into the same Microsoft account — cloud-only files need to hydrate (fully download) before
OpenCV can read them; either wait for that or use `attrib +P <file>` to force it.

## Pipeline overview

In dependency order:

- `detector.py` — `CombinedDetector`: two fine-tuned YOLO models (player/goalkeeper/referee/ball,
  plus a ball specialist since the merged model's ball recall was too weak on its own) with
  BoT-SORT tracking.
- `jersey_ocr.py` — EasyOCR digit reading on player crops, majority-voted per track ID.
- `track_merger.py` — stitches fragmented tracking IDs back into one persistent identity when
  jersey number agrees and the timing/position is physically plausible.
- `goal_calibration.py` — loads a human-marked 4-point goal-mouth calibration (2D image-space
  only, valid only for the camera framing it was marked on).
- `shot_detector.py` — chains sparse ball detections into trajectories, flags on-target/off-target
  shot attempts, heuristically classifies outcome (goal/save/blocked/uncertain) with an explicit
  confidence level rather than forcing a guess.

Two Artifact-based tools (published separately, not part of this repo's runtime — source copies
kept here for reference):

- `calibration_tool.html` — click-to-mark goal calibration UI (writes to `goal_calibration.json`).
- `event_log_tool.html` — review/correct detected shot events and manually log shot-attempt or
  restart (kickoff/goal-kick/corner/etc.) events as ground truth. Deliberately keeps "was this a
  shot, and what happened" separate from "what restart followed" — a restart only tells you which
  side touched the ball last, never whether a shot occurred (a missed shot and an unrelated
  turnover both produce a goal kick; a save and a routine defensive clearance both produce a
  corner).
