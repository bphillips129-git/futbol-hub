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

`libgl1`/`libglib2.0-0` are OpenCV's runtime deps; `ffmpeg` is used both for video clip
extraction/compression and by `extract_event_clips.py` (see below).

If `apt-get install` 404s on a specific package version from `security.ubuntu.com` (seen once with
`python3.14-venv`), that's a transient mirror-sync gap, not a real problem — `apt-get update` and
retry the install a minute later.

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

Then clone the repo on the **Windows side** (e.g. `C:\Users\<you>\GitHub\futbol-hub`) and install
from there:

```bash
cd ~/soccer-analytics
.venv/bin/python -m pip install --only-binary=:all: -r /mnt/c/Users/<you>/GitHub/futbol-hub/requirements.txt
```

Check afterward that only `opencv-python-headless` ended up installed, not both `opencv-python`
and `opencv-python-headless` — see the gotcha below; `roboflow` pulls in the non-headless one as a
transitive dependency even though `requirements.txt` pins headless directly.

## 4. Known dependency gotchas

- **`opencv-python` vs `opencv-python-headless`**: only one may be installed at a time — they ship
  the same `cv2` files and having both is unsupported/fragile. `requirements.txt` pins
  `opencv-python-headless`, but `roboflow` pulls in the non-headless `opencv-python` as a
  transitive dependency regardless, so a plain `pip install -r requirements.txt` typically leaves
  both installed. Fix: `pip uninstall opencv-python opencv-python-headless`, then
  `pip install opencv-python-headless==<pinned version>` alone.
- **`lap`** (a tracker dependency) is pinned in `requirements.txt` deliberately — if it's missing,
  Ultralytics silently auto-installs it mid-run the first time `.track()` is called, and since the
  newly installed module doesn't load into the already-running process, **tracking silently returns
  no IDs for the rest of that run** with no error. Always let `pip install -r requirements.txt`
  install it upfront rather than relying on the auto-install.
- **Tracker config**: `detector.py` explicitly passes `tracker="botsort.yaml"` rather than using
  Ultralytics' own unnamed default — the default silently drops all tracking IDs once a frame has
  ~40+ boxes (a real bug in this Ultralytics version, confirmed directly against this project's
  data). BoT-SORT also measurably outperformed ByteTrack on this footage (fewer fragmented IDs).
- **`jersey_ocr.py`'s `JerseyOCR` defaults to `gpu=True`** — meaningful runtime win on a full game
  (OCR runs on every Nth frame across the whole video), but means a GPU-less machine needs
  `JerseyOCR(gpu=False)` explicitly or every OCR call will error/fall back slowly. Check
  `torch.cuda.is_available()` first if setting up on new hardware.

## 5. Running scripts

Everything runs through the WSL venv, pointed at the Windows-side repo path:

```bash
wsl -d Ubuntu -- ~/soccer-analytics/.venv/bin/python /mnt/c/Users/<you>/GitHub/futbol-hub/test_tracker.py <video-path> [options]
```

From Git Bash on Windows specifically, prefix with `MSYS_NO_PATHCONV=1` or `/mnt/c/...`-style
paths get mangled by MSYS's automatic path translation:

```bash
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -- ~/soccer-analytics/.venv/bin/python /mnt/c/Users/<you>/GitHub/futbol-hub/test_shot_detection.py "..."
```

To run the **full** video rather than a sample window, pass `--center-seconds`/`--window-seconds`
covering the whole thing (e.g. half the video's duration and the full duration, respectively) -
there's no dedicated "whole video" flag.

## 6. VS Code setup (WSL-remote)

Native Windows Python doesn't work here (see above), so VS Code has to run **connected to the WSL
distro**, not as a plain local window - a local window's integrated terminal is `powershell.exe`
with a Windows-side Python on PATH, which is exactly the broken combination this whole doc exists
to avoid.

1. Install extensions: `ms-vscode-remote.remote-wsl` and `ms-python.python` (installs
   `ms-python.debugpy` and `ms-python.vscode-pylance` as dependencies).
2. Open the repo **through the WSL connection**, not as a plain local folder — `code --remote
   wsl+Ubuntu /mnt/c/Users/<you>/GitHub/futbol-hub`, or Command Palette → **"WSL: Reopen Folder in
   WSL"** if it's already open locally. Confirm this worked: a fresh integrated terminal
   (`` Ctrl+` ``) should default to `bash`, not PowerShell, with `pwd` under `/mnt/c/...`.
3. **Trust the workspace** when prompted (Restricted Mode blocks extension functionality,
   including the Python interpreter). Optionally turn this off entirely for less friction on future
   folders: user `settings.json` → `"security.workspace.trust.enabled": false`. Since Remote-WSL
   runs a *separate* extension host, this setting needs to be set on both sides if you want it
   everywhere - local Windows `settings.json` (`%APPDATA%\Code\User\settings.json`) and the WSL-side
   one (`~/.vscode-server/data/Machine/settings.json` inside the distro).
4. **Extensions installed locally do not carry over to the WSL-remote window** — they need
   installing again "into WSL" specifically. In the Extensions view, connected to the WSL window,
   look for an "Install in WSL: Ubuntu" affordance on each extension if it shows as local-only.
5. Command Palette → **"Python: Select Interpreter"** → **"Enter interpreter path..."** →
   `/home/YOURUSERNAME/soccer-analytics/.venv/bin/python`. [.vscode/settings.json](.vscode/settings.json)
   already sets this as the workspace default, but it may need picking manually on first connect.
6. [.vscode/launch.json](.vscode/launch.json) has a Run/Debug config per pipeline script
   (F5-able), each prompting for a video path at launch time.

A "this workspace is on the Windows file system" performance dialog is expected and safe to
dismiss permanently - the repo is deliberately kept on the Windows side (see step 3 above); only
the venv needs to be WSL-native.

## 7. Files not in this repo — copy these separately

- **`models/soccer_v1_best.pt`** and **`models/ball_v1_best.pt`** — trained model weights,
  gitignored (small, ~6MB each, but represent real training time — carry them over directly rather
  than retraining unless you actually want to).
- **`.env`** — needs `ROBOFLOW_API_KEY=<your key>` (get one free at roboflow.com), only required if
  you run `download_dataset.py` / `train.py` to rebuild or retrain the models. Not needed just to
  run the existing detection/tracking/OCR/shot-detection pipeline.
- **`models/action_classifier.joblib`** — the trained action-type classifier (see below), gitignored.
  Small and easy to regenerate (`train_action_classifier.py`), so worth copying over only if you
  don't want to redo that step; not required for the rest of the pipeline to run.
- **`shot_log*.json`** — per-run event logs from `test_shot_detection.py`, gitignored, regenerable
  by re-running the pipeline. `train_action_classifier.py` needs one of these present locally to
  join labels against trajectories.
- **`labeled_events_export.json`** — a snapshot of reviewed labels pulled from Match Event Log's
  live database (see below); gitignored since it goes stale the moment more reviewing happens.
  Regenerate by asking Claude to re-export it (only Claude's session has Artifact tool access to
  query that database directly - see the summary prompt for how to ask for this).

Video footage lives on OneDrive (`Videos/Soccer Videos/`) and is reachable from any machine signed
into the same Microsoft account — cloud-only files need to hydrate (fully download) before
OpenCV can read them; either wait for that or use `attrib +P <file>` to force it (needs an
elevated prompt; if that's unavailable, a plain read of the file, e.g. opening it once in any
player, also triggers hydration).

## Pipeline overview

In dependency order:

- `detector.py` — `CombinedDetector`: two fine-tuned YOLO models (player/goalkeeper/referee/ball,
  plus a ball specialist since the merged model's ball recall was too weak on its own) with
  BoT-SORT tracking.
- `jersey_ocr.py` — EasyOCR digit reading on player crops, majority-voted per track ID.
- `track_merger.py` — stitches fragmented tracking IDs back into one persistent identity when
  jersey number agrees and the timing/position is physically plausible.
- `goal_calibration.py` — loads a human-marked 4-point goal-mouth calibration (2D image-space
  only, valid only for the camera framing it was marked on). Also converts to real-world units:
  a calibration can carry a `format` (`"7v7"`/`"9v9"`/`"11v11"`) and/or an explicit
  `goal_width_feet`, used to compute `feet_per_pixel` from the goal's own measured pixel width -
  this is what lets distance/speed features mean the same thing across different cameras, zoom
  levels, and match formats instead of being raw-pixel numbers specific to one video.
- `shot_detector.py` — chains sparse ball detections into trajectories, flags on-target/off-target
  shot attempts, heuristically classifies outcome (goal/save/blocked/uncertain) with an explicit
  confidence level rather than forcing a guess. Known limitation: the "is this even a shot" gate is
  just a speed threshold near the goal, so passes/clearances/headers near goal get flagged too -
  that's what the action-type classifier below is for.
- `action_classifier.py` / `train_action_classifier.py` — a RandomForest prototype that classifies
  a ball-trajectory segment into an actual action type (shot/pass/cross/header/throw_in/
  tackle_or_block/dribble/not_a_ball_action) rather than assuming anything fast near the goal is a
  shot. Trains from human-reviewed labels (via Match Event Log, see below) joined against
  `shot_log*.json` trajectories by `video` + `start_frame`. As of this writing there are far too
  few labels (~10, almost all one class) for this to be a usable classifier yet - see the summary
  prompt for exactly where labeling stands. Usage:
  ```bash
  python train_action_classifier.py --labels labeled_events_export.json \
      --event-logs shot_log_full_game.json --model-out models/action_classifier.joblib
  ```
- `extract_event_clips.py` — pulls a short, padded video clip (ffmpeg trim) around each detected
  event's frame range, for reviewing in Match Event Log without hunting through raw frame numbers.
  Clips get uploaded as Artifact assets (only Claude's session can do this - see below) and linked
  into the corresponding event's `clip_url` field.

Two Artifact-based tools (published to claude.ai, not part of this repo's runtime — source copies
kept here for reference/editing, but the **live, working versions are the published URLs**, tied to
the account, not this machine):

- [`calibration_tool.html`](calibration_tool.html) → [Goal Line Calibration](https://claude.ai/code/artifact/e4e243df-4846-4e0b-9b3d-b096db1759e9)
  — click-to-mark goal calibration UI (writes to the `db` capability's `calibration/goal` doc; the
  version in this repo's `goal_calibration.json` was manually copied from there). Currently
  hardcoded to one video/frame/timestamp - calibrating a *new* video needs the tool generalized
  first (not done yet).
- [`event_log_tool.html`](event_log_tool.html) → [Match Event Log](https://claude.ai/code/artifact/f11b75b1-a9ee-48c2-ba89-e1bd51330879)
  — review/correct detected events (now against a SoccerNet-style action taxonomy: shot outcome,
  pass, cross, header, throw-in, tackle/block, dribble, not-a-ball-action) and manually log
  shot-attempt or restart (kickoff/goal-kick/corner/etc.) events as ground truth. Deliberately keeps
  "what ball action was this" separate from "what restart followed" — a restart only tells you
  which side touched the ball last, never whether a shot occurred. Declares the `db` and `assets`
  runtime capabilities (live shared database + uploaded video clips) - both are account-tied, so
  this tool and all its data are reachable from any device signed into the same claude.ai account,
  with no per-machine setup needed.
