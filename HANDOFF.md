# Handoff summary — futbol-hub soccer analytics project

*Paste this whole file's contents into a new Claude Code session on the other device to bring it
up to speed. Written 2026-09-11.*

## What this project is

A computer-vision pipeline for analyzing amateur/youth soccer match video: player/ball detection +
tracking (YOLO + BoT-SORT), jersey number OCR, goal-mouth calibration, and heuristic shot-on-goal
detection. The end goal is a broader analytics tool usable across **different match formats**
(7v7/9v9/11v11) and **different camera setups**, not just the one test video worked with so far.

Repo: `bphillips129-git/futbol-hub` on GitHub, main branch.

## Environment: read SETUP.md first

This only runs inside WSL2 Ubuntu — native Windows Python is blocked by Smart App Control on this
family of machines. [SETUP.md](SETUP.md) is fully up to date as of this handoff (WSL setup, apt
gotchas, opencv-python/headless conflict, VS Code Remote-WSL setup including the workspace-trust
and extension-installation quirks, and how to run everything). Follow it in order on the new
device; it was written from a completed real bootstrap, not from memory.

**Files to copy manually onto the new device** (all gitignored, not in the repo):
- `models/soccer_v1_best.pt` and `models/ball_v1_best.pt` — trained weights, ~6MB each
- `.env` with `ROBOFLOW_API_KEY` — only needed for `download_dataset.py`/`train.py`, skip if not
  retraining
- Everything else regenerable (see SETUP.md's "Files not in this repo" section) — not worth
  carrying over

Video footage lives on OneDrive (`Videos/Soccer Videos/`), reachable from any machine signed into
the same Microsoft account once files hydrate.

## Where things actually stand

1. **Core pipeline works end-to-end** — ran the full detection/tracking/OCR/shot-detection chain
   against a complete ~30 min game (`WIVY3619.MP4`, a U11 9v9 match) with GPU acceleration. Produced
   73 candidate shot events.
2. **The naive shot heuristic over-flags heavily.** `shot_detector.py`'s "is this a shot" gate is
   just "fast ball movement near the goal" — every reviewed candidate so far has turned out to be a
   pass, clearance, foul, or goalkeeper retrieval, not a shot. This is the core problem being worked
   on, not a bug.
3. **Built an action-type classifier prototype** (`action_classifier.py` / `train_action_classifier.py`)
   to fix that: a RandomForest over hand-engineered trajectory features (speed, direction-to-goal,
   straightness, distance-to-goal, etc.), trained from human-reviewed labels. **Status: not usable
   yet** — only ~10 labeled examples exist, almost all the same class ("not a shot"). It proves the
   label→train→predict pipeline works, nothing more. Needs a real spread of labels across
   shot/pass/cross/header/throw_in/tackle_or_block/dribble before it means anything.
4. **Features are now real-world units (feet, feet/second), not raw pixels/frames** — added
   `format`-aware (7v7/9v9/11v11) goal-width lookup and `feet_per_pixel` conversion to
   `goal_calibration.py` specifically so labeled data and a trained classifier could eventually
   generalize across different cameras and match formats, not just this one video's framing. This
   was a direct response to the user flagging that goal-to-field proportions genuinely differ by
   format, not just camera zoom.
5. **Two Artifact tools are published and live** (account-tied — reachable from any device signed
   into the same claude.ai account, no per-machine setup):
   - [Goal Line Calibration](https://claude.ai/code/artifact/e4e243df-4846-4e0b-9b3d-b096db1759e9) —
     currently hardcoded to one video/frame; generalizing it to calibrate a *new* video is an open
     task for whenever a new video/format actually gets calibrated.
   - [Match Event Log](https://claude.ai/code/artifact/f11b75b1-a9ee-48c2-ba89-e1bd51330879) —
     review tool for the 73 detected candidates + manual ground-truth logging. Recently overhauled:
     categories now mirror the SoccerNet Ball Action Spotting taxonomy (shot outcome / pass / cross
     / header / throw_in / tackle_or_block / dribble / not_a_ball_action) instead of shot-outcome-only,
     and all 62 still-pending candidates now have an embedded, playable video clip (2.5s padding)
     so reviewing doesn't mean guessing from a bare timestamp. Earlier had a "stuck on Loading"
     bug that traced to a stale runtime contract pin — upgraded to latest; should be resolved but
     wasn't re-confirmed after the fix.

## Immediate next step

Keep reviewing pending candidates in Match Event Log, but the priority now is **label diversity**,
not volume — deliberately picking a spread across pass/header/cross/throw_in/dribble/tackle_or_block,
not just confirming more "not a shot"s. Once there's a real spread (~8-10 examples across 4+
classes), ask Claude to:
1. Re-export current labels from the live database (only Claude's session has Artifact tool access
   to query it — that's why `labeled_events_export.json` isn't just checked into the repo)
2. Re-run `train_action_classifier.py`
3. If results look decent, wire the classifier into `test_shot_detection.py` as a pre-filter so the
   existing shot-outcome logic only runs on candidates actually classified as shots

## Longer-term open items (not started)

- Generalize `calibration_tool.html` to calibrate a new video (currently one-off, hardcoded)
- Add player-proximity features to the classifier (was a player stationary vs. sprinting at the
  event's start) — `test_shot_detection.py` doesn't currently persist player positions per event,
  which caps classifier accuracy until it does
- Everything here has only ever been run against one video of one format (9v9) — the multi-format
  design (real-world units, format-aware calibration) is untested against an actual second format
  until a 7v7 or 11v11 video gets run through it
