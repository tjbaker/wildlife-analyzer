# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp config.sample.py config.py  # then edit: set GOOGLE_API_KEY, VIDEO_PATH, LOCATION_CONTEXT
```

ffprobe is required for the `archive` mode: `brew install ffmpeg`

## Running

The tool is a single CLI with five modes run in sequence:

```bash
python wildlife_analyzer.py extract   # Extract sharp frames from video
python wildlife_analyzer.py tune      # Auto-calibrate thresholds from manual sort (optional)
python wildlife_analyzer.py refine    # Find sharper versions of manual selections (optional)
python wildlife_analyzer.py analyze   # Identify species with Gemini API
python wildlife_analyzer.py archive   # Append session log to master CSV
```

There are no tests and no linting configuration.

## Architecture

All logic lives in `wildlife_analyzer.py` (~1700 lines). `config.py` (gitignored, copy from `config.sample.py`) drives all behavior.

**CONFIG class** (line ~46) loads `config.py` at import time with fallback defaults. It owns all directory paths (`BASE_DIR/good`, `blurry`, `rejected`, `manual`) and all tunable parameters.

**Frame extraction pipeline** (`extract_frames`):
1. OpenCV opens the video and steps at `SNAPSHOT_INTERVAL` seconds
2. Each frame is scored: `is_blurry()` (Laplacian variance), `has_content()` (Canny edge density), `detect_motion()` (optical flow + affine compensation to isolate independent motion from camera shake)
3. `SMART_EXTRACT=True` scans the entire second to keep the sharpest frame instead of a single grab; blur scoring is parallelized across CPU cores via a shared `ThreadPoolExecutor`
4. Frames land in `good/`, `blurry/`, or `rejected/`

**AI analysis pipeline** (`analyze_frames`):
1. Loads frames from `good/` + `manual/`; skips already-processed filenames from existing CSV (resume-safe)
2. For each frame, `get_context_frames()` pulls `CONTEXT_FRAMES` evenly-spaced frames from that video second and sends them all to Gemini in one call for temporal context
3. Parses JSON response; filters out `INANIMATE_OBJECTS` (rocks, sand, water, humans, generic "fish", etc.)
4. **Smart retry**: triggers when confidence < `RETRY_CONFIDENCE_THRESHOLD` *or* scientific name is unknown — finds the sharpest frame in that second via `find_best_frame_at_timestamp()`, re-analyzes, accepts only if retry returns a real scientific name with equal or better confidence; otherwise skips the frame
5. Parallel execution via `ThreadPoolExecutor` (controlled by `MAX_WORKERS`); thread-safe rate limiting via `_RateLimiter`
6. Writes to `sighting_log_<video>.csv` with a CSV lock; rebuilds `youtube_chapters.txt` from the full CSV at the end (so resumes produce complete chapter files)
7. Chapter deduplication is case-insensitive; common names are normalized per scientific name (most-frequent wins) to eliminate AI casing variation
8. **Consensus pass**: after main analysis, groups consecutive same-type frames with conflicting IDs and re-queries the API with all frames together — see below

**Consensus pass** (`run_consensus_pass`):
Fires after the main analysis loop when `CONSENSUS_WINDOW > 0`. Fixes the common case where the same animal (e.g. an eel) gets inconsistent scientific names across consecutive frames because each API call is independent.

Algorithm:
1. Groups single-species frames into tight sub-clusters (gap ≤ `SNAPSHOT_INTERVAL + 1` s, then capped at `CONSENSUS_WINDOW` s)
2. Finds the most-repeated genus within each cluster — if no genus repeats, the frames likely show different animals and are skipped
3. Extracts the dominant type-word from the common names of the genus-overlap frames (e.g. "eel" from "Goldspotted Snake Eel", "Sharptail Snake Eel")
4. Sends only frames whose common name contains that type-word — bystander frames (a Frogfish or Sea Cucumber that happens to appear in the time window) are excluded and left unchanged
5. The consensus prompt names the type-word explicitly: *"sequential frames of the same eel"*

**Auto-tuning** (`tune`): reads blur/content scores from `good/`, `blurry/`, `rejected/` folders, calculates optimal thresholds, and rewrites the threshold values in `config.py`.

**Archive** (`archive_log`): enriches the session CSV with video creation date (via ffprobe), location, and video name, then deduplicates and appends to `master_sighting_log.csv`.

## Key config knobs

| Setting | Effect |
|---|---|
| `BLUR_THRESHOLD` / `MIN_CONTENT_THRESHOLD` | Frame quality filters — auto-tuned by `tune` mode |
| `MOTION_THRESHOLD` | Fraction of pixels that must change (0.10 = 10%) |
| `SMART_EXTRACT` | Scan full second for sharpest frame (slower, better quality) |
| `CONTEXT_FRAMES` | Number of frames sent per Gemini call (3 = temporal context, 1 = faster) |
| `SMART_RETRY` | Re-query Gemini with a better frame when confidence is low or scientific name is unknown |
| `RETRY_CONFIDENCE_THRESHOLD` | Confidence below which Smart Retry fires (default 0.60) |
| `CONFIDENCE_THRESHOLD` | Minimum confidence to appear in `youtube_chapters.txt` (default 0.80) |
| `CONSENSUS_WINDOW` | Seconds window for consensus pass; 0 = disabled (default 5) |
| `MAX_WORKERS` | Parallel analysis threads (1 = safe for free tier, 5 = paid tier) |
| `MODEL_NAME` | Gemini model — `gemini-2.0-flash` (fast/cheap) or `gemini-2.5-flash` (slower/more accurate) |
| `ANALYSIS_PROMPT` | Full Gemini prompt — customizable AI persona and instructions |
| `INANIMATE_OBJECTS` | Filter list — common names matched here are silently skipped |
| `LOG_LEVEL` | `DEBUG` for verbose output, `INFO` for normal |
| `API_DELAY` | Increase if hitting Gemini rate limits (429 errors) |

## See also

`NOTES.md` — running log of experiments, dead ends, and ideas for future work.
