# Wildlife Video Analyzer

An AI-powered tool for identifying and logging wildlife species from video footage — aquatic or terrestrial.

Built with **Python**, **OpenCV**, and **Google Gemini** or **Anthropic Claude**.

## Features

- **Smart Extraction** — finds the sharpest frame each second, filters blurry and empty shots
- **Dual AI Backend** — supports Google Gemini and Anthropic Claude; switch with one config line
- **Smart Retry** — when confidence is low (<60%) or scientific name is unknown, re-scans the video for a clearer frame and retries
- **Location-aware ID** — prompt enforces environment plausibility (won't identify freshwater species in ocean footage)
- **Auto-Tuning** — calibrates blur/content thresholds from your manual sorting feedback
- **Inanimate Filtering** — ignores rocks, sand, water, and generic "fish" to keep logs clean
- **Master Log** — consolidates sessions into `master_sighting_log.csv` with location and date metadata
- **Consensus Pass** — after analysis, re-queries the API with groups of consecutive frames that have conflicting IDs for the same animal type, producing a single consistent identification
- **YouTube Chapters** — generates timestamped species list ready to paste into video descriptions

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp config.sample.py config.py
```

Edit `config.py`:
- `VIDEO_PATH` — path to your video file
- `LOCATION_CONTEXT` — e.g. `"St. Lucia, Caribbean"` (helps AI identify regional species)
- `PROVIDER` — `"gemini"` (default) or `"anthropic"`
- API key for your chosen provider (see below)

`ffprobe` is used in `archive` and `analyze` modes to read the video creation date. Without it, date metadata will be missing. Install via: `brew install ffmpeg`

## Choosing a provider

All models were tested head-to-head on the same 56 frames of Caribbean snorkeling footage (St. Lucia, Dec 2025). Results below reflect real-world accuracy, not benchmarks.

| Provider | Model | Speed | Cost | Recommendation |
| :--- | :--- | :--- | :--- | :--- |
| Google Gemini | `gemini-2.5-flash` | ~9s/frame | Low | ✅ **Best overall** |
| Google Gemini | `gemini-2.0-flash` | ~3s/frame | Very low | Good for large batches |
| Anthropic | `claude-sonnet-4-6` | ~8s/frame | Medium | Comparable to Gemini, see notes |
| Anthropic | `claude-haiku-4-5` | ~3s/frame | Low | ❌ Not recommended |

### Findings

**Gemini 2.5 Flash** produced the cleanest results after manual frame curation — correct species, appropriate confidence scores, and good biodiversity coverage (Sergeant Major, Goatfish, Royal Gramma, Doctorfish). One known weakness: occasionally identifies freshwater species in marine footage when frames are ambiguous. The updated location-awareness prompt reduces this but doesn't eliminate it entirely — manual review of low-confidence frames is recommended.

**Claude Sonnet 4.6** showed comparable confidence calibration to Gemini but misidentified rock/coral texture as Spotted Scorpionfish (*Scorpaena plumieri*) across four consecutive frames — a consistent false positive that Gemini correctly identified as Sergeant Major. Sonnet may perform better on cleaner footage or different habitat types; not enough data to rule it out.

**Claude Haiku 4.5** is not suitable for species identification. It produced plausible-sounding but incorrect IDs (confirmed Caribbean Reef Octopus hallucination on a frame containing only Sergeant Major), and its lower confidence scores don't reliably indicate actual uncertainty. The cost saving over Sonnet is not worth the accuracy loss.

**Manual curation before analysis is strongly recommended** regardless of model. Removing empty water, rock, and low-quality frames before running `analyze` eliminates the majority of false positives and reduces API cost. In testing, culling 57% of frames (56 → 23) produced a 100% hit rate with no false positives.

Get a Gemini key at [Google AI Studio](https://aistudio.google.com/app/apikey) (free tier available).  
Get an Anthropic key at [console.anthropic.com](https://console.anthropic.com) (pay-as-you-go, ~$0.001/frame with Haiku, ~$0.01/frame with Sonnet).

## Workflow

### 1. Extract
```bash
python wildlife_analyzer.py extract
```
Scans the video and saves sharp, interesting frames to `extracted_images/good/`.

### 2. Curate *(recommended)*
After extraction, `good/` contains frames that passed the automated filters, and `blurry/` / `rejected/` contain the ones that didn't. Manually review `good/` in Finder and remove frames you don't want analyzed (empty water, rocks, bad angles). You can also promote frames from `blurry/` or `rejected/` back to `good/` if they contain something useful.

**One clean frame per continuous sighting is enough.** The YouTube chapters builder only writes a new entry when a species *reappears* after being absent — 20 consecutive frames of the same animal produce exactly one chapter line. Pruning redundant frames before `analyze` reduces API cost with no loss of output quality.

This step has the single biggest impact on result quality.

### 3. Tune *(optional)*
If you've done significant manual sorting, run the tuner to recalibrate thresholds for your video conditions:
```bash
python wildlife_analyzer.py tune
```
The tuner reads score distributions across `good/`, `blurry/`, and `rejected/` — don't delete frames from those folders before running it, as that skews the calibration.

### 4. Refine *(optional)*
Move any "almost good" frames to `extracted_images/manual/`, then:
```bash
python wildlife_analyzer.py refine
```
Re-scans the video at each manual frame's timestamp to find the sharpest version.

### 5. Analyze
```bash
python wildlife_analyzer.py analyze
```
Sends frames to the configured AI provider for identification. Outputs:
- `sighting_log_[video_name].csv` — all sightings for this session
- `youtube_chapters.txt` — timestamped species list

### 6. Archive
```bash
python wildlife_analyzer.py archive
```
Enriches the session log with date (from video metadata) and location, then appends to `master_sighting_log.csv`. Deduplicates on video + timestamp.

## Configuration (`config.py`)

| Key | Description | Default |
| :--- | :--- | :--- |
| `VIDEO_PATH` | Path to source video | `"wildlife_video.mp4"` |
| `LOCATION_CONTEXT` | Region context for species ID | `"Backyard, New England"` |
| `PROVIDER` | AI backend: `"gemini"` or `"anthropic"` | `"gemini"` |
| `MODEL_NAME` | Model for the selected provider | `"gemini-2.0-flash"` |
| `GOOGLE_API_KEY` | Gemini API key | — |
| `ANTHROPIC_API_KEY` | Anthropic API key | — |
| `SESSION_TAG` | Suffix appended to log filename (e.g. `"_test"`) — useful for side-by-side model comparisons | `""` |
| `API_DELAY` | Seconds between API calls (increase if seeing 429 errors) | `2.0` |
| `MAX_WORKERS` | Parallel analysis threads — keep at 1 on free-tier API plans to avoid 429 errors | `1` |
| `LOG_LEVEL` | Output verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) | `"INFO"` |
| `SMART_EXTRACT` | Scan full second for sharpest frame (slower, better quality) | `True` |
| `SMART_RETRY` | Re-query with a better frame when confidence is low | `True` |
| `BLUR_THRESHOLD` | Laplacian variance cutoff (auto-tuned) | `100.0` |
| `MIN_CONTENT_THRESHOLD` | Edge density cutoff (auto-tuned) | `0.005` |
| `CONFIDENCE_THRESHOLD` | Minimum confidence to include in YouTube chapters | `0.80` |
| `RETRY_CONFIDENCE_THRESHOLD` | Confidence below which Smart Retry fires | `0.60` |
| `CONTEXT_FRAMES` | Frames sent per AI call (3 = temporal context, 1 = faster) | `3` |
| `CONSENSUS_WINDOW` | Seconds window for post-analysis consensus pass; groups consecutive same-type frames with conflicting IDs and re-queries with all frames together; `0` = disabled | `5` |

### Custom Prompts

Override `ANALYSIS_PROMPT` in `config.py` to change the AI persona or focus. Use `{location}`, `{date}`, and `{n}` as placeholders:

```python
ANALYSIS_PROMPT = (
    "You are an ornithologist. Location: {location}. {date}"
    "Examine these {n} frames. Identify ALL bird species clearly visible — pay attention to "
    "plumage, beak shape, and size. Only identify species plausible for this location. "
    "Return ONLY valid JSON with no markdown — an array of objects, one per species. "
    "Return an empty array [] if no birds are visible. "
    '[{{"common_name": "string", "scientific_name": "string", '
    '"confidence": 0.0-1.0, "notes": "brief observation"}}]'
)
```

## Output Examples

**`sighting_log_[video].csv`**
```csv
"timestamp","file","common_name","scientific_name","confidence"
"00:01","frame_00m_01s.jpg","Long-spined Sea Urchin","Diadema antillarum","0.90"
"00:29","frame_00m_29s.jpg","Brown Chromis","Chromis multilineata","0.80"
```

**`master_sighting_log.csv`**
```csv
"date","location","video_name","timestamp","common_name","scientific_name","confidence"
"2025-12-21","St. Lucia","dive_trip.mp4","00:01","Fairy Basslet","Gramma loreto","0.95"
```

**`youtube_chapters.txt`** — paste into YouTube Studio description field; YouTube auto-generates clickable chapters.
```
00:01 - Long-spined Sea Urchin (Diadema antillarum) [0.90]
00:29 - Brown Chromis (Chromis multilineata) [0.80]
```

## License

Apache License, Version 2.0.
