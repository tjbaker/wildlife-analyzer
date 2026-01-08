# Wildlife Video Analyzer 🦁🦅🐠

An intelligent, AI-powered tool for analyzing video footage to identify and log wildlife species (aquatic or terrestrial).

Built with **Python**, **OpenCV**, and **Google Gemini 2.0 Flash**.

## Features

- **Smart Extraction**: Automatically finds the sharpest frames in your video, filtering out blurry or empty shots.
- **Auto-Tuning**: Intelligently adjusts blur and content thresholds based on your manual feedback.
- **AI Identification**: Uses Google's Gemini 2.0 Vision model to identify species with high accuracy.
- **Smart Retry**: If the AI is unsure (<85% confidence), the system automatically re-scans the video source for a clearer angle and retries.
- **Inanimate Filtering**: Automatically ignores rocks, sand, and water to keep your logs clean.
- **Master Log Archiving**: Consolidate sessions into a single `master_sighting_log.csv` with location and date metadata.
- **YouTube Chapters**: Generates improved timestamps and species names ready for video descriptions.

## Usage

### 1. Setup

1.  **Clone the repo** (or download files).
2.  **Set up Virtual Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configure**:
    ```bash
    cp config.sample.py config.py
    ```
    *   **Get your API Key**:
        1.  Go to [Google AI Studio](https://aistudio.google.com/app/apikey).
        2.  Click **Create API key**.
        3.  Select a project (or create a new one) and copy the key.
    *   Edit `config.py` and paste your `GOOGLE_API_KEY`.
    *   Update `VIDEO_PATH` to point to your video file.

### 2. Workflow

The tool operates in three simple modes:

#### Step 1: Extract
Scans your video and saves potential frames to `extracted_images/good`.
```bash
python wildlife_analyzer.py extract
```

#### Step 2: Tune (Optional but Recommended)
1.  Manually review `extracted_images/good`. Move any blurry or empty images to `extracted_images/blurry` or `extracted_images/rejected`.
    *   **Tip**: You can also check `extracted_images/blurry` and `extracted_images/rejected` and move any valid images *back* into `good`. The tuner learns from these corrections too!
2.  Run the tuner:
    ```bash
    python wildlife_analyzer.py tune
    ```
    This will analyze your sorting choices and automatically update `wildlife_config.py` (or `config.py`) with the perfect sensitivity settings for your specific video conditions.

#### Step 2.5: Refine (Optional)
If you manually move images or want to improve the quality of specific frames:
1.  Move any "almost good" images from `extracted_images/blurry` to `extracted_images/manual` (or just leave them in `good` if you moved them there).
2.  Run the refiner:
    ```bash
    python wildlife_analyzer.py refine
    ```
    This re-scans the video for the files in your manual folder to ensure you have the sharpest possible version of that specific moment.

#### Step 3: Analyze
Sends the approved frames to Gemini for identification.
```bash
python wildlife_analyzer.py analyze
```
*   **Outputs**:
    *   `sighting_log_[video_name].csv`: Comprehensive spreadsheet of all sightings in this session.
    *   `youtube_chapters.txt`: Timestamped list for video publishing.

#### Step 4: Archive
When finished with a session, add it to your lifetime master log.
```bash
python wildlife_analyzer.py archive
```
*   **Appends to**: `master_sighting_log.csv`
*   **Adds Metadata**: Automatically adds Date (from video file) and Location (from config) to every entry.

## Configuration (`config.py`)

| Key | Description | Default |
| :--- | :--- | :--- |
| `VIDEO_PATH` | Path to your source video file. | `"wildlife_video.mp4"` |
| `LOCATION_CONTEXT` | Helps AI identify region-specific species. | `"Backyard, New England"` |
| `MODEL_NAME` | The Gemini model to use. | `"gemini-2.0-flash"` |
| `API_DELAY` | Seconds to wait between API calls (prevents 429 errors). | `2.0` |
| `LOG_LEVEL` | Controls output verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`). | `"INFO"` |
| `SMART_EXTRACT` | If `true`, scans the full second to find the sharpest frame (slower but better). | `false` |
| `SMART_RETRY` | If `true`, re-scans video when AI confidence is low. | `true` |
| `BLUR_THRESHOLD` | Laplacian variance threshold (Auto-tuned). | `100.0` |
| `MIN_CONTENT_THRESHOLD` | Edge density threshold (Auto-tuned). | `0.005` |
| `CONFIDENCE_THRESHOLD` | Minimum confidence to log chapter. | `0.70` |

### Custom Analysis Prompts (`ANALYSIS_PROMPT`)

You can customize the AI persona and instructions in `config.py` to suit different types of videos.

#### Example: Bird Watching 🦅
```python
ANALYSIS_PROMPT = (
    "You are an Ornithologist. Location: {location}. {date} "
    "Identify the bird species in this image. "
    "Pay attention to plumage patterns, beak shape, and size. "
    "..."
)
```

#### Example: Safari / Wildlife 🐘
```python
ANALYSIS_PROMPT = (
    "You are a Safari Guide. Location: {location}. {date} "
    "Identify the animals in this shot. "
    "Look for camouflaged animals in the grass or trees. "
    "..."
)
```

### Output Examples

#### 1. Sighting Log (`sighting_log.csv`)
The session log contains every identified frame.
```csv
"timestamp","file","common_name","scientific_name","confidence"
"00:01","frame_00m_01s.jpg","Long-spined Sea Urchin","Diadema antillarum","0.90"
"00:29","frame_00m_29s.jpg","Brown Chromis","Chromis multilineata","0.80"
"00:33","frame_00m_33s.jpg","Blue Chromis","Chromis cyanea","0.90"
```

#### 2. Master Log (`master_sighting_log.csv`)
The master log aggregates all sessions with metadata.
```csv
"date","location","video_name","timestamp","common_name","scientific_name","confidence"
"2025-12-21","St. Lucia","dive_trip.mp4","00:01","Fairy Basslet","Gramma loreto","0.95"
"2025-12-21","St. Lucia","dive_trip.mp4","00:05","Blue Tang","Acanthurus coeruleus","0.92"
```

#### 3. YouTube Chapters (`youtube_chapters.txt`)
Ready-to-paste timestamps for video descriptions.
```text
00:01 - Long-spined Sea Urchin (Diadema antillarum) [0.90]
00:29 - Brown Chromis (Chromis multilineata) [0.80]
00:33 - Blue Chromis (Chromis cyanea) [0.90]
```
**How to use:**
1.  Open your video in YouTube Studio.
2.  Paste the contents of `youtube_chapters.txt` into the **Description** field.
3.  YouTube will automatically generate clickable chapters in the video timeline.


## Logging

You can control the verbosity of the output by changing `LOG_LEVEL` in `config.py`.
*   **INFO** (Default): Standard progress updates and essential information.
*   **DEBUG**: Detailed logs including API request/response times and internal logic decisions. Useful for troubleshooting.
*   **WARNING/ERROR**: Only verification warnings or critical errors.

## Troubleshooting

### "429 Resource Exhausted" Errors
If you see frequent 429 errors, it means you're hitting the free tier rate limits of the Gemini API.
*   **Increase Delay**: Increase `API_DELAY` in `config.py` (e.g., to 4.0 or 5.0 seconds).
*   **Upgrade Plan**: Consider enabling billing on your Google Cloud project to access higher rate limits (Paid Tier).

## License

Apache License, Version 2.0.
