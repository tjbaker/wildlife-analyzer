# Copyright 2025 Trevor Baker, all rights reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# SPDX-License-Identifier: Apache-2.0

"""
WILDLIFE ANALYZER CONFIGURATION
=============================

This file controls the behavior of the wildlife video analysis tool.
Copy this file to 'config.py' to use it.

"""

# ==============================================================================
# 1. PROJECT SETTINGS
# ==============================================================================

# THE VIDEO FILE
# Path to the source video file you want to analyze.
# Examples: "birdwatch.mp4", "/Users/me/movies/safari.mov"
VIDEO_PATH = "wildlife_video.mp4"

# OUTPUT DIRECTORY
# The root folder where all results will be stored.
# The tool will create subfolders:
#   - /good      : Sharp, interesting frames (used for analysis)
#   - /blurry    : Frames rejected because they were too blurry
#   - /rejected  : Frames rejected because they were empty (no edges/content)
#   - /manual    : Folder for you to manually force-include images
BASE_DIR = "extracted_images"

# LOCATION CONTEXT
# Providing the location helps the AI (Gemini) identify species correctly.
# E.g., "Backyard, New England", "Serengeti", "Amazon Rainforest".
LOCATION_CONTEXT = "St. Lucia, Caribbean"

# AI PROVIDER
#   "gemini"     — Google Gemini (default)
#   "anthropic"  — Anthropic Claude
PROVIDER = "gemini"

# API KEYS — set the one matching your provider
GOOGLE_API_KEY = "YOUR_API_KEY_HERE"
ANTHROPIC_API_KEY = "YOUR_API_KEY_HERE"

# MODEL NAME — must match the selected provider
#   Gemini:     "gemini-2.0-flash" (fast/cheap) | "gemini-2.5-flash" (more accurate)
#   Anthropic:  "claude-haiku-4-5" (fast/cheap) | "claude-sonnet-4-6" (more accurate)
MODEL_NAME = "gemini-2.0-flash"

# SESSION TAG — optional suffix appended to the log filename (e.g. "_test", "_anthropic")
# Useful for side-by-side comparisons without overwriting existing logs.
SESSION_TAG = ""

# API RATE LIMITING — increase if you see 429 / RESOURCE_EXHAUSTED errors
API_DELAY = 2.0

# PARALLEL WORKERS — 1 = safe for free tier (15 RPM); 5 = paid tier
MAX_WORKERS = 1

# CONTEXT FRAMES — frames sent per Gemini call; 3 = temporal context, 1 = faster
CONTEXT_FRAMES = 3

# LOG LEVEL — DEBUG for verbose output, INFO for normal
LOG_LEVEL = "INFO"


# ==============================================================================
# 2. EXTRACTION SETTINGS (Finding the images)
# ==============================================================================

# How often to grab a frame from the video (seconds)
SNAPSHOT_INTERVAL = 1

# SMART EXTRACT — scan full second for sharpest frame (True = better quality, slower)
SMART_EXTRACT = True

# BLUR / CONTENT THRESHOLDS — auto-tuned by `tune` mode; rarely set manually
BLUR_THRESHOLD = 100.0
MIN_CONTENT_THRESHOLD = 0.005
MOTION_THRESHOLD = 0.10


# ==============================================================================
# 3. ANALYSIS SETTINGS (Identifying species)
# ==============================================================================

# Minimum confidence to appear in youtube_chapters.txt (0.0–1.0)
CONFIDENCE_THRESHOLD = 0.70

# SMART RETRY — re-query with a sharper frame when confidence is low or scientific name unknown
SMART_RETRY = True
RETRY_CONFIDENCE_THRESHOLD = 0.60

# Inanimate Filter
# The AI will ignore objects with these names.
INANIMATE_OBJECTS = ["none", "null", "unknown", "rock", "sand", "gravel", "water", "coral rubble",
                     "reef fish", "fish", "marine fish", "tropical fish", "saltwater fish",
                     "human", "person", "snorkeler", "diver"]

# ANALYSIS PROMPT
# The instructions sent to Gemini.
# Placeholders: {location}, {date}, {n} (number of context frames).
ANALYSIS_PROMPT = (
    "You are a field biologist. Location: {location}. {date}"
    "Examine these {n} sequential video frames from the same moment. "
    "Identify ALL distinct living creatures that are CLEARLY VISIBLE across the frames. "
    "Only identify species plausible for this location and environment — "
    "never identify a freshwater species in ocean footage, or a terrestrial species underwater. "
    "When uncertain between two species, use location to break the tie. "
    "Do not guess based on shadows, blur, or ambiguous shapes. "
    "Return ONLY valid JSON with no markdown — an array of objects, one per species. "
    "Return an empty array [] if only background is visible (water, rocks, sand, coral, foliage). "
    '[{{"common_name": "string", "scientific_name": "string", '
    '"confidence": 0.0-1.0, "notes": "brief observation"}}]'
)
