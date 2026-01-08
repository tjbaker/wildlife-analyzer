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

# GOOGLE GEMINI API KEY
# Required for the 'analyze' step.
# You can get a free key at: https://aistudio.google.com/app/apikey
GOOGLE_API_KEY = "YOUR_API_KEY_HERE"

# GEMINI MODEL NAME
# The model to use for analysis.
# Options:
#   - "gemini-2.0-flash" (Fast, low cost)
#   - "gemini-2.0-pro"   (Higher reasoning)
#   - "gemini-3-flash-preview" (Newest, experimental)
# Default: "gemini-2.0-flash"
MODEL_NAME = "gemini-2.0-flash"

# API RATE LIMITING
# Delay in seconds between API calls to avoid hitting rate limits (429).
# Increase this if you see frequent "RESOURCE_EXHAUSTED" errors.
API_DELAY = 2.0

# LOG LEVEL
# Controls verbosity of the output.
# Options: "DEBUG", "INFO", "WARNING", "ERROR"
LOG_LEVEL = "INFO"


# ==============================================================================
# 2. EXTRACTION SETTINGS (Finding the images)
# ==============================================================================

# SNAPSHOT INTERVAL (Seconds)
# How often to grab a frame from the video.
#   1 = Check every second (High detail, more images)
#   5 = Check every 5 seconds (Faster, fewer images)
SNAPSHOT_INTERVAL = 1

# SMART EXTRACT (High Quality Mode)
#   False (Default): Grabs exactly one frame at the interval. Fast.
#   True           : Scans the *entire* second around the interval to find
#                    the absolutely sharpest frame. Slower, but better quality.
SMART_EXTRACT = False

# BLUR THRESHOLD (Auto-Tuned)
# A measure of image sharpness (Laplacian Variance).
#   Higher = Stricter (requires sharper images).
#   Lower  = Lenient (allows more blur).
# Run 'python wildlife_analyzer.py tune' to set this automatically.
BLUR_THRESHOLD = 100.0

# MIN CONTENT THRESHOLD (Auto-Tuned)
# A measure of 'stuff' in the image (Edge Density).
#   Higher = Stricter (requires complex scenes/animals).
#   Lower  = Lenient (allows empty blue water).
# Run 'python wildlife_analyzer.py tune' to set this automatically.
MIN_CONTENT_THRESHOLD = 0.005
MOTION_THRESHOLD = 0.10 # 10% of pixels must change (Ignores minor shake)


# ==============================================================================
# 3. ANALYSIS SETTINGS (Identifying the fish)
# ==============================================================================

# CONFIDENCE THRESHOLD
# How sure the AI must be to log a species in the 'YouTube Chapters' file.
#   0.70 = 70% Confidence.
# Range: 0.0 to 1.0.
CONFIDENCE_THRESHOLD = 0.70

# SMART RETRY (AI Double-Check)
# If True, when the AI is unsure (confidence < RETRY_CONFIDENCE_THRESHOLD),
# the script will pause, look at the video again to find a slightly different
# angle or sharper frame from that same moment, and ask the AI again.
# Significantly improves results for tricky shots.
SMART_RETRY = True

# RETRY THRESHOLD
# The confidence level that triggers a Smart Retry.
# If AI confidence is below this, we try again.
RETRY_CONFIDENCE_THRESHOLD = 0.85

# ANALYSIS PROMPT
# The instructions sent to Gemini.
# You can use {location} and {date} as placeholders.
ANALYSIS_PROMPT = (
    "You are a Marine Biologist. Location: {location}. {date} "
    "Identify the marine animal species in this image. "
    "Consider the location and date (seasonality) to identify species present in this region at this time. "
    "Only identify animals that are CLEARLY VISIBLE. Do not guess based on shadows, shapes, or blurry blobs. "
    "If the image contains only background (rocks, sand, water, coral rubble) or ambiguous shapes, return null for names. "
    "Return valid JSON only: "
    "{{'common_name': 'Common Name' or null, 'scientific_name': 'Scientific Name' or null, 'confidence': 0.0-1.0}}."
)
