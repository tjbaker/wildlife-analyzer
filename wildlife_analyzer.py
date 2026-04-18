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

import cv2
import os
import numpy as np
import sys
import glob
import json
import time
import datetime
import argparse
import certifi
import shutil
import threading
import base64
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import groupby
from tqdm import tqdm
from google import genai
from google.genai import types
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    retry_if_exception,
)
from google.api_core import exceptions
import ast
import re
import PIL.Image
import csv
import subprocess
import logging

try:
    import anthropic as _anthropic
except ImportError:
    _anthropic = None

# Fix SSL on Mac
os.environ["SSL_CERT_FILE"] = certifi.where()


class CONFIG:
    # Default values
    VIDEO_PATH = "wildlife_video.mp4"
    SNAPSHOT_INTERVAL = 1  # Seconds between checks
    BLUR_THRESHOLD = 100.0  # Variance of Laplacian. < 100 = blurry
    MODEL_NAME = "gemini-2.0-flash"

    # Directory Structure
    BASE_DIR = "extracted_images"
    FRAMES_DIR = os.path.join(BASE_DIR, "good")
    REJECTED_DIR = os.path.join(BASE_DIR, "rejected")
    BLURRY_DIR = os.path.join(BASE_DIR, "blurry")
    MANUAL_DIR = os.path.join(BASE_DIR, "manual")

    LOCATION_CONTEXT = "Backyard, New England"
    CONFIDENCE_THRESHOLD = 0.70
    # Fraction of pixels that must be edges (0.5%)
    MIN_CONTENT_THRESHOLD = 0.005
    MOTION_THRESHOLD = 0.10  # Reduced sensitivity. 10% change required.
    COST_PER_IMAGE = 0.0001
    SMART_EXTRACT = True

    # Smart Retry Config
    SMART_RETRY = True
    RETRY_CONFIDENCE_THRESHOLD = 0.60

    # Inanimate Filter
    INANIMATE_OBJECTS = [
        "none",
        "null",
        "unknown",
        "rock",
        "sand",
        "gravel",
        "water",
        "coral rubble",
        "reef fish",
        "fish",
        "marine fish",
        "tropical fish",
        "saltwater fish",
        "human",
        "person",
        "snorkeler",
        "diver",
    ]

    API_DELAY = 2.0
    LOG_LEVEL = "INFO"
    MAX_WORKERS = 1
    CONTEXT_FRAMES = 3

    # Provider
    PROVIDER = "gemini"  # "gemini" or "anthropic"
    ANTHROPIC_API_KEY = ""
    SESSION_TAG = ""  # appended to log filename, e.g. "_anthropic_test"

    # Default Prompt
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

    @classmethod
    def load(cls):
        try:
            import sys
            import importlib

            if os.getcwd() not in sys.path:
                sys.path.append(os.getcwd())

            config_name = os.environ.get("WILDLIFE_CONFIG", "config")
            user_config = importlib.import_module(config_name)
            importlib.reload(user_config)

            cls.VIDEO_PATH = getattr(user_config, "VIDEO_PATH", cls.VIDEO_PATH)
            cls.SNAPSHOT_INTERVAL = getattr(
                user_config, "SNAPSHOT_INTERVAL", cls.SNAPSHOT_INTERVAL
            )
            cls.BLUR_THRESHOLD = getattr(
                user_config, "BLUR_THRESHOLD", cls.BLUR_THRESHOLD
            )

            # Load Base Dir and update paths
            cls.BASE_DIR = getattr(user_config, "BASE_DIR", cls.BASE_DIR)
            cls.FRAMES_DIR = os.path.join(cls.BASE_DIR, "good")
            cls.REJECTED_DIR = os.path.join(cls.BASE_DIR, "rejected")
            cls.BLURRY_DIR = os.path.join(cls.BASE_DIR, "blurry")
            cls.MANUAL_DIR = os.path.join(cls.BASE_DIR, "manual")

            cls.LOCATION_CONTEXT = getattr(
                user_config, "LOCATION_CONTEXT", cls.LOCATION_CONTEXT
            )
            cls.CONFIDENCE_THRESHOLD = getattr(
                user_config, "CONFIDENCE_THRESHOLD", cls.CONFIDENCE_THRESHOLD
            )
            cls.MIN_CONTENT_THRESHOLD = getattr(
                user_config, "MIN_CONTENT_THRESHOLD", cls.MIN_CONTENT_THRESHOLD
            )
            cls.MOTION_THRESHOLD = getattr(
                user_config, "MOTION_THRESHOLD", cls.MOTION_THRESHOLD
            )
            cls.SMART_EXTRACT = getattr(user_config, "SMART_EXTRACT", cls.SMART_EXTRACT)

            cls.SMART_RETRY = getattr(user_config, "SMART_RETRY", cls.SMART_RETRY)
            cls.RETRY_CONFIDENCE_THRESHOLD = getattr(
                user_config,
                "RETRY_CONFIDENCE_THRESHOLD",
                cls.RETRY_CONFIDENCE_THRESHOLD,
            )
            cls.INANIMATE_OBJECTS = getattr(
                user_config, "INANIMATE_OBJECTS", cls.INANIMATE_OBJECTS
            )
            cls.ANALYSIS_PROMPT = getattr(
                user_config, "ANALYSIS_PROMPT", cls.ANALYSIS_PROMPT
            )
            cls.MODEL_NAME = getattr(user_config, "MODEL_NAME", "gemini-2.0-flash")
            cls.API_DELAY = getattr(user_config, "API_DELAY", cls.API_DELAY)
            cls.LOG_LEVEL = getattr(user_config, "LOG_LEVEL", cls.LOG_LEVEL)
            cls.MAX_WORKERS = getattr(user_config, "MAX_WORKERS", cls.MAX_WORKERS)
            cls.CONTEXT_FRAMES = getattr(
                user_config, "CONTEXT_FRAMES", cls.CONTEXT_FRAMES
            )
            cls.PROVIDER = getattr(user_config, "PROVIDER", cls.PROVIDER).lower()
            cls.ANTHROPIC_API_KEY = getattr(
                user_config, "ANTHROPIC_API_KEY", cls.ANTHROPIC_API_KEY
            )
            cls.SESSION_TAG = getattr(user_config, "SESSION_TAG", cls.SESSION_TAG)

            # Load API keys if present
            api_key = getattr(user_config, "GOOGLE_API_KEY", None)
            if api_key and api_key != "YOUR_API_KEY_HERE":
                os.environ["GOOGLE_API_KEY"] = api_key
            anthropic_key = getattr(user_config, "ANTHROPIC_API_KEY", None)
            if anthropic_key and anthropic_key != "YOUR_API_KEY_HERE":
                os.environ["ANTHROPIC_API_KEY"] = anthropic_key

            # Configure Logging
            numeric_level = getattr(logging, cls.LOG_LEVEL.upper(), logging.INFO)
            logging.basicConfig(
                level=numeric_level,
                format="%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%H:%M:%S",
            )

        except ImportError:
            # Configure Logging with defaults if config not found
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%H:%M:%S",
            )
            logging.warning("config.py not found. Using defaults.")


# Load config immediately
CONFIG.load()


class _RateLimiter:
    def __init__(self):
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self):
        with self._lock:
            gap = CONFIG.API_DELAY - (time.time() - self._last)
            if gap > 0:
                time.sleep(gap)
            self._last = time.time()


_api_rate_limiter = _RateLimiter()


def get_session_log_path():
    """Returns the session log filename, e.g. 'sighting_log_GX010070.csv'. SESSION_TAG is appended when set."""
    if not CONFIG.VIDEO_PATH:
        return f"sighting_log{CONFIG.SESSION_TAG}.csv"
    base = os.path.basename(CONFIG.VIDEO_PATH)
    name, _ = os.path.splitext(base)
    name = "".join([c for c in name if c.isalnum() or c in ("-", "_")])
    return f"sighting_log_{name}{CONFIG.SESSION_TAG}.csv"


def find_best_frame_at_timestamp(video_path, timestamp_str, scan_duration_sec=1.0):
    """Returns (frame, blur_score) for the sharpest frame within scan_duration_sec of timestamp_str, or (None, -1)."""
    try:
        if ":" not in timestamp_str:
            return None, -1

        parts = timestamp_str.split(":")
        minutes = int(parts[0])
        seconds = int(parts[1])
        target_time_sec = minutes * 60 + seconds
    except Exception:
        return None, -1

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, -1

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_to_scan = int(scan_duration_sec * fps)

    # Centre the scan: start = target - (duration / 2)
    center_frame = int(target_time_sec * fps)
    start_frame = max(0, center_frame - (frames_to_scan // 2))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    best_frame = None
    best_score = -1.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_available = min(frames_to_scan, total_frames - start_frame)

    for _ in range(frames_available):
        ret, frame = cap.read()
        if not ret:
            break

        _, score = is_blurry(frame)
        if score > best_score:
            best_score = score
            best_frame = frame

    cap.release()
    return best_frame, best_score


def is_blurry(image):
    """Returns (is_blurry, laplacian_variance)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < CONFIG.BLUR_THRESHOLD, variance


def has_content(image):
    """Returns (has_content, edge_density_ratio) using Canny edge detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    height, width = edges.shape
    total_pixels = height * width
    edge_pixels = cv2.countNonZero(edges)

    ratio = edge_pixels / total_pixels
    return ratio > CONFIG.MIN_CONTENT_THRESHOLD, ratio


def detect_motion(frame1, frame2):
    """Returns (motion_detected, ratio) using optical flow with affine stabilization to isolate subject motion from camera shake."""
    if frame1 is None or frame2 is None:
        return False, 0.0

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 1. Find features to track in previous frame
    # We look for corners/contrast (rocks, coral, debris)
    p0 = cv2.goodFeaturesToTrack(
        gray1, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3
    )

    if p0 is None or len(p0) < 8:
        # Not enough background features (blue water?) -> fall back to simple diff
        # OR: if it's just blue water, maybe we *don't* want to capture?
        # Let's assume safely: Simple diff logic for fallback
        diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        ratio = cv2.countNonZero(thresh) / (gray1.size)
        return ratio > CONFIG.MOTION_THRESHOLD, ratio

    # 2. Track features in current frame
    p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None)

    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    if len(good_new) < 8:
        # Tracking failed
        return False, 0.0

    # 3. Estimate Camera Movement (Affine Transform: Rotation + Translation + Scale)
    # estimateAffinePartial2D is more robust for simple camera shake than
    # findHomography
    matrix, inliers = cv2.estimateAffinePartial2D(good_old, good_new)

    if matrix is None:
        return False, 0.0

    # 4. Warp previous frame to match current frame (Stabilize)
    rows, cols = gray1.shape
    warped_gray1 = cv2.warpAffine(gray1, matrix, (cols, rows))

    # 5. Compute difference on STABILIZED frames
    # Areas that match (background) will cancel out.
    # Areas that move differently (fish) will show up.

    # We also mask out invalid borders from the warp
    # (Optional but cleaner)

    diff = cv2.absdiff(warped_gray1, gray2)

    # Threshold
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Erode to remove noise (optional)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=1)

    # Count
    changed_pixels = cv2.countNonZero(thresh)
    ratio = changed_pixels / (rows * cols)

    # Note: Stabilized difference usually yields MUCH smaller ratios than raw shake.
    # We might need to compare against a possibly different threshold,
    # but let's stick to the config one for now (or maybe scale it).

    return ratio > CONFIG.MOTION_THRESHOLD, ratio


def get_timestamp_from_filename(filename):
    """Parses 'frame_00m_05s[_suffix].jpg' → '00:05'."""
    basename = os.path.basename(filename)
    # Expected format: frame_MMm_SSs...jpg
    try:
        # Remove extension
        name_no_ext = os.path.splitext(basename)[0]
        parts = name_no_ext.split("_")
        # parts[0] = "frame", parts[1] = "00m", parts[2] = "05s"
        minutes = parts[1].replace("m", "")
        seconds = parts[2].replace("s", "")
        return f"{minutes}:{seconds}"
    except Exception:
        return "00:00"


def ensure_directories(clear=False):
    """Creates output directories. If clear=True, empties good/blurry/rejected but never manual/."""
    for directory in [
        CONFIG.FRAMES_DIR,
        CONFIG.REJECTED_DIR,
        CONFIG.BLURRY_DIR,
        CONFIG.MANUAL_DIR,
    ]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logging.debug(f"Created directory: {directory}")

    if clear:
        for directory in [CONFIG.FRAMES_DIR, CONFIG.REJECTED_DIR, CONFIG.BLURRY_DIR]:
            logging.info(f"Clearing existing frames in: {directory}")
            for f in glob.glob(os.path.join(directory, "*")):
                try:
                    os.remove(f)
                except Exception as e:
                    logging.warning(f"Failed to remove {f}: {e}")


def reclassify_frames():
    """Re-sorts existing images across good/blurry/rejected using current thresholds. Faster than re-extracting."""
    logging.info("--- Re-sorting Existing Images ---")
    logging.info(f"Blur Threshold: {CONFIG.BLUR_THRESHOLD}")
    logging.info(f"Content Threshold: {CONFIG.MIN_CONTENT_THRESHOLD}")

    # We look at all 3 automatic folders. We DO NOT touch 'manual'.
    # We need to handle files moving between them.
    # Strategy: Move everything to a temp list, then re-distribute.

    all_files = []

    dirs_to_scan = [CONFIG.FRAMES_DIR, CONFIG.BLURRY_DIR, CONFIG.REJECTED_DIR]

    # Ensure dirs exist
    ensure_directories(clear=False)  # Don't clear, we need the files!

    logging.info("Gathering files...")
    for d in dirs_to_scan:
        if os.path.exists(d):
            files = glob.glob(os.path.join(d, "*.jpg"))
            all_files.extend(files)

    logging.info(f"Found {len(all_files)} images to re-sort.")

    moves = {"good": 0, "blurry": 0, "rejected": 0}

    # Matches: frame_00m_00s (_blur...) (.jpg)
    suffix_pattern = re.compile(r"(_blur\d+|_empty[\d\.]+)(\.jpg)$")

    for fpath in tqdm(all_files, desc="Re-sorting"):
        image = cv2.imread(fpath)
        if image is None:
            continue

        # 1. Score
        _, blur_score = is_blurry(image)
        _, content_score = has_content(image)

        # 2. Decide destination
        dest_dir = ""
        suffix = ""

        is_blur = blur_score < CONFIG.BLUR_THRESHOLD

        # Re-eval content (simple check)
        # Note: In extract_frames, smart extract might have skipped empty seconds.
        # Here we just check the image itself.
        has_cont = content_score > CONFIG.MIN_CONTENT_THRESHOLD

        if is_blur:
            dest_dir = CONFIG.BLURRY_DIR
            suffix = f"_blur{int(blur_score)}"
            moves["blurry"] += 1
        elif not has_cont:
            dest_dir = CONFIG.REJECTED_DIR
            suffix = f"_empty{content_score:.4f}"
            moves["rejected"] += 1
        else:
            dest_dir = CONFIG.FRAMES_DIR
            # No suffix for good files
            moves["good"] += 1

        # 3. Construct new filename
        dirname, basename = os.path.split(fpath)

        # Strip old suffix if present
        # We assume base format is frame_MMm_SSs...
        # Let's try to find the standard timestamp part
        # A simpler way: just remove the regex match from the end

        clean_name = basename
        match = suffix_pattern.search(clean_name)
        if match:
            clean_name = clean_name[: match.start()] + match.group(2)  # Keep extension

        # Insert new suffix before extension
        name_root, ext = os.path.splitext(clean_name)
        new_name = f"{name_root}{suffix}{ext}"

        new_path = os.path.join(dest_dir, new_name)

        # 4. Move if different
        if new_path != fpath:
            shutil.move(fpath, new_path)

    logging.info("\nRe-sort Complete.")
    logging.info(f"  Good: {moves['good']}")
    logging.info(f"  Blurry: {moves['blurry']}")
    logging.info(f"  Rejected: {moves['rejected']}")


def extract_frames():
    """Extract quality frames from video into good/, blurry/, rejected/."""
    logging.info("--- Starting Extraction Mode ---")
    logging.info(f"Video: {CONFIG.VIDEO_PATH}")
    logging.info(f"Smart Extract: {CONFIG.SMART_EXTRACT}")

    if not os.path.exists(CONFIG.VIDEO_PATH):
        logging.error(f"Error: Video file '{CONFIG.VIDEO_PATH}' not found.")
        return

    # Prepare directories
    ensure_directories(clear=True)

    cap = cv2.VideoCapture(CONFIG.VIDEO_PATH)
    if not cap.isOpened():
        logging.error("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0

    logging.info(f"Duration: {duration_sec:.2f} seconds. FPS: {fps:.2f}")

    # Calculate frames to skip
    frames_to_skip = int(CONFIG.SNAPSHOT_INTERVAL * fps)

    saved_count = 0
    blurry_count = 0
    rejected_empty_count = 0

    # We will iterate through the video
    # Using tqdm for progress
    pbar = tqdm(total=total_frames, unit="frames")

    current_frame = 0

    while True:
        if current_frame >= total_frames:
            break

        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        seconds_total = int(timestamp_ms / 1000)
        minutes = seconds_total // 60
        seconds = seconds_total % 60
        base_filename = f"frame_{minutes:02d}m_{seconds:02d}s"

        final_frame = None
        best_blur_score = -1.0

        # Determine how many frames to process in this interval
        if CONFIG.SMART_EXTRACT:
            # Hybrid Smart Mode:
            # 1. Peek at the first frame to see if it's worth scanning the
            # whole second
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, first_frame = cap.read()
            if not ret:
                break

            has_cont, _ = has_content(first_frame)

            # SECOND CHANCE: Check Motion (Instantaneous)
            is_moving = False
            # Read next immediate frame to check for motion
            ret2, next_frame_check = cap.read()
            if ret2:
                is_moving, _ = detect_motion(first_frame, next_frame_check)

            if not has_cont and not is_moving:
                # OPTIMIZATION: First frame is empty water & no motion.
                # Assume the whole second is likely empty. Skip the expensive
                # scan.
                final_frame = first_frame
                # We still need a blur score for the logic below (even if
                # ignored for rejection)
                _, best_blur_score = is_blurry(final_frame)

                # Advance mostly quickly
                current_frame += frames_to_skip
                pbar.update(frames_to_skip)
            else:
                # Content OR Motion detected! Worth scanning for the sharpest
                # frame.
                frames_to_read = min(frames_to_skip, total_frames - current_frame)

                # Reset to start of interval (since we read one frame)
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

                for _ in range(frames_to_read):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # We need to score it to find the best
                    _, b_score = is_blurry(frame)

                    if b_score > best_blur_score:
                        best_blur_score = b_score
                        final_frame = frame

                # Advance counter
                current_frame += frames_to_read
                pbar.update(frames_to_read)

        else:
            # Fast Mode: Jump to current_frame and read one
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, final_frame = cap.read()

            if not ret:
                break

            # Advance counter
            current_frame += frames_to_skip
            pbar.update(frames_to_skip)

            # Calculate score for this single frame so we can use it below
            if final_frame is not None:
                _, best_blur_score = is_blurry(final_frame)

        # Process the chosen frame (final_frame)
        if final_frame is not None:
            # Re-evaluate with thresholds (since is_blurry logic is just
            # variance calc)
            is_blur = best_blur_score < CONFIG.BLUR_THRESHOLD
            has_cont, content_score = has_content(final_frame)

            # Re-check motion for the processing decision (Fast Mode needs
            # this)
            is_moving = False

            # We need to peek at the next frame to see if there is motion
            # Note: logic above might have advanced cap, so we need to be careful.
            # In Fast Mode, we just read one frame.
            # Let's verify if we need to read 'next' here.

            if not CONFIG.SMART_EXTRACT:
                # In fast mode, we are at pos+1. Read it.
                ret_next, next_frame_fast = cap.read()
                if ret_next:
                    is_moving, motion_score = detect_motion(
                        final_frame, next_frame_fast
                    )

            # In SMART_EXTRACT, we might have scanned multiple frames.
            # Ideally we check motion ON the chosen frame vs its neighbor.
            # But we already did a coarse check at the start of the second.
            # Let's stick to the coarse check for SMART mode (it passed the filter).
            # For FAST mode, use the check we just did.

            # If we are here in SMART mode, we passed the filter, so is_moving is effectively True/irrelevant
            # unless we fell through content check.
            # Wait, if has_cont is False, we rely on is_moving.
            # For SMART mode, we checked is_moving at the *start*.
            # That decision allowed us to enter the loop.
            # But wait, logic block 398 sets is_moving.
            # If we entered the loop, either has_cont=True or is_moving=True.
            # So we don't strictly need to re-calc is_moving for rejection logic
            # UNLESS we want to be strict about the specific *final_frame*.
            # Simplify: If we are in SMART mode, assume we keep it if we spent the effort to find it.
            # EXCEPT if the final frame itself is trash?
            # Let's rely on the previous logic:

            if CONFIG.SMART_EXTRACT:
                # We already decided it was worth keeping.
                # But wait, has_content(final_frame) might be low.
                # If we entered because of motion, we should keep it.
                # Reuse the initial is_moving? No, that variable is local to the block above.
                # Let's re-measure motion for the final frame if needed.
                pass
                # Be Safe: always check motion on final frame if content low.
                # Since we don't have next frame easily without seek,
                # maybe rely on the fact that we found *some* content?

                # Actually, let's keep it simple. If we are improving "Extraction Accuracy",
                # we trust the "Trigger" (Start of second).
                # If the Trigger said GO, we save whatever we found.
                is_moving = True

            if is_blur:
                fname = f"{base_filename}_blur{int(best_blur_score)}.jpg"
                save_path = os.path.join(CONFIG.BLURRY_DIR, fname)
                cv2.imwrite(save_path, final_frame)
                blurry_count += 1
            elif not has_cont and not is_moving:
                fname = f"{base_filename}_empty{content_score:.4f}.jpg"
                save_path = os.path.join(CONFIG.REJECTED_DIR, fname)
                cv2.imwrite(save_path, final_frame)
                rejected_empty_count += 1
            else:
                fname = f"{base_filename}.jpg"
                save_path = os.path.join(CONFIG.FRAMES_DIR, fname)
                cv2.imwrite(save_path, final_frame)
                saved_count += 1

    pbar.close()
    cap.release()

    estimated_cost = saved_count * CONFIG.COST_PER_IMAGE

    logging.info("Extraction complete.")
    logging.info(
        f"Saved: {saved_count} sharp, interesting frames to '{CONFIG.FRAMES_DIR}'."
    )
    logging.info(
        f"Rejected content: {rejected_empty_count} frames to '{CONFIG.REJECTED_DIR}'."
    )
    logging.info(f"Rejected blurry: {blurry_count} frames to '{CONFIG.BLURRY_DIR}'.")
    logging.info(
        f"Estimated Analysis Cost: ${estimated_cost:.4f} (based on Gemini Flash rates)."
    )
    logging.info(
        f"Please review rejected folders and move any missed fish to '{
            CONFIG.FRAMES_DIR
        }' before analysis."
    )


def parse_ai_response(text):
    """Parses JSON from AI response, handling markdown fences. Always returns a list of dicts."""
    text = text.strip()

    if "```" in text:
        parts = text.split("```")
        for part in parts:
            if "[" in part or "{" in part:
                text = re.sub(r"^json\s*", "", part).strip()
                break

    data = None
    try:
        data = json.loads(text)
    except Exception:
        try:
            data = ast.literal_eval(text)
        except Exception:
            pass

    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return []

    return [item for item in data if isinstance(item, dict)]


@retry(
    wait=wait_exponential(multiplier=2, min=4, max=120),
    stop=stop_after_attempt(10),
    retry=retry_if_exception_type(
        (
            exceptions.ResourceExhausted,
            exceptions.ServiceUnavailable,
            exceptions.DeadlineExceeded,
        )
    ),
)
def call_gemini_api(client, prompt, *images):
    try:
        _api_rate_limiter.wait()
        logging.debug(f"Sending request to Gemini ({CONFIG.MODEL_NAME})...")
        start_t = time.time()
        response = client.models.generate_content(
            model=CONFIG.MODEL_NAME,
            contents=[prompt, *images],
            config=types.GenerateContentConfig(temperature=0),
        )
        logging.debug(f"Response received in {time.time() - start_t:.2f}s")
        return response
    except Exception as e:
        logging.warning(f"API Error: {e}. Retrying...")
        raise e


def _is_anthropic_retryable(exc):
    """Only retry on rate limits and transient connection errors, not billing/auth failures."""
    if _anthropic is None:
        return False
    if isinstance(exc, _anthropic.RateLimitError):
        return True
    if isinstance(exc, _anthropic.APIConnectionError):
        return True
    if isinstance(exc, _anthropic.APIStatusError):
        return (
            exc.status_code >= 500
        )  # 5xx = server error, retryable; 4xx = client error, not retryable
    return False


@retry(
    wait=wait_exponential(multiplier=2, min=4, max=120),
    stop=stop_after_attempt(10),
    retry=retry_if_exception(_is_anthropic_retryable),
    reraise=True,
)
def call_anthropic_api(client, prompt, *images):
    """Encode PIL images as base64 and call the Anthropic Messages API."""
    if _anthropic is None:
        raise RuntimeError(
            "anthropic package not installed. Run: pip install anthropic"
        )
    _api_rate_limiter.wait()
    logging.debug(f"Sending request to Anthropic ({CONFIG.MODEL_NAME})...")
    start_t = time.time()
    content = []
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
        content.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
            }
        )
    content.append({"type": "text", "text": prompt})
    try:
        response = client.messages.create(
            model=CONFIG.MODEL_NAME,
            max_tokens=1024,
            messages=[{"role": "user", "content": content}],
        )
        logging.debug(f"Response received in {time.time() - start_t:.2f}s")
        return response.content[0].text
    except Exception as e:
        logging.warning(f"Anthropic API error: {e}")
        raise


def call_ai_api(client, prompt, *images):
    """Dispatch to the configured provider and return the response text."""
    if CONFIG.PROVIDER == "anthropic":
        return call_anthropic_api(client, prompt, *images)
    return call_gemini_api(client, prompt, *images).text


def get_video_creation_date(video_path):
    """Returns 'YYYY-MM-DD' from video metadata via ffprobe, or None."""
    try:
        # Command: ffprobe -v quiet -select_streams v:0 -show_entries
        # stream_tags=creation_time -of default=noprint_wrappers=1:nokey=1
        # [FILE]
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream_tags=creation_time",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        date_str = result.stdout.strip()

        if date_str:
            # Expected format: 2025-12-21T18:17:46.000000Z
            # We just want YYYY-MM-DD
            if "T" in date_str:
                return date_str.split("T")[0]
            return date_str[:10]  # Fallback for other ISO formats

    except Exception as e:
        logging.warning(f"Warning: Could not extract video date: {e}")

    return None


def sort_session_log(log_path):
    """Sorts the CSV log in-place by timestamp."""
    if not os.path.exists(log_path):
        return

    rows = []
    header = []

    try:
        with open(log_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                return  # Empty file
            rows = list(reader)

        # Sort by first column (timestamp)
        # Timestamp format 00:00 or 00:00:00 sorts correctly as string
        rows.sort(key=lambda x: x[0])

        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

        logging.info(f"Log file sorted: {os.path.basename(log_path)}")
    except Exception as e:
        logging.warning(f"Warning: Could not sort log file: {e}")


def get_context_frames(timestamp_str, count=3):
    """Return up to `count` PIL Images centered on timestamp_str for multi-frame context."""
    if count <= 1 or not os.path.exists(CONFIG.VIDEO_PATH):
        return []
    try:
        parts = timestamp_str.split(":")
        target_sec = int(parts[0]) * 60 + int(parts[1])
    except Exception:
        return []

    cap = cv2.VideoCapture(CONFIG.VIDEO_PATH)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    center = int(target_sec * fps)
    half = int(fps / 2)
    start = max(0, center - half)
    end = min(total_frames - 1, center + half)

    indices = [int(start + i * (end - start) / max(count - 1, 1)) for i in range(count)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    cap.release()
    return frames


def analyze_frames():
    """Identify species in extracted frames using the configured AI provider. Resume-safe; rebuilds youtube_chapters.txt on completion."""
    logging.info("--- Starting Analysis Mode ---")
    logging.info(f"Provider: {CONFIG.PROVIDER} | Model: {CONFIG.MODEL_NAME}")

    ensure_directories(clear=False)

    if CONFIG.PROVIDER == "anthropic":
        if _anthropic is None:
            logging.error("anthropic package not installed. Run: pip install anthropic")
            return
        api_key = os.environ.get("ANTHROPIC_API_KEY") or CONFIG.ANTHROPIC_API_KEY
        if not api_key:
            logging.error("ANTHROPIC_API_KEY not set in config.py or environment.")
            return
        client = _anthropic.Anthropic(api_key=api_key)
    else:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            logging.error("GOOGLE_API_KEY environment variable not set.")
            return
        client = genai.Client(api_key=api_key)

    files_main = glob.glob(os.path.join(CONFIG.FRAMES_DIR, "*.jpg"))
    files_manual = glob.glob(os.path.join(CONFIG.MANUAL_DIR, "*.jpg"))
    image_files = sorted(files_main + files_manual, key=lambda x: os.path.basename(x))

    if not image_files:
        logging.error(
            f"No images found in {CONFIG.FRAMES_DIR} or {CONFIG.MANUAL_DIR}. Run 'extract' mode first."
        )
        return

    log_file = get_session_log_path()
    processed_files = set()

    logging.info(f"Session Log File: {log_file}")

    if not os.path.exists(log_file):
        with open(log_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(
                ["timestamp", "file", "common_name", "scientific_name", "confidence"]
            )
    else:
        try:
            with open(log_file, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("file"):
                        processed_files.add(row["file"])
            logging.info(
                f"Resuming analysis. {len(processed_files)} images already processed."
            )
        except Exception as e:
            logging.error(f"Error reading existing log: {e}")

    logging.info(
        f"Found {len(image_files)} images total ({len(files_manual)} manual overrides)."
    )

    video_date = get_video_creation_date(CONFIG.VIDEO_PATH)
    date_context = f"Date: {video_date}. " if video_date else ""

    try:
        scan_prompt = CONFIG.ANALYSIS_PROMPT.format(
            location=CONFIG.LOCATION_CONTEXT,
            date=date_context,
            n=CONFIG.CONTEXT_FRAMES,
        )
    except Exception as e:
        scan_prompt = CONFIG.ANALYSIS_PROMPT
        logging.warning(f"Prompt formatting failed ({e}). Using raw prompt.")

    pending = [p for p in image_files if os.path.basename(p) not in processed_files]

    csv_lock = threading.Lock()
    chapter_entries = []

    _UNKNOWN_SCI = {"unknown", "null", "none", ""}

    def _filter_species(raw_list):
        """Filter a parsed species list, returning only valid identifiable entries."""
        results = []
        for item in raw_list:
            name = (item.get("common_name") or "").strip()
            sci = (item.get("scientific_name") or "").strip()
            conf = float(item.get("confidence") or 0.0)
            if not name or name.lower() in CONFIG.INANIMATE_OBJECTS:
                continue
            if sci.lower() in _UNKNOWN_SCI:
                continue
            results.append((name, sci, conf))
        return results

    def process_frame(img_path):
        base_name = os.path.basename(img_path)
        timestamp_str = get_timestamp_from_filename(img_path)

        try:
            images = get_context_frames(timestamp_str, CONFIG.CONTEXT_FRAMES)
            if not images:
                images = [PIL.Image.open(img_path)]

            text = call_ai_api(client, scan_prompt, *images)
            species = _filter_species(parse_ai_response(text))

            if not species and CONFIG.SMART_RETRY:
                logging.info(f"  > {base_name}: no valid IDs. Attempting Smart Retry...")
                better_frame, b_score = find_best_frame_at_timestamp(
                    CONFIG.VIDEO_PATH, timestamp_str
                )
                if better_frame is not None:
                    logging.info(
                        f"  > Found replacement frame (blur score: {b_score:.1f}). Re-analyzing..."
                    )
                    cv2.imwrite(img_path, better_frame)
                    pil_retry = PIL.Image.fromarray(
                        cv2.cvtColor(better_frame, cv2.COLOR_BGR2RGB)
                    )
                    try:
                        text2 = call_ai_api(client, scan_prompt, pil_retry)
                        species = _filter_species(parse_ai_response(text2))
                        if species:
                            logging.info(f"  > Retry found {len(species)} species.")
                        else:
                            logging.info("  > Retry returned no valid IDs. Skipping.")
                    except Exception as retry_err:
                        logging.warning(f"  > Retry failed: {retry_err}")
                else:
                    logging.warning("  > Could not find a better frame.")

            if not species:
                logging.info(f"  > {base_name}: no animals detected.")
                return []

            logging.info(
                f"  > {base_name}: {', '.join(n for n, _, _ in species)}"
            )
            return [
                {
                    "timestamp": timestamp_str,
                    "base_name": base_name,
                    "common_name": name,
                    "scientific_name": sci,
                    "confidence": conf,
                }
                for name, sci, conf in species
            ]

        except Exception as e:
            logging.error(f"\nError processing {img_path}: {e}")
            if hasattr(e, "last_attempt") and e.last_attempt.exception():
                logging.error(f"Original Exception: {e.last_attempt.exception()}")
            return []

    logging.info(
        f"Analyzing {len(pending)} frames with {CONFIG.MAX_WORKERS} worker(s)..."
    )

    with ThreadPoolExecutor(max_workers=CONFIG.MAX_WORKERS) as executor:
        futures = {executor.submit(process_frame, p): p for p in pending}

        for future in tqdm(as_completed(futures), total=len(pending), desc="Analyzing"):
            results = future.result()
            if not results:
                continue

            with csv_lock:
                with open(log_file, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                    for result in results:
                        writer.writerow(
                            [
                                result["timestamp"],
                                result["base_name"],
                                result["common_name"],
                                result["scientific_name"],
                                f"{result['confidence']:.2f}",
                            ]
                        )

            for result in results:
                if result["confidence"] >= CONFIG.CONFIDENCE_THRESHOLD:
                    entry = f"{result['common_name']} ({result['scientific_name']})"
                    chapter_entries.append(
                        (result["timestamp"], entry, result["confidence"])
                    )

    # Regenerate chapters from the full CSV so resumes produce a complete file.
    # Normalize common names: for each scientific name, use whichever common name
    # appears most frequently (ties broken by highest confidence).
    all_rows = []
    name_votes = {}  # scientific_name -> {common_name: count}
    try:
        with open(log_file, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                all_rows.append(row)
                sci = row.get("scientific_name", "").strip()
                common = row.get("common_name", "").strip()
                if sci and common:
                    name_votes.setdefault(sci, {})
                    name_votes[sci][common] = name_votes[sci].get(common, 0) + 1
    except Exception as e:
        logging.warning(f"Could not read log for chapters: {e}")

    canonical = {sci: max(counts, key=counts.get) for sci, counts in name_votes.items()}

    all_entries = []
    for row in all_rows:
        conf = float(row.get("confidence") or 0)
        sci = row.get("scientific_name", "").strip()
        common = canonical.get(sci, row.get("common_name", "")).strip()
        if (
            conf >= CONFIG.CONFIDENCE_THRESHOLD
            and common
            and common.lower() not in CONFIG.INANIMATE_OBJECTS
        ):
            entry = f"{common} ({sci})"
            all_entries.append((row["timestamp"], entry, conf))

    if all_entries:
        all_entries.sort(key=lambda x: x[0])
        chapters = []
        prev_species: set[str] = set()
        # Group by timestamp so multi-species frames suppress correctly
        for ts, group in groupby(all_entries, key=lambda x: x[0]):
            group = list(group)
            curr_species: set[str] = {e.lower() for _, e, _ in group}
            for _, entry, conf in group:
                if entry.lower() not in prev_species:
                    chapters.append(f"{ts} - {entry} [{conf:.2f}]")
            prev_species = curr_species
        with open("youtube_chapters.txt", "w") as f:
            f.write("\n".join(chapters))
        logging.info("Saved youtube_chapters.txt")

    sort_session_log(log_file)


def get_scores(directory):
    files = glob.glob(os.path.join(directory, "*.jpg"))
    scores = []  # list of (filename, blur_score, content_score)
    if not files:
        return scores

    for fpath in tqdm(files, desc=f"Scanning {os.path.basename(directory)}"):
        image = cv2.imread(fpath)
        if image is None:
            continue
        _, blur = is_blurry(image)
        _, content = has_content(image)
        scores.append(
            {"file": os.path.basename(fpath), "blur": blur, "content": content}
        )
    return scores


def tune():
    logging.info("--- Advanced Auto-Tuning Configuration ---")
    logging.info(
        "This script analyzes your manual sorting to find the best thresholds."
    )

    # Paths
    dir_good = CONFIG.FRAMES_DIR
    dir_blurry = CONFIG.BLURRY_DIR
    dir_rejected = CONFIG.REJECTED_DIR

    # 0. auto-refine manual frames
    logging.info("\n--- Step 0: Refining Manual Frames ---")
    refine_selected_frames()

    # 1. Analyze All Buckets
    logging.info("\n1. Analyzing image stats across all folders...")
    good_stats = get_scores(dir_good)
    blurry_stats = get_scores(dir_blurry)
    rejected_stats = get_scores(dir_rejected)

    if not good_stats:
        logging.error(
            f"Error: No images found in '{dir_good}'. You must have at least one 'good' frame to tune thresholds."
        )
        return

    # 2. Determine Optimal Thresholds
    # Constraint 1: Must accept ALL good files.
    # We find the 'worst' good file and set the bar just below it.
    min_good_blur = min(s["blur"] for s in good_stats)
    min_good_content = min(s["content"] for s in good_stats)

    # Proposed new thresholds (slightly relaxed to ensure inclusion)
    new_blur_thresh = max(0.0, min_good_blur - 0.1)
    new_content_thresh = max(0.0, min_good_content - 0.0001)

    # 3. Analyze Trade-offs (Overlap)
    false_positive_blurry = [s for s in blurry_stats if s["blur"] >= new_blur_thresh]
    false_positive_rejected = [
        s for s in rejected_stats if s["content"] > new_content_thresh
    ]

    logging.info("\n--- Analysis Results ---")
    logging.info(f"Total Good Files: {len(good_stats)}")
    logging.info(f"  Min Blur Score found: {min_good_blur:.2f}")
    logging.info(f"  Min Content Score found: {min_good_content:.5f}")

    logging.info("\nProposed Config:")
    logging.info(
        f"  BLUR_THRESHOLD:        {new_blur_thresh:.4f} (was {CONFIG.BLUR_THRESHOLD})"
    )
    logging.info(
        f"  MIN_CONTENT_THRESHOLD: {new_content_thresh:.5f} (was {
            CONFIG.MIN_CONTENT_THRESHOLD
        })"
    )

    logging.info("\n--- Impact Analysis ---")
    if false_positive_blurry:
        logging.warning(
            f"WARNING: {len(false_positive_blurry)} images from '{
                dir_blurry
            }' will now be considered SHARP."
        )
        logging.warning(f"  (They have blur scores >= {new_blur_thresh:.4f}).")
    else:
        logging.info(
            f"Perfect Separation! No images from '{dir_blurry}' will be falsely accepted."
        )

    if false_positive_rejected:
        logging.warning(
            f"WARNING: {len(false_positive_rejected)} images from '{
                dir_rejected
            }' will now be considered VALID CONTENT."
        )
        logging.warning(f"  (They have content scores > {new_content_thresh:.5f}).")
    else:
        logging.info(
            f"Perfect Separation! No images from '{dir_rejected}' will be falsely accepted."
        )

    # 4. Apply & Verify
    logging.info(
        "\nTo keep all your 'Good' files, we must update the config to the Proposed values."
    )
    confirm = input("Update config.py and RERUN extraction to verify? (y/n): ")

    if confirm.lower() == "y":
        # Update Config
        try:
            # Create Backup
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"config.py.bak.{timestamp}"
            shutil.copy("config.py", backup_name)
            logging.info(f"Created backup: {backup_name}")

            # Read existing config
            with open("config.py", "r") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                # Update BLUR_THRESHOLD
                if line.strip().startswith("BLUR_THRESHOLD"):
                    # Preserve comment if exists
                    comment = ""
                    if "#" in line:
                        comment = " # " + line.split("#", 1)[1].strip()
                    new_lines.append(f"BLUR_THRESHOLD = {new_blur_thresh}{comment}\n")

                # Update MIN_CONTENT_THRESHOLD
                elif line.strip().startswith("MIN_CONTENT_THRESHOLD"):
                    comment = ""
                    if "#" in line:
                        comment = " # " + line.split("#", 1)[1].strip()
                    new_lines.append(
                        f"MIN_CONTENT_THRESHOLD = {new_content_thresh}{comment}\n"
                    )
                else:
                    new_lines.append(line)

            with open("config.py", "w") as f:
                f.writelines(new_lines)

            logging.info("Successfully updated config.py.")

            # Since we changed a python module, we must reload config in memory
            CONFIG.load()

            logging.info("\n--- Verification ---")
            logging.info("To see the effect of your changes, you can either:")
            logging.info("  [s] Re-sort existing images (FAST, ~seconds)")
            logging.info("  [e] Re-extract from video (SLOW, ~minutes)")
            logging.info("  [n] Skip verification")

            choice = input("Choice [s/e/n]: ").lower()

            if choice == "s":
                reclassify_frames()
            elif choice == "e":
                logging.warning(
                    "WARNING: This will clear your manual sorting folders and re-extract from the video."
                )
                extract_frames()
            else:
                logging.info("Verification skipped.")

        except Exception as e:
            print(f"Error: {e}")
    else:
        logging.info("Cancelled. Config not updated.")


def refine_selected_frames():
    """Replace manual/ frames and suffixed good/ frames with the sharpest frame from the same video second."""
    logging.info("--- Starting Refine Mode ---")

    candidates = []  # List of (path, directory)

    # 1. Gather Manual Files (All)
    if os.path.exists(CONFIG.MANUAL_DIR):
        files = glob.glob(os.path.join(CONFIG.MANUAL_DIR, "*.jpg"))
        for f in files:
            candidates.append((f, CONFIG.MANUAL_DIR))

    # 2. Gather Good Files (Only suffixed)
    suffix_pattern = re.compile(r"(_blur\d+|_empty[\d\.]+)(\.jpg)$")

    if os.path.exists(CONFIG.FRAMES_DIR):
        files = glob.glob(os.path.join(CONFIG.FRAMES_DIR, "*.jpg"))
        for f in files:
            if suffix_pattern.search(f):
                candidates.append((f, CONFIG.FRAMES_DIR))

    if not candidates:
        logging.info("No files found to refine.")
        logging.info(
            f"Checked {CONFIG.MANUAL_DIR} (All) and {
                CONFIG.FRAMES_DIR
            } (Only files with _blur/_empty suffixes)."
        )
        return

    logging.info(f"Found {len(candidates)} images to refine.")

    cap = cv2.VideoCapture(CONFIG.VIDEO_PATH)
    if not cap.isOpened():
        logging.error("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_scan = int(1.0 * fps)  # Scan 1 second window

    refined_count = 0

    for img_path, dest_dir in tqdm(candidates, desc="Refining"):
        basename = os.path.basename(img_path)
        try:
            time_match = re.search(r"frame_(\d+)m_(\d+)s", basename)
            if not time_match:
                logging.debug(f"Skipping {basename}: cannot parse timestamp pattern.")
                continue

            minutes = int(time_match.group(1))
            seconds = int(time_match.group(2))

            target_time_sec = minutes * 60 + seconds
            start_frame = int(target_time_sec * fps)

        except Exception as e:
            logging.error(f"Skipping {basename}: error parsing. {e}")
            continue

        # Smart Scan logic
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        best_frame = None
        best_score = -1.0

        frames_available = min(frames_to_scan, total_frames - start_frame)
        if frames_available <= 0:
            continue

        for _ in range(frames_available):
            ret, frame = cap.read()
            if not ret:
                break

            _, score = is_blurry(frame)
            if score > best_score:
                best_score = score
                best_frame = frame

        # Overwrite content
        if best_frame is not None:
            # Construct clean name
            clean_name = f"frame_{minutes:02d}m_{seconds:02d}s.jpg"
            save_path = os.path.join(dest_dir, clean_name)

            # If we are effectively renaming (removing suffix), delete old
            if img_path != save_path:
                if os.path.exists(img_path):
                    os.remove(img_path)

            cv2.imwrite(save_path, best_frame)
            refined_count += 1

    cap.release()
    logging.info(f"Refined {refined_count} images.")


def archive_log():
    """Enrich session CSV with date/location/video metadata and append to master_sighting_log.csv. Deduplicates on video+timestamp."""
    logging.info("--- Archiving Session Log ---")

    current_log = get_session_log_path()
    master_log = "master_sighting_log.csv"

    logging.info(f"Reading from: {current_log}")

    if not os.path.exists(current_log):
        logging.warning(f"No {current_log} found to archive.")
        return

    # Gather Metadata
    video_name = os.path.basename(CONFIG.VIDEO_PATH)
    location = CONFIG.LOCATION_CONTEXT
    date_str = get_video_creation_date(CONFIG.VIDEO_PATH) or "Unknown Date"

    logging.info(f"Context: {date_str} | {location} | {video_name}")

    # Read current session entries
    new_entries = []

    try:
        with open(current_log, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Expects keys:
                # timestamp,file,common_name,scientific_name,confidence
                if not row.get("timestamp"):
                    continue

                # Create enriched entry
                entry = {
                    "date": date_str,
                    "location": location,
                    "video_name": video_name,
                    "timestamp": row["timestamp"],
                    "common_name": row["common_name"],
                    "scientific_name": row["scientific_name"],
                    "confidence": row["confidence"],
                }
                new_entries.append(entry)
    except Exception as e:
        logging.error(f"Error reading current log: {e}")
        return

    if not new_entries:
        logging.info("Current log is empty.")
        return

    # Check Master Log
    existing_keys = set()
    fieldnames = [
        "date",
        "location",
        "video_name",
        "timestamp",
        "common_name",
        "scientific_name",
        "confidence",
    ]
    file_exists = os.path.exists(master_log)

    if file_exists:
        try:
            with open(master_log, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Key: Video + Timestamp
                    # Handle potential missing keys if file is corrupted
                    v = row.get("video_name", "")
                    t = row.get("timestamp", "")
                    if v and t:
                        existing_keys.add(f"{v}_{t}")
        except Exception as e:
            logging.warning(
                f"Warning: Could not read master log (might be corrupt): {e}"
            )

    # Append
    added_count = 0
    mode = "a" if file_exists else "w"

    with open(master_log, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        if not file_exists:
            writer.writeheader()

        for entry in new_entries:
            key = f"{entry['video_name']}_{entry['timestamp']}"
            if key not in existing_keys:
                writer.writerow(entry)
                added_count += 1

    logging.info(f"Archived {added_count} new sightings to {master_log}.")
    logging.info(f"Total Master Log Size: {len(existing_keys) + added_count} entries.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wildlife Video Analyzer")
    parser.add_argument(
        "mode",
        choices=["extract", "analyze", "refine", "tune", "archive"],
        help="Operation mode",
    )

    # Handle the case where no args are passed
    if len(sys.argv) == 1:
        print("Wildlife Video Analyzer - Usage Guide")
        print("==================================")
        print("Usage: python wildlife_analyzer.py [mode]\n")
        print("Modes:")
        print("  extract   : Reads video and saves frames to 'extracted_images/good'.")
        print(
            "              - Use 'SMART_EXTRACT': true in config for hybrid scanning."
        )
        print("              - Filters out blurry and empty frames automatically.\n")

        print(
            "  tune      : Optimizes detection thresholds based on your manual sorting."
        )
        print(
            "              - Sort files into extracted_images/good (Keep) / _blurry / _rejected."
        )
        print(
            "              - Move blurry-but-good files to 'extracted_images/manual'."
        )
        print(
            "              - AUTO-REFINES manual frames (better quality) before tuning.\n"
        )

        print("  analyze   : Identifies species using Google Gemini.")
        print(
            "              - Smart Retry: Re-scans video if confidence < 60% or scientific name unknown."
        )
        print("              - Smart Filter: Ignores rocks/sand/gravel/humans.")
        print(
            "              - Generates 'sighting_log_{video}.csv' and 'youtube_chapters.txt'.\n"
        )

        print(
            "  refine    : (Manual Only) specialized command to re-scan just the manual folder."
        )
        print("              - Note: 'tune' runs this automatically.\n")

        print("Configuration:")
        print("  Edit 'config.py' to change thresholds, video path, or API key.")
        sys.exit(1)

    args = parser.parse_args()

    if args.mode == "extract":
        extract_frames()
    elif args.mode == "refine":
        refine_selected_frames()
    elif args.mode == "analyze":
        analyze_frames()
    elif args.mode == "tune":
        tune()
    elif args.mode == "archive":
        archive_log()
    else:
        logging.error(f"Unknown mode: {args.mode}")
