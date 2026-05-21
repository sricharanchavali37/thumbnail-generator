import cv2
import numpy as np
from datetime import datetime
from typing import List, Tuple
import asyncio

from backend.workers.celery_app import celery_app
from backend.services.s3 import (
    get_raw_video_url,
    upload_thumbnails_to_s3,
)
from backend.database.mongodb import get_videos_collection
from backend.utils.ffmpeg_helper import (
    get_video_metadata,
    calculate_frame_count,
    stream_frames_from_s3,
)


# ─────────────────────────────────────────────────────────
# SCORING WEIGHTS PER CONTENT TYPE
#
# These numbers control how much each score matters
# depending on what type of video we detected.
#
# Each row is one content type.
# Each column is one scoring criteria.
# Higher number = that criteria matters more.
# ─────────────────────────────────────────────────────────

CONTENT_WEIGHTS = {
    "vlog": {
        "blur": 0.25,
        "brightness": 0.20,
        "face": 0.40,       # Faces matter most in vlogs
        "motion": 0.15,
    },
    "sports": {
        "blur": 0.20,
        "brightness": 0.20,
        "face": 0.10,
        "motion": 0.50,     # Motion matters most in sports
    },
    "cinematic": {
        "blur": 0.25,
        "brightness": 0.50, # Brightness matters most in cinematic
        "face": 0.10,
        "motion": 0.15,
    },
    "general": {
        "blur": 0.25,
        "brightness": 0.25,
        "face": 0.25,
        "motion": 0.25,     # All equal for general content
    },
}


# ─────────────────────────────────────────────────────────
# FACE DETECTOR
#
# Haar Cascade is a pre-trained face detector.
# It comes built into OpenCV for free.
# No download needed. No API needed.
#
# cv2.data.haarcascades gives us the path
# to where OpenCV stores its pre-trained models.
# ─────────────────────────────────────────────────────────

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ═════════════════════════════════════════════════════════
# SECTION 1 — FRAME SCORING FUNCTIONS
#
# Each function takes one frame (numpy array)
# and returns a score from 0.0 to 1.0
# 0.0 = worst possible
# 1.0 = best possible
# ═════════════════════════════════════════════════════════

def score_blur(frame: np.ndarray) -> float:
    """
    Detects how blurry a frame is.

    Method: Laplacian variance
      Laplacian is a mathematical filter
      that detects edges in an image.
      Sharp image = many strong edges = high variance.
      Blurry image = few weak edges = low variance.

    We normalize the result to 0.0 - 1.0 range.
    Variance above 500 is considered perfectly sharp.
    """
    # Convert to grayscale first
    # Laplacian works on single channel images
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Laplacian filter and measure variance
    laplacian_variance = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Normalize to 0.0 - 1.0
    # min(variance / 500, 1.0) means:
    #   if variance is 500 or more → score is 1.0 (perfectly sharp)
    #   if variance is 250 → score is 0.5
    #   if variance is 0 → score is 0.0 (completely blurry)
    return min(laplacian_variance / 500.0, 1.0)


def score_brightness(frame: np.ndarray) -> float:
    """
    Scores how well-lit the frame is.

    Method: HSV histogram analysis
      HSV color format separates brightness (V channel)
      from color information (H and S channels).
      This makes brightness easy to measure independently.

    Ideal brightness is in the middle range.
    Too dark (under 50) = bad thumbnail.
    Too bright (over 200) = bad thumbnail.
    Sweet spot is 80 to 170.
    """
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Extract just the V channel (brightness)
    v_channel = hsv[:, :, 2]

    # Calculate average brightness across all pixels
    avg_brightness = np.mean(v_channel)

    # Score based on how close to ideal brightness
    # Ideal range is 80 to 170 out of 255
    if 80 <= avg_brightness <= 170:
        # In ideal range
        # Score is highest at the center (125) and
        # decreases toward the edges of the range
        score = 1.0 - abs(avg_brightness - 125) / 90.0
    elif avg_brightness < 80:
        # Too dark
        score = avg_brightness / 80.0
    else:
        # Too bright
        score = (255 - avg_brightness) / 85.0

    return max(0.0, min(score, 1.0))


def score_faces(frame: np.ndarray) -> float:
    """
    Detects faces in the frame and returns a score.

    Method: Haar Cascade face detection
      Scans the image in sliding windows.
      Each window is checked against trained face patterns.
      Returns list of rectangles where faces were found.

    No face    → 0.0
    1 face     → 0.8  (good, clear subject)
    2-3 faces  → 1.0  (great, social content)
    4+ faces   → 0.9  (crowd, slightly less focused)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    # scaleFactor: how much image size is reduced each scan
    # minNeighbors: how many detections needed to confirm a face
    # Higher minNeighbors = fewer false positives
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),   # Minimum face size to detect
    )

    face_count = len(faces)

    if face_count == 0:
        return 0.0
    elif face_count == 1:
        return 0.8
    elif face_count <= 3:
        return 1.0
    else:
        return 0.9


def score_motion(
    frame: np.ndarray,
    previous_frame: np.ndarray
) -> float:
    """
    Measures how much motion is in this frame
    compared to the previous frame.

    Method: Frame difference
      Convert both frames to grayscale.
      Calculate absolute difference pixel by pixel.
      Average that difference across all pixels.
      Higher average = more motion.

    This score is used for content classification.
    High motion across many frames = sports video.
    Low motion across many frames = vlog or cinematic.
    """
    if previous_frame is None:
        return 0.0

    # Convert both frames to grayscale
    gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_previous = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

    # Calculate pixel difference
    diff = cv2.absdiff(gray_current, gray_previous)

    # Average difference across all pixels
    # Normalize to 0.0 - 1.0
    # Max possible difference per pixel is 255
    motion_score = np.mean(diff) / 255.0

    return motion_score


def is_transition_frame(
    frame: np.ndarray,
    previous_frame: np.ndarray,
    threshold: float = 0.4
) -> bool:
    """
    Detects if this frame is a scene transition.

    Method: Histogram comparison
      A histogram shows the color distribution of an image.
      During a scene cut the colors change dramatically.
      We compare histograms of current and previous frame.
      If they are very different = transition = discard.

    Returns True if this is a transition frame.
    Returns False if this is a normal frame.
    """
    if previous_frame is None:
        return False

    # Calculate histogram for current frame
    hist_current = cv2.calcHist(
        [frame], [0, 1, 2], None,
        [8, 8, 8],          # 8 bins per channel
        [0, 256, 0, 256, 0, 256]
    )
    cv2.normalize(hist_current, hist_current)

    # Calculate histogram for previous frame
    hist_previous = cv2.calcHist(
        [previous_frame], [0, 1, 2], None,
        [8, 8, 8],
        [0, 256, 0, 256, 0, 256]
    )
    cv2.normalize(hist_previous, hist_previous)

    # Compare histograms
    # HISTCMP_CORREL returns value from -1 to 1
    # 1.0  = identical histograms (same scene)
    # 0.0  = no correlation (different scene)
    # -1.0 = opposite (very different)
    correlation = cv2.compareHist(
        hist_current,
        hist_previous,
        cv2.HISTCMP_CORREL
    )

    # If correlation is below threshold = transition
    return correlation < (1.0 - threshold)


# ═════════════════════════════════════════════════════════
# SECTION 2 — CONTENT CLASSIFICATION
# ═════════════════════════════════════════════════════════

def classify_content_type(
    avg_face_score: float,
    avg_motion_score: float,
    user_tag: str = None,
) -> str:
    """
    Determines what type of content the video is.

    First checks if user provided a tag during upload.
    User always knows their content better than algorithm.

    If no user tag: uses face and motion averages
    to make a best guess.
    """

    # User tag takes priority always
    if user_tag and user_tag in CONTENT_WEIGHTS:
        return user_tag

    # Classify based on scores
    if avg_face_score > 0.3 and avg_motion_score < 0.2:
        return "vlog"           # People talking, not much movement
    elif avg_motion_score > 0.3 and avg_face_score < 0.2:
        return "sports"         # Lots of movement, no clear faces
    elif avg_face_score < 0.2 and avg_motion_score < 0.2:
        return "cinematic"      # Calm, scenic, no people
    else:
        return "general"        # Mixed or unclear


# ═════════════════════════════════════════════════════════
# SECTION 3 — FRAME SELECTION
# ═════════════════════════════════════════════════════════

def select_top_5_frames(
    scored_frames: List[dict],
    duration: float,
) -> List[np.ndarray]:
    """
    Selects the best 5 frames spread across the video.

    Why spread them?
      If all top 5 are from minute 2 of a 10 minute video
      user gets 5 almost identical thumbnails.
      No real choice.

    How spreading works:
      Divide video into 5 equal sections.
      Pick the best scoring frame from each section.
      User gets one great thumbnail from each part of video.
    """

    if not scored_frames:
        return []

    # Divide video duration into 5 equal sections
    section_size = duration / 5
    selected_frames = []

    for section_index in range(5):
        # Calculate time range for this section
        section_start = section_index * section_size
        section_end = section_start + section_size

        # Find all frames that fall within this section
        section_frames = [
            f for f in scored_frames
            if section_start <= f["timestamp"] < section_end
        ]

        if section_frames:
            # Pick the highest scoring frame from this section
            best_in_section = max(
                section_frames,
                key=lambda f: f["final_score"]
            )
            selected_frames.append(best_in_section["frame"])

    # If we got fewer than 5 (short video edge case)
    # fill remaining slots from overall top scorers
    if len(selected_frames) < 5:
        already_selected = set(id(f) for f in selected_frames)
        remaining = [
            f for f in scored_frames
            if id(f["frame"]) not in already_selected
        ]
        remaining_sorted = sorted(
            remaining,
            key=lambda f: f["final_score"],
            reverse=True
        )
        for frame_data in remaining_sorted:
            if len(selected_frames) >= 5:
                break
            selected_frames.append(frame_data["frame"])

    return selected_frames[:5]


def resize_frame_to_thumbnail(frame: np.ndarray) -> np.ndarray:
    """
    Resizes a frame to standard thumbnail dimensions.

    Standard thumbnail size: 1280 x 720 pixels
    This is standard HD resolution.
    Same size YouTube uses for thumbnails.

    cv2.INTER_LANCZOS4 is the highest quality
    resize algorithm OpenCV provides.
    Slower than other methods but best visual result.
    """
    return cv2.resize(
        frame,
        (1280, 720),
        interpolation=cv2.INTER_LANCZOS4
    )


# ═════════════════════════════════════════════════════════
# SECTION 4 — MONGODB HELPERS
#
# Simple functions to update MongoDB status.
# Called throughout processing to keep
# frontend informed of progress.
# ═════════════════════════════════════════════════════════

def update_status(job_id: str, update_data: dict):
    """
    Updates the MongoDB document for this job.

    Called multiple times during processing:
      When processing starts    → status: processing
      During frame scoring      → progress: 10, 20, 30...
      When thumbnails uploaded  → status: done, 5 URLs
      If anything fails         → status: failed, error message
    """
    collection = get_videos_collection()

    # Run the async update in a sync context
    # Celery workers are synchronous
    # MongoDB motor driver is async
    # asyncio.get_event_loop().run_until_complete
    # bridges the two worlds
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(
            collection.update_one(
                {"job_id": job_id},
                {"$set": update_data}
            )
        )
    finally:
        loop.close()


# ═════════════════════════════════════════════════════════
# SECTION 5 — THE MAIN CELERY TASK
#
# This is the function that becomes a Celery task.
# @celery_app.task decoration means:
#   When someone calls process_video.delay(...)
#   This function does not run immediately.
#   It gets sent to Redis as a job.
#   A worker picks it up and runs it here.
# ═════════════════════════════════════════════════════════

@celery_app.task(
    name="process_video",
    bind=True,          # Gives us access to self (the task instance)
    max_retries=0,      # No automatic retries, user uses re-generate
)
def process_video(
    self,
    job_id: str,
    user_id: str,
    s3_raw_key: str,
    content_type_user_tag: str = None,
):
    """
    The main video processing task.

    This is what every Celery worker runs
    when it picks up a job from Redis.

    Parameters:
      job_id               unique ID for this job
      user_id              who uploaded the video
      s3_raw_key           path to raw video in S3
      content_type_user_tag optional tag from user
    """

    try:
        # ── STAGE 1: Mark as processing ──────────────────
        update_status(job_id, {
            "status": "processing",
            "progress_percent": 5,
        })

        # ── STAGE 2: Get video URL for FFmpeg ────────────
        # Generate pre-signed URL so FFmpeg can access
        # the private video in S3
        s3_url = get_raw_video_url(s3_raw_key)

        update_status(job_id, {"progress_percent": 10})

        # ── STAGE 3: Read video metadata ─────────────────
        metadata = get_video_metadata(s3_url)
        duration = metadata["duration"]
        width = metadata["width"]
        height = metadata["height"]

        # Save metadata to MongoDB
        # These fields are used for subscription checks later
        update_status(job_id, {
            "duration_seconds": duration,
            "format": metadata["format"],
            "resolution": f"{width}x{height}",
            "progress_percent": 15,
        })

        # ── STAGE 4: Calculate frame sample count ────────
        frame_count = calculate_frame_count(duration)

        # ── STAGE 5: Stream and score every frame ────────
        scored_frames = []      # All frames with their scores
        previous_frame = None   # Needed for motion and transition detection

        # Accumulators for content classification
        total_face_score = 0.0
        total_motion_score = 0.0
        frames_processed = 0

        for frame, timestamp in stream_frames_from_s3(
            s3_url, duration, width, height, frame_count
        ):
            # TEST 1: Is this a transition frame?
            # If yes: skip it entirely, do not score it
            if is_transition_frame(frame, previous_frame):
                previous_frame = frame
                continue

            # TEST 2: Score this frame on all criteria
            blur = score_blur(frame)
            brightness = score_brightness(frame)
            face = score_faces(frame)
            motion = score_motion(frame, previous_frame)

            # Accumulate for content classification later
            total_face_score += face
            total_motion_score += motion
            frames_processed += 1

            # Store frame with its individual scores and timestamp
            scored_frames.append({
                "frame": frame,
                "timestamp": timestamp,
                "scores": {
                    "blur": blur,
                    "brightness": brightness,
                    "face": face,
                    "motion": motion,
                },
                "final_score": 0.0,  # Calculated after classification
            })

            previous_frame = frame

            # Update progress as frames are processed
            # Progress goes from 15 to 70 during this stage
            progress = 15 + int(
                (frames_processed / frame_count) * 55
            )
            update_status(job_id, {"progress_percent": progress})

        # ── STAGE 6: Classify content type ───────────────
        avg_face = total_face_score / max(frames_processed, 1)
        avg_motion = total_motion_score / max(frames_processed, 1)

        content_type = classify_content_type(
            avg_face,
            avg_motion,
            content_type_user_tag,
        )

        update_status(job_id, {
            "content_type_detected": content_type,
            "progress_percent": 72,
        })

        # ── STAGE 7: Apply content-aware weights ──────────
        weights = CONTENT_WEIGHTS[content_type]

        for frame_data in scored_frames:
            s = frame_data["scores"]

            # Final score = weighted sum of all individual scores
            frame_data["final_score"] = (
                s["blur"]       * weights["blur"] +
                s["brightness"] * weights["brightness"] +
                s["face"]       * weights["face"] +
                s["motion"]     * weights["motion"]
            )

        update_status(job_id, {"progress_percent": 78})

        # ── STAGE 8: Select top 5 frames ─────────────────
        top_5_frames = select_top_5_frames(scored_frames, duration)

        if not top_5_frames:
            raise RuntimeError(
                "Could not extract any valid frames from video"
            )

        update_status(job_id, {"progress_percent": 82})

        # ── STAGE 9: Resize frames to thumbnail size ──────
        resized_frames = [
            resize_frame_to_thumbnail(frame)
            for frame in top_5_frames
        ]

        update_status(job_id, {"progress_percent": 88})

        # ── STAGE 10: Upload thumbnails to S3 ────────────
        thumbnail_urls = upload_thumbnails_to_s3(
            resized_frames,
            user_id,
            job_id,
        )

        update_status(job_id, {"progress_percent": 95})

        # ── STAGE 11: Mark as done ────────────────────────
        update_status(job_id, {
            "status": "done",
            "thumbnail_candidates": thumbnail_urls,
            "progress_percent": 100,
            "processed_at": datetime.utcnow().isoformat(),
        })

    except Exception as error:
        # ── FAILURE HANDLER ───────────────────────────────
        # If ANYTHING above fails we land here.
        # We update MongoDB with failed status.
        # Worker is then free for the next job.
        # Other videos are completely unaffected.
        update_status(job_id, {
            "status": "failed",
            "error_message": str(error),
            "progress_percent": 0,
            "processed_at": datetime.utcnow().isoformat(),
        })

        # Re-raise so Celery knows this task failed
        raise