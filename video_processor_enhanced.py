import cv2
import numpy as np
from datetime import datetime
from typing import List, Tuple
from pymongo import MongoClient

from backend.workers.celery_app import celery_app
from backend.services.s3 import (
    get_raw_video_url,
    upload_thumbnails_to_s3,
)
from backend.utils.ffmpeg_helper import (
    get_video_metadata,
    calculate_frame_count,
    stream_frames_from_s3,
)
from backend.services.thumbnail_enhancer import enhance_thumbnail


# ─────────────────────────────────────────────────────────
# CONTENT AWARE SCORING WEIGHTS
# ─────────────────────────────────────────────────────────

CONTENT_WEIGHTS = {
    "vlog": {
        "blur": 0.25,
        "brightness": 0.20,
        "face": 0.40,
        "motion": 0.15,
    },
    "sports": {
        "blur": 0.20,
        "brightness": 0.20,
        "face": 0.10,
        "motion": 0.50,
    },
    "cinematic": {
        "blur": 0.25,
        "brightness": 0.50,
        "face": 0.10,
        "motion": 0.15,
    },
    "general": {
        "blur": 0.25,
        "brightness": 0.25,
        "face": 0.25,
        "motion": 0.25,
    },
}


# ─────────────────────────────────────────────────────────
# FACE DETECTOR
# ─────────────────────────────────────────────────────────

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ═════════════════════════════════════════════════════════
# FRAME SCORING FUNCTIONS
# ═════════════════════════════════════════════════════════

def score_blur(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return min(laplacian_variance / 500.0, 1.0)


def score_brightness(frame: np.ndarray) -> float:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    avg_brightness = np.mean(v_channel)

    if 80 <= avg_brightness <= 170:
        score = 1.0 - abs(avg_brightness - 125) / 90.0
    elif avg_brightness < 80:
        score = avg_brightness / 80.0
    else:
        score = (255 - avg_brightness) / 85.0

    return max(0.0, min(score, 1.0))


def score_faces(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
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
    if previous_frame is None:
        return 0.0
    gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_previous = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_current, gray_previous)
    return np.mean(diff) / 255.0


def is_transition_frame(
    frame: np.ndarray,
    previous_frame: np.ndarray,
    threshold: float = 0.4
) -> bool:
    if previous_frame is None:
        return False

    hist_current = cv2.calcHist(
        [frame], [0, 1, 2], None,
        [8, 8, 8],
        [0, 256, 0, 256, 0, 256]
    )
    cv2.normalize(hist_current, hist_current)

    hist_previous = cv2.calcHist(
        [previous_frame], [0, 1, 2], None,
        [8, 8, 8],
        [0, 256, 0, 256, 0, 256]
    )
    cv2.normalize(hist_previous, hist_previous)

    correlation = cv2.compareHist(
        hist_current,
        hist_previous,
        cv2.HISTCMP_CORREL
    )

    return correlation < (1.0 - threshold)


# ═════════════════════════════════════════════════════════
# CONTENT CLASSIFICATION
# ═════════════════════════════════════════════════════════

def classify_content_type(
    avg_face_score: float,
    avg_motion_score: float,
    user_tag: str = None,
) -> str:
    if user_tag and user_tag in CONTENT_WEIGHTS:
        return user_tag

    if avg_face_score > 0.3 and avg_motion_score < 0.2:
        return "vlog"
    elif avg_motion_score > 0.3 and avg_face_score < 0.2:
        return "sports"
    elif avg_face_score < 0.2 and avg_motion_score < 0.2:
        return "cinematic"
    else:
        return "general"


# ═════════════════════════════════════════════════════════
# FRAME SELECTION AND EXPORT
# ═════════════════════════════════════════════════════════

def select_top_5_frames(
    scored_frames: List[dict],
    duration: float,
) -> List[np.ndarray]:
    if not scored_frames:
        return []

    section_size = duration / 5
    selected_frames = []

    for section_index in range(5):
        section_start = section_index * section_size
        section_end = section_start + section_size

        section_frames = [
            f for f in scored_frames
            if section_start <= f["timestamp"] < section_end
        ]

        if section_frames:
            best_in_section = max(
                section_frames,
                key=lambda f: f["final_score"]
            )
            selected_frames.append(best_in_section["frame"])

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
    Smart crop: centre crop to remove black bars.
    No letterboxing. Content fills full 1280x720.
    """
    h, w = frame.shape[:2]
    target_w, target_h = 1280, 720
    target_ratio = target_w / target_h
    current_ratio = w / h

    if current_ratio > target_ratio:
        new_w = int(h * target_ratio)
        start_x = (w - new_w) // 2
        frame = frame[:, start_x:start_x + new_w]
    else:
        new_h = int(w / target_ratio)
        start_y = (h - new_h) // 2
        frame = frame[start_y:start_y + new_h, :]

    return cv2.resize(
        frame,
        (target_w, target_h),
        interpolation=cv2.INTER_LANCZOS4
    )


# ═════════════════════════════════════════════════════════
# MONGODB UPDATE HELPER
# ═════════════════════════════════════════════════════════

def update_status(job_id: str, update_data: dict):
    from backend.config import settings

    client = MongoClient(settings.MONGODB_URL)
    db = client[settings.MONGODB_DB_NAME]
    collection = db["videos"]

    collection.update_one(
        {"job_id": job_id},
        {"$set": update_data}
    )
    client.close()


# ═════════════════════════════════════════════════════════
# MAIN CELERY TASK
# ═════════════════════════════════════════════════════════

@celery_app.task(
    name="process_video",
    bind=True,
    max_retries=0,
)
def process_video(
    self,
    job_id: str,
    user_id: str,
    s3_raw_key: str,
    content_type_user_tag: str = None,
    thumbnail_title: str = None,        # ← NEW FIELD
):
    """
    Main video processing task.

    thumbnail_title is the text user wants on their thumbnail.
    Example: "INDIA VS AUSTRALIA HIGHLIGHTS"
    If None: enhancement still applied but no title text shown.
    """

    try:
        # ── STAGE 1: Mark as processing ──────────────────
        update_status(job_id, {
            "status": "processing",
            "progress_percent": 5,
        })

        # ── STAGE 2: Get video path ───────────────────────
        s3_url = get_raw_video_url(s3_raw_key)
        update_status(job_id, {"progress_percent": 10})

        # ── STAGE 3: Read video metadata ─────────────────
        metadata = get_video_metadata(s3_url)
        duration = metadata["duration"]
        width = metadata["width"]
        height = metadata["height"]

        update_status(job_id, {
            "duration_seconds": duration,
            "format": metadata["format"],
            "resolution": f"{width}x{height}",
            "progress_percent": 15,
        })

        # ── STAGE 4: Calculate frame count ───────────────
        frame_count = calculate_frame_count(duration)

        # ── STAGE 5: Stream and score frames ─────────────
        scored_frames = []
        previous_frame = None
        total_face_score = 0.0
        total_motion_score = 0.0
        frames_processed = 0

        for frame, timestamp in stream_frames_from_s3(
            s3_url, duration, width, height, frame_count
        ):
            if is_transition_frame(frame, previous_frame):
                previous_frame = frame
                continue

            blur = score_blur(frame)
            brightness = score_brightness(frame)
            face = score_faces(frame)
            motion = score_motion(frame, previous_frame)

            total_face_score += face
            total_motion_score += motion
            frames_processed += 1

            scored_frames.append({
                "frame": frame,
                "timestamp": timestamp,
                "scores": {
                    "blur": blur,
                    "brightness": brightness,
                    "face": face,
                    "motion": motion,
                },
                "final_score": 0.0,
            })

            previous_frame = frame

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

        # ── STAGE 7: Apply content weights ───────────────
        weights = CONTENT_WEIGHTS[content_type]

        for frame_data in scored_frames:
            s = frame_data["scores"]
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

        # ── STAGE 9: Smart crop and resize ───────────────
        resized_frames = [
            resize_frame_to_thumbnail(frame)
            for frame in top_5_frames
        ]

        update_status(job_id, {"progress_percent": 86})

        # ── STAGE 10: ENHANCE THUMBNAILS ─────────────────
        # NEW STEP: Apply professional design to each frame.
        # Brightness boost + gradient + badge + title + border.
        enhanced_frames = [
            enhance_thumbnail(
                frame_bgr=frame,
                content_type=content_type,
                title=thumbnail_title,
            )
            for frame in resized_frames
        ]

        update_status(job_id, {"progress_percent": 92})

        # ── STAGE 11: Save thumbnails ─────────────────────
        thumbnail_urls = upload_thumbnails_to_s3(
            enhanced_frames,          # ← enhanced not raw
            user_id,
            job_id,
        )

        update_status(job_id, {"progress_percent": 96})

        # ── STAGE 12: Mark as done ────────────────────────
        update_status(job_id, {
            "status": "done",
            "thumbnail_candidates": thumbnail_urls,
            "progress_percent": 100,
            "processed_at": datetime.utcnow().isoformat(),
        })

    except Exception as error:
        update_status(job_id, {
            "status": "failed",
            "error_message": str(error),
            "progress_percent": 0,
            "processed_at": datetime.utcnow().isoformat(),
        })
        raise