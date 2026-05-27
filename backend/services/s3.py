import os
import shutil
import cv2
import numpy as np
from fastapi import UploadFile
from typing import List
from backend.config import settings


# ─────────────────────────────────────────────────────────
# LOCAL STORAGE SERVICE
#
# This is a LOCAL FILE SYSTEM replacement for AWS S3.
# All videos and thumbnails are stored on your machine.
#
# When AWS credentials are ready:
#   Replace this file with the original s3.py
#   Zero changes needed anywhere else in the project.
#
# Local folder structure mirrors S3 structure:
#   local_storage/
#     raw/{user_id}/{job_id}/original.ext
#     thumbnails/{user_id}/{job_id}/thumb_1..5.jpg
# ─────────────────────────────────────────────────────────

BASE_PATH = settings.LOCAL_STORAGE_PATH


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def build_raw_video_key(user_id: str, job_id: str, filename: str) -> str:
    extension = filename.rsplit(".", 1)[-1].lower()
    return f"raw/{user_id}/{job_id}/original.{extension}"


def build_thumbnail_key(user_id: str, job_id: str, thumb_number: int) -> str:
    return f"thumbnails/{user_id}/{job_id}/thumb_{thumb_number}.jpg"


def build_s3_url(s3_key: str) -> str:
    full_path = os.path.join(BASE_PATH, s3_key).replace("\\", "/")
    return f"file:///{full_path}"


def get_full_path(s3_key: str) -> str:
    return os.path.join(BASE_PATH, s3_key)


async def upload_video_to_s3(
    file: UploadFile,
    user_id: str,
    job_id: str,
) -> tuple:
    s3_key = build_raw_video_key(user_id, job_id, file.filename)
    full_path = get_full_path(s3_key)
    ensure_dir(os.path.dirname(full_path))

    chunk_size = 8 * 1024 * 1024
    total_bytes = 0

    with open(full_path, "wb") as f:
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            total_bytes += len(chunk)

    s3_url = build_s3_url(s3_key)
    print(f"Video saved locally: {full_path}")
    return s3_key, s3_url, total_bytes


def get_raw_video_url(s3_raw_key: str) -> str:
    full_path = get_full_path(s3_raw_key)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Video file not found: {full_path}")
    return full_path


def upload_thumbnails_to_s3(
    frames: List[np.ndarray],
    user_id: str,
    job_id: str,
) -> List[str]:
    thumbnail_urls = []

    for index, frame in enumerate(frames):
        thumb_number = index + 1
        success, jpeg_bytes = cv2.imencode(
            ".jpg", frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 92]
        )
        if not success:
            continue

        s3_key = build_thumbnail_key(user_id, job_id, thumb_number)
        full_path = get_full_path(s3_key)
        ensure_dir(os.path.dirname(full_path))

        with open(full_path, "wb") as f:
            f.write(jpeg_bytes.tobytes())

        thumbnail_url = build_s3_url(s3_key)
        thumbnail_urls.append(thumbnail_url)
        print(f"Thumbnail {thumb_number} saved: {full_path}")

    return thumbnail_urls


def delete_thumbnails_from_s3(user_id: str, job_id: str) -> None:
    for thumb_number in range(1, 6):
        s3_key = build_thumbnail_key(user_id, job_id, thumb_number)
        full_path = get_full_path(s3_key)
        if os.path.exists(full_path):
            os.remove(full_path)
            print(f"Deleted thumbnail: {full_path}")

    job_folder = os.path.join(BASE_PATH, "thumbnails", user_id, job_id)
    if os.path.exists(job_folder):
        try:
            os.rmdir(job_folder)
        except OSError:
            pass