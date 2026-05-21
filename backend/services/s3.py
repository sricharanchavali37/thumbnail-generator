import boto3
from botocore.exceptions import ClientError
from fastapi import UploadFile
from typing import List
import numpy as np
import cv2
import io
from backend.config import settings


# ─────────────────────────────────────────────────────────
# S3 CLIENT
#
# This is the connection to AWS S3.
# Created once when this file is imported.
# Reused for every S3 operation.
#
# boto3 is the official AWS library for Python.
# It reads our credentials from config:
#   AWS_ACCESS_KEY_ID
#   AWS_SECRET_ACCESS_KEY
#   AWS_REGION
# ─────────────────────────────────────────────────────────

s3_client = boto3.client(
    "s3",
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    region_name=settings.AWS_REGION,
)


# ─────────────────────────────────────────────────────────
# HELPER — BUILD S3 KEYS
#
# A key in S3 is the full path of a file inside the bucket.
# Like a file path on your computer but inside S3.
#
# raw_video_key builds the path for uploaded videos.
# thumbnail_key builds the path for each thumbnail.
# ─────────────────────────────────────────────────────────

def build_raw_video_key(user_id: str, job_id: str, filename: str) -> str:
    """
    Builds the S3 path where the raw uploaded video will be stored.

    Example output:
      raw/user_abc/job_123/original.mp4

    We always rename the file to original.{extension}
    so we always know exactly what to look for later.
    The original filename is saved in MongoDB separately.
    """
    # Extract the file extension from original filename
    # my_vacation.MP4 → .MP4 → .mp4
    extension = filename.rsplit(".", 1)[-1].lower()
    return f"raw/{user_id}/{job_id}/original.{extension}"


def build_thumbnail_key(user_id: str, job_id: str, thumb_number: int) -> str:
    """
    Builds the S3 path for one thumbnail image.

    Example output:
      thumbnails/user_abc/job_123/thumb_1.jpg

    thumb_number goes from 1 to 5.
    """
    return f"thumbnails/{user_id}/{job_id}/thumb_{thumb_number}.jpg"


def build_s3_url(s3_key: str) -> str:
    """
    Builds the full public URL for any S3 file.

    Example output:
      https://my-bucket.s3.ap-south-1.amazonaws.com/thumbnails/user_abc/job_123/thumb_1.jpg

    This URL is what gets saved in MongoDB.
    This URL is what frontend uses to display thumbnails.
    """
    return (
        f"https://{settings.S3_BUCKET_NAME}"
        f".s3.{settings.AWS_REGION}"
        f".amazonaws.com/{s3_key}"
    )


# ─────────────────────────────────────────────────────────
# FUNCTION 1 — UPLOAD RAW VIDEO TO S3
#
# Called by the router when user uploads a video.
# Streams the video chunk by chunk into S3.
# Never holds the full video in memory.
# ─────────────────────────────────────────────────────────

async def upload_video_to_s3(
    file: UploadFile,
    user_id: str,
    job_id: str,
) -> tuple[str, str, int]:
    """
    Uploads the raw video file to S3.

    How streaming works here:
      The video file arrives from frontend in chunks.
      We read one chunk at a time.
      We send that chunk to S3 immediately.
      We never hold the full video in memory.
      S3 assembles all chunks into the complete file.

    Returns:
      s3_key       → the path inside S3 bucket
      s3_url       → the full URL to access the file
      file_size    → total bytes uploaded
    """

    # Build where this video will live in S3
    s3_key = build_raw_video_key(user_id, job_id, file.filename)

    # We use S3 multipart upload for large files.
    # Multipart upload means:
    #   Tell S3 we are about to send a large file
    #   Send it in pieces
    #   Tell S3 we are done
    #   S3 puts the pieces together

    # Start the multipart upload
    # S3 gives us an upload_id to track this specific upload
    multipart = s3_client.create_multipart_upload(
        Bucket=settings.S3_BUCKET_NAME,
        Key=s3_key,
        ContentType=file.content_type or "video/mp4",
    )
    upload_id = multipart["UploadId"]

    parts = []          # Track each chunk we send
    part_number = 1     # S3 requires chunks to be numbered
    total_bytes = 0     # Track total file size
    chunk_size = 8 * 1024 * 1024  # 8MB per chunk
    # Why 8MB chunks?
    # Too small: too many network round trips, slow
    # Too large: uses too much memory
    # 8MB is the sweet spot for most video files

    try:
        while True:
            # Read one chunk from the incoming video stream
            chunk = await file.read(chunk_size)

            # Empty chunk means we have read the entire file
            if not chunk:
                break

            # Send this chunk to S3
            # S3 requires each chunk to be at least 5MB
            # except the very last chunk which can be smaller
            response = s3_client.upload_part(
                Bucket=settings.S3_BUCKET_NAME,
                Key=s3_key,
                PartNumber=part_number,
                UploadId=upload_id,
                Body=chunk,
            )

            # S3 gives back an ETag for each chunk
            # We collect all ETags to finalize the upload
            parts.append({
                "PartNumber": part_number,
                "ETag": response["ETag"],
            })

            total_bytes += len(chunk)
            part_number += 1

        # Tell S3 we are done sending chunks
        # S3 now assembles all chunks into one complete file
        s3_client.complete_multipart_upload(
            Bucket=settings.S3_BUCKET_NAME,
            Key=s3_key,
            UploadId=upload_id,
            MultipartUpload={"Parts": parts},
        )

    except Exception as error:
        # Something went wrong during upload
        # We MUST cancel the multipart upload
        # Otherwise S3 keeps the partial chunks forever
        # and charges you for that storage
        s3_client.abort_multipart_upload(
            Bucket=settings.S3_BUCKET_NAME,
            Key=s3_key,
            UploadId=upload_id,
        )
        # Re-raise the error so the router knows upload failed
        raise error

    s3_url = build_s3_url(s3_key)
    return s3_key, s3_url, total_bytes


# ─────────────────────────────────────────────────────────
# FUNCTION 2 — GET THE S3 URL FOR THE RAW VIDEO
#
# Called by the Celery worker when it starts processing.
# Worker needs the S3 URL so FFmpeg can stream from it.
# ─────────────────────────────────────────────────────────

def get_raw_video_url(s3_raw_key: str) -> str:
    """
    Returns a pre-signed URL for the raw video.

    A pre-signed URL is a temporary URL that gives
    time-limited access to a private S3 file.

    Why pre-signed and not public URL?
      Raw videos should not be publicly accessible.
      Anyone with the URL should not be able to
      download the user's original video.
      Pre-signed URLs expire after a set time.
      After expiry the URL stops working.

    We give it 3600 seconds = 1 hour.
    More than enough for FFmpeg to stream and process.
    """
    url = s3_client.generate_presigned_url(
        "get_object",
        Params={
            "Bucket": settings.S3_BUCKET_NAME,
            "Key": s3_raw_key,
        },
        ExpiresIn=3600,  # URL valid for 1 hour
    )
    return url


# ─────────────────────────────────────────────────────────
# FUNCTION 3 — UPLOAD THUMBNAILS TO S3
#
# Called by the Celery worker after scoring is done.
# Takes the 5 best frames as images.
# Uploads each one to S3.
# Returns the 5 public URLs.
# ─────────────────────────────────────────────────────────

def upload_thumbnails_to_s3(
    frames: List[np.ndarray],
    user_id: str,
    job_id: str,
) -> List[str]:
    """
    Uploads the 5 best frames as JPEG thumbnails to S3.

    frames is a list of 5 images.
    Each image is a numpy array (how OpenCV represents images).

    For each frame:
      Convert numpy array to JPEG bytes
      Upload those bytes to S3
      Build the public URL
      Add URL to our list

    Returns list of 5 public S3 URLs.
    """
    thumbnail_urls = []

    for index, frame in enumerate(frames):
        thumb_number = index + 1  # 1 to 5, not 0 to 4

        # OpenCV works with numpy arrays internally.
        # S3 needs bytes to upload.
        # We encode the numpy array as JPEG bytes here.
        #
        # cv2.imencode converts numpy array → JPEG bytes
        # [int(cv2.IMWRITE_JPEG_QUALITY), 92] sets quality to 92
        # 92 = high quality, reasonable file size
        # 100 = perfect quality, very large file
        # 70  = lower quality, small file but looks bad
        success, jpeg_bytes = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 92]
        )

        if not success:
            # If encoding fails skip this frame
            # Better to have 4 thumbnails than crash entirely
            continue

        # Convert to bytes object that boto3 can upload
        image_bytes = io.BytesIO(jpeg_bytes.tobytes())

        # Build the S3 path for this thumbnail
        s3_key = build_thumbnail_key(user_id, job_id, thumb_number)

        # Upload to S3
        # For thumbnails we use simple put_object not multipart
        # Because thumbnails are small (under 1MB each)
        # Multipart is only needed for large files
        s3_client.put_object(
            Bucket=settings.S3_BUCKET_NAME,
            Key=s3_key,
            Body=image_bytes,
            ContentType="image/jpeg",
        )

        # Build the public URL for this thumbnail
        # This URL goes into MongoDB
        # Frontend uses this URL to display the image
        thumbnail_url = build_s3_url(s3_key)
        thumbnail_urls.append(thumbnail_url)

    return thumbnail_urls


# ─────────────────────────────────────────────────────────
# FUNCTION 4 — DELETE OLD THUMBNAILS
#
# Called when user clicks re-generate on an old video.
# Deletes all 5 old thumbnails from S3.
# So the bucket does not fill up with old unused images.
# ─────────────────────────────────────────────────────────

def delete_thumbnails_from_s3(user_id: str, job_id: str) -> None:
    """
    Deletes all 5 thumbnail files for a specific job from S3.

    Called before re-generating thumbnails.
    Makes sure old thumbnails do not pile up in S3
    costing money for storage nobody needs.

    S3 lets us delete multiple files in one request.
    More efficient than deleting one by one.
    """

    # Build the list of all 5 thumbnail keys to delete
    keys_to_delete = [
        {"Key": build_thumbnail_key(user_id, job_id, thumb_number)}
        for thumb_number in range(1, 6)  # 1, 2, 3, 4, 5
    ]

    # Delete all 5 in one S3 request
    s3_client.delete_objects(
        Bucket=settings.S3_BUCKET_NAME,
        Delete={"Objects": keys_to_delete},
    )