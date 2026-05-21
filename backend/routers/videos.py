from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from typing import List
import uuid

from backend.database.mongodb import get_videos_collection
from backend.models.video import (
    VideoDocument,
    SelectThumbnailRequest,
    VideoUploadResponse,
    VideoStatusResponse,
    VideoHistoryItem,
)
from backend.services.s3 import upload_video_to_s3, delete_thumbnails_from_s3
from backend.workers.video_processor import process_video

router = APIRouter(prefix="/videos", tags=["videos"])


# ─────────────────────────────────────────────────────────
# ENDPOINT 1 — UPLOAD VIDEOS
#
# POST /videos/upload
#
# Frontend sends one or more video files here.
# For each video:
#   Stream it to S3
#   Create MongoDB record
#   Fire Celery task
# Return all job_ids immediately.
# Frontend uses job_ids to start polling.
# ─────────────────────────────────────────────────────────

@router.post("/upload", response_model=VideoUploadResponse)
async def upload_videos(
    request: Request,
    files: List[UploadFile] = File(...),
    content_type_tag: str = None,
):
    """
    Upload one or more videos for thumbnail generation.

    Accepts multiple files in one request.
    Each file gets its own job_id.
    Processing starts immediately in background.
    Returns job_ids for polling.
    """
    # Get user_id attached by auth middleware
    user_id = request.state.user_id
    collection = get_videos_collection()
    job_ids = []

    for file in files:
        # Generate unique job_id for this video
        job_id = str(uuid.uuid4())

        # Stream video directly to S3
        # Never saves full video to disk
        s3_key, s3_url, file_size = await upload_video_to_s3(
            file=file,
            user_id=user_id,
            job_id=job_id,
        )

        # Create MongoDB record for this video
        video_doc = VideoDocument(
            job_id=job_id,
            user_id=user_id,
            original_filename=file.filename,
            s3_raw_key=s3_key,
            file_size_bytes=file_size,
            content_type_user_tag=content_type_tag,
        )

        # Insert into MongoDB
        await collection.insert_one(video_doc.dict())

        # Fire Celery task
        # .delay() means: put this in Redis queue now
        # Do not wait for it to finish
        # Worker picks it up and runs it independently
        process_video.delay(
            job_id=job_id,
            user_id=user_id,
            s3_raw_key=s3_key,
            content_type_user_tag=content_type_tag,
        )

        job_ids.append(job_id)

    return VideoUploadResponse(
        job_ids=job_ids,
        message=f"{len(job_ids)} video(s) queued for processing"
    )


# ─────────────────────────────────────────────────────────
# ENDPOINT 2 — BATCH STATUS POLLING
#
# GET /videos/status?job_ids=id1,id2,id3
#
# Frontend calls this every 3 seconds.
# Returns status of all requested jobs in one response.
# One request for all jobs. Not one request per job.
# ─────────────────────────────────────────────────────────

@router.get("/status")
async def get_status(
    request: Request,
    job_ids: str,   # Comma separated job IDs from query string
):
    """
    Get status of multiple jobs in one request.

    Called by frontend every 3 seconds.
    Returns status, progress, and thumbnails for each job.
    Frontend updates each card on dashboard from this response.
    """
    user_id = request.state.user_id
    collection = get_videos_collection()

    # Split comma separated job_ids into a list
    # "id1,id2,id3" → ["id1", "id2", "id3"]
    job_id_list = [jid.strip() for jid in job_ids.split(",")]

    # Fetch all jobs in one MongoDB query
    # $in means: find documents where job_id is in this list
    cursor = collection.find({
        "job_id": {"$in": job_id_list},
        "user_id": user_id,     # Security: only return this user's jobs
    })

    results = []
    async for doc in cursor:
        results.append(VideoStatusResponse(
            job_id=doc["job_id"],
            status=doc["status"],
            progress_percent=doc.get("progress_percent", 0),
            thumbnail_candidates=doc.get("thumbnail_candidates", []),
            chosen_thumbnail=doc.get("chosen_thumbnail"),
            error_message=doc.get("error_message"),
        ))

    return results


# ─────────────────────────────────────────────────────────
# ENDPOINT 3 — SELECT THUMBNAIL
#
# PATCH /videos/{job_id}/select-thumbnail
#
# User picks one of the 5 thumbnails.
# We save their choice to MongoDB permanently.
# ─────────────────────────────────────────────────────────

@router.patch("/{job_id}/select-thumbnail")
async def select_thumbnail(
    job_id: str,
    body: SelectThumbnailRequest,
    request: Request,
):
    """
    Save the user's chosen thumbnail.

    Called when user clicks one of the 5 thumbnail options.
    Saves the chosen thumbnail URL to MongoDB.
    This is what appears in history forever.
    """
    user_id = request.state.user_id
    collection = get_videos_collection()

    result = await collection.update_one(
        {
            "job_id": job_id,
            "user_id": user_id,     # Security: only update own videos
        },
        {"$set": {"chosen_thumbnail": body.thumbnail_url}}
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Video not found")

    return {"message": "Thumbnail saved successfully"}


# ─────────────────────────────────────────────────────────
# ENDPOINT 4 — HISTORY
#
# GET /videos/history
#
# Returns all past videos for this user.
# Sorted newest first.
# Used for the History page.
# ─────────────────────────────────────────────────────────

@router.get("/history")
async def get_history(request: Request):
    """
    Get all past videos for the current user.

    Returns every video ever uploaded by this user.
    Sorted by upload date newest first.
    Each item includes chosen thumbnail and status.
    """
    user_id = request.state.user_id
    collection = get_videos_collection()

    # Fetch all videos for this user
    # Sort by uploaded_at descending (-1 = newest first)
    cursor = collection.find(
        {"user_id": user_id}
    ).sort("uploaded_at", -1)

    history = []
    async for doc in cursor:
        history.append(VideoHistoryItem(
            job_id=doc["job_id"],
            original_filename=doc["original_filename"],
            status=doc["status"],
            chosen_thumbnail=doc.get("chosen_thumbnail"),
            thumbnail_candidates=doc.get("thumbnail_candidates", []),
            uploaded_at=doc["uploaded_at"],
            processed_at=doc.get("processed_at"),
            content_type_detected=doc.get("content_type_detected"),
            duration_seconds=doc.get("duration_seconds"),
        ))

    return history


# ─────────────────────────────────────────────────────────
# ENDPOINT 5 — REGENERATE
#
# POST /videos/{job_id}/regenerate
#
# User wants fresh thumbnails for an old video.
# Delete old thumbnails from S3.
# Reset status in MongoDB.
# Fire new Celery task.
# ─────────────────────────────────────────────────────────

@router.post("/{job_id}/regenerate")
async def regenerate_thumbnails(
    job_id: str,
    request: Request,
):
    """
    Re-generate thumbnails for an existing video.

    Deletes old thumbnails from S3.
    Resets job status to queued.
    Fires a new processing task.
    Frontend resumes polling for this job.
    """
    user_id = request.state.user_id
    collection = get_videos_collection()

    # Find the existing video record
    doc = await collection.find_one({
        "job_id": job_id,
        "user_id": user_id,
    })

    if not doc:
        raise HTTPException(status_code=404, detail="Video not found")

    # Delete old thumbnails from S3
    delete_thumbnails_from_s3(user_id, job_id)

    # Reset MongoDB record for fresh processing
    await collection.update_one(
        {"job_id": job_id},
        {"$set": {
            "status": "queued",
            "progress_percent": 0,
            "thumbnail_candidates": [],
            "chosen_thumbnail": None,
            "error_message": None,
            "processed_at": None,
            "content_type_detected": None,
        }}
    )

    # Fire new Celery task with same video
    process_video.delay(
        job_id=job_id,
        user_id=user_id,
        s3_raw_key=doc["s3_raw_key"],
        content_type_user_tag=doc.get("content_type_user_tag"),
    )

    return {"message": "Re-generation started", "job_id": job_id}


# ─────────────────────────────────────────────────────────
# ENDPOINT 6 — SINGLE VIDEO DETAIL
#
# GET /videos/{job_id}
#
# Returns full detail of one specific video.
# ─────────────────────────────────────────────────────────

@router.get("/{job_id}")
async def get_video(job_id: str, request: Request):
    """
    Get full details of one specific video job.
    """
    user_id = request.state.user_id
    collection = get_videos_collection()

    doc = await collection.find_one({
        "job_id": job_id,
        "user_id": user_id,
    })

    if not doc:
        raise HTTPException(status_code=404, detail="Video not found")

    return VideoStatusResponse(
        job_id=doc["job_id"],
        status=doc["status"],
        progress_percent=doc.get("progress_percent", 0),
        thumbnail_candidates=doc.get("thumbnail_candidates", []),
        chosen_thumbnail=doc.get("chosen_thumbnail"),
        error_message=doc.get("error_message"),
    )