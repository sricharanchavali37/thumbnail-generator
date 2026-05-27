from fastapi import APIRouter, Request, UploadFile, File, HTTPException, Query
from typing import List, Optional
import uuid
from datetime import datetime
from pydantic import BaseModel

from backend.database.mongodb import get_videos_collection
from backend.services.s3 import upload_video_to_s3
from backend.workers.video_processor import process_video
from backend.models.video import (
    VideoDocument,
    VideoStatus,
    SelectThumbnailRequest,
    VideoUploadResponse,
)
from backend.config import settings
from backend.services.thumbnail_enhancer import add_custom_text_to_thumbnail

import os

router = APIRouter(prefix="/videos", tags=["videos"])


# ─────────────────────────────────────────────────────────
# REQUEST MODEL FOR CUSTOM TEXT
# ─────────────────────────────────────────────────────────

class AddTextRequest(BaseModel):
    """
    What user sends when adding custom text to a thumbnail.

    thumbnail_url: the file URL of chosen thumbnail
                   from thumbnail_candidates list
    custom_text:   what user wants written on thumbnail
                   Example: "ROHIT SHARMA WORLD CUP FINAL"
    position:      where text appears
                   "top" / "centre" / "bottom"
                   Default is "bottom"
    """
    thumbnail_url: str
    custom_text: str
    position: str = "bottom"


# ─────────────────────────────────────────────────────────
# EXISTING ENDPOINTS — unchanged
# ─────────────────────────────────────────────────────────

@router.post("/upload", response_model=VideoUploadResponse)
async def upload_videos(
    request: Request,
    files: List[UploadFile] = File(...),
    content_type_tag: Optional[str] = None,
    thumbnail_title: Optional[str] = None,
):
    user_id = request.state.user_id
    collection = get_videos_collection()
    job_ids = []

    for file in files:
        job_id = str(uuid.uuid4())

        s3_key, s3_url, file_size = await upload_video_to_s3(
            file=file,
            user_id=user_id,
            job_id=job_id,
        )

        video_doc = VideoDocument(
            job_id=job_id,
            user_id=user_id,
            original_filename=file.filename,
            s3_raw_key=s3_key,
            file_size_bytes=file_size,
            status=VideoStatus.QUEUED,
        )

        await collection.insert_one(video_doc.dict())

        process_video.delay(
            job_id=job_id,
            user_id=user_id,
            s3_raw_key=s3_key,
            content_type_user_tag=content_type_tag,
            thumbnail_title=thumbnail_title,
        )

        job_ids.append(job_id)

    return VideoUploadResponse(
        job_ids=job_ids,
        message=f"{len(job_ids)} video(s) queued for processing",
    )


@router.get("/status")
async def get_status(
    request: Request,
    job_ids: str = Query(...),
):
    user_id = request.state.user_id
    collection = get_videos_collection()
    job_id_list = [jid.strip() for jid in job_ids.split(",")]

    results = []
    async for doc in collection.find(
        {"job_id": {"$in": job_id_list}, "user_id": user_id}
    ):
        results.append({
            "job_id": doc["job_id"],
            "status": doc["status"],
            "progress_percent": doc.get("progress_percent", 0),
            "thumbnail_candidates": doc.get("thumbnail_candidates", []),
            "chosen_thumbnail": doc.get("chosen_thumbnail"),
            "error_message": doc.get("error_message"),
        })

    return results


@router.patch("/{job_id}/select-thumbnail")
async def select_thumbnail(
    job_id: str,
    body: SelectThumbnailRequest,
    request: Request,
):
    user_id = request.state.user_id
    collection = get_videos_collection()

    result = await collection.update_one(
        {"job_id": job_id, "user_id": user_id},
        {"$set": {"chosen_thumbnail": body.thumbnail_url}}
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Job not found")

    return {"message": "Thumbnail saved successfully"}


@router.get("/history")
async def get_history(request: Request):
    user_id = request.state.user_id
    collection = get_videos_collection()

    results = []
    async for doc in collection.find(
        {"user_id": user_id}
    ).sort("uploaded_at", -1).limit(50):
        results.append({
            "job_id": doc["job_id"],
            "original_filename": doc.get("original_filename"),
            "status": doc["status"],
            "chosen_thumbnail": doc.get("chosen_thumbnail"),
            "thumbnail_candidates": doc.get("thumbnail_candidates", []),
            "uploaded_at": doc.get("uploaded_at"),
            "content_type_detected": doc.get("content_type_detected"),
        })

    return results


@router.post("/{job_id}/regenerate")
async def regenerate_thumbnails(
    job_id: str,
    request: Request,
):
    user_id = request.state.user_id
    collection = get_videos_collection()

    doc = await collection.find_one(
        {"job_id": job_id, "user_id": user_id}
    )

    if not doc:
        raise HTTPException(status_code=404, detail="Job not found")

    await collection.update_one(
        {"job_id": job_id},
        {"$set": {
            "status": VideoStatus.QUEUED,
            "progress_percent": 0,
            "thumbnail_candidates": [],
            "chosen_thumbnail": None,
            "error_message": None,
        }}
    )

    process_video.delay(
        job_id=job_id,
        user_id=user_id,
        s3_raw_key=doc["s3_raw_key"],
        content_type_user_tag=doc.get("content_type_detected"),
    )

    return {"message": "Regeneration started", "job_id": job_id}


@router.get("/{job_id}")
async def get_video(job_id: str, request: Request):
    user_id = request.state.user_id
    collection = get_videos_collection()

    doc = await collection.find_one(
        {"job_id": job_id, "user_id": user_id}
    )

    if not doc:
        raise HTTPException(status_code=404, detail="Job not found")

    return doc


# ─────────────────────────────────────────────────────────
# NEW ENDPOINT — ADD CUSTOM TEXT TO THUMBNAIL
#
# User picks one of their 5 thumbnails.
# User types what text they want on it.
# User chooses position: top / centre / bottom.
# System adds yellow bold text in Ranveer Show style.
# Saves as final_thumb.jpg in same folder.
# Original thumbnail files are NOT changed.
# Both versions available to user.
# ─────────────────────────────────────────────────────────

@router.post("/{job_id}/add-text")
async def add_text_to_thumbnail(
    job_id: str,
    body: AddTextRequest,
    request: Request,
):
    """
    Adds user's custom text onto their chosen thumbnail.

    Takes:
        job_id        → which video job
        thumbnail_url → which of the 5 thumbnails to use
        custom_text   → what text to write on it
        position      → top / centre / bottom

    Does:
        Converts file:/// URL to actual disk path
        Opens the thumbnail JPEG file
        Adds yellow bold text in Ranveer Show style
        Saves as final_thumb.jpg (original unchanged)
        Returns URL of the new final thumbnail

    Returns:
        final_thumbnail_url → URL of the customised thumbnail
        message             → confirmation
    """
    user_id = request.state.user_id
    collection = get_videos_collection()

    # Verify this job belongs to this user
    doc = await collection.find_one(
        {"job_id": job_id, "user_id": user_id}
    )

    if not doc:
        raise HTTPException(status_code=404, detail="Job not found")

    # Validate position field
    valid_positions = ["top", "centre", "bottom"]
    if body.position not in valid_positions:
        raise HTTPException(
            status_code=400,
            detail=f"Position must be one of: {valid_positions}"
        )

    # Validate custom text is not empty
    if not body.custom_text or not body.custom_text.strip():
        raise HTTPException(
            status_code=400,
            detail="custom_text cannot be empty"
        )

    # Convert file:/// URL to actual disk path
    # file:///D:/mnr thumbnail/.../thumb_2.jpg
    # → D:/mnr thumbnail/.../thumb_2.jpg
    thumbnail_url = body.thumbnail_url
    if thumbnail_url.startswith("file:///"):
        thumbnail_path = thumbnail_url.replace("file:///", "")
        thumbnail_path = thumbnail_path.replace("/", os.sep)
    else:
        thumbnail_path = thumbnail_url

    # Check file exists on disk
    if not os.path.exists(thumbnail_path):
        raise HTTPException(
            status_code=404,
            detail="Thumbnail file not found on disk"
        )

    # Build output path for final thumbnail
    # Saved in same folder as original thumbnails
    folder = os.path.dirname(thumbnail_path)
    output_path = os.path.join(folder, "final_thumb.jpg")

    # Add custom text using Pillow
    try:
        final_path = add_custom_text_to_thumbnail(
            thumbnail_path=thumbnail_path,
            custom_text=body.custom_text,
            position=body.position,
            output_path=output_path,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to add text: {str(e)}"
        )

    # Build URL for the final thumbnail
    final_url = "file:///" + final_path.replace(os.sep, "/")

    # Save final thumbnail URL to MongoDB
    await collection.update_one(
        {"job_id": job_id},
        {"$set": {
            "final_thumbnail": final_url,
            "custom_text": body.custom_text,
            "text_position": body.position,
        }}
    )

    return {
        "final_thumbnail_url": final_url,
        "message": "Text added successfully",
        "custom_text": body.custom_text,
        "position": body.position,
    }
