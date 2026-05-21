from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum
import uuid


# ─────────────────────────────────────────────────────────
# ENUMS
#
# Enums are a way of defining a fixed set of allowed values.
# Instead of storing raw strings like "queued" everywhere
# we define an Enum so the allowed values are enforced.
#
# If someone tries to set status = "cooking"
# Python will raise an error immediately.
# Only the values defined below are allowed.
# ─────────────────────────────────────────────────────────

class VideoStatus(str, Enum):
    """
    The possible states a video job can be in.

    queued      → job created, waiting for a worker to pick it up
    processing  → a worker has picked it up and is working on it
    done        → processing finished, 5 thumbnails are ready
    failed      → something went wrong, error_message has details
    """
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class ContentType(str, Enum):
    """
    The type of content detected in the video.

    vlog      → person talking, face dominant
    sports    → action and motion dominant
    cinematic → scenic, calm, no people
    general   → mixed or unclear
    """
    VLOG = "vlog"
    SPORTS = "sports"
    CINEMATIC = "cinematic"
    GENERAL = "general"


# ─────────────────────────────────────────────────────────
# MAIN VIDEO MODEL
#
# This is the shape of every video document in MongoDB.
# Every field is defined here.
# ─────────────────────────────────────────────────────────

class VideoDocument(BaseModel):
    """
    Represents one video record in MongoDB.

    One document is created per video uploaded.
    This document is updated throughout processing.
    It holds the complete history of the video job.
    """

    # ── Identity ─────────────────────────────────────────
    job_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique ID for this job. Used for polling."
    )
    # uuid4() generates a random unique ID
    # Example: 550e8400-e29b-41d4-a716-446655440000
    # This is what React uses to track this specific video

    user_id: str = Field(
        description="Who uploaded this video. Decoded from JWT header."
    )

    # ── File Information ──────────────────────────────────
    original_filename: str = Field(
        description="The original name of the uploaded file."
        # Example: my_vacation_video.mp4
    )

    s3_raw_key: str = Field(
        description="Full S3 path where the raw video is stored."
        # Example: raw/user123/job456/original.mp4
    )

    file_size_bytes: Optional[int] = Field(
        default=None,
        description="Size of the video file in bytes. Stored for future subscription limits."
    )

    duration_seconds: Optional[float] = Field(
        default=None,
        description="Duration of video in seconds. Read by FFmpeg. Stored for subscription limits."
    )

    format: Optional[str] = Field(
        default=None,
        description="Video format detected by FFmpeg. Example: mp4, mov, avi"
    )

    resolution: Optional[str] = Field(
        default=None,
        description="Video resolution. Example: 1920x1080"
    )

    # ── Content Detection ─────────────────────────────────
    content_type_detected: Optional[ContentType] = Field(
        default=None,
        description="Content type our algorithm classified. Set after processing."
    )

    content_type_user_tag: Optional[str] = Field(
        default=None,
        description="Content type the user told us during upload. We trust this over our detection."
    )

    # ── Job Status ────────────────────────────────────────
    status: VideoStatus = Field(
        default=VideoStatus.QUEUED,
        description="Current state of this job."
    )
    # Starts as queued the moment the record is created
    # Worker changes it to processing when it starts
    # Worker changes it to done when finished
    # Worker changes it to failed if something breaks

    error_message: Optional[str] = Field(
        default=None,
        description="If status is failed, this explains why."
    )

    progress_percent: int = Field(
        default=0,
        description="How far along processing is. 0 to 100. Drives the progress bar on frontend."
    )

    # ── Thumbnails ────────────────────────────────────────
    thumbnail_candidates: List[str] = Field(
        default=[],
        description="The 5 S3 URLs of generated thumbnail options. Empty until status is done."
    )
    # Example:
    # [
    #   "https://s3.amazonaws.com/bucket/thumbnails/user123/job456/thumb_1.jpg",
    #   "https://s3.amazonaws.com/bucket/thumbnails/user123/job456/thumb_2.jpg",
    #   ...
    # ]

    chosen_thumbnail: Optional[str] = Field(
        default=None,
        description="The S3 URL of the thumbnail the user picked. Saved when user makes their choice."
    )

    # ── Timestamps ────────────────────────────────────────
    uploaded_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the video was uploaded. Set automatically."
    )

    processed_at: Optional[datetime] = Field(
        default=None,
        description="When processing finished. Set by worker when done or failed."
    )

    # ── Future Use ────────────────────────────────────────
    subscription_tier: str = Field(
        default="free",
        description="User subscription tier at time of upload. Hook for future limits."
    )
    # This field exists now so the data is there when we need it.
    # Currently all users are free tier.
    # When subscriptions are introduced we read this field
    # and enforce limits without changing the data structure.

    class Config:
        # Allow using Enum values directly
        use_enum_values = True


# ─────────────────────────────────────────────────────────
# REQUEST MODELS
#
# These define what the API expects to receive
# from the frontend in request bodies.
# ─────────────────────────────────────────────────────────

class SelectThumbnailRequest(BaseModel):
    """
    What frontend sends when user picks a thumbnail.

    PATCH /videos/{job_id}/select-thumbnail
    Body: { "thumbnail_url": "https://..." }
    """
    thumbnail_url: str = Field(
        description="The S3 URL of the thumbnail the user chose."
    )


# ─────────────────────────────────────────────────────────
# RESPONSE MODELS
#
# These define what our API sends back to frontend.
# We never send the raw MongoDB document.
# We always send a clean, controlled response.
# ─────────────────────────────────────────────────────────

class VideoStatusResponse(BaseModel):
    """
    What frontend receives when polling for status.

    GET /videos/status?job_ids=id1,id2,id3
    Returns a list of these objects, one per job.
    """
    job_id: str
    status: str
    progress_percent: int
    thumbnail_candidates: List[str]
    chosen_thumbnail: Optional[str]
    error_message: Optional[str]


class VideoUploadResponse(BaseModel):
    """
    What frontend receives immediately after uploading.

    POST /videos/upload
    Frontend uses job_ids to start polling.
    """
    job_ids: List[str]
    message: str


class VideoHistoryItem(BaseModel):
    """
    One item in the history list.

    GET /videos/history
    Returns a list of these, one per past video.
    """
    job_id: str
    original_filename: str
    status: str
    chosen_thumbnail: Optional[str]
    thumbnail_candidates: List[str]
    uploaded_at: datetime
    processed_at: Optional[datetime]
    content_type_detected: Optional[str]
    duration_seconds: Optional[float]