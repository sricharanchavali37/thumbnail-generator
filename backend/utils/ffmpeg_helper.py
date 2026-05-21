import subprocess
import json
import numpy as np
from typing import Generator, Tuple


# ─────────────────────────────────────────────────────────
# WHAT THIS FILE IS
#
# FFmpeg is a command line tool.
# We control it from Python using subprocess.
# subprocess lets Python run command line tools
# and talk to them directly from code.
#
# This file has 3 jobs:
#   1. Read video metadata from S3 URL
#   2. Decide how many frames to extract
#   3. Stream frames from S3 into Python memory
# ─────────────────────────────────────────────────────────


def get_video_metadata(s3_url: str) -> dict:
    """
    Reads video metadata without downloading the video.

    FFmpeg probes the S3 URL and returns:
      duration    how long the video is in seconds
      width       video width in pixels
      height      video height in pixels
      fps         frames per second
      format      file format like mp4 or mov

    How it works:
      We run FFmpeg's probe command.
      FFmpeg reads just the header of the video from S3.
      The header contains all metadata.
      FFmpeg does not download the full video.
      Just the first few kilobytes where metadata lives.
    """

    # Build the FFprobe command
    # FFprobe is a tool that comes with FFmpeg
    # specifically for reading video information
    command = [
        "ffprobe",
        "-v", "quiet",              # Do not print unnecessary output
        "-print_format", "json",    # Give us the result as JSON
        "-show_streams",            # Show stream information
        "-show_format",             # Show format information
        s3_url                      # The S3 URL to probe
    ]

    # Run the command and capture output
    result = subprocess.run(
        command,
        capture_output=True,    # Capture stdout and stderr
        text=True,              # Return output as string not bytes
        timeout=30,             # Give up after 30 seconds
    )

    if result.returncode != 0:
        raise RuntimeError(f"FFprobe failed: {result.stderr}")

    # Parse the JSON output
    probe_data = json.loads(result.stdout)

    # Find the video stream
    # A video file can have multiple streams
    # One for video, one for audio, sometimes subtitles
    # We only care about the video stream
    video_stream = None
    for stream in probe_data.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break

    if not video_stream:
        raise RuntimeError("No video stream found in file")

    # Extract duration
    # Duration can be in the stream or in the format section
    duration = float(
        video_stream.get("duration")
        or probe_data.get("format", {}).get("duration", 0)
    )

    # Extract FPS
    # FFmpeg returns FPS as a fraction like "30/1" or "24000/1001"
    # We calculate it by dividing
    fps_raw = video_stream.get("r_frame_rate", "30/1")
    fps_parts = fps_raw.split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1])

    return {
        "duration": duration,
        "width": int(video_stream.get("width", 1920)),
        "height": int(video_stream.get("height", 1080)),
        "fps": fps,
        "format": probe_data.get("format", {}).get("format_name", "unknown"),
    }


def calculate_frame_count(duration_seconds: float) -> int:
    """
    Decides how many frames to extract based on video length.

    Short video  = fewer frames needed
    Long video   = more frames needed for good coverage

    Under 2 minutes  →  30 frames
    2 to 10 minutes  →  60 frames
    Over 10 minutes  →  120 frames

    Why these numbers?
      We need enough frames to find 5 great thumbnails.
      Too few frames = might miss the best moments.
      Too many frames = slow processing, diminishing returns.
      These numbers are the sweet spot.
    """
    if duration_seconds < 120:      # Under 2 minutes
        return 30
    elif duration_seconds < 600:    # 2 to 10 minutes
        return 60
    else:                           # Over 10 minutes
        return 120


def calculate_skip_boundaries(duration_seconds: float) -> Tuple[float, float]:
    """
    Calculates which parts of the video to skip.

    We skip:
      First 5% → usually black screen, intro, logo
      Last  5% → usually outro, credits, logo

    Returns:
      start_time → when to start extracting frames
      end_time   → when to stop extracting frames

    Example for a 100 second video:
      start_time = 5 seconds
      end_time   = 95 seconds
      We sample from second 5 to second 95
    """
    start_time = duration_seconds * 0.05
    end_time = duration_seconds * 0.95
    return start_time, end_time


def stream_frames_from_s3(
    s3_url: str,
    duration: float,
    width: int,
    height: int,
    frame_count: int,
) -> Generator[Tuple[np.ndarray, float], None, None]:
    """
    Streams frames directly from S3 URL into Python memory.

    This is the core of the streaming approach.
    FFmpeg reads from S3.
    FFmpeg sends raw frame data through a pipe.
    Python reads that pipe one frame at a time.
    No video file ever touches disk.

    Yields:
      frame      → numpy array of pixel data (Height x Width x 3)
      timestamp  → where in the video this frame is in seconds

    How the pipe works:
      FFmpeg writes raw pixel bytes to stdout.
      Python reads those bytes from stdout.
      Each frame is exactly width * height * 3 bytes.
      We know exactly how many bytes to read per frame.
      We read that many bytes, convert to numpy array.
      That numpy array is one complete frame.
    """

    # Calculate which timestamps to extract frames from
    start_time, end_time = calculate_skip_boundaries(duration)
    usable_duration = end_time - start_time

    # Calculate timestamps evenly spread across usable video
    # If we want 60 frames from 100 seconds of usable video
    # we extract a frame every 100/60 = 1.67 seconds
    timestamps = [
        start_time + (i * usable_duration / frame_count)
        for i in range(frame_count)
    ]

    # Build FFmpeg command to stream frames
    command = [
        "ffmpeg",
        "-i", s3_url,               # Input: S3 URL directly
        "-vf", f"fps=1",            # Extract 1 frame per second filter
        "-f", "rawvideo",           # Output format: raw pixel data
        "-pix_fmt", "bgr24",        # Pixel format: BGR (OpenCV default)
        # bgr24 means each pixel has 3 values: Blue, Green, Red
        # Each value is 0-255
        # OpenCV uses BGR not RGB historically
        "pipe:1",                   # Send output to stdout (the pipe)
    ]

    # Start FFmpeg as a subprocess
    # stdout=subprocess.PIPE means we read its output in Python
    # stderr=subprocess.DEVNULL means we ignore error output
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    # Calculate exactly how many bytes one frame takes
    # Each pixel = 3 bytes (B, G, R)
    # Total bytes = width * height * 3
    frame_size_bytes = width * height * 3
    frame_index = 0

    try:
        while True:
            # Read exactly one frame worth of bytes from FFmpeg
            raw_bytes = process.stdout.read(frame_size_bytes)

            # If we got fewer bytes than expected
            # FFmpeg has finished sending frames
            if len(raw_bytes) < frame_size_bytes:
                break

            # Convert raw bytes to numpy array
            # This is how OpenCV represents an image
            frame = np.frombuffer(raw_bytes, dtype=np.uint8)
            frame = frame.reshape((height, width, 3))

            # Calculate the timestamp for this frame
            if frame_index < len(timestamps):
                timestamp = timestamps[frame_index]
            else:
                break

            # Yield this frame and its timestamp
            # yield means: give this frame to whoever called us
            # then pause here until they ask for the next one
            # Memory efficient: only one frame in memory at a time
            yield frame, timestamp
            frame_index += 1

    finally:
        # Always clean up the FFmpeg process
        # Whether we finished normally or something crashed
        process.stdout.close()
        process.wait()