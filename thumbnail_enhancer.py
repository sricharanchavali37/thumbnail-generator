from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import numpy as np
import os


# ─────────────────────────────────────────────────────────
# THUMBNAIL ENHANCER SERVICE
#
# Takes a raw extracted frame (numpy array from OpenCV)
# and transforms it into a professional YouTube thumbnail.
#
# What it adds:
#   1. Brightness boost for dark frames
#   2. Contrast boost to make colours pop
#   3. Dark gradient overlay on bottom 40%
#   4. Content type badge top left corner
#   5. Bold title text bottom left
#   6. Coloured border around thumbnail
#
# Tool: Pillow (PIL) — free, no GPU, no API, runs on CPU
# ─────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────
# CONTENT TYPE COLOUR SCHEME
# Each content type gets its own colour identity.
# Badge + border use the same colour per type.
# ─────────────────────────────────────────────────────────

CONTENT_COLORS = {
    "sports":    (255, 100, 0),    # Orange
    "vlog":      (0, 120, 255),    # Blue
    "cinematic": (150, 0, 255),    # Purple
    "general":   (200, 200, 200),  # Light grey
}

BADGE_LABELS = {
    "sports":    "SPORTS",
    "vlog":      "VLOG",
    "cinematic": "CINEMATIC",
    "general":   "VIDEO",
}


# ─────────────────────────────────────────────────────────
# FONT PATHS
# Using Windows built-in fonts.
# No download needed. Already on every Windows machine.
# ─────────────────────────────────────────────────────────

FONT_BOLD_PATH = "C:/Windows/Fonts/arialbd.ttf"
FONT_REGULAR_PATH = "C:/Windows/Fonts/arial.ttf"


def get_font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    """
    Loads Arial font at given size.
    Falls back to default if Arial not found.
    """
    try:
        path = FONT_BOLD_PATH if bold else FONT_REGULAR_PATH
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


# ═════════════════════════════════════════════════════════
# ENHANCEMENT FUNCTIONS
# Each function does one specific visual improvement.
# ═════════════════════════════════════════════════════════

def boost_brightness_contrast(image: Image.Image) -> Image.Image:
    """
    Automatically boosts brightness and contrast.

    Checks average brightness of image.
    If dark (night cricket, indoor scene):
        Boosts brightness so frame is usable.
    Always applies slight contrast boost.
    Makes colours pop. Makes action stand out.
    """
    # Check average brightness
    grayscale = image.convert("L")
    avg_brightness = np.mean(np.array(grayscale))

    # Boost brightness if too dark
    if avg_brightness < 80:
        # Very dark frame. Boost significantly.
        brightness_factor = 1.6
    elif avg_brightness < 120:
        # Moderately dark. Mild boost.
        brightness_factor = 1.3
    else:
        # Already bright enough. No change.
        brightness_factor = 1.0

    if brightness_factor > 1.0:
        image = ImageEnhance.Brightness(image).enhance(brightness_factor)

    # Always boost contrast slightly
    # Makes the thumbnail look vivid and professional
    image = ImageEnhance.Contrast(image).enhance(1.2)

    return image


def add_gradient_overlay(image: Image.Image) -> Image.Image:
    """
    Adds a dark gradient overlay on bottom 40% of image.

    Starts fully transparent at the top of that section.
    Becomes 80% opaque black at the very bottom.

    Why:
        Text placed on bright or busy backgrounds
        is completely unreadable.
        This dark area makes white text always visible
        regardless of what the video content shows.

    This is the exact technique used in:
        F1 Hamilton thumbnail (Image 2)
        Kabaddi thumbnail (Image 3)
        Cricket ODI thumbnail (Image 4)
    """
    width, height = image.size

    # Convert to RGBA to support transparency
    image = image.convert("RGBA")

    # Create transparent overlay same size as image
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Gradient covers bottom 40% of image
    gradient_start_y = int(height * 0.60)

    # Draw gradient line by line
    # Each line gets progressively more opaque
    for y in range(gradient_start_y, height):
        # How far through the gradient are we? 0.0 to 1.0
        progress = (y - gradient_start_y) / (height - gradient_start_y)

        # Alpha goes from 0 (transparent) to 200 (80% dark)
        alpha = int(progress * 200)

        draw.line([(0, y), (width, y)], fill=(0, 0, 0, alpha))

    # Merge overlay onto image
    image = Image.alpha_composite(image, overlay)

    # Convert back to RGB for JPEG saving
    image = image.convert("RGB")

    return image


def add_content_badge(
    image: Image.Image,
    content_type: str,
) -> Image.Image:
    """
    Adds a small coloured badge in top left corner.

    Content type is auto-detected by our classifier.
    We just display it as a label.

    sports    → orange  SPORTS badge
    vlog      → blue    VLOG badge
    cinematic → purple  CINEMATIC badge
    general   → grey    VIDEO badge

    Looks like the HIGHLIGHTS label in Kabaddi thumbnail.
    """
    draw = ImageDraw.Draw(image)

    color = CONTENT_COLORS.get(content_type, CONTENT_COLORS["general"])
    label = BADGE_LABELS.get(content_type, "VIDEO")

    font = get_font(22, bold=True)

    # Measure text size
    bbox = font.getbbox(label)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Badge padding
    pad_x = 14
    pad_y = 8

    # Badge position: top left with small margin
    badge_x = 20
    badge_y = 20
    badge_w = badge_x + text_width + (pad_x * 2)
    badge_h = badge_y + text_height + (pad_y * 2)

    # Draw filled badge rectangle
    draw.rectangle(
        [badge_x, badge_y, badge_w, badge_h],
        fill=color
    )

    # Draw text on badge
    draw.text(
        (badge_x + pad_x, badge_y + pad_y),
        label,
        font=font,
        fill=(255, 255, 255)
    )

    return image


def add_title_text(
    image: Image.Image,
    title: str,
    content_type: str,
) -> Image.Image:
    """
    Adds large bold title text in bottom left area.

    Text sits inside the dark gradient overlay zone.
    White text with black outline for maximum readability.

    User provides this text when uploading the video.
    Example: "INDIA VS AUSTRALIA HIGHLIGHTS"
             "HAMILTON WAS INVINCIBLE"
             "BOSE DEATH MYSTERY EXPLAINED"

    If title is empty or None:
        No text added. Gradient still looks professional.
    """
    if not title or title.strip() == "":
        return image

    draw = ImageDraw.Draw(image)
    width, height = image.size

    # Clean up title text
    title = title.strip().upper()

    # Split into lines if too long
    # Max 25 characters per line looks good at 1280px width
    words = title.split()
    lines = []
    current_line = ""

    for word in words:
        test_line = current_line + " " + word if current_line else word
        if len(test_line) <= 25:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    # Limit to 2 lines maximum
    lines = lines[:2]

    # Font sizes based on number of lines
    font_size = 80 if len(lines) == 1 else 65
    font = get_font(font_size, bold=True)

    # Position: bottom left inside gradient area
    # Start from 65% height downward
    text_x = 35
    line_height = font_size + 10
    total_text_height = len(lines) * line_height
    text_y = height - total_text_height - 35

    # Draw each line with black outline first then white text
    outline_range = 3

    for i, line in enumerate(lines):
        y = text_y + (i * line_height)

        # Black outline (draw text offset in all 8 directions)
        for dx in range(-outline_range, outline_range + 1):
            for dy in range(-outline_range, outline_range + 1):
                if dx != 0 or dy != 0:
                    draw.text(
                        (text_x + dx, y + dy),
                        line,
                        font=font,
                        fill=(0, 0, 0)
                    )

        # White text on top
        draw.text(
            (text_x, y),
            line,
            font=font,
            fill=(255, 255, 255)
        )

    return image


def add_border(
    image: Image.Image,
    content_type: str,
) -> Image.Image:
    """
    Adds a thin coloured border around the entire thumbnail.

    Border colour matches content type colour.
    sports    → orange border
    vlog      → blue border
    cinematic → purple border
    general   → white border

    Makes thumbnail look finished and polished.
    Helps it stand out in YouTube grid.
    """
    draw = ImageDraw.Draw(image)
    width, height = image.size

    color = CONTENT_COLORS.get(content_type, CONTENT_COLORS["general"])

    border_thickness = 6

    # Draw rectangle border along all 4 edges
    draw.rectangle(
        [0, 0, width - 1, height - 1],
        outline=color,
        width=border_thickness
    )

    return image


# ═════════════════════════════════════════════════════════
# MAIN FUNCTION — Called from video_processor.py
# ═════════════════════════════════════════════════════════

def enhance_thumbnail(
    frame_bgr: np.ndarray,
    content_type: str,
    title: str = None,
) -> np.ndarray:
    """
    Main enhancement function called from video_processor.py

    Takes raw frame from OpenCV (numpy array in BGR format).
    Returns enhanced professional thumbnail (numpy array in BGR).

    Steps applied in order:
      1. Convert BGR to RGB (OpenCV → Pillow format)
      2. Boost brightness if dark
      3. Boost contrast slightly
      4. Add dark gradient overlay bottom 40%
      5. Add content type badge top left
      6. Add title text bottom left (if provided)
      7. Add coloured border
      8. Convert RGB back to BGR (Pillow → OpenCV format)

    Args:
      frame_bgr:    numpy array from OpenCV (BGR, 1280x720)
      content_type: "sports" / "vlog" / "cinematic" / "general"
      title:        text to display on thumbnail (optional)

    Returns:
      numpy array in BGR format ready for cv2.imencode()
    """

    # ── CONVERT OpenCV BGR → Pillow RGB ──────────────────
    frame_rgb = frame_bgr[:, :, ::-1]  # BGR to RGB flip
    image = Image.fromarray(frame_rgb)

    # ── STEP 1: Brightness and contrast ──────────────────
    image = boost_brightness_contrast(image)

    # ── STEP 2: Dark gradient overlay ────────────────────
    image = add_gradient_overlay(image)

    # ── STEP 3: Content type badge ────────────────────────
    image = add_content_badge(image, content_type)

    # ── STEP 4: Title text ────────────────────────────────
    if title:
        image = add_title_text(image, title, content_type)

    # ── STEP 5: Border ────────────────────────────────────
    image = add_border(image, content_type)

    # ── CONVERT Pillow RGB → OpenCV BGR ──────────────────
    result_rgb = np.array(image)
    result_bgr = result_rgb[:, :, ::-1]  # RGB to BGR flip

    return result_bgr