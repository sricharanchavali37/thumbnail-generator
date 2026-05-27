from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import numpy as np
import os


CONTENT_COLORS = {
    "sports":    (255, 100, 0),
    "vlog":      (0, 120, 255),
    "cinematic": (150, 0, 255),
    "general":   (200, 200, 200),
}

BADGE_LABELS = {
    "sports":    "SPORTS",
    "vlog":      "VLOG",
    "cinematic": "CINEMATIC",
    "general":   "VIDEO",
}

FONT_BOLD_PATH = "C:/Windows/Fonts/arialbd.ttf"
FONT_REGULAR_PATH = "C:/Windows/Fonts/arial.ttf"


def get_font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    try:
        path = FONT_BOLD_PATH if bold else FONT_REGULAR_PATH
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def boost_brightness_contrast(image: Image.Image) -> Image.Image:
    grayscale = image.convert("L")
    avg_brightness = np.mean(np.array(grayscale))
    if avg_brightness < 80:
        brightness_factor = 1.6
    elif avg_brightness < 120:
        brightness_factor = 1.3
    else:
        brightness_factor = 1.0
    if brightness_factor > 1.0:
        image = ImageEnhance.Brightness(image).enhance(brightness_factor)
    image = ImageEnhance.Contrast(image).enhance(1.2)
    return image


def add_gradient_overlay(image: Image.Image) -> Image.Image:
    width, height = image.size
    image = image.convert("RGBA")
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    gradient_start_y = int(height * 0.60)
    for y in range(gradient_start_y, height):
        progress = (y - gradient_start_y) / (height - gradient_start_y)
        alpha = int(progress * 200)
        draw.line([(0, y), (width, y)], fill=(0, 0, 0, alpha))
    image = Image.alpha_composite(image, overlay)
    return image.convert("RGB")


def add_content_badge(image: Image.Image, content_type: str) -> Image.Image:
    draw = ImageDraw.Draw(image)
    color = CONTENT_COLORS.get(content_type, CONTENT_COLORS["general"])
    label = BADGE_LABELS.get(content_type, "VIDEO")
    font = get_font(22, bold=True)
    bbox = font.getbbox(label)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    pad_x, pad_y = 14, 8
    badge_x, badge_y = 20, 20
    badge_w = badge_x + text_width + (pad_x * 2)
    badge_h = badge_y + text_height + (pad_y * 2)
    draw.rectangle([badge_x, badge_y, badge_w, badge_h], fill=color)
    draw.text((badge_x + pad_x, badge_y + pad_y), label, font=font, fill=(255, 255, 255))
    return image


def add_border(image: Image.Image, content_type: str) -> Image.Image:
    draw = ImageDraw.Draw(image)
    width, height = image.size
    color = CONTENT_COLORS.get(content_type, CONTENT_COLORS["general"])
    draw.rectangle([0, 0, width - 1, height - 1], outline=color, width=6)
    return image


def enhance_thumbnail(
    frame_bgr: np.ndarray,
    content_type: str,
    title: str = None,
) -> np.ndarray:
    frame_rgb = frame_bgr[:, :, ::-1]
    image = Image.fromarray(frame_rgb)
    image = boost_brightness_contrast(image)
    image = add_gradient_overlay(image)
    image = add_content_badge(image, content_type)
    image = add_border(image, content_type)
    result_rgb = np.array(image)
    result_bgr = result_rgb[:, :, ::-1]
    return result_bgr


def add_custom_text_to_thumbnail(
    thumbnail_path: str,
    custom_text: str,
    position: str = "bottom",
    output_path: str = None,
) -> str:
    image = Image.open(thumbnail_path).convert("RGB")
    width, height = image.size
    custom_text = custom_text.strip().upper()
    words = custom_text.split()
    lines = []
    current_line = ""
    for word in words:
        test_line = current_line + " " + word if current_line else word
        if len(test_line) <= 20:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    lines = lines[:2]
    font_size = 90 if len(lines) == 1 else 72
    font = get_font(font_size, bold=True)
    line_height = font_size + 12
    total_text_height = len(lines) * line_height
    padding = 40
    if position == "top":
        text_y_start = padding + 60
    elif position == "centre":
        text_y_start = (height - total_text_height) // 2
    else:
        text_y_start = height - total_text_height - padding
    text_x = 40
    image = image.convert("RGBA")
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    gradient_padding = 20
    gradient_top = max(0, text_y_start - gradient_padding)
    gradient_bottom = min(height, text_y_start + total_text_height + gradient_padding)
    gradient_height = gradient_bottom - gradient_top
    for y in range(gradient_top, gradient_bottom):
        progress = (y - gradient_top) / max(gradient_height, 1)
        if progress < 0.2:
            alpha = int(progress * 5 * 160)
        elif progress > 0.8:
            alpha = int((1 - progress) * 5 * 160)
        else:
            alpha = 160
        draw_overlay.line([(0, y), (width, y)], fill=(0, 0, 0, alpha))
    image = Image.alpha_composite(image, overlay)
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    text_color = (255, 220, 0)
    outline_color = (0, 0, 0)
    outline_thickness = 4
    for i, line in enumerate(lines):
        y = text_y_start + (i * line_height)
        for dx in range(-outline_thickness, outline_thickness + 1):
            for dy in range(-outline_thickness, outline_thickness + 1):
                if dx != 0 or dy != 0:
                    draw.text((text_x + dx, y + dy), line, font=font, fill=outline_color)
        draw.text((text_x, y), line, font=font, fill=text_color)
    if output_path is None:
        folder = os.path.dirname(thumbnail_path)
        output_path = os.path.join(folder, "final_thumb.jpg")
    image.save(output_path, "JPEG", quality=95)
    print(f"Final thumbnail saved: {output_path}")
    return output_path