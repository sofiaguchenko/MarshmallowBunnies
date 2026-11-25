from pathlib import Path
from typing import Dict, List, Optional

import cv2
import easyocr
import numpy as np
from PIL import Image

_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def init_ocr_reader(
    languages: Optional[List[str]] = None, gpu: bool = True
) -> easyocr.Reader:
    """Initialize EasyOCR reader once for reuse across a dataset."""
    if languages is None:
        languages = ["en"]
    return easyocr.Reader(languages, gpu=gpu)


def extract_visual_attributes(image_path: Path) -> Dict[str, float]:
    """Extract brightness, saturation, color warmth, and face metrics from a cover."""
    with Image.open(image_path) as img:
        rgb = np.array(img.convert("RGB"))

    height, width = rgb.shape[:2]
    total_pixels = float(height * width)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]  # OpenCV scale: 0-179
    saturation = hsv[:, :, 1]  # Scale: 0-255
    value = hsv[:, :, 2]  # Scale: 0-255

    mean_brightness = float(np.mean(value) / 255.0)
    std_brightness = float(np.std(value) / 255.0)
    mean_saturation = float(np.mean(saturation) / 255.0)

    # Warm hues (reds, oranges, yellows: 0-25 or 165-179) with non-trivial saturation
    warm_mask = ((hue <= 25) | (hue >= 165)) & (saturation >= 40)
    warm_pixel_ratio = float(np.sum(warm_mask) / total_pixels)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(24, 24),
    )

    face_count = int(len(faces))
    face_area = sum(int(w * h) for (_, _, w, h) in faces)
    face_area_ratio = float(face_area / total_pixels)

    return {
        "mean_brightness": mean_brightness,
        "std_brightness": std_brightness,
        "mean_saturation": mean_saturation,
        "warm_pixel_ratio": warm_pixel_ratio,
        "face_count": face_count,
        "face_area_ratio": face_area_ratio,
    }


def mask_cover_text(
    image_path: Path,
    output_path: Path,
    reader: easyocr.Reader,
    min_conf: float = 0.4,
    pad_px: int = 4,
) -> Dict[str, object]:
    """Detect title/author text via OCR, blur it out, and save the masked cover."""
    with Image.open(image_path) as img:
        rgb = np.array(img.convert("RGB"))

    height, width = rgb.shape[:2]
    total_pixels = float(height * width)

    detections = reader.readtext(rgb)
    text_mask = np.zeros((height, width), dtype=np.uint8)
    extracted_tokens: List[str] = []

    for bbox, text, conf in detections:
        if conf < min_conf:
            continue
        extracted_tokens.append(text.strip())
        pts = np.array(bbox, dtype=np.int32)
        x_min = max(int(np.min(pts[:, 0])) - pad_px, 0)
        y_min = max(int(np.min(pts[:, 1])) - pad_px, 0)
        x_max = min(int(np.max(pts[:, 0])) + pad_px, width)
        y_max = min(int(np.max(pts[:, 1])) + pad_px, height)
        text_mask[y_min:y_max, x_min:x_max] = 255

    # Gaussian blur preserves cover color/brightness while removing readable text
    blurred = cv2.GaussianBlur(rgb, (51, 51), 0)
    masked_rgb = np.where(text_mask[:, :, None] == 255, blurred, rgb)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(masked_rgb).save(output_path)

    text_area_ratio = float(np.sum(text_mask > 0) / total_pixels)

    return {
        "masked_image_path": str(output_path),
        "text_area_ratio": text_area_ratio,
        "ocr_text": " ".join(extracted_tokens),
    }
