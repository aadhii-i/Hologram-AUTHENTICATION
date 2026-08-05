"""RGB->HSV conversion and HSV-threshold-based color classification.

Classifies each (preprocessed, ROI-cropped) frame into one of
`green` / `violet` / `neutral` / `unknown`, ignoring glare, dark, and
low-saturation pixels when voting on the dominant color.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from app.utils import get_logger

logger = get_logger(__name__)

COLOR_CLASSES = ("green", "violet", "neutral")
CLASS_SYMBOLS = {"green": "G", "violet": "V", "neutral": "N", "unknown": "U"}


def to_hsv(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image (OpenCV's native order) to HSV."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def _band_mask(hsv: np.ndarray, cfg: dict) -> np.ndarray:
    h, s, v = cv2.split(hsv)
    mask = (
        (h >= cfg.get("h_min", 0))
        & (h <= cfg.get("h_max", 179))
        & (s >= cfg.get("s_min", 0))
        & (s <= cfg.get("s_max", 255))
        & (v >= cfg.get("v_min", 0))
        & (v <= cfg.get("v_max", 255))
    )
    return mask.astype(np.uint8) * 255


def _neutral_mask(hsv: np.ndarray, cfg: dict) -> np.ndarray:
    h, s, v = cv2.split(hsv)
    mask = (
        (s <= cfg.get("s_max", 40))
        & (v >= cfg.get("v_min", 60))
        & (v <= cfg.get("v_max", 255))
    )
    return mask.astype(np.uint8) * 255


def glare_mask(hsv: np.ndarray, cfg: dict) -> np.ndarray:
    h, s, v = cv2.split(hsv)
    mask = (v >= cfg.get("v_min", 240)) & (s <= cfg.get("s_max", 30))
    return mask.astype(np.uint8) * 255


def dark_mask(hsv: np.ndarray, cfg: dict) -> np.ndarray:
    _, _, v = cv2.split(hsv)
    mask = v <= cfg.get("v_max", 30)
    return mask.astype(np.uint8) * 255


def build_masks(hsv: np.ndarray, hsv_config: dict) -> dict[str, np.ndarray]:
    """Build boolean-style (0/255) masks for every color class plus glare/dark."""
    masks = {
        "green": _band_mask(hsv, hsv_config.get("green", {})),
        "violet": _band_mask(hsv, hsv_config.get("violet", {})),
        "neutral": _neutral_mask(hsv, hsv_config.get("neutral", {})),
        "glare": glare_mask(hsv, hsv_config.get("glare", {})),
        "dark": dark_mask(hsv, hsv_config.get("dark", {})),
    }
    return masks


def classify_pixel_color(h: float, s: float, v: float, hsv_config: dict) -> str:
    """Classify a single H/S/V triple. Mainly used for unit testing the thresholds."""
    hsv_pixel = np.array([[[h, s, v]]], dtype=np.uint8)
    masks = build_masks(hsv_pixel, hsv_config)
    for cls in COLOR_CLASSES:
        if masks[cls][0, 0] > 0:
            return cls
    return "unknown"


@dataclass
class ColorResult:
    """Per-frame color classification output."""

    label: str
    confidence: float
    dominant_hue: float
    mean_hue: float
    median_hue: float
    pixel_pct: dict[str, float] = field(default_factory=dict)
    valid_ratio: float = 0.0
    is_glare: bool = False
    is_dark: bool = False
    is_low_saturation: bool = False


def classify_frame(
    roi_image: np.ndarray, hsv_config: dict, min_class_pixel_ratio: Optional[float] = None
) -> ColorResult:
    """Classify a ROI frame's dominant hologram color.

    Glare and near-black pixels are excluded from the voting population.
    A frame is `unknown` if no class clears `min_class_pixel_ratio` of the
    *valid* (non-glare, non-dark) pixel population.
    """
    min_ratio = (
        min_class_pixel_ratio
        if min_class_pixel_ratio is not None
        else hsv_config.get("min_class_pixel_ratio", 0.15)
    )

    hsv = to_hsv(roi_image)
    masks = build_masks(hsv, hsv_config)

    invalid = (masks["glare"] > 0) | (masks["dark"] > 0)
    valid = ~invalid
    total_px = roi_image.shape[0] * roi_image.shape[1]
    valid_ratio = float(valid.sum()) / total_px if total_px else 0.0

    pixel_pct: dict[str, float] = {}
    for cls in COLOR_CLASSES:
        cls_valid_px = (masks[cls] > 0) & valid
        denom = max(int(valid.sum()), 1)
        pixel_pct[cls] = float(cls_valid_px.sum()) / denom

    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    valid_hues = h_channel[valid & (s_channel > 15)]  # ignore near-gray for hue stats
    if valid_hues.size == 0:
        valid_hues = h_channel[valid] if valid.any() else h_channel.flatten()

    mean_hue = float(np.mean(valid_hues)) if valid_hues.size else 0.0
    median_hue = float(np.median(valid_hues)) if valid_hues.size else 0.0
    if valid_hues.size:
        hist, _ = np.histogram(valid_hues, bins=180, range=(0, 180))
        dominant_hue = float(np.argmax(hist))
    else:
        dominant_hue = 0.0

    best_cls = max(pixel_pct, key=pixel_pct.get) if pixel_pct else "unknown"
    label = best_cls if pixel_pct.get(best_cls, 0.0) >= min_ratio else "unknown"
    confidence = pixel_pct.get(best_cls, 0.0) if label != "unknown" else 0.0

    is_low_sat = float(np.median(s_channel[valid])) <= hsv_config.get(
        "low_saturation", {}
    ).get("s_max", 20) if valid.any() else True

    return ColorResult(
        label=label,
        confidence=confidence,
        dominant_hue=dominant_hue,
        mean_hue=mean_hue,
        median_hue=median_hue,
        pixel_pct=pixel_pct,
        valid_ratio=valid_ratio,
        is_glare=bool(masks["glare"].mean() / 255 > 0.5),
        is_dark=bool(masks["dark"].mean() / 255 > 0.5),
        is_low_saturation=is_low_sat,
    )


def adapt_thresholds(hue_samples: np.ndarray, base_config: dict, percentile_low: float = 5, percentile_high: float = 95) -> dict:
    """Bonus: widen/narrow green & violet hue bands to the observed distribution.

    Keeps the configured band centered but rescales its width toward the
    [percentile_low, percentile_high] spread of hues actually seen in the
    video, which compensates for lighting/foil differences between samples.
    """
    if hue_samples.size == 0:
        return base_config

    adapted = {k: dict(v) if isinstance(v, dict) else v for k, v in base_config.items()}
    lo, hi = np.percentile(hue_samples, [percentile_low, percentile_high])
    spread = max(float(hi - lo), 10.0)

    for cls in ("green", "violet"):
        cfg = adapted.get(cls, {})
        center = (cfg.get("h_min", 0) + cfg.get("h_max", 179)) / 2
        half_width = max(spread / 4, 10.0)
        adapted[cls] = {
            **cfg,
            "h_min": max(0, int(center - half_width)),
            "h_max": min(179, int(center + half_width)),
        }
    return adapted
