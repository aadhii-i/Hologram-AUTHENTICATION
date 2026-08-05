"""Unit tests for app.hsv: pixel classification and frame classification."""

from __future__ import annotations

import cv2
import numpy as np

from app.hsv import classify_frame, classify_pixel_color


def test_classify_pixel_green(hsv_config):
    assert classify_pixel_color(h=60, s=200, v=200, hsv_config=hsv_config) == "green"


def test_classify_pixel_violet(hsv_config):
    assert classify_pixel_color(h=140, s=150, v=150, hsv_config=hsv_config) == "violet"


def test_classify_pixel_neutral(hsv_config):
    assert classify_pixel_color(h=0, s=10, v=150, hsv_config=hsv_config) == "neutral"


def test_classify_pixel_unknown_for_red(hsv_config):
    assert classify_pixel_color(h=5, s=200, v=200, hsv_config=hsv_config) == "unknown"


def _solid_hsv_frame(h: int, s: int, v: int, size: int = 60) -> np.ndarray:
    hsv = np.full((size, size, 3), (h, s, v), dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def test_classify_frame_dominant_green(hsv_config):
    frame = _solid_hsv_frame(60, 200, 200)
    result = classify_frame(frame, hsv_config)
    assert result.label == "green"
    assert result.confidence > 0.9


def test_classify_frame_dominant_violet(hsv_config):
    frame = _solid_hsv_frame(140, 150, 150)
    result = classify_frame(frame, hsv_config)
    assert result.label == "violet"
    assert result.confidence > 0.9


def test_classify_frame_excludes_glare_and_dark():
    frame_white = np.full((40, 40, 3), 255, dtype=np.uint8)
    frame_black = np.zeros((40, 40, 3), dtype=np.uint8)
    from app.utils import load_hsv_config

    cfg = load_hsv_config()
    white_result = classify_frame(frame_white, cfg)
    black_result = classify_frame(frame_black, cfg)
    assert white_result.valid_ratio < 0.1
    assert black_result.valid_ratio < 0.1
