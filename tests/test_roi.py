"""Unit tests for app.roi: cropping, coordinate conversion, and ArUco fallback."""

from __future__ import annotations

import numpy as np

from app.roi import (
    apply_roi,
    detect_aruco_roi,
    normalized_to_pixels,
    pixels_to_normalized,
)


def test_apply_roi_crops_expected_region():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[25:75, 25:75] = 255  # bright square in the middle

    crop = apply_roi(image, (0.25, 0.25, 0.75, 0.75))
    assert crop.shape[0] == 50
    assert crop.shape[1] == 50
    assert crop.mean() > 250  # crop should be entirely inside the bright square


def test_apply_roi_with_padding_grows_crop():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    tight = apply_roi(image, (0.4, 0.4, 0.6, 0.6), padding_ratio=0.0)
    padded = apply_roi(image, (0.4, 0.4, 0.6, 0.6), padding_ratio=0.5)
    assert padded.shape[0] > tight.shape[0]
    assert padded.shape[1] > tight.shape[1]


def test_pixel_normalized_roundtrip():
    box_px = (10, 20, 90, 80)
    normalized = pixels_to_normalized(box_px, width=100, height=100)
    back = normalized_to_pixels(normalized, width=100, height=100)
    assert back == box_px


def test_detect_aruco_roi_returns_none_without_markers():
    rng = np.random.default_rng(0)
    noise_image = rng.integers(0, 255, size=(200, 200, 3), dtype=np.uint8)
    assert detect_aruco_roi(noise_image) is None
