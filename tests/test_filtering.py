"""Unit tests for app.tracker.FrameFilter: Phase 6 frame-quality filtering."""

from __future__ import annotations

import cv2
import numpy as np

from app.tracker import FrameFilter

SETTINGS = {
    "filtering": {
        "blur_laplacian_threshold": 60.0,
        "overexposure_ratio": 0.6,
        "overexposure_v": 250,
        "underexposure_ratio": 0.6,
        "underexposure_v": 15,
        "duplicate_hash_threshold": 3,
        "outlier_zscore": 3.0,
        "min_information_entropy": 2.5,
    }
}


def _checkerboard(size: int = 100) -> np.ndarray:
    board = np.zeros((size, size), dtype=np.uint8)
    board[::2, ::2] = 255
    board[1::2, 1::2] = 255
    return cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)


def test_is_blurry_detects_blur():
    filt = FrameFilter(SETTINGS)
    sharp = _checkerboard()
    blurred = cv2.GaussianBlur(sharp, (25, 25), 0)
    assert filt.is_blurry(blurred) is True
    assert filt.is_blurry(sharp) is False


def test_is_overexposed_flags_white_frame():
    filt = FrameFilter(SETTINGS)
    white = np.full((50, 50, 3), 255, dtype=np.uint8)
    assert filt.is_overexposed(white) is True


def test_is_underexposed_flags_black_frame():
    filt = FrameFilter(SETTINGS)
    black = np.zeros((50, 50, 3), dtype=np.uint8)
    assert filt.is_underexposed(black) is True


def test_is_duplicate_detects_repeated_frame():
    filt = FrameFilter(SETTINGS)
    frame = _checkerboard()
    assert filt.is_duplicate(frame) is False  # first sighting
    assert filt.is_duplicate(frame) is True   # identical repeat


def test_is_low_information_flags_flat_frame():
    # Entropy is a histogram-spread measure, not a spatial-frequency one: a
    # flat frame has a single intensity (entropy ~0 bits) while a frame with
    # many distinct gray levels (e.g. natural noise) has high entropy, even
    # though a binary checkerboard (only 2 levels) would also read as "low
    # information" by this metric despite its sharp edges.
    filt = FrameFilter(SETTINGS)
    flat = np.full((50, 50, 3), 128, dtype=np.uint8)
    rng = np.random.default_rng(0)
    noisy = rng.integers(0, 255, size=(50, 50, 3), dtype=np.uint8)
    assert filt.is_low_information(flat) is True
    assert filt.is_low_information(noisy) is False
