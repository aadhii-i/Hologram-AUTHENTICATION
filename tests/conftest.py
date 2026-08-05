"""Shared pytest fixtures: project path setup and synthetic test videos."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils import load_hsv_config  # noqa: E402

# BGR swatches that land inside the default hsv.yaml bands.
GREEN_BGR = (60, 200, 60)
VIOLET_BGR = (180, 40, 150)
NEUTRAL_BGR = (200, 200, 200)


@pytest.fixture(scope="session")
def hsv_config() -> dict:
    return load_hsv_config()


def _write_solid_frame_video(path: Path, colors_bgr: list[tuple[int, int, int]], frames_per_color: int = 5, size: int = 80):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (size, size))
    for color in colors_bgr:
        frame = np.full((size, size, 3), color, dtype=np.uint8)
        for _ in range(frames_per_color):
            writer.write(frame)
    writer.release()


@pytest.fixture()
def genuine_sequence_video(tmp_path: Path) -> Path:
    """A tiny synthetic video whose ROI transitions Green -> Neutral -> Violet."""
    path = tmp_path / "genuine.mp4"
    _write_solid_frame_video(path, [GREEN_BGR, NEUTRAL_BGR, VIOLET_BGR])
    return path


@pytest.fixture()
def counterfeit_sequence_video(tmp_path: Path) -> Path:
    """A tiny synthetic video whose ROI transitions Neutral -> Green -> Violet."""
    path = tmp_path / "counterfeit.mp4"
    _write_solid_frame_video(path, [NEUTRAL_BGR, GREEN_BGR, VIOLET_BGR])
    return path
