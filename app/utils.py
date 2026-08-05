"""Shared utilities: logging, config loading, and video I/O.

Everything else in `app/` depends on this module for config access and
frame extraction, so it deliberately has no dependencies on other
first-party modules (keeps the import graph acyclic).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_LOGGERS: dict[str, logging.Logger] = {}


def get_logger(name: str, config: Optional[dict] = None) -> logging.Logger:
    """Return a configured module-level logger, memoized by name."""
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    if not logger.handlers:
        level_name = "INFO"
        log_file = None
        if config:
            log_cfg = config.get("logging", {})
            level_name = log_cfg.get("level", "INFO")
            log_file = log_cfg.get("log_file")

        logger.setLevel(getattr(logging, level_name, logging.INFO))
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if log_file:
            log_path = resolve_path(log_file)
            ensure_dir(log_path.parent)
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        logger.propagate = False

    _LOGGERS[name] = logger
    return logger


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root unless already absolute."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if it doesn't exist; return it."""
    p = resolve_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a plain dict. Returns {} if the file is empty."""
    p = resolve_path(path)
    with open(p, "r") as f:
        data = yaml.safe_load(f)
    return data or {}


def save_yaml(path: str | Path, data: dict[str, Any]) -> None:
    """Write a dict to a YAML file, creating parent directories as needed."""
    p = resolve_path(path)
    ensure_dir(p.parent)
    with open(p, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def load_settings(path: str | Path = "configs/settings.yaml") -> dict[str, Any]:
    return load_yaml(path)


def load_hsv_config(path: str | Path = "configs/hsv.yaml") -> dict[str, Any]:
    return load_yaml(path)


def load_calibration(path: str | Path = "configs/calibration.yaml") -> dict[str, Any]:
    return load_yaml(path)


@dataclass
class Frame:
    """A single decoded video frame with metadata."""

    index: int
    timestamp_sec: float
    image: np.ndarray = field(repr=False)


class VideoLoadError(RuntimeError):
    """Raised when a video file cannot be opened or decoded."""


class VideoLoader:
    """Opens a video file and yields frames with numbering/timestamps.

    Supports mp4, mov, avi, mkv (anything OpenCV/FFmpeg can decode).
    """

    def __init__(
        self,
        video_path: str | Path,
        frame_skip: int = 1,
        max_frames: int = 0,
        supported_extensions: Optional[list[str]] = None,
    ):
        self.video_path = resolve_path(video_path)
        self.frame_skip = max(1, int(frame_skip))
        self.max_frames = max_frames
        self.supported_extensions = supported_extensions or [
            ".mp4",
            ".mov",
            ".avi",
            ".mkv",
        ]
        self.logger = get_logger(self.__class__.__name__)

        if not self.video_path.exists():
            raise VideoLoadError(f"Video file does not exist: {self.video_path}")

        if self.video_path.suffix.lower() not in self.supported_extensions:
            raise VideoLoadError(
                f"Unsupported video extension '{self.video_path.suffix}'. "
                f"Supported: {self.supported_extensions}"
            )

        self.capture = cv2.VideoCapture(str(self.video_path))
        if not self.capture.isOpened():
            raise VideoLoadError(f"OpenCV could not open video: {self.video_path}")

        self.fps = self.capture.get(cv2.CAP_PROP_FPS) or 0.0
        if self.fps <= 0:
            self.logger.warning("FPS metadata missing/zero; defaulting to 30.0")
            self.fps = 30.0

        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.logger.info(
            "Opened video %s (%dx%d, %.2f fps, ~%d frames)",
            self.video_path.name,
            self.width,
            self.height,
            self.fps,
            self.frame_count,
        )

    def __iter__(self) -> Iterator[Frame]:
        index = 0
        yielded = 0
        while True:
            ok, image = self.capture.read()
            if not ok:
                break
            if index % self.frame_skip == 0:
                timestamp = index / self.fps
                yield Frame(index=index, timestamp_sec=timestamp, image=image)
                yielded += 1
                if self.max_frames and yielded >= self.max_frames:
                    break
            index += 1

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()

    def __enter__(self) -> "VideoLoader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


def _natural_sort_key(path: Path) -> list:
    """Sort 'frame_2.jpg' before 'frame_10.jpg' by splitting digit runs."""
    import re

    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


class ImageSequenceLoader:
    """Reads a folder of tilt-angle photos as if it were a video.

    Many hologram datasets are captured as a handful of still photos per
    sample (one per tilt angle) rather than a video file. This loader
    presents the same interface as `VideoLoader` (fps, frame_count,
    width, height, iteration over `Frame`s, `release()`) so the rest of
    the pipeline doesn't need to know which source it's reading from.
    Images are read in natural filename order (frame_1, frame_2, ..., frame_10).
    """

    def __init__(
        self,
        directory: str | Path,
        fps: float = 5.0,
        image_extensions: Optional[list[str]] = None,
        max_frames: int = 0,
    ):
        self.directory = resolve_path(directory)
        self.fps = fps if fps > 0 else 5.0
        self.image_extensions = image_extensions or [".jpg", ".jpeg", ".png", ".bmp"]
        self.max_frames = max_frames
        self.logger = get_logger(self.__class__.__name__)

        if not self.directory.is_dir():
            raise VideoLoadError(f"Image sequence directory does not exist: {self.directory}")

        self.image_paths = sorted(
            (p for p in self.directory.iterdir() if p.suffix.lower() in self.image_extensions),
            key=_natural_sort_key,
        )
        if not self.image_paths:
            raise VideoLoadError(
                f"No images with extensions {self.image_extensions} found in {self.directory}"
            )

        first = cv2.imread(str(self.image_paths[0]))
        if first is None:
            raise VideoLoadError(f"Could not decode image: {self.image_paths[0]}")
        self.height, self.width = first.shape[:2]
        self.frame_count = len(self.image_paths)

        self.logger.info(
            "Opened image sequence %s (%d frames, %dx%d, assumed %.1f fps)",
            self.directory.name,
            self.frame_count,
            self.width,
            self.height,
            self.fps,
        )

    def __iter__(self) -> Iterator[Frame]:
        paths = self.image_paths[: self.max_frames] if self.max_frames else self.image_paths
        for index, path in enumerate(paths):
            image = cv2.imread(str(path))
            if image is None:
                self.logger.warning("Skipping unreadable image: %s", path)
                continue
            yield Frame(index=index, timestamp_sec=index / self.fps, image=image)

    def release(self) -> None:
        pass  # nothing to close; kept for interface parity with VideoLoader

    def __enter__(self) -> "ImageSequenceLoader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


FrameSource = "VideoLoader | ImageSequenceLoader"


def is_image_sequence_dir(path: str | Path, image_extensions: Optional[list[str]] = None) -> bool:
    """True if `path` is a directory containing at least one supported image."""
    p = resolve_path(path)
    if not p.is_dir():
        return False
    extensions = image_extensions or [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    return any(child.suffix.lower() in extensions for child in p.iterdir())


def open_frame_source(path: str | Path, settings: dict, max_frames_override: Optional[int] = None):
    """Factory: open `path` as a `VideoLoader` (file) or `ImageSequenceLoader` (folder).

    This is the single place that decides how a "sample" is read, so the
    rest of the pipeline (tracker, visualization, CLI) can stay agnostic
    to whether the input is a video file or a folder of tilt-angle photos.
    """
    video_cfg = settings.get("video", {})
    p = resolve_path(path)

    if p.is_dir():
        return ImageSequenceLoader(
            p,
            fps=video_cfg.get("image_sequence_fps", 5.0),
            image_extensions=video_cfg.get("image_extensions"),
            max_frames=max_frames_override if max_frames_override is not None else video_cfg.get("max_frames", 0),
        )

    return VideoLoader(
        p,
        frame_skip=video_cfg.get("frame_skip", 1),
        max_frames=max_frames_override if max_frames_override is not None else video_cfg.get("max_frames", 0),
        supported_extensions=video_cfg.get("supported_extensions"),
    )


def image_average_hash(image: np.ndarray, hash_size: int = 8) -> np.ndarray:
    """Compute a simple average-hash (aHash) for cheap duplicate-frame detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    resized = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    avg = resized.mean()
    return (resized > avg).flatten()


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def image_entropy(image: np.ndarray) -> float:
    """Shannon entropy of the grayscale intensity histogram; low = low-information frame."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-9)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def timer() -> float:
    return time.perf_counter()
