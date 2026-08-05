"""Frame filtering (Phase 6) and per-frame color tracking (Phase 7 input).

`FrameFilter` screens out frames that would corrupt the color timeline
(motion blur, over/under-exposure, reflection, duplicates, outliers,
low-information frames). `ColorTracker` drives the ROI -> preprocess ->
HSV-classify loop over a video and applies bonus smoothing (multi-frame
averaging + Kalman filtering of the hue signal).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

from app.hsv import ColorResult, classify_frame
from app.preprocessing import Preprocessor
from app.roi import Box, apply_roi
from app.utils import (
    Frame,
    VideoLoader,
    get_logger,
    hamming_distance,
    image_average_hash,
    image_entropy,
)

logger = get_logger(__name__)


@dataclass
class FrameRecord:
    """One row of the color timeline (Phase 7)."""

    frame_index: int
    timestamp_sec: float
    label: str
    raw_label: str
    confidence: float
    mean_hue: float
    median_hue: float
    dominant_hue: float
    pixel_pct: dict[str, float] = field(default_factory=dict)
    filtered_out: bool = False
    filter_reason: Optional[str] = None


class FrameFilter:
    """Stateful quality filter applied frame-by-frame in temporal order."""

    def __init__(self, config: dict):
        cfg = config.get("filtering", {})
        self.blur_threshold = cfg.get("blur_laplacian_threshold", 60.0)
        self.overexposure_ratio = cfg.get("overexposure_ratio", 0.6)
        self.overexposure_v = cfg.get("overexposure_v", 250)
        self.underexposure_ratio = cfg.get("underexposure_ratio", 0.6)
        self.underexposure_v = cfg.get("underexposure_v", 15)
        self.duplicate_hash_threshold = cfg.get("duplicate_hash_threshold", 3)
        self.outlier_zscore = cfg.get("outlier_zscore", 3.0)
        self.min_entropy = cfg.get("min_information_entropy", 2.5)

        self._prev_hash: Optional[np.ndarray] = None
        self._hue_history: list[float] = []
        self.logger = get_logger(self.__class__.__name__)

    def reset(self) -> None:
        self._prev_hash = None
        self._hue_history = []

    @staticmethod
    def blur_score(image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def is_blurry(self, image: np.ndarray) -> bool:
        return self.blur_score(image) < self.blur_threshold

    def is_overexposed(self, image: np.ndarray) -> bool:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bright_ratio = float(np.mean(gray >= self.overexposure_v))
        return bright_ratio >= self.overexposure_ratio

    def is_underexposed(self, image: np.ndarray) -> bool:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dark_ratio = float(np.mean(gray <= self.underexposure_v))
        return dark_ratio >= self.underexposure_ratio

    def is_duplicate(self, image: np.ndarray) -> bool:
        current_hash = image_average_hash(image)
        is_dup = False
        if self._prev_hash is not None:
            dist = hamming_distance(current_hash, self._prev_hash)
            is_dup = dist <= self.duplicate_hash_threshold
        self._prev_hash = current_hash
        return is_dup

    def is_low_information(self, image: np.ndarray) -> bool:
        return image_entropy(image) < self.min_entropy

    def is_hue_outlier(self, hue: float) -> bool:
        self._hue_history.append(hue)
        if len(self._hue_history) < 8:
            return False
        window = np.array(self._hue_history[-30:])
        mean, std = window.mean(), window.std()
        if std < 1e-6:
            return False
        z = abs(hue - mean) / std
        return bool(z > self.outlier_zscore)

    def evaluate(self, roi_image: np.ndarray, color: ColorResult) -> Optional[str]:
        """Return a filter-reason string if the frame should be dropped, else None."""
        if self.is_blurry(roi_image):
            return "motion_blur"
        if self.is_overexposed(roi_image) or color.is_glare:
            return "overexposure"
        if self.is_underexposed(roi_image) or color.is_dark:
            return "underexposure"
        # A confidently classified "neutral" frame *is* low-saturation by
        # definition (that's the class boundary) -- only treat low
        # saturation as a quality problem when it left the frame
        # unclassifiable, otherwise this filter would erase every genuine
        # neutral frame and make the G-N-V sequence undetectable.
        if color.is_low_saturation and color.label != "neutral":
            return "low_saturation"
        if self.is_low_information(roi_image):
            return "low_information"
        if self.is_duplicate(roi_image):
            return "duplicate"
        if self.is_hue_outlier(color.mean_hue):
            return "outlier"
        return None


class HueKalmanSmoother:
    """1D Kalman filter (position + velocity) smoothing the hue signal (bonus feature)."""

    def __init__(self, process_noise: float = 1e-2, measurement_noise: float = 4.0):
        self.kf = cv2.KalmanFilter(2, 1)
        self.kf.transitionMatrix = np.array([[1, 1], [0, 1]], dtype=np.float32)
        self.kf.measurementMatrix = np.array([[1, 0]], dtype=np.float32)
        self.kf.processNoiseCov = process_noise * np.eye(2, dtype=np.float32)
        self.kf.measurementNoiseCov = measurement_noise * np.eye(1, dtype=np.float32)
        self._initialized = False

    def smooth(self, hue: float) -> float:
        measurement = np.array([[np.float32(hue)]])
        if not self._initialized:
            self.kf.statePre = np.array([[hue], [0]], dtype=np.float32)
            self.kf.statePost = np.array([[hue], [0]], dtype=np.float32)
            self._initialized = True
        self.kf.predict()
        corrected = self.kf.correct(measurement)
        return float(corrected[0, 0])


def majority_vote_smooth(labels: list[str], window: int) -> list[str]:
    """Bonus: multi-frame averaging of *labels* via sliding majority vote."""
    if window <= 1 or len(labels) <= 1:
        return labels
    half = window // 2
    smoothed = []
    for i in range(len(labels)):
        lo, hi = max(0, i - half), min(len(labels), i + half + 1)
        window_labels = labels[lo:hi]
        counts = {lbl: window_labels.count(lbl) for lbl in set(window_labels)}
        smoothed.append(max(counts, key=counts.get))
    return smoothed


class ColorTracker:
    """Drives ROI cropping, preprocessing, HSV classification, filtering and smoothing."""

    def __init__(self, settings: dict, hsv_config: dict):
        self.settings = settings
        self.hsv_config = hsv_config
        self.preprocessor = Preprocessor(settings)
        self.frame_filter = FrameFilter(settings)
        self.logger = get_logger(self.__class__.__name__)

    def track(self, video_loader: VideoLoader, roi_box: Box) -> list[FrameRecord]:
        self.preprocessor.reset()
        self.frame_filter.reset()

        seq_cfg = self.settings.get("sequence", {})
        padding = self.settings.get("roi", {}).get("padding_ratio", 0.0)
        min_conf = seq_cfg.get("min_confidence", 0.35)

        raw_records: list[FrameRecord] = []
        hue_smoother = (
            HueKalmanSmoother() if seq_cfg.get("use_kalman_smoothing", True) else None
        )

        frame: Frame
        progress = tqdm(
            video_loader,
            total=video_loader.frame_count or None,
            desc="Tracking colors",
            unit="frame",
            leave=False,
        )
        for frame in progress:
            # Preprocess the full frame *before* cropping: white balance,
            # CLAHE, and brightness normalization rely on scene-level
            # statistics (e.g. gray-world assumes the frame averages to
            # gray). Running them on an already-cropped, near-solid-color
            # ROI would fight the very color signal we're trying to
            # measure, washing it out toward gray.
            processed_full = self.preprocessor.process(frame.image)
            processed = apply_roi(processed_full, roi_box, padding)
            if processed.size == 0:
                continue

            color = classify_frame(processed, self.hsv_config)

            # Quality checks (blur/exposure/duplicate/entropy) run on the
            # *raw* ROI, not the preprocessed one: the pipeline's own
            # Gaussian/median blur and denoising would otherwise make every
            # frame look "blurry" right after we've deliberately smoothed it.
            raw_roi = apply_roi(frame.image, roi_box, padding)
            reason = self.frame_filter.evaluate(raw_roi, color)
            filtered_out = reason is not None

            hue_value = color.mean_hue
            if hue_smoother is not None:
                hue_value = hue_smoother.smooth(color.mean_hue)

            label = color.label
            if not filtered_out and color.confidence < min_conf:
                label = "unknown"

            raw_records.append(
                FrameRecord(
                    frame_index=frame.index,
                    timestamp_sec=frame.timestamp_sec,
                    label=label,
                    raw_label=color.label,
                    confidence=color.confidence,
                    mean_hue=hue_value,
                    median_hue=color.median_hue,
                    dominant_hue=color.dominant_hue,
                    pixel_pct=color.pixel_pct,
                    filtered_out=filtered_out,
                    filter_reason=reason,
                )
            )

        window = seq_cfg.get("smoothing_window", 5)
        valid_indices = [i for i, r in enumerate(raw_records) if not r.filtered_out]
        valid_labels = [raw_records[i].label for i in valid_indices]
        smoothed_labels = majority_vote_smooth(valid_labels, window)
        for idx, new_label in zip(valid_indices, smoothed_labels):
            raw_records[idx].label = new_label

        n_kept = len(valid_indices)
        n_total = len(raw_records)
        self.logger.info(
            "Tracked %d frames, kept %d (%.1f%%) after filtering",
            n_total,
            n_kept,
            100.0 * n_kept / n_total if n_total else 0.0,
        )
        return raw_records
