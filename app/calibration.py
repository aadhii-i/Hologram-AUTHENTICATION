"""Interactive HSV threshold calibration tool.

Opens an OpenCV window with H/S/V min/max trackbars over a live mask
preview of a representative frame, so a user can tune the `green`,
`violet`, or `neutral` bands for a specific hologram/lighting setup and
persist the result to `configs/calibration.yaml` (which overrides
`configs/hsv.yaml` at runtime).

Requires a GUI display; raises RuntimeError in headless environments.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.hsv import build_masks, to_hsv
from app.utils import get_logger, load_calibration, load_hsv_config, save_yaml

logger = get_logger(__name__)

_TRACKBARS_BAND = [
    ("H min", "h_min", 179),
    ("H max", "h_max", 179),
    ("S min", "s_min", 255),
    ("S max", "s_max", 255),
    ("V min", "v_min", 255),
    ("V max", "v_max", 255),
]

_TRACKBARS_NEUTRAL = [
    ("S max", "s_max", 255),
    ("V min", "v_min", 255),
    ("V max", "v_max", 255),
]


class CalibrationTool:
    """Interactive per-class HSV threshold tuner."""

    def __init__(self, hsv_config_path: str = "configs/hsv.yaml",
                 calibration_path: str = "configs/calibration.yaml"):
        self.hsv_config_path = hsv_config_path
        self.calibration_path = calibration_path
        self.logger = get_logger(self.__class__.__name__)

    def _trackbar_spec(self, color_class: str):
        return _TRACKBARS_NEUTRAL if color_class == "neutral" else _TRACKBARS_BAND

    def calibrate(self, frame: np.ndarray, color_class: str) -> dict:
        """Run the interactive session on a single representative frame.

        Returns the tuned threshold dict for `color_class`. Controls:
          [s] save & continue to next class / exit
          [q] quit without saving remaining classes
        """
        if color_class not in ("green", "violet", "neutral"):
            raise ValueError(f"Unknown color class: {color_class}")

        base_config = load_hsv_config(self.hsv_config_path)
        current = dict(base_config.get(color_class, {}))
        spec = self._trackbar_spec(color_class)

        window = f"Calibrate: {color_class}"
        try:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        except cv2.error as exc:
            raise RuntimeError(
                "HSV calibration requires a GUI display (no headless support)."
            ) from exc

        def _noop(_):
            return None

        for label, key, max_val in spec:
            default = current.get(key, max_val if "max" in key else 0)
            cv2.createTrackbar(label, window, int(default), max_val, _noop)

        self.logger.info("Calibrating '%s' - press 's' to save, 'q' to quit", color_class)
        hsv = to_hsv(frame)

        while True:
            values = {key: cv2.getTrackbarPos(label, window) for label, key, _ in spec}
            if color_class == "neutral":
                preview_cfg = {"neutral": values}
            else:
                preview_cfg = {color_class: values, "neutral": base_config.get("neutral", {})}

            masks = build_masks(hsv, preview_cfg)
            mask = masks[color_class]
            overlay = cv2.bitwise_and(frame, frame, mask=mask)
            stacked = np.hstack([frame, overlay])
            cv2.imshow(window, stacked)

            key_pressed = cv2.waitKey(30) & 0xFF
            if key_pressed == ord("s"):
                current.update(values)
                break
            if key_pressed == ord("q"):
                break

        cv2.destroyWindow(window)
        return current

    def calibrate_all(self, frame: np.ndarray, classes: Optional[list[str]] = None) -> dict:
        classes = classes or ["green", "violet", "neutral"]
        overrides = {}
        for cls in classes:
            overrides[cls] = self.calibrate(frame, cls)
        return overrides

    def save(self, overrides: dict) -> None:
        calibration = load_calibration(self.calibration_path)
        hsv_overrides = calibration.setdefault("hsv_overrides", {})
        hsv_overrides.update(overrides)
        calibration["calibrated"] = True
        calibration["calibrated_on"] = _dt.datetime.now().isoformat(timespec="seconds")
        save_yaml(self.calibration_path, calibration)
        self.logger.info("Saved HSV calibration overrides to %s", self.calibration_path)


def merge_hsv_with_calibration(hsv_config: dict, calibration: dict) -> dict:
    """Apply any non-null `hsv_overrides` from calibration.yaml on top of hsv.yaml."""
    merged = {k: dict(v) if isinstance(v, dict) else v for k, v in hsv_config.items()}
    overrides = calibration.get("hsv_overrides", {}) or {}
    for cls, values in overrides.items():
        if values:
            merged.setdefault(cls, {}).update(values)
    return merged
