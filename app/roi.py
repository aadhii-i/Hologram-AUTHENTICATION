"""Region-of-interest (ROI) detection: automatic ArUco markers with a
manual-selection fallback, plus persistence of ROI coordinates to YAML.

ROI boxes are always stored/returned normalized to [0, 1] as
`(x1, y1, x2, y2)` so they remain valid across frames of the same
resolution and are easy to serialize.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from app.utils import get_logger, load_calibration, resolve_path, save_yaml

logger = get_logger(__name__)

Box = tuple[float, float, float, float]


def _clip_box(box: Box) -> Box:
    x1, y1, x2, y2 = box
    x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
    return (x1, y1, x2, y2)


def normalized_to_pixels(box: Box, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        int(round(x1 * width)),
        int(round(y1 * height)),
        int(round(x2 * width)),
        int(round(y2 * height)),
    )


def pixels_to_normalized(box: tuple[int, int, int, int], width: int, height: int) -> Box:
    x1, y1, x2, y2 = box
    return (x1 / width, y1 / height, x2 / width, y2 / height)


def apply_roi(image: np.ndarray, box: Box, padding_ratio: float = 0.0) -> np.ndarray:
    """Crop `image` to a normalized ROI box, with optional symmetric padding."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = _clip_box(box)

    if padding_ratio:
        bw, bh = x2 - x1, y2 - y1
        x1 -= bw * padding_ratio
        x2 += bw * padding_ratio
        y1 -= bh * padding_ratio
        y2 += bh * padding_ratio
        x1, y1, x2, y2 = _clip_box((x1, y1, x2, y2))

    px1, py1, px2, py2 = normalized_to_pixels((x1, y1, x2, y2), w, h)
    px2, py2 = max(px2, px1 + 1), max(py2, py1 + 1)
    return image[py1:py2, px1:px2]


def detect_aruco_roi(
    image: np.ndarray, dictionary_name: str = "DICT_4X4_50"
) -> Optional[Box]:
    """Detect ArUco markers and return the normalized bounding box enclosing all of them.

    Returns None if no markers are found or the aruco module is unavailable.
    """
    if not hasattr(cv2, "aruco"):
        logger.debug("cv2.aruco not available; skipping ArUco ROI detection")
        return None

    try:
        aruco_dict_id = getattr(cv2.aruco, dictionary_name)
    except AttributeError:
        logger.warning("Unknown ArUco dictionary '%s'; falling back", dictionary_name)
        return None

    dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(corners) == 0:
        return None

    all_pts = np.concatenate(corners, axis=1).reshape(-1, 2)
    x1, y1 = all_pts.min(axis=0)
    x2, y2 = all_pts.max(axis=0)

    h, w = image.shape[:2]
    box = pixels_to_normalized((int(x1), int(y1), int(x2), int(y2)), w, h)
    logger.info("ArUco ROI detected from %d marker(s): %s", len(corners), box)
    return _clip_box(box)


def select_roi_manual(image: np.ndarray, window_name: str = "Select ROI") -> Optional[Box]:
    """Open an interactive OpenCV window for the user to drag-select a ROI.

    Requires a display; raises RuntimeError in headless environments.
    Press ENTER/SPACE to confirm, 'c' to cancel.
    """
    h, w = image.shape[:2]
    try:
        x, y, bw, bh = cv2.selectROI(window_name, image, showCrosshair=True)
        cv2.destroyWindow(window_name)
    except cv2.error as exc:
        raise RuntimeError(
            "Manual ROI selection requires a GUI display (no headless support)."
        ) from exc

    if bw == 0 or bh == 0:
        return None
    return pixels_to_normalized((x, y, x + bw, y + bh), w, h)


def default_roi(config: dict) -> Box:
    box = config.get("roi", {}).get(
        "default_box", [0.35, 0.35, 0.65, 0.65]
    )
    return _clip_box(tuple(box))  # type: ignore[arg-type]


def save_roi(box: Box, path: str = "configs/calibration.yaml") -> None:
    calibration = load_calibration(path)
    calibration.setdefault("roi", {})["box"] = list(box)
    save_yaml(path, calibration)
    logger.info("Saved ROI %s to %s", box, path)


def load_saved_roi(path: str = "configs/calibration.yaml") -> Optional[Box]:
    calibration = load_calibration(path)
    box = calibration.get("roi", {}).get("box")
    return tuple(box) if box else None  # type: ignore[return-value]


class ROISelector:
    """Resolves a ROI for a frame using: saved calibration -> ArUco -> manual -> default."""

    def __init__(self, settings: dict):
        self.settings = settings.get("roi", {})
        self.calibration_path = settings.get("paths", {}).get(
            "calibration_config", "configs/calibration.yaml"
        )
        self.logger = get_logger(self.__class__.__name__)

    def resolve(
        self, first_frame: np.ndarray, mode: Optional[str] = None, manual_box: Optional[Box] = None
    ) -> tuple[Box, str]:
        """Return (box, source) where source in {'manual_arg', 'saved', 'aruco', 'manual_gui', 'default'}."""
        mode = mode or self.settings.get("mode", "auto")

        if manual_box is not None:
            return _clip_box(manual_box), "manual_arg"

        saved = load_saved_roi(self.calibration_path)
        if saved is not None:
            self.logger.info("Using saved ROI from calibration.yaml: %s", saved)
            return saved, "saved"

        if mode in ("auto", "aruco"):
            aruco_box = detect_aruco_roi(
                first_frame, self.settings.get("aruco_dictionary", "DICT_4X4_50")
            )
            if aruco_box is not None:
                return aruco_box, "aruco"
            if mode == "aruco":
                self.logger.warning("ArUco requested but no markers found; using default ROI")
                return default_roi({"roi": self.settings}), "default"
            self.logger.info("No ArUco markers found; falling back to manual/default ROI")

        if mode == "manual":
            try:
                manual = select_roi_manual(first_frame)
                if manual is not None:
                    return manual, "manual_gui"
            except RuntimeError as exc:
                self.logger.warning("%s Falling back to default ROI.", exc)

        return default_roi({"roi": self.settings}), "default"
