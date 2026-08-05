"""Classical image-preprocessing operations applied before HSV classification.

Every operation is a pure function on a BGR `np.ndarray` so it can be
unit-tested and composed independently. `Preprocessor` wires them together
into the pipeline driven by `configs/settings.yaml`.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from app.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Individual operations
# ---------------------------------------------------------------------------

def gaussian_blur(image: np.ndarray, kernel: int = 5) -> np.ndarray:
    kernel = kernel if kernel % 2 == 1 else kernel + 1
    return cv2.GaussianBlur(image, (kernel, kernel), 0)


def median_blur(image: np.ndarray, kernel: int = 5) -> np.ndarray:
    kernel = kernel if kernel % 2 == 1 else kernel + 1
    return cv2.medianBlur(image, kernel)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """Contrast-Limited Adaptive Histogram Equalization on the L channel of LAB."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    l_eq = clahe.apply(l)
    merged = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    if gamma <= 0:
        gamma = 1.0
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
    ).astype("uint8")
    return cv2.LUT(image, table)


def estimate_auto_gamma(image: np.ndarray, target_mean: float = 128.0) -> float:
    """Pick a gamma value that pushes mean brightness toward `target_mean`."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = max(float(gray.mean()), 1e-6)
    gamma = np.log(target_mean / 255.0) / np.log(mean_brightness / 255.0 + 1e-9)
    return float(np.clip(gamma, 0.3, 3.0))


def white_balance(image: np.ndarray) -> np.ndarray:
    """Gray-world white balance: scale each channel so its mean matches the overall gray mean."""
    result = image.astype(np.float32)
    b, g, r = cv2.split(result)
    gray_mean = (b.mean() + g.mean() + r.mean()) / 3.0
    for channel in (b, g, r):
        channel_mean = channel.mean()
        if channel_mean > 1e-6:
            channel *= gray_mean / channel_mean
    balanced = cv2.merge((b, g, r))
    return np.clip(balanced, 0, 255).astype(np.uint8)


def normalize_brightness(image: np.ndarray, target_mean: float = 128.0) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    current_mean = max(float(v.mean()), 1e-6)
    v *= target_mean / current_mean
    v = np.clip(v, 0, 255)
    hsv = cv2.merge((h, s, v)).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def reduce_reflection(image: np.ndarray, threshold: int = 245) -> np.ndarray:
    """Suppress specular highlights via inpainting on a bright-pixel mask."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    if mask.sum() == 0:
        return image
    return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)


def remove_noise(image: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    merged = cv2.merge((y_eq, cr, cb))
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)


# ---------------------------------------------------------------------------
# Frame stabilization (bonus: optical-flow based)
# ---------------------------------------------------------------------------

class FrameStabilizer:
    """Feature-tracking optical-flow stabilizer.

    Estimates a rigid transform between consecutive frames from sparse
    Lucas-Kanade optical flow and warps the current frame back into
    alignment with the first frame seen.
    """

    def __init__(self):
        self._prev_gray: Optional[np.ndarray] = None
        self._cum_transform = np.eye(2, 3, dtype=np.float32)
        self.logger = get_logger(self.__class__.__name__)

    def reset(self) -> None:
        self._prev_gray = None
        self._cum_transform = np.eye(2, 3, dtype=np.float32)

    def stabilize(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None:
            self._prev_gray = gray
            return image

        prev_pts = cv2.goodFeaturesToTrack(
            self._prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30
        )
        if prev_pts is None or len(prev_pts) < 6:
            self._prev_gray = gray
            return image

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, prev_pts, None)
        if curr_pts is None:
            self._prev_gray = gray
            return image

        valid = status.flatten() == 1
        prev_valid, curr_valid = prev_pts[valid], curr_pts[valid]
        if len(prev_valid) < 6:
            self._prev_gray = gray
            return image

        transform, _ = cv2.estimateAffinePartial2D(prev_valid, curr_valid)
        self._prev_gray = gray
        if transform is None:
            return image

        h, w = image.shape[:2]
        # Invert the estimated motion to cancel it out.
        inv = cv2.invertAffineTransform(transform)
        stabilized = cv2.warpAffine(image, inv, (w, h), borderMode=cv2.BORDER_REPLICATE)
        return stabilized


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Preprocessor:
    """Runs the configured sequence of preprocessing steps on each frame."""

    def __init__(self, config: dict):
        self.config = config.get("preprocessing", {})
        self.stabilizer = FrameStabilizer() if self.config.get("stabilize", True) else None
        self.logger = get_logger(self.__class__.__name__)

    def reset(self) -> None:
        if self.stabilizer:
            self.stabilizer.reset()

    def process(self, image: np.ndarray) -> np.ndarray:
        out = image

        if self.stabilizer is not None:
            out = self.stabilizer.stabilize(out)

        if self.config.get("white_balance", True):
            out = white_balance(out)

        gamma = self.config.get("gamma", 1.0)
        if self.config.get("auto_gamma", True):
            gamma = estimate_auto_gamma(out, self.config.get("brightness_target", 128))
        if gamma and abs(gamma - 1.0) > 1e-3:
            out = gamma_correction(out, gamma)

        out = normalize_brightness(out, self.config.get("brightness_target", 128))
        out = reduce_reflection(out)

        clip_limit = self.config.get("clahe_clip_limit", 2.0)
        tile_grid = self.config.get("clahe_tile_grid", 8)
        out = apply_clahe(out, clip_limit, tile_grid)

        gauss_k = self.config.get("gaussian_kernel", 5)
        if gauss_k and gauss_k > 1:
            out = gaussian_blur(out, gauss_k)

        median_k = self.config.get("median_kernel", 5)
        if median_k and median_k > 1:
            out = median_blur(out, median_k)

        return out
