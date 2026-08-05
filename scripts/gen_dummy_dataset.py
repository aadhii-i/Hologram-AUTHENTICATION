"""Regenerate the synthetic placeholder dataset (dataset/genuine, dataset/counterfeit).

Produces photo-folder samples simulating a hand-held tilt sequence through
the fixed color transition (GENUINE: G->N->V, COUNTERFEIT: N->G->V), with
per-frame jitter, speckle noise, and an occasional drifting specular
highlight so the frame-quality filters have something realistic to reject.
Colors are the same BGR swatches tests/conftest.py uses, guaranteed to land
inside the shipped configs/hsv.yaml bands.
"""

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CANVAS_W, CANVAS_H = 640, 480
ROI = (0.21583333333333332, 0.235625, 0.7933333333333333, 0.603125)
BG_BGR = (235, 235, 235)

GREEN_BGR = (60, 200, 60)
VIOLET_BGR = (180, 40, 150)
NEUTRAL_BGR = (200, 200, 200)

GENUINE_SEQUENCE = [GREEN_BGR, NEUTRAL_BGR, VIOLET_BGR]
COUNTERFEIT_SEQUENCE = [NEUTRAL_BGR, GREEN_BGR, VIOLET_BGR]
FRAMES_PER_STATE = 5  # must exceed sequence.smoothing_window (5) so majority-vote
                       # smoothing can't tie at state boundaries


def make_frame(color_bgr: tuple[int, int, int], seed: int, glare: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((CANVAS_H, CANVAS_W, 3), BG_BGR, dtype=np.uint8)

    x1, y1 = int(ROI[0] * CANVAS_W), int(ROI[1] * CANVAS_H)
    x2, y2 = int(ROI[2] * CANVAS_W), int(ROI[3] * CANVAS_H)

    patch = np.full((y2 - y1, x2 - x1, 3), color_bgr, dtype=np.int16)
    patch += rng.integers(-8, 8, size=3)  # per-frame color jitter
    patch += rng.normal(0, 6, patch.shape).astype(np.int16)  # speckle texture
    patch = np.clip(patch, 0, 255).astype(np.uint8)

    dx, dy = rng.integers(-4, 5), rng.integers(-4, 5)  # hand-held jitter
    img[y1 + dy : y2 + dy, x1 + dx : x2 + dx] = patch

    if glare:
        cx, cy = rng.integers(x1, x2), rng.integers(y1, y2)
        cv2.circle(img, (int(cx), int(cy)), 20, (255, 255, 255), -1)

    return img


def build_sample(folder: Path, sequence: list[tuple[int, int, int]], seed: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    idx = 1
    mid_state = len(sequence) // 2
    mid_frame = FRAMES_PER_STATE // 2
    for state_i, color in enumerate(sequence):
        for f in range(FRAMES_PER_STATE):
            glare = f == mid_frame and state_i == mid_state  # one glare frame, mid-run so it can't skew a boundary
            img = make_frame(color, seed=seed * 100 + idx, glare=glare)
            cv2.imwrite(str(folder / f"frame_{idx:02d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            idx += 1


def main() -> None:
    for i, name in enumerate(["sample_001", "sample_002"]):
        build_sample(ROOT / "dataset" / "genuine" / name, GENUINE_SEQUENCE, seed=10 + i)
        build_sample(ROOT / "dataset" / "counterfeit" / name, COUNTERFEIT_SEQUENCE, seed=20 + i)
    print("Generated synthetic genuine/counterfeit samples under dataset/")


if __name__ == "__main__":
    main()
