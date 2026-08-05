"""Phase 9 - visualization: ROI/mask snapshots, timeline & hue graphs,
transition diagrams, confidence graphs, histograms, and the annotated
output video.

Uses the non-interactive `Agg` matplotlib backend so it runs headlessly
(servers, CI, the FastAPI worker) without a display.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.hsv import build_masks, to_hsv
from app.roi import Box, apply_roi
from app.sequence import SequenceRun
from app.utils import Frame, ensure_dir, get_logger, open_frame_source
from app.tracker import FrameRecord

logger = get_logger(__name__)

LABEL_COLORS = {
    "green": "#2ecc71",
    "violet": "#8e44ad",
    "neutral": "#95a5a6",
    "unknown": "#2c3e50",
}
LABEL_BGR = {
    "green": (60, 180, 75),
    "violet": (150, 40, 130),
    "neutral": (150, 150, 150),
    "unknown": (60, 60, 60),
}


def save_roi_snapshot(roi_image: np.ndarray, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    cv2.imwrite(str(output_path), roi_image)
    return output_path


def save_mask_images(
    roi_image: np.ndarray, hsv_config: dict, output_dir: str | Path, prefix: str = "frame"
) -> dict[str, Path]:
    """Save one PNG per color-class mask overlay for a representative frame."""
    output_dir = ensure_dir(output_dir)
    hsv = to_hsv(roi_image)
    masks = build_masks(hsv, hsv_config)

    paths: dict[str, Path] = {}
    for cls, mask in masks.items():
        overlay = cv2.bitwise_and(roi_image, roi_image, mask=mask)
        path = output_dir / f"{prefix}_mask_{cls}.png"
        cv2.imwrite(str(path), overlay)
        paths[cls] = path
    return paths


def plot_color_timeline(df: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    fig, ax = plt.subplots(figsize=(11, 2.5))
    for label, color in LABEL_COLORS.items():
        subset = df[(df["label"] == label) & (~df["filtered_out"])]
        if subset.empty:
            continue
        ax.scatter(subset["timestamp_sec"], [1] * len(subset), color=color, s=14, label=label)

    filtered = df[df["filtered_out"]]
    if not filtered.empty:
        ax.scatter(
            filtered["timestamp_sec"], [1] * len(filtered), color="#e74c3c", s=6, marker="x",
            label="filtered", alpha=0.5,
        )

    ax.set_yticks([])
    ax.set_xlabel("Time (s)")
    ax.set_title("Color Timeline")
    ax.legend(loc="upper right", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_hue_graph(df: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    fig, ax = plt.subplots(figsize=(11, 3))
    valid = df[~df["filtered_out"]]
    ax.plot(valid["timestamp_sec"], valid["mean_hue"], color="#2980b9", linewidth=1.2, label="mean hue (smoothed)")
    ax.plot(valid["timestamp_sec"], valid["median_hue"], color="#e67e22", linewidth=0.8, alpha=0.6, label="median hue")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Hue (0-179)")
    ax.set_title("Hue Over Time")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_transition_graph(runs: list[SequenceRun], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    fig, ax = plt.subplots(figsize=(11, 1.8))
    left = 0
    for run in runs:
        color = LABEL_COLORS.get(run.label, "#000000")
        ax.barh(0, run.frame_count, left=left, color=color, edgecolor="white")
        ax.text(
            left + run.frame_count / 2, 0, run.symbol,
            ha="center", va="center", color="white", fontsize=11, fontweight="bold",
        )
        left += run.frame_count

    ax.set_xlim(0, max(left, 1))
    ax.set_yticks([])
    ax.set_xlabel("Frame count")
    sequence_str = "".join(r.symbol for r in runs)
    ax.set_title(f"Color Transition: {sequence_str or '(none)'}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_confidence_graph(df: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    fig, ax = plt.subplots(figsize=(11, 2.5))
    valid = df[~df["filtered_out"]]
    ax.plot(valid["timestamp_sec"], valid["confidence"], color="#27ae60", linewidth=1.2)
    ax.fill_between(valid["timestamp_sec"], valid["confidence"], alpha=0.2, color="#27ae60")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Confidence")
    ax.set_title("Per-Frame Classification Confidence")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_color_histogram(roi_image: np.ndarray, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    hsv = to_hsv(roi_image)
    hue = hsv[:, :, 0].flatten()

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(hue, bins=90, range=(0, 180), color="#34495e")
    ax.set_xlabel("Hue (0-179)")
    ax.set_ylabel("Pixel count")
    ax.set_title("Hue Histogram")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def render_annotated_video(
    video_path: str | Path,
    records: list[FrameRecord],
    roi_box: Box,
    output_path: str | Path,
    settings: dict,
) -> Path:
    """Re-play the source video (or image-sequence folder) with ROI box,
    label, and confidence overlaid."""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    loader = open_frame_source(video_path, settings)
    records_by_index = {r.frame_index: r for r in records}

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path), fourcc, loader.fps, (loader.width, loader.height)
    )

    frame: Frame
    for frame in loader:
        img = frame.image.copy()
        h, w = img.shape[:2]
        x1, y1, x2, y2 = [
            int(v) for v in (roi_box[0] * w, roi_box[1] * h, roi_box[2] * w, roi_box[3] * h)
        ]

        record = records_by_index.get(frame.index)
        label = record.label if record else "unknown"
        confidence = record.confidence if record else 0.0
        color = LABEL_BGR.get(label, (255, 255, 255))

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        caption = f"{label.upper()} ({confidence:.2f})"
        cv2.putText(img, caption, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(
            img, f"frame {frame.index}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )
        writer.write(img)

    loader.release()
    writer.release()
    logger.info("Wrote annotated video to %s", output_path)
    return output_path
