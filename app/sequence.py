"""Phase 7 - sequence tracking: turn per-frame labels into a timeline table
and a compressed color-transition string (e.g. "GGGGNNNNVVVV" -> "GNV").
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.hsv import CLASS_SYMBOLS
from app.tracker import FrameRecord
from app.utils import get_logger

logger = get_logger(__name__)


def build_timeline(records: list[FrameRecord]) -> pd.DataFrame:
    """Return a tidy DataFrame timeline, one row per frame (Phase 7 table)."""
    rows = [
        {
            "frame_index": r.frame_index,
            "timestamp_sec": round(r.timestamp_sec, 3),
            "label": r.label,
            "raw_label": r.raw_label,
            "confidence": round(r.confidence, 4),
            "mean_hue": round(r.mean_hue, 2),
            "median_hue": round(r.median_hue, 2),
            "dominant_hue": round(r.dominant_hue, 2),
            "filtered_out": r.filtered_out,
            "filter_reason": r.filter_reason or "",
        }
        for r in records
    ]
    return pd.DataFrame(rows)


@dataclass
class SequenceRun:
    label: str
    symbol: str
    start_frame: int
    end_frame: int
    frame_count: int
    mean_confidence: float


def run_length_encode(records: list[FrameRecord]) -> list[SequenceRun]:
    """Collapse consecutive identical labels among *valid* (non-filtered) frames."""
    valid = [r for r in records if not r.filtered_out]
    runs: list[SequenceRun] = []
    for r in valid:
        if runs and runs[-1].label == r.label:
            run = runs[-1]
            run.end_frame = r.frame_index
            run.frame_count += 1
            run.mean_confidence = (
                run.mean_confidence * (run.frame_count - 1) + r.confidence
            ) / run.frame_count
        else:
            runs.append(
                SequenceRun(
                    label=r.label,
                    symbol=CLASS_SYMBOLS.get(r.label, "U"),
                    start_frame=r.frame_index,
                    end_frame=r.frame_index,
                    frame_count=1,
                    mean_confidence=r.confidence,
                )
            )
    return runs


def compress_sequence(
    records: list[FrameRecord], min_run_length: int = 2
) -> tuple[str, list[SequenceRun]]:
    """Compress the frame-level timeline into a short color-transition string.

    Steps: run-length encode -> drop "unknown" runs and runs shorter than
    `min_run_length` (treated as noise) -> merge any now-adjacent runs of
    the same label -> emit symbols (G/N/V).
    """
    runs = run_length_encode(records)
    kept = [
        run for run in runs if run.label != "unknown" and run.frame_count >= min_run_length
    ]

    merged: list[SequenceRun] = []
    for run in kept:
        if merged and merged[-1].label == run.label:
            prev = merged[-1]
            prev.end_frame = run.end_frame
            total = prev.frame_count + run.frame_count
            prev.mean_confidence = (
                prev.mean_confidence * prev.frame_count + run.mean_confidence * run.frame_count
            ) / total
            prev.frame_count = total
        else:
            merged.append(run)

    symbol_sequence = "".join(run.symbol for run in merged)
    logger.info("Compressed sequence: %s (from %d raw run(s))", symbol_sequence, len(runs))
    return symbol_sequence, merged


def compress_labels(labels: list[str], min_run_length: int = 1) -> str:
    """Lightweight helper for tests/utilities: compress a plain label list
    (e.g. ["green"]*4 + ["neutral"]*4 + ["violet"]*4) straight to "GNV",
    without needing full FrameRecord objects.
    """
    if not labels:
        return ""
    runs: list[tuple[str, int]] = []
    for label in labels:
        if runs and runs[-1][0] == label:
            runs[-1] = (label, runs[-1][1] + 1)
        else:
            runs.append((label, 1))
    kept = [(lbl, cnt) for lbl, cnt in runs if lbl != "unknown" and cnt >= min_run_length]

    merged: list[str] = []
    for lbl, _ in kept:
        symbol = CLASS_SYMBOLS.get(lbl, "U")
        if not merged or merged[-1] != symbol:
            merged.append(symbol)
    return "".join(merged)
