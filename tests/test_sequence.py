"""Unit tests for app.sequence: run-length encoding and sequence compression."""

from __future__ import annotations

from app.sequence import compress_labels, compress_sequence
from app.tracker import FrameRecord


def _make_records(labels: list[str]) -> list[FrameRecord]:
    return [
        FrameRecord(
            frame_index=i,
            timestamp_sec=i / 10.0,
            label=label,
            raw_label=label,
            confidence=0.9,
            mean_hue=0.0,
            median_hue=0.0,
            dominant_hue=0.0,
            filtered_out=False,
        )
        for i, label in enumerate(labels)
    ]


def test_compress_labels_genuine_pattern():
    labels = ["green"] * 4 + ["neutral"] * 4 + ["violet"] * 4
    assert compress_labels(labels) == "GNV"


def test_compress_labels_counterfeit_pattern():
    labels = ["neutral"] * 4 + ["green"] * 4 + ["violet"] * 4
    assert compress_labels(labels) == "NGV"


def test_compress_labels_ignores_unknown():
    labels = ["unknown"] * 3 + ["green"] * 4 + ["unknown"] * 2 + ["violet"] * 4
    assert compress_labels(labels) == "GV"


def test_compress_sequence_drops_short_noise_runs():
    labels = ["green"] * 4 + ["violet"] * 1 + ["neutral"] * 4 + ["violet"] * 4
    records = _make_records(labels)
    sequence_str, runs = compress_sequence(records, min_run_length=2)
    assert sequence_str == "GNV"
    assert [r.frame_count for r in runs] == [4, 4, 4]


def test_compress_sequence_merges_adjacent_after_noise_removal():
    # A 1-frame blip of "neutral" between two green runs should be dropped,
    # merging the surrounding green runs back into one.
    labels = ["green"] * 3 + ["neutral"] * 1 + ["green"] * 3 + ["violet"] * 4
    records = _make_records(labels)
    sequence_str, runs = compress_sequence(records, min_run_length=2)
    assert sequence_str == "GV"
    assert runs[0].frame_count == 6
