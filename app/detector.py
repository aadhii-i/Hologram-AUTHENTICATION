"""Top-level orchestrator: wires ROI -> preprocessing -> HSV classification
-> filtering -> sequence compression -> decision -> visualization ->
report generation into a single `HologramDetector.process_video()` call.

This is the module the CLI, FastAPI service, and Streamlit UI all import.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from app.calibration import merge_hsv_with_calibration
from app.decision import DecisionEngine, DecisionResult
from app.roi import Box, ROISelector, apply_roi
from app.sequence import build_timeline, compress_sequence
from app.tracker import ColorTracker, FrameRecord
from app.utils import (
    ensure_dir,
    get_logger,
    load_calibration,
    load_hsv_config,
    load_settings,
    open_frame_source,
    resolve_path,
    timer,
)
from app.visualization import (
    plot_color_histogram,
    plot_color_timeline,
    plot_confidence_graph,
    plot_hue_graph,
    plot_transition_graph,
    render_annotated_video,
    save_mask_images,
    save_roi_snapshot,
)

logger = get_logger(__name__)


@dataclass
class DetectionResult:
    video_path: str
    video_id: str
    label: str
    sequence: str
    confidence: float
    reasoning: str
    roi_box: Box
    roi_source: str
    processing_time_sec: float
    frame_count: int
    kept_frame_count: int
    fps: float
    generated_at: str
    output_paths: dict[str, str] = field(default_factory=dict)

    def to_json_dict(self) -> dict:
        d = asdict(self)
        return d


class HologramDetector:
    """End-to-end pipeline for a single video."""

    def __init__(
        self,
        settings_path: str = "configs/settings.yaml",
        hsv_config_path: str = "configs/hsv.yaml",
        calibration_path: str = "configs/calibration.yaml",
    ):
        self.settings = load_settings(settings_path)
        raw_hsv = load_hsv_config(hsv_config_path)
        calibration = load_calibration(calibration_path)
        self.hsv_config = merge_hsv_with_calibration(raw_hsv, calibration)
        self.calibration_path = calibration_path

        self.roi_selector = ROISelector(self.settings)
        self.decision_engine = DecisionEngine(self.settings)
        self.logger = get_logger(self.__class__.__name__)

    # -- pipeline ----------------------------------------------------------

    def process_video(
        self,
        video_path: str,
        roi_mode: Optional[str] = None,
        manual_roi: Optional[Box] = None,
        save_outputs: bool = True,
        video_id: Optional[str] = None,
        generate_report: bool = True,
        generate_annotated_video: bool = True,
    ) -> DetectionResult:
        start = timer()
        video_path = str(resolve_path(video_path))
        video_id = video_id or Path(video_path).stem

        # First pass: peek at frame 1 to resolve the ROI. Works whether
        # `video_path` is a video file or a folder of tilt-angle photos.
        probe_loader = open_frame_source(video_path, self.settings, max_frames_override=1)
        first_frame = next(iter(probe_loader))
        fps = probe_loader.fps
        probe_loader.release()

        roi_box, roi_source = self.roi_selector.resolve(
            first_frame.image, mode=roi_mode, manual_box=manual_roi
        )
        self.logger.info("ROI resolved via '%s': %s", roi_source, roi_box)

        # Second pass: full tracking run.
        tracker = ColorTracker(self.settings, self.hsv_config)
        track_loader = open_frame_source(video_path, self.settings)
        records = tracker.track(track_loader, roi_box)
        track_loader.release()

        timeline_df = build_timeline(records)
        min_run = self.settings.get("sequence", {}).get("min_run_length", 2)
        sequence_str, runs = compress_sequence(records, min_run_length=min_run)
        decision: DecisionResult = self.decision_engine.decide(sequence_str, runs)

        processing_time = timer() - start
        kept = int((~timeline_df["filtered_out"]).sum()) if not timeline_df.empty else 0

        result = DetectionResult(
            video_path=video_path,
            video_id=video_id,
            label=decision.label,
            sequence=decision.sequence,
            confidence=decision.confidence,
            reasoning=decision.reasoning,
            roi_box=roi_box,
            roi_source=roi_source,
            processing_time_sec=round(processing_time, 3),
            frame_count=len(records),
            kept_frame_count=kept,
            fps=fps,
            generated_at=_dt.datetime.now().isoformat(timespec="seconds"),
        )

        if save_outputs:
            self._save_outputs(
                video_path,
                video_id,
                result,
                records,
                timeline_df,
                runs,
                roi_box,
                generate_report=generate_report,
                generate_annotated_video=generate_annotated_video,
            )

        self.logger.info(
            "Processed '%s' -> %s (sequence=%s, confidence=%.2f, %.2fs)",
            video_id,
            decision.label,
            decision.sequence,
            decision.confidence,
            processing_time,
        )
        return result

    # -- outputs -------------------------------------------------------

    def _save_outputs(
        self,
        video_path: str,
        video_id: str,
        result: DetectionResult,
        records: list[FrameRecord],
        timeline_df: pd.DataFrame,
        runs,
        roi_box: Box,
        generate_report: bool = True,
        generate_annotated_video: bool = True,
    ) -> None:
        paths = self.settings.get("paths", {})
        frames_dir = ensure_dir(paths.get("frames_dir", "outputs/frames")) / video_id
        graphs_dir = ensure_dir(paths.get("graphs_dir", "outputs/graphs")) / video_id
        roi_dir = ensure_dir(paths.get("roi_dir", "outputs/roi")) / video_id
        reports_dir = ensure_dir(paths.get("reports_dir", "outputs/reports"))
        videos_dir = ensure_dir(paths.get("videos_dir", "outputs/videos"))
        ensure_dir(frames_dir)
        ensure_dir(graphs_dir)
        ensure_dir(roi_dir)

        csv_path = frames_dir / "timeline.csv"
        timeline_df.to_csv(csv_path, index=False)
        result.output_paths["timeline_csv"] = str(csv_path)

        # Representative ROI snapshot + masks from the temporal midpoint frame.
        loader = open_frame_source(video_path, self.settings)
        mid_index = len(records) // 2
        sample_roi = None
        for frame in loader:
            if frame.index >= mid_index:
                sample_roi = apply_roi(frame.image, roi_box)
                break
        loader.release()

        if sample_roi is not None:
            roi_png = roi_dir / "roi_sample.png"
            save_roi_snapshot(sample_roi, roi_png)
            result.output_paths["roi_sample"] = str(roi_png)
            mask_paths = save_mask_images(sample_roi, self.hsv_config, roi_dir, prefix="sample")
            result.output_paths.update({f"mask_{k}": str(v) for k, v in mask_paths.items()})
            hist_path = graphs_dir / "hue_histogram.png"
            plot_color_histogram(sample_roi, hist_path)
            result.output_paths["hue_histogram"] = str(hist_path)

        timeline_png = graphs_dir / "color_timeline.png"
        plot_color_timeline(timeline_df, timeline_png)
        result.output_paths["color_timeline_graph"] = str(timeline_png)

        hue_png = graphs_dir / "hue_graph.png"
        plot_hue_graph(timeline_df, hue_png)
        result.output_paths["hue_graph"] = str(hue_png)

        transition_png = graphs_dir / "transition_graph.png"
        plot_transition_graph(runs, transition_png)
        result.output_paths["transition_graph"] = str(transition_png)

        confidence_png = graphs_dir / "confidence_graph.png"
        plot_confidence_graph(timeline_df, confidence_png)
        result.output_paths["confidence_graph"] = str(confidence_png)

        if generate_annotated_video:
            video_out = videos_dir / f"{video_id}_annotated.mp4"
            render_annotated_video(video_path, records, roi_box, video_out, self.settings)
            result.output_paths["annotated_video"] = str(video_out)

        json_path = reports_dir / f"{video_id}_result.json"
        ensure_dir(reports_dir)
        with open(json_path, "w") as f:
            json.dump(result.to_json_dict(), f, indent=2)
        result.output_paths["result_json"] = str(json_path)

        if generate_report:
            pdf_path = reports_dir / f"{video_id}_report.pdf"
            self._generate_pdf_report(result, timeline_df, pdf_path)
            result.output_paths["pdf_report"] = str(pdf_path)

    def _generate_pdf_report(
        self, result: DetectionResult, timeline_df: pd.DataFrame, output_path: Path
    ) -> None:
        """Phase 10: single PDF combining metadata, sequence, and graphs."""
        with PdfPages(output_path) as pdf:
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis("off")
            lines = [
                "Hologram Genuineness Detection Report",
                "=" * 45,
                f"Video: {Path(result.video_path).name}",
                f"Generated: {result.generated_at}",
                "",
                f"Classification: {result.label}",
                f"Detected sequence: {result.sequence or '(none)'}",
                f"Confidence: {result.confidence:.2%}",
                "",
                f"ROI source: {result.roi_source}",
                f"ROI box (normalized): {tuple(round(v, 3) for v in result.roi_box)}",
                f"FPS: {result.fps:.2f}",
                f"Frames processed: {result.frame_count}",
                f"Frames kept after filtering: {result.kept_frame_count}",
                f"Processing time: {result.processing_time_sec:.2f}s",
                "",
                "Reasoning:",
                _wrap(result.reasoning, 90),
            ]
            ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=11, family="monospace")
            pdf.savefig(fig)
            plt.close(fig)

            for key in ("color_timeline_graph", "hue_graph", "transition_graph", "confidence_graph"):
                path = result.output_paths.get(key)
                if not path or not Path(path).exists():
                    continue
                img = plt.imread(path)
                fig, ax = plt.subplots(figsize=(8.5, 4))
                ax.imshow(img)
                ax.axis("off")
                ax.set_title(key.replace("_", " ").title())
                pdf.savefig(fig)
                plt.close(fig)

        self.logger.info("Generated PDF report at %s", output_path)


def _wrap(text: str, width: int) -> str:
    import textwrap

    return "\n".join(textwrap.wrap(text, width=width))
