#!/usr/bin/env python3
"""Command-line entry point for the Hologram Genuineness Detector.

Subcommands
-----------
  detect     Run the full pipeline on a single video.
  batch      Run the pipeline over every video in a directory (or the
             dataset/{genuine,counterfeit,test} folders by default).
  calibrate  Launch the interactive HSV threshold calibration tool.
  roi        Launch interactive manual ROI selection and save it.
  webcam     Real-time webcam mode with a live confidence meter.

A "sample" passed to `detect`/`batch`/`calibrate`/`roi` may be either a
video file (mp4/mov/avi/mkv) or a folder of tilt-angle photos
(e.g. dataset/genuine/sample_001/ containing frame_01.jpg, frame_02.jpg, ...).

Examples
--------
  python run.py detect dataset/test/sample.mp4
  python run.py detect dataset/genuine/sample_001
  python run.py batch dataset/genuine --roi-mode auto
  python run.py calibrate dataset/genuine/sample_001 --classes green violet neutral
  python run.py roi dataset/genuine/sample_001
  python run.py webcam --camera-index 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.calibration import CalibrationTool, merge_hsv_with_calibration
from app.detector import HologramDetector
from app.hsv import classify_frame
from app.roi import ROISelector, apply_roi, save_roi, select_roi_manual
from app.utils import (
    ensure_dir,
    get_logger,
    is_image_sequence_dir,
    load_calibration,
    load_hsv_config,
    load_settings,
    open_frame_source,
)

logger = get_logger("cli")


def cmd_detect(args: argparse.Namespace) -> None:
    detector = HologramDetector()
    result = detector.process_video(
        args.video,
        roi_mode=args.roi_mode,
        video_id=args.video_id,
        generate_report=not args.no_report,
        generate_annotated_video=not args.no_video,
    )
    print(json.dumps(result.to_json_dict(), indent=2))


def cmd_batch(args: argparse.Namespace) -> None:
    settings = load_settings()
    if args.directory:
        directories = [Path(args.directory)]
    else:
        dataset_dir = Path(settings.get("paths", {}).get("dataset_dir", "dataset"))
        directories = [dataset_dir / "genuine", dataset_dir / "counterfeit", dataset_dir / "test"]

    video_cfg = settings.get("video", {})
    supported = tuple(video_cfg.get("supported_extensions", [".mp4"]))
    image_extensions = video_cfg.get("image_extensions")

    videos: list[Path] = []
    for d in directories:
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix.lower() in supported:
                videos.append(p)
            elif is_image_sequence_dir(p, image_extensions):
                videos.append(p)

    if not videos:
        logger.warning("No videos or image-sequence sample folders found in %s", directories)
        return

    detector = HologramDetector()
    rows = []
    for video_path in videos:
        try:
            result = detector.process_video(
                str(video_path),
                roi_mode=args.roi_mode,
                generate_report=not args.no_report,
                generate_annotated_video=not args.no_video,
            )
            rows.append(
                {
                    "video": video_path.name,
                    "label": result.label,
                    "sequence": result.sequence,
                    "confidence": result.confidence,
                    "processing_time_sec": result.processing_time_sec,
                }
            )
        except Exception as exc:  # keep batch runs alive on a single bad file
            logger.exception("Failed to process %s", video_path)
            rows.append({"video": video_path.name, "label": "ERROR", "sequence": "", "confidence": 0.0, "processing_time_sec": 0.0})

    summary_df = pd.DataFrame(rows)
    out_path = ensure_dir(settings.get("paths", {}).get("reports_dir", "outputs/reports")) / "batch_summary.csv"
    summary_df.to_csv(out_path, index=False)
    print(summary_df.to_string(index=False))
    print(f"\nSaved batch summary to {out_path}")


def cmd_calibrate(args: argparse.Namespace) -> None:
    settings = load_settings()
    loader = open_frame_source(args.video, settings)
    frame = None
    for f in loader:
        if f.index >= args.frame_index:
            frame = f.image
            break
    loader.release()
    if frame is None:
        raise SystemExit(f"Could not read frame {args.frame_index} from {args.video}")

    roi_selector = ROISelector(settings)
    roi_box, source = roi_selector.resolve(frame, mode="auto")
    roi_crop = apply_roi(frame, roi_box)
    print(f"Using ROI from '{source}': {roi_box}")

    tool = CalibrationTool()
    overrides = tool.calibrate_all(roi_crop, classes=args.classes)
    tool.save(overrides)
    print("Saved calibration overrides:", overrides)


def cmd_roi(args: argparse.Namespace) -> None:
    settings = load_settings()
    loader = open_frame_source(args.video, settings)
    first = next(iter(loader))
    loader.release()

    box = select_roi_manual(first.image)
    if box is None:
        print("No ROI selected; aborting.")
        return
    save_roi(box)
    print(f"Saved ROI {box} to configs/calibration.yaml")


def cmd_webcam(args: argparse.Namespace) -> None:
    """Bonus: real-time webcam mode with a live confidence meter overlay."""
    settings = load_settings()
    raw_hsv = load_hsv_config()
    calibration = load_calibration()
    hsv_config = merge_hsv_with_calibration(raw_hsv, calibration)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera_index}")

    roi_box = tuple(settings.get("roi", {}).get("default_box", [0.35, 0.35, 0.65, 0.65]))
    print("Press 'q' to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            roi_crop = apply_roi(frame, roi_box)
            color = classify_frame(roi_crop, hsv_config)

            h, w = frame.shape[:2]
            x1, y1, x2, y2 = [int(v) for v in (roi_box[0] * w, roi_box[1] * h, roi_box[2] * w, roi_box[3] * h)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, f"{color.label.upper()}", (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
            )

            meter_w = int(200 * color.confidence)
            cv2.rectangle(frame, (10, h - 40), (210, h - 20), (60, 60, 60), -1)
            cv2.rectangle(frame, (10, h - 40), (10 + meter_w, h - 20), (46, 204, 113), -1)
            cv2.putText(
                frame, f"confidence {color.confidence:.2f}", (10, h - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )

            cv2.imshow("Hologram Detector - Webcam", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hologram Genuineness Detector CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_detect = sub.add_parser("detect", help="Run detection on a single video")
    p_detect.add_argument("video", help="Path to the input video")
    p_detect.add_argument("--roi-mode", choices=["auto", "aruco", "manual"], default=None)
    p_detect.add_argument("--video-id", default=None)
    p_detect.add_argument("--no-report", action="store_true")
    p_detect.add_argument("--no-video", action="store_true")
    p_detect.set_defaults(func=cmd_detect)

    p_batch = sub.add_parser("batch", help="Run detection over a directory of videos")
    p_batch.add_argument("directory", nargs="?", default=None)
    p_batch.add_argument("--roi-mode", choices=["auto", "aruco", "manual"], default=None)
    p_batch.add_argument("--no-report", action="store_true")
    p_batch.add_argument("--no-video", action="store_true")
    p_batch.set_defaults(func=cmd_batch)

    p_calib = sub.add_parser("calibrate", help="Interactive HSV threshold calibration")
    p_calib.add_argument("video")
    p_calib.add_argument("--frame-index", type=int, default=0)
    p_calib.add_argument("--classes", nargs="+", default=["green", "violet", "neutral"])
    p_calib.set_defaults(func=cmd_calibrate)

    p_roi = sub.add_parser("roi", help="Interactive manual ROI selection")
    p_roi.add_argument("video")
    p_roi.set_defaults(func=cmd_roi)

    p_webcam = sub.add_parser("webcam", help="Real-time webcam mode")
    p_webcam.add_argument("--camera-index", type=int, default=0)
    p_webcam.set_defaults(func=cmd_webcam)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
