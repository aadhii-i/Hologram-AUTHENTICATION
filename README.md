# Hologram Genuineness Detector

A classical computer-vision pipeline that classifies security holograms as
**GENUINE** or **COUNTERFEIT** from the color sequence they produce while
being tilted — **no deep learning, no training data required.**

The distinguishing signal is already known and fixed:

```
GENUINE:     Green  →  Neutral  →  Violet     (G → N → V)
COUNTERFEIT: Neutral →  Green   →  Violet     (N → G → V)
```

This project detects that transition using ROI extraction, HSV
thresholding, frame-quality filtering, and rule-based sequence matching —
nothing is learned from labeled examples.

---

## Table of contents

- [Quickstart](#quickstart)
- [Input formats: video or photo sequence](#input-formats-video-or-photo-sequence)
- [The dummy dataset](#the-dummy-dataset)
- [Usage](#usage)
  - [CLI](#cli)
  - [Streamlit UI](#streamlit-ui)
  - [FastAPI](#fastapi)
- [Architecture](#architecture)
- [Folder structure](#folder-structure)
- [Theory: why HSV instead of RGB](#theory-why-hsv-instead-of-rgb)
- [Algorithm flowchart](#algorithm-flowchart)
- [Outputs](#outputs)
- [Configuration](#configuration)
- [Testing](#testing)
- [Limitations](#limitations)
- [Future improvements](#future-improvements)

---

## Quickstart

Requires **Python 3.12+** (tested on 3.12–3.14).

```bash
cd hologram-detector
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run the pipeline on the bundled placeholder dataset
python run.py batch dataset/genuine
python run.py batch dataset/counterfeit
```

Expected output: every `sample_*` folder classified as `GENUINE`
(sequence `GNV`) under `dataset/genuine/`, and `COUNTERFEIT` (sequence
`NGV`) under `dataset/counterfeit/`, each with >95% confidence. Full
reports land in `outputs/`.

> If you're on a very new Python (3.13+) and a dependency has no
> prebuilt wheel yet, `pip install` will try to build it from source and
> may need system build tools (`gcc`, `python3-dev`). Using Python 3.12
> avoids this entirely.

## Input formats: video or photo sequence

A "sample" fed into the pipeline can be **either**:

1. **A video file** (`.mp4`, `.mov`, `.avi`, `.mkv`) of the hologram being
   tilted, or
2. **A folder of still photos**, one per tilt angle, named so they sort in
   capture order (`frame_01.jpg`, `frame_02.jpg`, ... or `img_1.png`,
   `img_2.png`, ...).

Both are read through the same internal `Frame` interface
(`app/utils.py`: `VideoLoader` / `ImageSequenceLoader`, selected
automatically by `open_frame_source()`), so every stage downstream —
preprocessing, ROI, HSV classification, filtering, sequence compression —
is agnostic to which one it's looking at. A photo folder is treated as a
fixed-fps "video" (`video.image_sequence_fps` in `configs/settings.yaml`,
default 5 fps) purely so timestamps and Kalman smoothing have something
to work with.

```
python run.py detect dataset/genuine/sample_001        # folder of photos
python run.py detect dataset/test/some_clip.mp4         # video file
```

## The dummy dataset

`dataset/genuine/` and `dataset/counterfeit/` currently contain
**synthetic placeholder samples** (`sample_001/`, `sample_002/`, plus one
in `dataset/test/`) generated for you to confirm the pipeline runs
end-to-end before wiring up real data. Each is a folder of 9 JPEGs
simulating a hand-held tilt sequence with the correct color transition,
plus jitter, speckle texture, and a drifting specular highlight so the
frame-quality filters (blur/duplicate/glare) have something realistic to
reject.

**To use your own images:** replace the contents of each
`sample_NNN/` folder (or add new ones) with your own tilt-angle photos
in capture order — same folder layout, real images instead of synthetic
ones:

```
dataset/
├── genuine/
│   ├── sample_001/        ← original hologram, tilt-angle photos
│   │   ├── frame_01.jpg
│   │   ├── frame_02.jpg
│   │   └── ...
│   └── sample_002/
├── counterfeit/
│   ├── sample_001/        ← duplicate/fake hologram, tilt-angle photos
│   └── sample_002/
├── validation/             ← optional: held-out samples for threshold tuning
└── test/                   ← optional: samples you haven't labeled yet
```

No code changes are needed — `python run.py batch dataset/genuine` (etc.)
will pick up any subfolder containing images automatically. If your real
photos don't classify correctly out of the box, recalibrate the HSV
thresholds against them (see [Configuration](#configuration)) rather than
editing the detection logic — the color bands in `configs/hsv.yaml` were
tuned against the placeholder foil colors, not yours.

## Usage

### CLI

```bash
# Single sample (video file or photo folder)
python run.py detect dataset/genuine/sample_001

# Every sample under a directory (defaults to dataset/{genuine,counterfeit,test})
python run.py batch dataset/genuine
python run.py batch                       # all three default folders

# Interactive HSV threshold calibration (opens a GUI window)
python run.py calibrate dataset/genuine/sample_001 --classes green violet neutral

# Interactive manual ROI selection, saved to configs/calibration.yaml
python run.py roi dataset/genuine/sample_001

# Real-time webcam mode with a live confidence meter overlay
python run.py webcam --camera-index 0
```

`detect`/`batch` accept `--roi-mode {auto,aruco,manual}`,
`--no-report`, and `--no-video` to skip the PDF/annotated-video outputs
for faster iteration.

### Streamlit UI

```bash
streamlit run ui/streamlit_app.py
```

Upload a video, choose automatic (ArUco) or manual (slider-defined) ROI,
run detection, and inspect the ROI crop, HSV masks, timeline/hue/
confidence graphs, frame table, and downloadable PDF report — all in a
dark-themed, single-page app.

### FastAPI

```bash
uvicorn api.main:app --reload
```

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/upload` | Upload a video, get back a `video_id` |
| `POST` | `/detect` | `{"video_id": "...", "roi_mode": "auto"}` → runs the pipeline |
| `GET` | `/result/{video_id}` | Fetch the stored result JSON |
| `GET` | `/download/{video_id}/{artifact}` | Download `pdf_report`, `timeline_csv`, `annotated_video`, etc. |

Interactive docs at `http://localhost:8000/docs`.

## Architecture

```
Video file  ──┐
              ├─→ open_frame_source() ─→ Frame(index, timestamp, image)
Photo folder ─┘

Frame ─→ ROISelector ─────────────► ROI box (normalized, cached in calibration.yaml)
              │  saved → ArUco → manual GUI → default box

Frame ─→ Preprocessor (full frame) ─→ ROI crop ─→ HSV classify (app/hsv.py)
    white balance → auto-gamma → brightness norm → reflection removal
    → CLAHE → Gaussian/median blur → (optical-flow stabilization)

  raw ROI ─→ FrameFilter (app/tracker.py)
    blur (Laplacian var) · exposure · glare · low-saturation ambiguity
    · low-information (entropy) · duplicate (aHash) · hue outlier (z-score)

  kept frames ─→ Kalman-smoothed hue + majority-vote label smoothing
       ─→ SequenceRun run-length encoding (app/sequence.py)
       ─→ compressed symbol string, e.g. "GGGGNNNNVVVV" → "GNV"

  sequence ─→ DecisionEngine (app/decision.py)
       "GNV" → GENUINE | "NGV" → COUNTERFEIT | else → UNKNOWN
       (+ confidence = frame-count-weighted mean of per-run confidence)

  result ─→ visualization.py (graphs, masks, annotated video)
         ─→ detector.py._generate_pdf_report() (PDF via matplotlib)
         ─→ CSV timeline, JSON result
```

**Why preprocessing runs on the full frame, not the ROI crop:** white
balance and brightness normalization assume scene-level statistics (e.g.
gray-world white balance assumes the *frame* averages to gray). Running
them on an already-cropped, near-solid-color hologram patch would fight
the very color signal being measured. Preprocessing therefore always
sees the full frame; only classification and filtering work on the crop.

**Why blur/exposure/duplicate checks run on the *raw* ROI, not the
preprocessed one:** the pipeline deliberately Gaussian/median-blurs and
denoises frames for classification robustness — checking "is this frame
blurry" *after* that would just measure how much we blurred it
ourselves. Quality filters look at the frame as captured; only
color-band checks (glare/dark/low-saturation) use the corrected version.

**Why "neutral" is exempt from the low-saturation quality filter:** the
neutral class is *defined* by low saturation. A blanket "reject
washed-out frames" filter would silently erase every legitimate neutral
frame and make the middle state of G→N→V undetectable. The filter only
rejects low-saturation frames that *couldn't* be confidently classified.

## Folder structure

```
hologram-detector/
├── app/                  # Pipeline library
│   ├── utils.py           # logging, config, VideoLoader, ImageSequenceLoader
│   ├── roi.py              # ArUco + manual ROI, normalized box persistence
│   ├── hsv.py               # HSV conversion, color-class masks, per-frame classification
│   ├── preprocessing.py    # blur/CLAHE/gamma/white-balance/stabilization
│   ├── calibration.py       # interactive HSV trackbar tuner
│   ├── tracker.py            # frame-quality filtering + color tracking loop
│   ├── sequence.py            # timeline table + run-length sequence compression
│   ├── decision.py             # GNV/NGV rule-based decision engine
│   ├── visualization.py         # graphs, mask overlays, annotated video
│   └── detector.py               # HologramDetector orchestrator + PDF report
├── api/main.py             # FastAPI service
├── ui/streamlit_app.py     # Streamlit UI
├── configs/                # hsv.yaml, settings.yaml, calibration.yaml
├── dataset/{genuine,counterfeit,validation,test}/
├── outputs/{frames,graphs,roi,reports,videos}/
├── notebooks/demo.ipynb
├── tests/
├── run.py                  # CLI: detect / batch / calibrate / roi / webcam
└── requirements.txt
```

## Theory: why HSV instead of RGB

RGB encodes color as three tightly coupled channels — changing the
lighting intensity on a hologram shifts *all three* R, G, and B values
simultaneously, because brightness is smeared across the whole triplet.
Detecting "is this green?" in RGB means reasoning about a moving target
in a 3D cube every time the light changes.

HSV separates that out:

- **Hue** — the actual color identity (green vs. violet vs. neutral),
  largely invariant to lighting intensity and camera exposure.
- **Saturation** — colorfulness. This is what makes "neutral" a clean,
  separate concept: it isn't a hue at all, it's *low saturation*
  regardless of hue.
- **Value** — brightness, which is exactly the channel that glare,
  shadows, and exposure changes hit hardest — and it's isolated where
  it can be filtered out (glare/dark masks) without touching the hue
  signal used for classification.

Concretely: a genuine hologram's green patch under dim light and under
bright light will land at very different points in RGB space, but at
roughly the same *hue* in HSV — only its `V` channel moves. That's why
`configs/hsv.yaml` expresses the green/violet bands as `h_min`/`h_max`
ranges rather than RGB cube corners, and why "neutral" is defined purely
by `s_max` (low saturation), independent of hue entirely.

## Algorithm flowchart

```
┌─────────────┐   ┌──────────────┐   ┌───────────────────┐
│ Video / photo│──▶│ ROI resolve  │──▶│ Per-frame loop:    │
│ sequence     │   │ (ArUco/saved/│   │  preprocess (full) │
└─────────────┘   │  manual/def.)│   │  → crop ROI         │
                   └──────────────┘   │  → HSV classify     │
                                       │  → quality filter    │
                                       └──────────┬───────────┘
                                                  ▼
                              ┌───────────────────────────────┐
                              │ Kalman hue smoothing +          │
                              │ majority-vote label smoothing    │
                              └──────────────┬────────────────┘
                                             ▼
                        ┌────────────────────────────────────┐
                        │ Run-length encode → drop short/     │
                        │ unknown runs → compress to symbols  │
                        │ e.g. GGGGNNNNVVVV → "GNV"            │
                        └───────────────────┬──────────────────┘
                                            ▼
                      "GNV"?──────▶ GENUINE
                      "NGV"?──────▶ COUNTERFEIT
                      else ───────▶ UNKNOWN
                                            │
                                            ▼
                     graphs · masks · annotated video · CSV
                     · JSON · PDF report (outputs/)
```

## Outputs

Every processed sample produces, under `outputs/`:

| File | Location |
|---|---|
| Per-frame timeline (label, confidence, hue stats, filter reason) | `outputs/frames/<id>/timeline.csv` |
| Color timeline / hue / transition / confidence graphs (PNG) | `outputs/graphs/<id>/` |
| ROI sample crop + per-class HSV mask overlays | `outputs/roi/<id>/` |
| Annotated video (ROI box + label + confidence overlay) | `outputs/videos/<id>_annotated.mp4` |
| Machine-readable result | `outputs/reports/<id>_result.json` |
| Human-readable PDF report | `outputs/reports/<id>_report.pdf` |
| Batch summary across a directory | `outputs/reports/batch_summary.csv` |

## Configuration

Everything is config-driven — no thresholds are hardcoded in the pipeline:

- **`configs/hsv.yaml`** — HSV bands per color class, glare/dark/
  low-saturation cutoffs, minimum pixel ratio to call a class "dominant."
- **`configs/settings.yaml`** — video/image-sequence handling, ROI mode,
  preprocessing toggles, frame-filtering thresholds, sequence smoothing,
  decision confidence floor, output paths.
- **`configs/calibration.yaml`** — populated by `run.py calibrate` /
  `run.py roi`; overrides `hsv.yaml` and the auto-ROI detector when
  present, so recalibrating never means editing the shipped defaults.

Recalibrate against your own hologram photos with:

```bash
python run.py roi dataset/genuine/sample_001         # pick the ROI once
python run.py calibrate dataset/genuine/sample_001    # tune H/S/V sliders per class
```

## Testing

```bash
pytest tests/ -v
```

Covers ROI cropping/ArUco fallback, HSV pixel/frame classification,
sequence compression (including noise-run dropping and re-merging),
the decision engine (all three outcomes + confidence downgrade), frame
quality filtering (blur/exposure/duplicate/entropy), and the FastAPI
upload → detect → result → download flow end-to-end via `TestClient`.

## Limitations

- **Rule-based sequence matching is strict.** A real capture with heavy
  motion blur or an interrupted tilt can legitimately produce sequences
  like `"GV"` (missing neutral) or `"GNVG"` (camera moved back), which
  fall through to `UNKNOWN` rather than a fuzzy best-guess. This is a
  deliberate precision-over-recall choice for a genuineness check, but it
  means capture quality matters.
- **HSV thresholds are foil- and lighting-specific.** The shipped bands
  were tuned against the placeholder dataset's synthetic colors; real
  foils under real lighting will likely need `run.py calibrate`.
- **ArUco ROI detection requires markers in-frame.** Without them (most
  real capture setups), the pipeline falls back to a manual or default
  ROI, which assumes the hologram is roughly centered.
- **Image-sequence timestamps are synthetic.** A folder of photos has no
  real frame timing, so it's treated as a fixed-fps clip
  (`image_sequence_fps`) purely to drive Kalman smoothing — the actual
  physical tilt speed isn't known.
- **No deep learning fallback.** If HSV thresholding can't separate the
  colors on a particular foil (e.g. very low-saturation "green"), there's
  no learned model to fall back on by design — that's the point of the
  challenge, but it's also the ceiling of this approach.

## Future improvements

- Fuzzy/edit-distance sequence matching (e.g. allow `"GNVN"` to still
  read as genuine with a confidence penalty) instead of strict equality.
- Multi-marker ArUco pose estimation to correct for perspective, not just
  bound a box.
- Per-foil-batch HSV auto-calibration from a handful of known-genuine
  reference samples, rather than one global manual calibration.
- Cross-validate the classical thresholds against a held-out
  `dataset/validation/` split to report precision/recall, not just
  per-sample confidence.
- Package the CLI as an installable console script (`pip install -e .`)
  instead of `python run.py`.
