"""Streamlit UI for the Hologram Genuineness Detection pipeline.

Run with: `streamlit run ui/streamlit_app.py`
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.detector import HologramDetector  # noqa: E402
from app.roi import apply_roi  # noqa: E402
from app.utils import ensure_dir, load_settings  # noqa: E402

st.set_page_config(
    page_title="Hologram Genuineness Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_CSS = """
<style>
    .stApp { background-color: #0e1117; color: #e6e6e6; }
    .badge {
        display: inline-block; padding: 0.4rem 1rem; border-radius: 999px;
        font-weight: 700; font-size: 1.1rem; letter-spacing: 0.03em;
    }
    .badge-genuine { background: #1e5631; color: #7CFC9A; border: 1px solid #2ecc71; }
    .badge-counterfeit { background: #5a1e1e; color: #ff8080; border: 1px solid #e74c3c; }
    .badge-unknown { background: #3a3a3a; color: #cccccc; border: 1px solid #888888; }
    .metric-card {
        background: #161b22; padding: 1rem; border-radius: 0.5rem; border: 1px solid #30363d;
    }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)


@st.cache_resource
def get_detector() -> HologramDetector:
    return HologramDetector()


def badge_html(label: str) -> str:
    css_class = {
        "GENUINE": "badge-genuine",
        "COUNTERFEIT": "badge-counterfeit",
    }.get(label, "badge-unknown")
    return f'<span class="badge {css_class}">{label}</span>'


def main() -> None:
    st.title("🔬 Hologram Genuineness Detection")
    st.caption(
        "Classical computer-vision pipeline: detects the G→N→V (genuine) vs. "
        "N→G→V (counterfeit) color-tilt signature — no deep learning involved."
    )

    settings = load_settings()
    upload_dir = ensure_dir(settings.get("api", {}).get("upload_dir", "outputs/uploads"))

    with st.sidebar:
        st.header("1. Upload video")
        uploaded_file = st.file_uploader(
            "Hologram tilt video", type=["mp4", "mov", "avi", "mkv"]
        )

        st.header("2. Select ROI")
        roi_choice = st.radio(
            "ROI mode", ["Automatic (ArUco / default)", "Manual coordinates"], index=0
        )

        manual_roi = None
        run_clicked = False

        if uploaded_file is not None:
            video_id = st.session_state.setdefault(
                f"video_id_{uploaded_file.name}", uuid.uuid4().hex[:10]
            )
            video_path = upload_dir / f"{video_id}{Path(uploaded_file.name).suffix}"
            if not video_path.exists():
                video_path.write_bytes(uploaded_file.getvalue())

            cap = cv2.VideoCapture(str(video_path))
            ok, first_frame = cap.read()
            cap.release()

            if ok:
                if roi_choice == "Manual coordinates":
                    st.caption("Normalized box [x1, y1, x2, y2]")
                    x1 = st.slider("x1", 0.0, 1.0, 0.35, 0.01)
                    y1 = st.slider("y1", 0.0, 1.0, 0.35, 0.01)
                    x2 = st.slider("x2", 0.0, 1.0, 0.65, 0.01)
                    y2 = st.slider("y2", 0.0, 1.0, 0.65, 0.01)
                    manual_roi = (x1, y1, x2, y2)

                    preview = first_frame.copy()
                    h, w = preview.shape[:2]
                    cv2.rectangle(
                        preview,
                        (int(x1 * w), int(y1 * h)),
                        (int(x2 * w), int(y2 * h)),
                        (46, 204, 113),
                        3,
                    )
                    st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), caption="ROI preview")
                else:
                    st.image(
                        cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB),
                        caption="First frame (ArUco markers auto-detected if present)",
                    )

            st.header("3. Run")
            run_clicked = st.button("Run Detection", type="primary", use_container_width=True)

    if uploaded_file is None:
        st.info("Upload a hologram tilt video from the sidebar to begin.")
        return

    if run_clicked:
        detector = get_detector()
        roi_mode = "manual" if roi_choice == "Manual coordinates" else "auto"
        with st.spinner("Running detection pipeline (ROI → HSV → sequence → decision)..."):
            result = detector.process_video(
                str(video_path),
                roi_mode=roi_mode,
                manual_roi=manual_roi,
                video_id=video_id,
            )
        st.session_state["last_result"] = result

    result = st.session_state.get("last_result")
    if result is None:
        st.warning("Click **Run Detection** to process the uploaded video.")
        return

    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(badge_html(result.label), unsafe_allow_html=True)
        st.caption("Classification")
    with col2:
        st.markdown(f'<div class="metric-card"><b>{result.sequence or "—"}</b></div>', unsafe_allow_html=True)
        st.caption("Detected sequence")
    with col3:
        st.markdown(f'<div class="metric-card"><b>{result.confidence:.1%}</b></div>', unsafe_allow_html=True)
        st.caption("Confidence")
    with col4:
        st.markdown(f'<div class="metric-card"><b>{result.processing_time_sec:.2f}s</b></div>', unsafe_allow_html=True)
        st.caption("Processing time")

    st.info(result.reasoning)

    tab_video, tab_roi, tab_graphs, tab_timeline, tab_report = st.tabs(
        ["Original / Annotated Video", "ROI & HSV Masks", "Graphs", "Timeline Table", "Report"]
    )

    with tab_video:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Original")
            st.video(str(video_path))
        with c2:
            st.subheader("Annotated")
            annotated = result.output_paths.get("annotated_video")
            if annotated and Path(annotated).exists():
                st.video(annotated)
            else:
                st.write("Annotated video not available.")

    with tab_roi:
        roi_sample = result.output_paths.get("roi_sample")
        if roi_sample:
            st.image(roi_sample, caption="Sample ROI crop", width=300)
        mask_cols = st.columns(3)
        for i, cls in enumerate(("green", "violet", "neutral")):
            path = result.output_paths.get(f"mask_{cls}")
            if path and Path(path).exists():
                with mask_cols[i]:
                    st.image(path, caption=f"{cls.title()} mask")

    with tab_graphs:
        for key, title in (
            ("color_timeline_graph", "Color Timeline"),
            ("hue_graph", "Hue Over Time"),
            ("transition_graph", "Transition Graph"),
            ("confidence_graph", "Confidence Graph"),
            ("hue_histogram", "Hue Histogram"),
        ):
            path = result.output_paths.get(key)
            if path and Path(path).exists():
                st.subheader(title)
                st.image(path, use_container_width=True)

    with tab_timeline:
        csv_path = result.output_paths.get("timeline_csv")
        if csv_path and Path(csv_path).exists():
            df = pd.read_csv(csv_path)
            st.dataframe(df, use_container_width=True, height=400)

    with tab_report:
        st.subheader("Download outputs")
        for key, label in (
            ("pdf_report", "📄 PDF report"),
            ("timeline_csv", "📊 Timeline CSV"),
            ("result_json", "🧾 Result JSON"),
            ("annotated_video", "🎬 Annotated video"),
        ):
            path = result.output_paths.get(key)
            if path and Path(path).exists():
                with open(path, "rb") as f:
                    st.download_button(label, f.read(), file_name=Path(path).name, use_container_width=True)


if __name__ == "__main__":
    main()
