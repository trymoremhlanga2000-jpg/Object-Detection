"""
app.py — VISION Real-Time Object Detection Dashboard
Stable version for Python 3.14+ and Streamlit 1.35+
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import cv2
from PIL import Image
from streamlit_autorefresh import st_autorefresh

from detector import ObjectDetector, EMOJI_MAP


# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────

st.set_page_config(
    page_title="VISION • Object Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ─────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────

if "last_dets" not in st.session_state:
    st.session_state["last_dets"] = []

if "last_frame_id" not in st.session_state:
    st.session_state["last_frame_id"] = None


# ─────────────────────────────────────────────────────────
# LOAD DETECTOR (Singleton)
# ─────────────────────────────────────────────────────────

@st.cache_resource
def load_detector():
    return ObjectDetector(
        model_name="yolov8n.pt",
        conf_threshold=0.45
    )

detector = load_detector()


# ─────────────────────────────────────────────────────────
# AUTO REFRESH
# ─────────────────────────────────────────────────────────

st_autorefresh(interval=800, key="vision_refresh")


# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────

with st.sidebar:

    st.markdown("## ⚙️ SETTINGS")

    conf_val = st.slider(
        "Confidence Threshold",
        0.10,
        0.90,
        0.45,
        0.05
    )

    detector.set_confidence(conf_val)

    st.markdown("---")

    st.markdown("## 📸 CAMERA MODE")
    st.markdown(
        """
        1. Allow camera access  
        2. Point camera at objects  
        3. Click **Take Photo**  
        4. Watch detections
        """
    )

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        if st.button("🔄 RESET"):
            detector.reset()
            st.session_state["last_dets"] = []
            st.success("Statistics cleared")

    with c2:
        stats_sidebar = detector.get_stats()
        st.metric("Frames", stats_sidebar["frames"])


# ─────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────

st.title("🎯 VISION — Real-Time Object Detection")


# ─────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────

tab_live, tab_analytics, tab_log = st.tabs(
    [
        "📹 LIVE FEED",
        "📊 ANALYTICS",
        "📋 DETECTION LOG"
    ]
)


# =========================================================
# TAB 1 — LIVE FEED
# =========================================================

with tab_live:

    col_cam, col_panel = st.columns([3, 1])

    with col_cam:

        st.subheader("Camera")

        camera_image = st.camera_input(
            "Capture",
            label_visibility="collapsed"
        )

        if camera_image is not None and camera_image.id != st.session_state["last_frame_id"]:

            st.session_state["last_frame_id"] = camera_image.id

            try:

                pil_img = Image.open(camera_image).convert("RGB")
                frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

                annotated, dets = detector.detect_and_draw(frame)

                annotated_rgb = cv2.cvtColor(
                    annotated,
                    cv2.COLOR_BGR2RGB
                )

                st.image(
                    annotated_rgb,
                    use_container_width=True,
                    caption=f"Frame {detector.frame_count} | {len(dets)} objects"
                )

                st.session_state["last_dets"] = dets

            except Exception as e:

                st.error(f"Detection error: {e}")

        elif camera_image is None:

            st.info("Click **Take Photo** to start detection")


    # ─────────────────────────────────────────
    # RIGHT METRICS PANEL
    # ─────────────────────────────────────────

    with col_panel:

        st.subheader("Live Metrics")

        stats = detector.get_stats()
        last_dets = st.session_state["last_dets"]

        st.metric("In Frame", len(last_dets))
        st.metric("Total Detected", stats["total"])
        st.metric("Unique Classes", stats["unique_classes"])
        st.metric("Frames", stats["frames"])

        st.markdown("---")
        st.markdown("**Detections**")

        if last_dets:

            for det in sorted(last_dets, key=lambda x: -x["confidence"]):

                emoji = det.get("emoji", "📦")
                cls = det["class"].title()
                pct = int(det["confidence"] * 100)

                st.markdown(
                    f"{emoji} **{cls}** — {pct}%"
                )

        else:
            st.caption("No detections yet")


# =========================================================
# TAB 2 — ANALYTICS
# =========================================================

with tab_analytics:

    stats = detector.get_stats()

    if not stats["class_counts"]:

        st.info("Capture frames first to generate analytics")

    else:

        k1, k2, k3, k4 = st.columns(4)

        k1.metric("Total Detections", stats["total"])
        k2.metric("Frames", stats["frames"])
        k3.metric("Classes", stats["unique_classes"])
        k4.metric("Avg / Frame", stats["det_per_frame"])

        st.markdown("---")

        df = pd.DataFrame(
            list(stats["class_counts"].items()),
            columns=["Class", "Count"]
        ).sort_values("Count")

        fig = px.bar(
            df,
            x="Count",
            y="Class",
            orientation="h",
            title="Top Detected Classes"
        )

        st.plotly_chart(fig, use_container_width=True)


        # Timeline

        if stats["timeline"]:

            tl = pd.DataFrame(stats["timeline"])

            fig2 = go.Figure()

            fig2.add_trace(
                go.Scatter(
                    x=tl["frame"],
                    y=tl["count"],
                    mode="lines+markers",
                    name="Objects"
                )
            )

            fig2.update_layout(
                title="Objects Detected Over Time",
                xaxis_title="Frame",
                yaxis_title="Count"
            )

            st.plotly_chart(fig2, use_container_width=True)


# =========================================================
# TAB 3 — DETECTION LOG
# =========================================================

with tab_log:

    stats = detector.get_stats()

    if not stats["class_counts"]:

        st.info("No detection data yet")

    else:

        rows = []
        total = max(stats["total"], 1)

        for cls, cnt in sorted(
            stats["class_counts"].items(),
            key=lambda x: -x[1]
        ):

            avg_c = stats["conf_avg"].get(cls, 0)

            rows.append(
                {
                    "Icon": EMOJI_MAP.get(cls, "📦"),
                    "Class": cls.title(),
                    "Count": cnt,
                    "Share (%)": round(cnt / total * 100, 1),
                    "Avg Confidence (%)": avg_c
                }
            )

        df_log = pd.DataFrame(rows)

        st.dataframe(
            df_log,
            use_container_width=True,
            hide_index=True
        )

        c1, c2 = st.columns(2)

        with c1:

            st.download_button(
                "⬇️ Export CSV",
                df_log.to_csv(index=False),
                "vision_detections.csv",
                "text/csv"
            )

        with c2:

            st.download_button(
                "⬇️ Export JSON",
                df_log.to_json(
                    orient="records",
                    indent=2
                ),
                "vision_detections.json",
                "application/json"
            )


        if stats["timeline"]:

            st.markdown("---")

            tl2 = pd.DataFrame(stats["timeline"])

            tl2.columns = ["Frame", "Objects", "Time"]

            st.dataframe(
                tl2[::-1].reset_index(drop=True),
                use_container_width=True,
                hide_index=True
            )
