"""
app.py — VISION: Real-Time Object Detection Dashboard
Streamlit + streamlit-webrtc + YOLOv8
"""

import streamlit as st
import av
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase, WebRtcMode
from streamlit_autorefresh import st_autorefresh
from detector import ObjectDetector, EMOJI_MAP

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VISION • Object Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global detector (shared across WebRTC thread + Streamlit) ───────────────
# Using module-level singleton so the VideoProcessor thread can always reach it
if "detector" not in st.session_state:
    st.session_state.detector = ObjectDetector(model_name="yolov8n.pt", conf_threshold=0.45)

_DETECTOR: ObjectDetector = st.session_state.detector

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;500;700&display=swap');

/* App background */
[data-testid="stAppViewContainer"] { background: #07090c !important; }
[data-testid="stSidebar"]          { background: #0c1014 !important; border-right: 1px solid #162016; }
[data-testid="stSidebar"] *        { color: #8aaa8a !important; }

/* Headers */
h1, h2, h3, h4 {
    font-family: 'Share Tech Mono', monospace !important;
    color: #00ff88 !important;
    letter-spacing: 3px;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #0c1014;
    border: 1px solid #162016;
    border-radius: 3px;
    padding: 14px 16px !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important;
    font-size: 2rem !important;
    color: #00ff88 !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.6rem !important;
    letter-spacing: 2px !important;
    color: #3a5a3a !important;
}
[data-testid="stMetricDelta"] { color: #00cc6a !important; font-size: 0.75rem !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"]  { background: #0c1014; border-bottom: 1px solid #162016; gap: 4px; }
.stTabs [data-baseweb="tab"]       { font-family: 'Share Tech Mono', monospace; font-size: 0.75rem;
                                     letter-spacing: 2px; color: #3a5a3a !important;
                                     padding: 10px 20px; border: none; background: transparent; }
.stTabs [aria-selected="true"]     { color: #00ff88 !important; border-bottom: 2px solid #00ff88 !important; }

/* Buttons */
div.stButton > button {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 2px;
    background: transparent;
    color: #00ff88;
    border: 1px solid #00ff88;
    border-radius: 2px;
    padding: 8px 18px;
    transition: all .2s;
}
div.stButton > button:hover { background: #00ff8822; box-shadow: 0 0 10px #00ff8844; }

/* Slider */
[data-testid="stSlider"] { padding: 0 4px; }
.stSlider > div > div > div > div { background: #00ff88 !important; }

/* Select boxes */
[data-testid="stSelectbox"] > div > div {
    background: #0c1014;
    border: 1px solid #162016;
    color: #8aaa8a;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #162016;
    background: #0c1014;
}
.dvn-scroller { background: #0c1014 !important; }

/* Info / warning boxes */
[data-testid="stInfoMessage"]    { background: #0c1014 !important; border-color: #00ff8844 !important; color: #8aaa8a !important; }
[data-testid="stSuccessMessage"] { background: #0c1014 !important; border-color: #00ff88 !important; }

/* Scrollbar */
::-webkit-scrollbar       { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #162016; border-radius: 2px; }

/* Video element */
video { border: 1px solid #162016; }
</style>
""", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
  background:#0c1014;
  border-bottom:1px solid #162016;
  padding:12px 24px;
  margin:-2rem -4rem 1.5rem -4rem;
  display:flex;
  align-items:center;
  justify-content:space-between;
">
  <div>
    <span style="font-family:'Share Tech Mono',monospace;font-weight:700;font-size:1.5rem;
                 color:#00ff88;letter-spacing:6px;text-shadow:0 0 20px #00ff8866">VISION</span>
    <span style="font-family:'Share Tech Mono',monospace;color:#3a5a3a;font-size:0.8rem;
                 margin-left:12px">// REAL-TIME OBJECT DETECTION</span>
  </div>
  <div style="font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:#3a5a3a;letter-spacing:2px">
    YOLOV8 &nbsp;|&nbsp; 80 CLASSES &nbsp;|&nbsp; COCO DATASET
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Auto-refresh every 1.5s so stats update live ────────────────────────────
st_autorefresh(interval=1500, key="live_refresh")

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ SETTINGS")
    st.markdown("---")

    # Confidence threshold
    conf_val = st.slider(
        "Confidence Threshold",
        min_value=0.10, max_value=0.90, value=0.45, step=0.05,
        help="Minimum detection confidence (lower = more detections, more noise)",
    )
    _DETECTOR.set_confidence(conf_val)

    # Model selector
    model_choice = st.selectbox(
        "Model Variant",
        options=["yolov8n.pt — Nano (fastest)", "yolov8s.pt — Small", "yolov8m.pt — Medium"],
        index=0,
        help="Nano runs at 25-40 FPS. Medium is 2× more accurate but slower.",
    )

    st.markdown("---")
    st.markdown("## 🔧 CONTROLS")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 RESET"):
            _DETECTOR.reset()
            st.success("Stats cleared!")
    with col2:
        stats_sidebar = _DETECTOR.get_stats()
        st.metric("FPS", stats_sidebar['fps'])

    st.markdown("---")
    st.markdown("## ℹ️ ABOUT")
    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:#3a5a3a;line-height:2">
    Model &nbsp;&nbsp;&nbsp;: YOLOv8 Nano<br>
    Backend &nbsp;: PyTorch (CPU)<br>
    Classes &nbsp;: 80 (COCO)<br>
    Latency &nbsp;: ~30ms/frame<br>
    </div>
    """, unsafe_allow_html=True)

# ─── WebRTC Config (STUN servers) ────────────────────────────────────────────
RTC_CONFIG = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
    ]
})

# ─── Video Processor (runs in WebRTC thread) ─────────────────────────────────
class VideoProcessor(VideoProcessorBase):
    """Passes each video frame through the YOLOv8 detector."""
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        # Flip so it acts as a mirror
        img = cv2.flip(img, 1)
        img, _ = _DETECTOR.detect_and_draw(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_live, tab_analytics, tab_log = st.tabs([
    "  📹  LIVE FEED  ",
    "  📊  ANALYTICS  ",
    "  📋  DETECTION LOG  ",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE FEED
# ═══════════════════════════════════════════════════════════════════════════════
with tab_live:
    col_video, col_live = st.columns([3, 1], gap="medium")

    with col_video:
        st.markdown("#### 🎥 CAMERA FEED")
        webrtc_ctx = webrtc_streamer(
            key="vision-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIG,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if not webrtc_ctx.state.playing:
            st.info(
                "👆 Click **START** above → allow camera in browser popup → detection begins automatically."
            )

    with col_live:
        st.markdown("#### 📡 LIVE METRICS")
        stats = _DETECTOR.get_stats()

        st.metric("Objects in Frame",  stats['current_count'])
        st.metric("Total Detected",    stats['total'])
        st.metric("Unique Classes",    stats['unique_classes'])
        st.metric("Frames Processed",  stats['frames'])

        st.markdown("---")
        st.markdown("**Current Frame:**")

        if stats['current_dets']:
            for det in sorted(stats['current_dets'], key=lambda x: -x['confidence']):
                em   = det.get('emoji', '📦')
                cls  = det['class'].title()
                pct  = int(det['confidence'] * 100)
                bar_w = int(pct * 1.4)  # scale to ~140px max
                st.markdown(
                    f"""<div style="margin:4px 0;padding:6px 8px;
                        background:#0c1014;border:1px solid #162016;
                        font-family:'Share Tech Mono',monospace;font-size:.7rem">
                        {em} <b style="color:#c8d8c8">{cls}</b>
                        <span style="float:right;color:#00ff88">{pct}%</span>
                        <div style="margin-top:4px;height:2px;background:#162016">
                          <div style="width:{pct}%;height:100%;background:#00ff88;transition:width .3s"></div>
                        </div>
                    </div>""",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                "<div style='color:#3a5a3a;font-family:monospace;font-size:.7rem;padding:8px'>NO OBJECTS DETECTED</div>",
                unsafe_allow_html=True,
            )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_analytics:
    stats = _DETECTOR.get_stats()

    if not stats['class_counts']:
        st.markdown("""
        <div style="text-align:center;padding:60px;font-family:'Share Tech Mono',monospace;color:#3a5a3a">
            <div style="font-size:3rem">🎯</div>
            <div style="margin-top:16px;letter-spacing:3px">START THE CAMERA TO SEE ANALYTICS</div>
            <div style="margin-top:8px;font-size:.7rem">Go to LIVE FEED tab and press START</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── Row 1: KPI metrics ──────────────────────────────────────────────
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Detections",  stats['total'])
        k2.metric("Frames Processed",  stats['frames'])
        k3.metric("Unique Classes",     stats['unique_classes'])
        k4.metric("Avg Detections/Frame", stats['det_per_frame'])
        mins = stats['elapsed_sec'] // 60
        secs = stats['elapsed_sec'] % 60
        k5.metric("Session Duration",   f"{mins:02d}:{secs:02d}")

        st.markdown("---")

        # ── Row 2: Bar charts ───────────────────────────────────────────────
        col_a, col_b = st.columns(2, gap="medium")

        _CHART_THEME = dict(
            paper_bgcolor='#0c1014', plot_bgcolor='#07090c',
            font=dict(color='#8aaa8a', family='Share Tech Mono'),
            title_font=dict(color='#00ff88', size=13),
            margin=dict(l=10, r=10, t=40, b=10),
        )

        with col_a:
            cls_df = (
                pd.DataFrame(list(stats['class_counts'].items()), columns=['Class', 'Count'])
                .sort_values('Count', ascending=True)
                .tail(12)
            )
            fig = px.bar(
                cls_df, x='Count', y='Class', orientation='h',
                title='▸ Top Detected Classes',
                color='Count',
                color_continuous_scale=[[0,'#162016'],[0.5,'#00cc6a'],[1,'#00ff88']],
            )
            fig.update_layout(**_CHART_THEME, showlegend=False, coloraxis_showscale=False)
            fig.update_xaxes(gridcolor='#162016', color='#3a5a3a', zeroline=False)
            fig.update_yaxes(gridcolor='#162016', color='#c8d8c8')
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            conf_df = pd.DataFrame([
                {'Class': k, 'Avg Confidence (%)': v}
                for k, v in stats['conf_avg'].items()
            ]).sort_values('Avg Confidence (%)', ascending=True)

            fig2 = px.bar(
                conf_df, x='Avg Confidence (%)', y='Class', orientation='h',
                title='▸ Avg Confidence by Class',
                color='Avg Confidence (%)',
                color_continuous_scale=[[0,'#162016'],[0.5,'#00aaff'],[1,'#00ccff']],
                range_x=[0, 100],
            )
            fig2.update_layout(**_CHART_THEME, showlegend=False, coloraxis_showscale=False)
            fig2.update_xaxes(gridcolor='#162016', color='#3a5a3a')
            fig2.update_yaxes(gridcolor='#162016', color='#c8d8c8')
            st.plotly_chart(fig2, use_container_width=True)

        # ── Row 3: Timeline ─────────────────────────────────────────────────
        if stats['timeline']:
            tl_df = pd.DataFrame(stats['timeline'])
            # Rolling average for smoothness
            tl_df['smooth'] = tl_df['count'].rolling(5, min_periods=1).mean()

            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=tl_df['frame'], y=tl_df['count'],
                mode='lines', name='Raw',
                line=dict(color='#162016', width=1),
                fill='tozeroy', fillcolor='#00ff8808',
            ))
            fig3.add_trace(go.Scatter(
                x=tl_df['frame'], y=tl_df['smooth'],
                mode='lines', name='Smoothed',
                line=dict(color='#00ff88', width=2),
            ))
            fig3.update_layout(
                **_CHART_THEME,
                title='▸ Objects Detected Over Time',
                showlegend=True,
                legend=dict(font=dict(color='#8aaa8a'), bgcolor='rgba(0,0,0,0)'),
                xaxis_title='Frame', yaxis_title='Count',
            )
            fig3.update_xaxes(gridcolor='#162016', color='#3a5a3a', zeroline=False)
            fig3.update_yaxes(gridcolor='#162016', color='#3a5a3a', zeroline=False)
            st.plotly_chart(fig3, use_container_width=True)

        # ── Row 4: Pie + Histogram ──────────────────────────────────────────
        col_c, col_d = st.columns(2, gap="medium")

        with col_c:
            pie_df = pd.DataFrame(list(stats['class_counts'].items()), columns=['Class', 'Count'])
            fig4 = px.pie(
                pie_df, values='Count', names='Class',
                title='▸ Detection Distribution',
                hole=0.45,
                color_discrete_sequence=px.colors.sequential.Greens_r,
            )
            fig4.update_layout(**_CHART_THEME)
            fig4.update_traces(textfont_color='#c8d8c8')
            st.plotly_chart(fig4, use_container_width=True)

        with col_d:
            all_confs, all_cls = [], []
            for cls, vals in stats['conf_history'].items():
                all_confs.extend([v * 100 for v in vals])
                all_cls.extend([cls] * len(vals))

            if all_confs:
                hist_df = pd.DataFrame({'Confidence (%)': all_confs, 'Class': all_cls})
                fig5 = px.histogram(
                    hist_df, x='Confidence (%)', nbins=25,
                    title='▸ Confidence Score Distribution',
                    color_discrete_sequence=['#00ff88'],
                )
                fig5.update_layout(**_CHART_THEME, showlegend=False,
                                   bargap=0.05, yaxis_title='Count')
                fig5.update_xaxes(gridcolor='#162016', color='#3a5a3a', range=[0, 100])
                fig5.update_yaxes(gridcolor='#162016', color='#3a5a3a')
                st.plotly_chart(fig5, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DETECTION LOG
# ═══════════════════════════════════════════════════════════════════════════════
with tab_log:
    stats = _DETECTOR.get_stats()

    if not stats['class_counts']:
        st.info("No detection data yet. Start the camera on the Live Feed tab.")
    else:
        st.markdown("#### 📋 CLASS SUMMARY TABLE")

        rows = []
        total = max(stats['total'], 1)
        for cls, cnt in sorted(stats['class_counts'].items(), key=lambda x: -x[1]):
            avg_c = stats['conf_avg'].get(cls, 0)
            em    = EMOJI_MAP.get(cls, '📦')
            rows.append({
                'Icon':          em,
                'Class':         cls.title(),
                'Total Count':   cnt,
                'Share (%)':     round(cnt / total * 100, 1),
                'Avg Confidence': f"{avg_c:.1f}%",
                'Min Conf':      f"{min(stats['conf_history'].get(cls,[0]))*100:.1f}%",
                'Max Conf':      f"{max(stats['conf_history'].get(cls,[0]))*100:.1f}%",
            })

        log_df = pd.DataFrame(rows)
        st.dataframe(log_df, use_container_width=True, hide_index=True)

        # ── Download ────────────────────────────────────────────────────────
        col_dl1, col_dl2, _ = st.columns([1, 1, 3])
        with col_dl1:
            csv_data = log_df.to_csv(index=False)
            st.download_button(
                "⬇️ EXPORT CSV",
                data=csv_data,
                file_name="vision_detections.csv",
                mime="text/csv",
            )
        with col_dl2:
            json_data = log_df.to_json(orient='records', indent=2)
            st.download_button(
                "⬇️ EXPORT JSON",
                data=json_data,
                file_name="vision_detections.json",
                mime="application/json",
            )

        # ── Timeline table ──────────────────────────────────────────────────
        if stats['timeline']:
            st.markdown("---")
            st.markdown("#### 📈 FRAME TIMELINE (last 150 frames)")
            tl_df = pd.DataFrame(stats['timeline'])
            tl_df.columns = ['Frame #', 'Objects', 'Timestamp']
            st.dataframe(tl_df[::-1].reset_index(drop=True),  # newest first
                         use_container_width=True, hide_index=True, height=280)
