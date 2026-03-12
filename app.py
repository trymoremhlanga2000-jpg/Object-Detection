"""
app.py - VISION: Real-Time Object Detection Dashboard
Uses st.camera_input() - NO WebRTC, NO system packages, works on Python 3.14+
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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VISION • Object Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Singleton detector (persists across reruns) ───────────────────────────────
@st.cache_resource
def load_detector():
    return ObjectDetector(model_name="yolov8n.pt", conf_threshold=0.45)

detector = load_detector()

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;500;700&display=swap');

[data-testid="stAppViewContainer"] { background:#07090c !important; }
[data-testid="stSidebar"]          { background:#0c1014 !important; border-right:1px solid #162016; }
section[data-testid="stSidebar"] * { color:#8aaa8a !important; }

h1,h2,h3,h4 {
    font-family:'Share Tech Mono',monospace !important;
    color:#00ff88 !important; letter-spacing:3px;
}
p, label, div { color:#8aaa8a; }

[data-testid="metric-container"] {
    background:#0c1014; border:1px solid #162016;
    border-radius:3px; padding:14px 16px !important;
}
[data-testid="stMetricValue"]  { font-family:'Rajdhani',sans-serif !important;
    font-weight:700 !important; font-size:2rem !important; color:#00ff88 !important; }
[data-testid="stMetricLabel"]  { font-family:'Share Tech Mono',monospace !important;
    font-size:.6rem !important; letter-spacing:2px !important; color:#3a5a3a !important; }

.stTabs [data-baseweb="tab-list"]  { background:#0c1014; border-bottom:1px solid #162016; gap:4px; }
.stTabs [data-baseweb="tab"]       { font-family:'Share Tech Mono',monospace; font-size:.75rem;
    letter-spacing:2px; color:#3a5a3a !important; padding:10px 20px;
    border:none; background:transparent; }
.stTabs [aria-selected="true"]     { color:#00ff88 !important; border-bottom:2px solid #00ff88 !important; }

div.stButton > button {
    font-family:'Share Tech Mono',monospace; font-size:.65rem;
    letter-spacing:2px; background:transparent; color:#00ff88;
    border:1px solid #00ff88; border-radius:2px; padding:8px 18px; transition:all .2s;
}
div.stButton > button:hover { background:#00ff8822; box-shadow:0 0 10px #00ff8844; }

[data-testid="stImage"] img { border:1px solid #162016; border-radius:2px; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:#0c1014;border-bottom:1px solid #162016;
  padding:12px 24px;margin:-2rem -4rem 1.5rem -4rem;
  display:flex;align-items:center;justify-content:space-between">
  <div>
    <span style="font-family:'Share Tech Mono',monospace;font-weight:700;font-size:1.5rem;
                 color:#00ff88;letter-spacing:6px;text-shadow:0 0 20px #00ff8866">VISION</span>
    <span style="font-family:'Share Tech Mono',monospace;color:#3a5a3a;font-size:.8rem;
                 margin-left:12px">// REAL-TIME OBJECT DETECTION</span>
  </div>
  <div style="font-family:'Share Tech Mono',monospace;font-size:.6rem;color:#3a5a3a;letter-spacing:2px">
    YOLOV8 &nbsp;|&nbsp; 80 CLASSES &nbsp;|&nbsp; COCO DATASET
  </div>
</div>
""", unsafe_allow_html=True)

# ── Auto-refresh ──────────────────────────────────────────────────────────────
st_autorefresh(interval=800, key="live_refresh")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ SETTINGS")
    st.markdown("---")

    conf_val = st.slider("Confidence Threshold", 0.10, 0.90, 0.45, 0.05,
        help="Lower = more detections (more noise). Higher = fewer but surer.")
    detector.set_confidence(conf_val)

    st.markdown("---")
    st.markdown("## 📸 CAMERA MODE")
    st.markdown("""
    <div style='font-family:"Share Tech Mono",monospace;font-size:.65rem;
                color:#3a5a3a;line-height:2'>
    1. Allow camera in browser<br>
    2. Point at objects<br>
    3. Watch detections live<br>
    4. Check Analytics tab for stats
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 RESET"):
            detector.reset()
            st.success("Stats cleared!")
    with col2:
        stats_s = detector.get_stats()
        st.metric("Frames", stats_s['frames'])

    st.markdown("---")
    st.markdown("""
    <div style='font-family:"Share Tech Mono",monospace;font-size:.6rem;color:#3a5a3a;line-height:2'>
    Model &nbsp;: YOLOv8 Nano<br>
    Backend : PyTorch CPU<br>
    Classes : 80 COCO<br>
    Python &nbsp;: 3.14 ✅
    </div>""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_live, tab_analytics, tab_log = st.tabs([
    "  📹  LIVE FEED  ",
    "  📊  ANALYTICS  ",
    "  📋  DETECTION LOG  ",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE FEED
# ═════════════════════════════════════════════════════════════════════════════
with tab_live:
    col_cam, col_panel = st.columns([3, 1], gap="medium")

    with col_cam:
        st.markdown("#### 📷 CAMERA — Point at objects and detections appear instantly")
        camera_image = st.camera_input(
            label="camera",
            label_visibility="collapsed",
        )

        if camera_image is not None:
            # Convert uploaded image bytes → numpy BGR
            pil_img  = Image.open(camera_image).convert("RGB")
            frame    = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            # Run detection
            annotated, dets = detector.detect_and_draw(frame)

            # Convert back to RGB for display
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, use_column_width=True,
                     caption=f"Frame #{detector.frame_count} | {len(dets)} objects detected")

            # Store last dets in session for sidebar
            st.session_state['last_dets'] = dets
        else:
            st.markdown("""
            <div style="border:1px solid #162016;background:#0c1014;padding:40px;
                        text-align:center;font-family:'Share Tech Mono',monospace;
                        font-size:.75rem;color:#3a5a3a;letter-spacing:2px">
                📷 CLICK "TAKE PHOTO" ABOVE TO START<br><br>
                <span style="font-size:.6rem">Allow camera access when browser asks</span>
            </div>""", unsafe_allow_html=True)
            st.session_state.setdefault('last_dets', [])

    # ── Right panel ──────────────────────────────────────────────────────────
    with col_panel:
        st.markdown("#### 📡 LIVE METRICS")
        stats = detector.get_stats()
        last_dets = st.session_state.get('last_dets', [])

        st.metric("In This Frame",    len(last_dets))
        st.metric("Total Detected",   stats['total'])
        st.metric("Unique Classes",   stats['unique_classes'])
        st.metric("Frames Processed", stats['frames'])

        st.markdown("---")
        st.markdown("**Detections:**")

        if last_dets:
            for det in sorted(last_dets, key=lambda x: -x['confidence']):
                em  = det.get('emoji','📦')
                cls = det['class'].title()
                pct = int(det['confidence'] * 100)
                st.markdown(
                    f"""<div style="margin:4px 0;padding:6px 8px;background:#0c1014;
                        border:1px solid #162016;
                        font-family:'Share Tech Mono',monospace;font-size:.7rem">
                        {em} <b style="color:#c8d8c8">{cls}</b>
                        <span style="float:right;color:#00ff88">{pct}%</span>
                        <div style="margin-top:4px;height:2px;background:#162016">
                          <div style="width:{pct}%;height:100%;background:#00ff88"></div>
                        </div></div>""",
                    unsafe_allow_html=True)
        else:
            st.markdown(
                "<div style='color:#3a5a3a;font-family:monospace;font-size:.7rem;padding:8px'>"
                "NO DETECTIONS YET</div>", unsafe_allow_html=True)

    # ── Tips ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.info("📸 **Tip:** Each photo = one detection pass. Take multiple photos to build up stats.")
    c2.info("💡 **Tip:** Good lighting = better accuracy. Try holding objects closer to camera.")
    c3.info("⚡ **Tip:** Use the Analytics tab after several frames to see trends & statistics.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════
with tab_analytics:
    stats = detector.get_stats()

    if not stats['class_counts']:
        st.markdown("""
        <div style="text-align:center;padding:60px;font-family:'Share Tech Mono',monospace;color:#3a5a3a">
            <div style="font-size:3rem">📊</div>
            <div style="margin-top:16px;letter-spacing:3px">CAPTURE SOME FRAMES TO SEE ANALYTICS</div>
            <div style="margin-top:8px;font-size:.7rem">Go to LIVE FEED → take photos → come back here</div>
        </div>""", unsafe_allow_html=True)
    else:
        THEME = dict(
            paper_bgcolor='#0c1014', plot_bgcolor='#07090c',
            font=dict(color='#8aaa8a', family='Share Tech Mono'),
            title_font=dict(color='#00ff88', size=13),
            margin=dict(l=10,r=10,t=40,b=10),
        )

        # KPIs
        k1,k2,k3,k4,k5 = st.columns(5)
        k1.metric("Total Detections",     stats['total'])
        k2.metric("Frames Processed",     stats['frames'])
        k3.metric("Unique Classes",        stats['unique_classes'])
        k4.metric("Avg Dets / Frame",     stats['det_per_frame'])
        m,s = stats['elapsed_sec']//60, stats['elapsed_sec']%60
        k5.metric("Session Time",          f"{m:02d}:{s:02d}")

        st.markdown("---")

        # Row 1: bar charts
        ca, cb = st.columns(2, gap="medium")
        with ca:
            df = (pd.DataFrame(list(stats['class_counts'].items()), columns=['Class','Count'])
                  .sort_values('Count').tail(12))
            fig = px.bar(df, x='Count', y='Class', orientation='h',
                         title='▸ Top Detected Classes',
                         color='Count',
                         color_continuous_scale=[[0,'#162016'],[.5,'#00cc6a'],[1,'#00ff88']])
            fig.update_layout(**THEME, showlegend=False, coloraxis_showscale=False)
            fig.update_xaxes(gridcolor='#162016',color='#3a5a3a',zeroline=False)
            fig.update_yaxes(gridcolor='#162016',color='#c8d8c8')
            st.plotly_chart(fig, use_container_width=True)

        with cb:
            df2 = pd.DataFrame([{'Class':k,'Avg Confidence (%)':v}
                                 for k,v in stats['conf_avg'].items()]
                               ).sort_values('Avg Confidence (%)')
            fig2 = px.bar(df2, x='Avg Confidence (%)', y='Class', orientation='h',
                          title='▸ Avg Confidence by Class',
                          color='Avg Confidence (%)',
                          color_continuous_scale=[[0,'#162016'],[.5,'#00aaff'],[1,'#00ccff']],
                          range_x=[0,100])
            fig2.update_layout(**THEME, showlegend=False, coloraxis_showscale=False)
            fig2.update_xaxes(gridcolor='#162016',color='#3a5a3a')
            fig2.update_yaxes(gridcolor='#162016',color='#c8d8c8')
            st.plotly_chart(fig2, use_container_width=True)

        # Timeline
        if stats['timeline']:
            tl = pd.DataFrame(stats['timeline'])
            tl['smooth'] = tl['count'].rolling(5, min_periods=1).mean()
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=tl['frame'],y=tl['count'],mode='lines',
                name='Raw',line=dict(color='#162016',width=1),
                fill='tozeroy',fillcolor='#00ff8808'))
            fig3.add_trace(go.Scatter(x=tl['frame'],y=tl['smooth'],mode='lines',
                name='Smoothed (5f avg)',line=dict(color='#00ff88',width=2)))
            fig3.update_layout(**THEME, title='▸ Objects Detected Over Time',
                showlegend=True, legend=dict(font=dict(color='#8aaa8a'),bgcolor='rgba(0,0,0,0)'),
                xaxis_title='Frame', yaxis_title='Object Count')
            fig3.update_xaxes(gridcolor='#162016',color='#3a5a3a',zeroline=False)
            fig3.update_yaxes(gridcolor='#162016',color='#3a5a3a',zeroline=False)
            st.plotly_chart(fig3, use_container_width=True)

        # Pie + Histogram
        cc2, cd = st.columns(2, gap="medium")
        with cc2:
            pdf = pd.DataFrame(list(stats['class_counts'].items()),columns=['Class','Count'])
            fig4 = px.pie(pdf,values='Count',names='Class',
                          title='▸ Detection Distribution',hole=0.45,
                          color_discrete_sequence=px.colors.sequential.Greens_r)
            fig4.update_layout(**THEME)
            fig4.update_traces(textfont_color='#c8d8c8')
            st.plotly_chart(fig4, use_container_width=True)

        with cd:
            all_c,all_l=[],[]
            for cls,vals in stats['conf_history'].items():
                all_c.extend([v*100 for v in vals]); all_l.extend([cls]*len(vals))
            if all_c:
                hdf = pd.DataFrame({'Confidence (%)':all_c,'Class':all_l})
                fig5 = px.histogram(hdf, x='Confidence (%)', nbins=25,
                    title='▸ Confidence Score Distribution',
                    color_discrete_sequence=['#00ff88'])
                fig5.update_layout(**THEME,showlegend=False,bargap=0.05,yaxis_title='Count')
                fig5.update_xaxes(gridcolor='#162016',color='#3a5a3a',range=[0,100])
                fig5.update_yaxes(gridcolor='#162016',color='#3a5a3a')
                st.plotly_chart(fig5, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — DETECTION LOG
# ═════════════════════════════════════════════════════════════════════════════
with tab_log:
    stats = detector.get_stats()
    if not stats['class_counts']:
        st.info("No data yet. Capture some frames on the Live Feed tab first.")
    else:
        st.markdown("#### 📋 CLASS SUMMARY TABLE")
        rows=[]
        total=max(stats['total'],1)
        for cls,cnt in sorted(stats['class_counts'].items(),key=lambda x:-x[1]):
            avg_c=stats['conf_avg'].get(cls,0)
            ch=stats['conf_history'].get(cls,[0])
            rows.append({
                'Icon':         EMOJI_MAP.get(cls,'📦'),
                'Class':        cls.title(),
                'Total Count':  cnt,
                'Share (%)':    round(cnt/total*100,1),
                'Avg Conf':     f"{avg_c:.1f}%",
                'Min Conf':     f"{min(ch)*100:.1f}%",
                'Max Conf':     f"{max(ch)*100:.1f}%",
            })
        df_log=pd.DataFrame(rows)
        st.dataframe(df_log, use_container_width=True, hide_index=True)

        c1,c2,_=st.columns([1,1,3])
        with c1:
            st.download_button("⬇️ EXPORT CSV",  df_log.to_csv(index=False),
                               "vision_detections.csv","text/csv")
        with c2:
            st.download_button("⬇️ EXPORT JSON", df_log.to_json(orient='records',indent=2),
                               "vision_detections.json","application/json")

        if stats['timeline']:
            st.markdown("---")
            st.markdown("#### 📈 FRAME TIMELINE")
            tl2=pd.DataFrame(stats['timeline'])
            tl2.columns=['Frame #','Objects','Timestamp']
            st.dataframe(tl2[::-1].reset_index(drop=True),
                         use_container_width=True,hide_index=True,height=280)
