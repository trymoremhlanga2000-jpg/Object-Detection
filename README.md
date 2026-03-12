# 🎯 VISION — Real-Time Object Detection

A production-grade web app for real-time object detection with deep analytics.

**Stack:** Streamlit · YOLOv8 · streamlit-webrtc · Plotly · OpenCV

---

## Features

- 📹 **Live webcam feed** with bounding boxes, corner accents, confidence labels
- 📊 **Deep analytics**: top classes, confidence distribution, detection timeline
- 📋 **Detection log** with CSV/JSON export
- ⚙️ **Configurable**: confidence threshold, model variant
- 🎨 **Military-grade dark UI**

---

## Local Setup

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/vision-detection.git
cd vision-detection

# 2. Install
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Deploy to Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Click **Deploy** — done!

---

## Model Variants

| Model | Speed | Accuracy | Size |
|-------|-------|----------|------|
| `yolov8n.pt` | ⚡⚡⚡ Fastest | Good   | 6MB  |
| `yolov8s.pt` | ⚡⚡    Balanced | Better | 22MB |
| `yolov8m.pt` | ⚡      Slowest  | Best   | 52MB |

Default is `yolov8n` (nano) — best for real-time on CPU.

---

## Troubleshooting

**Camera not working?**
→ Allow camera permission in your browser popup

**WebRTC won't connect on deployed app?**
→ Try a different browser (Chrome recommended)
→ Check your firewall / VPN settings

**Low FPS?**
→ Close other browser tabs
→ Use `yolov8n.pt` (default, fastest)
