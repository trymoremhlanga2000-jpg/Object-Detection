"""
detector.py — YOLOv8 Object Detection Engine
Thread-safe detection + deep statistics collection
"""

import cv2
import numpy as np
import threading
import time
import colorsys
from collections import defaultdict, deque
from datetime import datetime


# ── Emoji map for 80 COCO classes ─────────────────────────────────────────────
EMOJI_MAP = {
    'person': '🧍', 'car': '🚗', 'bicycle': '🚲', 'motorcycle': '🏍',
    'airplane': '✈️', 'bus': '🚌', 'train': '🚂', 'truck': '🚛', 'boat': '⛵',
    'traffic light': '🚦', 'fire hydrant': '🚒', 'stop sign': '🛑',
    'parking meter': '🅿️', 'bench': '🪑', 'bird': '🐦', 'cat': '🐱',
    'dog': '🐶', 'horse': '🐴', 'sheep': '🐑', 'cow': '🐄', 'elephant': '🐘',
    'bear': '🐻', 'zebra': '🦓', 'giraffe': '🦒', 'backpack': '🎒',
    'umbrella': '☂️', 'handbag': '👜', 'tie': '👔', 'suitcase': '🧳',
    'frisbee': '🥏', 'skis': '🎿', 'snowboard': '🏂', 'sports ball': '⚽',
    'kite': '🪁', 'baseball bat': '🏏', 'baseball glove': '🧤',
    'skateboard': '🛹', 'surfboard': '🏄', 'tennis racket': '🎾',
    'bottle': '🍾', 'wine glass': '🍷', 'cup': '☕', 'fork': '🍴',
    'knife': '🔪', 'spoon': '🥄', 'bowl': '🥣', 'banana': '🍌',
    'apple': '🍎', 'sandwich': '🥪', 'orange': '🍊', 'broccoli': '🥦',
    'carrot': '🥕', 'hot dog': '🌭', 'pizza': '🍕', 'donut': '🍩',
    'cake': '🎂', 'chair': '🪑', 'couch': '🛋', 'potted plant': '🪴',
    'bed': '🛏', 'dining table': '🪞', 'toilet': '🚽', 'tv': '📺',
    'laptop': '💻', 'mouse': '🖱', 'remote': '📱', 'keyboard': '⌨️',
    'cell phone': '📱', 'microwave': '📦', 'oven': '🍳', 'toaster': '🍞',
    'sink': '🚿', 'refrigerator': '🧊', 'book': '📚', 'clock': '⏰',
    'vase': '🏺', 'scissors': '✂️', 'teddy bear': '🧸',
    'hair drier': '💨', 'toothbrush': '🪥',
}


class ObjectDetector:
    """
    YOLOv8-based detector with:
    - Thread-safe statistics engine
    - Per-class confidence tracking
    - Frame timeline
    - Pretty bounding box rendering
    """

    def __init__(self, model_name: str = 'yolov8n.pt', conf_threshold: float = 0.45):
        from ultralytics import YOLO
        self.model        = YOLO(model_name)
        self.conf_thresh  = conf_threshold
        self._lock        = threading.Lock()
        self._color_cache: dict = {}
        self._reset_state()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _reset_state(self):
        self.total_detections = 0
        self.frame_count      = 0
        self.class_counts     = defaultdict(int)
        self.conf_sums        = defaultdict(float)
        self.conf_history     = defaultdict(lambda: deque(maxlen=60))
        self.timeline         = deque(maxlen=150)   # ~2.5 min of history
        self.current_dets     = []
        self.session_start    = datetime.now()
        self._fps             = 0.0
        self._fps_frames      = 0
        self._fps_t           = time.time()

    def _color_for(self, label: str):
        """Deterministic, vibrant color per class."""
        if label not in self._color_cache:
            h = (hash(label) * 137) % 360
            r, g, b = colorsys.hsv_to_rgb(h / 360, 0.80, 0.95)
            self._color_cache[label] = (int(b * 255), int(g * 255), int(r * 255))
        return self._color_cache[label]

    # ── Core detection ────────────────────────────────────────────────────────

    def detect_and_draw(self, frame: np.ndarray):
        """
        Run YOLOv8 inference on a BGR frame.
        Returns (annotated_frame, list_of_detections).
        """
        results = self.model(frame, conf=self.conf_thresh, verbose=False)[0]

        dets = []
        for box in results.boxes:
            cls_id   = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            conf     = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            dets.append({
                'class':      cls_name,
                'confidence': conf,
                'bbox':       (x1, y1, x2, y2),
                'emoji':      EMOJI_MAP.get(cls_name, '📦'),
            })

        # Draw all boxes
        for det in dets:
            self._draw_box(frame, det)

        # Overlay: object count badge top-left
        self._draw_hud(frame, len(dets))

        # Thread-safe stats update
        with self._lock:
            self.current_dets   = dets
            self.frame_count   += 1
            self.total_detections += len(dets)
            for d in dets:
                self.class_counts[d['class']] += 1
                self.conf_sums[d['class']]    += d['confidence']
                self.conf_history[d['class']].append(d['confidence'])
            self.timeline.append({
                'frame': self.frame_count,
                'count': len(dets),
                'time':  datetime.now().strftime('%H:%M:%S'),
            })
            # FPS calc
            self._fps_frames += 1
            now = time.time()
            if now - self._fps_t >= 1.0:
                self._fps      = self._fps_frames / (now - self._fps_t)
                self._fps_frames = 0
                self._fps_t    = now

        return frame, dets

    # ── Drawing helpers ───────────────────────────────────────────────────────

    def _draw_box(self, frame: np.ndarray, det: dict):
        x1, y1, x2, y2 = det['bbox']
        color  = self._color_for(det['class'])
        conf   = det['confidence']
        label  = f"  {det['class']}  {conf:.0%}  "

        # Subtle fill
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.07, frame, 0.93, 0, frame)

        # Main border
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Corner accents
        c, t = 16, 3
        for (sx, sy, dx, dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(frame, (sx, sy), (sx + dx*c, sy), color, t)
            cv2.line(frame, (sx, sy), (sx, sy + dy*c), color, t)

        # Label background + text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        ty = (y1 - th - 10) if y1 > (th + 14) else (y2 + 4)
        cv2.rectangle(frame, (x1, ty), (x1 + tw + 4, ty + th + 8), color, -1)
        cv2.putText(frame, label, (x1 + 2, ty + th + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (15, 15, 15), 1, cv2.LINE_AA)

    def _draw_hud(self, frame: np.ndarray, count: int):
        """Draw a minimal HUD showing object count and FPS."""
        h, w = frame.shape[:2]
        txt_count = f" OBJECTS: {count} "
        txt_fps   = f" FPS: {self._fps:.1f} "
        font, sc, th = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        (cw, ch), _ = cv2.getTextSize(txt_count, font, sc, th)
        (fw, fh), _ = cv2.getTextSize(txt_fps,   font, sc, th)
        cv2.rectangle(frame, (8, 8),  (8 + cw + 4, 8 + ch + 8),  (0, 255, 136), -1)
        cv2.rectangle(frame, (8, 26 + ch), (8 + fw + 4, 26 + ch + fh + 8), (0, 0, 0), -1)
        cv2.putText(frame, txt_count, (10, 8 + ch + 2),   font, sc, (10,10,10), th, cv2.LINE_AA)
        cv2.putText(frame, txt_fps,   (10, 26 + ch*2 + 4), font, sc, (0,255,136), th, cv2.LINE_AA)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return a snapshot of all statistics (thread-safe)."""
        with self._lock:
            cc = dict(self.class_counts)
            total = self.total_detections
            conf_avg = {
                k: round(self.conf_sums[k] / cc[k] * 100, 1)
                for k in cc if cc[k] > 0
            }
            conf_hist = {k: list(v) for k, v in self.conf_history.items()}
            tl = list(self.timeline)
            cur = list(self.current_dets)
            elapsed = (datetime.now() - self.session_start).seconds

            return {
                'fps':            round(self._fps, 1),
                'total':          total,
                'frames':         self.frame_count,
                'current_count':  len(cur),
                'unique_classes': len(cc),
                'class_counts':   cc,
                'conf_avg':       conf_avg,
                'conf_history':   conf_hist,
                'timeline':       tl,
                'current_dets':   cur,
                'elapsed_sec':    elapsed,
                'det_per_frame':  round(total / max(self.frame_count, 1), 2),
            }

    def set_confidence(self, value: float):
        self.conf_thresh = value

    def reset(self):
        with self._lock:
            self._reset_state()
