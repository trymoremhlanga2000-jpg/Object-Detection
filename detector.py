"""
detector.py — Robust YOLOv8 Detection Engine
"""

import cv2
import numpy as np
import colorsys
from collections import defaultdict, deque
from datetime import datetime

EMOJI_MAP = {
    'person':'🧍','car':'🚗','bicycle':'🚲','motorcycle':'🏍','dog':'🐶',
    'cat':'🐱','chair':'🪑','laptop':'💻','cell phone':'📱','book':'📚'
}

_COLOR_CACHE = {}

def color_for(label):
    if label not in _COLOR_CACHE:
        h = (hash(label) % 360) / 360
        r,g,b = colorsys.hsv_to_rgb(h,0.8,0.95)
        _COLOR_CACHE[label] = (int(b*255),int(g*255),int(r*255))
    return _COLOR_CACHE[label]


class ObjectDetector:

    def __init__(self, model_name="yolov8n.pt", conf_threshold=0.45):

        from ultralytics import YOLO

        try:
            self.model = YOLO(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")

        self.conf_thresh = conf_threshold
        self._reset_stats()


    def _reset_stats(self):

        self.total_detections = 0
        self.frame_count = 0

        self.class_counts = defaultdict(int)
        self.conf_sums = defaultdict(float)

        self.conf_history = defaultdict(lambda: deque(maxlen=200))
        self.timeline = deque(maxlen=300)

        self.session_start = datetime.now()


    def detect_and_draw(self, frame):

        dets = []

        try:
            results = self.model(frame, conf=self.conf_thresh, verbose=False)[0]
        except Exception:
            return frame, dets

        if results.boxes is None:
            return frame, dets

        for box in results.boxes:

            try:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())

                x1,y1,x2,y2 = box.xyxy[0].tolist()
                x1,y1,x2,y2 = map(int,[x1,y1,x2,y2])

            except Exception:
                continue

            cls_name = self.model.names.get(cls_id,"object")

            det = {
                "class":cls_name,
                "confidence":conf,
                "bbox":(x1,y1,x2,y2),
                "emoji":EMOJI_MAP.get(cls_name,"📦")
            }

            dets.append(det)

        for d in dets:
            self._draw_box(frame,d)

        self._draw_hud(frame,len(dets))

        self._update_stats(dets)

        return frame,dets


    def _update_stats(self,dets):

        self.frame_count += 1
        self.total_detections += len(dets)

        for d in dets:
            cls = d["class"]

            self.class_counts[cls]+=1
            self.conf_sums[cls]+=d["confidence"]

            self.conf_history[cls].append(d["confidence"])

        self.timeline.append({
            "frame":self.frame_count,
            "count":len(dets),
            "timestamp":datetime.now().strftime("%H:%M:%S")
        })


    def _draw_box(self,frame,det):

        h,w,_ = frame.shape
        x1,y1,x2,y2 = det["bbox"]

        x1=max(0,x1)
        y1=max(0,y1)
        x2=min(w,x2)
        y2=min(h,y2)

        col = color_for(det["class"])

        cv2.rectangle(frame,(x1,y1),(x2,y2),col,2)

        label = f"{det['class']} {det['confidence']:.0%}"

        cv2.putText(
            frame,label,
            (x1,max(20,y1-5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            col,
            1,
            cv2.LINE_AA
        )


    def _draw_hud(self,frame,count):

        txt = f"Objects: {count} | Frame: {self.frame_count}"

        cv2.putText(
            frame,
            txt,
            (10,25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,136),
            2,
            cv2.LINE_AA
        )


    def get_stats(self):

        cc = dict(self.class_counts)

        conf_avg = {
            k:round(self.conf_sums[k]/cc[k]*100,1)
            for k in cc if cc[k]>0
        }

        elapsed = int((datetime.now()-self.session_start).total_seconds())

        return {
            "total":self.total_detections,
            "frames":self.frame_count,
            "unique_classes":len(cc),
            "class_counts":cc,
            "conf_avg":conf_avg,
            "conf_history":{k:list(v) for k,v in self.conf_history.items()},
            "timeline":list(self.timeline),
            "elapsed_sec":elapsed,
            "det_per_frame":round(self.total_detections/max(self.frame_count,1),2)
        }


    def set_confidence(self,val):
        self.conf_thresh = float(val)


    def reset(self):
        self._reset_stats()
