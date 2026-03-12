"""
detector.py - YOLOv8 Object Detection Engine
Works with st.camera_input() - no WebRTC, no av, no system libs needed
"""

import cv2
import numpy as np
import colorsys
from collections import defaultdict, deque
from datetime import datetime

EMOJI_MAP = {
    'person':'🧍','car':'🚗','bicycle':'🚲','motorcycle':'🏍','airplane':'✈️',
    'bus':'🚌','train':'🚂','truck':'🚛','boat':'⛵','traffic light':'🚦',
    'fire hydrant':'🚒','stop sign':'🛑','parking meter':'🅿️','bench':'🪑',
    'bird':'🐦','cat':'🐱','dog':'🐶','horse':'🐴','sheep':'🐑','cow':'🐄',
    'elephant':'🐘','bear':'🐻','zebra':'🦓','giraffe':'🦒','backpack':'🎒',
    'umbrella':'☂️','handbag':'👜','tie':'👔','suitcase':'🧳','frisbee':'🥏',
    'skis':'🎿','snowboard':'🏂','sports ball':'⚽','kite':'🪁',
    'baseball bat':'🏏','baseball glove':'🧤','skateboard':'🛹',
    'surfboard':'🏄','tennis racket':'🎾','bottle':'🍾','wine glass':'🍷',
    'cup':'☕','fork':'🍴','knife':'🔪','spoon':'🥄','bowl':'🥣',
    'banana':'🍌','apple':'🍎','sandwich':'🥪','orange':'🍊','broccoli':'🥦',
    'carrot':'🥕','hot dog':'🌭','pizza':'🍕','donut':'🍩','cake':'🎂',
    'chair':'🪑','couch':'🛋','potted plant':'🪴','bed':'🛏',
    'dining table':'🪞','toilet':'🚽','tv':'📺','laptop':'💻','mouse':'🖱',
    'remote':'📱','keyboard':'⌨️','cell phone':'📱','microwave':'📦',
    'oven':'🍳','toaster':'🍞','sink':'🚿','refrigerator':'🧊','book':'📚',
    'clock':'⏰','vase':'🏺','scissors':'✂️','teddy bear':'🧸',
    'hair drier':'💨','toothbrush':'🪥',
}

_COLOR_CACHE = {}

def color_for(label):
    if label not in _COLOR_CACHE:
        h = (hash(label) * 137) % 360
        r, g, b = colorsys.hsv_to_rgb(h / 360, 0.82, 0.96)
        _COLOR_CACHE[label] = (int(b*255), int(g*255), int(r*255))
    return _COLOR_CACHE[label]


class ObjectDetector:
    def __init__(self, model_name='yolov8n.pt', conf_threshold=0.45):
        from ultralytics import YOLO
        self.model       = YOLO(model_name)
        self.conf_thresh = conf_threshold
        self._reset_stats()

    def _reset_stats(self):
        self.total_detections = 0
        self.frame_count      = 0
        self.class_counts     = defaultdict(int)
        self.conf_sums        = defaultdict(float)
        self.conf_history     = defaultdict(lambda: deque(maxlen=200))
        self.timeline         = deque(maxlen=300)
        self.session_start    = datetime.now()

    def detect_and_draw(self, frame):
        results = self.model(frame, conf=self.conf_thresh, verbose=False)[0]
        dets = []
        for box in results.boxes:
            cls_id   = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            conf     = float(box.conf[0])
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            dets.append({'class':cls_name,'confidence':conf,
                         'bbox':(x1,y1,x2,y2),'emoji':EMOJI_MAP.get(cls_name,'📦')})

        for d in dets:
            self._draw_box(frame, d)
        self._draw_hud(frame, dets)

        self.frame_count      += 1
        self.total_detections += len(dets)
        for d in dets:
            self.class_counts[d['class']] += 1
            self.conf_sums[d['class']]    += d['confidence']
            self.conf_history[d['class']].append(d['confidence'])
        self.timeline.append({'frame':self.frame_count,'count':len(dets),
                              'time':datetime.now().strftime('%H:%M:%S')})
        return frame, dets

    def _draw_box(self, frame, det):
        x1,y1,x2,y2 = det['bbox']
        col   = color_for(det['class'])
        label = f"  {det['class']}  {det['confidence']:.0%}  "
        overlay = frame.copy()
        cv2.rectangle(overlay,(x1,y1),(x2,y2),col,-1)
        cv2.addWeighted(overlay,0.08,frame,0.92,0,frame)
        cv2.rectangle(frame,(x1,y1),(x2,y2),col,2)
        c=14
        for sx,sy,dx,dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(frame,(sx,sy),(sx+dx*c,sy),col,3)
            cv2.line(frame,(sx,sy),(sx,sy+dy*c),col,3)
        font=cv2.FONT_HERSHEY_SIMPLEX
        (tw,th),_=cv2.getTextSize(label,font,0.52,1)
        ty=(y1-th-10) if y1>(th+14) else (y2+4)
        cv2.rectangle(frame,(x1,ty),(x1+tw+4,ty+th+8),col,-1)
        cv2.putText(frame,label,(x1+2,ty+th+3),font,0.52,(15,15,15),1,cv2.LINE_AA)

    def _draw_hud(self, frame, dets):
        font=cv2.FONT_HERSHEY_SIMPLEX
        txt=f"  OBJECTS: {len(dets)}  FRAME #{self.frame_count}  "
        (tw,th),_=cv2.getTextSize(txt,font,0.5,1)
        cv2.rectangle(frame,(8,8),(8+tw+4,8+th+10),(0,255,136),-1)
        cv2.putText(frame,txt,(10,8+th+4),font,0.5,(10,10,10),1,cv2.LINE_AA)

    def get_stats(self):
        cc=dict(self.class_counts)
        conf_avg={k:round(self.conf_sums[k]/cc[k]*100,1) for k in cc if cc[k]>0}
        conf_hist={k:list(v) for k,v in self.conf_history.items()}
        elapsed=(datetime.now()-self.session_start).seconds
        return {
            'total':self.total_detections,
            'frames':self.frame_count,
            'unique_classes':len(cc),
            'class_counts':cc,
            'conf_avg':conf_avg,
            'conf_history':conf_hist,
            'timeline':list(self.timeline),
            'elapsed_sec':elapsed,
            'det_per_frame':round(self.total_detections/max(self.frame_count,1),2),
        }

    def set_confidence(self, v):
        self.conf_thresh = v

    def reset(self):
        self._reset_stats()
