import os
from ultralytics import YOLO
import numpy as np
import cv2


class ObjectDetector:
    """
    Robust YOLOv8 detector wrapper
    Handles model loading, inference, and result parsing.
    """

    def __init__(self, model_path: str = "yolov8n.pt"):
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")

    def detect(self, image: np.ndarray):
        """
        Run detection on an image
        """
        try:
            results = self.model(image)
            return results
        except Exception as e:
            raise RuntimeError(f"Detection failed: {e}")

    def draw_boxes(self, image: np.ndarray, results):
        """
        Draw bounding boxes on image
        """
        try:
            annotated = image.copy()

            for r in results:
                boxes = r.boxes

                if boxes is None:
                    continue

                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])

                    label = f"{self.model.names[cls]} {conf:.2f}"

                    cv2.rectangle(
                        annotated,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2,
                    )

                    cv2.putText(
                        annotated,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

            return annotated

        except Exception as e:
            raise RuntimeError(f"Failed to draw boxes: {e}")
