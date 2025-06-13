import torch
from ultralytics import YOLO
import cv2

class PlayerDetector:
    def __init__(self, weights_path: str, conf_threshold: float = 0.4):
        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold

    def detect_players(self, frame):
        """
        Run YOLOv11 detection on the input frame and return player bounding boxes.
        """
        results = self.model.predict(frame, conf=self.conf_threshold, classes=[0], verbose=False)  # class 0 = 'person'
        bboxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                bboxes.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": confidence
                })
        return bboxes
