import cv2
import numpy as np
from ultralytics import YOLO

PROHIBITED_CLASSES = {
    67: "Cell Phone"
}

class ObjectDetector:
    def __init__(self, model_name="yolov8n.pt", conf_thresh=0.35):
        self.model = YOLO(model_name)
        self.conf_thresh = conf_thresh

    def detect(self, frame):
        results = self.model(frame, verbose=False, conf=self.conf_thresh)[0]
        person_boxes = []
        prohibited_objects = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = [int(v) for v in box.xyxy[0].tolist()]

            if cls_id == 0:
                person_boxes.append(xyxy)
            elif cls_id in PROHIBITED_CLASSES:
                prohibited_objects.append({
                    "box": xyxy,
                    "label": PROHIBITED_CLASSES[cls_id],
                    "conf": round(conf, 2)
                })

        return person_boxes, prohibited_objects

    def match_objects_to_students(self, students, prohibited_objects):
        student_objects = {s["id"]: [] for s in students}

        for obj in prohibited_objects:
            ox1, oy1, ox2, oy2 = obj["box"]
            obj_cx = (ox1 + ox2) / 2
            obj_cy = (oy1 + oy2) / 2

            closest_student = None
            min_dist = float('inf')

            for s in students:
                sx1, sy1, sx2, sy2 = s["box"]
                expand_x = (sx2 - sx1) * 0.3
                expand_y = (sy2 - sy1) * 0.4
                expanded_box = [sx1 - expand_x, sy1 - expand_y, sx2 + expand_x, sy2 + expand_y]

                if (expanded_box[0] <= obj_cx <= expanded_box[2] and
                    expanded_box[1] <= obj_cy <= expanded_box[3]):
                    dist = ((obj_cx - (sx1 + sx2) / 2) ** 2 + (obj_cy - (sy1 + sy2) / 2) ** 2) ** 0.5
                    if dist < min_dist:
                        min_dist = dist
                        closest_student = s["id"]

            if closest_student is not None:
                student_objects[closest_student].append(obj)

        return student_objects
