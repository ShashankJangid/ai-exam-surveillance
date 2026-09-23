import numpy as np
from ultralytics import YOLO

class PoseAnalyzer:
    def __init__(self, model_name="yolov8n-pose.pt"):
        self.model = YOLO(model_name)
        self.baselines = {}

    def reset_baselines(self):
        self.baselines = {}

    def analyze(self, frame, tracked_students):
        student_poses = {}
        for s in tracked_students:
            student_poses[s["id"]] = {
                "action": "WRITING_NORMAL",
                "head_turn": "FORWARD",
                "dyaw": 0.0,
                "dpitch": 0.0,
                "suspicious": False,
                "violation_type": None,
                "confidence": 0.90
            }

        results = self.model(frame, verbose=False, conf=0.30, iou=0.45)[0]

        if results.boxes is None or len(results.boxes) == 0:
            return student_poses

        boxes = results.boxes.xyxy.cpu().numpy()
        kpts_data = results.keypoints.data.cpu().numpy()

        for box, kpts in zip(boxes, kpts_data):
            nose = kpts[0]
            l_sh, r_sh = kpts[5], kpts[6]

            if nose[2] < 0.35 or l_sh[2] < 0.20 or r_sh[2] < 0.20:
                continue

            sh_mid_x = (l_sh[0] + r_sh[0]) / 2.0
            sh_mid_y = (l_sh[1] + r_sh[1]) / 2.0
            sh_w = max(70.0, abs(r_sh[0] - l_sh[0]))

            norm_yaw = (nose[0] - sh_mid_x) / sh_w
            norm_pitch = (nose[1] - sh_mid_y) / sh_w

            matched_id = None
            best_dist = float("inf")
            bcx = (box[0] + box[2]) / 2.0
            bcy = (box[1] + box[3]) / 2.0

            for s in tracked_students:
                sx1, sy1, sx2, sy2 = s["box"]
                scx = (sx1 + sx2) / 2.0
                scy = (sy1 + sy2) / 2.0
                dist = ((bcx - scx) ** 2 + (bcy - scy) ** 2) ** 0.5
                if dist < 160.0 and dist < best_dist:
                    best_dist = dist
                    matched_id = s["id"]

            if matched_id is None:
                continue

            if matched_id not in self.baselines:
                self.baselines[matched_id] = {
                    "yaws": [],
                    "pitches": [],
                    "base_yaw": 0.0,
                    "base_pitch": 0.0,
                    "calibrated": False
                }

            b_data = self.baselines[matched_id]
            if not b_data["calibrated"]:
                b_data["yaws"].append(norm_yaw)
                b_data["pitches"].append(norm_pitch)
                if len(b_data["yaws"]) >= 8:
                    b_data["base_yaw"] = float(np.median(b_data["yaws"]))
                    b_data["base_pitch"] = float(np.median(b_data["pitches"]))
                    b_data["calibrated"] = True
                continue

            dyaw = norm_yaw - b_data["base_yaw"]
            dpitch = norm_pitch - b_data["base_pitch"]

            action = "WRITING_NORMAL"
            suspicious = False
            violation_type = None
            conf = 0.92

            if dpitch < -0.30 and dyaw > 0.16:
                action = "PEEKING_FRONT"
                suspicious = True
                violation_type = "Peeking & Copying from Student in Front"
                conf = 0.96
            elif dyaw > 0.35 and norm_yaw > 0.15 and dpitch < 0.15:
                action = "PEEKING_RIGHT"
                suspicious = True
                violation_type = "Peeking / Communicating Across Aisle (Right)"
                conf = 0.94
            elif dyaw < -0.35 and norm_yaw < -0.35 and dpitch < 0.15:
                action = "PEEKING_LEFT"
                suspicious = True
                violation_type = "Peeking / Communicating Across Aisle (Left)"
                conf = 0.94
            elif dpitch < -0.45:
                action = "LOOKING_UP"
                suspicious = True
                violation_type = "Looking Away from Exam Paper"
                conf = 0.89

            student_poses[matched_id] = {
                "action": action,
                "head_turn": action,
                "dyaw": round(float(dyaw), 2),
                "dpitch": round(float(dpitch), 2),
                "suspicious": suspicious,
                "violation_type": violation_type,
                "confidence": conf
            }

        return student_poses
