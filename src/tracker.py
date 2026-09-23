import numpy as np

class StudentTracker:
    def __init__(self, max_distance=160.0, init_window_sec=2.5):
        self.next_id = 1
        self.tracks = {}
        self.max_distance = max_distance
        self.init_window_sec = init_window_sec

    def reset(self):
        self.next_id = 1
        self.tracks = {}

    def update(self, detected_boxes, current_timestamp=0.0):
        if len(detected_boxes) == 0:
            return [
                {"id": tid, "label": d["label"], "box": d["box"]}
                for tid, d in self.tracks.items()
            ]

        det_centers = [
            ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
            for b in detected_boxes
        ]

        matched_tracks = {}
        used_dets = set()

        for tid, tinfo in self.tracks.items():
            tcx, tcy = tinfo["center"]
            best_dist = float("inf")
            best_d_idx = None
            for i, (dcx, dcy) in enumerate(det_centers):
                if i in used_dets:
                    continue
                dist = np.sqrt((tcx - dcx) ** 2 + (tcy - dcy) ** 2)
                if dist < self.max_distance and dist < best_dist:
                    best_dist = dist
                    best_d_idx = i

            if best_d_idx is not None:
                matched_tracks[tid] = best_d_idx
                used_dets.add(best_d_idx)
                new_cx, new_cy = det_centers[best_d_idx]
                tinfo["center"] = (tcx * 0.8 + new_cx * 0.2, tcy * 0.8 + new_cy * 0.2)
                tinfo["box"] = detected_boxes[best_d_idx]
                tinfo["disappeared"] = 0

        if current_timestamp <= self.init_window_sec or len(self.tracks) == 0:
            for i, (cx, cy) in enumerate(det_centers):
                if i not in used_dets:
                    self.tracks[self.next_id] = {
                        "center": (cx, cy),
                        "box": detected_boxes[i],
                        "label": f"Student #{self.next_id}",
                        "disappeared": 0
                    }
                    used_dets.add(i)
                    self.next_id += 1

        return [
            {"id": tid, "label": d["label"], "box": d["box"]}
            for tid, d in sorted(self.tracks.items())
        ]
