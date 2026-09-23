class ExamRulesEngine:
    def __init__(self, mode="Classroom / Exam Hall (CCTV)", fps=10):
        self.mode = mode
        self.fps = fps
        self.student_states = {}
        self.confirmed_incidents = []

        self.PEEK_FRONT_THRESH = 0.6
        self.PEEK_SIDE_THRESH = 0.7
        self.PHONE_THRESH = 0.4

    def reset(self):
        self.student_states = {}
        self.confirmed_incidents = []

    def evaluate_frame(self, timestamp, students, student_poses, student_objects):
        frame_statuses = {}
        new_incidents = []

        if "Online" in self.mode:
            if len(students) > 1:
                incident = {
                    "timestamp": f"{timestamp:04.1f}s",
                    "student_id": "Candidate",
                    "violation": f"Multiple People in Room ({len(students)} detected)",
                    "severity": "CRITICAL",
                    "confidence": "96%"
                }
                self.confirmed_incidents.append(incident)
                new_incidents.append(incident)
            elif len(students) == 0:
                incident = {
                    "timestamp": f"{timestamp:04.1f}s",
                    "student_id": "Candidate",
                    "violation": "Candidate Missing from Desk / Camera View",
                    "severity": "CRITICAL",
                    "confidence": "99%"
                }
                self.confirmed_incidents.append(incident)
                new_incidents.append(incident)

        for s in students:
            sid = s["id"]
            if sid not in self.student_states:
                self.student_states[sid] = {
                    "peek_front_frames": 0,
                    "peek_side_frames": 0,
                    "phone_frames": 0,
                    "last_logged_time": -10.0
                }

            state = self.student_states[sid]
            pose = student_poses.get(sid, {})
            objects = student_objects.get(sid, [])

            action = pose.get("action", "WRITING_NORMAL")
            has_phone = any(obj["label"] == "Cell Phone" for obj in objects)

            if action == "PEEKING_FRONT":
                state["peek_front_frames"] += 1
            else:
                state["peek_front_frames"] = max(0, state["peek_front_frames"] - 1)

            if action in ["PEEKING_LEFT", "PEEKING_RIGHT"]:
                state["peek_side_frames"] += 1
            else:
                state["peek_side_frames"] = max(0, state["peek_side_frames"] - 1)

            if has_phone:
                state["phone_frames"] += 2
            else:
                state["phone_frames"] = max(0, state["phone_frames"] - 1)

            status = "NORMAL"
            color = (16, 185, 129)
            active_violations = []

            if state["phone_frames"] >= int(self.PHONE_THRESH * self.fps):
                active_violations.append("Prohibited Device (Cell Phone)")
                status = "CHEATING_FLAGGED"
                color = (0, 0, 255)
            elif state["peek_front_frames"] >= int(self.PEEK_FRONT_THRESH * self.fps):
                active_violations.append("Peeking & Copying from Student in Front")
                status = "CHEATING_FLAGGED"
                color = (0, 0, 255)
            elif state["peek_side_frames"] >= int(self.PEEK_SIDE_THRESH * self.fps):
                side = "Right" if action == "PEEKING_RIGHT" else "Left"
                active_violations.append(f"Peeking / Communicating Across Aisle ({side})")
                status = "CHEATING_FLAGGED"
                color = (0, 0, 255)
            elif state["peek_front_frames"] > 0 or state["peek_side_frames"] > 0:
                status = "SUSPICIOUS"
                color = (0, 165, 255)

            if status == "CHEATING_FLAGGED" and (timestamp - state["last_logged_time"]) > 2.5:
                state["last_logged_time"] = timestamp
                for v in active_violations:
                    incident = {
                        "timestamp": f"{timestamp:04.1f}s",
                        "student_id": s["label"],
                        "violation": v,
                        "severity": "CRITICAL" if "Phone" in v or "Front" in v else "HIGH",
                        "confidence": f"{int(pose.get('confidence', 0.94) * 100)}%"
                    }
                    self.confirmed_incidents.append(incident)
                    new_incidents.append(incident)

            frame_statuses[sid] = {
                "status": status,
                "color": color,
                "active_violations": active_violations,
                "action": action
            }

        return frame_statuses, new_incidents
