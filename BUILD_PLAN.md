# 📋 Engineering Build Plan: AI Exam Surveillance Suite

A structured, end-to-end execution roadmap detailing technical tasks, deliverables, ownership, milestones, and required proficiencies for developing, deploying, and maintaining the autonomous AI Exam Surveillance platform.

---

## 🎯 Project Overview & Objectives
- **System:** Aegis AI // Exam Surveillance Suite
- **Core Technology:** YOLOv8 (Object & 17-Keypoint Pose), Centroid Desk Tracking, Scale-Invariant Normalization, Temporal State Machine, OpenH264, Gradio Web UI.
- **Repository:** [`ShashankJangid/ai-exam-surveillance`](https://github.com/ShashankJangid/ai-exam-surveillance)

---

## 🏗️ Phased Build Plan Matrix

| Task | Deliverables | Assigned To (can be the mentor) | Deadline | Resources/Skills Required |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1: Environment Setup & Computer Vision Backbone** | • Working Python 3.12 / `uv` virtual environment.<br>• Verified inference pipeline with `yolov8n.pt` (contraband/phone) and `yolov8n-pose.pt` (17 keypoints).<br>• Initial test suite benchmarking person and object detection confidence. | AI Lead / Developer (Mentored by Computer Vision Lead) | Week 1 | • Python 3.10+, PyTorch, Ultralytics YOLOv8<br>• CUDA / cuDNN environment<br>• OpenCV |
| **Phase 2: Spatial Desk-Centric Multi-Candidate Tracking** | • `src/tracker.py`: Euclidean centroid tracker with Exponential Moving Average (EMA).<br>• Desk anchoring logic ($< 160\text{px}$ radius) to eliminate track fragmentation and ID swaps during posture shifts.<br>• Unit test verifying 100% ID persistence on seated candidates. | Computer Vision Engineer (Mentored by System Architect) | Week 2 | • NumPy vector mathematics<br>• Object tracking algorithms (Centroid / Hungarian / SORT principles)<br>• Video annotation |
| **Phase 3: Scale-Invariant Normalization & Baseline Calibration** | • `src/pose_analyzer.py`: Clamped shoulder-width normalization ($w_{\text{clamped}} = \max(70.0, \|x_r - x_l\|)$).<br>• Initial 8-sample ($2.5\text{s}$) baseline calibration per candidate desk.<br>• Perspective-invariant angular gaze deviation ($\Delta\text{yaw}, \Delta\text{pitch}$). | Deep Learning Engineer (Mentored by Lead AI Researcher) | Week 3 | • Human pose estimation & 2D keypoint kinematics<br>• Statistical distribution modeling (median filtering)<br>• Coordinate geometry |
| **Phase 4: Multi-Vector Temporal Rules Engine** | • `src/rules_engine.py`: Multi-directional temporal state machine.<br>• Detection of: (1) Forward desk copying ($\Delta\text{pitch} < -0.30, \Delta\text{yaw} > 0.16$), (2) Bilateral aisle collusion ($\Delta\text{yaw} \gtrless \pm 0.35$ with anatomical constraint), (3) Prohibited devices.<br>• Sustained temporal thresholds ($\ge 0.6\text{s}-0.7\text{s}$) with 2.5s cooldown logging. | Software Engineer / ML Engineer (Mentored by Lead Proctoring Consultant) | Week 4 | • Finite State Machines (FSM)<br>• Temporal signal filtering<br>• Behavioral anomaly detection |
| **Phase 5: Native Browser H.264 Pipeline & Visual Proof Extraction** | • `src/pipeline.py`: Hardware-accelerated OpenH264 (`avc1`) encoding pipeline.<br>• Automated incident snapshot capture engine saving timestamped keyframes.<br>• Dynamic surveillance HUD overlay with color-coded status badges (Green = Normal, Red = Violation, Blue = Contraband). | Systems Engineer (Mentored by Media & Codec Specialist) | Week 5 | • OpenH264 library & Cisco codec integration<br>• OpenCV `VideoWriter` & video transcoding<br>• File I/O & temp file management |
| **Phase 6: Executive Web Dashboard & User Experience** | • `app.py`: High-performance Gradio interface.<br>• 4 KPI executive scorecards (Candidates, Violations, Clean Candidates, Integrity Index).<br>• Clickable Visual Forensic Proof Gallery with image zoom.<br>• One-click sample video loading panel.<br>• Exportable machine-readable JSON & CSV audit dossiers. | Full-Stack AI Engineer (Mentored by Product / UX Mentor) | Week 6 | • Gradio Framework, HTML5/CSS3<br>• Pandas DataFrames, JSON schema<br>• UI/UX design & responsive layouts |
| **Phase 7: Cross-Video Validation & Zero-False-Alarm Tuning** | • Dual-video benchmark suite testing oblique CCTV and front aisle views.<br>• Formal validation report confirming zero false alarms on innocent candidates.<br>• Automated regression test script (`test_pipeline.py`). | QA / ML Validation Engineer (Mentored by Lead AI Architect) | Week 7 | • Quality Assurance & automated regression testing<br>• Model calibration & precision-recall tuning<br>• Confusion matrix & benchmark metrics |
| **Phase 8: Production Hardening, Edge Streaming & Cloud Packaging** | • Multi-threaded RTSP / IP camera live stream ingestion (low latency).<br>• Containerized Dockerfile and `docker-compose.yml`.<br>• GPU acceleration profile for real-time edge processing (NVIDIA Jetson / Cloud VM).<br>• Production release tagged on GitHub. | DevOps / MLOps Engineer (Mentored by Infrastructure Mentor) | Week 8 | • Docker & Containerization<br>• RTSP streaming & GStreamer/FFmpeg<br>• MLOps & CI/CD deployment |

---

## 👥 Roles & Mentorship Responsibilities

### Mentorship Checkpoints
1. **Milestone Review 1 (End of Week 2):** Audit pose stability and desk tracking centroid accuracy under seated occlusion.
2. **Milestone Review 2 (End of Week 4):** Verify false-positive mitigation on innocent students writing naturally.
3. **Milestone Review 3 (End of Week 6):** User acceptance testing on the executive web dashboard and JSON evidence export.
4. **Final Sign-off (End of Week 8):** Edge deployment readiness, latency benchmark (<50ms per frame), and compliance sign-off.
