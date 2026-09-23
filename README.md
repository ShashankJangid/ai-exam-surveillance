# Aegis AI // Exam Surveillance Suite

An autonomous, multi-student examination proctoring and integrity surveillance system built with YOLOv8, pose estimation, spatial desk tracking, and temporal behavioral analysis.

Designed to detect student cheating in classroom environments and remote webcams across diverse camera perspectives (including oblique/diagonal CCTV views and straight-on classroom aisles).

---

## Key Features

- **Multi-Candidate Tracking:** Anchors candidates to physical desk coordinates to ensure consistent tracking IDs across the entire exam without ID-swapping or fragmentation.
- **Adaptive Posture & Gaze Estimation:** Calibrates an individualized baseline posture per candidate over the opening seconds of the examination, neutralizing natural seating offsets and camera angle tilt.
- **Cross-Video & Multi-Vector Violation Detection:**
  - **Peeking Ahead / Copying:** Detects forward desk intrusion and head lift when copying from a peer in front.
  - **Aisle Collusion (Lateral Turns):** Bilateral recognition for candidates turning left or right to communicate across aisles.
  - **Contraband Detection:** Identifies prohibited devices (e.g., cell phones) in the candidate workspace.
- **Visual Evidence Extraction:** Automatically captures high-resolution keyframe snapshots of confirmed infractions for auditor review.
- **Hardware-Compatible Video Export:** Produces standard H.264 (`avc1`) video files with visual HUD overlays for native browser playback.
- **Executive Web Dashboard:** Interactive Gradio interface with KPI cards, evidence gallery, live filtering, and exportable JSON audit reports.

---

## Project Structure

```text
├── sample_videos/
│   ├── 155164-809618953.mp4           # Test Video 1 (Diagonal / Front Copying)
│   └── 7092083-hd_1920_1080_30fps.mp4 # Test Video 2 (Classroom / Aisle Collusion)
├── src/
│   ├── __init__.py
│   ├── detector.py                    # Object & contraband detection
│   ├── pose_analyzer.py               # Scale-invariant pose & gaze estimation
│   ├── tracker.py                     # Seated candidate desk tracker
│   ├── rules_engine.py                # Temporal violation state machine
│   └── pipeline.py                    # End-to-end stream orchestrator
├── app.py                             # Gradio web dashboard
├── openh264-2.5.0-win64.dll           # OpenH264 codec for Windows H.264 encoding
├── yolov8n.pt                         # YOLOv8 object detection weights
├── yolov8n-pose.pt                    # YOLOv8 pose estimation weights
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
└── README.md
```

---

## Installation & Setup

### Prerequisites

- Python 3.10+
- (Optional) CUDA-compatible GPU for accelerated inference

### 1. Clone the Repository

```bash
git clone https://github.com/ShashankJangid/ai-exam-surveillance.git
cd ai-exam-surveillance
```

### 2. Install Dependencies

Using `pip`:

```bash
pip install -r requirements.txt
```

Or using `uv`:

```bash
uv pip install -r requirements.txt
```

---

## Running the Application

Launch the local web dashboard:

```bash
python app.py
```

Open your browser and navigate to:
```text
http://127.0.0.1:7860
```

1. Upload any exam video (or select from `sample_videos/`) or capture directly via webcam.
2. Select the operational mode (`Classroom / Exam Hall` or `Remote Online Exam`).
3. Click **Analyze Stream** to view real-time detections, KPI metrics, flagged evidence frames, and download the incident audit log.

---

## Benchmark Evaluation

Tested on the included sample datasets:

| Video | Scenario | Candidates Monitored | Flagged Suspects | Clean Candidates | False Alarms | Primary Infraction |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`155164-809618953.mp4`** | Diagonal camera angle | 5 | 1 | 4 | 0 | `Peeking & Copying from Student in Front` |
| **`7092083-hd_1920_1080_30fps.mp4`** | Front classroom view | 5 | 2 | 3 | 0 | `Peeking / Communicating Across Aisle` |

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
