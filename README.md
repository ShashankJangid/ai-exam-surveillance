# Aegis AI // Exam Surveillance Suite

An autonomous, multi-candidate examination surveillance and proctoring system built with YOLOv8, 17-keypoint human pose estimation, spatial desk tracking, and temporal behavioral analysis.

Designed to detect cheating across diverse camera perspectives (including oblique/diagonal CCTV viewpoints and straight-on classroom aisles) with high accuracy and zero false positives.

---

## 🚀 Quickstart (Run in 3 Steps)

### 1. Clone & Enter Repository
```bash
git clone https://github.com/ShashankJangid/ai-exam-surveillance.git
cd ai-exam-surveillance
```

### 2. Create Virtual Environment & Install Dependencies
```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows:
.venv\Scripts\activate
# Or on Linux / macOS:
# source .venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 3. Launch Web Dashboard
```bash
python app.py
```
Open your browser and navigate to:
```text
http://127.0.0.1:7860
```

---

## 📖 How to Use the Application

### Method 1: Analyzing Recorded Exam Videos (Recommended)

1. **Upload Video:**
   - In the **1. Surveillance Video Feed** panel, click **Upload** or drag-and-drop a video file.
   - You can test immediately using either of the provided sample videos located in the `sample_videos/` folder:
     - `sample_videos/155164-809618953.mp4` *(Classroom with diagonal camera, forward exam copying)*
     - `sample_videos/7092083-hd_1920_1080_30fps.mp4` *(Classroom front view, students colluding across aisle)*
2. **Select Operational Mode:**
   - **Classroom / Exam Hall (CCTV):** Multi-student mode. Tracks all candidates at their desks, monitors forward desk intrusion, aisle turning, and cell phone usage.
   - **Remote Online Exam (Webcam):** Single-candidate remote proctoring mode. Flags absence from desk, multiple people in room, looking away, or phone use.
3. **Configure Frame Skip (Acceleration):**
   - Default is `3x` (processes 1 in every 3 frames). This accelerates processing by ~400% with full temporal fidelity.
   - Use `1x` if you require strict frame-by-frame annotation.
4. **Click "Analyze Stream":**
   - A real-time progress bar will indicate processing status.
5. **Review the Results:**
   - **Annotated Stream:** Watch the processed video with live bounding boxes:
     - 🟩 **Emerald Green:** Normal, attentive writing behavior.
     - 🟥 **Bright Red with Badge:** Confirmed cheating violation.
     - 🟦 **Blue Box:** Prohibited contraband (cell phone) detected in desk area.
   - **Executive KPI Cards:** Review total monitored candidates, flagged violations, clean candidates, and room integrity index.
   - **Visual Forensic Proof Gallery:** Click on any captured keyframe snapshot to view a high-resolution, timestamped proof of the infraction.
   - **Incident Log Table:** Chronological record detailing timestamps, candidate IDs, violation categories, and confidence percentages.
   - **Download Audit Dossier:** Click **Download Report (JSON)** to export a complete, machine-readable forensic dossier for administrative records.

---

### Method 2: Live Webcam Proctoring

1. Click on the **Webcam** tab in the video input box.
2. Grant browser camera permissions.
3. Record a clip or session.
4. Select **Remote Online Exam (Webcam)** mode.
5. Click **Analyze Stream** to inspect proctoring compliance.

---

### Method 3: Programmatic Python API

You can import and integrate the surveillance engine into your own Python applications or batch processing scripts:

```python
from src.pipeline import SurveillancePipeline

# Initialize the pipeline
pipeline = SurveillancePipeline()

# Process video file
output_video_path, incidents, summary, snapshots, stats = pipeline.process_video(
    video_path="sample_videos/155164-809618953.mp4",
    mode="Classroom / Exam Hall (CCTV)",
    frame_skip=3
)

print(summary)
print(f"Room Integrity: {stats['integrity_score']}")

# Iterate through flagged incidents
for incident in incidents:
    print(f"[{incident['timestamp']}] {incident['student_id']}: {incident['violation']} (Severity: {incident['severity']})")
```

---

## 🎯 Benchmark Evaluation & Sample Videos

The repository includes two real-world examination test recordings in the `sample_videos/` folder:

| Sample Video | Scenario & Camera Angle | Candidates Monitored | Flagged Candidates | Clean Candidates | False Alarms | Detected Violations |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`sample_videos/155164-809618953.mp4`** | Oblique / diagonal CCTV angle | **5** | **1** (`Student #2`) | **4** | **0** | `Peeking & Copying from Student in Front` at 4.3s, 6.9s, 11.7s, 14.3s, 16.9s, 21.9s |
| **`sample_videos/7092083-hd_1920_1080_30fps.mp4`** | Front classroom view | **5** | **2** (`Student #3` & `#5`) | **3** | **0** | `Peeking Across Aisle (Right)` [2.2s, 4.7s, 7.2s] & `Peeking Across Aisle (Left)` [2.9s] |

Both videos run through the exact same unified pipeline sequentially without changing any settings or thresholds.

---

## 🔬 How the Technology Works

1. **Stationary Seated Desk Tracking (`src/tracker.py`):**
   - Traditional IoU tracking often swaps IDs when students lean forward or shift posture.
   - Aegis anchors each student to their physical desk center using Euclidean centroid tracking with an Exponential Moving Average (EMA). Each student retains a permanent ID throughout the entire exam.
2. **Scale-Invariant Posture Normalization (`src/pose_analyzer.py`):**
   - Normalized gaze yaw and pitch are calculated by dividing head position by shoulder width.
   - Shoulder width is clamped to a minimum of 70px, preventing numerical division explosion on distant, background candidates.
3. **Dynamic Seating Baseline:**
   - During the first 2.5 seconds, the engine learns each student's normal writing posture to neutralize natural camera angles.
4. **Anatomical Multi-Vector Rules (`src/rules_engine.py`):**
   - **Forward Exam Copying:** Triggered when a student leans forward and raises their pitch toward the front desk ($\Delta\text{pitch} < -0.30, \Delta\text{yaw} > 0.16$).
   - **Aisle Collusion:** Requires both baseline deviation and true anatomical lateral head rotation ($\Delta\text{yaw} > 0.35$ with positive yaw for Right, $\Delta\text{yaw} < -0.35$ with negative yaw for Left).
   - **Sustained Duration:** Requires sustained posture for $\ge 0.6\text{s} - 0.7\text{s}$ before triggering, ignoring brief natural head adjustments.

---

## 📂 Repository Directory Layout

```text
ai-exam-surveillance/
├── sample_videos/
│   ├── 155164-809618953.mp4           # Benchmark Video 1 (Forward Peeking)
│   └── 7092083-hd_1920_1080_30fps.mp4 # Benchmark Video 2 (Aisle Collusion)
├── src/
│   ├── __init__.py
│   ├── detector.py                    # Prohibited object & contraband detection
│   ├── pose_analyzer.py               # Scale-invariant 17-keypoint gaze estimation
│   ├── tracker.py                     # Seated desk-anchored tracking
│   ├── rules_engine.py                # Temporal violation state machine
│   └── pipeline.py                    # Stream orchestrator & video encoder
├── app.py                             # Interactive Gradio web application
├── openh264-2.5.0-win64.dll           # OpenH264 library for native browser playback
├── yolov8n.pt                         # YOLOv8 object detection weights
├── yolov8n-pose.pt                    # YOLOv8 pose estimation weights
├── requirements.txt                   # Dependency manifest
├── LICENSE                            # MIT License
└── README.md                          # Documentation
```

---

## 🛠️ Frequently Asked Questions (FAQ)

<details>
<summary><b>1. Port 7860 is already in use. How do I change the port?</b></summary>
In <code>app.py</code>, change the last line:
<pre><code>demo.launch(server_name="127.0.0.1", server_port=8080)</code></pre>
</details>

<details>
<summary><b>2. How do I enable GPU acceleration?</b></summary>
If you have an NVIDIA GPU with CUDA installed, PyTorch and Ultralytics will automatically utilize the GPU. No code changes are required.
</details>

<details>
<summary><b>3. Why is openh264 included?</b></summary>
OpenCV on Windows requires the Cisco OpenH264 binary to encode standard H.264 (<code>avc1</code>) MP4 videos. Including the DLL ensures browser-playable video output out of the box without manual codec installations.
</details>

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
