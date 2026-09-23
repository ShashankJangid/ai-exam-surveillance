import gradio as gr
import pandas as pd
import json
import os
import tempfile
from src.detector import ObjectDetector
from src.pose_analyzer import PoseAnalyzer
from src.pipeline import SurveillancePipeline

global_detector = ObjectDetector(model_name="yolov8n.pt", conf_thresh=0.35)
global_pose_analyzer = PoseAnalyzer(model_name="yolov8n-pose.pt")
pipeline = SurveillancePipeline(obj_detector=global_detector, pose_analyzer=global_pose_analyzer)

def format_kpi_cards(stats=None):
    if not stats:
        stats = {
            "total_students": "-",
            "flagged_incidents": "-",
            "clean_students": "-",
            "integrity_score": "--%"
        }
    return f"""
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 16px 0 20px 0;">
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
            <div style="font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b;">Candidates Monitored</div>
            <div style="font-size: 28px; font-weight: 700; color: #0f172a; margin-top: 4px;">{stats['total_students']}</div>
        </div>
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
            <div style="font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #ef4444;">Flagged Violations</div>
            <div style="font-size: 28px; font-weight: 700; color: #dc2626; margin-top: 4px;">{stats['flagged_incidents']}</div>
        </div>
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
            <div style="font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #10b981;">Clean Candidates</div>
            <div style="font-size: 28px; font-weight: 700; color: #059669; margin-top: 4px;">{stats['clean_students']}</div>
        </div>
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
            <div style="font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #3b82f6;">Integrity Index</div>
            <div style="font-size: 28px; font-weight: 700; color: #2563eb; margin-top: 4px;">{stats['integrity_score']}</div>
        </div>
    </div>
    """

def run_surveillance(video_file, mode, frame_skip, progress=gr.Progress()):
    if not video_file:
        empty_df = pd.DataFrame(columns=["Timestamp", "Student ID", "Violation", "Severity", "Confidence"])
        return None, empty_df, None, None, format_kpi_cards(), "Please provide a video file or record from webcam."

    progress(0.04, desc="Initializing detection...")

    def on_progress(pct, desc):
        progress(pct, desc=desc)

    output_video, incidents, summary, snapshots, stats = pipeline.process_video(
        video_path=video_file,
        mode=mode,
        frame_skip=int(frame_skip),
        progress_callback=on_progress
    )

    if not incidents:
        df = pd.DataFrame([{
            "Timestamp": "00:00",
            "Student ID": "All Candidates",
            "Violation": "No cheating or suspicious behavior detected",
            "Severity": "CLEAN",
            "Confidence": "100%"
        }])
    else:
        df = pd.DataFrame(incidents)

    log_temp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump({
        "summary": summary,
        "mode": mode,
        "stats": stats,
        "incidents": incidents
    }, log_temp, indent=4)
    log_temp.close()

    kpi_html = format_kpi_cards(stats)
    progress(1.0, desc="Audit complete.")

    return output_video, df, snapshots, log_temp.name, kpi_html, summary

custom_css = """
body, .gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    background-color: #f8fafc !important;
}
.header-wrapper {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border-radius: 12px;
    padding: 24px 30px;
    margin-bottom: 20px;
    color: #ffffff;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.15);
}
.app-title {
    font-size: 24px;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #ffffff;
    margin: 0;
}
.app-subtitle {
    font-size: 14px;
    color: #94a3b8;
    margin-top: 4px;
}
.badge-online {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.3);
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
}
.section-title {
    font-size: 15px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #334155;
    margin-bottom: 10px;
}
"""

with gr.Blocks(title="Aegis AI Surveillance Suite") as demo:
    gr.HTML("""
    <div class="header-wrapper">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 class="app-title">AEGIS // EXAM SURVEILLANCE SUITE</h1>
                <div class="app-subtitle">Autonomous Examination Integrity Verification & Anomaly Detection</div>
            </div>
            <div class="badge-online">
                <span style="font-size: 10px;">●</span> SYSTEM READY
            </div>
        </div>
    </div>
    """)

    kpi_display = gr.HTML(format_kpi_cards())

    with gr.Row():
        with gr.Column(scale=5):
            gr.Markdown("<div class='section-title'>1. Surveillance Video Feed</div>")
            video_input = gr.Video(
                label="Source Video Stream",
                sources=["upload", "webcam"],
                interactive=True
            )

            with gr.Accordion("Settings", open=True):
                mode_select = gr.Radio(
                    choices=[
                        "Classroom / Exam Hall (CCTV)",
                        "Remote Online Exam (Webcam)"
                    ],
                    value="Classroom / Exam Hall (CCTV)",
                    label="Environment"
                )
                frame_skip_slider = gr.Slider(
                    minimum=1,
                    maximum=6,
                    value=3,
                    step=1,
                    label="Frame Skip (Acceleration)",
                    info="Skip rate for processing acceleration"
                )

            analyze_btn = gr.Button("Analyze Stream", variant="primary", size="lg")

        with gr.Column(scale=6):
            gr.Markdown("<div class='section-title'>2. Annotated Output</div>")
            status_banner = gr.Markdown("Ready for examination video stream.")
            video_output = gr.Video(
                label="Surveillance Output",
                interactive=False
            )

    with gr.Row():
        with gr.Column():
            gr.Markdown("<div class='section-title'>3. Visual Evidence Gallery</div>")
            evidence_gallery = gr.Gallery(
                label="Flagged Incidents",
                show_label=False,
                columns=4,
                rows=1,
                height="auto",
                object_fit="contain"
            )

    with gr.Row():
        with gr.Column():
            gr.Markdown("<div class='section-title'>4. Incident Log</div>")
            incident_table = gr.DataFrame(
                headers=["Timestamp", "Student ID", "Violation", "Severity", "Confidence"],
                datatype=["str", "str", "str", "str", "str"],
                label="Recorded Violations",
                interactive=False
            )
            with gr.Row():
                download_json_btn = gr.File(label="Download Report (JSON)")

    analyze_btn.click(
        fn=run_surveillance,
        inputs=[video_input, mode_select, frame_skip_slider],
        outputs=[video_output, incident_table, evidence_gallery, download_json_btn, kpi_display, status_banner]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
