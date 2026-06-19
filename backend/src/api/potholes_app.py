import gradio as gr
import cv2
import os
import csv
import shutil
import tempfile
import subprocess
import glob
from ultralytics import YOLO
from huggingface_hub import snapshot_download
import imageio_ffmpeg

# Set up ffmpeg path so video conversion works automatically
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_exe)

# Download model from Hugging Face
print("Downloading Pothole model...")
try:
    model_path = snapshot_download(repo_id="Harisanth/Pothole-Finetuned-YOLOv8", allow_patterns=["*.pt"])
    pt_files = glob.glob(os.path.join(model_path, "*.pt"))
    if not pt_files:
        raise ValueError("No .pt files found in the Hugging Face repository.")
    model = YOLO(pt_files[0])
    print(f"Pothole Model loaded from {pt_files[0]}")
except Exception as e:
    print(f"Failed to load pothole model: {e}")

def process_video(video_path, conf_threshold=0.50, enable_augmentation=True, filter_white_lines=True, save_csv=True, save_screenshots=True, progress=gr.Progress()):
    # Setup output directories
    temp_dir = tempfile.mkdtemp()
    screenshots_dir = os.path.join(temp_dir, "screenshots")
    if save_screenshots:
        os.makedirs(screenshots_dir, exist_ok=True)
    
    csv_file_path = os.path.join(temp_dir, "detections.csv") if save_csv else ""
    raw_video_path = os.path.join(temp_dir, "raw_output.mp4")
    output_video_path = os.path.join(temp_dir, "output_video.mp4")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30.0 # Default if unknown
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(raw_video_path, fourcc, fps, (width, height))
    
    csv_file = None
    csv_writer = None
    if save_csv:
        csv_file = open(csv_file_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Frame_Number', 'Timestamp_Seconds', 'Confidence_Score', 'Severity', 'Bounding_Box_Coordinates_XYXY'])
    
    frame_number = 0
    pothole_detected = False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 1
        
    screenshot_count = 0
    confidence_scores = []
    severity_counts = {"Small": 0, "Medium": 0, "Large": 0}
    logged_pothole_ids = set()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        progress(frame_number / total_frames, desc=f"Processing frame {frame_number}/{total_frames}")
            
        results = model.track(
            frame, 
            conf=conf_threshold, 
            augment=enable_augmentation, 
            iou=0.45, 
            persist=True,
            verbose=False,
            tracker="botsort.yaml"
        )
        result = results[0]
        
        annotated_frame = frame.copy()
        frame_pothole_detected = False
        new_pothole_in_frame = False
        
        if len(result.boxes) > 0:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf >= conf_threshold:
                    coords = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])

                    if filter_white_lines:
                        box_width = x2 - x1
                        box_height = y2 - y1
                        if box_height > 0:
                            aspect_ratio = box_width / box_height
                            if aspect_ratio > 4.0 or aspect_ratio < 0.25:
                                continue

                        cropped = frame[y1:y2, x1:x2]
                        if cropped.size > 0:
                            gray_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                            avg_brightness = gray_crop.mean()
                            if avg_brightness > 150:
                                continue

                    box_id = int(box.id[0]) if box.id is not None else None
                    is_new_pothole = True
                    if box_id is not None:
                        if box_id in logged_pothole_ids:
                            is_new_pothole = False
                        else:
                            logged_pothole_ids.add(box_id)

                    frame_pothole_detected = True
                    pothole_detected = True
                    
                    if is_new_pothole:
                        new_pothole_in_frame = True
                        det_width = x2 - x1
                        det_height = y2 - y1
                        det_area = det_width * det_height
                        if det_area > 8000:
                            severity = "Large"
                        elif det_area > 2000:
                            severity = "Medium"
                        else:
                            severity = "Small"
                            
                        severity_counts[severity] += 1
                        confidence_scores.append(conf)
                        
                        if save_csv and csv_writer:
                            coords_str = f"[{x1}, {y1}, {x2}, {y2}]"
                            timestamp = frame_number / fps
                            csv_writer.writerow([frame_number, f"{timestamp:.2f}", f"{conf:.2f}", severity, coords_str])
                    
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    
                    if box_id is not None:
                        label = f"Pothole #{box_id} {conf:.2f}"
                    else:
                        label = f"Pothole {conf:.2f}"
                        
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(annotated_frame, (x1, max(0, y1 - 25)), (x1 + w, y1), (0, 0, 255), -1)
                    cv2.putText(annotated_frame, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        out.write(annotated_frame)
        
        try:
            progress(frame_number / total_frames, desc=f"Processing frame {frame_number}/{total_frames}", vis_frame=annotated_frame)
        except TypeError:
            progress(frame_number / total_frames, desc=f"Processing frame {frame_number}/{total_frames}")
        
        if new_pothole_in_frame and save_screenshots:
            screenshot_count += 1
            timestamp_sec = int(frame_number / fps)
            max_conf = max([float(box.conf[0]) for box in result.boxes if float(box.conf[0]) >= conf_threshold])
            screenshot_name = f"pothole_shot{screenshot_count:04d}_conf{max_conf:.2f}_time{timestamp_sec}s.jpg"
            screenshot_path = os.path.join(screenshots_dir, screenshot_name)
            cv2.imwrite(screenshot_path, annotated_frame)
            
        frame_number += 1

    cap.release()
    out.release()
    if csv_file:
        csv_file.close()
    
    progress(1.0, desc="Finalizing output...")
    
    screenshots_zip = ""
    if save_screenshots:
        screenshots_zip = os.path.join(temp_dir, "screenshots.zip")
        shutil.make_archive(screenshots_zip.replace('.zip', ''), 'zip', screenshots_dir)
    
    try:
        subprocess.run([ffmpeg_exe, '-i', raw_video_path, '-vcodec', 'libx264', '-crf', '28', output_video_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        final_video_path = output_video_path
    except Exception as e:
        print(f"Failed to convert video with ffmpeg: {e}")
        final_video_path = raw_video_path

    video_duration = frame_number / fps
    minutes = int(video_duration // 60)
    seconds = int(video_duration % 60)
    avg_conf = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
    
    summary = f"""Detection Summary
- Video Duration: {minutes} min {seconds} sec
- Frames Processed: {frame_number:,}
- Average Confidence: {avg_conf:.2f}

Severity Breakdown:
- 🟡 Small: {severity_counts['Small']}
- 🟠 Medium: {severity_counts['Medium']}
- 🔴 Large: {severity_counts['Large']}"""

    return final_video_path, csv_file_path, screenshots_zip, summary

def create_pothole_app():
    hawkeye_theme = gr.themes.Base(
        primary_hue=gr.themes.Color(c50="#e0fbfc", c100="#c1f8fa", c200="#a2f4f8", c300="#83f1f6", c400="#64edf4", c500="#00f0ff", c600="#00c0cc", c700="#009099", c800="#006066", c900="#003033", c950="#00181a"),
        secondary_hue=gr.themes.Color(c50="#f1e0ff", c100="#e3c1ff", c200="#d5a2ff", c300="#c783ff", c400="#b964ff", c500="#7000ff", c600="#5a00cc", c700="#430099", c800="#2d0066", c900="#160033", c950="#0b001a"),
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Outfit"), "sans-serif"]
    ).set(
        body_background_fill="#07080d",
        body_background_fill_dark="#07080d",
        body_text_color="#f0f4ff",
        background_fill_primary="#07080d",
        background_fill_secondary="#07080d",
        border_color_accent="rgba(0, 240, 255, 0.3)",
        border_color_primary="rgba(255, 255, 255, 0.07)",
        block_background_fill="rgba(255, 255, 255, 0.025)",
        block_border_width="1px",
        block_border_color="rgba(255, 255, 255, 0.07)",
        block_radius="16px",
        button_primary_background_fill="linear-gradient(135deg, #7000ff, #3b00ff)",
        button_primary_background_fill_hover="linear-gradient(135deg, #00f0ff, #7000ff)",
        button_primary_text_color="white",
        slider_color="#00f0ff",
        panel_background_fill="rgba(255, 255, 255, 0.025)",
        block_title_text_color="#00f0ff",
        block_label_text_color="#f0f4ff",
    )
    
    with gr.Blocks(theme=hawkeye_theme, title="Pothole Detection UI") as demo:
        gr.Markdown("# Pothole Intelligence Console")
        gr.Markdown("Real-Time Pothole Detection with YOLOv8. Upload a dashcam video to detect potholes and retrieve a comprehensive dataset.")
        
        with gr.Row():
            with gr.Column():
                video_input = gr.File(label="Upload Dashcam Video (.avi, .mp4, .mkv, etc.)", file_types=["video"])
                conf_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.50, step=0.05, label="Confidence Threshold (0.50 recommended)")
                
                with gr.Accordion("Advanced AI & Vision Enhancements", open=False):
                    gr.Markdown("Enable Test-Time Augmentation (TTA). The AI will process augmented versions of the frame to drastically reduce false positives (shadows, patches) at the cost of slight processing speed.")
                    enable_augmentation = gr.Checkbox(label="Enable AI Augmentation (Maximum Precision)", value=True)
                    gr.Markdown("Uses OpenCV Computer Vision to filter noise. It ignores tiny microscopic cracks, as well as bright, elongated white lane markings.")
                    filter_white_lines = gr.Checkbox(label="Enable OpenCV Noise & White Line Filters", value=True)
                    
                with gr.Accordion("Output Options", open=False):
                    save_csv = gr.Checkbox(label="Save Detections to CSV", value=True)
                    save_screenshots = gr.Checkbox(label="Capture Screenshots of Potholes", value=True)

                process_btn = gr.Button("Detect Potholes", variant="primary")
                
            with gr.Column():
                summary_output = gr.Markdown(label="Detection Summary")
                video_output = gr.Video(label="Processed Video")
                with gr.Row():
                    csv_output = gr.File(label="Detections CSV")
                    zip_output = gr.File(label="Screenshots ZIP")
                
        process_btn.click(
            fn=process_video,
            inputs=[video_input, conf_slider, enable_augmentation, filter_white_lines, save_csv, save_screenshots],
            outputs=[video_output, csv_output, zip_output, summary_output]
        )
        
    return demo
