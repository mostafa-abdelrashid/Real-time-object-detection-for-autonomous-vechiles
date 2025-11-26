import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
from PIL import Image
import numpy as np
import mlflow
import mlflow.pytorch
import time

# Set page config
st.set_page_config(page_title="Autonomous Vehicle Detector", page_icon="🚗", layout="wide")

# Title
st.title("🚗 Autonomous Vehicle Object Detection")
st.markdown("Real-time object detection with MLflow experiment tracking")

# MLflow Tracking Section
st.sidebar.header("🔬 MLflow Experiment Tracking")

if st.sidebar.button("Start New Experiment Run"):
    with mlflow.start_run():
        st.sidebar.success("🎯 MLflow run started!")
        mlflow.log_param("app_mode", "streamlit_demo")
        mlflow.log_param("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))

# Load model with MLflow logging
@st.cache_resource
def load_model():
    with mlflow.start_run(nested=True):
        model = YOLO('best.pt')
        mlflow.log_param("model_loaded", "YOLO11s_KITTI")
        return model

model = load_model()

# FIXED: Universal detection function that handles ANY image format
def run_detection_with_tracking(frame, detection_type):
    start_time = time.time()
    
    # UNIVERSAL FIX: Handle any image format and ensure 3 channels (RGB)
    if len(frame.shape) == 3:  # Color image
        if frame.shape[2] == 4:  # RGBA (4 channels) - PNG with transparency
            # Remove alpha channel - convert RGBA to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        elif frame.shape[2] == 1:  # Grayscale (1 channel)
            # Convert grayscale to RGB by duplicating channels
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        # If already 3 channels (RGB), no conversion needed
    else:  # Grayscale (2D array)
        # Convert 2D grayscale to 3D RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    
    # Ensure the frame is in the correct data type (uint8)
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    
    # Run detection
    results = model(frame, conf=0.5, verbose=False)
    annotated_frame = results[0].plot()
    
    # Calculate performance metrics
    inference_time = time.time() - start_time
    fps = 1.0 / inference_time if inference_time > 0 else 0
    num_detections = len(results[0].boxes) if len(results) > 0 else 0
    
    # Log to MLflow
    try:
        mlflow.log_metrics({
            f'{detection_type}_fps': fps,
            f'{detection_type}_detections': num_detections,
            f'{detection_type}_inference_time': inference_time
        })
    except:
        pass  # MLflow not required
    
    return annotated_frame, fps, num_detections

# FIXED: Universal image preprocessing function
def preprocess_image(uploaded_image):
    """Convert any image format to standard 3-channel RGB numpy array"""
    # Load image
    image = Image.open(uploaded_image)
    
    # Convert to RGB (3 channels) - handles PNG, JPG, JPEG, etc.
    if image.mode in ('RGBA', 'LA', 'P'):
        # Create a white background for transparent PNGs
        background = Image.new('RGB', image.size, (255, 255, 255))
        if image.mode == 'RGBA':
            # Paste the image onto white background
            background.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
            image = background
        else:
            image = image.convert('RGB')
    elif image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array
    image_np = np.array(image)
    
    # Final safety check - ensure 3 channels
    if len(image_np.shape) == 3 and image_np.shape[2] == 4:
        image_np = image_np[:, :, :3]  # Remove alpha channel
    
    return image_np, image

# Detection options
option = st.radio("Choose detection mode:", 
                 ["Webcam Live Detection", "Upload Video File", "Upload Image"])

if option == "Webcam Live Detection":
    st.header("🎥 Live Webcam Detection")
    
    run_webcam = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.image([])
    stats_placeholder = st.empty()
    
    if run_webcam:
        cap = cv2.VideoCapture(0)
        fps_history = []
        detection_history = []
        
        while run_webcam:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam")
                break
                
            annotated_frame, fps, detections = run_detection_with_tracking(frame, "webcam")
            fps_history.append(fps)
            detection_history.append(detections)
            
            # Keep only last 30 readings
            if len(fps_history) > 30:
                fps_history.pop(0)
            if len(detection_history) > 30:
                detection_history.pop(0)
                
            avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0
            avg_detections = sum(detection_history) / len(detection_history) if detection_history else 0
            
            # Convert for display
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(annotated_frame_rgb)
            
            # Show stats
            stats_placeholder.metric("📊 Current FPS", f"{avg_fps:.1f}")
            stats_placeholder.metric("🎯 Detections", f"{avg_detections:.1f}")
            
        cap.release()

elif option == "Upload Video File":
    st.header("🎬 Video File Detection")
    
    uploaded_video = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
    
    if uploaded_video is not None:
        # Save uploaded video
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_video.read())
            video_path = tmp_file.name
        
        if st.button("Process Video with MLflow Tracking"):
            # Start MLflow run for video processing
            with mlflow.start_run(run_name="video_processing"):
                mlflow.log_param("video_file", uploaded_video.name)
                
                st.info("Processing video with MLflow tracking...")
                
                cap = cv2.VideoCapture(video_path)
                total_frames = 0
                total_detections = 0
                start_time = time.time()
                
                video_placeholder = st.empty()
                progress_bar = st.progress(0)
                
                while True:
                    success, frame = cap.read()
                    if not success:
                        break
                    
                    annotated_frame, fps, detections = run_detection_with_tracking(frame, "video")
                    total_frames += 1
                    total_detections += detections
                    
                    # Update progress
                    if total_frames % 10 == 0:
                        progress = min(total_frames / 300, 1.0)  # Assume max 300 frames for demo
                        progress_bar.progress(progress)
                    
                    display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(display_frame)
                
                processing_time = time.time() - start_time
                
                # Log final metrics
                mlflow.log_metrics({
                    'total_video_frames': total_frames,
                    'total_video_detections': total_detections,
                    'video_processing_time': processing_time,
                    'avg_detections_per_frame': total_detections / total_frames if total_frames > 0 else 0
                })
                
                cap.release()
                os.unlink(video_path)
                
                st.success(f"✅ Video processing completed! Processed {total_frames} frames")

elif option == "Upload Image":
    st.header("🖼️ Image Detection - UNIVERSAL FORMAT SUPPORT")
    
    uploaded_image = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'])
    
    if uploaded_image is not None:
        with mlflow.start_run(nested=True, run_name="image_detection"):
            mlflow.log_param("image_file", uploaded_image.name)
            mlflow.log_param("image_format", uploaded_image.type)
            
            # FIXED: Use universal image preprocessing
            image_np, original_image = preprocess_image(uploaded_image)
            
            st.success(f"✅ Image loaded successfully: {uploaded_image.name} ({original_image.mode} mode)")
            
            annotated_image, fps, detections = run_detection_with_tracking(image_np, "image")
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(original_image, use_container_width=True)
                st.caption(f"Format: {uploaded_image.type} | Mode: {original_image.mode}")
            with col2:
                st.subheader("Detected Objects")
                st.image(annotated_image_rgb, use_container_width=True)
                st.caption(f"Processed: 3-channel RGB | Detections: {detections}")
            
            st.metric("🎯 Objects Detected", detections)
            st.metric("⚡ Inference FPS", f"{fps:.1f}")
            
            mlflow.log_metric("image_detections", detections)
            mlflow.log_metric("image_inference_fps", fps)

# MLflow UI Access
st.sidebar.markdown("---")
st.sidebar.header("MLflow UI")
if st.sidebar.button("Launch MLflow UI"):
    st.sidebar.info("Run in terminal: mlflow ui --port 5000 --host 127.0.0.1")
    st.sidebar.markdown("Then visit: http://localhost:5000")

st.sidebar.markdown("""
### 📊 Tracked Metrics:
- FPS (Frames per second)
- Detection counts  
- Inference times
- Processing performance

### 🖼️ Supported Image Formats:
- ✅ JPG, JPEG (3 channels)
- ✅ PNG (4 channels with transparency)
- ✅ BMP, TIFF
- ✅ WebP
- ✅ Grayscale images
""")

# Footer
st.markdown("---")
st.markdown("### 🚦 Detection Classes:")
st.write("""
- **🚗 Car** - Passenger vehicles
- **🚐 Van** - Commercial vans  
- **🚚 Truck** - Large trucks
- **🚶 Pedestrian** - People walking
- **🚴 Cyclist** - Bicycle riders
- **🚊 Tram** - Light rail vehicles
- **🚜 Misc** - Other vehicles
""")