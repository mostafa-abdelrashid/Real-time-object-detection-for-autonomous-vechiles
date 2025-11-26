import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
from PIL import Image
import numpy as np

# Set page config
st.set_page_config(
    page_title="Autonomous Vehicle Detector",
    page_icon="🚗",
    layout="wide"
)

# Title and description
st.title("🚗 Autonomous Vehicle Object Detection")
st.markdown("Real-time object detection for autonomous vehicles using YOLO11")

# Sidebar for model info
st.sidebar.header("Model Information")
st.sidebar.metric("Model", "YOLO11s")
st.sidebar.metric("Dataset", "KITTI")
st.sidebar.metric("mAP50", "89.8%")

# Main options
option = st.radio(
    "Choose detection mode:",
    ["Webcam Live Detection", "Upload Video File", "Upload Image"]
)

# Load model once
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

if option == "Webcam Live Detection":
    st.header("🎥 Live Webcam Detection")
    
    # Webcam access
    run_webcam = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.image([])
    
    if run_webcam:
        cap = cv2.VideoCapture(0)
        
        while run_webcam:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam")
                break
                
            # Run detection
            results = model(frame, conf=0.5, verbose=False)
            annotated_frame = results[0].plot()
            
            # Convert BGR to RGB for Streamlit
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(annotated_frame_rgb)
            
        cap.release()

elif option == "Upload Video File":
    st.header("🎬 Video File Detection")
    
    uploaded_video = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
    
    if uploaded_video is not None:
        # Save uploaded video to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_video.read())
            video_path = tmp_file.name
        
        # Process video
        if st.button("Process Video"):
            st.info("Processing video... This may take a few minutes.")
            
            cap = cv2.VideoCapture(video_path)
            success = True
            
            # Create placeholder for video frames
            video_placeholder = st.empty()
            
            while success:
                success, frame = cap.read()
                if not success:
                    break
                
                # Run detection
                results = model(frame, conf=0.5, verbose=False)
                annotated_frame = results[0].plot()
                
                # Convert for display
                display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(display_frame)
            
            cap.release()
            os.unlink(video_path)  # Clean up temp file
            
            st.success("Video processing completed!")

elif option == "Upload Image":
    st.header("🖼️ Image Detection")
    
    uploaded_image = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_image is not None:
        # Load image
        image = Image.open(uploaded_image)
        image_np = np.array(image)
        
        # Run detection
        results = model(image_np, conf=0.5, verbose=False)
        annotated_image = results[0].plot()
        
        # Convert for display
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        
        # Show results
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
        with col2:
            st.subheader("Detected Objects")
            st.image(annotated_image_rgb, use_column_width=True)
        
        # Show detection stats
        if len(results) > 0:
            num_detections = len(results[0].boxes)
            st.info(f"🎯 Detected {num_detections} objects")

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