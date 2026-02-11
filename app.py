import streamlit as st
from ultralytics import YOLO
import cv2
import PIL.Image
import numpy as np
import tempfile


st.set_page_config(page_title="Fire Smoke Detector 🔥", layout="wide")
st.title("🔥 Fire & Smoke Detection System")

model = YOLO('best.pt')

source_mode = st.sidebar.radio("Select Source:", ["Image", "Video", "Webcam"])
conf_threshold = st.sidebar.slider("Confidence", 0.1, 1.0, 0.4)


if source_mode == "Image":
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        img = PIL.Image.open(uploaded_file)
        results = model.predict(img, conf=conf_threshold)
        st.image(results[0].plot()[:,:,::-1], caption="Detection", use_container_width=True)


elif source_mode == "Video":
    uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'mov', 'avi'])
    
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        
        if st.button("🚀 Start Fast Analysis"):
            vf = cv2.VideoCapture(tfile.name)
            
            # Video Properties
            width = int(vf.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vf.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(vf.get(cv2.CAP_PROP_FPS))
            total_frames = int(vf.get(cv2.CAP_PROP_FRAME_COUNT))

            # --- THE SPEED FIX: STRIDE ---
            # Set to 2 (skip every other frame) or 3 (skip two frames)
            stride = 2 
            new_fps = fps // stride
            
            output_path = "output.webm"
            fourcc = cv2.VideoWriter_fourcc(*'VP80')
            out = cv2.VideoWriter(output_path, fourcc, new_fps, (width, height))
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            frame_count = 0
            while vf.isOpened():
                ret, frame = vf.read()
                if not ret: break
                
                # Only process the frame if it matches our stride
                if frame_count % stride == 0:
                    # imgsz=320 makes the AI math 4x faster than default 640
                    results = model.predict(frame, conf=conf_threshold, imgsz=320, verbose=False)
                    out.write(results[0].plot())
                
                frame_count += 1
                if frame_count % 20 == 0:
                    progress_bar.progress(min(frame_count / total_frames, 1.0))
                    status_text.text(f"Analyzing: {frame_count}/{total_frames} frames...")

            vf.release()
            out.release()
            
            with open(output_path, "rb") as f:
                st.video(f.read())
            st.success(f"Done! Processed at {stride}x speed.")


elif source_mode == "Webcam":
    st.info("Click 'Stop' at Top Right to Turn Off Camera.")
    cap = cv2.VideoCapture(0) 
    st_frame = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        results = model.predict(frame, conf=conf_threshold)
        st_frame.image(results[0].plot(), channels="BGR", use_container_width=True)
    cap.release()

