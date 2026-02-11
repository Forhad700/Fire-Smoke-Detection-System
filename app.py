import streamlit as st
from ultralytics import YOLO
import cv2
import PIL.Image
import numpy as np
import tempfile

st.set_page_config(page_title="Fire Smoke Detector 🔥", layout="wide")
st.title("🔥 Fire & Smoke Detection System")

# Cache the model to prevent reloading on every click
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

source_mode = st.sidebar.radio("Select Source:", ["Image", "Video", "Webcam"])
conf_threshold = st.sidebar.slider("Confidence", 0.1, 1.0, 0.4)

if source_mode == "Image":
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        img = PIL.Image.open(uploaded_file)
        # Standard prediction for static images
        results = model.predict(img, conf=conf_threshold)
        st.image(results[0].plot()[:,:,::-1], caption="Detection", use_container_width=True)

elif source_mode == "Video":
    uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'mov', 'avi'])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        vf = cv2.VideoCapture(tfile.name)
        st_frame = st.empty()
        
        frame_count = 0
        while vf.isOpened():
            ret, frame = vf.read()
            if not ret: break
            
            # --- SPEED OPTIMIZATION ---
            # Only process every 3rd frame to stop "Slow Motion"
            if frame_count % 3 == 0:
                # imgsz=320 is critical for CPU speed
                results = model.predict(frame, conf=conf_threshold, imgsz=320, verbose=False)
                st_frame.image(results[0].plot(), channels="BGR", use_container_width=True)
            
            frame_count += 1
        vf.release()

elif source_mode == "Webcam":
    st.info("Click 'Stop' at Top Right to Turn Off Camera.")
    cap = cv2.VideoCapture(0) 
    st_frame = st.empty()
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Process every 2nd frame for Webcam to reduce lag
        if frame_count % 2 == 0:
            results = model.predict(frame, conf=conf_threshold, imgsz=320, verbose=False)
            st_frame.image(results[0].plot(), channels="BGR", use_container_width=True)
        
        frame_count += 1
    cap.release()
