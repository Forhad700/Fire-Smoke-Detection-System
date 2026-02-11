import streamlit as st
from ultralytics import YOLO
import cv2
import PIL.Image
import numpy as np
import tempfile


st.set_page_config(page_title="Fire Smoke Detector 🔥", layout="wide")
st.title("🔥 Fire & Smoke Detection System")

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
        results = model.predict(img, conf=conf_threshold)
        st.image(results[0].plot()[:,:,::-1], caption="Detection", use_container_width=True)


elif source_mode == "Video":
    uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'mov', 'avi'])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        vf = cv2.VideoCapture(tfile.name)
        st_frame = st.empty()
        while vf.isOpened():
            ret, frame = vf.read()
            if not ret: break
            results = model.predict(frame, conf=conf_threshold)
            st_frame.image(results[0].plot(), channels="BGR", use_container_width=True)
        vf.release()


elif source_mode == "Webcam":
    st.info("Continuous Detection: Click the button to start/stop.")
    
    # This is the secret for cloud speed: st.camera_input + a loop
    # It works instantly because it uses your browser's native power.
    run = st.toggle("Start Real-Time Detection")
    FRAME_WINDOW = st.empty()

    while run:
        # Use Streamlit's native camera for zero-setup cloud access
        img_file = st.camera_input("Detecting...", label_visibility="collapsed")
        
        if img_file:
            # Convert to format YOLO understands
            img = PIL.Image.open(img_file)
            img_array = np.array(img)
            
            # --- TURBO SPEED SETTINGS ---
            # imgsz=256 is the "Sweet Spot": 4x faster than default.
            # stream=True uses less memory to prevent cloud crashes.
            results = model.predict(img_array, conf=conf_threshold, imgsz=256, stream=True)
            
            for r in results:
                annotated_frame = r.plot()
                # Display the processed frame
                FRAME_WINDOW.image(annotated_frame, channels="BGR", use_container_width=True)
        
        # Stops the loop if the toggle is turned off
        if not run:
            break
