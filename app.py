import streamlit as st
from ultralytics import YOLO
import cv2
import PIL.Image
import numpy as np
import tempfile
from camera_input_live import camera_input_live


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
    st.info("Live stream active. Point camera at fire/smoke source.")
    
    # This component captures a frame every few milliseconds automatically
    image = camera_input_live(show_controls=False)
    
    if image:
        # Convert bytes to an image for YOLO
        img = PIL.Image.open(image)
        img_array = np.array(img)
        
        # Run detection (imgsz=320 makes it fast enough for a live web stream)
        results = model.predict(img_array, conf=conf_threshold, imgsz=320, verbose=False)
        
        # Display the detection
        st.image(results[0].plot(), caption="Live Detection", use_container_width=True)
