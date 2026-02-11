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
    st.info("Point camera at fire/smoke source. Detection updates every 0.5s.")
    
    # --- THE SPEED FIX ---
    # debounce=500 means "wait 500ms before sending the next frame"
    # This prevents the Cloud CPU from crashing/lagging.
    image = camera_input_live(debounce=500, show_controls=False)
    
    if image:
        img = PIL.Image.open(image)
        img_array = np.array(img)
        
        # imgsz=160 or 320 makes the AI math 4-8 times faster on CPU
        results = model.predict(img_array, conf=conf_threshold, imgsz=256, verbose=False)
        
        st.image(results[0].plot(), caption="Live Cloud Detection", use_container_width=True)
