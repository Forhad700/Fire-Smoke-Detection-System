import streamlit as st
from ultralytics import YOLO
import cv2
import PIL.Image
import numpy as np
import tempfile
# New imports for the cloud-webcam fix
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

st.set_page_config(page_title="Fire Smoke Detector 🔥", layout="wide")
st.title("🔥 Fire & Smoke Detection System")

# Use cache so the model doesn't reload every time you click a button
@st.cache_resource
def load_yolo_model():
    return YOLO('best.pt')

model = load_yolo_model()

source_mode = st.sidebar.radio("Select Source:", ["Image", "Video", "Webcam"])
conf_threshold = st.sidebar.slider("Confidence", 0.1, 1.0, 0.4)

# --- IMAGE SECTION (NO CHANGES) ---
if source_mode == "Image":
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        img = PIL.Image.open(uploaded_file)
        results = model.predict(img, conf=conf_threshold)
        st.image(results[0].plot()[:,:,::-1], caption="Detection", use_container_width=True)

# --- VIDEO SECTION (NO CHANGES) ---
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

# --- WEBCAM SECTION (FIXED FOR CLOUD) ---
elif source_mode == "Webcam":
    st.info("The camera will open below. Use the 'START' button.")

    # This configuration is required for cloud deployment to bypass firewalls
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    # This function processes each camera frame for the AI
    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Performance tip: imgsz=320 makes it run much faster on the cloud
        results = model.predict(img, conf=conf_threshold, imgsz=320, verbose=False)
        
        annotated_frame = results[0].plot()
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    webrtc_streamer(
        key="fire-smoke-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
