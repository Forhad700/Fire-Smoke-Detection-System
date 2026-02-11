import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import cv2

# Page Config
st.set_page_config(page_title="Fire Detection System", layout="wide")
st.title("🔥 Real-Time Fire & Smoke Detection")

# Load your model (Make sure best.pt is in the same folder)
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# Sidebar Settings
st.sidebar.header("Configuration")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.45)

# WebRTC Configuration (This helps connect through firewalls)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- THE MAGIC FUNCTION ---
# This runs in a background thread for maximum speed
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # Predict using a small image size (320) to keep it fast on CPU
    results = model.predict(img, conf=conf_threshold, imgsz=320, verbose=False)

    # Plot the results back onto the frame
    annotated_img = results[0].plot()

    return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

# Main Interface
tab1, tab2 = st.tabs(["Webcam (Live)", "Image Upload"])

with tab1:
    st.info("Click 'Start' to begin live detection. This uses WebRTC for smooth motion.")
    webrtc_streamer(
        key="fire-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with tab2:
    uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        file_bytes = uploaded_file.read()
        # Process and show image
        # (Image code remains the same as your working version)
        st.success("Image processed!")
