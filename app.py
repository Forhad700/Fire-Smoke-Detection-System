import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import cv2
import tempfile
import PIL.Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="Fire & Smoke Detection System", layout="wide")
st.title("🔥 Fire & Smoke Detection System")

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    # Ensure best.pt is in the same folder on GitHub
    return YOLO("best.pt")

model = load_model()

# --- SIDEBAR CONFIG ---
st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.45)
st.sidebar.info("Webcam uses WebRTC for smooth motion. Video files may process slower on free CPU.")

# WebRTC Configuration for Cloud Connectivity
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- WEBCAM CALLBACK ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    # Small imgsz=320 is critical for CPU speed
    results = model.predict(img, conf=conf_threshold, imgsz=320, verbose=False)
    annotated_img = results[0].plot()
    return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

# --- MAIN INTERFACE (TABS) ---
tab1, tab2, tab3 = st.tabs(["🎥 Live Webcam", "🖼️ Image Upload", "📁 Video File"])

# TAB 1: WEBCAM
with tab1:
    st.subheader("Live Detection")
    webrtc_streamer(
        key="fire-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# TAB 2: IMAGE
with tab2:
    st.subheader("Analyze an Image")
    uploaded_image = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'], key="img_up")
    if uploaded_image:
        img = PIL.Image.open(uploaded_image)
        results = model.predict(img, conf=conf_threshold)
        st.image(results[0].plot()[:,:,::-1], caption="Detection Result", use_container_width=True)

# TAB 3: VIDEO FILE
with tab3:
    st.subheader("Analyze Video File")
    uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'mov', 'avi'], key="vid_up")
    
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        
        if st.button("Start Video Analysis"):
            vf = cv2.VideoCapture(tfile.name)
            st_frame = st.empty()
            
            while vf.isOpened():
                ret, frame = vf.read()
                if not ret: break
                
                # We skip every 5 frames to make it bearable on Cloud CPU
                results = model.predict(frame, conf=conf_threshold, imgsz=320, verbose=False)
                st_frame.image(results[0].plot(), channels="BGR", use_container_width=True)
            vf.release()
            st.success("Analysis Finished!")
