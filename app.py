import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import cv2
import tempfile
import PIL.Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="Fire & Smoke Detector", layout="wide")
st.title("🔥 Professional Fire & Smoke Detection")

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    # TIP: If you export your model to OpenVINO locally first, 
    # use YOLO('best_openvino_model/') for 3x more speed!
    return YOLO("best.pt") 

model = load_model()

# --- SIDEBAR ---
st.sidebar.header("Control Panel")
conf_threshold = st.sidebar.slider("Confidence", 0.1, 1.0, 0.45)

# WebRTC Config (Essential for Cloud deployment)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- WEBCAM LOGIC (WebRTC) ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    # imgsz=320 is the "Secret Sauce" for CPU speed
    results = model.predict(img, conf=conf_threshold, imgsz=320, verbose=False)
    return av.VideoFrame.from_ndarray(results[0].plot(), format="bgr24")

# --- MAIN TABS ---
tab1, tab2, tab3 = st.tabs(["🎥 Live Webcam", "🖼️ Image Upload", "📁 Video File"])

with tab1:
    st.info("Webcam mode uses WebRTC for smooth real-time motion on Cloud.")
    webrtc_streamer(
        key="fire-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with tab2:
    uploaded_image = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
    if uploaded_image:
        img = PIL.Image.open(uploaded_image)
        results = model.predict(img, conf=conf_threshold)
        st.image(results[0].plot()[:,:,::-1], use_container_width=True)

with tab3:
    uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'mov'])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        
        if st.button("🚀 Fast Analysis"):
            vf = cv2.VideoCapture(tfile.name)
            st_frame = st.empty()
            
            count = 0
            while vf.isOpened():
                ret, frame = vf.read()
                if not ret: break
                
                # SKIP FRAMES: Only process every 5th frame to respect the recruiter's time
                if count % 5 == 0:
                    results = model.predict(frame, conf=conf_threshold, imgsz=320, verbose=False)
                    st_frame.image(results[0].plot(), channels="BGR", use_container_width=True)
                count += 1
            vf.release()
