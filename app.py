import streamlit as st
from ultralytics import YOLO
import cv2
import PIL.Image
import numpy as np
import tempfile
import os


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
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        output_path = "processed_video.mp4"
        if st.button("Start Analysis"):
            vf = cv2.VideoCapture(tfile.name)
            width = int(vf.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vf.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(vf.get(cv2.CAP_PROP_FPS))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            progress_bar = st.progress(0)
            status_text = st.empty()
            total_frames = int(vf.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0
            while vf.isOpened():
                ret, frame = vf.read()
                if not ret:
                    break
                results = model.predict(frame, conf=conf_threshold, imgsz=320, verbose=False)
                out.write(results[0].plot())
                frame_count += 1
                progress_bar.progress(frame_count / total_frames)
                status_text.text(f"Processing frame {frame_count}/{total_frames}...")
            vf.release()
            out.release()
            st.success("Analysis Complete! Viewing smooth playback below:")
            video_file = open(output_path, 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)


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


