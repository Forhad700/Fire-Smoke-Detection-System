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
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        
        # Start button to prevent auto-running
        if st.button("🚀 Fast Analysis"):
            vf = cv2.VideoCapture(tfile.name)
            st_frame = st.empty()
            
            # --- THE SPEED TWEAKS ---
            # Increase this number (e.g., 10 or 15) to make it even faster
            SKIP_FRAMES = 10 
            count = 0
            
            while vf.isOpened():
                ret, frame = vf.read()
                if not ret: break
                
                # Only process one out of every 10 frames
                if count % SKIP_FRAMES == 0:
                    # imgsz=160 is TINY and UGLY, but it is the FASTEST possible
                    results = model.predict(frame, conf=conf_threshold, imgsz=160, verbose=False)
                    
                    # Display result
                    res_plotted = results[0].plot()
                    st_frame.image(res_plotted, channels="BGR", use_container_width=True)
                
                count += 1
            
            vf.release()
            st.success("Done!")


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




