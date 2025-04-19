import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import os
import shutil
import time
from streamlit_webrtc import webrtc_streamer
import av
import datetime

# Load the YOLO model
model = YOLO("./weights/best.pt")

# Streamlit app
st.title("YOLO Object Detection")
st.write("Choose an input method: Image, Video, or Live Camera")

# Selection for mode
mode = st.radio("Select Mode", ["Image", "Video", "Live Camera"], index=0)

# Confidence slider
conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)

if mode == "Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        st.write("Processing...")

        input_image_path = f"./temp/{uploaded_file.name}"
        os.makedirs("./temp", exist_ok=True)
        with open(input_image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        results = model.predict(source=input_image_path, conf=conf_threshold, save=True, save_txt=False)
        output_dir = results[0].save_dir
        output_image_path = os.path.join(output_dir, uploaded_file.name)
        st.image(output_image_path, caption="Detected Image", use_column_width=True)
        st.write(f"Detection completed! Saved at: {output_image_path}")
        os.remove(input_image_path)

elif mode == "Video":
    uploaded_video = st.file_uploader("Upload a video...", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video is not None:
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, uploaded_video.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())
        
        st.video(video_path)
        st.write("Processing Video...")
        
        output_video_path = os.path.join(temp_dir, "output.mp4")
        model.predict(source=video_path, conf=conf_threshold, save=True, save_txt=False)
        
        output_dir = os.path.join("runs", "detect")
        latest_result = sorted(os.listdir(output_dir))[-1]
        final_output_path = os.path.join(output_dir, latest_result, uploaded_video.name)
        
        st.video(final_output_path)
        st.write(f"Detection completed! Saved at: {final_output_path}")
        shutil.rmtree(temp_dir)

elif mode == "Live Camera":
    st.write("Starting Live Camera with real-time object detection for 5 seconds...")

    start_time = time.time()

    # Stop After 5 Seconds And Takes A Screenshot
    def video_frame_callback(frame):

        current_time = time.time()
        elapsed_time = current_time - start_time

        img = frame.to_ndarray(format="bgr24")
        results = model(img, conf=conf_threshold)
        for r in results:
            img = r.plot()

        if elapsed_time > 5:
            # Take a screenshot and save it before stopping
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = f"./Live Camera Results/img_{timestamp}.png"
            cv2.imwrite(screenshot_path, img)
            st.success(f"Screenshot saved at {screenshot_path}")
            # Stop streaming by returning None
            return None

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(key="live-stream", video_frame_callback=video_frame_callback)
