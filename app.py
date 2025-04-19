import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

# Load the YOLO model
model = YOLO("./weights/best.pt")

# Streamlit app
st.title("YOLO Object Detection")
st.write("Upload an image to detect objects using the YOLO model.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Confidence slider
conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")
    
    # Save the uploaded file to a temp directory
    input_image_path = f"./temp/{uploaded_file.name}"
    os.makedirs("./temp", exist_ok=True)
    with open(input_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Run YOLO prediction
    results = model.predict(source=input_image_path, conf=conf_threshold, save=True, save_txt=False)
    print(results)

    # Get the output image path
    output_dir = results[0].save_dir
    output_image_path = os.path.join(output_dir, uploaded_file.name)

    # Display the result
    st.image(output_image_path, caption="Detected Image", use_column_width=True)
    st.write(f"Detection completed! Saved at: {output_image_path}")

    # Cleanup
    os.remove(input_image_path)
