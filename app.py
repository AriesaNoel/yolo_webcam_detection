import streamlit as st
import torch
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(page_title="Live YOLOv5 Detection")

st.title("🔥 Live Webcam Object Detection")
st.write("Click below to capture image from webcam")

# Load model only once
@st.cache_resource
def load_model():
    model = torch.hub.load(
        'ultralytics/yolov5',
        'custom',
        path='best.pt',
        force_reload=False
    )
    return model

model = load_model()

# Open webcam
camera_image = st.camera_input("Open Camera")

if camera_image is not None:
    # Convert to PIL image
    image = Image.open(camera_image)

    st.image(image, caption="Captured Image")

    # Run detection
    results = model(image)

    # Render results
    results.render()

    # Convert result to image
    detected_image = Image.fromarray(results.ims[0])

    st.image(detected_image, caption="Detected Objects")