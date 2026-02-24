from ultralytics import YOLO
import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(page_title="Live YOLO Detection")

st.title("🔥 Live Webcam Object Detection")

@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    return model

model = load_model()

camera_image = st.camera_input("Capture Image")

if camera_image is not None:
    image = Image.open(camera_image)

    results = model(image)

    result_image = results[0].plot()

    st.image(result_image, caption="Detected Objects")