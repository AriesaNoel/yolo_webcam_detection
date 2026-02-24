st.write("NEW VERSION DEPLOYED ✅")

import streamlit as st
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="YOLOv5 ONNX Detection")
st.title("🔥 YOLOv5 Webcam Object Detection (ONNX)")

@st.cache_resource
def load_model():
    session = ort.InferenceSession("best.onnx")
    return session

session = load_model()

camera_image = st.camera_input("Capture Image")

if camera_image is not None:
    image = Image.open(camera_image)
    img = np.array(image)

    # Resize to YOLO input size
    img_resized = cv2.resize(img, (640, 640))

    # Convert HWC to CHW
    img_resized = img_resized.transpose(2, 0, 1)

    # Add batch dimension
    img_resized = np.expand_dims(img_resized, axis=0).astype(np.float32)

    # Normalize
    img_resized /= 255.0

    # Run inference
    outputs = session.run(
        None,
        {session.get_inputs()[0].name: img_resized}
    )

    st.image(image, caption="Image Captured")
    st.success("Model ran successfully! 🎉")
