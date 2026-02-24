import streamlit as st
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image

class_names = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="YOLOv5 ONNX Detection", layout="wide")

st.title("🔥 YOLOv5 Webcam Object Detection (ONNX)")
st.markdown("Deployed using ONNX Runtime on Streamlit Cloud")

# ---------------- CONFIDENCE SLIDER ----------------
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    0.0, 1.0, 0.3, 0.05
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    session = ort.InferenceSession(
        "best.onnx",
        providers=["CPUExecutionProvider"]
    )
    return session

session = load_model()

# ---------------- POSTPROCESS FUNCTION ----------------
def postprocess(predictions, original_image):
    boxes = []
    scores = []
    class_ids = []

    pred = predictions[0]

    # If shape is (1, 85, 25200), transpose it
    if pred.shape[1] == 85:
        pred = np.transpose(pred, (0, 2, 1))

    pred = pred[0]  # remove batch

    img_h, img_w = original_image.shape[:2]

    for det in pred:
        if len(det) < 6:
            continue

        obj_conf = det[4]
        class_scores = det[5:]

        if len(class_scores) == 0:
            continue

        class_id = np.argmax(class_scores)
        confidence = obj_conf * class_scores[class_id]

        if confidence > confidence_threshold:
            x, y, w, h = det[0:4]

            x = x * img_w / 640
            y = y * img_h / 640
            w = w * img_w / 640
            h = h * img_h / 640

            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(float(confidence))
            class_ids.append(class_id)

    if len(boxes) > 0:
        indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_threshold, 0.4)

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                label = f"{class_names[class_ids[i]]}: {scores[i]:.2f}"

                cv2.rectangle(original_image, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(
                    original_image,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,255,0),
                    2
                )

    return original_image

# ---------------- WEBCAM INPUT ----------------
camera_image = st.camera_input("Capture Image")

if camera_image is not None:
    image = Image.open(camera_image)
    img = np.array(image)

    # Resize to YOLO input size
    img_resized = cv2.resize(img, (640, 640))

    # Convert HWC → CHW
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

    # Postprocess & draw boxes
    result_img = postprocess(outputs, img.copy())

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original Image")

    with col2:
        st.image(result_img, caption="Detection Result")
