import streamlit as st
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# ---------------- CLASS NAMES ----------------
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
st.set_page_config(page_title="YOLOv5 Live Detection", layout="wide")
st.title("🔥 YOLOv5 ONNX Object Detection")
st.markdown("Live + Photo Detection Modes")

# ---------------- CONFIDENCE SLIDER ----------------
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    0.0, 1.0, 0.3, 0.05
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return ort.InferenceSession(
        "best.onnx",
        providers=["CPUExecutionProvider"]
    )

session = load_model()

# ---------------- DETECTION FUNCTION ----------------
def detect(image):
    img_h, img_w = image.shape[:2]

    img_resized = cv2.resize(image, (640, 640))
    img_resized = img_resized.transpose(2, 0, 1)
    img_resized = np.expand_dims(img_resized, axis=0).astype(np.float32)
    img_resized /= 255.0

    outputs = session.run(
        None,
        {session.get_inputs()[0].name: img_resized}
    )

    pred = outputs[0]

    if pred.shape[1] == 85:
        pred = np.transpose(pred, (0, 2, 1))

    pred = pred[0]

    boxes = []
    scores = []
    class_ids = []

    for det in pred:
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

                cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(
                    image,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2
                )

    return image

# ---------------- MODE SELECTION ----------------
mode = st.radio("Select Mode:", ["📸 Capture Image", "🎥 Live Webcam"])

# ---------------- IMAGE MODE ----------------
if mode == "📸 Capture Image":
    camera_image = st.camera_input("Capture Image")

    if camera_image is not None:
        image = Image.open(camera_image)
        img_np = np.array(image)

        result = detect(img_np.copy())

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original")
        with col2:
            st.image(result, caption="Detection Result")

# ---------------- LIVE MODE ----------------
elif mode == "🎥 Live Webcam":

    class VideoProcessor(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            result = detect(img)
            return result

    webrtc_streamer(
        key="live",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )
