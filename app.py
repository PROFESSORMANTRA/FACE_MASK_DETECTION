# ===========================================
# 😷 Face Mask Detection App (Streamlit + OpenCV DNN)
# Developed by Soham Saykar & Dakshata Wakode
# ===========================================

import streamlit as st
import cv2
import numpy as np
import time
import os
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ---------------------
# 🧩 Page Setup
# ---------------------
st.set_page_config(page_title="😷 Face Mask Detector", page_icon="🧠", layout="centered")
st.title("🧠 Real-time Face Mask Detection")
st.markdown("Upload an image or use your live webcam to detect masks using your trained CNN model.")
st.markdown("---")

# ---------------------
# 📁 File Paths
# ---------------------
MODEL_KERAS = "mask_detector_model.keras"
PROTO_TXT = "deploy.prototxt"
CAFFE_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"

# ---------------------
# ✅ Check Files
# ---------------------
missing_files = []
for f in [MODEL_KERAS, PROTO_TXT, CAFFE_MODEL]:
    if not os.path.exists(f):
        missing_files.append(f)

if missing_files:
    st.error(f"🚨 Missing file(s): {', '.join(missing_files)}. Please add them to your app folder.")
    st.stop()

# ---------------------
# 🧠 Load Models
# ---------------------
@st.cache_resource
def load_models():
    try:
        mask_model = load_model(MODEL_KERAS)
        net = cv2.dnn.readNetFromCaffe(PROTO_TXT, CAFFE_MODEL)

        inp = mask_model.input_shape
        h = int(inp[1]) if inp[1] else 128
        w = int(inp[2]) if inp[2] else 128
        input_size = (w, h)

        return mask_model, net, input_size
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

model, FACE_NET, MODEL_INPUT_SIZE = load_models()

if model is None or FACE_NET is None:
    st.stop()

# 🔁 You can swap these labels if you want the UI inverted.
# Example: {0: "No Mask", 1: "Mask"} will display inverted text (but percentages will still match).
CLASS_LABELS = {0: "No Mask", 1: "Mask"}

# ---------------------
# 🔍 DNN Face Detection
# ---------------------
def detect_faces_dnn(frame):
    """Detect faces using OpenCV DNN and return bounding boxes."""
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 1.0, (300, 300), (104.0, 177.0, 123.0))
    FACE_NET.setInput(blob)
    detections = FACE_NET.forward()
    faces = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # threshold
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            faces.append((x1, y1, x2 - x1, y2 - y1))
    return faces

# ---------------------
# 🧠 Mask Prediction (FIXED: label & confidence consistent)
# ---------------------
def predict_face_mask(face_bgr):
    """
    Predict mask presence on a cropped BGR face image.
    This implementation returns (confidence, label) where `confidence` is
    the probability for the displayed `label` (consistent).
    """
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(face_rgb, MODEL_INPUT_SIZE)
    arr = img_to_array(resized).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr, verbose=0)[0]

    # Interpret outputs robustly for both sigmoid (1) and softmax (2) shapes
    if preds.size == 1:
        # Single sigmoid output -> treat index 0 as "Mask prob"
        mask_prob = float(preds[0])
        no_mask_prob = 1.0 - mask_prob
        # predicted_index = 0 means model favors "Mask"
        predicted_index = 0 if mask_prob >= no_mask_prob else 1
    elif preds.size == 2:
        # Softmax two-class output, assume order [Mask, No Mask]
        mask_prob = float(preds[0])
        no_mask_prob = float(preds[1])
        predicted_index = 0 if mask_prob >= no_mask_prob else 1
    else:
        return 0.0, "ERROR"

    # Use predicted_index to pick UI label and the corresponding probability
    label = CLASS_LABELS.get(predicted_index, "Unknown")
    confidence = mask_prob if predicted_index == 0 else no_mask_prob

    return confidence, label

# ---------------------
# ✏️ Annotate Frame
# ---------------------
def annotate_frame(frame):
    faces = detect_faces_dnn(frame)
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        if face_roi.size == 0:
            continue

        confidence, label = predict_face_mask(face_roi)
        # color follows displayed label (green = Mask, red = No Mask)
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        text = f"{label} ({confidence*100:.1f}%)"
        cv2.putText(frame, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    return frame, len(faces)

# ---------------------
# 🧰 Streamlit UI
# ---------------------
mode = st.radio(
    "Choose Input Source:",
    ["Upload Image", "Live Webcam"],
    horizontal=True
)
st.markdown("---")

st.sidebar.header("Model Information")
st.sidebar.write(f"**Mask Model:** {MODEL_KERAS}")
st.sidebar.write(f"**Input Size:** {MODEL_INPUT_SIZE[0]}x{MODEL_INPUT_SIZE[1]}")
st.sidebar.write(f"**Face Detector:** OpenCV DNN")

# ---------------------
# 🖼️ Upload Image Mode
# ---------------------
if mode == "Upload Image":
    st.subheader("🖼️ Image Detection")
    uploaded = st.file_uploader("Upload image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded:
        try:
            image = Image.open(uploaded).convert("RGB")
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Resize large images for better processing
            max_dim = 800
            if max(image.size) > max_dim:
                scale = max_dim / max(image.size)
                new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
                image = image.resize(new_size)
                frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Image", use_container_width=True)

            if st.button("🔍 Detect Mask"):
                with st.spinner("Detecting faces and masks..."):
                    annotated, faces_found = annotate_frame(frame.copy())
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    with col2:
                        st.image(annotated_rgb, caption=f"Detected {faces_found} face(s)", use_container_width=True)

                    if faces_found == 0:
                        st.warning("No faces detected. Try another image.")
                    else:
                        st.success("Detection complete!")
        except Exception as e:
            st.error(f"Error processing image: {e}")

# ---------------------
# 📹 Live Webcam Mode
# ---------------------
else:
    st.subheader("📹 Live Webcam Detection")
    st.info("Click **Start Camera** to begin detection. Grant browser permission if prompted.")

    if "webcam_running" not in st.session_state:
        st.session_state.webcam_running = False

    cols = st.columns([1, 1, 6])
    start = cols[0].button("▶️ Start Camera")
    stop = cols[1].button("⏹️ Stop Camera")

    if start:
        st.session_state.webcam_running = True
    if stop:
        st.session_state.webcam_running = False

    video_placeholder = st.empty()

    if st.session_state.webcam_running:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("🚨 Could not access webcam.")
            st.session_state.webcam_running = False
        else:
            try:
                while st.session_state.webcam_running:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to read from webcam.")
                        break
                    frame = cv2.flip(frame, 1)
                    annotated, faces_found = annotate_frame(frame)
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(
                        annotated_rgb,
                        caption=f"Live Feed: {faces_found} face(s) detected",
                        use_container_width=True,
                    )
                    time.sleep(0.01)
            except Exception as e:
                st.error(f"Webcam processing error: {e}")
            finally:
                cap.release()
                video_placeholder.info("Webcam feed stopped. Click Start Camera to restart.")
    else:
        video_placeholder.info("Click **Start Camera** to begin live detection.")

# ---------------------
# 🏁 Footer
# ---------------------
st.markdown("---")
st.markdown("*Project:* Real-time Face Mask Detection  \n*Developed by Soham Saykar & Dakshata Wakode*")




# cd FACE_MASK_DETECTOR
# .\venv\Scripts\Activate.ps1
# streamlit run app.py
