import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import sys

# ==============================
# PATHS
# ==============================
MODEL_H5_PATH = "mask_detector_model.h5"
MODEL_KERAS_PATH = "mask_detector_model.keras"
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# ==============================
# LOAD MODEL
# ==============================
print("🔍 Loading model...")
if os.path.exists(MODEL_KERAS_PATH):
    model = load_model(MODEL_KERAS_PATH)
    print("✅ Loaded model from mask_detector_model.keras")
elif os.path.exists(MODEL_H5_PATH):
    model = load_model(MODEL_H5_PATH)
    print("✅ Loaded model from mask_detector_model.h5")
else:
    print("❌ Error: No model file found! Make sure mask_detector_model.h5 or mask_detector_model.keras exists.")
    sys.exit()

face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

# ==============================
# GPU / CPU SETUP
# ==============================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("✅ GPU memory growth enabled")
    except RuntimeError as e:
        print(f"⚠️ GPU setup warning: {e}")
else:
    print("⚠️ No GPU detected, using CPU instead.")

# ==============================
# SETTINGS
# ==============================
IMG_SIZE = (128, 128)
LABELS = {0: "No Mask", 1: "Mask"}
COLOR_MAP = {0: (0, 0, 255), 1: (0, 255, 0)}  # red=no mask, green=mask


# ==============================
# FUNCTIONS
# ==============================
def detect_faces_in_frame(frame):
    """Detect faces in a frame and classify them."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, IMG_SIZE)
        face_normalized = face_resized / 255.0
        face_input = np.expand_dims(face_normalized, axis=0)

        prediction = model.predict(face_input, verbose=0)[0][0]
        label = 1 if prediction > 0.5 else 0
        confidence = prediction if label == 1 else 1 - prediction

        color = COLOR_MAP[label]
        text = f"{LABELS[label]} ({confidence*100:.2f}%)"

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return frame


def detect_from_image_path(image_path):
    """Detect mask from a single image path."""
    if not os.path.exists(image_path):
        print(f"❌ Error: File not found — {image_path}")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Unable to open image: {image_path}")
        return

    result = detect_faces_in_frame(img)
    cv2.imshow("🖼️ Face Mask Detection Result", result)
    print("📸 Press any key to close the image window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_from_webcam():
    """Run real-time webcam detection."""
    print("🎥 Starting webcam... Press 'q' to quit, 's' to save snapshot.")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    snapshot_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)
        result = detect_faces_in_frame(frame)
        cv2.imshow("🧠 Real-Time Face Mask Detection", result)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("👋 Webcam detection stopped.")
            break
        elif key == ord('s'):
            snapshot_name = f"snapshot_{snapshot_count}.jpg"
            cv2.imwrite(snapshot_name, result)
            print(f"💾 Snapshot saved: {snapshot_name}")
            snapshot_count += 1

    cap.release()
    cv2.destroyAllWindows()


# ==============================
# MAIN EXECUTION
# ==============================
print("\nSelect Mode:")
print("1️⃣ Real-time Webcam Detection")
print("2️⃣ Detect from a Single Image File")

choice = input("👉 Enter your choice (1 or 2): ").strip()

if choice == "1":
    detect_from_webcam()
elif choice == "2":
    image_path = input("🖼️ Enter full path of the image file: ").strip('"')
    detect_from_image_path(image_path)
else:
    print("❌ Invalid choice. Please run again and enter 1 or 2.")



# cd FACE_MASK_DETECTOR
# .\venv\Scripts\Activate.ps1
# python real_time_detection.py
