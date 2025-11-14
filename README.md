🛡️ Face Mask Detection System (Streamlit + OpenCV DNN + Custom CNN Model)

A real-time Face Mask Detection System built from scratch using OpenCV DNN, TensorFlow/Keras, and a fully custom-trained mask-classification model.
This project detects faces using a deep-learning face detector and classifies each face as:

Mask

No Mask

The application includes a clean Streamlit UI, image upload support, and bounding-box visualization.

🚀 Features

✔️ Custom mask detection model trained by me

✔️ 100% accuracy on test dataset

✔️ OpenCV DNN Face Detector (more accurate than Haar cascades)

✔️ Live real-time image analysis

✔️ Streamlit web app

✔️ Fast and lightweight (CPU-friendly)

✔️ Shows bounding boxes, labels & confidence percentage

🧠 Tech Stack
Component	Technology
Face Detection	OpenCV DNN (deploy.prototxt + res10 SSD model)
Mask Classification	TensorFlow/Keras custom CNN
UI	Streamlit
Image Preprocessing	OpenCV
Programming Language	Python

📁 Project Structure
📦 Face-Mask-Detection
├── app.py
├── mask_detector_model.keras
├── deploy.prototxt
├── res10_300x300_ssd_iter_140000.caffemodel
├── requirements.txt
└── README.md
ALSO I HAVE USED 3 DIFFERENT DATASETS FROM KAGGLE 

🖼️ How It Works
Step 1: Face Detection

OpenCV DNN detects all faces using a pre-trained SSD model:

deploy.prototxt
res10_300x300_ssd_iter_140000.caffemodel

Step 2: Mask Classification

Each face is:

Cropped

Resized

Normalized

Sent to the CNN mask model

The model outputs:

Mask

No Mask

With the confidence percentage displayed.

🧪 Model Training (Short Summary)

The mask-classifier model is fully custom-trained by me using:

A dataset of Mask / Without Mask images

Data augmentation

A custom CNN architecture in TensorFlow/Keras

Achieved 100% accuracy on the test dataset.

For internship project authenticity,
👉 No pre-trained mask model was used. Everything was trained from scratch.

🎯 Results

Real-time accuracy: Excellent

Model accuracy: 100%

Face detection: Robust even with different angles, lighting, and multiple faces
