# utils/video_inference.py

import cv2
import numpy as np
import tempfile
from fastapi import UploadFile, HTTPException
from tensorflow.keras.models import load_model

MAX_FRAMES = 20
FRAME_HEIGHT = 160
FRAME_WIDTH = 160
CLASS_NAMES = ['J', 'Z']

# Cache model
model_dynamic = None

def load_dynamic_model():
    global model_dynamic
    if model_dynamic is None:
        model_dynamic = load_model("model/3dcnn_efficient.h5")
    return model_dynamic

def preprocess_video(file_path):
    cap = cv2.VideoCapture(file_path)
    frames = []

    if not cap.isOpened():
        return None, "Cannot open video"

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sampling = max(1, total // MAX_FRAMES)

    for i in range(MAX_FRAMES):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * sampling)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame / 255.0
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        return None, "No valid frames"

    while len(frames) < MAX_FRAMES:
        frames.append(frames[-1].copy())

    return np.array([frames]), None

def predict_video(file_path, model):
    video_data, error = preprocess_video(file_path)
    if error:
        return error, 0.0, None

    preds = model.predict(video_data, verbose=0)[0]
    class_id = np.argmax(preds)
    confidence = preds[class_id]
    return f"{CLASS_NAMES[class_id]} ({confidence*100:.1f}%)", confidence, CLASS_NAMES[class_id]

# Fungsi tambahan: untuk FastAPI endpoint
async def predict_from_video_file(file: UploadFile):
    try:
        # Simpan sementara file video untuk dibaca dengan OpenCV
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        model = load_dynamic_model()
        label, confidence, class_label = predict_video(tmp_path, model)

        return {
            "prediction": label,
            "confidence": float(confidence),
            "class": class_label
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
