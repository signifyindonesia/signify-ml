import cv2
import numpy as np
from tensorflow.keras.models import load_model

MAX_FRAMES = 20
FRAME_HEIGHT = 160
FRAME_WIDTH = 160
class_names = ['J', 'Z']
model = load_model("models/3dcnn_efficient.h5")

def load_and_preprocess_video(file_path):
    frames = []
    cap = cv2.VideoCapture(file_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sampling = max(1, total // MAX_FRAMES)
    idx = 0
    while len(frames) < MAX_FRAMES and idx < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx * sampling)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame / 255.0
        frames.append(frame)
        idx += 1
    cap.release()

    while len(frames) < MAX_FRAMES:
        frames.append(np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3)))

    return np.array([frames])

def predict_from_video(file_path):
    video = load_and_preprocess_video(file_path)
    if video is None:
        return {"label": "Error processing video", "confidence": 0.0}
    preds = model.predict(video, verbose=0)[0]
    idx = np.argmax(preds)
    return {"label": class_names[idx], "confidence": float(preds[idx])}
