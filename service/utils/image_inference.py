# utils/image_inference.py

import cv2
import numpy as np
import json
import io
from fastapi import UploadFile, HTTPException
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from utils.hand_detector import HandDetector

# Load model, labels, and detector
model_static, labels_static, detector_static = None, None, None

def load_static_model():
    global model_static, labels_static, detector_static
    if model_static is None or labels_static is None or detector_static is None:
        model_static = load_model("model/resnet_signify.h5")
        with open("labels/label_map.json", "r") as f:
            labels_static = json.load(f)
        detector_static = HandDetector()
    return model_static, labels_static, detector_static

def predict_static_image(image, model, labels, detector):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    frame_with_drawing, bbox = detector.find_hand(image)

    if bbox:
        x1, y1, x2, y2 = bbox
        hand_roi = rgb_image[y1:y2, x1:x2]

        if hand_roi.size > 0:
            try:
                img = cv2.resize(hand_roi, (224, 224))
                img = preprocess_input(img)
                img = np.expand_dims(img, axis=0)

                pred = model.predict(img, verbose=0)[0]
                class_id = np.argmax(pred)
                confidence = pred[class_id]
                label = f"{labels[class_id]} ({confidence*100:.1f}%)"

                return label, confidence, labels[class_id]
            except Exception as e:
                return f"Prediction error: {e}", 0.0, None

    return "No hand detected", 0.0, None

async def predict_from_image_file(file: UploadFile):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("File tidak bisa didecode menjadi gambar. Pastikan format file valid.")

        model, labels, detector = load_static_model()
        label, confidence, class_label = predict_static_image(image, model, labels, detector)

        return {
            "prediction": label,
            "confidence": float(confidence),
            "class": class_label
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
