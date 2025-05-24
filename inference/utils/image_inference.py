import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import json
from utils.hand_detector import HandDetector

model_path = "models/efficientnet_signify.h5"
label_path = "labels/label_map.json"

model = load_model(model_path)
with open(label_path) as f:
    labels = json.load(f)
detector = HandDetector()

def predict_from_image(image_array):
    rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    _, bbox = detector.find_hand(image_array)

    if bbox:
        x1, y1, x2, y2 = bbox
        roi = rgb_image[y1:y2, x1:x2]
        if roi.size == 0:
            return {"label": "No hand detected", "confidence": 0.0}

        roi = cv2.resize(roi, (224, 224))
        roi = preprocess_input(roi)
        roi = np.expand_dims(roi, axis=0)
        preds = model.predict(roi, verbose=0)[0]
        class_id = np.argmax(preds)
        return {
            "label": labels[class_id],
            "confidence": float(preds[class_id])
        }
    return {"label": "No hand detected", "confidence": 0.0}
