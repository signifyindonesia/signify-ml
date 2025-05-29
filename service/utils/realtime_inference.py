# utils/realtime_inference.py
import cv2
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from utils.hand_detector import HandDetector

def load_realtime_resources():
    model = load_model("model/resnet_signify.h5")
    with open("labels/label_map.json", "r") as f:
        labels = json.load(f)
    detector = HandDetector()
    return model, labels, detector

def predict_frame(frame, model, labels, detector):
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_with_drawing, bbox = detector.find_hand(frame)

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
                label = labels[class_id]

                return label, confidence
            except:
                return "Prediction Error", 0.0

    return "No hand", 0.0

