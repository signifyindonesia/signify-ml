# inference.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from utils.hand_detector import HandDetector
import string
import json

# Load model
model = load_model('models/{your_model}')
with open("labels/label_map.json", "r") as f:
    labels = json.load(f)

# Inisialisasi detektor tangan
detector = HandDetector()

# Mulai webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_with_drawing, bbox = detector.find_hand(frame)

    if bbox:
        x1, y1, x2, y2 = bbox
        hand_roi = rgb_frame[y1:y2, x1:x2]  # ROI dari frame RGB

        if hand_roi.size > 0 and hand_roi.shape[0] > 0 and hand_roi.shape[1] > 0:
            try:
                # Resize dan preprocessing sesuai model
                img = cv2.resize(hand_roi, (224, 224))
                img = preprocess_input(img)
                img = np.expand_dims(img, axis=0)

                # Prediksi
                pred = model.predict(img, verbose=0)[0]
                class_id = np.argmax(pred)
                confidence = pred[class_id]

                # Tampilkan hasil
                label_text = f"{labels[class_id]} ({confidence*100:.1f}%)"
                cv2.rectangle(frame_with_drawing, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(frame_with_drawing, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

                print(f"Predicted: {labels[class_id]} - Confidence: {confidence:.4f}")

                # (Opsional) Tampilkan ROI untuk debugging
                cv2.imshow("Hand ROI", cv2.cvtColor(img[0], cv2.COLOR_RGB2BGR))

            except Exception as e:
                print("Error saat prediksi:", e)

    cv2.imshow("Signify - SIBI Detection", frame_with_drawing)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

