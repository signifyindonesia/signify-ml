import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from utils.hand_detector import HandDetector
import string
import json
import os

def load_resources():
    """Load model and labels"""
    model = load_model('model/resnet_signify.h5')
    with open("labels/label_map.json", "r") as f:
        labels = json.load(f)
    detector = HandDetector()
    return model, labels, detector

def predict_image(model, labels, detector, image):
    """Process and predict a single image"""
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    frame_with_drawing, bbox = detector.find_hand(image)
    
    if bbox:
        x1, y1, x2, y2 = bbox
        hand_roi = rgb_image[y1:y2, x1:x2]
        
        if hand_roi.size > 0 and hand_roi.shape[0] > 0 and hand_roi.shape[1] > 0:
            try:
                img = cv2.resize(hand_roi, (224, 224))
                img = preprocess_input(img)
                img = np.expand_dims(img, axis=0)

                pred = model.predict(img, verbose=0)[0]
                class_id = np.argmax(pred)
                confidence = pred[class_id]
                
                label_text = f"{labels[class_id]} ({confidence*100:.1f}%)"
                cv2.rectangle(frame_with_drawing, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(frame_with_drawing, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                
                print(f"Predicted: {labels[class_id]} - Confidence: {confidence:.4f}")
                return frame_with_drawing, label_text
                
            except Exception as e:
                print("Error during prediction:", e)
                return frame_with_drawing, "Prediction Error"
    
    return frame_with_drawing, "No hand detected"

def realtime_detection(model, labels, detector):
    """Real-time detection from webcam"""
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror
        result_frame, _ = predict_image(model, labels, detector, frame)
        
        cv2.imshow("Signify - Real-time Detection", result_frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

def image_detection(model, labels, detector):
    """Detection from local image file or camera capture"""
    print("\nOptions:")
    print("1. Load image from file")
    print("2. Capture image from camera")
    choice = input("Select option (1/2): ")
    
    if choice == '1':
        file_path = input("Enter image file path: ")
        if not os.path.exists(file_path):
            print("Error: File not found!")
            return
            
        image = cv2.imread(file_path)
        if image is None:
            print("Error: Could not read image!")
            return
            
    elif choice == '2':
        cap = cv2.VideoCapture(0)
        print("Press SPACE to capture, ESC to cancel")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)  # 🔄 Mirror frame before display
            cv2.imshow("Capture Image - Press SPACE to capture", frame)
            key = cv2.waitKey(1)
            
            if key == 27:  # ESC
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == 32:  # SPACE
                image = frame.copy()
                cap.release()
                cv2.destroyAllWindows()
                break
    else:
        print("Invalid choice!")
        return
    
    result_frame, prediction = predict_image(model, labels, detector, image)
    cv2.imshow("Detection Result - " + prediction, result_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    model, labels, detector = load_resources()
    
    while True:
        print("\nSign Language Detection Menu:")
        print("1. Real-time detection (webcam)")
        print("2. Image detection (from file or camera)")
        print("3. Exit")
        
        choice = input("Select mode (1/2/3): ")
        
        if choice == '1':
            realtime_detection(model, labels, detector)
        elif choice == '2':
            image_detection(model, labels, detector)
        elif choice == '3':
            print("Exiting program...")
            break
        else:
            print("Invalid choice! Please try again.")

if __name__ == "__main__":
    main()
