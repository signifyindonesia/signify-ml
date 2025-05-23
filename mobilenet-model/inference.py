import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import argparse

def load_sibi_model(model_path='model_terbaik.h5'):
    """
    Load the trained SIBI sign language model
    """
    try:
        model = load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_image(img_path):
    """
    Preprocess the image to match the model's expected input
    """
    try:
        # Load and resize image
        img = image.load_img(img_path, target_size=(224, 224))
        
        # Convert to array and normalize
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize to [0,1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        return img_array, img
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None, None

def predict_sibi(model, img_array):
    """
    Make prediction using the model
    """
    try:
        # Get model prediction
        predictions = model.predict(img_array)
        
        # Get the predicted class index
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx] * 100
        
        # Map index to class label (A-Z)
        class_labels = [chr(i + 65) for i in range(26)]  # A-Z
        predicted_label = class_labels[predicted_class_idx]
        
        # # Get top 5 predictions for display
        # top5_indices = np.argsort(predictions[0])[-5:][::-1]
        # top5_labels = [class_labels[i] for i in top5_indices]
        # top5_confidences = [predictions[0][i] * 100 for i in top5_indices]
        
        # # Print top 5 predictions
        # print("\nTop 5 Predictions:")
        # for label, conf in zip(top5_labels, top5_confidences):
        #     print(f"  {label}: {conf:.2f}%")
        
        return predicted_label, confidence, predictions[0]
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None, None, None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='SIBI Sign Language Prediction')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--model', type=str, default='model_terbaik.h5', help='Path to the model file')
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found")
        return
    
    # Load model
    model = load_sibi_model(args.model)
    if model is None:
        return
    
    # Process image
    img_array, img = preprocess_image(args.image)
    if img_array is None:
        return
    
    # Make prediction
    predicted_label, confidence, all_confidences = predict_sibi(model, img_array)
    if predicted_label is None:
        return
    
    # Display results
    print(f"Predicted letter: {predicted_label}")
    print(f"Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    main()