# File: inference.py
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import argparse

# Parameter untuk preprocessing
MAX_FRAMES = 20
FRAME_HEIGHT = 160
FRAME_WIDTH = 160

video_path = "D:\DBS-coding-camp\inference\video"  # Ganti dengan path video yang sesuai

def load_and_preprocess_video(video_path, max_frames=MAX_FRAMES):
    """
    Muat dan preprocessing video untuk inferensi
    
    Args:
        video_path: Path ke file video
        max_frames: Jumlah frame maksimal yang diproses
    
    Returns:
        Array numpy berisi frame-frame yang telah diproses
    """
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Tidak dapat membuka video: {video_path}")
            return None
            
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Hitung sampling rate
        sampling_rate = max(1, frame_count // max_frames)
        
        frame_idx = 0
        while len(frames) < max_frames and frame_idx < frame_count:
            # Seek ke posisi frame yang diinginkan
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx * sampling_rate)
            ret, frame = cap.read()
            if not ret:
                break
                
            # Proses frame
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame / 255.0  # Normalisasi
            frames.append(frame)
            
            frame_idx += 1
            
        cap.release()
        
        # Handle jika kita tidak mendapatkan cukup frame
        if len(frames) == 0:
            print(f"Tidak ada frame valid dalam video {video_path}")
            return None
        
        # Pad jika perlu
        while len(frames) < max_frames:
            if frames:
                frames.append(frames[-1].copy())
            else:
                frames.append(np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3)))
                
        return np.array([frames])  # Tambahkan dimensi batch
        
    except Exception as e:
        print(f"Error saat memproses video {video_path}: {str(e)}")
        return None

def predict_sign_language(model, video_data, class_names):
    """
    Lakukan prediksi bahasa isyarat menggunakan model yang telah dilatih
    
    Args:
        model: Model TensorFlow yang sudah dimuat
        video_data: Data video yang telah diproses
        class_names: Daftar nama kelas
    
    Returns:
        Kelas prediksi dan skor probabilitas
    """
    if video_data is None:
        return None, None
        
    # Lakukan prediksi
    predictions = model.predict(video_data)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    
    return class_names[predicted_class_idx], confidence

def visualize_prediction(video_path, predicted_class, confidence):
    """
    Visualisasikan prediksi dengan menampilkan beberapa frame dari video
    dan hasil prediksi
    
    Args:
        video_path: Path ke file video
        predicted_class: Kelas yang diprediksi
        confidence: Nilai keyakinan (confidence) prediksi
    """
    import matplotlib.pyplot as plt
    
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Ambil beberapa frame untuk visualisasi
    num_frames = min(5, frame_count)
    frames = []
    
    for i in range(num_frames):
        frame_idx = i * frame_count // num_frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    
    # Tampilkan frame dan hasil prediksi
    plt.figure(figsize=(15, 8))
    
    for i, frame in enumerate(frames):
        plt.subplot(1, num_frames, i+1)
        plt.imshow(frame)
        plt.axis('off')
        plt.title(f"Frame {i+1}")
    
    plt.suptitle(f"Prediksi: {predicted_class} (Confidence: {confidence:.2f})", fontsize=16)
    plt.tight_layout()
    plt.savefig('prediction_result.png')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Inferensi klasifikasi bahasa isyarat dari video")
    parser.add_argument("--model", type=str, default="model/3dcnn_efficient.h5",
                       help="Path ke file model")
    parser.add_argument("--video", type=str, required=True,
                       help="Path ke file video untuk inferensi")
    args = parser.parse_args()
    
    # Definisikan nama kelas
    class_names = ['J', 'Z']
    
    # Cek apakah model ada
    if not os.path.exists(args.model):
        print(f"Model tidak ditemukan di {args.model}")
        print("Jalankan download_model.py terlebih dahulu untuk mengunduh model")
        return
    
    # Cek apakah video ada
    if not os.path.exists(args.video):
        print(f"File video tidak ditemukan: {args.video}")
        return
    
    # Muat model
    print(f"Memuat model dari {args.model}...")
    model = load_model(args.model)
    
    # Preprocess video
    print(f"Memproses video {args.video}...")
    video_data = load_and_preprocess_video(args.video)
    
    if video_data is not None:
        # Lakukan prediksi
        print("Melakukan prediksi...")
        predicted_class, confidence = predict_sign_language(model, video_data, class_names)
        
        if predicted_class is not None:
            print(f"\nHasil Prediksi:")
            print(f"Kelas: {predicted_class}")
            print(f"Confidence: {confidence:.4f}")
            
            # Visualisasikan hasil
            # visualize_prediction(args.video, predicted_class, confidence)
        else:
            print("Gagal melakukan prediksi.")
    else:
        print("Gagal memproses video.")

if __name__ == "__main__":
    main()