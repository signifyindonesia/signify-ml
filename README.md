## Signify: SIBI Detection with Deep Learning & MediaPipe

Signify adalah proyek *Capstone* berbasis Machine Learning yang bertujuan untuk mendeteksi dan mengenali bahasa isyarat SIBI (Sistem Isyarat Bahasa Indonesia) huruf A sampai Z menggunakan webcam secara real-time. Sistem ini mengombinasikan model Transfer Learning berbasis CNN (seperti EfficientNet/MobileNet) dengan ekstraksi pose tangan dari MediaPipe. Dataset dapat diakses di [Kaggle - SIBI Sign Language Dataset](https://www.kaggle.com/datasets/alvinbintang/sibi-dataset). 

<br>

### 📌 Fitur Utama

* 🔤 Mendeteksi abjad A–Z dari gestur tangan satu per satu
* 🎥 Inference real-time menggunakan OpenCV + MediaPipe
* 🧠 Model akurasi tinggi (hingga 96% validasi) menggunakan arsitektur CNN
* 📦 Modularisasi pipeline: `hand_detector.py`, `inference.py`, dan model terpisah
* 🖼️ Mendukung integrasi dengan webcam secara langsung

---

## 🧠 Arsitektur Model

Model deep learning yang digunakan berupa varian dari:

* ✅ **EfficientNetB0** (ringan, akurat, dan cepat)
* ✅ **MobileNetV2** (baseline awal)

Model dilatih dengan dataset hasil augmentasi dari gambar tangan melakukan pose A-Z, dengan input image 224x224 piksel dan output klasifikasi 26 kelas (A-Z).

---

## 🔧 Instalasi

1. **Buat dan aktifkan virtual environment (opsional):**

```bash
python -m venv venv
source venv/bin/activate  # atau venv\Scripts\activate di Windows
```

3. **Install dependensi:**

```bash
pip install -r signify_ml_requirements.txt
```

---

## 🚀 Cara Menjalankan

1. Pastikan webcam aktif
2. Jalankan skrip inference:

```bash
python inference.py
```

3. Tunjukkan gestur tangan huruf A-Z ke kamera. Model akan memprediksi huruf dan menampilkannya di layar.

---

## 📊 Hasil & Evaluasi

* **Akurasi Validasi:** 96%
* **Ukuran input model:** 224x224 piksel
* **Arsitektur:** EfficientNetB0 (pretrained ImageNet)
* **Evaluasi:** Confusion Matrix & Real-Time Inference

---

## 🧪 Teknologi yang Digunakan

| Teknologi          | Keterangan                          |
| ------------------ | ----------------------------------- |
| TensorFlow / Keras | Model CNN dan training              |
| MediaPipe          | Deteksi tangan dan landmark         |
| OpenCV             | Akses webcam dan tampilan real-time |
| Python             | Bahasa pemrograman utama            |

---

### 📌 Catatan Tambahan

* Model hanya mendeteksi satu gestur huruf dalam satu waktu.
* Untuk gestur dua tangan atau kata, pengembangan lanjutan dibutuhkan.

---
