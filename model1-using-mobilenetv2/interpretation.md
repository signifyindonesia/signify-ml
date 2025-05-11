# 📊 Evaluasi Model MobileNetV2 untuk Klasifikasi Bahasa Isyarat SIBI (A-Z)

## 📌 Arsitektur Model

Model yang digunakan berbasis **MobileNetV2** dengan transfer learning. Detail arsitektur:

* **Backbone**: `MobileNetV2` (pre-trained pada ImageNet, dibekukan/freeze)
* **Head Layers**:

  * `GlobalAveragePooling2D` untuk meratakan fitur spasial.
  * `Dense(256, activation='relu')` untuk fitur learning.
  * `BatchNormalization` untuk menstabilkan dan mempercepat pelatihan.
  * `Dropout(0.5)` untuk mencegah overfitting.
  * `Dense(26, activation='softmax')` sebagai layer output untuk klasifikasi 26 huruf.

## ⚙️ Detail Training

* **Loss Function**: Categorical Crossentropy
* **Optimizer**: Adam
* **Callback**:

  * `EarlyStopping` dengan `patience=5`
  * `ReduceLROnPlateau` jika `val_loss` stagnan

## 📈 Hasil Pelatihan

Visualisasi berikut menunjukkan performa model selama 30+ epoch:

![Training and Validation Plots](attachment)

### 🔍 Interpretasi:

#### **1. Loss (Kerugian)**

* **Training Loss** menurun secara stabil dari >2.0 ke <0.1, menandakan model belajar dengan baik.
* **Validation Loss** juga menurun dengan tren yang sama, walau terdapat sedikit fluktuasi pada beberapa epoch.
* Tidak terdapat tanda **overfitting signifikan**, karena jarak antara training dan validation loss kecil.

#### **2. Accuracy (Akurasi)**

* **Training Accuracy** meningkat drastis dari 40% ke hampir 99%.
* **Validation Accuracy** juga meningkat dari sekitar 60% ke lebih dari **95%**, menunjukkan generalisasi model yang baik.
* Beberapa fluktuasi kecil pada akurasi validasi di awal tidak signifikan dan diatasi oleh callbacks.

## ✅ Kesimpulan

Model MobileNetV2 berhasil melakukan transfer learning dengan sangat baik. Dengan kombinasi regularisasi (Dropout, BatchNormalization) dan callbacks, model mencapai:

* **Akurasi validasi tinggi (\~95%)**
* **Generalisasi yang kuat**
* **Tanpa tanda overfitting**
