### 📈 **1. Training dan Validation Loss**

* **Kedua loss (training & validation) turun secara konsisten** hingga mendekati nol.
* Tidak ada tanda-tanda **overfitting signifikan**, karena:

  * Validation loss mengikuti pola training loss.
  * Tidak ada gap besar antara keduanya.
![Image](https://github.com/user-attachments/assets/a629797f-17d3-4f50-bc11-a4e04b50f963)

✅ **Interpretasi:** Model belajar dengan sangat baik; tidak hanya menghafal data training, tapi juga mampu generalisasi ke data validasi.

### 📊 **2. Training dan Validation Accuracy**

* Akurasi training dan validasi **naik tajam** dan mencapai **97%**.
* Kurva validasi sangat stabil dan **tinggi secara konsisten**.

![Image](https://github.com/user-attachments/assets/00fd2c06-4f3e-4913-9800-4263d48f30e9)

✅ **Interpretasi:** Model sangat akurat dan tidak mengalami masalah seperti underfitting atau overfitting.

### ✅ Kesimpulan:

Model ResNet50 **berhasil belajar dengan sangat baik**, ditunjukkan oleh:

* Loss turun konsisten.
* Akurasi validasi tinggi dan stabil.
* Tidak ada overfitting besar.
