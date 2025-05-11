### 📈 **1. Training dan Validation Loss**

* **Kedua loss (training & validation) turun secara konsisten** hingga mendekati nol.
* Tidak ada tanda-tanda **overfitting signifikan**, karena:

  * Validation loss mengikuti pola training loss.
  * Tidak ada gap besar antara keduanya.

✅ **Interpretasi:** Model belajar dengan sangat baik; tidak hanya menghafal data training, tapi juga mampu generalisasi ke data validasi.

### 📊 **2. Training dan Validation Accuracy**

* Akurasi training dan validasi **naik tajam** dan mencapai:

  * > 98% di sekitar epoch ke-10.
  * Hampir **99% di akhir** pelatihan.
* Kurva validasi sangat stabil dan **tinggi secara konsisten**.

✅ **Interpretasi:** Model sangat akurat dan tidak mengalami masalah seperti underfitting atau overfitting.

### ✅ Kesimpulan:

Model EfficientNetB0 **berhasil belajar dengan sangat baik**, ditunjukkan oleh:

* Loss turun konsisten.
* Akurasi validasi tinggi dan stabil.
* Tidak ada overfitting besar.
