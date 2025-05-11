# signify-ml
## Proses Modelling Masih Mencari Model Terbaik
📌  **Hasil Modelling 4 Model**

1. Hasil Model 1 :

![Image](https://github.com/user-attachments/assets/9d26d3ed-f588-4802-bfb6-f8c3130c9197)

2. Hasil Model 2 :

![Image](https://github.com/user-attachments/assets/16ae1f6b-e15c-42ef-8b12-d331237c0453)

3. Hasil Model 3 :

![Image](https://github.com/user-attachments/assets/ede11f36-e949-4164-aab8-662d8dc59c4b)

4. Hasil Model 4 :
   
![Image](https://github.com/user-attachments/assets/05276014-f88d-4bb4-aaf7-f6e0938f2760)

# Panduan Pengembangan
## Setup Awal Menyiapkan Web
1. Clone repository
   ```bash
   git clone https://github.com/signifyindonesia/signify-web.git
   cd signify-web
   ```
2. Install dependencies
   ```bash
   npm install
   ```
3. Jalankan development server
   ```bash
   npm run dev
   ```

## Setup Awal Git
1. Buat branch baru dari `main`
   ```bash
   git checkout -b feature/nama-fitur
   ```
2. Lakukan perubahan dan commit dengan pesan yang deskriptif
   ```bash
   git add .
   git commit -m "feat: tambah halaman autentikasi"
   ```
3. Push branch ke remote
   ```bash
   git push origin feature/nama-fitur
   ```
4. Buat Pull Request ke branch `main`

## Merge Branch ke Main
1. Menuju ke branch `main`
    ```bash
    git checkout main
    ```
2. Lakukan merge dengan branch yang sudah ditambahkan
    ```bash
    git merge feature/nama-fitur
    ```
3. Push perubahan ke remote
    ```bash
    git push origin main
    ```
