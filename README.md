# 🧠 AttentionAid - ADHD Prediction System

**AttentionAid** adalah aplikasi prediksi kemungkinan ADHD berbasis web yang dikembangkan menggunakan framework **Streamlit**. Aplikasi ini memungkinkan pengguna mengisi lima jenis kuesioner psikologis, memilih model Machine Learning, dan mendapatkan hasil prediksi serta saran penanganan secara langsung.

---

## 📌 Daftar Isi

- [Fitur Utama](#-fitur-utama)
- [Cara Menjalankan](#-cara-menjalankan)
- [Struktur Proyek](#-struktur-proyek)
- [Tentang Dataset](#-tentang-dataset)
- [Model Machine Learning](#-model-machine-learning)
- [Hasil & Output](#-hasil--output)
- [Tampilan Antarmuka](#-tampilan-antarmuka)
- [Disclaimer](#️-disclaimer)
- [Lisensi](#-lisensi)
- [Kontribusi](#-kontribusi)

---

## ✅ Fitur Utama

- 🔢 **Input Dinamis**: Pengguna dapat menjawab pertanyaan dari 5 jenis skala psikologis:
  - BDI (Beck Depression Inventory)
  - AUDIT (Alcohol Use Disorders Identification Test)
  - ASRS (Adult ADHD Self-Report Scale)
  - AAS (Adult ADHD Rating Scale)
  - BAI (Beck Anxiety Inventory)

- 🧠 **Model Prediksi ADHD**:
  - Custom **Naive Bayes**
  - Custom **K-Nearest Neighbors (KNN)**

- 📈 **Hasil & Rekomendasi**:
  - Skor total dari tiap skala
  - Prediksi kemungkinan ADHD
  - Probabilitas dalam persentase
  - Rekomendasi langkah selanjutnya

- 💾 **Penyimpanan Data**: Seluruh input dan hasil akan disimpan ke file `adhd_data.csv` untuk keperluan analisis lanjutan.

- 🎨 **Tampilan Custom**: Menggunakan CSS untuk mempercantik tampilan sidebar, tab, slider, dan tombol.

---

## 🚀 Cara Menjalankan
Link aplikasi: https://mini-proyek-adhd-ti.streamlit.app/

---

## 📁 Struktur Proyek
```bash
.
├── adhd_prediction_app.py      # File utama aplikasi
├── ADHD.xlsx                   # Dataset untuk pelatihan model
├── adhd_data.csv               # Hasil input pengguna (dihasilkan saat runtime)
├── adhd.png                    # Gambar logo aplikasi
├── requirements.txt            # Daftar pustaka yang dibutuhkan
└── README.md                   # Dokumentasi ini
```

---

## 📊 Tentang Dataset
Model dilatih menggunakan dataset `ADHD.xlsx` yang berisi data historis dari kuesioner.
- **Fitur:** `bdi1_total`, `audit1_total`, `aas1_total`, `asrs1_total.x`, `bai1_total`
- **Target:** `adhd_label` (0 = tidak ADHD, 1 = ADHD)
- **Pembagian Data:** 80% untuk pelatihan dan 20% untuk pengujian.

---

## 🤖 Model Machine Learning
Aplikasi ini mengimplementasikan dua algoritma klasifikasi sederhana **tanpa `scikit-learn`** untuk tujuan pembelajaran:
1. **Naive Bayes**: Menggunakan distribusi Gaussian untuk menghitung probabilitas.
2. **K-Nearest Neighbors (KNN)**: Menggunakan jarak Euclidean dan voting dari `k=5` tetangga terdekat.

---

## 📤 Hasil & Output
Setiap kali pengguna menekan tombol **"Calculate ADHD Prediction"**, aplikasi akan:
1.  **Melakukan Prediksi** menggunakan model yang dipilih.
2.  **Menampilkan Hasil** berupa probabilitas, status prediksi (Likely/Unlikely), dan rekomendasi.
3.  **Menyimpan Data** input pengguna ke `adhd_data.csv`.

---

## 🖼 Tampilan Antarmuka
Aplikasi didesain menggunakan custom CSS untuk menambah kenyamanan pengguna, termasuk:
- Warna background yang lembut
- Tab per kuesioner
- Slider yang responsif
- Desain sidebar dan tombol yang interaktif

---

## ⚠️ Disclaimer
Aplikasi ini **bukan alat diagnosis medis resmi**. Hasil prediksi hanya berfungsi sebagai alat bantu untuk mengenali kemungkinan adanya gangguan ADHD. Konsultasi dengan profesional kesehatan jiwa tetap diperlukan untuk diagnosis dan penanganan yang tepat.

---

