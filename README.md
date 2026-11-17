# 📊 Dashboard Analisis Tingkat Pengangguran Terbuka (TPT) Provinsi Jawa Tengah
### Integrasi Peramalan Deep Learning LSTM & Klasterisasi Wilayah K-Means

Aplikasi web interaktif siap produksi (*production-ready*) berbasis **Streamlit** dan **Plotly**, dirancang khusus sebagai luaran utama skripsi untuk menganalisis, memvisualisasikan, dan meramalkan dinamika ketenagakerjaan pada 35 kabupaten/kota di Provinsi Jawa Tengah.

---

## 🌟 Fitur Utama Dashboard

1. **🌐 Peta Kloropleth Spasial Interaktif**:
   - Pemetaan berbasis poligon wilayah Jawa Tengah (35 Kabupaten/Kota).
   - Mode ganda: visualisasi gradasi TPT kontinu atau zonasi klaster diskrit K-Means.
   - Slider temporal dinamis dari data historis (2015) hingga proyeksi masa depan (2027).

2. **📈 Peramalan Time Series Deep Learning LSTM (Long Short-Term Memory)**:
   - Visualisasi kurva data aktual (2015–2024), estimasi in-sample, dan proyeksi out-of-sample (2025–2027).
   - Pita ketidakpastian (*95% Confidence Interval Band*).
   - Diagnostik galat residual dan evaluasi metrik performa: **RMSE**, **MAE**, dan **MAPE**.
   - Simulasi skenario masa depan (*Baseline*, *Optimis*, *Pesimis*, atau *Kustom Shock*).

3. **🧩 Klasterisasi Multidimensi K-Means**:
   - Uji validasi jumlah klaster optimal via **Elbow Method** (Inersia WCSS) dan **Silhouette Score**.
   - Proyeksi 2D *Principal Component Analysis (PCA)* dengan rasio variansi terjelaskan.
   - *Radar Chart (Spider Plot)* untuk membaca karakteristik tipologi sosio-ekonomi tiap klaster.
   - Tabel profil klaster lengkap dengan rekomendasi kebijakan ketenagakerjaan daerah.

4. **📚 Modul Edukasi & Sintesis Kebijakan**:
   - Penjelasan matematis arsitektur LSTM (*Forget Gate*, *Input Gate*, *Cell State*, *Output Gate*).
   - Formulasi objektif K-Means (*WCSS*) dan Silhouette Coefficient.
   - Sintesis tiga pilar kebijakan pembangunan ketenagakerjaan Jawa Tengah.

---

## 📁 Struktur Direktori & Modularitas Kode

Proyek dirancang dengan prinsip *Separation of Concerns* (SoC) yang rapi dan modular:

```text
tpt_jateng_dashboard/
├── app.py                      # Titik masuk utama aplikasi Streamlit (Layout, Routing & Interaksi)
├── config.py                   # Konfigurasi global (Daftar 35 Kab/Kota Jateng, Koordinat, Palet Warna)
├── requirements.txt            # Daftar pustaka dependensi Python
├── README.md                   # Dokumentasi teknis & operasional
│
├── data/
│   ├── data_loader.py          # Modul pemuatan data, caching, generator data ilmiah terkalibrasi BPS
│   ├── geojson_jateng.py       # Generator GeoJSON poligon batas 35 wilayah Jawa Tengah
│   └── raw/                    # Direktori penyimpanan dataset CSV/Excel kustom
│
├── models/
│   ├── lstm_module.py          # Logika peramalan LSTM, simulasi skenario, dan perhitungan pertumbuhan
│   └── kmeans_module.py        # Algoritma K-Means, StandardScaler, PCA 2D, Elbow & Silhouette
│
├── components/
│   ├── kpi_metrics.py          # Komponen kartu metrik KPI eksekutif & badge klaster
│   ├── map_view.py             # Visualisasi Peta Kloropleth Jawa Tengah berbasis Plotly Mapbox
│   ├── lstm_charts.py          # Grafik deret waktu LSTM, residual bar, dan peringkat proyeksi
│   └── kmeans_charts.py        # Grafik evaluasi Elbow/Silhouette, scatter PCA, radar chart, boxplot
│
└── utils/
    └── helpers.py              # Injeksi CSS kustom modern dan fungsi konversi ekspor data (CSV)
```

---

## 🚀 Panduan Instalasi dan Menjalankan Dashboard

### 1. Prasyarat Sistem
- Python 3.10+ terpasang di sistem.

### 2. Instal Dependensi
Buka terminal / PowerShell dan arahkan ke direktori proyek:
```bash
cd d:\laravel_projects\tpt_jateng_dashboard
pip install -r requirements.txt
```

### 3. Jalankan Aplikasi
Eksekusi perintah Streamlit berikut:
```bash
streamlit run app.py
```
Aplikasi akan otomatis terbuka pada peramban web di alamat:  
👉 `http://localhost:8501`

---

## 📊 Format Dataset Eksternal (Opsional)

Jika Anda ingin mengganti dataset bawaan dengan file data riil hasil pelatihan skripsi Anda, letakkan file CSV di `data/raw/data_tpt_jateng.csv` dengan struktur kolom berikut:

| wilayah | tipe_wilayah | tahun | tpt_aktual | tpt_prediksi_lstm | lower_ci | upper_ci | tipe_data |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Kota Semarang | Kota | 2024 | 5.85 | 5.79 | null | null | Historis |
| Kota Semarang | Kota | 2025 | null | 5.54 | 5.26 | 5.82 | Proyeksi LSTM |

Jika file tidak disediakan, sistem secara cerdas akan menggunakan data terkalibrasi BPS Jawa Tengah yang telah dikonfigurasi dalam sistem.
