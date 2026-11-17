"""
Konfigurasi Global Aplikasi Dashboard Analisis TPT Jawa Tengah
---------------------------------------------------------------
Modul ini mendefinisikan konstanta, skema warna, daftar wilayah Jawa Tengah (35 Kab/Kota),
dan konfigurasi antarmuka untuk visualisasi terpadu.
"""

# Konfigurasi Halaman Streamlit
APP_CONFIG = {
    "page_title": "Dashboard Analisis TPT Jawa Tengah | LSTM & K-Means",
    "page_icon": "📊",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# Palet Warna Visualisasi (Modern Executive Academic)
COLORS = {
    "primary": "#1E3A8A",      # Deep Navy
    "secondary": "#0284C7",    # Sky Blue
    "accent": "#F59E0B",       # Amber
    "success": "#10B981",      # Emerald Green (Klaster Rendah)
    "warning": "#F59E0B",      # Amber (Klaster Sedang)
    "danger": "#EF4444",       # Crimson (Klaster Tinggi)
    "background_dark": "#0F172A",
    "card_bg": "#1E293B",
    "text_light": "#F8FAFC",
    "text_muted": "#94A3B8",
}

# Skema Warna Klaster K-Means
CLUSTER_COLORS = {
    0: {"label": "Klaster Rendah (Baik)", "color": "#10B981", "badge": "success"},
    1: {"label": "Klaster Sedang (Waspada)", "color": "#F59E0B", "badge": "warning"},
    2: {"label": "Klaster Tinggi (Prioritas)", "color": "#EF4444", "badge": "danger"},
}

# Daftar Lengkap 35 Kabupaten/Kota di Jawa Tengah beserta Titik Koordinat Representatif
WILAYAH_JATENG = {
    "Kabupaten Cilacap": {"lat": -7.5500, "lon": 109.0000, "tipe": "Kabupaten"},
    "Kabupaten Banyumas": {"lat": -7.4500, "lon": 109.2000, "tipe": "Kabupaten"},
    "Kabupaten Purbalingga": {"lat": -7.3000, "lon": 109.3500, "tipe": "Kabupaten"},
    "Kabupaten Banjarnegara": {"lat": -7.3500, "lon": 109.6000, "tipe": "Kabupaten"},
    "Kabupaten Kebumen": {"lat": -7.6800, "lon": 109.6500, "tipe": "Kabupaten"},
    "Kabupaten Purworejo": {"lat": -7.7100, "lon": 110.0100, "tipe": "Kabupaten"},
    "Kabupaten Wonosobo": {"lat": -7.3600, "lon": 109.9000, "tipe": "Kabupaten"},
    "Kabupaten Magelang": {"lat": -7.5000, "lon": 110.2200, "tipe": "Kabupaten"},
    "Kabupaten Boyolali": {"lat": -7.5300, "lon": 110.5900, "tipe": "Kabupaten"},
    "Kabupaten Klaten": {"lat": -7.7000, "lon": 110.6000, "tipe": "Kabupaten"},
    "Kabupaten Sukoharjo": {"lat": -7.6800, "lon": 110.8300, "tipe": "Kabupaten"},
    "Kabupaten Wonogiri": {"lat": -7.8100, "lon": 110.9200, "tipe": "Kabupaten"},
    "Kabupaten Karanganyar": {"lat": -7.6000, "lon": 110.9500, "tipe": "Kabupaten"},
    "Kabupaten Sragen": {"lat": -7.4200, "lon": 111.0200, "tipe": "Kabupaten"},
    "Kabupaten Grobogan": {"lat": -7.1200, "lon": 110.9100, "tipe": "Kabupaten"},
    "Kabupaten Blora": {"lat": -7.0000, "lon": 111.4200, "tipe": "Kabupaten"},
    "Kabupaten Rembang": {"lat": -6.7100, "lon": 111.3400, "tipe": "Kabupaten"},
    "Kabupaten Pati": {"lat": -6.7500, "lon": 111.0300, "tipe": "Kabupaten"},
    "Kabupaten Kudus": {"lat": -6.8000, "lon": 110.8400, "tipe": "Kabupaten"},
    "Kabupaten Jepara": {"lat": -6.5800, "lon": 110.6700, "tipe": "Kabupaten"},
    "Kabupaten Demak": {"lat": -6.8900, "lon": 110.6400, "tipe": "Kabupaten"},
    "Kabupaten Semarang": {"lat": -7.2100, "lon": 110.4300, "tipe": "Kabupaten"},
    "Kabupaten Temanggung": {"lat": -7.2500, "lon": 110.1700, "tipe": "Kabupaten"},
    "Kabupaten Kendal": {"lat": -7.0200, "lon": 110.2000, "tipe": "Kabupaten"},
    "Kabupaten Batang": {"lat": -7.0000, "lon": 109.8500, "tipe": "Kabupaten"},
    "Kabupaten Pekalongan": {"lat": -7.0500, "lon": 109.6300, "tipe": "Kabupaten"},
    "Kabupaten Pemalang": {"lat": -7.0000, "lon": 109.3800, "tipe": "Kabupaten"},
    "Kabupaten Tegal": {"lat": -7.0500, "lon": 109.1300, "tipe": "Kabupaten"},
    "Kabupaten Brebes": {"lat": -7.0000, "lon": 108.9500, "tipe": "Kabupaten"},
    "Kota Magelang": {"lat": -7.4700, "lon": 110.2170, "tipe": "Kota"},
    "Kota Surakarta": {"lat": -7.5700, "lon": 110.8200, "tipe": "Kota"},
    "Kota Salatiga": {"lat": -7.3300, "lon": 110.5000, "tipe": "Kota"},
    "Kota Semarang": {"lat": -6.9900, "lon": 110.4200, "tipe": "Kota"},
    "Kota Pekalongan": {"lat": -6.8900, "lon": 109.6800, "tipe": "Kota"},
    "Kota Tegal": {"lat": -6.8700, "lon": 109.1300, "tipe": "Kota"},
}

# Tahun Analisis Historis dan Proyeksi
HISTORICAL_YEARS = list(range(2015, 2025))  # 2015 - 2024
FORECAST_YEARS = [2025, 2026, 2027]          # Prediksi LSTM
ALL_YEARS = HISTORICAL_YEARS + FORECAST_YEARS

# Variabel Indikator K-Means (Fitur Sosio-Ekonomi)
KMEANS_FEATURES = [
    "tpt",                 # Tingkat Pengangguran Terbuka (%)
    "ipm",                 # Indeks Pembangunan Manusia
    "pertumbuhan_ekonomi", # Laju Pertumbuhan Ekonomi (%)
    "tpak",                # Tingkat Partisipasi Angkatan Kerja (%)
    "upah_minimum",        # UMK (Juta Rupiah)
]
