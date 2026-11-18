"""
========================================================================================
DASHBOARD INTERAKTIF ANALISIS TINGKAT PENGANGGURAN TERBUKA (TPT) PROVINSI JAWA TENGAH
Integrasi Model Peramalan Deep Learning LSTM & Klasterisasi Wilayah K-Means
========================================================================================
Luaran Skripsi: Analisis Spasial, Temporal, dan Tipologi Ketenagakerjaan Daerah
Pengembang: Expert Python Data Science & Web Developer
Framework: Streamlit & Plotly Interactive Graphics
========================================================================================
"""

import os
import streamlit as st
import pandas as pd
import numpy as np

# 1. Impor Konfigurasi & Modul Data
from config import APP_CONFIG, WILAYAH_JATENG, ALL_YEARS, HISTORICAL_YEARS, FORECAST_YEARS
from data.data_loader import load_master_dataset
from models.kmeans_module import (
    calculate_elbow_and_silhouette,
    fit_kmeans_clustering,
    get_cluster_profiles
)
from models.lstm_module import (
    simulate_lstm_scenarios,
    calculate_time_series_growth
)

# 2. Impor Komponen Visualisasi
from components.kpi_metrics import render_overview_kpis, render_cluster_badges
from components.map_view import render_choropleth_map
from components.lstm_charts import (
    render_lstm_forecast_chart,
    render_lstm_residual_chart,
    render_ranking_projection_chart
)
from components.kmeans_charts import (
    render_elbow_silhouette_chart,
    render_pca_scatter_chart,
    render_radar_cluster_chart,
    render_feature_boxplots
)
from utils.helpers import inject_custom_css, convert_df_to_csv

# Inisialisasi Halaman Streamlit
st.set_page_config(
    page_title=APP_CONFIG["page_title"],
    page_icon=APP_CONFIG["page_icon"],
    layout=APP_CONFIG["layout"],
    initial_sidebar_state=APP_CONFIG["initial_sidebar_state"]
)

# Injeksi Gaya Visual CSS Modern
inject_custom_css()

# ======================================================================================
# PEMUATAN DATASET MASTER (CACHED)
# ======================================================================================
df_panel, df_kmeans_raw, eval_metrics = load_master_dataset()

# ======================================================================================
# SIDEBAR KONTROL DAN NAVIGASI INTERAKTIF
# ======================================================================================
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 10px 4px 18px 4px; border-bottom: 1px solid rgba(255, 255, 255, 0.08); margin-bottom: 18px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="width: 38px; height: 38px; border-radius: 10px; background: linear-gradient(135deg, #6366F1 0%, #3B82F6 100%); display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4);">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6"></polygon><line x1="8" y1="2" x2="8" y2="18"></line><line x1="16" y1="6" x2="16" y2="22"></line></svg>
                </div>
                <div>
                    <div style="font-size: 15px; font-weight: 800; color: #FFFFFF; letter-spacing: -0.3px;">TPT JAWA TENGAH</div>
                    <div style="font-size: 10px; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.6px; font-weight: 600;">Spatial & Machine Learning</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<p style='font-size: 11px; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 6px;'>Navigasi Modul</p>", unsafe_allow_html=True)
    menu_pilihan = st.radio(
        label="Pilih Modul Tampilan:",
        options=[
            "Peta Spasial & Sebaran",
            "Peramalan LSTM (Time Series)",
            "Klasterisasi K-Means",
            "Metodologi & Landasan Skripsi"
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 11px; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 6px;'>Parameter Analisis</p>", unsafe_allow_html=True)

    # Filter Wilayah
    opsi_wilayah = ["Seluruh Jawa Tengah (Rata-rata)"] + sorted(list(WILAYAH_JATENG.keys()))
    wilayah_terpilih = st.selectbox(
        label="Fokus Analisis Wilayah:",
        options=opsi_wilayah,
        index=0,
        help="Pilih kabupaten/kota spesifik untuk melihat rincian pergerakan TPT dan hasil peramalan LSTM."
    )

    # Parameter K-Means (Jumlah Klaster k)
    k_clusters = st.slider(
        label="Jumlah Klaster K-Means (k):",
        min_value=2,
        max_value=5,
        value=3,
        step=1,
        help="Rekomendasi nilai k optimal berdasarkan Elbow Method & Silhouette Score adalah k=3."
    )

    # Parameter Tahun Peta
    default_year = 2025 if 2025 in ALL_YEARS else ALL_YEARS[-1]
    tahun_peta = st.select_slider(
        label="Tahun Peta Spasial:",
        options=ALL_YEARS,
        value=default_year,
        help="Pilih tahun pengamatan untuk melihat evolusi spasial TPT dari 2020 hingga proyeksi 2026."
    )

    # Skenario Proyeksi LSTM
    skenario_lstm = st.selectbox(
        label="Skenario Proyeksi LSTM (2026):",
        options=[
            "Baseline",
            "Optimis (Pertumbuhan Investasi & Padat Karya)",
            "Pesimis (Disrupsi & Perlambatan Ekonomi)",
            "Kustom"
        ],
        index=0
    )

    shock_slider = 0.0
    if skenario_lstm == "Kustom":
        shock_slider = st.slider(
            label="Penyesuaian Manual TPT (% poin):",
            min_value=-1.5,
            max_value=1.5,
            value=0.0,
            step=0.1,
            help="Simulasikan dampak guncangan ekonomi atau intervensi kebijakan pasar kerja terhadap nilai TPT."
        )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.divider()

    # Upload Data Eksternal (Opsional - CSV atau Excel)
    with st.expander("Unggah Dataset Mandiri (CSV / Excel)"):
        st.caption(
            "Unggah file data panel (CSV atau Excel `.xlsx`) "
            "dengan kolom minimal: **wilayah**, **tahun**, **tpt_aktual**."
        )
        uploaded_file = st.file_uploader(
            "Pilih file",
            type=["csv", "xlsx", "xls"],
            label_visibility="collapsed"
        )
        if uploaded_file is not None:
            try:
                fname = uploaded_file.name.lower()
                if fname.endswith(".csv"):
                    df_raw = pd.read_csv(uploaded_file)
                else:
                    xl = pd.ExcelFile(uploaded_file)
                    sheet = "Data" if "Data" in xl.sheet_names else xl.sheet_names[0]
                    # Jika ada beberapa sheet, beri pilihan
                    if len(xl.sheet_names) > 1:
                        sheet = st.selectbox(
                            "Pilih Sheet Excel:",
                            xl.sheet_names,
                            index=0 if "Data" not in xl.sheet_names else xl.sheet_names.index("Data")
                        )
                    df_raw = xl.parse(sheet)

                actual_cols = list(df_raw.columns)
                st.success(f"✅ File **{uploaded_file.name}** terbaca! ({len(df_raw):,} baris)")

                # Coba deteksi kolom secara cerdas (case-insensitive + fuzzy label)
                col_lower_map = {c.lower().strip(): c for c in actual_cols}

                CANDIDATES = {
                    "wilayah":    ["wilayah", "kabupaten", "kota", "kabupaten/kota", "region",
                                   "nama_wilayah", "nama wilayah", "daerah", "kab/kota"],
                    "tahun":      ["tahun", "year", "periode", "waktu", "thn"],
                    "tpt_aktual": ["tpt_aktual", "tpt", "tingkat pengangguran", "pengangguran",
                                   "tpt aktual", "tpt_actual", "nilai tpt", "tpt (%)", "tpt(%)"],
                }

                auto_match = {}
                for target, candidates in CANDIDATES.items():
                    for cand in candidates:
                        if cand in col_lower_map:
                            auto_match[target] = col_lower_map[cand]
                            break

                st.markdown("**🔧 Petakan Kolom File ke Kolom Sistem:**")
                st.caption("Kolom yang terdeteksi otomatis sudah dipilihkan. Sesuaikan jika perlu.")

                col_wilayah = st.selectbox(
                    "Kolom 'Nama Wilayah':",
                    actual_cols,
                    index=actual_cols.index(auto_match["wilayah"]) if "wilayah" in auto_match else 0
                )
                col_tahun = st.selectbox(
                    "Kolom 'Tahun':",
                    actual_cols,
                    index=actual_cols.index(auto_match["tahun"]) if "tahun" in auto_match else 0
                )
                col_tpt = st.selectbox(
                    "Kolom 'Nilai TPT (%)':",
                    actual_cols,
                    index=actual_cols.index(auto_match["tpt_aktual"]) if "tpt_aktual" in auto_match else 0
                )

                if st.button("✅ Terapkan & Simpan Dataset", use_container_width=True):
                    df_mapped = df_raw.rename(columns={
                        col_wilayah: "wilayah",
                        col_tahun:   "tahun",
                        col_tpt:     "tpt_aktual"
                    })
                    df_mapped["tahun"] = pd.to_numeric(df_mapped["tahun"], errors="coerce").astype("Int64")
                    df_mapped = df_mapped.dropna(subset=["wilayah", "tahun", "tpt_aktual"])

                    # Simpan sebagai CSV ke data/raw/ agar dipakai saat restart
                    save_path = os.path.join(
                        os.path.dirname(__file__), "data", "raw", "data_tpt_jateng.csv"
                    )
                    df_mapped.to_csv(save_path, index=False)
                    st.success(
                        f"✅ Dataset disimpan! ({len(df_mapped):,} baris, "
                        f"{df_mapped['wilayah'].nunique()} wilayah). "
                        "Klik **Rerun** untuk memuat ulang dashboard."
                    )
                    st.balloons()

                with st.expander("👁️ Preview 5 Baris Pertama"):
                    st.dataframe(df_raw.head(5), use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"❌ Gagal membaca file: {e}")

    # Unggah & Pemrosesan Shapefile GADM (Level 2 Kab/Kota)
    with st.expander("Unggah Peta GIS / GADM (Shapefile / ZIP)"):
        st.caption(
            "Mendukung file GADM Indonesia Level 2 (180MB+). "
            "Sistem akan otomatis mengekstrak, memotong hanya 35 Kab/Kota Jateng, "
            "dan mengompresnya menjadi GeoJSON ringan (~500KB)."
        )
        gadm_file = st.file_uploader(
            "Pilih arsip ZIP atau file SHP GADM",
            type=["zip", "shp"],
            key="gadm_uploader",
            label_visibility="collapsed"
        )
        if gadm_file is not None:
            if st.button("⚡ Ekstrak & Konversi Peta Jawa Tengah", key="btn_conv_gadm", use_container_width=True):
                with st.spinner("Memproses Shapefile & mengekstrak 35 Kab/Kota Jawa Tengah..."):
                    try:
                        from utils.gadm_processor import process_gadm_zip_or_shp
                        matched, out_path = process_gadm_zip_or_shp(gadm_file)
                        st.success(f"✅ Berhasil mengekstrak {matched} wilayah Jawa Tengah menjadi GeoJSON resmi!")
                        st.balloons()
                        st.rerun()
                    except Exception as ex:
                        st.error(f"❌ Gagal memproses Shapefile: {ex}")

        # Opsi cepat jika file diletakkan langsung di folder data/raw/
        raw_dir = os.path.join(os.path.dirname(__file__), "data", "raw")
        raw_files = [
            f for f in os.listdir(raw_dir)
            if f.lower().endswith((".zip", ".shp")) and not f.startswith("_")
        ] if os.path.exists(raw_dir) else []

        if raw_files:
            st.markdown("---")
            st.caption("💡 **Atau proses berkas yang sudah ada di folder `data/raw/`:**")
            selected_local_shp = st.selectbox("Pilih berkas lokal:", raw_files, key="select_local_gadm")
            if st.button("🚀 Proses Berkas Lokal", key="btn_proc_local", use_container_width=True):
                with st.spinner("Mengonversi berkas lokal..."):
                    try:
                        from utils.gadm_processor import process_gadm_zip_or_shp
                        local_path = os.path.join(raw_dir, selected_local_shp)
                        matched, _ = process_gadm_zip_or_shp(local_path)
                        st.success(f"✅ Selesai! {matched} wilayah berhasil diperbarui.")
                        st.rerun()
                    except Exception as ex:
                        st.error(f"❌ Gagal memproses: {ex}")

    st.caption("Luaran Utama Skripsi © 2026 | Analisis Ketenagakerjaan Jawa Tengah")

# ======================================================================================
# PEMROSESAN MODEL SESUAI PARAMETER INPUT PENGGUNA
# ======================================================================================
# 1. Eksekusi K-Means dengan k terpilih
df_clustered, kmeans_model, scaler_model, explained_var = fit_kmeans_clustering(
    df_kmeans_raw, n_clusters=k_clusters
)
cluster_profiles = get_cluster_profiles(df_clustered)
elbow_data = calculate_elbow_and_silhouette(df_kmeans_raw, max_k=6)

# 2. Filter data wilayah untuk LSTM
if wilayah_terpilih == "Seluruh Jawa Tengah (Rata-rata)":
    # Agregasi rerata seluruh Jawa Tengah
    df_wilayah = df_panel.groupby("tahun").agg({
        "tpt_aktual": "mean",
        "tpt_prediksi_lstm": "mean",
        "lower_ci": "mean",
        "upper_ci": "mean",
        "tipe_data": "first"
    }).reset_index()
    df_wilayah["wilayah"] = "Seluruh Jawa Tengah (Rata-rata)"
else:
    df_wilayah = df_panel[df_panel["wilayah"] == wilayah_terpilih].copy()

# Terapkan skenario peramalan LSTM
df_wilayah_simulated = simulate_lstm_scenarios(df_wilayah, scenario=skenario_lstm, shock_percent=shock_slider)

# ======================================================================================
# HEADER UTAMA DASHBOARD
# ======================================================================================
st.markdown(
    """
    <div class="executive-header">
        <div>
            <div class="header-meta">
                <span class="meta-pill meta-pill-indigo">LUARAN SKRIPSI ILMIAH</span>
                <span class="meta-pill meta-pill-emerald">● SYSTEM OPERATIONAL</span>
                <span style="font-size: 11px; color: #64748B;">PSTI &bull; Univ. Muhammadiyah Purworejo</span>
            </div>
            <h1 class="header-title" style="margin-top: 10px;">ANALISIS TINGKAT PENGANGGURAN TERBUKA (TPT) PROVINSI JAWA TENGAH</h1>
            <p class="header-subtitle">
                Integrasi Spasial-Temporal: Peramalan Deret Waktu Deep Learning <b>LSTM</b> & Pemetaan Tipologi <b>K-Means Clustering</b> (35 Kabupaten/Kota).
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Tampilkan Baris KPI Metrik Teratas
render_overview_kpis(df_panel, df_clustered, eval_metrics)
st.write("")
render_cluster_badges(df_clustered)
st.write("")

# ======================================================================================
# TAMPILAN BERDASARKAN NAVIGASI MODUL
# ======================================================================================

# --------------------------------------------------------------------------------------
# MODUL 1: PETA SPASIAL & SEBARAN WILAYAH
# --------------------------------------------------------------------------------------
if menu_pilihan == "Peta Spasial & Sebaran":
    st.markdown("### Peta Distribusi Spasial Wilayah Jawa Tengah")
    st.write(
        """
        Peta kloropleth interaktif di bawah ini menyajikan distribusi geografis Tingkat Pengangguran Terbuka (TPT)
        pada 35 kabupaten/kota di Provinsi Jawa Tengah. Anda dapat mengubah tahun observasi melalui slider di sidebar
        serta memilih mode pewarnaan: gradasi TPT kontinu atau zonasi klaster diskrit hasil K-Means.
        """
    )

    peta_col1, peta_col2 = st.columns([8, 4])

    with peta_col1:
        tab_peta_mode = st.radio(
            label="Pilih Mode Visualisasi Peta:",
            options=["Distribusi Nilai TPT (%)", "Zonasi Klaster K-Means"],
            horizontal=True
        )
        mode_str = "tpt" if tab_peta_mode == "Distribusi Nilai TPT (%)" else "cluster"
        fig_map = render_choropleth_map(df_panel, df_clustered, selected_year=tahun_peta, map_mode=mode_str)
        st.plotly_chart(fig_map, use_container_width=True)
        st.caption(
            "📐 **Keterangan Kartografis:** Sistem Proyeksi: Geografis (WGS 84) &bull; "
            "Sumber Data: Badan Pusat Statistik (BPS) & Global Administrative Areas (GADM) &bull; "
            "Peneliti: **Muhammad Safiq (NIM: 232520069)** — TI Universitas Muhammadiyah Purworejo (2026)"
        )

    with peta_col2:
        st.markdown(f"#### Ringkasan Wilayah Tahun {tahun_peta}")
        df_target_year = df_panel[df_panel["tahun"] == tahun_peta].copy()
        df_target_year["tpt_tampil"] = df_target_year["tpt_aktual"].fillna(df_target_year["tpt_prediksi_lstm"])
        
        # Sort tertinggi dan terendah
        highest = df_target_year.sort_values("tpt_tampil", ascending=False).iloc[0]
        lowest = df_target_year.sort_values("tpt_tampil", ascending=True).iloc[0]
        mean_val = df_target_year["tpt_tampil"].mean()

        st.info(
            f"""
            **Rata-rata TPT Provinsi:** `{mean_val:.2f}%`  
            **Wilayah Tertinggi:** **{highest['wilayah']}** (`{highest['tpt_tampil']:.2f}%`)  
            **Wilayah Terendah:** **{lowest['wilayah']}** (`{lowest['tpt_tampil']:.2f}%`)
            """
        )

        st.markdown("##### Tabel Data Wilayah:")
        tabel_ringkas = df_target_year[["wilayah", "tipe_wilayah", "tpt_tampil", "tipe_data"]].rename(
            columns={
                "wilayah": "Kabupaten/Kota",
                "tipe_wilayah": "Tipe",
                "tpt_tampil": f"TPT {tahun_peta} (%)",
                "tipe_data": "Status"
            }
        ).sort_values(f"TPT {tahun_peta} (%)", ascending=False)

        st.dataframe(
            tabel_ringkas,
            height=300,
            use_container_width=True,
            hide_index=True
        )

# --------------------------------------------------------------------------------------
# MODUL 2: PERAMALAN DEEP LEARNING LSTM
# --------------------------------------------------------------------------------------
elif menu_pilihan == "Peramalan LSTM (Time Series)":
    st.markdown(f"### Analisis Deret Waktu & Proyeksi Model LSTM: {wilayah_terpilih}")
    st.write(
        f"""
        Model **Long Short-Term Memory (LSTM)** menangani autokorelasi temporal non-linear data TPT (2020–2025)
        dan memproyeksikan pergerakan target tahun **2026** (sesuai luaran skripsi). Pada saat ini skenario yang aktif adalah:
        **`{skenario_lstm}`**.
        """
    )

    # 1. Grafik Utama Deret Waktu
    fig_lstm = render_lstm_forecast_chart(df_wilayah_simulated, wilayah_name=wilayah_terpilih)
    st.plotly_chart(fig_lstm, use_container_width=True)

    # Baris Statistik Detail Pertumbuhan & Residual
    col_stat1, col_stat2 = st.columns([6, 6])

    with col_stat1:
        st.markdown("#### Diagnostik Residual Model LSTM (In-Sample)")
        fig_residual = render_lstm_residual_chart(df_wilayah_simulated)
        st.plotly_chart(fig_residual, use_container_width=True)

    with col_stat2:
        st.markdown("#### Evaluasi Metrik Akurasi Model LSTM")
        growth = calculate_time_series_growth(df_panel, wilayah_terpilih)
        
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("RMSE", f"{eval_metrics['rmse']:.3f}", help="Root Mean Squared Error (Semakin mendekati 0 semakin baik)")
        with m_col2:
            st.metric("MAE", f"{eval_metrics['mae']:.3f}", help="Mean Absolute Error")
        with m_col3:
            st.metric("MAPE", f"{eval_metrics['mape']:.2f}%", help="Mean Absolute Percentage Error (<10% = Sangat Akurat)")

        st.markdown("##### Analisis Perubahan Historis & Estimasi Masa Depan:")
        st.markdown(
            f"""
            - **Titik Awal Periode (2020):** `{growth['tpt_2020']:.2f}%`
            - **Posisi Historis Terkini (2025):** `{growth['tpt_2025_latest']:.2f}%` *(Tren 2020-2025: `{growth['delta_2025_vs_2020']:+.2f}%` poin)*
            - **Proyeksi Target Skripsi (2026):** `{growth['tpt_2026_proj']:.2f}%` *(Estimasi peramalan: `{growth['projected_change_2026']:+.2f}%` poin)*
            """
        )

    st.divider()

    # Perbandingan Peringkat Proyeksi 2026 Antar Wilayah
    st.markdown("#### Peringkat Proyeksi TPT Antar Kabupaten/Kota Tahun 2026")
    fig_rank = render_ranking_projection_chart(df_panel, target_year=2026)
    st.plotly_chart(fig_rank, use_container_width=True)

# --------------------------------------------------------------------------------------
# MODUL 3: KLASTERISASI SOSIO-EKONOMI K-MEANS
# --------------------------------------------------------------------------------------
elif menu_pilihan == "Klasterisasi K-Means":
    st.markdown(f"### Segmentasi Tipologi Wilayah Menggunakan K-Means Clustering (k={k_clusters})")
    st.write(
        """
        Pengelompokan wilayah mengintegrasikan variabel multidimensi: **TPT (Tingkat Pengangguran Terbuka)**,
        **IPM (Indeks Pembangunan Manusia)**, **Pertumbuhan Ekonomi**, **TPAK**, dan **Upah Minimum (UMK)**.
        Standarisasi *StandardScaler* digunakan sebelum pelatihan model untuk meniadakan bias skala satuan.
        """
    )

    col_k1, col_k2 = st.columns([6, 6])

    with col_k1:
        # Evaluasi Elbow & Silhouette
        fig_elbow = render_elbow_silhouette_chart(elbow_data, optimal_k=k_clusters)
        st.plotly_chart(fig_elbow, use_container_width=True)

    with col_k2:
        # Scatter Proyeksi PCA 2D
        fig_pca = render_pca_scatter_chart(df_clustered, explained_var)
        st.plotly_chart(fig_pca, use_container_width=True)

    col_k3, col_k4 = st.columns([6, 6])

    with col_k3:
        # Radar Chart Tipologi Klaster
        fig_radar = render_radar_cluster_chart(df_clustered)
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_k4:
        # Boxplot Distribusi TPT
        fig_box = render_feature_boxplots(df_clustered)
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("#### Tabel Profil Klaster & Rekomendasi Kebijakan Ketenagakerjaan")
    st.dataframe(
        cluster_profiles.rename(
            columns={
                "cluster_id": "ID",
                "nama_klaster": "Tipologi Klaster",
                "jumlah_wilayah": "Jumlah Wilayah",
                "tpt": "Rata-rata TPT (%)",
                "ipm": "Rata-rata IPM",
                "pertumbuhan_ekonomi": "Pertumbuhan (%)",
                "tpak": "TPAK (%)",
                "upah_minimum": "UMK (Jt Rp)",
                "rekomendasi_kebijakan": "Rekomendasi Kebijakan Daerah"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    # Tombol Ekspor Data CSV
    csv_clustered = convert_df_to_csv(df_clustered)
    st.download_button(
        label="Unduh Data Hasil Klasterisasi (CSV)",
        data=csv_clustered,
        file_name="hasil_klasterisasi_tpt_jateng.csv",
        mime="text/csv"
    )

# --------------------------------------------------------------------------------------
# MODUL 4: METODOLOGI & SINTESIS KEBIJAKAN SKRIPSI
# --------------------------------------------------------------------------------------
elif menu_pilihan == "Metodologi & Landasan Skripsi":
    st.markdown("### Kerangka Metodologi Penelitian & Sintesis Akademik")
    st.write(
        """
        Modul ini mendokumentasikan landasan teoritis, alur integrasi analitik,
        serta justifikasi saintifik yang digunakan dalam skripsi.
        """
    )

    with st.expander("🔬 1. Arsitektur Model Deep Learning LSTM (Long Short-Term Memory)", expanded=True):
        st.markdown(
            r"""
            LSTM merupakan varian dari *Recurrent Neural Network* (RNN) yang dirancang khusus untuk mengatasi masalah
            *vanishing gradient* pada data deret waktu berjarak panjang. Struktur sel LSTM dikendalikan oleh tiga gerbang (*gates*):
            
            1. **Forget Gate ($f_t$):** Menentukan informasi masa lalu yang harus dibuang dari memori internal.
               $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
            2. **Input Gate ($i_t$) & Candidate Memory ($\tilde{C}_t$):** Mengatur informasi baru yang disimpan.
               $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
               $$\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$
            3. **Cell State Update ($C_t$):** Memperbarui memori sel jangka panjang.
               $$C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$$
            4. **Output Gate ($o_t$) & Hidden State ($h_t$):** Menghasilkan keluaran prediksi TPT.
               $$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
               $$h_t = o_t * \tanh(C_t)$$
            
            *Lookback Window* yang diterapkan pada skripsi ini adalah **3 tahun**, memungkinkan model mempelajari pola
            siklus ketenagakerjaan daerah dari data historis (2020–2025) sebelum memproyeksikan target tahun 2026.
            """
        )

    with st.expander("🧩 2. Algoritma Klasterisasi K-Means & Validasi Klaster", expanded=True):
        st.markdown(
            r"""
            Klasterisasi K-Means mengelompokkan 35 kabupaten/kota ke dalam $k$ kelompok homogen berdasarkan kesamaan
            vektor fitur sosio-ekonomi terstandarisasi.
            
            **Fungsi Objektif (Within-Cluster Sum of Squares / WCSS):**
            $$J = \sum_{j=1}^{k} \sum_{i \in S_j} ||x_i - \mu_j||^2$$
            
            **Evaluasi Silhouette Coefficient ($s_i$):**
            $$s_i = \frac{b_i - a_i}{\max(a_i, b_i)}$$
            di mana $a_i$ adalah jarak rata-rata ke sampel seklaster, dan $b_i$ adalah jarak rata-rata terdekat ke klaster lain.
            """
        )

    with st.expander("🏛️ 3. Sintesis Kebijakan Pembangunan Ketenagakerjaan Jawa Tengah", expanded=True):
        st.markdown(
            """
            Berdasarkan konvergensi hasil proyeksi LSTM dan klasterisasi K-Means, dirumuskan tiga pilar rekomendasi strategis:
            
            1. **Klaster Prioritas Tinggi (Urban & Pusat Industri):**
               - Memperkuat integrasi vokasi link-and-match dengan kawasan industri (KIT Batang, Kendal, Wijayakusuma).
               - Memfasilitasi bursa kerja digital berkala untuk mengatasi pengangguran friksional lulusan baru.
            2. **Klaster Menengah (Daerah Penyangga & Aglomerasi):**
               - Program reskilling tenaga kerja agraris menuju sektor jasa dan manufaktur bernilai tambah.
               - Akses permodalan kredit usaha rakyat (KUR) bagi wirausaha pemula.
            3. **Klaster Rendah (Daerah Agraris & TPT Stabil):**
               - Penguatan rantai pasok agroindustri daerah guna mengurangi fenomena *setengah penganggur (underemployment)*.
               - Modernisasi mekanisasi pertanian untuk menarik minat generasi muda produktif.
            """
        )

# ======================================================================================
# FOOTER DASHBOARD
# ======================================================================================
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748B; font-size: 12px; padding-bottom: 20px;">
        <b>Dashboard Analisis Tingkat Pengangguran Terbuka (TPT) Provinsi Jawa Tengah</b><br>
        Sistem Informasi Eksekutif Berbasis Streamlit, Deep Learning LSTM & Algoritma K-Means.<br>
        Data bersumber dari Badan Pusat Statistik (BPS) Provinsi Jawa Tengah.
    </div>
    """,
    unsafe_allow_html=True
)
