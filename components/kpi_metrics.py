"""
Komponen Metrik KPI dan Kartu Indikator Utama (Modern SaaS UI)
--------------------------------------------------------------
Menyajikan ringkasan indikator kinerja utama (KPI) dengan kartu kaca
minimalis, tipografi eksekutif, dan ikon vektor garis tipis (SVG).
"""

import streamlit as st
import pandas as pd

def render_overview_kpis(df_panel, df_clustered, eval_metrics):
    """
    Menampilkan 4 kartu metrik modern:
    1. Rata-rata TPT Terkini (2024) beserta tren YoY.
    2. Proyeksi TPT LSTM (2027) beserta arah estimasi.
    3. Akurasi Model LSTM (MAPE & Interpretasi).
    4. Wilayah Prioritas (Klaster Tinggi / TPT Tertinggi).
    """
    # 1. Rata-rata TPT 2024 vs 2023
    tpt_2024 = df_panel[df_panel["tahun"] == 2024]["tpt_aktual"].mean()
    tpt_2023 = df_panel[df_panel["tahun"] == 2023]["tpt_aktual"].mean()
    delta_tpt = round(tpt_2024 - tpt_2023, 2)

    # 2. Proyeksi 2027 vs 2024
    tpt_2027 = df_panel[df_panel["tahun"] == 2027]["tpt_prediksi_lstm"].mean()
    delta_proj = round(tpt_2027 - tpt_2024, 2)

    # 3. Akurasi LSTM (MAPE)
    mape_val = eval_metrics["mape"]
    if mape_val < 10:
        mape_desc = "Sangat Akurat (<10%)"
    elif mape_val < 20:
        mape_desc = "Baik (10-20%)"
    else:
        mape_desc = "Cukup"

    # 4. Wilayah Tertinggi
    df_2024_all = df_panel[df_panel["tahun"] == 2024]
    highest_row = df_2024_all.sort_values("tpt_aktual", ascending=False).iloc[0]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card-pro">
                <div class="metric-card-header">
                    <span class="metric-title">Rerata TPT Jateng (2024)</span>
                    <div class="metric-icon-wrap">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#60A5FA" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>
                    </div>
                </div>
                <div class="metric-val">{tpt_2024:.2f}%</div>
                <div class="metric-delta-row">
                    <span class="delta-badge-green">↓ {delta_tpt:+.2f}%</span>
                    <span style="color: #64748B; font-size: 11px;">vs 2023 ({tpt_2023:.2f}%)</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card-pro">
                <div class="metric-card-header">
                    <span class="metric-title">Proyeksi LSTM (2027)</span>
                    <div class="metric-icon-wrap">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline><polyline points="17 6 23 6 23 12"></polyline></svg>
                    </div>
                </div>
                <div class="metric-val">{tpt_2027:.2f}%</div>
                <div class="metric-delta-row">
                    <span class="delta-badge-green">↓ {delta_proj:+.2f}%</span>
                    <span style="color: #64748B; font-size: 11px;">Target Jangka Menengah</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card-pro">
                <div class="metric-card-header">
                    <span class="metric-title">Akurasi Model (MAPE)</span>
                    <div class="metric-icon-wrap">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#34D399" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
                    </div>
                </div>
                <div class="metric-val">{mape_val:.2f}%</div>
                <div class="metric-delta-row">
                    <span class="delta-badge-green">✓ {mape_desc}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card-pro">
                <div class="metric-card-header">
                    <span class="metric-title">TPT Tertinggi (2024)</span>
                    <div class="metric-icon-wrap">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#F87171" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
                    </div>
                </div>
                <div class="metric-val">{highest_row['tpt_aktual']:.2f}%</div>
                <div class="metric-delta-row">
                    <span class="delta-badge-red">{highest_row['wilayah'].replace('Kabupaten ', 'Kab. ')}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

def render_cluster_badges(df_clustered):
    """
    Menampilkan ringkasan status tipologi klaster K-Means dalam kartu modern.
    """
    counts = df_clustered.groupby("cluster_id")["wilayah"].count()
    cols = st.columns(len(counts))

    dot_classes = ["dot-green", "dot-yellow", "dot-red", "dot-indigo"]

    for idx, (cid, count) in enumerate(counts.items()):
        nama = df_clustered[df_clustered["cluster_id"] == cid]["nama_klaster"].iloc[0]
        # sederhanakan label
        nama_clean = nama.replace(" (Kondisi Relatif Baik)", "").replace(" (Kondisi Waspada)", "").replace(" (Prioritas Intervensi)", "")
        with cols[idx]:
            st.markdown(
                f"""
                <div class="cluster-pill-card">
                    <div class="cluster-indicator">
                        <span class="pulse-dot {dot_classes[idx % len(dot_classes)]}"></span>
                        <span class="cluster-name-text">{nama_clean.upper()}</span>
                    </div>
                    <div>
                        <span class="cluster-count-badge">{count}</span>
                        <span class="cluster-count-sub">Wilayah</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
