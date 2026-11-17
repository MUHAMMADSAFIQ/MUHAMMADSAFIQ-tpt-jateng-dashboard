"""
Modul Pemodelan LSTM (Long Short-Term Memory) & Skenario Proyeksi
----------------------------------------------------------------
Mengatur logika interpretasi model Deep Learning Time-Series LSTM,
penghitungan metrik performa (RMSE, MAE, MAPE), residual diagnostics,
dan simulasi skenario masa depan (Optimis, Baseline, Pesimis).
"""

import numpy as np
import pandas as pd

def simulate_lstm_scenarios(df_wilayah_panel, scenario="Baseline", shock_percent=0.0):
    """
    Melakukan simulasi skenario peramalan TPT berbasis model LSTM:
    - Baseline: Mengikuti proyeksi murni model LSTM.
    - Skenario Optimis: Kebijakan pasar kerja agresif / pertumbuhan investasi (TPT turun lebih cepat -0.3% s/d -0.6%).
    - Skenario Pesimis: Perlambatan ekonomi global atau disrupsi industri (TPT naik +0.4% s/d +0.8%).
    - Custom Shock: Pengaruh intervensi kustom berbasis input pengguna slider.
    """
    df_sim = df_wilayah_panel.copy()

    scenario_adjustments = {
        "Baseline": 0.0,
        "Optimis (Pertumbuhan Investasi & Padat Karya)": -0.45,
        "Pesimis (Disrupsi & Perlambatan Ekonomi)": 0.55,
        "Kustom": shock_percent
    }

    adj = scenario_adjustments.get(scenario, 0.0)

    # Terapkan penyesuaian hanya pada tahun-tahun proyeksi (>= 2025)
    for idx, row in df_sim.iterrows():
        if row["tahun"] >= 2025:
            step = row["tahun"] - 2024
            cumulative_adj = round(adj * (0.7 + 0.3 * step), 2)
            adjusted_pred = max(1.5, round(row["tpt_prediksi_lstm"] + cumulative_adj, 2))
            df_sim.at[idx, "tpt_prediksi_lstm"] = adjusted_pred
            
            if row["lower_ci"] is not None:
                df_sim.at[idx, "lower_ci"] = max(1.0, round(row["lower_ci"] + cumulative_adj, 2))
            if row["upper_ci"] is not None:
                df_sim.at[idx, "upper_ci"] = round(row["upper_ci"] + cumulative_adj, 2)

    return df_sim

def calculate_time_series_growth(df_panel, wilayah=None):
    """
    Menghitung laju pertumbuhan TPT (YoY - Year on Year) dan perubahan absolut
    antara tahun awal, tahun terakhir historis (2024), dan akhir proyeksi (2027).
    """
    if wilayah and wilayah != "Seluruh Jawa Tengah (Rata-rata)":
        df_target = df_panel[df_panel["wilayah"] == wilayah].sort_values("tahun")
    else:
        # Agregat provinsi
        df_target = df_panel.groupby("tahun").agg({
            "tpt_aktual": "mean",
            "tpt_prediksi_lstm": "mean"
        }).reset_index().sort_values("tahun")

    # Hitung TPT gabungan (Aktual untuk <= 2024, Prediksi untuk >= 2025)
    tpt_series = []
    for _, row in df_target.iterrows():
        val = row["tpt_aktual"] if not np.isnan(row["tpt_aktual"]) and row["tpt_aktual"] is not None else row["tpt_prediksi_lstm"]
        tpt_series.append(round(val, 2))
    df_target["tpt_final"] = tpt_series

    val_2015 = df_target[df_target["tahun"] == 2015]["tpt_final"].values[0]
    val_2020 = df_target[df_target["tahun"] == 2020]["tpt_final"].values[0]
    val_2024 = df_target[df_target["tahun"] == 2024]["tpt_final"].values[0]
    val_2027 = df_target[df_target["tahun"] == 2027]["tpt_final"].values[0]

    growth_stats = {
        "tpt_2015": val_2015,
        "tpt_2020_peak": val_2020,
        "tpt_2024_latest": val_2024,
        "tpt_2027_proj": val_2027,
        "delta_2020_covid": round(val_2020 - df_target[df_target["tahun"] == 2019]["tpt_final"].values[0], 2),
        "delta_2024_vs_2020": round(val_2024 - val_2020, 2),
        "projected_change_2027": round(val_2027 - val_2024, 2),
    }

    return growth_stats
