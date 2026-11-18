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

    # Terapkan penyesuaian hanya pada tahun proyeksi (>= 2026)
    for idx, row in df_sim.iterrows():
        if row["tahun"] >= 2026:
            step = row["tahun"] - 2025
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
    antara tahun awal dataset (2020), tahun terakhir historis (2025), dan proyeksi LSTM (2026).
    """
    if wilayah and wilayah != "Seluruh Jawa Tengah (Rata-rata)":
        df_target = df_panel[df_panel["wilayah"] == wilayah].sort_values("tahun")
    else:
        # Agregat provinsi
        df_target = df_panel.groupby("tahun").agg({
            "tpt_aktual": "mean",
            "tpt_prediksi_lstm": "mean"
        }).reset_index().sort_values("tahun")

    # Hitung TPT gabungan (Aktual untuk <= 2025, Prediksi untuk >= 2026)
    tpt_series = []
    for _, row in df_target.iterrows():
        val = row["tpt_aktual"] if not np.isnan(row["tpt_aktual"]) and row["tpt_aktual"] is not None else row["tpt_prediksi_lstm"]
        tpt_series.append(round(val, 2))
    df_target["tpt_final"] = tpt_series

    def get_val(yr, default_idx=0):
        m = df_target[df_target["tahun"] == yr]["tpt_final"]
        return m.values[0] if len(m) > 0 else df_target["tpt_final"].iloc[default_idx]

    val_2020 = get_val(2020, 0)
    val_2025 = get_val(2025, -2 if len(df_target) >= 2 else 0)
    val_2026 = get_val(2026, -1)

    growth_stats = {
        "tpt_2020": val_2020,
        "tpt_2025_latest": val_2025,
        "tpt_2026_proj": val_2026,
        "delta_2025_vs_2020": round(val_2025 - val_2020, 2),
        "projected_change_2026": round(val_2026 - val_2025, 2),
    }

    return growth_stats
