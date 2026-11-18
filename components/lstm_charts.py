"""
Komponen Visualisasi Peramalan Model LSTM (Long Short-Term Memory)
------------------------------------------------------------------
Menyajikan grafik interaktif peramalan time-series, pita ketidakpastian
(Confidence Interval 95%), evaluasi residual, serta perbandingan antar wilayah.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def render_lstm_forecast_chart(df_wilayah, wilayah_name="Seluruh Jawa Tengah"):
    """
    Menampilkan grafik pergerakan historis TPT (2020-2025), hasil estimasi in-sample LSTM,
    dan proyeksi masa depan (2026) lengkap dengan pita interval kepercayaan 95%.
    """
    fig = go.Figure()

    df_sorted = df_wilayah.sort_values("tahun")
    has_2025 = 2025 in df_sorted["tahun"].values
    hist_cut_year = 2025 if has_2025 else 2024

    df_hist = df_sorted[df_sorted["tahun"] <= hist_cut_year]
    df_proj = df_sorted[df_sorted["tahun"] >= hist_cut_year]  # Mulai dari tahun historis terakhir agar bersambung

    # 1. Garis Data Aktual Historis (2020-2025)
    fig.add_trace(go.Scatter(
        x=df_hist["tahun"],
        y=df_hist["tpt_aktual"],
        mode="lines+markers",
        name="Data Aktual BPS (2020-2025)",
        line=dict(color="#1E3A8A", width=3),
        marker=dict(size=8, symbol="circle", color="#1E3A8A"),
        hovertemplate="<b>Tahun %{x} (Aktual)</b><br>TPT: %{y:.2f}%<extra></extra>"
    ))

    # 2. Garis Estimasi In-Sample LSTM
    fig.add_trace(go.Scatter(
        x=df_hist["tahun"],
        y=df_hist["tpt_prediksi_lstm"],
        mode="lines",
        name="Prediksi In-Sample LSTM",
        line=dict(color="#64748B", width=2, dash="dash"),
        hovertemplate="<b>Tahun %{x} (In-Sample)</b><br>TPT Prediksi: %{y:.2f}%<extra></extra>"
    ))

    # 3. Pita Interval Kepercayaan 95% (Confidence Interval 2026)
    df_ci = df_sorted[df_sorted["tahun"] >= hist_cut_year].copy()
    tpt_cut_val = df_hist[df_hist["tahun"] == hist_cut_year]["tpt_aktual"].values[0]
    ci_years = df_ci["tahun"].tolist()
    ci_upper = [tpt_cut_val if y == hist_cut_year else u for y, u in zip(ci_years, df_ci["upper_ci"])]
    ci_lower = [tpt_cut_val if y == hist_cut_year else l for y, l in zip(ci_years, df_ci["lower_ci"])]

    # Upper bound
    fig.add_trace(go.Scatter(
        x=ci_years,
        y=ci_upper,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip"
    ))
    # Lower bound dengan shading fill
    fig.add_trace(go.Scatter(
        x=ci_years,
        y=ci_lower,
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(245, 158, 11, 0.18)",
        name="Interval Kepercayaan 95%",
        hoverinfo="skip"
    ))

    # 4. Garis Proyeksi Masa Depan LSTM (Fokus 2026)
    proj_vals = [tpt_cut_val if y == hist_cut_year else p for y, p in zip(df_proj["tahun"], df_proj["tpt_prediksi_lstm"])]
    fig.add_trace(go.Scatter(
        x=df_proj["tahun"],
        y=proj_vals,
        mode="lines+markers",
        name="Proyeksi LSTM (2026)",
        line=dict(color="#F59E0B", width=3.5),
        marker=dict(size=10, symbol="diamond", color="#D97706", line=dict(width=2, color="#FFFFFF")),
        hovertemplate="<b>Tahun %{x} (Proyeksi)</b><br>TPT Estimasi: %{y:.2f}%<extra></extra>"
    ))

    # Garis Pembatas Vertikal Titik Awal Proyeksi
    fig.add_vline(
        x=hist_cut_year + 0.5,
        line_width=1.5,
        line_dash="dot",
        line_color="#EF4444",
        annotation_text="Batas Proyeksi",
        annotation_position="top right",
        annotation_font=dict(color="#EF4444", size=11)
    )

    fig.update_layout(
        title=f"<b>Tren Historis (2020-2025) & Proyeksi LSTM (2026): {wilayah_name}</b>",
        xaxis=dict(
            title="Tahun",
            tickmode="linear",
            tick0=2020,
            dtick=1,
            gridcolor="rgba(148, 163, 184, 0.15)"
        ),
        yaxis=dict(
            title="Tingkat Pengangguran Terbuka (%)",
            ticksuffix="%",
            gridcolor="rgba(148, 163, 184, 0.15)"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="center",
            x=0.5
        ),
        hovermode="x unified",
        margin=dict(l=40, r=20, t=80, b=40),
        height=450,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )

    return fig

def render_lstm_residual_chart(df_wilayah):
    """
    Menampilkan grafik selisih residual (Aktual - Prediksi LSTM) untuk memeriksa
    apakah galat model terdistribusi acak di sekitar nol (zero-mean error).
    """
    has_2025 = 2025 in df_wilayah["tahun"].values
    hist_cut = 2025 if has_2025 else 2024
    df_hist = df_wilayah[df_wilayah["tahun"] <= hist_cut].copy()
    residuals = df_hist["tpt_aktual"] - df_hist["tpt_prediksi_lstm"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_hist["tahun"],
        y=residuals,
        marker_color=["#10B981" if r >= 0 else "#EF4444" for r in residuals],
        name="Residual (Aktual - Prediksi)",
        hovertemplate="Tahun %{x}<br>Residual: %{y:.3f}%<extra></extra>"
    ))

    fig.add_hline(y=0, line_color="#94A3B8", line_width=1.5)

    fig.update_layout(
        title="<b>Analisis Residual Model LSTM (Data Historis)</b>",
        xaxis=dict(title="Tahun", tickmode="linear", dtick=1),
        yaxis=dict(title="Residual (%)", ticksuffix="%"),
        height=280,
        margin=dict(l=40, r=20, t=50, b=30),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig

def render_ranking_projection_chart(df_panel, target_year=2026):
    """
    Menampilkan visualisasi peringkat 10 wilayah dengan TPT tertinggi dan terendah pada tahun proyeksi.
    """
    df_target = df_panel[df_panel["tahun"] == target_year].sort_values("tpt_prediksi_lstm", ascending=True)

    top5_lowest = df_target.head(5)
    top5_highest = df_target.tail(5)
    df_combined = pd.concat([top5_lowest, top5_highest])

    fig = px.bar(
        df_combined,
        x="tpt_prediksi_lstm",
        y="wilayah",
        orientation="h",
        color="tipe_wilayah",
        color_discrete_map={"Kota": "#6366F1", "Kabupaten": "#0284C7"},
        title=f"<b>Sorotan Wilayah: Proyeksi TPT Terendah vs Tertinggi ({target_year})</b>",
        labels={"tpt_prediksi_lstm": f"Proyeksi TPT {target_year} (%)", "wilayah": "Wilayah"},
        text="tpt_prediksi_lstm"
    )

    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(
        height=380,
        margin=dict(l=20, r=40, t=50, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig
