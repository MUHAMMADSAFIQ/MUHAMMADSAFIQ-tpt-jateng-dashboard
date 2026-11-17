"""
Komponen Visualisasi Klasterisasi K-Means (Unsupervised Learning)
-----------------------------------------------------------------
Menyajikan grafik interaktif Elbow Method, Silhouette Score, proyeksi 2D PCA,
Radar Chart karakteristik klaster, serta boxplot distribusi fitur antar klaster.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def render_elbow_silhouette_chart(elbow_data, optimal_k=3):
    """
    Menampilkan grafik kombinasi Elbow Method (Inersia WCSS) dan Silhouette Score
    untuk membuktikan justifikasi ilmiah pemilihan jumlah klaster optimal (k).
    """
    k_vals = elbow_data["k_range"]
    inertias = elbow_data["inertias"]
    silhouettes = elbow_data["silhouette_scores"]

    # Buat subplot dengan 2 sumbu Y
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 1. Garis Inersia (Elbow / WCSS)
    fig.add_trace(
        go.Scatter(
            x=k_vals,
            y=inertias,
            name="Inersia (Elbow WCSS)",
            mode="lines+markers",
            line=dict(color="#1E3A8A", width=3),
            marker=dict(size=8, symbol="circle"),
            hovertemplate="k = %{x}<br>Inersia: %{y:.2f}<extra></extra>"
        ),
        secondary_y=False
    )

    # 2. Garis Silhouette Score
    fig.add_trace(
        go.Scatter(
            x=k_vals,
            y=silhouettes,
            name="Silhouette Score",
            mode="lines+markers",
            line=dict(color="#10B981", width=2.5, dash="dash"),
            marker=dict(size=8, symbol="diamond"),
            hovertemplate="k = %{x}<br>Silhouette: %{y:.3f}<extra></extra>"
        ),
        secondary_y=True
    )

    # Tandai K Optimal terpilih
    fig.add_vline(
        x=optimal_k,
        line_width=2,
        line_dash="dot",
        line_color="#EF4444",
        annotation_text=f"k={optimal_k} Optimal",
        annotation_position="top left",
        annotation_font=dict(color="#EF4444", size=11)
    )

    fig.update_layout(
        title="<b>Evaluasi Penentuan Jumlah Klaster Optimal (Elbow & Silhouette)</b>",
        xaxis=dict(title="Jumlah Klaster (k)", tickmode="linear", dtick=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="center", x=0.5),
        height=360,
        margin=dict(l=40, r=40, t=60, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )

    fig.update_yaxes(title_text="Inersia (Sum of Squared Distances)", secondary_y=False)
    fig.update_yaxes(title_text="Silhouette Score", secondary_y=True, range=[0, 0.7])

    return fig

def render_pca_scatter_chart(df_clustered, explained_var):
    """
    Menampilkan scatter plot 2D hasil reduksi dimensi PCA untuk memvisualisasikan
    separasi dan kedekatan antar klaster wilayah.
    """
    color_map = {
        "Klaster Rendah (Kondisi Relatif Baik)": "#10B981",
        "Klaster Sedang (Kondisi Waspada)": "#F59E0B",
        "Klaster Tinggi (Prioritas Intervensi)": "#EF4444",
        "Klaster TPT Terendah": "#10B981",
        "Klaster TPT Tertinggi (Prioritas)": "#EF4444",
    }

    fig = px.scatter(
        df_clustered,
        x="pca_x",
        y="pca_y",
        color="nama_klaster",
        color_discrete_map=color_map,
        hover_name="wilayah",
        hover_data={
            "pca_x": False,
            "pca_y": False,
            "tpt": ":.2f%",
            "ipm": ":.2f",
            "pertumbuhan_ekonomi": ":.2f%",
            "upah_minimum": ":.2f Jt",
            "nama_klaster": False
        },
        title=f"<b>Proyeksi Ruang Fitur 2D (PCA) Klaster Wilayah Jawa Tengah</b><br><sup>Komponen Utama 1 ({explained_var[0]}%) & Komponen Utama 2 ({explained_var[1]}%)</sup>",
        labels={"pca_x": f"PC 1 ({explained_var[0]}% Variansi)", "pca_y": f"PC 2 ({explained_var[1]}% Variansi)"}
    )

    fig.update_traces(
        marker=dict(size=12, line=dict(width=1.5, color="#FFFFFF")),
        selector=dict(mode="markers")
    )

    fig.update_layout(
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=40, r=20, t=70, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )

    return fig

def render_radar_cluster_chart(df_clustered, feature_cols=None):
    """
    Menampilkan Radar Chart (Spider Plot) untuk membandingkan karakteristik rata-rata
    fitur sosio-ekonomi per klaster yang telah dinormalisasi min-max (0 - 1).
    """
    if feature_cols is None:
        feature_cols = ["tpt", "ipm", "pertumbuhan_ekonomi", "tpak", "upah_minimum"]

    labels_id = {
        "tpt": "Tingkat Pengangguran",
        "ipm": "Indeks Pembangunan Manusia",
        "pertumbuhan_ekonomi": "Pertumbuhan Ekonomi",
        "tpak": "Partisipasi Angkatan Kerja",
        "upah_minimum": "Upah Minimum (UMK)"
    }

    # Normalisasi fitur ke skala 0 - 1 untuk perbandingan adil pada radar chart
    df_norm = df_clustered.copy()
    for col in feature_cols:
        c_min = df_norm[col].min()
        c_max = df_norm[col].max()
        df_norm[col] = (df_norm[col] - c_min) / (c_max - c_min + 1e-6)

    mean_per_cluster = df_norm.groupby("nama_klaster")[feature_cols].mean()

    fig = go.Figure()
    colors = ["#10B981", "#F59E0B", "#EF4444", "#6366F1"]

    categories = [labels_id.get(col, col) for col in feature_cols]
    categories_closed = categories + [categories[0]]

    for idx, (klaster_name, row) in enumerate(mean_per_cluster.iterrows()):
        values = row.tolist()
        values_closed = values + [values[0]]

        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            fill="toself",
            name=klaster_name,
            line=dict(color=colors[idx % len(colors)], width=2.5),
            opacity=0.65
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0.25, 0.5, 0.75, 1.0],
                ticktext=["Rendah", "Sedang", "Tinggi", "Maksimal"]
            )
        ),
        title="<b>Karakteristik Tipologi Klaster Wilayah (Radar Profil)</b>",
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="center", x=0.5),
        height=450,
        margin=dict(l=40, r=40, t=70, b=40),
        paper_bgcolor="rgba(0,0,0,0)"
    )

    return fig

def render_feature_boxplots(df_clustered):
    """
    Menampilkan visualisasi boxplot perbandingan distribusi TPT dan IPM antar klaster.
    """
    color_map = {
        "Klaster Rendah (Kondisi Relatif Baik)": "#10B981",
        "Klaster Sedang (Kondisi Waspada)": "#F59E0B",
        "Klaster Tinggi (Prioritas Intervensi)": "#EF4444",
        "Klaster TPT Terendah": "#10B981",
        "Klaster TPT Tertinggi (Prioritas)": "#EF4444",
    }

    fig = px.box(
        df_clustered,
        x="nama_klaster",
        y="tpt",
        color="nama_klaster",
        color_discrete_map=color_map,
        title="<b>Distribusi Sebaran TPT per Klaster Wilayah</b>",
        labels={"nama_klaster": "Klaster K-Means", "tpt": "Tingkat Pengangguran Terbuka (%)"},
        points="all"
    )

    fig.update_layout(
        height=380,
        showlegend=False,
        margin=dict(l=40, r=20, t=50, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )

    return fig
