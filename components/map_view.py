"""
Komponen Visualisasi Spasial (Choropleth Mapping Jawa Tengah)
------------------------------------------------------------
Mengintegrasikan peta kloropleth interaktif wilayah Jawa Tengah berbasis Plotly Mapbox
untuk menampilkan sebaran spasial TPT antar tahun serta peta zonasi klaster K-Means.
"""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from data.geojson_jateng import get_jateng_geojson
from config import CLUSTER_COLORS

def render_choropleth_map(df_panel, df_clustered, selected_year=2024, map_mode="tpt"):
    """
    Menampilkan peta kloropleth interaktif Jawa Tengah.
    
    Parameter:
    - df_panel: DataFrame time series TPT (2015-2027)
    - df_clustered: DataFrame hasil klasterisasi K-Means
    - selected_year: Tahun yang dipilih pengguna
    - map_mode: 'tpt' (gradasi nilai TPT) atau 'cluster' (warna zonasi klaster)
    """
    geojson_data = get_jateng_geojson()

    # Gabungkan data panel tahun terpilih dengan klaster K-Means
    df_map = df_panel[df_panel["tahun"] == selected_year].copy()
    df_map = df_map.merge(
        df_clustered[["wilayah", "cluster_id", "nama_klaster", "ipm", "pertumbuhan_ekonomi", "upah_minimum"]],
        on="wilayah",
        how="left"
    )

    # Tentukan nilai TPT yang ditampilkan (Aktual untuk historis, Prediksi untuk tahun proyeksi 2026)
    if "tpt_prediksi_lstm" in df_map.columns:
        df_map["tpt_tampil"] = df_map["tpt_aktual"].fillna(df_map["tpt_prediksi_lstm"])
    else:
        df_map["tpt_tampil"] = df_map["tpt_aktual"]
        
    df_map["tpt_tampil"] = pd.to_numeric(df_map["tpt_tampil"], errors="coerce")
    df_map["status_data"] = df_map["tipe_data"] if "tipe_data" in df_map.columns else ("Historis" if selected_year <= 2025 else "Proyeksi LSTM")

    # Plotly v7+ menggunakan px.choropleth_map (bukan choropleth_mapbox)
    # dengan parameter map_style menggantikan mapbox_style
    if map_mode == "tpt":
        # Gradasi warna berdasarkan tingkat TPT (Yellow-Orange-Red)
        fig = px.choropleth_map(
            df_map,
            geojson=geojson_data,
            locations="wilayah",
            featureidkey="properties.name",
            color="tpt_tampil",
            color_continuous_scale="YlOrRd",
            range_color=[df_map["tpt_tampil"].min() * 0.9, df_map["tpt_tampil"].max() * 1.05],
            map_style="carto-positron",
            zoom=7.3,
            center={"lat": -7.15, "lon": 110.15},
            opacity=0.82,
            hover_name="wilayah",
            hover_data={
                "wilayah": False,
                "tpt_tampil": ":.2f",
                "status_data": True,
                "nama_klaster": True,
                "ipm": ":.2f",
                "pertumbuhan_ekonomi": ":.2f%"
            },
            labels={
                "tpt_tampil": f"TPT {selected_year} (%)",
                "status_data": "Status Data",
                "nama_klaster": "Klaster K-Means",
                "ipm": "IPM",
                "pertumbuhan_ekonomi": "Pertumbuhan"
            }
        )
        fig.update_layout(
            margin={"r": 0, "t": 40, "l": 0, "b": 0},
            title_text=f"<b>Peta Distribusi Spasial TPT Jawa Tengah Tahun {selected_year}</b>",
            title_x=0.02,
            coloraxis_colorbar=dict(
                title="TPT (%)",
                thicknessmode="pixels",
                thickness=15,
                lenmode="pixels",
                len=240,
                yanchor="middle",
                y=0.5,
            )
        )
    else:
        # Peta Zonasi Klaster K-Means (Sesuai Layout GIS Skripsi: Hijau, Kuning, Merah)
        color_discrete_map = {
            "Klaster Rendah (Kondisi Relatif Baik)": "#22C55E",
            "Klaster Sedang (Kondisi Waspada)": "#EAB308",
            "Klaster Tinggi (Prioritas Intervensi)": "#EF4444",
            "Klaster TPT Terendah": "#22C55E",
            "Klaster TPT Tertinggi (Prioritas)": "#EF4444",
        }
        fig = px.choropleth_map(
            df_map,
            geojson=geojson_data,
            locations="wilayah",
            featureidkey="properties.name",
            color="nama_klaster",
            color_discrete_map=color_discrete_map,
            map_style="carto-positron",
            zoom=7.3,
            center={"lat": -7.15, "lon": 110.15},
            opacity=0.90,
            hover_name="wilayah",
            hover_data={
                "wilayah": False,
                "tpt_tampil": ":.2f",
                "ipm": ":.2f",
                "pertumbuhan_ekonomi": ":.2f%",
                "upah_minimum": ":.2f Jt"
            },
            labels={
                "nama_klaster": "Klaster K-Means",
                "tpt_tampil": "TPT (%)",
                "ipm": "IPM",
                "pertumbuhan_ekonomi": "Pertumbuhan (%)",
                "upah_minimum": "UMK (Jt Rp)"
            }
        )
        fig.update_layout(
            margin={"r": 0, "t": 40, "l": 0, "b": 0},
            title_text=f"<b>Peta Zonasi Klaster K-Means Wilayah Jawa Tengah</b>",
            title_x=0.02,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

    # Berikan garis batas antar kabupaten yang tajam & jelas seperti layout GIS
    fig.update_traces(
        marker_line_width=1.2,
        marker_line_color="#0F172A",
        selector=dict(type="choroplethmap")
    )

    # Tambahkan layer scatter / label penanda titik pusat kota/kabupaten
    # Plotly v7+ menggunakan go.Scattermap (bukan go.Scattermapbox)
    fig.add_trace(
        go.Scattermap(
            lat=df_map["lat"],
            lon=df_map["lon"],
            mode="text",
            text=df_map["wilayah"].str.replace("Kabupaten ", "Kab. ").str.replace("Kota ", "Kota "),
            textposition="top center",
            textfont=dict(size=9, color="#1E293B"),
            hoverinfo="none",
            showlegend=False
        )
    )

    fig.update_layout(
        height=520,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig
