"""
Modul Pemodelan K-Means Clustering (Machine Learning Unsupervised)
-----------------------------------------------------------------
Menangani standarisasi fitur, penghitungan Elbow Method & Silhouette Score,
eksekusi algoritma K-Means, pemetaan urutan klaster yang konsisten,
reduksi dimensi PCA untuk visualisasi 2D, serta pembuatan profil klaster.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from config import KMEANS_FEATURES

def calculate_elbow_and_silhouette(df_features, feature_cols=None, max_k=8):
    """
    Menghitung inersia (Elbow Method) dan Silhouette Score untuk rentang k = 2 hingga max_k
    guna menentukan jumlah klaster optimal secara ilmiah (rekomendasi skripsi).
    """
    if feature_cols is None:
        feature_cols = KMEANS_FEATURES

    X = df_features[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k_range = list(range(2, max_k + 1))
    inertias = []
    silhouette_scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        score = silhouette_score(X_scaled, kmeans.labels_)
        silhouette_scores.append(round(score, 3))

    return {
        "k_range": k_range,
        "inertias": inertias,
        "silhouette_scores": silhouette_scores,
    }

def fit_kmeans_clustering(df_features, n_clusters=3, feature_cols=None):
    """
    Melakukan proses klasterisasi K-Means dengan penskalaan fitur StandardScaler.
    Menjamin konsistensi pelabelan klaster:
    - Klaster 0: Rata-rata TPT Terendah ("Kondisi Baik")
    - Klaster n-1: Rata-rata TPT Tertinggi ("Prioritas Intervensi")
    """
    if feature_cols is None:
        feature_cols = KMEANS_FEATURES

    df_result = df_features.copy()
    X = df_result[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=15)
    raw_labels = kmeans.fit_predict(X_scaled)

    # Urutkan klaster berdasarkan rata-rata TPT (dari terendah ke tertinggi)
    # agar warna dan makna klaster selalu konsisten antar sesi
    df_temp = pd.DataFrame({"raw_cluster": raw_labels, "tpt": df_result["tpt"]})
    cluster_order = df_temp.groupby("raw_cluster")["tpt"].mean().sort_values().index.tolist()
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(cluster_order)}

    df_result["cluster_id"] = [label_mapping[lbl] for lbl in raw_labels]

    # Beri label semantik yang ramah pengguna
    semantic_names = {
        0: "Klaster Rendah (Kondisi Relatif Baik)",
        1: "Klaster Sedang (Kondisi Waspada)",
        2: "Klaster Tinggi (Prioritas Intervensi)",
    }
    # Jika k > 3, sesuaikan penamaan secara dinamis
    def assign_label(cid):
        if n_clusters == 3:
            return semantic_names.get(cid, f"Klaster {cid}")
        if cid == 0:
            return "Klaster TPT Terendah"
        elif cid == n_clusters - 1:
            return "Klaster TPT Tertinggi (Prioritas)"
        else:
            return f"Klaster Menengah {cid}"

    df_result["nama_klaster"] = df_result["cluster_id"].apply(assign_label)

    # Reduksi Dimensi Menggunakan PCA untuk Proyeksi 2D Scatter
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)
    df_result["pca_x"] = np.round(pca_coords[:, 0], 3)
    df_result["pca_y"] = np.round(pca_coords[:, 1], 3)
    explained_var = [round(var * 100, 1) for var in pca.explained_variance_ratio_]

    return df_result, kmeans, scaler, explained_var

def get_cluster_profiles(df_clustered, feature_cols=None):
    """
    Menghasilkan tabel profil ringkasan statistik (Mean, Min, Max) per klaster
    beserta rekomendasi kebijakan spesifik untuk luaran skripsi.
    """
    if feature_cols is None:
        feature_cols = KMEANS_FEATURES

    summary_df = df_clustered.groupby(["cluster_id", "nama_klaster"])[feature_cols].mean().round(2).reset_index()
    counts = df_clustered.groupby("cluster_id")["wilayah"].count().values
    summary_df["jumlah_wilayah"] = counts

    # Rekomendasi Kebijakan Edukatif / Analisis Skripsi
    recommendations = {
        0: "Fokus mempertahankan stabilitas penyerapan tenaga kerja, penguatan kewirausahaan lokal, dan peningkatan produktivitas berbasis teknologi.",
        1: "Peningkatan program link-and-match SMK/Universitas dengan industri lokal, pelatihan reskilling tenaga kerja, dan stimulus UMKM daerah.",
        2: "Prioritas utama intervensi ketenagakerjaan daerah: pembukaan bursa kerja terpadu (Job Fair), insentif investasi padat karya, serta program padat karya tunai (cash for work).",
    }
    summary_df["rekomendasi_kebijakan"] = summary_df["cluster_id"].apply(lambda c: recommendations.get(c, "Intervensi disesuaikan dengan profil fitur."))

    return summary_df
