"""
Modul GeoJSON Wilayah Jawa Tengah
---------------------------------
Menyediakan poligon batas wilayah atau titik representatif untuk 35 Kabupaten/Kota
di Provinsi Jawa Tengah, baik melalui berkas GeoJSON eksternal (jika tersedia)
maupun melalui representasi spasial bawaan.
"""

import json
import os
import math
from config import WILAYAH_JATENG

GEOJSON_FILE_PATH = os.path.join(os.path.dirname(__file__), "raw", "jawa_tengah.geojson")

def generate_fallback_geojson():
    """
    Membuat GeoJSON sintetis berbasis centroid kabupaten/kota Jawa Tengah
    dengan poligon teratur (bounding octagons) agar Plotly Choropleth Mapbox
    dapat merender visualisasi wilayah secara mandiri tanpa dependensi internet.
    """
    features = []
    
    # Delta derajat ~ 12-15 km radius representasi wilayah
    delta = 0.11
    
    for nama_wilayah, info in WILAYAH_JATENG.items():
        lat = info["lat"]
        lon = info["lon"]
        
        # Buat poligon 8 sisi (oktagon) di sekitar pusat koordinat
        coords = []
        for i in range(8):
            angle = i * (2 * math.pi / 8)
            # Berikan variasi rasio agar tidak berbentuk lingkaran sempurna
            r_lat = delta * (0.85 + 0.15 * math.sin(i * 2))
            r_lon = delta * 1.15 * (0.85 + 0.15 * math.cos(i * 2))
            
            p_lat = lat + r_lat * math.sin(angle)
            p_lon = lon + r_lon * math.cos(angle)
            coords.append([round(p_lon, 5), round(p_lat, 5)])
            
        # Tutup poligon
        coords.append(coords[0])
        
        feature = {
            "type": "Feature",
            "id": nama_wilayah,
            "properties": {
                "name": nama_wilayah,
                "tipe": info["tipe"],
                "lat": lat,
                "lon": lon
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords]
            }
        }
        features.append(feature)
        
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    return geojson

def get_jateng_geojson():
    """
    Mengambil data GeoJSON Jawa Tengah.
    Memprioritaskan file eksternal jika ada di data/raw/jawa_tengah.geojson,
    jika tidak ada akan menggunakan generator bawaan.
    """
    if os.path.exists(GEOJSON_FILE_PATH):
        try:
            with open(GEOJSON_FILE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Peringatan] Gagal memuat berkas GeoJSON lokal: {e}. Menggunakan fallback.")
            
    return generate_fallback_geojson()
