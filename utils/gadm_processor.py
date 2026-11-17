"""
Modul Pemroses Shapefile GADM (Global Administrative Areas)
-----------------------------------------------------------
Mengekstrak, memfilter hanya wilayah Jawa Tengah (35 Kab/Kota),
menyederhanakan geometri (Shapely simplify) agar ringan dan responsif,
serta mengonversi Shapefile 180MB menjadi GeoJSON (~500KB) yang siap dirender Plotly.
"""

import os
import zipfile
import json
import shapefile
from shapely.geometry import shape, mapping
from shapely.ops import transform
from config import WILAYAH_JATENG

OUTPUT_GEOJSON_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "jawa_tengah.geojson")

def process_gadm_zip_or_shp(file_bytes_or_path, output_path=OUTPUT_GEOJSON_PATH, tolerance=0.003):
    """
    Memproses file Shapefile GADM (berkas .zip atau .shp langsung)
    Filter khusus Provinsi Jawa Tengah dan simpan sebagai GeoJSON ringkas.
    """
    extract_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "_temp_gadm")
    os.makedirs(extract_dir, exist_ok=True)

    shp_path = None

    # Jika input berupa bytes (dari st.file_uploader)
    if hasattr(file_bytes_or_path, "read"):
        temp_zip_path = os.path.join(extract_dir, "uploaded_gadm.zip")
        with open(temp_zip_path, "wb") as f:
            f.write(file_bytes_or_path.getbuffer() if hasattr(file_bytes_or_path, "getbuffer") else file_bytes_or_path.read())
        file_bytes_or_path = temp_zip_path

    # Jika file berupa ZIP, ekstrak isinya
    if isinstance(file_bytes_or_path, str) and file_bytes_or_path.lower().endswith(".zip"):
        with zipfile.ZipFile(file_bytes_or_path, "r") as z:
            z.extractall(extract_dir)

        # Cari file .shp level 2 (Kabupaten/Kota)
        for root, _, files in os.walk(extract_dir):
            for file in files:
                if file.lower().endswith(".shp") and ("2" in file or "kab" in file.lower() or "kota" in file.lower()):
                    shp_path = os.path.join(root, file)
                    break
            if shp_path:
                break
        
        # Fallback jika tidak ada angka 2, ambil .shp pertama yang ditemukan
        if not shp_path:
            for root, _, files in os.walk(extract_dir):
                for file in files:
                    if file.lower().endswith(".shp"):
                        shp_path = os.path.join(root, file)
                        break
                if shp_path:
                    break
    elif isinstance(file_bytes_or_path, str) and file_bytes_or_path.lower().endswith(".shp"):
        shp_path = file_bytes_or_path

    if not shp_path or not os.path.exists(shp_path):
        raise FileNotFoundError("Berkas .shp tidak ditemukan di dalam arsip yang diunggah.")

    # Baca Shapefile menggunakan PyShp
    sf = shapefile.Reader(shp_path)
    fields = [field[0] for field in sf.fields[1:]]
    
    # Identifikasi kolom nama provinsi dan nama kabupaten (Mendukung GADM, BIG, BPS, dan QGIS)
    prov_col = None
    kab_col = None
    for col in fields:
        col_lower = col.lower()
        if col_lower in ["name_1", "provinsi", "province", "propinsi", "wadmpr", "prov"]:
            prov_col = col
        if col_lower in ["name_2", "kabupaten", "kota", "kab_kota", "kabkota", "wadmkk", "namobj", "regency", "district", "nm_kabkota", "wilayah"]:
            kab_col = col

    features = []
    matched_count = 0

    daftar_resmi = list(WILAYAH_JATENG.keys())

    for shape_rec in sf.shapeRecords():
        rec_dict = dict(zip(fields, shape_rec.record))
        
        # Filter Provinsi Jawa Tengah
        is_jateng = False
        if prov_col and rec_dict.get(prov_col):
            prov_val = str(rec_dict[prov_col]).lower()
            if "jawa tengah" in prov_val or "central java" in prov_val:
                is_jateng = True
        else:
            # Jika tidak ada kolom provinsi, cek apakah nama kabupaten cocok dengan daftar Jateng
            is_jateng = True

        if not is_jateng:
            continue

        raw_name = str(rec_dict.get(kab_col, "")).strip() if kab_col else ""
        if not raw_name:
            continue

        # Standardisasi nama wilayah agar cocok dengan 35 Kab/Kota di config
        matched_name = None
        raw_name_clean = raw_name.lower().replace("kabupaten ", "").replace("kab. ", "").replace("kota ", "").strip()
        
        for resmi in daftar_resmi:
            resmi_clean = resmi.lower().replace("kabupaten ", "").replace("kab. ", "").replace("kota ", "").strip()
            if raw_name_clean == resmi_clean:
                # Periksa kecocokan Kota vs Kabupaten
                if "kota" in raw_name.lower() and "kota" in resmi.lower():
                    matched_name = resmi
                    break
                elif "kota" not in raw_name.lower() and "kabupaten" in resmi.lower():
                    matched_name = resmi
                    break
                elif not matched_name:
                    matched_name = resmi

        if not matched_name:
            # Fallback jika belum match
            for resmi in daftar_resmi:
                if raw_name_clean in resmi.lower():
                    matched_name = resmi
                    break

        if not matched_name:
            continue

        # Konversi geometri shapefile ke Shapely object & sederhanakan (simplify)
        geom = shape(shape_rec.shape.__geo_interface__)
        if tolerance and tolerance > 0:
            geom = geom.simplify(tolerance=tolerance, preserve_topology=True)

        feature = {
            "type": "Feature",
            "id": matched_name,
            "properties": {
                "name": matched_name,
                "raw_gadm_name": raw_name,
                "lat": WILAYAH_JATENG.get(matched_name, {}).get("lat", 0),
                "lon": WILAYAH_JATENG.get(matched_name, {}).get("lon", 0),
            },
            "geometry": mapping(geom)
        }
        features.append(feature)
        matched_count += 1

    if len(features) == 0:
        raise ValueError("Tidak ada poligon wilayah Jawa Tengah yang berhasil diidentifikasi dari shapefile ini.")

    geojson_out = {
        "type": "FeatureCollection",
        "features": features
    }

    # Simpan ke target path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson_out, f, ensure_ascii=False)

    return matched_count, output_path
