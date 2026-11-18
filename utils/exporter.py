"""
Modul Generator Laporan Excel & Data Estetik (Executive Academic Exporter)
--------------------------------------------------------------------------
Menghasilkan berkas Excel (.xlsx) dengan tata letak profesional, tipografi rapi,
palet warna korporat/akademik, border halus, auto-fit lebar kolom, dan penyorotan
status klaster untuk lampiran sidang skripsi.
"""

import io
from datetime import datetime
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# Skema Palet Warna Excel Estetik
PRIMARY_COLOR = "1E3A8A"      # Deep Navy
ACCENT_COLOR = "3B82F6"       # Royal Blue
HEADER_FILL = "1E293B"        # Dark Slate
ZEBRA_FILL = "F8FAFC"         # Very Light Gray
BORDER_COLOR = "CBD5E1"       # Light Slate Border

# Warna Klaster
CLUSTER_FILLS = {
    "rendah": {"fill": "DCFCE7", "font": "166534"},  # Emerald Green Soft
    "sedang": {"fill": "FEF9C3", "font": "854D0E"},  # Amber Yellow Soft
    "tinggi": {"fill": "FEE2E2", "font": "991B1B"},  # Crimson Red Soft
}

def _apply_sheet_header(ws, title_text, subtitle_text):
    """
    Menambahkan blok header dokumen skripsi pada bagian atas lembar kerja.
    """
    ws.views.sheetView[0].showGridLines = True

    # Row 1: Banner Judul Utama
    ws.merge_cells("A1:G1")
    title_cell = ws["A1"]
    title_cell.value = title_text.upper()
    title_cell.font = Font(name="Calibri", size=14, bold=True, color="FFFFFF")
    title_cell.fill = PatternFill(start_color=PRIMARY_COLOR, end_color=PRIMARY_COLOR, fill_type="solid")
    title_cell.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 28

    # Row 2: Subjudul
    ws.merge_cells("A2:G2")
    sub_cell = ws["A2"]
    sub_cell.value = subtitle_text
    sub_cell.font = Font(name="Calibri", size=10, italic=True, color="475569")
    sub_cell.fill = PatternFill(start_color="F1F5F9", end_color="F1F5F9", fill_type="solid")
    sub_cell.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[2].height = 18

    # Row 3: Metadata Peneliti Skripsi
    ws.merge_cells("A3:G3")
    meta_cell = ws["A3"]
    meta_cell.value = f"Peneliti: Muhammad Safiq (NIM: 232520069) | PSTI Univ. Muhammadiyah Purworejo | Tanggal Cetak: {datetime.now().strftime('%d %B %Y')}"
    meta_cell.font = Font(name="Calibri", size=9, color="64748B")
    meta_cell.fill = PatternFill(start_color="F1F5F9", end_color="F1F5F9", fill_type="solid")
    meta_cell.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[3].height = 16

    # Row 4: Blank Spacing
    ws.row_dimensions[4].height = 8

def _style_table(ws, start_row, df, num_cols_align_right=None, cluster_col_idx=None):
    """
    Menata tabel data dengan format header gelap, zebra row, border, dan auto-width.
    """
    thin_border = Border(
        left=Side(style="thin", color=BORDER_COLOR),
        right=Side(style="thin", color=BORDER_COLOR),
        top=Side(style="thin", color=BORDER_COLOR),
        bottom=Side(style="thin", color=BORDER_COLOR)
    )

    # 1. Header Tabel
    ws.row_dimensions[start_row].height = 24
    for col_idx, col_name in enumerate(df.columns, start=1):
        cell = ws.cell(row=start_row, column=col_idx)
        cell.value = col_name
        cell.font = Font(name="Calibri", size=10, bold=True, color="FFFFFF")
        cell.fill = PatternFill(start_color=HEADER_FILL, end_color=HEADER_FILL, fill_type="solid")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = thin_border

    # 2. Baris Data
    data_start_row = start_row + 1
    for r_idx, (_, row) in enumerate(df.iterrows(), start=data_start_row):
        ws.row_dimensions[r_idx].height = 20
        is_even = (r_idx % 2 == 0)
        row_fill = PatternFill(start_color=ZEBRA_FILL, end_color=ZEBRA_FILL, fill_type="solid") if is_even else PatternFill(fill_type=None)

        for c_idx, val in enumerate(row, start=1):
            cell = ws.cell(row=r_idx, column=c_idx)
            cell.value = val
            cell.font = Font(name="Calibri", size=10, color="0F172A")
            cell.border = thin_border
            cell.fill = row_fill

            # Alignment
            if isinstance(val, (int, float)):
                cell.alignment = Alignment(horizontal="right", vertical="center")
                # Format persentase atau desimal
                if "tpt" in df.columns[c_idx - 1].lower() or "%" in df.columns[c_idx - 1].lower():
                    cell.number_format = "0.00"
                elif isinstance(val, float):
                    cell.number_format = "0.00"
            elif c_idx == 1:
                # No / Urut
                cell.alignment = Alignment(horizontal="center", vertical="center")
            else:
                cell.alignment = Alignment(horizontal="left", vertical="center")

            # Pewarnaan Khusus Kolom Klaster
            if cluster_col_idx and c_idx == cluster_col_idx:
                val_str = str(val).lower()
                for key, style in CLUSTER_FILLS.items():
                    if key in val_str:
                        cell.fill = PatternFill(start_color=style["fill"], end_color=style["fill"], fill_type="solid")
                        cell.font = Font(name="Calibri", size=10, bold=True, color=style["font"])
                        break

    # 3. Auto-Fit Column Width
    for col in ws.columns:
        max_len = 0
        col_letter = get_column_letter(col[0].column)
        for cell in col:
            # Abaikan baris merger header (1, 2, 3)
            if cell.row < start_row:
                continue
            if cell.value:
                val_str = str(cell.value)
                max_len = max(max_len, len(val_str))
        ws.column_dimensions[col_letter].width = max(max_len + 4, 12)

def generate_aesthetic_kmeans_excel(df_clustered, cluster_profiles=None):
    """
    Menghasilkan file Excel estetik untuk luaran klasterisasi K-Means Jawa Tengah.
    """
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Hasil Klasterisasi K-Means"

    _apply_sheet_header(
        ws,
        title_text="HASIL ANALISIS KLASTERISASI K-MEANS TPT PROVINSI JAWA TENGAH",
        subtitle_text="Segmentasi Tipologi Ketenagakerjaan Daerah (35 Kabupaten/Kota) Berbasis Fitur Multidimensi"
    )

    # Siapkan DataFrame yang rapi
    df_export = df_clustered.copy()
    columns_map = {
        "wilayah": "Kabupaten / Kota",
        "tipe_wilayah": "Tipe",
        "tpt": "TPT Terkini (%)",
        "ipm": "Indeks Pembangunan Manusia (IPM)",
        "pertumbuhan_ekonomi": "Pertumbuhan Ekonomi (%)",
        "tpak": "TPAK (%)",
        "upah_minimum": "UMK (Juta Rp)",
        "nama_klaster": "Tipologi Klaster K-Means"
    }

    cols_exist = [c for c in columns_map.keys() if c in df_export.columns]
    df_clean = df_export[cols_exist].rename(columns={c: columns_map[c] for c in cols_exist})
    df_clean.insert(0, "No", range(1, len(df_clean) + 1))

    # Tentukan index kolom nama_klaster
    cluster_idx = None
    for i, col in enumerate(df_clean.columns, start=1):
        if "klaster" in col.lower():
            cluster_idx = i
            break

    _style_table(ws, start_row=5, df=df_clean, cluster_col_idx=cluster_idx)

    # Simpan ke buffer in-memory
    output = io.BytesIO()
    wb.save(output)
    output.seek(0)
    return output.getvalue()

def generate_aesthetic_lstm_excel(df_panel):
    """
    Menghasilkan file Excel estetik untuk hasil peramalan LSTM 2026 dan time series historis.
    """
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Peramalan LSTM 2026"

    _apply_sheet_header(
        ws,
        title_text="HASIL PERAMALAN TPT MODEL DEEP LEARNING LSTM TAHUN 2026",
        subtitle_text="Proyeksi Tingkat Pengangguran Terbuka 35 Kabupaten/Kota Provinsi Jawa Tengah & Data Historis"
    )

    # Pivot atau rekap per wilayah
    wilayah_list = sorted(df_panel["wilayah"].unique())
    records = []

    for idx, w in enumerate(wilayah_list, start=1):
        w_data = df_panel[df_panel["wilayah"] == w]
        tipe = w_data["tipe_wilayah"].iloc[0] if "tipe_wilayah" in w_data.columns else "Kabupaten"
        
        row_dict = {
            "No": idx,
            "Kabupaten / Kota": w,
            "Tipe": tipe
        }

        # Kolom tahun historis (2020-2025)
        for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
            val_hist = w_data[w_data["tahun"] == yr]["tpt_aktual"]
            row_dict[f"TPT {yr} (%)"] = round(float(val_hist.values[0]), 2) if len(val_hist) > 0 and pd.notnull(val_hist.values[0]) else None

        # Kolom Prediksi LSTM 2026
        pred_2026 = w_data[w_data["tahun"] == 2026]["tpt_prediksi_lstm"]
        pred_val = round(float(pred_2026.values[0]), 2) if len(pred_2026) > 0 and pd.notnull(pred_2026.values[0]) else None
        row_dict["Prediksi LSTM 2026 (%)"] = pred_val

        # Interval Kepercayaan
        low_ci = w_data[w_data["tahun"] == 2026]["lower_ci"]
        up_ci = w_data[w_data["tahun"] == 2026]["upper_ci"]
        row_dict["Batas Bawah 95%"] = round(float(low_ci.values[0]), 2) if len(low_ci) > 0 and pd.notnull(low_ci.values[0]) else None
        row_dict["Batas Atas 95%"] = round(float(up_ci.values[0]), 2) if len(up_ci) > 0 and pd.notnull(up_ci.values[0]) else None

        # Arah Perubahan vs 2025
        val_2025 = row_dict.get("TPT 2025 (%)")
        if pred_val is not None and val_2025 is not None:
            delta = round(pred_val - val_2025, 2)
            row_dict["Estimasi Perubahan (poin)"] = delta
            row_dict["Tren"] = "Turun (Membaik)" if delta < 0 else ("Naik (Waspada)" if delta > 0 else "Stabil")
        else:
            row_dict["Estimasi Perubahan (poin)"] = 0.0
            row_dict["Tren"] = "Stabil"

        records.append(row_dict)

    df_export = pd.DataFrame(records)
    _style_table(ws, start_row=5, df=df_export)

    output = io.BytesIO()
    wb.save(output)
    output.seek(0)
    return output.getvalue()
