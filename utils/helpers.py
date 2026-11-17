"""
Modul Utilitas dan Fungsi Pembantu (Helpers & Styling)
------------------------------------------------------
Menyediakan injeksi CSS kustom ultra-modern (SaaS/Executive Dashboard),
styling hamburger menu responsif, kartu metrik kaca (glassmorphism),
serta fungsi ekspor data CSV.
"""

import streamlit as st
import pandas as pd

def inject_custom_css():
    """
    Menyuntikkan gaya CSS kustom untuk menghasilkan tampilan antarmuka yang
    minimalis, modern, bersih, profesional, dan responsif.
    """
    custom_css = """
    <style>
        /* Import Font Modern: Plus Jakarta Sans & Inter */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

        :root {
            --bg-main: #0B0F19;
            --bg-card: rgba(17, 24, 39, 0.7);
            --bg-card-hover: rgba(30, 41, 59, 0.85);
            --border-subtle: rgba(255, 255, 255, 0.08);
            --border-hover: rgba(99, 102, 241, 0.4);
            --text-primary: #F8FAFC;
            --text-secondary: #94A3B8;
            --text-muted: #64748B;
            --accent-primary: #6366F1;
            --accent-emerald: #10B981;
            --accent-amber: #F59E0B;
            --accent-crimson: #EF4444;
        }

        html, body, [class*="css"], .stApp {
            font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
            background-color: var(--bg-main) !important;
            color: var(--text-primary) !important;
        }

        /* Hilangkan padding default berlebih di atas halaman */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 3rem !important;
            max-width: 95% !important;
        }

        /* =======================================================
           HAMBURGER MENU & SIDEBAR RESPONSIVE STYLING
           ======================================================= */
        /* Tombol Toggle Sidebar Streamlit (Hamburger Look) */
        button[data-testid="stSidebarCollapseButton"],
        [data-testid="baseButton-headerNoPadding"],
        [data-testid="stSidebarHeader"] button {
            background: rgba(30, 41, 59, 0.8) !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: 10px !important;
            color: var(--text-primary) !important;
            backdrop-filter: blur(8px) !important;
            transition: all 0.2s ease-in-out !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25) !important;
        }

        button[data-testid="stSidebarCollapseButton"]:hover,
        [data-testid="baseButton-headerNoPadding"]:hover {
            border-color: var(--accent-primary) !important;
            background: rgba(99, 102, 241, 0.25) !important;
            transform: scale(1.05) !important;
        }

        /* Sidebar Styling: Clean Dark Glass */
        section[data-testid="stSidebar"] {
            background-color: #090D16 !important;
            border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
            box-shadow: 4px 0 24px rgba(0, 0, 0, 0.4) !important;
        }

        /* Navigasi Radio di Sidebar (Pill Style) */
        div[data-testid="stSidebar"] div[role="radiogroup"] {
            gap: 6px !important;
        }

        div[data-testid="stSidebar"] div[role="radiogroup"] label {
            background: rgba(255, 255, 255, 0.02) !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
            padding: 10px 14px !important;
            border-radius: 10px !important;
            transition: all 0.2s ease !important;
            margin-bottom: 2px !important;
        }

        div[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background: rgba(99, 102, 241, 0.1) !important;
            border-color: rgba(99, 102, 241, 0.3) !important;
            transform: translateX(3px) !important;
        }

        div[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.25) 0%, rgba(59, 130, 246, 0.15) 100%) !important;
            border-color: var(--accent-primary) !important;
            box-shadow: 0 0 16px rgba(99, 102, 241, 0.2) !important;
        }

        /* =======================================================
           HEADER & BRANDING STYLING
           ======================================================= */
        .executive-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 16px;
            padding: 22px 28px;
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(17, 24, 39, 0.85) 100%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(12px);
            box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
            margin-bottom: 24px;
        }

        .header-title {
            font-size: 24px;
            font-weight: 800;
            color: #FFFFFF;
            letter-spacing: -0.4px;
            margin: 0;
            line-height: 1.2;
        }

        .header-subtitle {
            font-size: 13px;
            color: #94A3B8;
            margin-top: 6px;
            font-weight: 400;
        }

        .header-meta {
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
        }

        .meta-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 5px 12px;
            border-radius: 9999px;
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.3px;
        }

        .meta-pill-indigo {
            background: rgba(99, 102, 241, 0.12);
            color: #A5B4FC;
            border: 1px solid rgba(99, 102, 241, 0.25);
        }

        .meta-pill-emerald {
            background: rgba(16, 185, 129, 0.12);
            color: #6EE7B7;
            border: 1px solid rgba(16, 185, 129, 0.25);
        }

        /* =======================================================
           MODERN METRIC CARD (GLASSMORPHISM)
           ======================================================= */
        .metric-card-pro {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            padding: 18px 20px;
            border-radius: 16px;
            backdrop-filter: blur(12px);
            transition: all 0.25s ease;
            position: relative;
            overflow: hidden;
            height: 100%;
        }

        .metric-card-pro:hover {
            border-color: var(--border-hover);
            transform: translateY(-3px);
            box-shadow: 0 12px 24px -6px rgba(0, 0, 0, 0.4);
        }

        .metric-card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 10px;
        }

        .metric-title {
            font-size: 11px;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.8px;
        }

        .metric-icon-wrap {
            width: 32px;
            height: 32px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.06);
            color: var(--text-primary);
        }

        .metric-val {
            font-size: 28px;
            font-weight: 800;
            color: #FFFFFF;
            letter-spacing: -0.5px;
            line-height: 1.1;
        }

        .metric-delta-row {
            display: flex;
            align-items: center;
            gap: 6px;
            margin-top: 8px;
            font-size: 12px;
            font-weight: 500;
        }

        .delta-badge-green {
            color: #34D399;
            background: rgba(16, 185, 129, 0.12);
            padding: 2px 8px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 11px;
        }

        .delta-badge-neutral {
            color: #93C5FD;
            background: rgba(59, 130, 246, 0.12);
            padding: 2px 8px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 11px;
        }

        .delta-badge-red {
            color: #F87171;
            background: rgba(239, 68, 68, 0.12);
            padding: 2px 8px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 11px;
        }

        /* =======================================================
           CLUSTER SUMMARY BADGES
           ======================================================= */
        .cluster-pill-card {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 14px;
            padding: 14px 18px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            transition: all 0.2s ease;
        }

        .cluster-pill-card:hover {
            border-color: rgba(255, 255, 255, 0.15);
            background: var(--bg-card-hover);
        }

        .cluster-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .pulse-dot {
            width: 9px;
            height: 9px;
            border-radius: 50%;
            display: inline-block;
        }

        .dot-green { background: #10B981; box-shadow: 0 0 10px #10B981; }
        .dot-yellow { background: #F59E0B; box-shadow: 0 0 10px #F59E0B; }
        .dot-red { background: #EF4444; box-shadow: 0 0 10px #EF4444; }

        .cluster-name-text {
            font-size: 12px;
            font-weight: 600;
            color: var(--text-primary);
        }

        .cluster-count-badge {
            font-size: 18px;
            font-weight: 800;
            color: #FFFFFF;
        }

        .cluster-count-sub {
            font-size: 11px;
            font-weight: 400;
            color: var(--text-muted);
            margin-left: 2px;
        }

        /* =======================================================
           MAP & CARD CONTAINER STYLING
           ======================================================= */
        .content-card-wrap {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 18px;
            padding: 20px;
            margin-bottom: 20px;
            backdrop-filter: blur(8px);
        }

        /* Modern Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px !important;
            border-bottom: 1px solid var(--border-subtle) !important;
            padding-bottom: 6px !important;
        }

        .stTabs [data-baseweb="tab"] {
            background: transparent !important;
            border: 1px solid transparent !important;
            border-radius: 8px !important;
            padding: 8px 16px !important;
            color: var(--text-secondary) !important;
            font-size: 13px !important;
            font-weight: 500 !important;
        }

        .stTabs [aria-selected="true"] {
            background: rgba(99, 102, 241, 0.15) !important;
            border-color: rgba(99, 102, 241, 0.3) !important;
            color: #FFFFFF !important;
            font-weight: 600 !important;
        }

        /* Radio Buttons horizontal look like toggle pills */
        div[data-testid="stRadio"] div[role="radiogroup"] {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }

        /* Form Inputs & Sliders */
        div[data-baseweb="select"] > div,
        div[data-testid="stSlider"] {
            border-radius: 10px !important;
        }

        /* Scrollbar Halus */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: #090D16;
        }
        ::-webkit-scrollbar-thumb {
            background: #1E293B;
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #334155;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def convert_df_to_csv(df):
    """
    Mengonversi pandas DataFrame ke bytes CSV yang siap diunduh.
    """
    return df.to_csv(index=False).encode("utf-8")
