"""
=============================================================
  Image Color Distribution Analyzer — Streamlit Web App
  Alemeno Product Management Internship Assignment
=============================================================
  A production-quality single-page dashboard that:
    • Accepts drag-and-drop image uploads (JPG/PNG up to 20 MB)
    • Classifies every pixel into 11 colour categories (HSV rules)
    • Renders interactive pie chart, bar chart, palette strip,
      side-by-side heatmap comparison
    • Provides CSV / JSON / PNG download buttons
    • Supports batch upload with comparison table

  Run:
      streamlit run app.py
=============================================================
"""

import io
import json
import base64
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

# ── Page config (must be FIRST Streamlit call) ─────────────
st.set_page_config(
    page_title="Image Color Analyzer",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════

CATEGORIES = [
    "White", "Black", "Gray", "Red", "Orange",
    "Yellow", "Green", "Blue", "Purple", "Pink", "Brown",
]

COLOR_HEX = {
    "White":  "#F5F5F5",
    "Black":  "#1C1C1C",
    "Gray":   "#888888",
    "Red":    "#E53935",
    "Orange": "#FB8C00",
    "Yellow": "#FDD835",
    "Green":  "#43A047",
    "Blue":   "#1E88E5",
    "Purple": "#8E24AA",
    "Pink":   "#EC407A",
    "Brown":  "#6D4C41",
}

# Pure RGB used when recolouring pixels in the heatmap
COLOR_RGB = {
    "White":  (245, 245, 245),
    "Black":  (28,  28,  28 ),
    "Gray":   (136, 136, 136),
    "Red":    (229, 57,  53 ),
    "Orange": (251, 140, 0  ),
    "Yellow": (253, 216, 53 ),
    "Green":  (67,  160, 71 ),
    "Blue":   (30,  136, 229),
    "Purple": (142, 36,  170),
    "Pink":   (236, 64,  122),
    "Brown":  (109, 76,  65 ),
}

MAX_FILE_MB = 20

# ═══════════════════════════════════════════════════════════
#  GLOBAL CSS  – modern dashboard aesthetic
# ═══════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ── Base ──────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background: #0F1117;
    color: #E8EAF0;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
[data-testid="stHeader"]  { background: transparent; }
[data-testid="stToolbar"] { display: none; }

/* ── Cards ─────────────────────────────────────────────── */
.card {
    background: #1A1D2E;
    border: 1px solid #2A2D3E;
    border-radius: 14px;
    padding: 22px 26px;
    margin-bottom: 18px;
}
.card-sm {
    background: #1A1D2E;
    border: 1px solid #2A2D3E;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
}

/* ── Hero header ───────────────────────────────────────── */
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #7C6FFF, #FF6FD8, #FFB347);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
}
.hero-sub {
    color: #9095A8;
    font-size: 1.05rem;
    margin-bottom: 30px;
}

/* ── Section labels ────────────────────────────────────── */
.section-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #7C6FFF;
    margin-bottom: 10px;
}

/* ── Stat boxes ────────────────────────────────────────── */
.stat-box {
    background: #22263A;
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
}
.stat-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #E8EAF0;
}
.stat-label {
    font-size: 0.75rem;
    color: #8085A0;
    margin-top: 2px;
}

/* ── Dominant colour card ──────────────────────────────── */
.dominant-card {
    border-radius: 12px;
    padding: 20px 24px;
    color: white;
    font-weight: 700;
    font-size: 1.25rem;
}

/* ── Colour table ──────────────────────────────────────── */
.color-row {
    display: flex;
    align-items: center;
    padding: 7px 0;
    border-bottom: 1px solid #2A2D3E;
    gap: 12px;
}
.color-swatch {
    width: 18px; height: 18px;
    border-radius: 4px;
    flex-shrink: 0;
}
.color-name  { flex: 1; font-size: 0.92rem; }
.color-pct   { font-size: 0.92rem; color: #9095A8; }
.color-bar-bg {
    flex: 2;
    height: 8px;
    background: #2A2D3E;
    border-radius: 4px;
    overflow: hidden;
}
.color-bar-fill {
    height: 100%;
    border-radius: 4px;
}

/* ── Palette strip ─────────────────────────────────────── */
.palette-outer {
    border-radius: 10px;
    overflow: hidden;
    display: flex;
    height: 54px;
    margin-top: 8px;
}
.palette-segment {
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.65rem;
    font-weight: 700;
    color: rgba(255,255,255,0.85);
    transition: flex 0.4s;
    white-space: nowrap;
    overflow: hidden;
}

/* ── Upload area ────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    border: 2px dashed #3A3D50 !important;
    border-radius: 14px !important;
    padding: 30px !important;
    background: #141623 !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #7C6FFF !important;
}

/* ── Buttons ────────────────────────────────────────────── */
.stDownloadButton > button {
    background: #22263A !important;
    border: 1px solid #3A3D50 !important;
    border-radius: 8px !important;
    color: #E8EAF0 !important;
    font-size: 0.83rem !important;
}
.stDownloadButton > button:hover {
    border-color: #7C6FFF !important;
    background: #2E3350 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #7C6FFF, #FF6FD8) !important;
    border: none !important;
    border-radius: 10px !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 12px 36px !important;
}

/* ── Progress / spinner ─────────────────────────────────── */
[data-testid="stProgress"] > div {
    background: linear-gradient(90deg, #7C6FFF, #FF6FD8) !important;
    border-radius: 6px !important;
}

/* ── Tabs ───────────────────────────────────────────────── */
[data-testid="stTabs"] button {
    color: #9095A8;
    font-weight: 600;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #7C6FFF;
    border-bottom: 2px solid #7C6FFF;
}

/* ── DataFrame ──────────────────────────────────────────── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── Divider ────────────────────────────────────────────── */
hr { border-color: #2A2D3E; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  COLOUR CLASSIFICATION ENGINE  (fully vectorised)
# ═══════════════════════════════════════════════════════════

def classify_pixels(rgb_array: np.ndarray) -> np.ndarray:
    """
    Classify every pixel using HSV-based rules.
    Returns flat array of string labels, length H*W.
    """
    h, w, _ = rgb_array.shape
    total   = h * w

    # RGB → HSV (OpenCV: H 0-180, S 0-255, V 0-255)
    hsv_img  = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
    hsv_flat = hsv_img.reshape(total, 3).astype(np.float32)

    H = hsv_flat[:, 0] * 2.0       # → 0-360°
    S = hsv_flat[:, 1] / 255.0     # → 0-1
    V = hsv_flat[:, 2] / 255.0     # → 0-1

    labels     = np.empty(total, dtype=object)
    labels[:]  = ""

    # ── Achromatic first ────────────────────────────────────
    mask_white = (V > 0.90) & (S < 0.15)
    labels[mask_white] = "White"

    mask_black = (~mask_white) & (V < 0.15)
    labels[mask_black] = "Black"

    mask_gray = (labels == "") & (S < 0.20)
    labels[mask_gray] = "Gray"

    # ── Brown (before Orange – same hue range, lower V) ─────
    rem = labels == ""
    mask_brown = rem & (H >= 10) & (H <= 40) & (V >= 0.20) & (V <= 0.65) & (S > 0.25)
    labels[mask_brown] = "Brown"

    # ── Chromatic hue buckets ────────────────────────────────
    rem = labels == ""
    labels[rem & ((H <= 15) | (H >= 345))]           = "Red"
    rem = labels == ""
    labels[rem & (H > 15) & (H <= 40)]               = "Orange"
    rem = labels == ""
    labels[rem & (H > 40) & (H <= 70)]               = "Yellow"
    rem = labels == ""
    labels[rem & (H > 70) & (H <= 170)]              = "Green"
    rem = labels == ""
    labels[rem & (H > 170) & (H <= 260)]             = "Blue"
    rem = labels == ""
    labels[rem & (H > 260) & (H <= 300)]             = "Purple"
    rem = labels == ""
    labels[rem & (H > 300) & (H <= 345)]             = "Pink"

    # ── Safety net for any floating-point edge cases ─────────
    still_empty = labels == ""
    if still_empty.any():
        H_rem = H[still_empty]
        fallback = np.where(H_rem <= 40, "Red",
                   np.where(H_rem <= 70, "Yellow",
                   np.where(H_rem <= 170, "Green",
                   np.where(H_rem <= 260, "Blue",
                   np.where(H_rem <= 300, "Purple", "Pink")))))
        labels[still_empty] = fallback

    return labels


def analyze_image_array(rgb_array: np.ndarray):
    """
    Run classification, return:
      - results DataFrame (Color, Pixels, Percentage)
      - heatmap rgb_array (same shape, pixels replaced by category colour)
    """
    h, w = rgb_array.shape[:2]
    total   = h * w

    labels = classify_pixels(rgb_array)

    # ── Counts & percentages ─────────────────────────────────
    counts   = {cat: int(np.sum(labels == cat)) for cat in CATEGORIES}
    percents = {cat: round(counts[cat] / total * 100, 2) for cat in CATEGORIES}

    df = pd.DataFrame({
        "Color":      list(percents.keys()),
        "Pixels":     list(counts.values()),
        "Percentage": list(percents.values()),
    }).sort_values("Percentage", ascending=False).reset_index(drop=True)

    # ── Heatmap: replace each pixel with its category colour ─
    rgb_map   = np.array([COLOR_RGB[lbl] for lbl in labels], dtype=np.uint8)
    heatmap   = rgb_map.reshape(h, w, 3)

    return df, heatmap


# ═══════════════════════════════════════════════════════════
#  CHART GENERATORS  (return PNG bytes)
# ═══════════════════════════════════════════════════════════

def _buf(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


def make_pie_chart(df: pd.DataFrame, title: str = "") -> bytes:
    visible = df[df["Percentage"] > 0]
    colors  = [COLOR_HEX[c] for c in visible["Color"]]
    labels  = [
        f"{r['Color']}\n{r['Percentage']:.1f}%" if r["Percentage"] >= 2 else ""
        for _, r in visible.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(7, 6), facecolor="#1A1D2E")
    ax.set_facecolor("#1A1D2E")
    wedges, texts = ax.pie(
        visible["Percentage"], labels=labels, colors=colors,
        startangle=140,
        wedgeprops=dict(linewidth=0.8, edgecolor="#1A1D2E"),
        textprops=dict(fontsize=8, color="white"),
    )
    handles = [
        mpatches.Patch(color=COLOR_HEX[r["Color"]],
                       label=f"{r['Color']}  {r['Percentage']:.1f}%")
        for _, r in visible.iterrows() if r["Percentage"] > 0
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=7,
              framealpha=0, labelcolor="white", title_fontsize=8,
              title="Colors")
    ax.set_title(title or "Color Distribution", color="white",
                 fontsize=11, fontweight="bold", pad=12)
    return _buf(fig)


def make_bar_chart(df: pd.DataFrame, title: str = "") -> bytes:
    visible = df[df["Percentage"] > 0]
    colors  = [COLOR_HEX[c] for c in visible["Color"][::-1]]

    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor="#1A1D2E")
    ax.set_facecolor("#1A1D2E")
    bars = ax.barh(
        visible["Color"][::-1], visible["Percentage"][::-1],
        color=colors, edgecolor="#1A1D2E", linewidth=0.5, height=0.65,
    )
    for bar, pct in zip(bars, visible["Percentage"][::-1]):
        ax.text(bar.get_width() + 0.3,
                bar.get_y() + bar.get_height() / 2,
                f"{pct:.2f}%", va="center", ha="left",
                fontsize=8.5, color="white")
    ax.set_xlabel("Percentage (%)", color="white", fontsize=9)
    ax.set_title(title or "Color Breakdown", color="white",
                 fontsize=11, fontweight="bold", pad=12)
    ax.set_xlim(0, max(visible["Percentage"]) * 1.18)
    ax.spines[:].set_color("#2A2D3E")
    ax.tick_params(colors="white", labelsize=9)
    ax.xaxis.label.set_color("white")
    return _buf(fig)


# ═══════════════════════════════════════════════════════════
#  HELPER UTILITIES
# ═══════════════════════════════════════════════════════════

def pil_to_b64(pil_img: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def bytes_to_b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def palette_html(df: pd.DataFrame) -> str:
    visible = df[df["Percentage"] >= 0.5].copy()
    segments = ""
    for _, row in visible.iterrows():
        name  = row["Color"]
        pct   = row["Percentage"]
        hex_c = COLOR_HEX[name]
        label = f"{name} {pct:.1f}%" if pct >= 5 else ""
        segments += (
            f'<div class="palette-segment" '
            f'style="flex:{pct};background:{hex_c};'
            f'min-width:{max(pct,0.4):.1f}%">'
            f'{label}</div>'
        )
    return f'<div class="palette-outer">{segments}</div>'


def color_table_html(df: pd.DataFrame) -> str:
    rows = ""
    max_pct = df["Percentage"].max()
    for i, row in df.iterrows():
        name    = row["Color"]
        pct     = row["Percentage"]
        pixels  = f"{int(row['Pixels']):,}"
        hex_c   = COLOR_HEX[name]
        bar_w   = (pct / max_pct * 100) if max_pct > 0 else 0
        bold    = "font-weight:700;color:#E8EAF0;" if i == 0 else ""
        rows += f"""
        <div class="color-row" style="{bold}">
          <div class="color-swatch" style="background:{hex_c}"></div>
          <div class="color-name">{name}</div>
          <div class="color-bar-bg">
            <div class="color-bar-fill" style="width:{bar_w:.1f}%;background:{hex_c}"></div>
          </div>
          <div class="color-pct">{pixels} px &nbsp; <b>{pct:.2f}%</b></div>
        </div>"""
    return f'<div>{rows}</div>'


def stat_box_html(value: str, label: str) -> str:
    return f"""
    <div class="stat-box">
      <div class="stat-value">{value}</div>
      <div class="stat-label">{label}</div>
    </div>"""


def make_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def make_json_bytes(df: pd.DataFrame) -> bytes:
    result = {row["Color"]: row["Percentage"] for _, row in df.iterrows()}
    return json.dumps(result, indent=2).encode()


def make_full_report_bytes(
    df: pd.DataFrame,
    img_name: str,
    width: int,
    height: int,
    file_size: str,
) -> bytes:
    """Simple HTML report bundling the key data."""
    rows = "".join(
        f"<tr><td>{r['Color']}</td><td>{int(r['Pixels']):,}</td>"
        f"<td>{r['Percentage']:.2f}%</td></tr>"
        for _, r in df.iterrows()
    )
    top3 = df.head(3)
    top3_str = ", ".join(
        f"{r['Color']} ({r['Percentage']:.2f}%)" for _, r in top3.iterrows()
    )
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
    <title>Color Analysis Report – {img_name}</title>
    <style>
      body{{font-family:sans-serif;max-width:800px;margin:40px auto;color:#222}}
      h1{{color:#5A4FCC}}table{{border-collapse:collapse;width:100%}}
      th,td{{border:1px solid #ddd;padding:8px 12px;text-align:left}}
      th{{background:#f4f4f4}}tr:nth-child(even){{background:#fafafa}}
    </style></head><body>
    <h1>🎨 Color Analysis Report</h1>
    <p><b>File:</b> {img_name} &nbsp;|&nbsp;
       <b>Dimensions:</b> {width}×{height} px &nbsp;|&nbsp;
       <b>Total Pixels:</b> {width*height:,} &nbsp;|&nbsp;
       <b>Size:</b> {file_size}</p>
    <p><b>Dominant Colour:</b> {df.iloc[0]['Color']} ({df.iloc[0]['Percentage']:.2f}%)</p>
    <p><b>Top 3:</b> {top3_str}</p>
    <h2>Breakdown</h2>
    <table><tr><th>Color</th><th>Pixels</th><th>Percentage</th></tr>
    {rows}</table></body></html>"""
    return html.encode()


# ═══════════════════════════════════════════════════════════
#  SINGLE-IMAGE ANALYSIS PAGE
# ═══════════════════════════════════════════════════════════

def render_single(uploaded_file):
    # ── Load & validate ──────────────────────────────────────
    file_bytes = uploaded_file.read()
    file_size  = f"{len(file_bytes)/1024:.1f} KB" if len(file_bytes) < 1_048_576 \
                 else f"{len(file_bytes)/1_048_576:.2f} MB"

    if len(file_bytes) > MAX_FILE_MB * 1_048_576:
        st.error(f"File exceeds {MAX_FILE_MB} MB limit. Please upload a smaller image.")
        return

    try:
        pil_img   = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        rgb_array = np.array(pil_img, dtype=np.uint8)
    except Exception as e:
        st.error(f"Could not open image: {e}")
        return

    W, H   = pil_img.width, pil_img.height
    total  = W * H

    # ── Analyse button ───────────────────────────────────────
    st.markdown('<div class="section-label">📂 Image Preview</div>',
                unsafe_allow_html=True)
    col_prev, col_meta = st.columns([1, 1], gap="large")

    with col_prev:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(pil_img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_meta:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Image Metadata</div>',
                    unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m1.markdown(stat_box_html(str(W), "Width (px)"),    unsafe_allow_html=True)
        m2.markdown(stat_box_html(str(H), "Height (px)"),   unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        m3, m4 = st.columns(2)
        m3.markdown(stat_box_html(f"{total:,}", "Total Pixels"), unsafe_allow_html=True)
        m4.markdown(stat_box_html(file_size,    "File Size"),     unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Analyse button ───────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col_btn = st.columns([1, 2, 1])[1]
    with col_btn:
        run_analysis = st.button("🔍  Analyze Colors", use_container_width=True)

    if not run_analysis:
        return

    # ── Processing ───────────────────────────────────────────
    with st.spinner("Analyzing image at pixel level…"):
        progress = st.progress(0)
        progress.progress(20, "Loading pixel data…")
        df, heatmap = analyze_image_array(rgb_array)
        progress.progress(65, "Generating charts…")
        pie_bytes = make_pie_chart(df, uploaded_file.name)
        bar_bytes = make_bar_chart(df, uploaded_file.name)
        progress.progress(100, "Done!")
        progress.empty()

    st.success("✅ Analysis complete!")
    st.markdown("<hr>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    #  SECTION 1 – DOMINANT COLOUR INSIGHTS
    # ════════════════════════════════════════════════════════
    st.markdown('<div class="section-label">🏆 Dominant Colour Insights</div>',
                unsafe_allow_html=True)

    dom   = df.iloc[0]
    top3  = df.head(3)
    dom_hex = COLOR_HEX[dom["Color"]]

    ins1, ins2 = st.columns([1, 2], gap="large")
    with ins1:
        st.markdown(
            f'<div class="dominant-card" style="background:{dom_hex};">'
            f'<div style="font-size:2rem">●</div>'
            f'<div style="font-size:1.5rem;margin:4px 0">{dom["Color"]}</div>'
            f'<div style="font-size:1.8rem">{dom["Percentage"]:.1f}%</div>'
            f'<div style="font-size:0.8rem;opacity:0.85">dominant colour</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with ins2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Top 3 Colours</div>',
                    unsafe_allow_html=True)
        for rank, (_, row) in enumerate(top3.iterrows(), 1):
            c_hex = COLOR_HEX[row["Color"]]
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:10px;'
                f'padding:8px 0;border-bottom:1px solid #2A2D3E">'
                f'<span style="font-size:1.1rem;color:#7C6FFF;font-weight:700">#{rank}</span>'
                f'<span style="width:14px;height:14px;border-radius:3px;'
                f'background:{c_hex};display:inline-block"></span>'
                f'<span style="flex:1">{row["Color"]}</span>'
                f'<span style="color:#9095A8">{int(row["Pixels"]):,} px</span>'
                f'<span style="font-weight:700;color:{c_hex}">{row["Percentage"]:.2f}%</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    #  SECTION 2 – COLOUR BREAKDOWN TABLE
    # ════════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">📋 Colour Breakdown Table</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(color_table_html(df), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    #  SECTION 3 – PALETTE STRIP
    # ════════════════════════════════════════════════════════
    st.markdown('<div class="section-label">🎨 Colour Palette Strip</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(palette_html(df), unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.75rem;color:#666;margin-top:8px">'
        'Strip width ∝ percentage of total pixels. Segments &lt;0.5% hidden.</p>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    #  SECTION 4 – CHARTS  (tabs)
    # ════════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">📊 Charts</div>',
                unsafe_allow_html=True)

    tab_pie, tab_bar = st.tabs(["🥧 Pie Chart", "📊 Bar Chart"])
    with tab_pie:
        st.image(pie_bytes, use_container_width=True)
    with tab_bar:
        st.image(bar_bytes, use_container_width=True)

    # ════════════════════════════════════════════════════════
    #  SECTION 5 – HEATMAP COMPARISON
    # ════════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">🔬 Pixel Classification Heatmap</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#9095A8;font-size:0.87rem;margin-bottom:10px">'
        'Every pixel is replaced by its classified category colour — '
        'a visual sanity-check of the algorithm.</p>',
        unsafe_allow_html=True,
    )
    hm_col1, hm_col2 = st.columns(2, gap="medium")
    with hm_col1:
        st.markdown('<div class="card-sm">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Original</div>',
                    unsafe_allow_html=True)
        st.image(pil_img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with hm_col2:
        st.markdown('<div class="card-sm">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Classified</div>',
                    unsafe_allow_html=True)
        st.image(Image.fromarray(heatmap), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    #  SECTION 6 – DOWNLOADS
    # ════════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">⬇️ Download Results</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    dl1, dl2, dl3, dl4, dl5 = st.columns(5, gap="small")

    stem = uploaded_file.name.rsplit(".", 1)[0]

    dl1.download_button("📄 CSV",          make_csv_bytes(df),   f"{stem}_results.csv",  "text/csv")
    dl2.download_button("{ } JSON",        make_json_bytes(df),  f"{stem}_results.json", "application/json")
    dl3.download_button("🥧 Pie Chart",    pie_bytes,            f"{stem}_pie.png",      "image/png")
    dl4.download_button("📊 Bar Chart",    bar_bytes,            f"{stem}_bar.png",      "image/png")
    dl5.download_button("📑 Full Report",
                        make_full_report_bytes(df, uploaded_file.name, W, H, file_size),
                        f"{stem}_report.html", "text/html")

    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  BATCH ANALYSIS PAGE
# ═══════════════════════════════════════════════════════════

def render_batch(uploaded_files):
    st.markdown(
        '<div class="section-label">📦 Batch Analysis</div>',
        unsafe_allow_html=True,
    )
    results = []

    for uf in uploaded_files:
        try:
            file_bytes = uf.read()
            if len(file_bytes) > MAX_FILE_MB * 1_048_576:
                st.warning(f"⚠️ {uf.name} skipped – exceeds {MAX_FILE_MB} MB.")
                continue
            pil_img   = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            rgb_array = np.array(pil_img, dtype=np.uint8)

            with st.spinner(f"Analysing {uf.name}…"):
                df, _   = analyze_image_array(rgb_array)

            dom = df.iloc[0]
            results.append({
                "Image":           uf.name,
                "Dominant Color":  dom["Color"],
                "Coverage":        f"{dom['Percentage']:.2f}%",
                "Top 3":           " | ".join(
                    f"{r['Color']} {r['Percentage']:.1f}%"
                    for _, r in df.head(3).iterrows()
                ),
                "W×H":             f"{pil_img.width}×{pil_img.height}",
                "Pixels":          f"{pil_img.width*pil_img.height:,}",
            })

            # Individual charts
            with st.expander(f"📷 {uf.name} — {dom['Color']} ({dom['Percentage']:.2f}%)"):
                ec1, ec2 = st.columns(2, gap="medium")
                with ec1:
                    st.image(pil_img, use_container_width=True)
                    st.markdown(palette_html(df), unsafe_allow_html=True)
                with ec2:
                    st.image(make_bar_chart(df, uf.name), use_container_width=True)

        except Exception as e:
            st.error(f"Error processing {uf.name}: {e}")

    # ── Comparison table ─────────────────────────────────────
    if results:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">📊 Comparison Dashboard</div>',
                    unsafe_allow_html=True)
        compare_df = pd.DataFrame(results)
        st.dataframe(compare_df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download Comparison CSV",
            compare_df.to_csv(index=False).encode(),
            "batch_comparison.csv",
            "text/csv",
        )


# ═══════════════════════════════════════════════════════════
#  MAIN  LAYOUT
# ═══════════════════════════════════════════════════════════

def main():
    # ── Hero ─────────────────────────────────────────────────
    st.markdown("""
        <div style="text-align:center;padding:30px 0 10px">
          <div class="hero-title">🎨 Image Color Distribution Analyzer</div>
          <div class="hero-sub">
            Upload an image and instantly see pixel-level color composition
            across <b>11 standardized color categories</b>.
          </div>
        </div>
    """, unsafe_allow_html=True)

    # ── Mode selector ────────────────────────────────────────
    mode = st.radio(
        "Mode",
        ["Single Image", "Batch Upload"],
        horizontal=True,
        label_visibility="collapsed",
    )

    # ── Upload ───────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)

    if mode == "Single Image":
        uploaded = st.file_uploader(
            "Drop your image here or click to browse",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=False,
            help=f"Supports JPG, JPEG, PNG, WebP up to {MAX_FILE_MB} MB",
        )
        if uploaded:
            render_single(uploaded)
        else:
            st.markdown("""
            <div style="text-align:center;padding:40px;color:#555">
              <div style="font-size:3rem">⬆️</div>
              <div style="margin-top:10px;font-size:0.95rem;color:#666">
                Upload an image above to get started
              </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        uploaded_files = st.file_uploader(
            "Drop your images here (up to 10)",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            help=f"Supports JPG, JPEG, PNG, WebP up to {MAX_FILE_MB} MB each",
        )
        if uploaded_files:
            if len(uploaded_files) > 10:
                st.warning("Please upload at most 10 images at a time.")
                uploaded_files = uploaded_files[:10]
            if st.button("🔍  Analyze All Images", use_container_width=False):
                render_batch(uploaded_files)
        else:
            st.markdown("""
            <div style="text-align:center;padding:40px;color:#555">
              <div style="font-size:3rem">📦</div>
              <div style="margin-top:10px;font-size:0.95rem;color:#666">
                Upload multiple images to compare them side-by-side
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Footer ───────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;margin-top:60px;padding:20px;
                color:#444;font-size:0.78rem;border-top:1px solid #2A2D3E">
      Built with Claude · Alemeno PM Internship Assignment · 
      HSV pixel classification · NumPy vectorised
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
