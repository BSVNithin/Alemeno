"""
=============================================================
  Pixel-Level Image Color Analyzer
  Alemeno Product Management Internship Assignment
=============================================================
  Classifies every pixel in an image into one of 11 color
  categories using HSV color space rules.

  Categories: White, Black, Gray, Red, Orange, Yellow,
              Green, Blue, Purple, Pink, Brown

  Usage:
      python color_analyzer.py --image path/to/image.jpg
      python color_analyzer.py --image path/to/image.jpg --output_dir outputs/

  Author  : Built with Claude (Anthropic)
  Version : 1.0
=============================================================
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import cv2


# ─────────────────────────────────────────────────────────────
#  COLOUR DEFINITIONS
#  Visual hex values used ONLY for charts – not for detection.
# ─────────────────────────────────────────────────────────────
COLOR_META = {
    "White":  "#F5F5F5",
    "Black":  "#1A1A1A",
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

# Canonical order for output tables / charts
CATEGORIES = list(COLOR_META.keys())


# ─────────────────────────────────────────────────────────────
#  CORE CLASSIFICATION  (fully vectorised with NumPy)
# ─────────────────────────────────────────────────────────────

def classify_pixels(rgb_array: np.ndarray) -> np.ndarray:
    """
    Classify every pixel in *rgb_array* (shape H×W×3, dtype uint8)
    into one of 11 colour categories.

    Returns
    -------
    labels : np.ndarray  shape (H*W,)  dtype object  – category name per pixel
    """
    h, w, _ = rgb_array.shape
    total   = h * w

    # ── Step 1 : Convert RGB → HSV via OpenCV ───────────────
    # OpenCV HSV ranges: H 0-180, S 0-255, V 0-255
    hsv_img  = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
    hsv_flat = hsv_img.reshape(total, 3).astype(np.float32)

    # Normalise to [0,1] / [0,360] ranges expected by the rules
    H = hsv_flat[:, 0] * 2.0          # 0-360 degrees
    S = hsv_flat[:, 1] / 255.0        # 0-1
    V = hsv_flat[:, 2] / 255.0        # 0-1

    # ── Step 2 : Pre-allocate label array ───────────────────
    labels = np.empty(total, dtype=object)
    labels[:] = ""                     # sentinel – must be filled for every pixel

    # ── Step 3 : Achromatic classes first (order matters!) ──

    # WHITE  : high brightness, very low saturation
    mask_white = (V > 0.90) & (S < 0.15)
    labels[mask_white] = "White"

    # BLACK  : very low brightness
    mask_black = (V < 0.15) & ~mask_white
    labels[mask_black] = "Black"

    # GRAY   : low saturation, mid brightness (not already W/B)
    mask_gray = (S < 0.20) & (labels == "")
    labels[mask_gray] = "Gray"

    # ── Step 4 : Chromatic pixels – use Hue angle ───────────
    remaining = labels == ""           # pixels not yet classified

    # BROWN  : detected BEFORE Orange because it overlaps Hue 10-40
    #          Brown = warm low-brightness orange
    mask_brown = (
        remaining &
        (H >= 10) & (H <= 40) &
        (V >= 0.20) & (V <= 0.65) &
        (S > 0.25)
    )
    labels[mask_brown] = "Brown"
    remaining = labels == ""

    # RED    : hue wraps around 0°
    mask_red = remaining & ((H <= 15) | (H >= 345))
    labels[mask_red] = "Red"
    remaining = labels == ""

    # ORANGE : hue 15-40  (higher brightness than Brown)
    mask_orange = remaining & (H > 15) & (H <= 40)
    labels[mask_orange] = "Orange"
    remaining = labels == ""

    # YELLOW : hue 40-70
    mask_yellow = remaining & (H > 40) & (H <= 70)
    labels[mask_yellow] = "Yellow"
    remaining = labels == ""

    # GREEN  : hue 70-170
    mask_green = remaining & (H > 70) & (H <= 170)
    labels[mask_green] = "Green"
    remaining = labels == ""

    # BLUE   : hue 170-260
    mask_blue = remaining & (H > 170) & (H <= 260)
    labels[mask_blue] = "Blue"
    remaining = labels == ""

    # PURPLE : hue 260-300
    mask_purple = remaining & (H > 260) & (H <= 300)
    labels[mask_purple] = "Purple"
    remaining = labels == ""

    # PINK   : hue 300-345  (remaining after Brown claimed 10-40)
    mask_pink = remaining & (H > 300) & (H <= 345)
    labels[mask_pink] = "Pink"
    remaining = labels == ""

    # ── Step 5 : Safety net – classify anything uncaught ────
    # (should be empty with complete rules, but guards against
    #  floating-point edge cases at hue boundaries)
    if remaining.any():
        # Fallback: nearest hue bucket
        leftover_H = H[remaining]
        fallback = np.where(leftover_H <= 40, "Red",
                   np.where(leftover_H <= 70, "Yellow",
                   np.where(leftover_H <= 170, "Green",
                   np.where(leftover_H <= 260, "Blue",
                   np.where(leftover_H <= 300, "Purple", "Pink")))))
        labels[remaining] = fallback

    return labels


# ─────────────────────────────────────────────────────────────
#  MAIN ANALYSIS FUNCTION
# ─────────────────────────────────────────────────────────────

def analyze_image(image_path: str, output_dir: str = "outputs") -> dict:
    """
    Full pipeline: load → classify → tabulate → charts → JSON/CSV.

    Parameters
    ----------
    image_path : str   – path to input image (JPG / PNG / JPEG)
    output_dir : str   – folder for saved outputs

    Returns
    -------
    dict  { "White": xx.xx, "Black": xx.xx, … }  (percentages)
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # ── Load image ──────────────────────────────────────────
    pil_img   = Image.open(image_path).convert("RGB")
    rgb_array = np.array(pil_img, dtype=np.uint8)
    total_px  = rgb_array.shape[0] * rgb_array.shape[1]

    print(f"\n{'='*55}")
    print(f"  Analysing : {image_path}")
    print(f"  Size      : {pil_img.width} × {pil_img.height} px  ({total_px:,} pixels)")
    print(f"{'='*55}")

    # ── Classify every pixel ────────────────────────────────
    labels = classify_pixels(rgb_array)

    # ── Count & compute percentages ─────────────────────────
    counts  = {cat: int(np.sum(labels == cat)) for cat in CATEGORIES}
    percents = {cat: round(counts[cat] / total_px * 100, 2) for cat in CATEGORIES}

    # Sanity check
    total_classified = sum(counts.values())
    assert total_classified == total_px, (
        f"Classification mismatch: {total_classified} ≠ {total_px} pixels"
    )

    # ── Build results DataFrame (sorted descending) ─────────
    df = pd.DataFrame({
        "Color":      list(percents.keys()),
        "Pixels":     list(counts.values()),
        "Percentage": list(percents.values()),
    }).sort_values("Percentage", ascending=False).reset_index(drop=True)

    # ── Print summary table ──────────────────────────────────
    print(f"\n{'Color':<12} {'Pixels':>12} {'Percentage':>12}")
    print("-" * 38)
    for _, row in df.iterrows():
        bar = "█" * int(row["Percentage"] / 2)
        print(f"  {row['Color']:<10} {int(row['Pixels']):>12,} {row['Percentage']:>10.2f}%  {bar}")
    print("-" * 38)
    print(f"  {'TOTAL':<10} {total_px:>12,} {'100.00%':>11}")

    dominant = df.iloc[0]
    top3     = df.head(3)
    print(f"\n  Dominant Color : {dominant['Color']}  ({dominant['Percentage']:.2f}%)")
    print(f"  Top 3 Colors   : " +
          "  |  ".join(f"{r['Color']} {r['Percentage']:.2f}%" for _, r in top3.iterrows()))

    # ── Save CSV ─────────────────────────────────────────────
    csv_path = os.path.join(output_dir, f"{base_name}_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  CSV saved  → {csv_path}")

    # ── Save JSON ────────────────────────────────────────────
    json_path = os.path.join(output_dir, f"{base_name}_results.json")
    with open(json_path, "w") as f:
        json.dump(percents, f, indent=2)
    print(f"  JSON saved → {json_path}")

    # ── Generate charts ──────────────────────────────────────
    _make_pie_chart(df, base_name, output_dir, pil_img)
    _make_bar_chart(df, base_name, output_dir)
    _make_combined_chart(df, base_name, output_dir, pil_img)

    return percents


# ─────────────────────────────────────────────────────────────
#  CHART HELPERS
# ─────────────────────────────────────────────────────────────

def _get_colors(df: pd.DataFrame) -> list:
    """Return hex colour list in the order rows appear in df."""
    return [COLOR_META[c] for c in df["Color"]]


def _make_pie_chart(df, base_name, output_dir, pil_img):
    """Pie chart – only slices ≥ 0.5% are labelled to avoid clutter."""
    fig, ax = plt.subplots(figsize=(8, 7), facecolor="#FAFAFA")

    sizes  = df["Percentage"].values
    colors = _get_colors(df)
    labels = [
        f"{row['Color']}\n{row['Percentage']:.1f}%" if row["Percentage"] >= 1.5 else ""
        for _, row in df.iterrows()
    ]

    wedges, texts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        startangle=140,
        wedgeprops=dict(linewidth=0.8, edgecolor="white"),
        textprops=dict(fontsize=9),
    )

    # Legend for slices that were too small to label
    legend_handles = [
        mpatches.Patch(color=COLOR_META[row["Color"]],
                       label=f"{row['Color']}  {row['Percentage']:.1f}%")
        for _, row in df.iterrows() if row["Percentage"] > 0
    ]
    ax.legend(handles=legend_handles, loc="lower right",
              fontsize=8, framealpha=0.9, title="Colors")

    ax.set_title(f"Color Distribution — {base_name}",
                 fontsize=13, fontweight="bold", pad=14)
    plt.tight_layout()

    path = os.path.join(output_dir, f"{base_name}_pie.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Pie chart  → {path}")


def _make_bar_chart(df, base_name, output_dir):
    """Horizontal bar chart sorted descending."""
    visible = df[df["Percentage"] > 0].copy()

    fig, ax = plt.subplots(figsize=(9, 6), facecolor="#FAFAFA")

    bars = ax.barh(
        visible["Color"][::-1],          # reverse so highest is on top
        visible["Percentage"][::-1],
        color=[COLOR_META[c] for c in visible["Color"][::-1]],
        edgecolor="white",
        linewidth=0.6,
        height=0.65,
    )

    # Value labels on bars
    for bar, pct in zip(bars, visible["Percentage"][::-1]):
        ax.text(
            bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{pct:.2f}%", va="center", ha="left", fontsize=9,
        )

    ax.set_xlabel("Percentage of Total Pixels (%)", fontsize=10)
    ax.set_title(f"Color Breakdown — {base_name}",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(0, max(visible["Percentage"]) * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=10)
    plt.tight_layout()

    path = os.path.join(output_dir, f"{base_name}_bar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Bar chart  → {path}")


def _make_combined_chart(df, base_name, output_dir, pil_img):
    """
    Single combined figure:
      [original image]  |  [pie chart]
      [horizontal bar chart spanning full width]
    """
    fig = plt.figure(figsize=(14, 10), facecolor="#F8F8F8")
    fig.suptitle(f"Pixel-Level Color Analysis — {base_name}",
                 fontsize=15, fontweight="bold", y=0.97)

    gs  = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.4], hspace=0.35, wspace=0.25)
    ax_img = fig.add_subplot(gs[0, 0])
    ax_pie = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[1, :])

    # ── Original image ───────────────────────────────────────
    ax_img.imshow(pil_img)
    ax_img.set_title("Original Image", fontsize=11)
    ax_img.axis("off")

    # ── Pie ──────────────────────────────────────────────────
    sizes  = df["Percentage"].values
    colors = _get_colors(df)
    labels_pie = [
        f"{r['Color']}\n{r['Percentage']:.1f}%" if r["Percentage"] >= 2.0 else ""
        for _, r in df.iterrows()
    ]
    ax_pie.pie(
        sizes, labels=labels_pie, colors=colors, startangle=140,
        wedgeprops=dict(linewidth=0.8, edgecolor="white"),
        textprops=dict(fontsize=8),
    )
    ax_pie.set_title("Pie Chart", fontsize=11)

    # ── Bar ──────────────────────────────────────────────────
    visible = df[df["Percentage"] > 0]
    bars = ax_bar.barh(
        visible["Color"][::-1],
        visible["Percentage"][::-1],
        color=[COLOR_META[c] for c in visible["Color"][::-1]],
        edgecolor="white", linewidth=0.6, height=0.65,
    )
    for bar, pct in zip(bars, visible["Percentage"][::-1]):
        ax_bar.text(
            bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
            f"{pct:.2f}%", va="center", ha="left", fontsize=9,
        )
    ax_bar.set_xlabel("Percentage (%)", fontsize=10)
    ax_bar.set_title("Bar Chart", fontsize=11)
    ax_bar.set_xlim(0, max(visible["Percentage"]) * 1.15)
    ax_bar.spines[["top", "right"]].set_visible(False)

    # ── Footer stats ─────────────────────────────────────────
    top3_str = "  |  ".join(
        f"{r['Color']} {r['Percentage']:.2f}%"
        for _, r in df.head(3).iterrows()
    )
    fig.text(0.5, 0.01, f"Dominant: {df.iloc[0]['Color']} ({df.iloc[0]['Percentage']:.2f}%)     "
             f"Top 3: {top3_str}",
             ha="center", fontsize=9, color="#555555")

    path = os.path.join(output_dir, f"{base_name}_combined.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Combined   → {path}")


# ─────────────────────────────────────────────────────────────
#  CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pixel-level image color analyzer (11 categories)"
    )
    parser.add_argument(
        "--image", required=True,
        help="Path to input image (jpg / png / jpeg)"
    )
    parser.add_argument(
        "--output_dir", default="outputs",
        help="Directory to save output files (default: outputs/)"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"ERROR: File not found → {args.image}")
        sys.exit(1)

    result = analyze_image(args.image, args.output_dir)

    print(f"\n  JSON result:")
    print(json.dumps(result, indent=4))
    print(f"\n{'='*55}\n  Done.\n{'='*55}\n")


if __name__ == "__main__":
    main()
