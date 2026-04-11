"""
Ch02 — Slide image generator
Saves PNG diagrams to:
  2-selection_cleaning_preparing/01-slides/

Images are referenced in ch02_slides.md as ./X.png (relative path).
  • Slidev: slides.sh symlinks current.md → ch02_slides.md; Vite follows
    the symlink so ./X.png resolves against 01-slides/ correctly.
  • VSCode markdown preview: ./X.png is relative to the .md file → works.

Run from anywhere inside the project:
  python 2-selection_cleaning_preparing/generate_images.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as FancyArrow
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# ── Output directory ───────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
OUT_DIR = SCRIPT_DIR / "01-slides"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def save(fig, name: str):
    """Save figure and close it."""
    fig.savefig(OUT_DIR / name)
    plt.close(fig)
    print(f"  ✓  {name}")

# ── Brand colours (GVZ / APML UZH theme) ─────────────────────────────────
TEAL      = "#00CCCC"
TEAL_DARK = "#009090"
DARK      = "#1a1a1a"
MUTED     = "#666666"
BORDER    = "#e0e0e0"
BG        = "#ffffff"

def apml_style():
    """Apply a clean, on-brand matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    BG,
        "axes.edgecolor":    BORDER,
        "axes.labelcolor":   DARK,
        "axes.titlecolor":   DARK,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "xtick.color":       MUTED,
        "ytick.color":       MUTED,
        "grid.color":        BORDER,
        "grid.linewidth":    0.8,
        "font.family":       "sans-serif",
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.titleweight":  "bold",
        "savefig.dpi":       150,
        "savefig.bbox":      "tight",
        "savefig.facecolor": BG,
    })

apml_style()

# ── Shared dataset ─────────────────────────────────────────────────────────
np.random.seed(42)
raw_data = {
    "student_id": range(1, 21),
    "age": [22, 25, 23, None, 21, 999, 24, 22, 26, 23,
            20, 25, None, 22, 24, 21, 23, 25, 22, 24],
    "gender": ["Male","female","MALE","Female",None,"male","Female","Male",
               "M","Female","male","Female","Male",None,"female","Male",
               "Female","male","FEMALE","Male"],
    "study_hours": [15, 20, None, 18, 25, 12, 30, 22, None, 16,
                    19, 28, 14, 21, 17, None, 23, 26, 18, 150],
    "grade": ["A","B","A","C","B",None,"A","B","C","A",
               "B","A","C","B",None,"A","B","C","A","B"],
    "faculty": ["Science","Arts","Science","Engineering","Arts",
                "Science",None,"Arts","Engineering","Science",
                "Arts","Science","Engineering","Arts","Science",
                "Engineering","Arts","Science","Arts","Engineering"],
    "score": [82,75,91,68,78,55,95,80,72,88,76,92,65,79,83,70,84,77,90,73],
}
df_raw = pd.DataFrame(raw_data)


# ══════════════════════════════════════════════════════════════════════════
# 1. pipeline_overview.png
# ══════════════════════════════════════════════════════════════════════════
def make_pipeline_overview():
    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.axis("off")

    steps = ["Raw\nData", "Inspect", "Clean", "Encode", "Scale", "Split", "Model"]
    colors = [MUTED, TEAL, TEAL, TEAL, TEAL, TEAL, TEAL_DARK]
    n = len(steps)
    xs = np.linspace(0.5, 9.5, n)

    for i, (x, label, color) in enumerate(zip(xs, steps, colors)):
        box = mpatches.FancyBboxPatch(
            (x - 0.55, 0.22), 1.1, 0.56,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor="white", linewidth=1.5,
            zorder=2,
        )
        ax.add_patch(box)
        ax.text(x, 0.50, label, ha="center", va="center",
                color="white", fontsize=9.5, fontweight="bold", zorder=3)
        if i < n - 1:
            ax.annotate("", xy=(xs[i+1] - 0.58, 0.50),
                        xytext=(x + 0.57, 0.50),
                        arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.5),
                        zorder=1)

    ax.text(5.0, 0.92, "The ML Pipeline", ha="center", va="center",
            fontsize=13, fontweight="bold", color=DARK)
    save(fig, "pipeline_overview.png")


# ══════════════════════════════════════════════════════════════════════════
# 2. missing_values_heatmap.png
# ══════════════════════════════════════════════════════════════════════════
def make_missing_values_heatmap():
    df = df_raw[["age", "gender", "study_hours", "grade", "faculty"]].copy()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    sns.heatmap(
        df.isnull(), cbar=False, yticklabels=False,
        cmap=["#f0f0f0", TEAL], linewidths=0.5, linecolor=BG, ax=ax,
    )
    ax.set_title("Missing Values (teal = missing)", pad=10)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0, labelsize=10)
    save(fig, "missing_values_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════
# 3. outlier_boxplot.png
# ══════════════════════════════════════════════════════════════════════════
def make_outlier_boxplot():
    df = df_raw.copy()
    df["age"] = df["age"].fillna(df["age"].median())
    df["study_hours"] = df["study_hours"].fillna(df["study_hours"].median())

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.4))
    cols = ["age", "study_hours", "score"]
    labels = ["Age", "Study hours / week", "Score"]

    for ax, col, lbl in zip(axes, cols, labels):
        bp = ax.boxplot(
            df[col].dropna(), patch_artist=True, widths=0.45,
            boxprops=dict(facecolor=TEAL, alpha=0.55, linewidth=1.2),
            medianprops=dict(color=TEAL_DARK, linewidth=2),
            whiskerprops=dict(color=MUTED, linewidth=1.2),
            capprops=dict(color=MUTED, linewidth=1.2),
            flierprops=dict(marker="o", markerfacecolor="#e74c3c",
                            markeredgecolor="white", markersize=8),
        )
        ax.set_title(lbl, fontsize=11)
        ax.set_xticks([])
        ax.yaxis.grid(True)

    axes[1].annotate("outlier!", xy=(1.18, 150), xytext=(1.38, 120),
                     fontsize=9, color="#e74c3c", fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.2))

    fig.suptitle("Boxplot — Outlier Detection", fontsize=13,
                 fontweight="bold", color=DARK, y=1.02)
    save(fig, "outlier_boxplot.png")


# ══════════════════════════════════════════════════════════════════════════
# 4. onehot_encoding.png
# ══════════════════════════════════════════════════════════════════════════
def make_onehot_encoding():
    before = pd.DataFrame({"faculty": ["Science", "Arts", "Engineering",
                                       "Arts", "Science"]})
    after  = pd.get_dummies(before["faculty"], prefix="faculty", dtype=int)
    combined = pd.concat([before, after], axis=1)

    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.axis("off")

    cols = list(combined.columns)
    col_w = 1.0 / len(cols)
    header_bg = TEAL
    even_bg   = "#f6fdfd"

    for ci, col in enumerate(cols):
        x = ci * col_w
        is_new = col != "faculty"
        # header
        ax.add_patch(plt.Rectangle((x, 0.78), col_w, 0.22,
                                   facecolor=header_bg, edgecolor=BG, lw=1))
        ax.text(x + col_w/2, 0.89, col, ha="center", va="center",
                color="white", fontsize=8.5, fontweight="bold")
        # rows
        for ri, val in enumerate(combined[col]):
            y = 0.78 - (ri + 1) * 0.15
            fill = even_bg if ri % 2 == 0 else BG
            if is_new and val == 1:
                fill = "#e8f9f9"
            ax.add_patch(plt.Rectangle((x, y), col_w, 0.15,
                                       facecolor=fill, edgecolor=BORDER, lw=0.5))
            txt_color = TEAL_DARK if (is_new and val == 1) else DARK
            ax.text(x + col_w/2, y + 0.075, str(val), ha="center",
                    va="center", fontsize=9, color=txt_color,
                    fontweight="bold" if (is_new and val == 1) else "normal")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_title("One-Hot Encoding — faculty column",
                 fontsize=12, fontweight="bold", color=DARK, pad=6)
    save(fig, "onehot_encoding.png")


# ══════════════════════════════════════════════════════════════════════════
# 5. feature_scaling.png
# ══════════════════════════════════════════════════════════════════════════
def make_feature_scaling():
    df = df_raw.copy()
    for col in ["age", "study_hours"]:
        df[col] = df[col].fillna(df[col].median())
    df.loc[df["age"] > 100, "age"] = df["age"].median()
    df["study_hours"] = df["study_hours"].clip(upper=40)

    feats = ["age", "study_hours", "score"]
    labels = ["Age", "Study hrs", "Score"]

    ss = StandardScaler()
    df_std = pd.DataFrame(ss.fit_transform(df[feats]), columns=feats)

    mm = MinMaxScaler()
    df_mm = pd.DataFrame(mm.fit_transform(df[feats]), columns=feats)

    fig, axes = plt.subplots(3, 3, figsize=(9, 5.5))
    row_titles = ["Original", "StandardScaler\n(mean=0, std=1)", "MinMaxScaler\n([0, 1])"]
    dfs = [df, df_std, df_mm]
    colors = [MUTED, TEAL, TEAL_DARK]

    for ri, (d, rtitle, color) in enumerate(zip(dfs, row_titles, colors)):
        for ci, (col, lbl) in enumerate(zip(feats, labels)):
            ax = axes[ri, ci]
            ax.hist(d[col], bins=8, color=color, alpha=0.72,
                    edgecolor="white", linewidth=0.7)
            if ri == 0:
                ax.set_title(lbl, fontsize=10, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(rtitle, fontsize=8.5, color=DARK)
            ax.yaxis.grid(True, alpha=0.6)
            ax.tick_params(labelsize=8)

    fig.suptitle("Feature Scaling — Before vs After",
                 fontsize=13, fontweight="bold", color=DARK, y=1.01)
    fig.tight_layout()
    save(fig, "feature_scaling.png")


# ══════════════════════════════════════════════════════════════════════════
# 6. train_test_split.png
# ══════════════════════════════════════════════════════════════════════════
def make_train_test_split():
    fig, ax = plt.subplots(figsize=(8, 2.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Full dataset bar
    ax.add_patch(plt.Rectangle((0, 0.55), 10, 0.38,
                               facecolor=BORDER, edgecolor=BG, lw=0))
    ax.text(5, 0.74, "Full dataset  (n = 20)", ha="center", va="center",
            fontsize=10, color=DARK, fontweight="bold")

    # Split bar
    split = 8.0  # 80%
    ax.add_patch(plt.Rectangle((0, 0.07), split, 0.38,
                               facecolor=TEAL, edgecolor=BG, lw=0))
    ax.add_patch(plt.Rectangle((split, 0.07), 10 - split, 0.38,
                               facecolor="#e74c3c", alpha=0.75,
                               edgecolor=BG, lw=0))
    ax.text(split/2, 0.26, "Train  80%", ha="center", va="center",
            fontsize=10, color="white", fontweight="bold")
    ax.text(split + (10 - split)/2, 0.26, "Test  20%",
            ha="center", va="center", fontsize=10, color="white",
            fontweight="bold")

    # Arrow
    ax.annotate("", xy=(5, 0.07), xytext=(5, 0.54),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.5))

    ax.set_title("Train / Test Split  (random_state=42, stratify=y)",
                 fontsize=12, fontweight="bold", color=DARK, pad=4)
    save(fig, "train_test_split.png")


# ══════════════════════════════════════════════════════════════════════════
# 7. data_leakage.png
# ══════════════════════════════════════════════════════════════════════════
def make_data_leakage():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.0))

    def draw_panel(ax, title, split_first, ok):
        ax.set_xlim(0, 10); ax.set_ylim(0, 1); ax.axis("off")
        color = TEAL if ok else "#e74c3c"
        icon  = "✓" if ok else "✗"

        if split_first:
            # correct: split → impute each part separately
            boxes = [
                (0.2, "All data",  MUTED),
                (2.5, "Split",     color),
                (5.0, "Impute\n(train)", color),
                (7.5, "Impute\n(test)",  color),
            ]
        else:
            # wrong: impute whole dataset first
            boxes = [
                (0.2, "All data",  MUTED),
                (2.5, "Impute\n(all!)",  "#e74c3c"),
                (5.0, "Split",     MUTED),
                (7.5, "Train/Test", MUTED),
            ]

        for i, (x, lbl, c) in enumerate(boxes):
            ax.add_patch(mpatches.FancyBboxPatch(
                (x - 0.85, 0.28), 1.7, 0.44,
                boxstyle="round,pad=0.05",
                facecolor=c, edgecolor="white", lw=1.5, zorder=2))
            ax.text(x, 0.50, lbl, ha="center", va="center",
                    color="white", fontsize=8.5, fontweight="bold", zorder=3)
            if i < len(boxes) - 1:
                ax.annotate("", xy=(boxes[i+1][0] - 0.87, 0.50),
                            xytext=(x + 0.87, 0.50),
                            arrowprops=dict(arrowstyle="->",
                                            color=MUTED, lw=1.4))

        label = f"{icon} {title}"
        ax.text(5, 0.90, label, ha="center", va="center",
                fontsize=11, fontweight="bold", color=color)
        if not ok:
            ax.text(3.5, 0.10,
                    "Test data leaked into imputer fit!",
                    ha="center", va="center", fontsize=8.5, color="#e74c3c",
                    style="italic")

    draw_panel(axes[0], "Wrong — leakage", split_first=False, ok=False)
    draw_panel(axes[1], "Correct — Pipeline", split_first=True, ok=True)

    fig.suptitle("Data Leakage", fontsize=13,
                 fontweight="bold", color=DARK, y=1.02)
    fig.tight_layout()
    save(fig, "data_leakage.png")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Generating ch02 slide images → {OUT_DIR}\n")
    make_pipeline_overview()
    make_missing_values_heatmap()
    make_outlier_boxplot()
    make_onehot_encoding()
    make_feature_scaling()
    make_train_test_split()
    make_data_leakage()
    print(f"\nDone — {len(list(OUT_DIR.glob('*.png')))} images in {OUT_DIR}")
