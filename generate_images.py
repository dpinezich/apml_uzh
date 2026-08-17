"""
APML UZH — Master image generator
===================================
Generates all slide images for all 12 chapters (legacy generators here +
per-chapter modules in imagegen/chNN.py incl. GIF animations).

Usage (from repo root):
    python generate_images.py

Each image is written to:
  1. The chapter's 01-slides/ folder  (Vite resolves from here at runtime)
  2. slidev/public/                   (static fallback + PDF export)

Images referenced in .md files as  ![alt](./filename.png).
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import make_blobs

# ── Root paths ──────────────────────────────────────────────────────────────
ROOT   = Path(__file__).parent
PUBLIC = ROOT / "slidev" / "public"
PUBLIC.mkdir(parents=True, exist_ok=True)

CHAPTERS = {
    "ch01": ROOT / "1-introduction" / "01-slides",
    "ch02": ROOT / "2-selection_cleaning_preparing" / "01-slides",
    "ch03": ROOT / "3-supervised_learning" / "01-slides",
    "ch04": ROOT / "3-supervised_learning" / "01-slides",
    "ch05": ROOT / "3-supervised_learning" / "01-slides",
    "ch06": ROOT / "3-supervised_learning" / "01-slides",
    "ch07": ROOT / "4-unsupervised_learning" / "01-slides",
    "ch08": ROOT / "4-unsupervised_learning" / "01-slides",
    "ch09": ROOT / "4-unsupervised_learning" / "01-slides",
    "ch10": ROOT / "5-reinforcement_learning" / "01-slides",
    "ch11": ROOT / "5-reinforcement_learning" / "01-slides",
    "ch12": ROOT / "6-capstone_ml" / "01-slides",
}

for d in set(CHAPTERS.values()):
    d.mkdir(parents=True, exist_ok=True)

# ── Brand colours ────────────────────────────────────────────────────────────
TEAL      = "#00CCCC"
TEAL_DARK = "#009090"
DARK      = "#1a1a1a"
MUTED     = "#666666"
BORDER    = "#e0e0e0"
BG        = "#ffffff"
RED       = "#e74c3c"
GREEN     = "#2ecc71"
BLUE      = "#3498db"
ORANGE    = "#e67e22"

def apml_style():
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
np.random.seed(42)


def save(fig, name: str, chapters: list):
    """Save fig to every listed chapter's 01-slides/ AND to slidev/public/."""
    paths_written = set()
    for ch in chapters:
        dest = CHAPTERS[ch] / name
        dest_dir = dest.parent
        if str(dest_dir) not in paths_written:
            fig.savefig(dest)
            paths_written.add(str(dest_dir))
    fig.savefig(PUBLIC / name)
    plt.close(fig)
    print(f"  ✓  {name}")


# ══════════════════════════════════════════════════════════════════════════════
#  Ch01 — Introduction
# ══════════════════════════════════════════════════════════════════════════════

def make_workflow_cycle():
    """7-step DS workflow as a circular ring diagram."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal")
    ax.axis("off")

    steps = [
        "① Define\nProblem",
        "② Collect\nData",
        "③ EDA &\nClean",
        "④ Preprocess\n& Engineer",
        "⑤ Train\nModel",
        "⑥ Evaluate",
        "⑦ Deploy\n& Monitor",
    ]
    n = len(steps)
    angles = [np.pi / 2 - 2 * np.pi * i / n for i in range(n)]
    radius = 1.1
    box_r  = 0.33

    colors = [TEAL_DARK, TEAL, TEAL, TEAL, TEAL, TEAL, TEAL_DARK]

    for i, (angle, step, color) in enumerate(zip(angles, steps, colors)):
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        circle = mpatches.FancyBboxPatch(
            (x - box_r, y - box_r * 0.85), 2 * box_r, 2 * box_r * 0.85,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor="white", linewidth=2, zorder=3,
        )
        ax.add_patch(circle)
        ax.text(x, y, step, ha="center", va="center",
                color="white", fontsize=9, fontweight="bold", zorder=4)

        # Arrow to next step
        next_angle = angles[(i + 1) % n]
        nx = radius * np.cos(next_angle)
        ny = radius * np.sin(next_angle)
        # Midpoint arrow (shortened)
        dx, dy = nx - x, ny - y
        ax.annotate("",
                    xy=(x + 0.72 * dx, y + 0.72 * dy),
                    xytext=(x + 0.28 * dx, y + 0.28 * dy),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED,
                                    lw=1.8, mutation_scale=14),
                    zorder=2)

    ax.text(0, 0, "Data\nScience\nWorkflow", ha="center", va="center",
            fontsize=12, fontweight="bold", color=DARK)

    ax.set_title("This is a cycle — not a pipeline", fontsize=13,
                 fontweight="bold", color=DARK, pad=10)
    save(fig, "workflow_cycle.png", ["ch01", "ch12"])


def make_ml_paradigms():
    """Three-column comparison: Supervised / Unsupervised / Reinforcement."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))

    titles = ["Supervised Learning", "Unsupervised Learning", "Reinforcement Learning"]
    subtitles = ["(X, y) → predict y", "X only → find structure", "trial & error → max reward"]
    colors = [TEAL, BLUE, ORANGE]

    # --- Panel 0: Supervised ---
    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    pts = [(2,8,"●","#e74c3c","Spam"), (3,6,"●","#e74c3c","Spam"),
           (7,7,"○",TEAL,"Ham"), (6,5,"○",TEAL,"Ham"), (5,3,"○",TEAL,"Ham")]
    for x, y, m, c, lbl in pts:
        ax.text(x, y, m, ha="center", va="center", fontsize=22, color=c)
        ax.text(x+0.7, y, lbl, ha="left", va="center", fontsize=9, color=DARK)
    ax.text(5, 1.2, "Labels provided → Model learns boundary",
            ha="center", fontsize=9, color=MUTED, style="italic")

    # --- Panel 1: Unsupervised ---
    ax = axes[1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    for x, y in [(2,7),(2.5,8),(3,7.5),(3.5,8.5),(2,8.5)]:
        ax.text(x, y, "●", ha="center", va="center", fontsize=18, color=TEAL)
    for x, y in [(7,3),(7.5,4),(8,3.5),(6.5,4),(7,4.5)]:
        ax.text(x, y, "●", ha="center", va="center", fontsize=18, color=BLUE)
    for x, y in [(5,6),(5.5,5.5),(4.5,5)]:
        ax.text(x, y, "●", ha="center", va="center", fontsize=18, color=ORANGE)
    ax.text(5, 1.2, "No labels → Algorithm discovers groups",
            ha="center", fontsize=9, color=MUTED, style="italic")

    # --- Panel 2: Reinforcement ---
    ax = axes[2]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    # Agent box
    ax.add_patch(mpatches.FancyBboxPatch((1, 4), 3, 2.5,
        boxstyle="round,pad=0.1", facecolor=ORANGE, edgecolor="white", lw=2))
    ax.text(2.5, 5.25, "Agent", ha="center", va="center",
            color="white", fontsize=12, fontweight="bold")
    # Env box
    ax.add_patch(mpatches.FancyBboxPatch((6, 4), 3, 2.5,
        boxstyle="round,pad=0.1", facecolor=TEAL, edgecolor="white", lw=2))
    ax.text(7.5, 5.25, "Environ-\nment", ha="center", va="center",
            color="white", fontsize=11, fontweight="bold")
    # Arrows
    ax.annotate("", xy=(6.0, 6.0), xytext=(4.0, 6.0),
                arrowprops=dict(arrowstyle="-|>", color=DARK, lw=2, mutation_scale=15))
    ax.text(5.0, 6.4, "Action", ha="center", fontsize=9, color=DARK)
    ax.annotate("", xy=(4.0, 4.5), xytext=(6.0, 4.5),
                arrowprops=dict(arrowstyle="-|>", color=DARK, lw=2, mutation_scale=15))
    ax.text(5.0, 3.9, "State + Reward", ha="center", fontsize=9, color=DARK)
    ax.text(5, 1.2, "Goal: maximize cumulative reward",
            ha="center", fontsize=9, color=MUTED, style="italic")

    for ax, title, subtitle, color in zip(axes, titles, subtitles, colors):
        ax.set_title(f"{title}\n{subtitle}", fontsize=11,
                     fontweight="bold", color=color)

    fig.suptitle("Three Paradigms of Machine Learning", fontsize=14,
                 fontweight="bold", color=DARK, y=1.02)
    fig.tight_layout()
    save(fig, "ml_paradigms.png", ["ch01"])


# ══════════════════════════════════════════════════════════════════════════════
#  Ch02 — Data Cleaning (moved from 2-selection_cleaning_preparing/generate_images.py)
# ══════════════════════════════════════════════════════════════════════════════

import pandas as pd
from sklearn.model_selection import train_test_split

raw_data = {
    "student_id": range(1, 21),
    "age": [22,25,23,None,21,999,24,22,26,23,20,25,None,22,24,21,23,25,22,24],
    "gender": ["Male","female","MALE","Female",None,"male","Female","Male",
               "M","Female","male","Female","Male",None,"female","Male",
               "Female","male","FEMALE","Male"],
    "study_hours": [15,20,None,18,25,12,30,22,None,16,19,28,14,21,17,None,23,26,18,150],
    "grade": ["A","B","A","C","B",None,"A","B","C","A","B","A","C","B",None,"A","B","C","A","B"],
    "faculty": ["Science","Arts","Science","Engineering","Arts","Science",None,"Arts",
                "Engineering","Science","Arts","Science","Engineering","Arts","Science",
                "Engineering","Arts","Science","Arts","Engineering"],
    "score": [82,75,91,68,78,55,95,80,72,88,76,92,65,79,83,70,84,77,90,73],
}
df_raw = pd.DataFrame(raw_data)


def make_pipeline_overview():
    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 1); ax.axis("off")
    steps = ["Raw\nData", "Inspect", "Clean", "Encode", "Split", "Scale", "Model"]
    colors = [MUTED, TEAL, TEAL, TEAL, TEAL, TEAL, TEAL_DARK]
    n = len(steps)
    xs = np.linspace(0.5, 9.5, n)
    for i, (x, label, color) in enumerate(zip(xs, steps, colors)):
        box = mpatches.FancyBboxPatch((x-0.55, 0.22), 1.1, 0.56,
            boxstyle="round,pad=0.05", facecolor=color, edgecolor="white",
            linewidth=1.5, zorder=2)
        ax.add_patch(box)
        ax.text(x, 0.50, label, ha="center", va="center",
                color="white", fontsize=9.5, fontweight="bold", zorder=3)
        if i < n - 1:
            ax.annotate("", xy=(xs[i+1]-0.58, 0.50), xytext=(x+0.57, 0.50),
                        arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.5), zorder=1)
    ax.text(5.0, 0.92, "The ML Pipeline", ha="center", va="center",
            fontsize=13, fontweight="bold", color=DARK)
    save(fig, "pipeline_overview.png", ["ch02"])


def make_missing_values_heatmap():
    df = df_raw[["age","gender","study_hours","grade","faculty"]].copy()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False,
                cmap=["#f0f0f0", TEAL], linewidths=0.5, linecolor=BG, ax=ax)
    ax.set_title("Missing Values (teal = missing)", pad=10)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0, labelsize=10)
    save(fig, "missing_values_heatmap.png", ["ch02"])


def make_outlier_boxplot():
    df = df_raw.copy()
    df["age"] = df["age"].fillna(df["age"].median())
    df["study_hours"] = df["study_hours"].fillna(df["study_hours"].median())
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.4))
    for ax, col, lbl in zip(axes, ["age","study_hours","score"], ["Age","Study hours / week","Score"]):
        ax.boxplot(df[col].dropna(), patch_artist=True, widths=0.45,
                   boxprops=dict(facecolor=TEAL, alpha=0.55, linewidth=1.2),
                   medianprops=dict(color=TEAL_DARK, linewidth=2),
                   whiskerprops=dict(color=MUTED, linewidth=1.2),
                   capprops=dict(color=MUTED, linewidth=1.2),
                   flierprops=dict(marker="o", markerfacecolor=RED,
                                   markeredgecolor="white", markersize=8))
        ax.set_title(lbl, fontsize=11); ax.set_xticks([]); ax.yaxis.grid(True)
    axes[1].annotate("outlier!", xy=(1.18, 150), xytext=(1.38, 120),
                     fontsize=9, color=RED, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    fig.suptitle("Boxplot — Outlier Detection", fontsize=13,
                 fontweight="bold", color=DARK, y=1.02)
    save(fig, "outlier_boxplot.png", ["ch02"])


def make_onehot_encoding():
    before = pd.DataFrame({"faculty": ["Science","Arts","Engineering","Arts","Science"]})
    after  = pd.get_dummies(before["faculty"], prefix="faculty", dtype=int)
    combined = pd.concat([before, after], axis=1)
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.axis("off")
    cols = list(combined.columns)
    col_w = 1.0 / len(cols)
    for ci, col in enumerate(cols):
        x = ci * col_w
        is_new = col != "faculty"
        ax.add_patch(plt.Rectangle((x, 0.78), col_w, 0.22,
                                   facecolor=TEAL, edgecolor=BG, lw=1))
        ax.text(x+col_w/2, 0.89, col, ha="center", va="center",
                color="white", fontsize=8.5, fontweight="bold")
        for ri, val in enumerate(combined[col]):
            y = 0.78 - (ri+1)*0.15
            fill = "#e8f9f9" if (is_new and val==1) else ("#f6fdfd" if ri%2==0 else BG)
            ax.add_patch(plt.Rectangle((x, y), col_w, 0.15,
                                       facecolor=fill, edgecolor=BORDER, lw=0.5))
            txt_c = TEAL_DARK if (is_new and val==1) else DARK
            ax.text(x+col_w/2, y+0.075, str(val), ha="center", va="center",
                    fontsize=9, color=txt_c,
                    fontweight="bold" if (is_new and val==1) else "normal")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.set_title("One-Hot Encoding — faculty column",
                 fontsize=12, fontweight="bold", color=DARK, pad=6)
    save(fig, "onehot_encoding.png", ["ch02"])


def make_feature_scaling():
    df = df_raw.copy()
    for col in ["age","study_hours"]:
        df[col] = df[col].fillna(df[col].median())
    df.loc[df["age"] > 100, "age"] = df["age"].median()
    df["study_hours"] = df["study_hours"].clip(upper=40)
    feats = ["age","study_hours","score"]
    labels = ["Age","Study hrs","Score"]
    df_std = pd.DataFrame(StandardScaler().fit_transform(df[feats]), columns=feats)
    df_mm  = pd.DataFrame(MinMaxScaler().fit_transform(df[feats]), columns=feats)
    fig, axes = plt.subplots(3, 3, figsize=(9, 5.5))
    for ri, (d, rtitle, color) in enumerate(zip(
            [df, df_std, df_mm],
            ["Original","StandardScaler\n(mean=0, std=1)","MinMaxScaler\n([0, 1])"],
            [MUTED, TEAL, TEAL_DARK])):
        for ci, (col, lbl) in enumerate(zip(feats, labels)):
            ax = axes[ri, ci]
            ax.hist(d[col], bins=8, color=color, alpha=0.72, edgecolor="white", lw=0.7)
            if ri == 0: ax.set_title(lbl, fontsize=10, fontweight="bold")
            if ci == 0: ax.set_ylabel(rtitle, fontsize=8.5, color=DARK)
            ax.yaxis.grid(True, alpha=0.6); ax.tick_params(labelsize=8)
    fig.suptitle("Feature Scaling — Before vs After",
                 fontsize=13, fontweight="bold", color=DARK, y=1.01)
    fig.tight_layout()
    save(fig, "feature_scaling.png", ["ch02"])


def make_train_test_split():
    fig, ax = plt.subplots(figsize=(8, 2.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0.55), 10, 0.38, facecolor=BORDER, edgecolor=BG, lw=0))
    ax.text(5, 0.74, "Full dataset  (n = 20)", ha="center", va="center",
            fontsize=10, color=DARK, fontweight="bold")
    ax.add_patch(plt.Rectangle((0, 0.07), 8.0, 0.38, facecolor=TEAL, edgecolor=BG, lw=0))
    ax.add_patch(plt.Rectangle((8.0, 0.07), 2.0, 0.38, facecolor=RED, alpha=0.75, edgecolor=BG, lw=0))
    ax.text(4.0, 0.26, "Train  80%", ha="center", va="center",
            fontsize=10, color="white", fontweight="bold")
    ax.text(9.0, 0.26, "Test  20%", ha="center", va="center",
            fontsize=10, color="white", fontweight="bold")
    ax.annotate("", xy=(5, 0.07), xytext=(5, 0.54),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.5))
    ax.set_title("Train / Test Split  (random_state=42, stratify=y)",
                 fontsize=12, fontweight="bold", color=DARK, pad=4)
    save(fig, "train_test_split.png", ["ch02"])



# ══════════════════════════════════════════════════════════════════════════════
#  Ch03 — Supervised Learning Intro
# ══════════════════════════════════════════════════════════════════════════════

def make_overfit_curves():
    """3-panel: underfit / good fit / overfit on noisy sine."""
    np.random.seed(42)
    X = np.linspace(0, 2*np.pi, 15)
    y = np.sin(X) + np.random.normal(0, 0.2, 15)
    X_fine = np.linspace(0, 2*np.pi, 200)

    from numpy.polynomial import polynomial as P
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ["Underfitting\n(degree=1)", "Good Fit\n(degree=3)", "Overfitting\n(degree=12)"]
    degrees = [1, 3, 12]
    colors_fit = [BLUE, TEAL_DARK, RED]

    for ax, deg, title, col in zip(axes, degrees, titles, colors_fit):
        pipe = Pipeline([("poly", PolynomialFeatures(degree=deg)),
                         ("lr", LinearRegression())])
        pipe.fit(X.reshape(-1,1), y)
        y_fine = pipe.predict(X_fine.reshape(-1,1))
        ax.scatter(X, y, color=DARK, s=40, zorder=5, label="Data")
        ax.plot(X_fine, np.sin(X_fine), color=MUTED, lw=1.5, linestyle="--", label="True")
        ax.plot(X_fine, y_fine.clip(-2, 2), color=col, lw=2.5, label=f"Fit (d={deg})")
        ax.set_title(title, fontsize=12, color=col)
        ax.set_ylim(-2, 2)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.4)

    fig.suptitle("Underfitting vs Good Fit vs Overfitting", fontsize=14,
                 fontweight="bold", color=DARK, y=1.02)
    fig.tight_layout()
    save(fig, "overfit_curves.png", ["ch03"])


def make_bias_variance():
    """Classic U-shaped bias-variance decomposition."""
    complexity = np.linspace(0, 10, 200)
    bias2   = 2.5 * np.exp(-0.4 * complexity) + 0.1
    variance= 0.05 * np.exp(0.4 * complexity) - 0.04
    variance= np.clip(variance, 0, None)
    noise   = np.full_like(complexity, 0.15)
    total   = bias2 + variance + noise

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(complexity, 0, bias2, alpha=0.25, color=BLUE, label="Bias²")
    ax.fill_between(complexity, bias2, bias2+variance, alpha=0.25,
                    color=ORANGE, label="Variance")
    ax.fill_between(complexity, bias2+variance, total, alpha=0.2,
                    color=MUTED, label="Irreducible Noise")
    ax.plot(complexity, total, color=RED, lw=2.5, label="Total Error")
    ax.plot(complexity, bias2, color=BLUE, lw=1.5, linestyle="--")
    ax.plot(complexity, bias2+variance, color=ORANGE, lw=1.5, linestyle="--")

    best_idx = np.argmin(total)
    ax.axvline(complexity[best_idx], color=TEAL_DARK, lw=2, linestyle="-.",
               label="Sweet spot")
    ax.annotate("Sweet spot\n(best generalization)", xy=(complexity[best_idx], total[best_idx]),
                xytext=(complexity[best_idx]+1.5, total[best_idx]+0.3),
                arrowprops=dict(arrowstyle="->", color=TEAL_DARK, lw=1.5),
                fontsize=10, color=TEAL_DARK, fontweight="bold")

    ax.set_xlabel("Model Complexity", fontsize=12)
    ax.set_ylabel("Error", fontsize=12)
    ax.set_title("Bias-Variance Tradeoff", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, None)
    ax.set_xlim(0, 10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, "bias_variance.png", ["ch03"])


def make_cross_val_folds():
    """5-fold cross-validation diagram."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(-0.5, 11)
    ax.set_ylim(-0.5, 6.5)
    ax.axis("off")

    n_folds = 5
    n_blocks = 5
    block_w = 1.8
    block_h = 0.7

    for fold in range(n_folds):
        y_pos = n_folds - fold - 1
        ax.text(-0.3, y_pos * 1.1 + block_h/2,
                f"Fold {fold+1}", ha="right", va="center",
                fontsize=10, color=DARK, fontweight="bold")
        for block in range(n_blocks):
            is_test = (block == fold)
            color = RED if is_test else TEAL
            alpha = 0.85
            rect = mpatches.FancyBboxPatch(
                (block * (block_w + 0.1), y_pos * 1.1),
                block_w, block_h,
                boxstyle="round,pad=0.04",
                facecolor=color, edgecolor="white", linewidth=1.5,
                alpha=alpha
            )
            ax.add_patch(rect)
            lbl = "TEST" if is_test else "TRAIN"
            ax.text(block * (block_w + 0.1) + block_w/2,
                    y_pos * 1.1 + block_h/2,
                    lbl, ha="center", va="center",
                    color="white", fontsize=9.5, fontweight="bold")
        score_x = n_blocks * (block_w + 0.1) + 0.2
        ax.text(score_x, y_pos * 1.1 + block_h/2,
                f"→ score₍{fold+1}₎", ha="left", va="center",
                fontsize=9, color=MUTED)

    final_y = -0.3
    ax.text(n_blocks * (block_w + 0.1) / 2, final_y,
            "Final score = mean(score₁, score₂, score₃, score₄, score₅)",
            ha="center", va="center", fontsize=11,
            color=TEAL_DARK, fontweight="bold")

    # Legend
    ax.add_patch(mpatches.FancyBboxPatch((0, 5.7), 0.6, 0.4,
        boxstyle="round,pad=0.03", facecolor=TEAL, edgecolor="white", lw=1))
    ax.text(0.7, 5.9, "Train", ha="left", va="center", fontsize=9, color=DARK)
    ax.add_patch(mpatches.FancyBboxPatch((2.0, 5.7), 0.6, 0.4,
        boxstyle="round,pad=0.03", facecolor=RED, edgecolor="white", lw=1))
    ax.text(2.7, 5.9, "Test (held out)", ha="left", va="center", fontsize=9, color=DARK)

    ax.set_title("5-Fold Cross-Validation", fontsize=14,
                 fontweight="bold", color=DARK, pad=12)
    save(fig, "cross_val_folds.png", ["ch03"])


# ══════════════════════════════════════════════════════════════════════════════
#  Ch04 — Regression
# ══════════════════════════════════════════════════════════════════════════════

def make_linear_reg_fit():
    """Scatter + regression line with annotated residual."""
    np.random.seed(7)
    X = np.linspace(20, 120, 30)
    y = 1.8 * X + 50 + np.random.normal(0, 18, 30)

    from numpy.polynomial import polynomial as P
    coeffs = np.polyfit(X, y, 1)
    y_fit  = np.polyval(coeffs, X)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(X, y, color=TEAL, s=55, zorder=5, alpha=0.85, edgecolors="white", lw=0.5)
    ax.plot(X, y_fit, color=RED, lw=2.5, label="Regression line  ŷ = β₀ + β₁x")

    # Annotate one residual
    idx = 15
    ax.plot([X[idx], X[idx]], [y[idx], y_fit[idx]],
            color=ORANGE, lw=2.5, zorder=6)
    ax.annotate("residual\n(actual − predicted)",
                xy=(X[idx], (y[idx]+y_fit[idx])/2),
                xytext=(X[idx]+12, (y[idx]+y_fit[idx])/2),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.5),
                fontsize=10, color=ORANGE, fontweight="bold")

    ax.set_xlabel("Area (sqm)", fontsize=12)
    ax.set_ylabel("Price (k€)", fontsize=12)
    ax.set_title("Linear Regression — Fit & Residual", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, "linear_reg_fit.png", ["ch04"])



# ══════════════════════════════════════════════════════════════════════════════
#  Ch05 — Classification
# ══════════════════════════════════════════════════════════════════════════════

def make_sigmoid_curve():
    """Clean sigmoid plot with decision boundary annotation."""
    z = np.linspace(-6, 6, 300)
    sigma = 1 / (1 + np.exp(-z))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(z, sigma, color=TEAL, lw=3)
    ax.fill_between(z, 0, sigma, alpha=0.08, color=TEAL)

    # Decision boundary
    ax.axvline(0, color=MUTED, lw=1.5, linestyle="--")
    ax.axhline(0.5, color=MUTED, lw=1.5, linestyle="--")
    ax.plot(0, 0.5, "o", color=RED, markersize=10, zorder=5)
    ax.annotate("σ(0) = 0.5\n(decision boundary)", xy=(0, 0.5),
                xytext=(1.2, 0.38),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5),
                fontsize=10, color=RED, fontweight="bold")

    # Region labels
    ax.text(-3.5, 0.12, "→ Predict class 0", fontsize=11, color=BLUE,
            fontweight="bold")
    ax.text(1.5, 0.88, "→ Predict class 1", fontsize=11, color=TEAL_DARK,
            fontweight="bold")

    ax.set_xlabel("z  (linear combination of features)", fontsize=12)
    ax.set_ylabel("σ(z) = P(y=1 | X)", fontsize=12)
    ax.set_title("Sigmoid Function — Logistic Regression Output", fontsize=14,
                 fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, "sigmoid_curve.png", ["ch05"])


def make_svm_margin():
    """SVM: decision boundary + margin + support vectors."""
    np.random.seed(3)
    X1 = np.random.randn(20, 2) + np.array([-2, -1])
    X2 = np.random.randn(20, 2) + np.array([2, 1])

    from sklearn.svm import SVC
    X_all = np.vstack([X1, X2])
    y_all = np.array([0]*20 + [1]*20)
    svm = SVC(kernel="linear", C=1.0)
    svm.fit(X_all, y_all)

    w = svm.coef_[0]
    b = svm.intercept_[0]
    xx = np.linspace(-5, 5, 200)
    yy_boundary = -(w[0]*xx + b) / w[1]
    yy_margin1  = -(w[0]*xx + b + 1) / w[1]
    yy_margin2  = -(w[0]*xx + b - 1) / w[1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X1[:,0], X1[:,1], color=BLUE, s=55, alpha=0.8,
               edgecolors="white", label="Class 0", zorder=5)
    ax.scatter(X2[:,0], X2[:,1], color=ORANGE, s=55, alpha=0.8,
               edgecolors="white", label="Class 1", zorder=5)

    ax.plot(xx, yy_boundary, color=DARK, lw=2.5, label="Decision boundary")
    ax.plot(xx, yy_margin1,  color=TEAL, lw=1.5, linestyle="--", label="Margin")
    ax.plot(xx, yy_margin2,  color=TEAL, lw=1.5, linestyle="--")
    ax.fill_between(xx, yy_margin1, yy_margin2, alpha=0.12, color=TEAL)

    # Support vectors
    sv = svm.support_vectors_
    ax.scatter(sv[:,0], sv[:,1], s=180, facecolors="none",
               edgecolors=RED, linewidths=2.5, zorder=6, label="Support vectors")

    # Annotate margin
    mid_idx = 100
    ax.annotate("", xy=(xx[mid_idx], yy_margin1[mid_idx]),
                xytext=(xx[mid_idx], yy_margin2[mid_idx]),
                arrowprops=dict(arrowstyle="<->", color=TEAL_DARK, lw=2.0))
    ax.text(xx[mid_idx]+0.2, (yy_margin1[mid_idx]+yy_margin2[mid_idx])/2,
            "margin\n(maximised)", fontsize=10, color=TEAL_DARK, fontweight="bold")

    ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
    ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2")
    ax.set_title("Support Vector Machine — Maximum Margin", fontsize=14,
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, "svm_margin.png", ["ch05"])



# ══════════════════════════════════════════════════════════════════════════════
#  Ch06 — Metrics & Evaluation
# ══════════════════════════════════════════════════════════════════════════════


def make_roc_curve():
    """ROC curve with AUC shading."""
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_curve, auc

    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color=TEAL, lw=2.5, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.fill_between(fpr, tpr, alpha=0.15, color=TEAL)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.5, linestyle="--", label="Random (AUC = 0.5)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title("ROC Curve & AUC Score", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, "roc_curve.png", ["ch06"])



# ══════════════════════════════════════════════════════════════════════════════
#  Ch07 — Unsupervised Intro
# ══════════════════════════════════════════════════════════════════════════════

def make_supervised_vs_unsupervised():
    """Side-by-side: labeled vs unlabeled data."""
    np.random.seed(42)
    X1 = np.random.randn(30, 2) + [2, 2]
    X2 = np.random.randn(30, 2) + [-2, -1]
    X3 = np.random.randn(20, 2) + [2, -2]
    X_all = np.vstack([X1, X2, X3])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Unlabeled (unsupervised view)
    axes[0].scatter(X_all[:,0], X_all[:,1], color="steelblue", alpha=0.6,
                    s=40, edgecolors="white")
    axes[0].set_title("Unsupervised: X only\n'What is in this data?'",
                       fontsize=12, fontweight="bold", color=BLUE)
    axes[0].set_xlabel("Feature 1"); axes[0].set_ylabel("Feature 2")
    axes[0].grid(True, alpha=0.3)

    # Labeled (supervised view)
    colors_lbl = [RED, TEAL, ORANGE]
    class_labels = ["Class A", "Class B", "Class C"]
    for X_g, color, lbl in zip([X1, X2, X3], colors_lbl, class_labels):
        axes[1].scatter(X_g[:,0], X_g[:,1], color=color, alpha=0.7,
                        s=40, edgecolors="white", label=lbl)
    axes[1].set_title("Supervised: (X, y) pairs\n'Predict the label'",
                       fontsize=12, fontweight="bold", color=TEAL_DARK)
    axes[1].set_xlabel("Feature 1"); axes[1].set_ylabel("Feature 2")
    axes[1].legend(fontsize=10); axes[1].grid(True, alpha=0.3)

    fig.suptitle("Same Data — Different Perspective",
                 fontsize=14, fontweight="bold", color=DARK, y=1.02)
    fig.tight_layout()
    save(fig, "supervised_vs_unsupervised.png", ["ch07"])


# ══════════════════════════════════════════════════════════════════════════════
#  Ch08 — Clustering
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
#  Ch09 — Dimensionality Reduction
# ══════════════════════════════════════════════════════════════════════════════



# ══════════════════════════════════════════════════════════════════════════════
#  Ch10 — RL Intro
# ══════════════════════════════════════════════════════════════════════════════

def make_rl_loop():
    """Polished Agent ↔ Environment loop diagram."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

    # Agent box
    ax.add_patch(mpatches.FancyBboxPatch((0.5, 1.8), 3.0, 2.4,
        boxstyle="round,pad=0.15", facecolor=ORANGE, edgecolor="white", lw=3))
    ax.text(2.0, 3.0, "Agent\n(π)", ha="center", va="center",
            color="white", fontsize=16, fontweight="bold")

    # Environment box
    ax.add_patch(mpatches.FancyBboxPatch((6.5, 1.8), 3.0, 2.4,
        boxstyle="round,pad=0.15", facecolor=TEAL, edgecolor="white", lw=3))
    ax.text(8.0, 3.0, "Environment", ha="center", va="center",
            color="white", fontsize=16, fontweight="bold")

    # Arrow: Action (top)
    ax.annotate("", xy=(6.5, 4.0), xytext=(3.5, 4.0),
                arrowprops=dict(arrowstyle="-|>", color=DARK, lw=2.5, mutation_scale=20))
    ax.text(5.0, 4.35, "Action  aₜ", ha="center", fontsize=12,
            fontweight="bold", color=DARK)

    # Arrow: State (bottom-left)
    ax.annotate("", xy=(3.5, 2.2), xytext=(6.5, 2.2),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.5, mutation_scale=20))
    ax.text(5.0, 2.55, "State  sₜ₊₁", ha="center", fontsize=12,
            fontweight="bold", color=BLUE)

    # Arrow: Reward (further bottom)
    ax.annotate("", xy=(3.5, 1.55), xytext=(6.5, 1.55),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.5, mutation_scale=20))
    ax.text(5.0, 1.15, "Reward  rₜ", ha="center", fontsize=12,
            fontweight="bold", color=RED)

    ax.set_title("The Reinforcement Learning Loop",
                 fontsize=15, fontweight="bold", color=DARK, pad=12)
    ax.text(5.0, 0.4, "Goal: maximize total cumulative reward",
            ha="center", fontsize=11, color=MUTED, style="italic")
    save(fig, "rl_loop.png", ["ch01", "ch10"])


def make_discount_factor():
    """Timeline showing how γ discounts future rewards."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(-0.5, 9); ax.set_ylim(-0.5, 3.5); ax.axis("off")

    gamma = 0.9
    rewards = [1.0, 1.0, 1.0, 1.0]
    labels = ["r₀", "r₁", "r₂", "r₃"]
    times = [1, 3, 5, 7]

    for t, (r, lbl) in enumerate(zip(rewards, labels)):
        x = times[t]
        disc = gamma ** t
        height = disc * 2.2

        # Bar
        ax.bar(x, height, width=0.9, color=TEAL, alpha=0.7 + 0.08*t,
               edgecolor="white", lw=1.5)
        # Reward label
        ax.text(x, height + 0.1, f"{lbl}", ha="center", fontsize=12,
                fontweight="bold", color=DARK)
        # Discount factor
        ax.text(x, height/2, f"×{disc:.3f}", ha="center", fontsize=9.5,
                color="white", fontweight="bold")
        # Time label
        ax.text(x, -0.3, f"t = {t}", ha="center", fontsize=10, color=MUTED)

    ax.set_title(f"Discount Factor γ = {gamma}: Future Rewards Are Worth Less",
                 fontsize=13, fontweight="bold", color=DARK, pad=10)
    ax.text(4.0, -0.5, "← Nearby rewards matter more than distant rewards →",
            ha="center", fontsize=10, color=MUTED, style="italic")
    save(fig, "discount_factor.png", ["ch10"])


# ══════════════════════════════════════════════════════════════════════════════
#  Ch11 — RL Algorithms
# ══════════════════════════════════════════════════════════════════════════════

def make_td_error_flow():
    """Horizontal flow: Bellman / TD error diagram."""
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 3); ax.axis("off")

    boxes = [
        (1.0, "Current\nQ(s, a)", BLUE),
        (3.5, "Observed\nReward  r", GREEN),
        (6.0, "γ · max Q(s', a')\n(best future)", TEAL),
        (8.8, "TD Error\n(how wrong?)", RED),
        (11.2, "Updated\nQ(s, a)", ORANGE),
    ]
    box_w, box_h = 1.8, 1.1

    for x, label, color in boxes:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - box_w/2, 1.0), box_w, box_h,
            boxstyle="round,pad=0.08",
            facecolor=color, edgecolor="white", lw=2, zorder=3))
        ax.text(x, 1.55, label, ha="center", va="center",
                color="white", fontsize=8.5, fontweight="bold", zorder=4)

    # Arrows
    pairs = [(1.0, 8.8), (3.5, 8.8), (6.0, 8.8), (8.8, 11.2)]
    for x_from, x_to in pairs:
        ax.annotate("", xy=(x_to - box_w/2, 1.55),
                    xytext=(x_from + box_w/2, 1.55),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED,
                                    lw=1.8, mutation_scale=14))

    # Labels on arrows
    ax.text(5.1, 1.85, "=   target  −  current",
            ha="center", fontsize=9, color=RED, fontweight="bold")
    ax.text(10.0, 1.85, "+ α × TD Error",
            ha="center", fontsize=9, color=ORANGE, fontweight="bold")

    ax.set_title("Bellman Update — Q-Learning Step",
                 fontsize=13, fontweight="bold", color=DARK, pad=10)
    ax.text(6.0, 0.3, "Q(s,a) ← Q(s,a) + α · [r + γ · max Q(s',a') − Q(s,a)]",
            ha="center", fontsize=11, color=DARK,
            fontfamily="monospace")
    save(fig, "td_error_flow.png", ["ch11"])



# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("APML UZH — Generating all slide images\n")
    from imagegen import ch01, ch02, ch03, ch04, ch05, ch06, ch07, ch08, ch09, ch10, ch11, ch12

    print("── Ch01: Introduction ─────────────────────────────────")
    make_workflow_cycle()
    make_ml_paradigms()
    ch01.generate()

    print("── Ch02: Data Cleaning ────────────────────────────────")
    make_pipeline_overview()
    make_missing_values_heatmap()
    make_outlier_boxplot()
    make_onehot_encoding()
    make_feature_scaling()
    make_train_test_split()
    ch02.generate()

    print("── Ch03: Supervised Intro ─────────────────────────────")
    make_overfit_curves()
    make_bias_variance()
    make_cross_val_folds()
    ch03.generate()

    print("── Ch04: Regression ───────────────────────────────────")
    make_linear_reg_fit()
    ch04.generate()

    print("── Ch05: Classification ───────────────────────────────")
    make_sigmoid_curve()
    make_svm_margin()
    ch05.generate()

    print("── Ch06: Metrics & Evaluation ─────────────────────────")
    make_roc_curve()
    ch06.generate()

    print("── Ch07: Unsupervised Intro ────────────────────────────")
    make_supervised_vs_unsupervised()
    ch07.generate()

    print("── Ch08: Clustering ───────────────────────────────────")
    ch08.generate()

    print("── Ch09: Dimensionality Reduction ─────────────────────")
    ch09.generate()

    print("── Ch10: RL Intro ─────────────────────────────────────")
    make_rl_loop()
    make_discount_factor()
    ch10.generate()

    print("── Ch11: RL Algorithms ────────────────────────────────")
    make_td_error_flow()
    ch11.generate()

    print("── Ch12: Capstone ─────────────────────────────────────")
    ch12.generate()

    print(f"\n✅  Done — images written to each chapter's 01-slides/ and slidev/public/")
