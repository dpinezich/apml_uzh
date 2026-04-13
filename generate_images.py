"""
APML UZH — Master image generator
===================================
Generates all slide images for all 12 chapters.

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


def make_data_leakage():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.0))
    def draw_panel(ax, title, split_first, ok):
        ax.set_xlim(0, 10); ax.set_ylim(0, 1); ax.axis("off")
        color = TEAL if ok else RED
        icon  = "✓" if ok else "✗"
        if split_first:
            boxes = [(0.2,"All data",MUTED),(2.5,"Split",color),
                     (5.0,"Impute\n(train)",color),(7.5,"Impute\n(test)",color)]
        else:
            boxes = [(0.2,"All data",MUTED),(2.5,"Impute\n(all!)",RED),
                     (5.0,"Split",MUTED),(7.5,"Train/Test",MUTED)]
        for i, (x, lbl, c) in enumerate(boxes):
            ax.add_patch(mpatches.FancyBboxPatch((x-0.85, 0.28), 1.7, 0.44,
                boxstyle="round,pad=0.05", facecolor=c, edgecolor="white", lw=1.5, zorder=2))
            ax.text(x, 0.50, lbl, ha="center", va="center",
                    color="white", fontsize=8.5, fontweight="bold", zorder=3)
            if i < len(boxes)-1:
                ax.annotate("", xy=(boxes[i+1][0]-0.87, 0.50), xytext=(x+0.87, 0.50),
                            arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.4))
        ax.text(5, 0.90, f"{icon} {title}", ha="center", va="center",
                fontsize=11, fontweight="bold", color=color)
        if not ok:
            ax.text(3.5, 0.10, "Test data leaked into imputer fit!",
                    ha="center", va="center", fontsize=8.5, color=RED, style="italic")
    draw_panel(axes[0], "Wrong — leakage", split_first=False, ok=False)
    draw_panel(axes[1], "Correct — Pipeline", split_first=True, ok=True)
    fig.suptitle("Data Leakage", fontsize=13, fontweight="bold", color=DARK, y=1.02)
    fig.tight_layout()
    save(fig, "data_leakage.png", ["ch02"])


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


def make_regularization_coefs():
    """Grouped bar: coefficient magnitudes for Linear / Ridge / Lasso."""
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    np.random.seed(42)
    n, p = 80, 8
    X = np.random.randn(n, p)
    # Only first 4 features are truly relevant
    true_coef = np.array([3.0, -2.0, 1.5, -1.0, 0.1, 0.05, -0.02, 0.01])
    y = X @ true_coef + np.random.randn(n) * 0.5

    feature_names = [f"f{i+1}" for i in range(p)]
    models = [
        ("Linear", LinearRegression().fit(X, y), BLUE),
        ("Ridge (α=1)", Ridge(alpha=1.0).fit(X, y), TEAL),
        ("Lasso (α=0.1)", Lasso(alpha=0.1, max_iter=5000).fit(X, y), ORANGE),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(p)
    width = 0.26
    for i, (name, model, color) in enumerate(models):
        ax.bar(x + i*width, model.coef_, width, label=name,
               color=color, alpha=0.82, edgecolor="white", lw=0.8)

    ax.axhline(0, color=DARK, lw=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(feature_names, fontsize=10)
    ax.set_xlabel("Feature", fontsize=12)
    ax.set_ylabel("Coefficient value", fontsize=12)
    ax.set_title("Regularization: How Coefficients Shrink\n"
                 "(Features 5–8 are irrelevant noise)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "regularization_coefs.png", ["ch04"])


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


def make_decision_boundaries():
    """5 classifiers decision boundaries on same 2D data."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    np.random.seed(0)
    X, y = make_blobs(n_samples=200, centers=2, cluster_std=1.6, random_state=4)
    ss = StandardScaler()
    X_sc = ss.fit_transform(X)

    classifiers = [
        ("Logistic\nRegression", LogisticRegression()),
        ("KNN\n(k=5)", KNeighborsClassifier(n_neighbors=5)),
        ("Decision\nTree", DecisionTreeClassifier(max_depth=4)),
        ("Random\nForest", RandomForestClassifier(n_estimators=50, random_state=42)),
        ("SVM\n(RBF)", SVC(kernel="rbf", C=1.0, probability=True)),
    ]

    h = 0.05
    x_min, x_max = X_sc[:,0].min()-1, X_sc[:,0].max()+1
    y_min, y_max = X_sc[:,1].min()-1, X_sc[:,1].max()+1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                          np.arange(y_min, y_max, h))

    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    colors_cls = [BLUE, ORANGE]

    for ax, (name, clf) in zip(axes, classifiers):
        clf.fit(X_sc, y)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        acc = clf.score(X_sc, y)
        ax.contourf(xx, yy, Z, alpha=0.25,
                    cmap=plt.cm.RdYlGn)
        for cls, color in enumerate(colors_cls):
            mask = y == cls
            ax.scatter(X_sc[mask,0], X_sc[mask,1], color=color,
                       s=20, alpha=0.8, edgecolors="white", lw=0.3)
        ax.set_title(f"{name}\nacc={acc:.2f}", fontsize=9.5,
                     fontweight="bold", color=DARK)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Decision Boundaries — Same Data, Five Classifiers",
                 fontsize=13, fontweight="bold", color=DARK, y=1.02)
    fig.tight_layout()
    save(fig, "decision_boundaries.png", ["ch05"])


# ══════════════════════════════════════════════════════════════════════════════
#  Ch06 — Metrics & Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def make_confusion_matrix_viz():
    """Color-coded 2×2 confusion matrix with labels and icons."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.set_xlim(0, 4); ax.set_ylim(0, 3.5); ax.axis("off")

    cells = [
        # (col, row, color, label, formula, icon)
        (0, 1, "#d5f5e3", "True Negative\n(TN)", "Pred=0, Actual=0", "✓"),
        (1, 1, "#fadbd8", "False Positive\n(FP)", "Pred=1, Actual=0", "✗"),
        (0, 0, "#fadbd8", "False Negative\n(FN)", "Pred=0, Actual=1", "✗"),
        (1, 0, "#d5f5e3", "True Positive\n(TP)", "Pred=1, Actual=1", "✓"),
    ]
    icon_colors = {"✓": GREEN, "✗": RED}

    for col, row, color, label, formula, icon in cells:
        x0, y0 = col * 2, row * 1.4 + 0.4
        ax.add_patch(plt.Rectangle((x0, y0), 2.0, 1.35,
                                   facecolor=color, edgecolor="white", lw=3))
        ax.text(x0+1.0, y0+0.95, label, ha="center", va="center",
                fontsize=10, fontweight="bold", color=DARK)
        ax.text(x0+1.0, y0+0.55, formula, ha="center", va="center",
                fontsize=8.5, color=MUTED)
        ax.text(x0+1.0, y0+0.18, icon, ha="center", va="center",
                fontsize=18, color=icon_colors[icon], fontweight="bold")

    # Headers
    ax.text(1.0, 3.15, "Predicted: 0", ha="center", va="center",
            fontsize=11, fontweight="bold", color=DARK)
    ax.text(3.0, 3.15, "Predicted: 1", ha="center", va="center",
            fontsize=11, fontweight="bold", color=DARK)
    ax.text(-0.35, 2.07, "Actual: 0", ha="center", va="center",
            fontsize=11, fontweight="bold", color=DARK, rotation=90)
    ax.text(-0.35, 0.77, "Actual: 1", ha="center", va="center",
            fontsize=11, fontweight="bold", color=DARK, rotation=90)

    ax.set_title("Confusion Matrix — Binary Classification",
                 fontsize=13, fontweight="bold", color=DARK, pad=8)
    save(fig, "confusion_matrix_viz.png", ["ch06"])


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


def make_precision_recall():
    """Precision-Recall tradeoff vs threshold."""
    thresholds = np.linspace(0.01, 0.99, 200)
    # Simulate realistic curves
    recall    = 1 / (1 + np.exp(6 * (thresholds - 0.4)))
    precision = 1 / (1 + np.exp(-6 * (thresholds - 0.55)))
    precision = np.clip(precision, 0.5, 1)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: P & R vs threshold
    ax = axes[0]
    ax.plot(thresholds, precision, color=BLUE, lw=2.5, label="Precision")
    ax.plot(thresholds, recall,    color=RED,  lw=2.5, label="Recall")
    ax.axvline(0.5, color=MUTED, lw=1.5, linestyle="--", label="Default (0.5)")
    ax.set_xlabel("Decision Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Precision vs Recall vs Threshold", fontsize=12, fontweight="bold")
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Panel 2: F1 peak
    ax = axes[1]
    ax.plot(thresholds, f1, color=TEAL_DARK, lw=2.5, label="F1 Score")
    best_t = thresholds[np.argmax(f1)]
    ax.axvline(best_t, color=ORANGE, lw=2, linestyle="-.",
               label=f"Best threshold ≈ {best_t:.2f}")
    ax.set_xlabel("Decision Threshold", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("F1 Score Peaks at Optimal Threshold", fontsize=12, fontweight="bold")
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    fig.suptitle("Precision-Recall Tradeoff — Adjusting the Decision Threshold",
                 fontsize=13, fontweight="bold", color=DARK, y=1.02)
    fig.tight_layout()
    save(fig, "precision_recall.png", ["ch06"])


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

def make_kmeans_steps():
    """4-panel K-Means convergence: Init → Assign → Move → Converged."""
    from sklearn.cluster import KMeans

    np.random.seed(42)
    X, _ = make_blobs(n_samples=80, centers=3, cluster_std=0.9, random_state=7)
    colors_k = [RED, TEAL, BLUE]

    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    titles = ["Step 1: Initialize\n(random centroids)",
              "Step 2: Assign\n(color by nearest centroid)",
              "Step 3: Move\n(recompute centroids)",
              "Converged\n(no more changes)"]

    # Step 1: random initial centroids
    init_centroids = X[np.random.choice(len(X), 3, replace=False)]
    axes[0].scatter(X[:,0], X[:,1], color=MUTED, s=30, alpha=0.6)
    axes[0].scatter(init_centroids[:,0], init_centroids[:,1],
                    marker="X", s=220, c=colors_k, edgecolors="white", lw=1.5, zorder=6)

    # Step 2–4: run KMeans with increasing iterations
    for ax, title, n_iter in zip(axes[1:], titles[1:], [1, 2, 100]):
        km = KMeans(n_clusters=3, max_iter=n_iter, n_init=1,
                    init=init_centroids, random_state=0)
        km.fit(X)
        labels = km.labels_
        for k, color in enumerate(colors_k):
            mask = labels == k
            ax.scatter(X[mask,0], X[mask,1], color=color, s=30, alpha=0.6)
        ax.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1],
                   marker="X", s=220, c=colors_k, edgecolors="white", lw=1.5, zorder=6)

    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("K-Means: Step-by-Step Convergence",
                 fontsize=13, fontweight="bold", color=DARK, y=1.02)
    fig.tight_layout()
    save(fig, "kmeans_steps.png", ["ch08"])


def make_elbow_curve():
    """Inertia vs k with annotated elbow."""
    from sklearn.cluster import KMeans

    np.random.seed(42)
    X, _ = make_blobs(n_samples=200, centers=3, cluster_std=0.9, random_state=42)
    ks = range(1, 10)
    inertias = [KMeans(n_clusters=k, n_init=10, random_state=42).fit(X).inertia_
                for k in ks]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(ks), inertias, "o-", color=TEAL, lw=2.5, markersize=8)
    ax.axvline(3, color=RED, lw=2, linestyle="--", label="Optimal k = 3")
    ax.scatter([3], [inertias[2]], color=RED, s=120, zorder=6)
    ax.annotate("Elbow point", xy=(3, inertias[2]),
                xytext=(4.5, inertias[2]+30),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5),
                fontsize=11, color=RED, fontweight="bold")
    ax.set_xlabel("Number of clusters (k)", fontsize=12)
    ax.set_ylabel("Inertia (within-cluster sum of squares)", fontsize=12)
    ax.set_title("Elbow Method — Choosing k", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, "elbow_curve.png", ["ch08"])


def make_dbscan_points():
    """DBSCAN: core, border, noise points with epsilon circles."""
    from sklearn.cluster import DBSCAN

    np.random.seed(42)
    X_core = np.random.randn(40, 2) * 0.5 + [2, 2]
    X_noise = np.array([[-1, 4], [5, 0], [0, -1], [4, 4.5]])
    X = np.vstack([X_core, X_noise])

    eps = 0.8
    db = DBSCAN(eps=eps, min_samples=4).fit(X)
    labels = db.labels_

    fig, ax = plt.subplots(figsize=(8, 6))

    # Classify points
    core_samples = set(db.core_sample_indices_)
    for i, (x, y_) in enumerate(X):
        if labels[i] == -1:
            ax.scatter(x, y_, color=RED, s=80, marker="x", lw=2.5, zorder=5)
        elif i in core_samples:
            ax.scatter(x, y_, color=TEAL, s=60, zorder=5)
        else:
            ax.scatter(x, y_, color=TEAL, s=60, facecolors="none",
                       edgecolors=TEAL, lw=2, zorder=5)

    # Draw epsilon circles around 2 core points
    for idx in list(db.core_sample_indices_)[:2]:
        circle = plt.Circle(X[idx], eps, color=TEAL, fill=False,
                             linestyle="--", lw=1.5, alpha=0.6)
        ax.add_patch(circle)
        ax.annotate(f"ε = {eps}", xy=X[idx],
                    xytext=(X[idx][0]+eps+0.1, X[idx][1]),
                    fontsize=8.5, color=TEAL_DARK)

    # Legend
    ax.scatter([], [], color=TEAL, s=60, label="Core point")
    ax.scatter([], [], color=TEAL, s=60, facecolors="none",
               edgecolors=TEAL, lw=2, label="Border point")
    ax.scatter([], [], color=RED, s=80, marker="x", lw=2.5, label="Noise (−1)")

    ax.set_title("DBSCAN — Core, Border, and Noise Points",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, "dbscan_points.png", ["ch08"])


# ══════════════════════════════════════════════════════════════════════════════
#  Ch09 — Dimensionality Reduction
# ══════════════════════════════════════════════════════════════════════════════

def make_pca_directions():
    """2D scatter of correlated data with PC1 and PC2 arrows."""
    from sklearn.decomposition import PCA

    np.random.seed(42)
    cov = [[3, 2], [2, 2]]
    X = np.random.multivariate_normal([0, 0], cov, 150)
    pca = PCA(n_components=2)
    pca.fit(X)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X[:,0], X[:,1], color=TEAL, s=30, alpha=0.5, edgecolors="white")

    origin = np.mean(X, axis=0)
    scale = 2.5
    for i, (comp, color, label) in enumerate(zip(
            pca.components_,
            [RED, BLUE],
            [f"PC1 ({pca.explained_variance_ratio_[0]:.0%} variance)",
             f"PC2 ({pca.explained_variance_ratio_[1]:.0%} variance)"])):
        length = np.sqrt(pca.explained_variance_[i]) * scale
        ax.annotate("", xy=origin + comp * length,
                    xytext=origin,
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    lw=3, mutation_scale=18))
        ax.text(*(origin + comp * (length + 0.4)), label,
                color=color, fontsize=10, fontweight="bold", ha="center")

    ax.set_xlabel("Feature 1", fontsize=12)
    ax.set_ylabel("Feature 2", fontsize=12)
    ax.set_title("PCA — Directions of Maximum Variance", fontsize=13,
                 fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, "pca_directions.png", ["ch09"])


def make_scree_plot():
    """Cumulative explained variance with 95% threshold annotated."""
    from sklearn.decomposition import PCA
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    X_sc = StandardScaler().fit_transform(data.data)
    pca = PCA().fit(X_sc)
    cumvar = pca.explained_variance_ratio_.cumsum()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(1, len(cumvar)+1), cumvar, "o-", color=TEAL, lw=2.5, markersize=5)
    ax.fill_between(range(1, len(cumvar)+1), 0, cumvar, alpha=0.1, color=TEAL)

    thresh = 0.95
    n_comp = int(np.argmax(cumvar >= thresh) + 1)
    ax.axhline(thresh, color=RED, lw=2, linestyle="--")
    ax.axvline(n_comp, color=RED, lw=2, linestyle="--")
    ax.scatter([n_comp], [cumvar[n_comp-1]], color=RED, s=100, zorder=6)
    ax.annotate(f"95% variance\n→ {n_comp} components",
                xy=(n_comp, thresh),
                xytext=(n_comp+2.5, thresh-0.12),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5),
                fontsize=10, color=RED, fontweight="bold")

    ax.set_xlabel("Number of Principal Components", fontsize=12)
    ax.set_ylabel("Cumulative Explained Variance", fontsize=12)
    ax.set_title("PCA Scree Plot — How Many Components to Keep?",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, "scree_plot.png", ["ch09"])


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
    save(fig, "rl_loop.png", ["ch10"])


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


def make_learning_curve():
    """Simulated Q-Learning training curve: reward vs episodes."""
    np.random.seed(42)
    n_episodes = 2000
    window = 50

    # Simulate: early random, then learning, then convergence
    t = np.arange(n_episodes)
    success = 1 / (1 + np.exp(-0.006 * (t - 600)))
    noise = np.random.randn(n_episodes) * 0.15
    raw = success + noise
    rolling = np.convolve(raw, np.ones(window)/window, mode="valid")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(rolling)), rolling, color=TEAL, lw=2.5)
    ax.fill_between(range(len(rolling)), 0, rolling, alpha=0.12, color=TEAL)

    ax.axvline(300,  color=MUTED, lw=1.5, linestyle=":", alpha=0.7)
    ax.axvline(1000, color=MUTED, lw=1.5, linestyle=":", alpha=0.7)
    ax.text(150,  0.85, "Exploration\n(mostly random)", ha="center",
            fontsize=10, color=MUTED)
    ax.text(650,  0.85, "Learning\n(improving)", ha="center",
            fontsize=10, color=TEAL_DARK, fontweight="bold")
    ax.text(1400, 0.85, "Exploitation\n(converged)", ha="center",
            fontsize=10, color=ORANGE, fontweight="bold")

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel(f"Success Rate (rolling {window}-episode mean)", fontsize=12)
    ax.set_title("Q-Learning: Agent Learns Over Time", fontsize=14,
                 fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, "learning_curve.png", ["ch11"])


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("APML UZH — Generating all slide images\n")

    print("── Ch01: Introduction ─────────────────────────────────")
    make_workflow_cycle()
    make_ml_paradigms()

    print("── Ch02: Data Cleaning ────────────────────────────────")
    make_pipeline_overview()
    make_missing_values_heatmap()
    make_outlier_boxplot()
    make_onehot_encoding()
    make_feature_scaling()
    make_train_test_split()
    make_data_leakage()

    print("── Ch03: Supervised Intro ─────────────────────────────")
    make_overfit_curves()
    make_bias_variance()
    make_cross_val_folds()

    print("── Ch04: Regression ───────────────────────────────────")
    make_linear_reg_fit()
    make_regularization_coefs()

    print("── Ch05: Classification ───────────────────────────────")
    make_sigmoid_curve()
    make_svm_margin()
    make_decision_boundaries()

    print("── Ch06: Metrics & Evaluation ─────────────────────────")
    make_confusion_matrix_viz()
    make_roc_curve()
    make_precision_recall()

    print("── Ch07: Unsupervised Intro ────────────────────────────")
    make_supervised_vs_unsupervised()

    print("── Ch08: Clustering ───────────────────────────────────")
    make_kmeans_steps()
    make_elbow_curve()
    make_dbscan_points()

    print("── Ch09: Dimensionality Reduction ─────────────────────")
    make_pca_directions()
    make_scree_plot()

    print("── Ch10: RL Intro ─────────────────────────────────────")
    make_rl_loop()
    make_discount_factor()

    print("── Ch11: RL Algorithms ────────────────────────────────")
    make_td_error_flow()
    make_learning_curve()

    print(f"\n✅  Done — images written to each chapter's 01-slides/ and slidev/public/")
