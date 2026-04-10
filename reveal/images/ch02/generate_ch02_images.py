"""
Generates all images used in ch02_slides.md.

Run from the project root:
    python reveal/images/ch02/generate_ch02_images.py

Or from this directory:
    cd reveal/images/ch02 && python generate_ch02_images.py

Output: PNG files saved to reveal/images/ch02/ (this directory).
Referenced in slides as: ../../reveal/images/ch02/<name>.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

OUT = "."

# ── Shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 13,
})


# ── 1. ML Pipeline — where does cleaning fit? ────────────────────────────────
def pipeline_overview():
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.axis("off")

    steps = ["Raw Data", "Clean &\nPrepare", "Feature\nEngineering", "Train\nModel", "Evaluate"]
    colors = ["#e0e0e0", "#ff7043", "#e0e0e0", "#e0e0e0", "#e0e0e0"]
    xs = [0.5, 2.5, 4.5, 6.5, 8.5]

    for i, (label, color, x) in enumerate(zip(steps, colors, xs)):
        box = mpatches.FancyBboxPatch(
            (x - 0.8, 0.2), 1.6, 0.6,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor="#333",
            linewidth=1.5,
        )
        ax.add_patch(box)
        ax.text(x, 0.5, label, ha="center", va="center", fontsize=11, fontweight="bold" if i == 1 else "normal")
        if i < len(steps) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.85, 0.5), xytext=(x + 0.85, 0.5),
                        arrowprops=dict(arrowstyle="->", color="#555", lw=2))

    ax.text(xs[1], 0.08, "← You are here", ha="center", color="#ff7043", fontsize=10, style="italic")
    fig.tight_layout()
    fig.savefig(f"{OUT}/pipeline_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("pipeline_overview.png")


# ── 2. Missing values heatmap ─────────────────────────────────────────────────
def missing_values_heatmap():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 1, (8, 6))
    missing_mask = rng.random((8, 6)) < 0.3
    data[missing_mask] = np.nan

    cols = ["age", "salary", "city", "edu", "score", "churn"]
    df = pd.DataFrame(data, columns=cols)

    fig, ax = plt.subplots(figsize=(7, 4))
    matrix = df.isnull().values.astype(float)
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha="right")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([f"Row {i}" for i in range(len(df))], fontsize=10)
    ax.set_title("Missing Values Heatmap  (red = NaN)", fontsize=13, pad=10)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            txt = "NaN" if matrix[i, j] == 1 else ""
            ax.text(j, i, txt, ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT}/missing_values_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("missing_values_heatmap.png")


# ── 3. Outlier — boxplot + IQR annotation ────────────────────────────────────
def outlier_boxplot():
    rng = np.random.default_rng(7)
    data = rng.normal(50, 10, 60).tolist() + [120, 130, -15]  # obvious outliers

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1, 2]})

    # Boxplot
    bp = axes[0].boxplot(data, patch_artist=True,
                         boxprops=dict(facecolor="#90caf9", color="#1565c0"),
                         medianprops=dict(color="#d32f2f", linewidth=2),
                         flierprops=dict(marker="o", color="#d32f2f", markersize=8))
    axes[0].set_title("Boxplot", fontsize=13)
    axes[0].set_ylabel("Value")
    axes[0].set_xticks([])

    # Histogram with IQR shading
    arr = np.array(data)
    q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    axes[1].hist(arr, bins=20, color="#90caf9", edgecolor="#1565c0", alpha=0.8)
    axes[1].axvline(lower, color="#d32f2f", linestyle="--", lw=2, label=f"Q1 − 1.5·IQR = {lower:.0f}")
    axes[1].axvline(upper, color="#d32f2f", linestyle="--", lw=2, label=f"Q3 + 1.5·IQR = {upper:.0f}")
    axes[1].axvspan(lower, upper, alpha=0.12, color="green", label="Normal range")
    axes[1].legend(fontsize=9)
    axes[1].set_title("Histogram with IQR boundaries", fontsize=13)
    axes[1].set_xlabel("Value")
    axes[1].set_ylabel("Count")

    fig.suptitle("Outlier Detection", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT}/outlier_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("outlier_boxplot.png")


# ── 4. One-Hot Encoding visual ────────────────────────────────────────────────
def onehot_visual():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    # Original column
    cities = ["NYC", "LA", "CHI", "NYC", "LA"]
    ax = axes[0]
    ax.axis("off")
    ax.set_title("Before: City (text)", fontsize=12, fontweight="bold", pad=8)
    for i, c in enumerate(["City"] + cities):
        color = "#bbdefb" if i == 0 else "white"
        rect = mpatches.FancyBboxPatch((0.2, 0.82 - i * 0.14), 0.6, 0.12,
                                       boxstyle="round,pad=0.01", facecolor=color,
                                       edgecolor="#90a4ae")
        ax.add_patch(rect)
        ax.text(0.5, 0.88 - i * 0.14, c, ha="center", va="center",
                fontweight="bold" if i == 0 else "normal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # After one-hot
    ax2 = axes[1]
    ax2.axis("off")
    ax2.set_title("After: One-Hot Encoding", fontsize=12, fontweight="bold", pad=8)
    headers = ["is_NYC", "is_LA", "is_CHI"]
    rows_data = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]]
    all_rows = [headers] + rows_data
    col_xs = [0.15, 0.5, 0.85]
    for i, row in enumerate(all_rows):
        for j, val in enumerate(row):
            bg = "#bbdefb" if i == 0 else ("#c8e6c9" if val == 1 else "#ffcdd2")
            rect = mpatches.FancyBboxPatch((col_xs[j] - 0.14, 0.82 - i * 0.14), 0.28, 0.12,
                                           boxstyle="round,pad=0.01", facecolor=bg,
                                           edgecolor="#90a4ae")
            ax2.add_patch(rect)
            ax2.text(col_xs[j], 0.88 - i * 0.14, str(val), ha="center", va="center",
                     fontweight="bold" if i == 0 else "normal")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(f"{OUT}/onehot_encoding.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("onehot_encoding.png")


# ── 5. Feature Scaling comparison ────────────────────────────────────────────
def feature_scaling():
    rng = np.random.default_rng(0)
    age = rng.uniform(20, 60, 200)
    salary = rng.uniform(30000, 120000, 200)

    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    scaler_std = StandardScaler()
    scaler_mm = MinMaxScaler()

    age_std = scaler_std.fit_transform(age.reshape(-1, 1)).ravel()
    age_mm = scaler_mm.fit_transform(age.reshape(-1, 1)).ravel()

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    titles = ["Original (age)", "StandardScaler (z-score)", "MinMaxScaler [0, 1]"]
    datasets = [age, age_std, age_mm]
    colors = ["#90caf9", "#a5d6a7", "#ffcc80"]

    for ax, d, title, color in zip(axes, datasets, titles, colors):
        ax.hist(d, bins=20, color=color, edgecolor="white")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(f"μ={d.mean():.1f}  σ={d.std():.1f}  range=[{d.min():.1f}, {d.max():.1f}]",
                      fontsize=8)
        ax.set_ylabel("Count")

    fig.suptitle("Effect of Feature Scaling on Distribution", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT}/feature_scaling.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("feature_scaling.png")


# ── 6. Train / Test Split diagram ────────────────────────────────────────────
def train_test_split_visual():
    fig, ax = plt.subplots(figsize=(9, 2.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Full dataset bar
    full = mpatches.FancyBboxPatch((0, 0.55), 10, 0.3, boxstyle="round,pad=0.05",
                                   facecolor="#e3f2fd", edgecolor="#1565c0", linewidth=2)
    ax.add_patch(full)
    ax.text(5, 0.7, "Full Dataset  (N samples)", ha="center", va="center",
            fontsize=12, fontweight="bold", color="#1565c0")

    # Arrow
    ax.annotate("", xy=(3, 0.3), xytext=(2, 0.5),
                arrowprops=dict(arrowstyle="->", color="#555"))
    ax.annotate("", xy=(7.5, 0.3), xytext=(8.5, 0.5),
                arrowprops=dict(arrowstyle="->", color="#555"))

    # Train bar
    train = mpatches.FancyBboxPatch((0, 0), 7.8, 0.28, boxstyle="round,pad=0.05",
                                    facecolor="#a5d6a7", edgecolor="#2e7d32", linewidth=2)
    ax.add_patch(train)
    ax.text(3.9, 0.14, "Train (80%)  →  model.fit(X_train, y_train)", ha="center", va="center",
            fontsize=11, color="#1b5e20")

    # Test bar
    test = mpatches.FancyBboxPatch((7.9, 0), 2.0, 0.28, boxstyle="round,pad=0.05",
                                   facecolor="#ffcdd2", edgecolor="#c62828", linewidth=2)
    ax.add_patch(test)
    ax.text(8.9, 0.14, "Test\n(20%)", ha="center", va="center", fontsize=11, color="#b71c1c")

    ax.text(10, -0.12, "Never seen\nduring training!", ha="center", va="top",
            fontsize=9, color="#b71c1c", style="italic")

    fig.tight_layout()
    fig.savefig(f"{OUT}/train_test_split.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("train_test_split.png")


# ── 7. Data Leakage diagram ───────────────────────────────────────────────────
def data_leakage():
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))

    for ax, title, bad in zip(axes,
                               ["Wrong (Data Leakage)", "Correct (Pipeline)"],
                               [True, False]):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(title, fontsize=13, fontweight="bold",
                     color="#c62828" if bad else "#2e7d32", pad=8)

        if bad:
            # Full data imputation
            full = mpatches.FancyBboxPatch((0.5, 0.7), 9, 0.22, boxstyle="round,pad=0.04",
                                           facecolor="#ffcdd2", edgecolor="#c62828", lw=2)
            ax.add_patch(full)
            ax.text(5, 0.81, "Impute on full dataset  (mean computed on ALL rows, incl. test!)",
                    ha="center", va="center", fontsize=9, color="#b71c1c")

            # Split after
            train = mpatches.FancyBboxPatch((0.5, 0.3), 6.3, 0.22, boxstyle="round,pad=0.04",
                                            facecolor="#ef9a9a", edgecolor="#c62828", lw=1.5)
            ax.add_patch(train)
            ax.text(3.65, 0.41, "Train", ha="center", va="center", fontsize=10)

            test = mpatches.FancyBboxPatch((7, 0.3), 2.5, 0.22, boxstyle="round,pad=0.04",
                                           facecolor="#ef9a9a", edgecolor="#c62828", lw=1.5)
            ax.add_patch(test)
            ax.text(8.25, 0.41, "Test", ha="center", va="center", fontsize=10)

            ax.text(5, 0.1, "Test info leaked into training!  Results too optimistic.",
                    ha="center", va="center", fontsize=9, color="#b71c1c", style="italic")

        else:
            # Split first
            train = mpatches.FancyBboxPatch((0.5, 0.7), 6.3, 0.22, boxstyle="round,pad=0.04",
                                            facecolor="#a5d6a7", edgecolor="#2e7d32", lw=2)
            ax.add_patch(train)
            ax.text(3.65, 0.81, "Train", ha="center", va="center", fontsize=10)

            test = mpatches.FancyBboxPatch((7, 0.7), 2.5, 0.22, boxstyle="round,pad=0.04",
                                           facecolor="#ffcdd2", edgecolor="#c62828", lw=2)
            ax.add_patch(test)
            ax.text(8.25, 0.81, "Test  (never touched)", ha="center", va="center",
                    fontsize=9, color="#b71c1c")

            # Fit on train, transform both
            pipe_train = mpatches.FancyBboxPatch((0.5, 0.3), 6.3, 0.22, boxstyle="round,pad=0.04",
                                                 facecolor="#c8e6c9", edgecolor="#2e7d32", lw=1.5)
            ax.add_patch(pipe_train)
            ax.text(3.65, 0.41, "pipe.fit_transform(X_train)", ha="center", va="center", fontsize=9)

            pipe_test = mpatches.FancyBboxPatch((7, 0.3), 2.5, 0.22, boxstyle="round,pad=0.04",
                                                facecolor="#ffcdd2", edgecolor="#c62828", lw=1.5)
            ax.add_patch(pipe_test)
            ax.text(8.25, 0.41, "pipe.transform\n(X_test)", ha="center", va="center", fontsize=8)

            ax.text(5, 0.1, "No leakage — imputation uses only training statistics.",
                    ha="center", va="center", fontsize=9, color="#2e7d32", style="italic")

    fig.tight_layout()
    fig.savefig(f"{OUT}/data_leakage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("data_leakage.png")


if __name__ == "__main__":
    pipeline_overview()
    missing_values_heatmap()
    outlier_boxplot()
    onehot_visual()
    feature_scaling()
    train_test_split_visual()
    data_leakage()
    print("\nAll images saved to:", OUT)
