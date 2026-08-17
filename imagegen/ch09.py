"""Ch09 — Dimensionality reduction: images and GIF animations.

Run:  .venv/bin/python -c "from imagegen import ch09; ch09.generate()"
"""
from imagegen.common import *
from sklearn.datasets import load_digits, load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


# ── 1. Curse of dimensionality GIF: distances concentrate ──────────────────
def make_curse_gif():
    rng = np.random.default_rng(0)
    dims = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000]
    n = 300
    ratios, dists = [], []
    for d in dims:
        X = rng.uniform(0, 1, size=(n, d))
        D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
        tri = D[np.triu_indices(n, 1)]
        dists.append(tri / tri.mean())        # normalised so histograms are comparable
        np.fill_diagonal(D, np.inf)
        nearest = D.min(1)
        np.fill_diagonal(D, -np.inf)
        farthest = D.max(1)
        ratios.append((nearest / farthest).mean())   # per point: nearest ÷ farthest neighbour

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    def update(i):
        for ax in axes:
            ax.clear()
        d = dims[i]
        axes[0].hist(dists[i], bins=40, color=TEAL, alpha=0.85, range=(0, 2.5))
        axes[0].set_xlim(0, 2.5)
        axes[0].set_xlabel("pairwise distance  (÷ mean distance)")
        axes[0].set_ylabel("count")
        axes[0].set_title(f"{d} dim{'s' if d > 1 else ''}: distances between 300 random points")
        axes[1].plot(dims[: i + 1], ratios[: i + 1], "o-", color=RED, lw=2.5, ms=7)
        axes[1].set_xscale("log")
        axes[1].set_xlim(0.8, 1500); axes[1].set_ylim(0, 1)
        axes[1].set_xlabel("number of dimensions (log scale)")
        axes[1].set_ylabel("nearest ÷ farthest neighbour (mean over points)")
        axes[1].set_title("nearest ÷ farthest neighbour → 1")
        axes[1].grid(True, alpha=0.3)
        axes[1].annotate(f"{ratios[i]:.2f}", (d, ratios[i]), textcoords="offset points",
                         xytext=(8, 6), color=RED, fontweight="bold")
        fig.suptitle("Curse of dimensionality: in high-D 'everything is equally far away'",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()

    save_gif(fig, update, len(dims), "curse_dimensionality.gif", ["ch09"], fps=1, hold_last=3)


# ── 2. PCA rotating axis GIF: variance captured by a direction ─────────────
def make_pca_rotation_gif():
    rng = np.random.default_rng(3)
    X = rng.multivariate_normal([0, 0], [[3.0, 1.7], [1.7, 1.8]], size=200)
    X -= X.mean(0)
    total_var = X.var(0).sum()
    pca = PCA(2).fit(X)
    pc1_angle = np.degrees(np.arctan2(pca.components_[0, 1], pca.components_[0, 0])) % 180
    angles = list(np.linspace(0, 180, 24, endpoint=False))
    # make sure the true PC1 angle is a frame, and end there
    angles = sorted(set(round(a, 1) for a in angles) | {round(pc1_angle, 1)})
    order = angles[angles.index(round(pc1_angle, 1)) + 1:] + angles[: angles.index(round(pc1_angle, 1)) + 1]
    var_frac = []
    for a in order:
        u = np.array([np.cos(np.radians(a)), np.sin(np.radians(a))])
        var_frac.append((X @ u).var() / total_var)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})
    L = 5.5

    def update(i):
        for ax in axes:
            ax.clear()
        a = order[i]
        u = np.array([np.cos(np.radians(a)), np.sin(np.radians(a))])
        proj = (X @ u)[:, None] * u[None, :]
        is_last = i == len(order) - 1
        col = TEAL_DARK if is_last else BLUE
        axes[0].scatter(X[:, 0], X[:, 1], color=MUTED, s=22, alpha=0.5)
        axes[0].plot([-L * u[0], L * u[0]], [-L * u[1], L * u[1]], color=col, lw=3)
        for p, q in zip(X[::4], proj[::4]):
            axes[0].plot([p[0], q[0]], [p[1], q[1]], color=col, lw=0.6, alpha=0.4)
        axes[0].scatter(proj[:, 0], proj[:, 1], color=col, s=14, alpha=0.9, zorder=5)
        axes[0].set_xlim(-L, L); axes[0].set_ylim(-L, L); axes[0].set_aspect("equal")
        axes[0].set_xticks([]); axes[0].set_yticks([])
        axes[0].set_title(("PC1 = the direction that keeps the MOST variance"
                           if is_last else f"Project the data onto a line at {a:.0f}°"),
                          color=col)
        axes[1].plot(sorted(order), [var_frac[order.index(a)] for a in sorted(order)],
                     "-", color=BORDER, lw=1.5, zorder=1)
        axes[1].scatter(order[: i + 1], var_frac[: i + 1], color=col, s=28, zorder=3)
        axes[1].scatter([a], [var_frac[i]], color=col, s=90, zorder=4)
        axes[1].set_xlim(0, 180); axes[1].set_ylim(0, 1)
        axes[1].set_xlabel("angle of the line (°)")
        axes[1].set_ylabel("share of total variance kept on that line")
        axes[1].set_title(f"variance kept: {var_frac[i]:.0%}", color=col)
        axes[1].grid(True, alpha=0.3)
        if is_last:
            axes[1].axhline(pca.explained_variance_ratio_[0], color=RED, ls="--", lw=1.2)
            axes[1].text(2, pca.explained_variance_ratio_[0] + 0.02,
                         f"maximum = PC1 ({pca.explained_variance_ratio_[0]:.0%})", color=RED)
        fig.suptitle("PCA: rotate the axis, keep the direction with the largest spread",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()

    save_gif(fig, update, len(order), "pca_rotation.gif", ["ch09"], fps=3, dpi=80, hold_last=6)


# ── 3. PCA loadings: a PC is a weighted mix of ALL features ─────────────────
def make_pca_loadings():
    data = load_breast_cancer()
    Xs = StandardScaler().fit_transform(data.data)
    pca = PCA(2).fit(Xs)
    names = [n.replace("mean ", "").replace(" error", " err") for n in data.feature_names]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=False)
    for ax, k in zip(axes, [0, 1]):
        w = pca.components_[k]
        cols = [TEAL if v >= 0 else RED for v in w]
        ax.bar(range(len(w)), w, color=cols)
        ax.axhline(0, color=DARK, lw=0.8)
        ax.set_xticks(range(len(w)))
        ax.set_xticklabels(names, rotation=90, fontsize=7)
        ax.set_ylabel("weight (loading)")
        ax.set_title(f"PC{k + 1} = weighted sum of ALL 30 features "
                     f"({pca.explained_variance_ratio_[k]:.0%} of variance)")
    fig.suptitle("Breast cancer, scaled: PCA builds NEW axes — it does not pick features",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "pca_loadings.png", ["ch09"])


# ── 4. Two regimes: 2 PCs for a picture vs. ≥95 % for preprocessing ─────────
def make_pca_two_regimes():
    d = load_digits()
    Xs = StandardScaler().fit_transform(d.data)
    pca = PCA().fit(Xs)
    cum = pca.explained_variance_ratio_.cumsum()
    X2 = Xs @ pca.components_[:2].T
    n95 = int(np.argmax(cum >= 0.95) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sc = axes[0].scatter(X2[:, 0], X2[:, 1], c=d.target, cmap="tab10", s=10, alpha=0.6)
    axes[0].set_title(f"Regime 1 — VISUALIZE: 2 PCs, only {cum[1]:.0%} of variance,\n"
                      "yet the digit groups are visible (colours = labels, added afterwards)")
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    axes[1].plot(range(1, 65), cum, "o-", color=TEAL, ms=4)
    axes[1].axhline(0.95, color=RED, ls="--"); axes[1].axvline(n95, color=RED, ls="--")
    axes[1].axvline(2, color=BLUE, ls=":", lw=2)
    axes[1].text(3, 0.12, f"2 PCs → {cum[1]:.0%}", color=BLUE, fontweight="bold")
    axes[1].text(n95 + 1.5, 0.5, f"{n95} PCs → 95%", color=RED, fontweight="bold")
    axes[1].set_xlabel("number of components"); axes[1].set_ylabel("cumulative explained variance")
    axes[1].set_title(f"Regime 2 — PREPROCESS: keep enough PCs for ≥95%\n(here {n95} of 64)")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("Digits (64 pixels): the SAME PCA serves two different jobs",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "pca_two_regimes.png", ["ch09"])


# ── 5. PCA vs t-SNE on digits ──────────────────────────────────────────────
def make_pca_vs_tsne():
    d = load_digits()
    Xs = StandardScaler().fit_transform(d.data)
    Xp = PCA(2).fit_transform(Xs)
    Xt = TSNE(2, perplexity=30, random_state=42).fit_transform(Xs)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax, X2, title in [(axes[0], Xp, "PCA — linear projection, axes = variance, distances meaningful"),
                          (axes[1], Xt, "t-SNE — non-linear, keeps NEIGHBOURS; distances between blobs are NOT meaningful")]:
        ax.scatter(X2[:, 0], X2[:, 1], c=d.target, cmap="tab10", s=10, alpha=0.7)
        for k in range(10):
            m = d.target == k
            ax.annotate(str(k), (np.median(X2[m, 0]), np.median(X2[m, 1])), fontsize=12,
                        fontweight="bold", ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec=BORDER))
        ax.set_title(title, fontsize=10.5)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Digits: same 64-D data, two 2-D pictures (colours = true labels, added afterwards)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "pca_vs_tsne.png", ["ch09"])


def generate():
    print("Ch09 images:")
    make_curse_gif()
    make_pca_rotation_gif()
    make_pca_loadings()
    make_pca_two_regimes()
    make_pca_vs_tsne()


if __name__ == "__main__":
    generate()
