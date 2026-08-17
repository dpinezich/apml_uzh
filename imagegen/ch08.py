"""Ch08 — Clustering: images and GIF animations.

Run:  .venv/bin/python -c "from imagegen import ch08; ch08.generate()"
"""
from imagegen.common import *
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

CLUSTER_COLORS = [RED, BLUE, GREEN, ORANGE, PURPLE, TEAL_DARK]


def _blobs4():
    X, _ = make_blobs(n_samples=300, centers=[[0, 0], [5, 0], [0, 5], [5, 5]],
                      cluster_std=0.7, random_state=3)
    return X


def _lloyd_history(X, k, seed, max_iter=50):
    """Plain Lloyd iterations from a random-point init; returns [(centroids, labels)]."""
    rng = np.random.default_rng(seed)
    c = X[rng.choice(len(X), k, replace=False)].copy()
    hist = []
    for _ in range(max_iter):
        lab = np.argmin(((X[:, None] - c[None]) ** 2).sum(2), 1)
        hist.append((c.copy(), lab.copy()))
        nc = np.array([X[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
        if np.allclose(nc, c):
            break
        c = nc
    return hist


# ── 1. K-Means iterations GIF (assign → update) ────────────────────────────
def make_kmeans_iterations_gif():
    X = _blobs4()
    hist = _lloyd_history(X, 4, seed=9)          # 6 update steps
    # frames: (step, phase) — phase 0 = assign (points coloured, old centroids),
    # phase 1 = update (centroids moved, arrows)
    frames = []
    for i, (c, lab) in enumerate(hist):
        frames.append(("assign", i))
        if i < len(hist) - 1:
            frames.append(("update", i))
    frames.append(("done", len(hist) - 1))

    fig, ax = plt.subplots(figsize=(6.5, 6))

    def update(f):
        ax.clear()
        phase, i = frames[f]
        c, lab = hist[i]
        ax.scatter(X[:, 0], X[:, 1], c=[CLUSTER_COLORS[l] for l in lab],
                   s=32, alpha=0.65, edgecolors="white", lw=0.4)
        if phase == "update":
            c_new = hist[i + 1][0]
            for j in range(4):
                ax.annotate("", xy=c_new[j], xytext=c[j],
                            arrowprops=dict(arrowstyle="->", color=DARK, lw=2))
            ax.scatter(c[:, 0], c[:, 1], marker="X", s=180, c=CLUSTER_COLORS[:4],
                       alpha=0.35, edgecolors=DARK, lw=1)
            ax.scatter(c_new[:, 0], c_new[:, 1], marker="X", s=260, c=CLUSTER_COLORS[:4],
                       edgecolors=DARK, lw=1.5, zorder=6)
            title = f"Iteration {i + 1} — UPDATE: move centroids to cluster means"
        else:
            ax.scatter(c[:, 0], c[:, 1], marker="X", s=260, c=CLUSTER_COLORS[:4],
                       edgecolors=DARK, lw=1.5, zorder=6)
            if phase == "done":
                title = f"Converged after {len(hist) - 1} iterations — centroids stop moving"
            elif i == 0:
                title = "Iteration 1 — INIT: pick 4 random points, ASSIGN each point to nearest"
            else:
                title = f"Iteration {i + 1} — ASSIGN: each point → nearest centroid"
        inertia = ((X - c[lab]) ** 2).sum()
        ax.set_title(title, fontsize=11)
        ax.text(0.02, 0.97, f"inertia = {inertia:,.0f}", transform=ax.transAxes,
                va="top", fontsize=11, fontweight="bold", color=DARK,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=BORDER))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-2.5, 7.5); ax.set_ylim(-2.5, 7.5)

    save_gif(fig, update, len(frames), "kmeans_iterations.gif", ["ch08"],
             fps=1, hold_last=3)


# ── 2. K-Means++ vs random init ─────────────────────────────────────────────
def make_kmeans_init():
    X = _blobs4()
    hist_bad = _lloyd_history(X, 4, seed=1)      # ends in a local optimum
    c_bad, lab_bad = hist_bad[-1]
    km = KMeans(n_clusters=4, init="k-means++", n_init=10, random_state=42).fit(X)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, c, lab, title, col in [
        (axes[0], c_bad, lab_bad,
         "Random init, single run\n→ stuck in a local optimum", RED),
        (axes[1], km.cluster_centers_, km.labels_,
         "k-means++ init, n_init=10 (sklearn default idea)\n→ global optimum", TEAL_DARK),
    ]:
        ax.scatter(X[:, 0], X[:, 1], c=[CLUSTER_COLORS[l] for l in lab],
                   s=30, alpha=0.6, edgecolors="white", lw=0.4)
        ax.scatter(c[:, 0], c[:, 1], marker="X", s=240, c=CLUSTER_COLORS[:4],
                   edgecolors=DARK, lw=1.5, zorder=6)
        inertia = ((X - c[lab]) ** 2).sum()
        ax.set_title(title, color=col)
        ax.text(0.02, 0.97, f"inertia = {inertia:,.0f}", transform=ax.transAxes,
                va="top", fontsize=12, fontweight="bold", color=col,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=BORDER))
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Same data, same k=4 — the initialization decides", fontsize=14,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "kmeans_init.png", ["ch08"])


# ── 3. Elbow + silhouette side by side ─────────────────────────────────────
def make_elbow_silhouette():
    X, _ = make_blobs(n_samples=400, centers=4, cluster_std=0.8, random_state=42)
    ks = list(range(1, 11))
    inertias, sils = [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(X, km.labels_) if k > 1 else np.nan)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].plot(ks, inertias, "o-", color=TEAL, lw=2.5, ms=7)
    axes[0].axvline(4, color=RED, ls="--", lw=1.5, label="k = 4")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia (within-cluster SS)")
    axes[0].set_title("Elbow method — where does the drop flatten?")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(ks[1:], sils[1:], "s-", color=GREEN, lw=2.5, ms=7)
    axes[1].axvline(4, color=RED, ls="--", lw=1.5, label="k = 4")
    axes[1].set_xlabel("k"); axes[1].set_ylabel("Mean silhouette score")
    axes[1].set_title("Silhouette — higher is better (max at k = 4)")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    fig.suptitle("Choosing k on 4 well-separated blobs: both methods agree",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "elbow_silhouette.png", ["ch08"])


# ── 4. Clustering structureless data still "works" ─────────────────────────
def make_kmeans_uniform():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 10, size=(400, 2))
    ks = list(range(1, 11))
    inertias = [KMeans(n_clusters=k, n_init=10, random_state=0).fit(X).inertia_ for k in ks]
    km3 = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    axes[0].scatter(X[:, 0], X[:, 1], color=MUTED, s=22, alpha=0.6)
    axes[0].set_title("Uniform random noise — no groups at all")
    axes[1].scatter(X[:, 0], X[:, 1], c=[CLUSTER_COLORS[l] for l in km3.labels_], s=22, alpha=0.7)
    axes[1].scatter(km3.cluster_centers_[:, 0], km3.cluster_centers_[:, 1], marker="X",
                    s=220, c=CLUSTER_COLORS[:3], edgecolors=DARK, lw=1.5, zorder=6)
    axes[1].set_title("K-Means(k=3) happily returns 3 'clusters'", color=RED)
    axes[2].plot(ks, inertias, "o-", color=TEAL, lw=2.5, ms=7)
    axes[2].set_xlabel("k"); axes[2].set_ylabel("Inertia")
    axes[2].set_title("…and the elbow plot looks 'normal' (no sharp bend)")
    axes[2].grid(True, alpha=0.3)
    for ax in axes[:2]:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("K-Means ALWAYS finds k clusters — even when there are none",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "kmeans_uniform.png", ["ch08"])


# ── 5. Scaling matters for distance-based clustering ───────────────────────
def make_scaling_kmeans():
    rng = np.random.default_rng(1)
    # two groups separated in feature 2 (small scale), spread out in feature 1 (large scale)
    n = 150
    f1 = np.clip(np.concatenate([rng.normal(50, 22, n), rng.normal(50, 22, n)]), 0, 100)
    f2 = np.concatenate([rng.normal(0.2, 0.06, n), rng.normal(0.8, 0.06, n)])  # e.g. ratio 0-1
    X = np.column_stack([f1, f2])
    truth = np.array([0] * n + [1] * n)
    lab_raw = KMeans(2, n_init=10, random_state=0).fit_predict(X)
    lab_sc = KMeans(2, n_init=10, random_state=0).fit_predict(StandardScaler().fit_transform(X))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, lab, title, col in [
        (axes[0], truth, "Data: two real groups (feature 2 separates them)", DARK),
        (axes[1], lab_raw, "K-Means on RAW data\n→ splits along the big-number feature", RED),
        (axes[2], lab_sc, "K-Means after StandardScaler\n→ finds the real groups", TEAL_DARK),
    ]:
        ax.scatter(X[:, 0], X[:, 1], c=[CLUSTER_COLORS[l] for l in lab], s=26, alpha=0.7,
                   edgecolors="white", lw=0.3)
        ax.set_title(title, color=col, fontsize=11)
        ax.set_xlabel("feature 1  (range 0–100)"); ax.set_ylabel("feature 2  (range 0–1)")
    fig.suptitle("Distances are dominated by the feature with the largest numbers → scale first",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "scaling_kmeans.png", ["ch08"])


# ── 6. K-Means vs DBSCAN on moons ──────────────────────────────────────────
def make_kmeans_vs_dbscan():
    X, _ = make_moons(n_samples=300, noise=0.07, random_state=42)
    Xs = StandardScaler().fit_transform(X)
    lab_km = KMeans(2, n_init=10, random_state=42).fit_predict(Xs)
    lab_db = DBSCAN(eps=0.25, min_samples=5).fit_predict(Xs)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].scatter(X[:, 0], X[:, 1], c=[CLUSTER_COLORS[l] for l in lab_km], s=30, alpha=0.7)
    axes[0].set_title("K-Means (k=2)\nassumes round blobs → cuts the moons", color=RED)
    cols = [MUTED if l == -1 else CLUSTER_COLORS[l] for l in lab_db]
    axes[1].scatter(X[:, 0], X[:, 1], c=cols, s=30, alpha=0.7)
    n_noise = int((lab_db == -1).sum())
    axes[1].set_title(f"DBSCAN (eps=0.25, scaled)\nfollows density; {n_noise} noise points (grey)",
                      color=TEAL_DARK)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    save(fig, "kmeans_vs_dbscan.png", ["ch08"])


# ── 7. Dendrogram (bonus / appendix) ───────────────────────────────────────
def make_dendrogram():
    X, _ = make_blobs(n_samples=40, centers=3, cluster_std=0.8, random_state=4)
    Z = linkage(X, method="ward")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.6]})
    from scipy.cluster.hierarchy import fcluster
    cut = (Z[-3, 2] + Z[-2, 2]) / 2       # between the 3rd- and 2nd-last merges → 3 clusters
    lab = fcluster(Z, t=cut, criterion="distance") - 1
    axes[0].scatter(X[:, 0], X[:, 1], c=[CLUSTER_COLORS[l] for l in lab], s=45, edgecolors="white")
    for i, (x, y) in enumerate(X):
        axes[0].annotate(str(i), (x, y), fontsize=6, ha="center", va="center", color="white")
    axes[0].set_title("40 points, coloured by cutting the tree into 3")
    axes[0].set_xticks([]); axes[0].set_yticks([])
    dendrogram(Z, ax=axes[1], color_threshold=cut, leaf_font_size=7,
               above_threshold_color=MUTED)
    axes[1].axhline(cut, color=RED, ls="--", lw=1.5)
    axes[1].text(2, cut + 0.3, "cut here → 3 clusters", color=RED, fontsize=10, fontweight="bold")
    axes[1].set_title("Dendrogram (Ward linkage): y = distance at which clusters merge")
    axes[1].set_ylabel("merge distance")
    fig.tight_layout()
    save(fig, "dendrogram.png", ["ch08"])


def generate():
    print("Ch08 images:")
    make_kmeans_iterations_gif()
    make_kmeans_init()
    make_elbow_silhouette()
    make_kmeans_uniform()
    make_scaling_kmeans()
    make_kmeans_vs_dbscan()
    make_dendrogram()


if __name__ == "__main__":
    generate()
