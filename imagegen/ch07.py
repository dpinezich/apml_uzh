"""Ch07 — Intro to unsupervised learning: images and GIF animations.

Run:  .venv/bin/python -c "from imagegen import ch07; ch07.generate()"
"""
from imagegen.common import *
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

CLUSTER_COLORS = [RED, BLUE, GREEN, ORANGE, PURPLE]


# ── 1. "Find the groups" reveal GIF ────────────────────────────────────────
def make_find_groups_gif():
    sets = [
        ("Blobs", *make_blobs(n_samples=300, centers=4, cluster_std=0.7, random_state=42)),
        ("Moons", *make_moons(n_samples=300, noise=0.07, random_state=42)),
        ("Circles", *make_circles(n_samples=300, noise=0.05, factor=0.4, random_state=42)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    # frames: 0 = all grey ("what the algorithm sees"), 1..3 reveal one panel each
    def update(f):
        for ax, (title, X, y) in zip(axes, sets):
            ax.clear()
            idx = [s[0] for s in sets].index(title)
            revealed = f > idx
            cols = [CLUSTER_COLORS[l] for l in y] if revealed else MUTED
            ax.scatter(X[:, 0], X[:, 1], c=cols, s=24, alpha=0.7, edgecolors="white", lw=0.3)
            ax.set_title(f"{title} — {'the groups (labels we did NOT have)' if revealed else 'how many groups do you see?'}",
                         fontsize=10.5, color=TEAL_DARK if revealed else DARK)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle("Unsupervised learning: find the structure WITHOUT being told the answer"
                     if f == 0 else "Reveal: your eyes clustered without labels — can an algorithm?",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
    save_gif(fig, update, 4, "find_groups.gif", ["ch07"], fps=1, hold_last=3)


# ── 2. Three tasks concept card ────────────────────────────────────────────
def make_unsupervised_tasks():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    # clustering
    X, y = make_blobs(n_samples=240, centers=3, cluster_std=0.8, random_state=1)
    lab = KMeans(3, n_init=10, random_state=0).fit_predict(X)
    axes[0].scatter(X[:, 0], X[:, 1], c=[CLUSTER_COLORS[l] for l in lab], s=26, alpha=0.7)
    axes[0].set_title("1 · Clustering", color=TEAL_DARK)
    axes[0].set_xlabel("'Which samples belong together?' → group ids", fontsize=10)
    # dimensionality reduction: 2-D correlated cloud projected onto PC1
    rng = np.random.default_rng(2)
    Z = rng.multivariate_normal([0, 0], [[3, 1.7], [1.7, 1.8]], 200)
    Z -= Z.mean(0)
    pc = PCA(1).fit(Z)
    u = pc.components_[0]
    proj = (Z @ u)[:, None] * u
    axes[1].scatter(Z[:, 0], Z[:, 1], color=MUTED, s=20, alpha=0.5, label="original (2 features)")
    axes[1].plot([-5 * u[0], 5 * u[0]], [-5 * u[1], 5 * u[1]], color=TEAL_DARK, lw=3)
    axes[1].scatter(proj[:, 0], proj[:, 1], color=TEAL, s=14, zorder=5, label="compressed (1 feature)")
    axes[1].set_aspect("equal"); axes[1].legend(fontsize=8, loc="lower right")
    axes[1].set_title("2 · Dimensionality reduction", color=TEAL_DARK)
    axes[1].set_xlabel("'Fewer features, most of the information' → new coordinates", fontsize=10)
    # anomaly detection
    Xn = rng.normal(0, 1, size=(200, 2))
    out = np.array([[4.2, 3.5], [-4, 3], [3.8, -3.8], [-3.5, -4.2], [0.3, 4.6]])
    Xa = np.vstack([Xn, out])
    iso = IsolationForest(contamination=0.03, random_state=0).fit(Xa)
    flag = iso.predict(Xa) == -1
    axes[2].scatter(Xa[~flag, 0], Xa[~flag, 1], color=MUTED, s=22, alpha=0.6, label="normal")
    axes[2].scatter(Xa[flag, 0], Xa[flag, 1], color=RED, s=70, marker="X", label="flagged anomaly")
    axes[2].legend(fontsize=8, loc="lower left")
    axes[2].set_title("3 · Anomaly detection (density)", color=TEAL_DARK)
    axes[2].set_xlabel("'Which samples are unusual?' → outlier flags", fontsize=10)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Three questions you can ask data that has NO labels", fontsize=14,
                 fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "unsupervised_tasks.png", ["ch07"])


# ── 3. Evaluation without labels: two valid groupings ──────────────────────
def make_two_valid_groupings():
    rng = np.random.default_rng(5)
    # 4 blobs arranged in a 2x2 grid; k=2 splits left/right OR top/bottom, k=4 gives 4
    X, _ = make_blobs(n_samples=320, centers=[[0, 0], [4, 0], [0, 4], [4, 4]],
                      cluster_std=0.6, random_state=5)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    labs = [(X[:, 0] > 2).astype(int), (X[:, 1] > 2).astype(int),
            KMeans(4, n_init=10, random_state=0).fit_predict(X)]
    titles = ["Grouping A: k=2 (left / right)", "Grouping B: k=2 (top / bottom)", "Grouping C: k=4"]
    for ax, l, t in zip(axes, labs, titles):
        ax.scatter(X[:, 0], X[:, 1], c=[CLUSTER_COLORS[i] for i in l], s=24, alpha=0.7)
        ax.set_title(t); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Which one is 'correct'? — All three are valid. Which one is USEFUL depends on your question.",
                 fontsize=12.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "two_valid_groupings.png", ["ch07"])


def generate():
    print("Ch07 images:")
    make_find_groups_gif()
    make_unsupervised_tasks()
    make_two_valid_groupings()


if __name__ == "__main__":
    generate()
