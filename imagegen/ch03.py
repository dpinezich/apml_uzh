"""Ch03 — Introduction to Supervised Learning (KNN red thread).

Images:
  knn_vote.png            concept card: a query point, its k nearest neighbours, majority vote
  knn_boundary_k.gif/.png decision boundary on make_moons as k grows 1 → 25 (train/test acc in title)
  knn_k_sweep.png         train vs test accuracy vs k (same data as the GIF)
  knn_scaling.png         why KNN needs feature scaling (neighbours change with the scale)
  hyperparam_vs_param.png concept card: hyperparameter (chosen by you) vs parameter (learned)
"""
from imagegen.common import *
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CH = ["ch03"]
K_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 40, 50, 75, 100]


def _moons():
    X, y = make_moons(n_samples=300, noise=0.3, random_state=42)
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


def knn_vote():
    rng = np.random.default_rng(3)
    A = rng.normal([2.0, 2.0], 0.9, (18, 2))
    B = rng.normal([4.2, 3.6], 0.9, (18, 2))
    q = np.array([3.1, 3.0])
    pts = np.vstack([A, B]); lab = np.array([0] * 18 + [1] * 18)
    d = np.linalg.norm(pts - q, axis=1)
    idx = np.argsort(d)[:5]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.scatter(A[:, 0], A[:, 1], c=BLUE, s=55, edgecolors="white", label="class A (blue)")
    ax.scatter(B[:, 0], B[:, 1], c=ORANGE, s=55, edgecolors="white", label="class B (orange)")
    ax.scatter(*q, marker="*", s=420, c=DARK, zorder=5, label="new point (?)")
    circ = plt.Circle(q, d[idx[-1]] + 0.03, fill=False, ls="--", lw=1.6, color=TEAL_DARK)
    ax.add_patch(circ)
    for i in idx:
        ax.plot([q[0], pts[i, 0]], [q[1], pts[i, 1]], color=TEAL_DARK, lw=1.2, alpha=0.9)
    votes_a = int((lab[idx] == 0).sum()); votes_b = 5 - votes_a
    winner = "A (blue)" if votes_a > votes_b else "B (orange)"
    ax.text(0.02, 0.97, f"k = 5 nearest neighbours\nvotes: {votes_a} × A, {votes_b} × B\n→ predict class {winner}",
            transform=ax.transAxes, va="top", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=TEAL_DARK))
    ax.set_title("K-Nearest Neighbours: look at the k closest points, let them vote")
    ax.set_xlabel("feature 1"); ax.set_ylabel("feature 2")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9, frameon=True)
    ax.set_aspect("equal")
    fig.tight_layout()
    save(fig, "knn_vote.png", CH)


def knn_boundary_gif():
    X_tr, X_te, y_tr, y_te = _moons()
    scaler = StandardScaler().fit(X_tr)
    Xs_tr, Xs_te = scaler.transform(X_tr), scaler.transform(X_te)
    h = 0.03
    x_min, x_max = Xs_tr[:, 0].min() - 0.6, Xs_tr[:, 0].max() + 0.6
    y_min, y_max = Xs_tr[:, 1].min() - 0.6, Xs_tr[:, 1].max() + 0.6
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    frames = []
    for k in K_VALUES:
        knn = KNeighborsClassifier(n_neighbors=k).fit(Xs_tr, y_tr)
        frames.append((k, knn.predict(grid).reshape(xx.shape),
                       knn.score(Xs_tr, y_tr), knn.score(Xs_te, y_te)))

    cmap_bg = ListedColormap(["#cfe6fb", "#fde3c8"])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.6),
                                  gridspec_kw={"width_ratios": [1.35, 1]})
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    ax.set_xlabel("feature 1 (scaled)"); ax.set_ylabel("feature 2 (scaled)")
    ax.scatter(Xs_tr[:, 0], Xs_tr[:, 1], c=[BLUE if v == 0 else ORANGE for v in y_tr],
               s=28, edgecolors="white", linewidths=0.4, zorder=3, label="train")
    ax.scatter(Xs_te[:, 0], Xs_te[:, 1], c=[BLUE if v == 0 else ORANGE for v in y_te],
               marker="^", s=34, edgecolors=DARK, linewidths=0.5, zorder=4, label="test")
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    mesh = [None]
    ttl = ax.set_title("")
    tag = ax.text(0.02, 0.03, "", transform=ax.transAxes, fontsize=10, color="white",
                  bbox=dict(boxstyle="round,pad=0.3", fc=GREEN, ec="none"))

    ks = [f[0] for f in frames]
    tr = [f[2] for f in frames]; te = [f[3] for f in frames]
    ax2.plot(ks, tr, "o-", color=BLUE, lw=2, ms=4, label="train accuracy")
    ax2.plot(ks, te, "s-", color=RED, lw=2, ms=4, label="test accuracy")
    ax2.set_xscale("log"); ax2.set_xticks([1, 3, 10, 30, 100]); ax2.set_xticklabels([1, 3, 10, 30, 100])
    ax2.set_xlabel("k (log scale)"); ax2.set_ylabel("accuracy")
    ax2.set_ylim(0.75, 1.02); ax2.grid(True, alpha=0.5)
    ax2.legend(loc="lower left", fontsize=9)
    ax2.set_title("Train vs test accuracy")
    m_tr, = ax2.plot([], [], "o", color=BLUE, ms=11, zorder=5)
    m_te, = ax2.plot([], [], "s", color=RED, ms=11, zorder=5)
    vl = ax2.axvline(1, color=DARK, lw=1, ls="--", alpha=0.6)
    fig.tight_layout()

    def update(i):
        k, Z, a_tr, a_te = frames[i]
        if mesh[0] is not None:
            mesh[0].remove()
        mesh[0] = ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], cmap=cmap_bg, zorder=1)
        ttl.set_text(f"KNN decision boundary — k = {k}")
        if k <= 3:
            txt, col = "jagged → memorises noise (overfitting)", RED
        elif k >= 50:
            txt, col = "too smooth → misses the moons (underfitting)", ORANGE
        else:
            txt, col = "smooth but follows the shape", GREEN
        tag.set_text(f"train {a_tr:.2f} | test {a_te:.2f}   {txt}")
        tag.get_bbox_patch().set_facecolor(col)
        m_tr.set_data([k], [a_tr]); m_te.set_data([k], [a_te]); vl.set_xdata([k, k])

    save_gif(fig, update, len(frames), "knn_boundary_k.gif", CH, fps=2, hold_last=4)

    # static k-sweep figure (same numbers)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(ks, tr, "o-", color=BLUE, lw=2, label="train accuracy")
    ax.plot(ks, te, "s-", color=RED, lw=2, label="test accuracy")
    ax.set_xscale("log"); ax.set_xticks([1, 3, 10, 30, 100]); ax.set_xticklabels([1, 3, 10, 30, 100])
    best = ks[int(np.argmax(te))]
    ax.axvline(best, color=GREEN, ls="--", lw=1.5, label=f"best test acc at k = {best}")
    ax.annotate("k = 1: train 100 %\nbut test much lower\n= overfitting", xy=(1, tr[0]),
                xytext=(1.6, 0.86), fontsize=9, color=MUTED, arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.annotate("large k: both drop\n= underfitting", xy=(100, te[-1]), xytext=(20, 0.79),
                fontsize=9, color=MUTED, arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.set_xlabel("k = number of neighbours (log scale)"); ax.set_ylabel("accuracy")
    ax.set_ylim(0.75, 1.02); ax.grid(True, alpha=0.5)
    ax.set_title("KNN on make_moons: sweeping the hyperparameter k")
    ax.legend(loc="lower left", fontsize=9)
    save(fig, "knn_k_sweep.png", CH)


def knn_scaling():
    rng = np.random.default_rng(4)
    n = 40
    age = np.r_[rng.uniform(22, 38, n // 2), rng.uniform(42, 60, n // 2)]
    inc = rng.uniform(40000, 140000, n)          # CHF, range 100 000
    lab = (age > 40).astype(int)                 # class depends on age only
    q = np.array([30.0, 95000.0])                # clearly "younger"
    pts_raw = np.c_[age, inc]
    sc = StandardScaler().fit(pts_raw)
    panels = [
        (pts_raw, q, "Raw features: distance ≈ income difference only", "age (years)", "income (CHF)"),
        (sc.transform(pts_raw), sc.transform([q])[0], "After StandardScaler: both features count", "age (scaled)", "income (scaled)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for ax, (pts, Q, title, xl, yl) in zip(axes, panels):
        d = np.linalg.norm(pts - Q, axis=1); idx = np.argsort(d)[:5]
        ax.scatter(pts[lab == 0, 0], pts[lab == 0, 1], c=BLUE, s=45, edgecolors="white", label="younger (<40)")
        ax.scatter(pts[lab == 1, 0], pts[lab == 1, 1], c=ORANGE, s=45, edgecolors="white", label="older (>40)")
        ax.scatter(*Q, marker="*", s=380, c=DARK, zorder=5, label="new person: age 30")
        for i in idx:
            ax.plot([Q[0], pts[i, 0]], [Q[1], pts[i, 1]], color=TEAL_DARK, lw=1.3)
        va = int((lab[idx] == 0).sum())
        pred = "younger ✓" if va > 2 else "older ✗"
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.text(0.98, 0.03, f"5 neighbours: {va} younger / {5 - va} older\n→ predicts {pred}",
                transform=ax.transAxes, ha="right", fontsize=9.5,
                bbox=dict(boxstyle="round", fc="white", ec=TEAL_DARK))
        ax.legend(loc="upper left", fontsize=8)
    axes[0].ticklabel_format(axis="y", style="plain")
    fig.suptitle("KNN measures distance — unscaled features silently decide who the neighbours are",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "knn_scaling.png", CH)


def hyperparam_vs_param():
    fig, ax = plt.subplots(figsize=(9, 3.6)); ax.axis("off")
    boxes = [
        (0.02, "Hyperparameter", TEAL_DARK,
         "Chosen by YOU before training\n\nKNN: k (number of neighbours)\nPolynomial: degree\nTree: max depth\n\nTuned with validation data / CV"),
        (0.54, "Parameter", ORANGE,
         "Learned by the ALGORITHM from data\n\nLinear regression: slope, intercept\nLogistic regression: weights\n\n(KNN has none — it stores the data)"),
    ]
    for x, title, col, body in boxes:
        ax.add_patch(mpatches.FancyBboxPatch((x, 0.05), 0.44, 0.9, boxstyle="round,pad=0.02", fc="white", ec=col, lw=2))
        ax.text(x + 0.22, 0.86, title, ha="center", va="center", fontsize=14, fontweight="bold", color=col)
        ax.text(x + 0.22, 0.42, body, ha="center", va="center", fontsize=10.5, color=DARK, linespacing=1.5)
    save(fig, "hyperparam_vs_param.png", CH)


def generate():
    print("Ch03 images:")
    knn_vote()
    knn_boundary_gif()
    knn_scaling()
    hyperparam_vs_param()
