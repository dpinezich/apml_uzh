"""Ch05 — Classification: decision boundaries (DecisionBoundaryDisplay),
decision-tree depth sweep GIF, confusion-matrix concept card."""
from imagegen.common import *
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay
from matplotlib.colors import ListedColormap

CMAP2 = ListedColormap(["#cfe9f5", "#f9d0cb"])   # benign-ish blue, malignant-ish red


# ── 1. Decision boundaries of 4 classifiers on 2 breast-cancer features ────
def _decision_boundaries():
    d = load_breast_cancer()
    y = 1 - d.target                       # malignant = 1 = positive class
    names = list(d.feature_names)
    idx = [names.index("worst radius"), names.index("worst texture")]
    X = d.data[:, idx]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "KNN (k=5)":           make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
        "Decision Tree (depth 4)": DecisionTreeClassifier(max_depth=4, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    for ax, (name, m) in zip(axes.flat, models.items()):
        m.fit(Xtr, ytr)
        DecisionBoundaryDisplay.from_estimator(m, X, response_method="predict",
                                               cmap=CMAP2, alpha=0.9, ax=ax,
                                               grid_resolution=300)
        ax.scatter(Xte[yte == 0, 0], Xte[yte == 0, 1], c=BLUE, s=28, edgecolor="white",
                   lw=0.5, label="benign (0)")
        ax.scatter(Xte[yte == 1, 0], Xte[yte == 1, 1], c=RED, s=28, edgecolor="white",
                   lw=0.5, label="malignant (1)")
        ax.set_title(f"{name}  —  test acc {m.score(Xte, yte):.2f}", fontsize=11)
        ax.set_xlabel("worst radius"); ax.set_ylabel("worst texture")
        ax.set_xlim(X[:, 0].min() - 1, X[:, 0].max() + 1); ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Same data, four models, four different boundaries", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "decision_boundaries_2d.png", ["ch05"])


# ── 2. Decision tree depth sweep on make_moons (overfitting spiral) ─────────
def _tree_depth_gif():
    X, y = make_moons(n_samples=300, noise=0.3, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
    depths = list(range(1, 13))
    fig, (ax_b, ax_acc) = plt.subplots(1, 2, figsize=(11, 4.6),
                                       gridspec_kw={"width_ratios": [1.15, 1]})
    tr_acc, te_acc = [], []
    fitted = []
    for dep in depths:
        t = DecisionTreeClassifier(max_depth=dep, random_state=0).fit(Xtr, ytr)
        fitted.append(t); tr_acc.append(t.score(Xtr, ytr)); te_acc.append(t.score(Xte, yte))
    ax_acc.set_xlim(0.5, 12.5); ax_acc.set_ylim(0.6, 1.02); ax_acc.set_xticks(depths)
    ax_acc.set_xlabel("max_depth"); ax_acc.set_ylabel("accuracy")
    ax_acc.set_title("Train vs. test accuracy"); ax_acc.grid(True, axis="y")
    l_tr, = ax_acc.plot([], [], "o-", color=BLUE, lw=2, label="train")
    l_te, = ax_acc.plot([], [], "s-", color=RED, lw=2, label="test")
    ax_acc.legend(loc="lower right", fontsize=9)

    def update(i):
        ax_b.cla()
        DecisionBoundaryDisplay.from_estimator(fitted[i], X, response_method="predict",
                                               cmap=CMAP2, alpha=0.9, ax=ax_b, grid_resolution=250)
        ax_b.scatter(Xtr[ytr == 0, 0], Xtr[ytr == 0, 1], c=BLUE, s=22, edgecolor="white", lw=0.4)
        ax_b.scatter(Xtr[ytr == 1, 0], Xtr[ytr == 1, 1], c=RED, s=22, edgecolor="white", lw=0.4)
        ax_b.set_title(f"DecisionTree(max_depth={depths[i]})  —  training points shown")
        ax_b.set_xticks([]); ax_b.set_yticks([])
        l_tr.set_data(depths[: i + 1], tr_acc[: i + 1])
        l_te.set_data(depths[: i + 1], te_acc[: i + 1])

    fig.tight_layout()
    save_gif(fig, update, len(depths), "tree_depth_sweep.gif", ["ch05"], fps=2, hold_last=5)


# ── 3. Confusion-matrix concept card (positive = malignant) ─────────────────
def _cm_card():
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.set_xlim(0, 4); ax.set_ylim(0, 3.4); ax.axis("off")
    cells = [  # (x, y, colour, big, small)
        (1, 1.7, "#d5f5e3", "TN", "healthy, predicted healthy\n✓ correct"),
        (2.5, 1.7, "#fde2c8", "FP", "healthy, predicted cancer\nfalse alarm (Type I)"),
        (1, 0.4, "#f9d0cb", "FN", "cancer, predicted healthy\nMISSED CASE (Type II) ⚠"),
        (2.5, 0.4, "#d5f5e3", "TP", "cancer, predicted cancer\n✓ correct"),
    ]
    for x, y, c, big, small in cells:
        ax.add_patch(mpatches.FancyBboxPatch((x, y), 1.4, 1.2, boxstyle="round,pad=0.02",
                                             fc=c, ec=BORDER))
        ax.text(x + 0.7, y + 0.85, big, ha="center", va="center", fontsize=22,
                fontweight="bold", color=DARK)
        ax.text(x + 0.7, y + 0.35, small, ha="center", va="center", fontsize=8.5, color=MUTED)
    ax.text(1.7, 3.15, "Predicted: healthy (0)", ha="center", fontsize=11, fontweight="bold")
    ax.text(3.2, 3.15, "Predicted: cancer (1)", ha="center", fontsize=11, fontweight="bold")
    ax.text(0.85, 2.3, "Actual:\nhealthy (0)", ha="right", va="center", fontsize=11, fontweight="bold")
    ax.text(0.85, 1.0, "Actual:\ncancer (1)", ha="right", va="center", fontsize=11, fontweight="bold")
    ax.text(2.0, 0.05, "rows = truth,  columns = prediction  (sklearn convention) — positive class = 1 = the thing you look for",
            ha="center", fontsize=9, color=TEAL_DARK, fontweight="bold")
    save(fig, "confusion_matrix_card.png", ["ch05", "ch06"])


def generate():
    print("ch05:")
    _decision_boundaries()
    _tree_depth_gif()
    _cm_card()


if __name__ == "__main__":
    generate()
