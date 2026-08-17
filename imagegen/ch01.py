"""Ch01 — Introduction to ML.

Images:
  rules_vs_data.png        concept card: traditional programming vs machine learning (replaces ASCII text)
  workflow_cycle_anim.gif  the 7-step workflow lights up step by step, then loops (cycle, not pipeline)
  ml_learns_boundary.gif   fun mini-demo: KNN on iris petals — the boundary is "learned" from data
"""
from imagegen.common import *
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

CH = ["ch01"]


def rules_vs_data():
    fig, ax = plt.subplots(figsize=(9.5, 3.9)); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 4)

    def row(y, title, col, boxes, out):
        ax.text(0.1, y + 1.05, title, fontsize=13, fontweight="bold", color=col, va="center")
        x = 0.1
        for txt in boxes:
            ax.add_patch(mpatches.FancyBboxPatch((x, y), 1.8, 0.75, boxstyle="round,pad=0.03", fc="white", ec=col, lw=2))
            ax.text(x + 0.9, y + 0.375, txt, ha="center", va="center", fontsize=11, color=DARK, fontweight="bold")
            x += 2.2
        ax.add_patch(FancyArrowPatch((x - 0.25, y + 0.375), (x + 0.45, y + 0.375), arrowstyle="-|>", mutation_scale=18, color=MUTED, lw=2))
        ax.add_patch(mpatches.FancyBboxPatch((x + 0.5, y), 3.0, 0.75, boxstyle="round,pad=0.03", fc=col, ec="none"))
        ax.text(x + 2.0, y + 0.375, out, ha="center", va="center", fontsize=11, color="white", fontweight="bold")
        ax.text(2.15, y + 0.375, "+", ha="center", va="center", fontsize=16, color=MUTED, fontweight="bold")

    row(2.55, "Traditional programming", MUTED, ["Rules\n(hand-written)", "Data"], "Answers")
    row(0.55, "Machine learning", TEAL_DARK, ["Data", "Answers\n(labels)"], "Rules  (= the model)")
    ax.text(4.5, 1.95, "The computer writes the rules — we supply examples.", ha="center", fontsize=10.5, style="italic", color=DARK)
    save(fig, "rules_vs_data.png", CH)


def workflow_cycle_gif():
    steps = ["① Define\nProblem", "② Collect\nData", "③ EDA &\nClean", "④ Preprocess\n& Engineer",
             "⑤ Train\nModel", "⑥ Evaluate", "⑦ Deploy\n& Monitor"]
    hints = ["What exactly do we predict? For whom?", "Where does the data come from? Is it representative?",
             "Look at it! Missing values, outliers, typos (Ch02)", "Impute, encode, scale — after the split (Ch02)",
             "fit() — the algorithm learns (Ch03 →)", "Score on held-out data. Better than a baseline?",
             "A model in a notebook helps no one. Monitor drift → loop again"]
    n = len(steps)
    angles = [np.pi / 2 - 2 * np.pi * i / n for i in range(n)]
    R, br = 1.15, 0.27
    fig, ax = plt.subplots(figsize=(7.4, 7.4))
    ax.set_xlim(-1.75, 1.75); ax.set_ylim(-1.75, 1.75); ax.set_aspect("equal"); ax.axis("off")
    boxes, arrows = [], []
    for i, (a, s) in enumerate(zip(angles, steps)):
        x, y = R * np.cos(a), R * np.sin(a)
        b = mpatches.FancyBboxPatch((x - br, y - br * 0.85), 2 * br, 2 * br * 0.85, boxstyle="round,pad=0.05",
                                    fc=BORDER, ec="white", lw=2, zorder=3)
        ax.add_patch(b); boxes.append(b)
        ax.text(x, y, s, ha="center", va="center", color="white", fontsize=8.5, fontweight="bold", zorder=4)
        na = angles[(i + 1) % n]; nx, ny = R * np.cos(na), R * np.sin(na)
        dx, dy = nx - x, ny - y
        arr = FancyArrowPatch((x + 0.31 * dx, y + 0.31 * dy), (x + 0.69 * dx, y + 0.69 * dy),
                              arrowstyle="-|>", color=BORDER, lw=1.8, mutation_scale=14, zorder=2)
        ax.add_patch(arr); arrows.append(arr)
    centre = ax.text(0, 0, "", ha="center", va="center", fontsize=11, fontweight="bold", color=DARK, wrap=True)
    ttl = ax.set_title("", fontsize=13, fontweight="bold", color=DARK, pad=10)
    frames = list(range(n)) + ["loop"]

    def update(f):
        step = frames[f]
        for i, (b, a) in enumerate(zip(boxes, arrows)):
            if step == "loop":
                b.set_facecolor(TEAL); a.set_color(TEAL_DARK)
            else:
                b.set_facecolor(TEAL_DARK if i == step else (TEAL if i < step else BORDER))
                a.set_color(TEAL_DARK if i < step else BORDER)
        if step == "loop":
            ttl.set_text("…and back to ①: this is a cycle, not a pipeline")
            centre.set_text("Loop 2, 3, …\nExpect to iterate!")
        else:
            ttl.set_text(f"Step {step + 1} of 7")
            centre.set_text("\n".join(_wrap(hints[step], 22)))

    save_gif(fig, update, len(frames), "workflow_cycle_anim.gif", CH, fps=1, hold_last=2)


def _wrap(text, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    lines.append(cur)
    return lines


def ml_learns_boundary_gif():
    """KNN (k=15) on iris petal features, trained on a growing subset of the data: the boundary is learned."""
    iris = load_iris()
    X = iris.data[:, 2:4]; y = iris.target
    rng = np.random.default_rng(0); order = rng.permutation(len(y))
    h = 0.03
    xx, yy = np.meshgrid(np.arange(0.5, 7.5, h), np.arange(-0.2, 3.0, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    sizes = [3, 6, 10, 15, 25, 40, 60, 90, 120, 150]
    cols = [RED, BLUE, GREEN]
    cmap = ListedColormap(["#f9d4cf", "#cfe6fb", "#d3f5e0"])
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.set_xlim(0.5, 7.5); ax.set_ylim(-0.2, 3.0)
    ax.set_xlabel("petal length (cm)"); ax.set_ylabel("petal width (cm)")
    mesh = [None]; sc = [None]
    ttl = ax.set_title("")
    for c, name in zip(cols, iris.target_names):
        ax.scatter([], [], c=c, label=name, s=40, edgecolors="white")
    ax.legend(loc="upper left", fontsize=9)

    def update(i):
        m = sizes[i]; idx = order[:m]
        k = min(15, max(1, m // 4))
        knn = KNeighborsClassifier(n_neighbors=k).fit(X[idx], y[idx])
        Z = knn.predict(grid).reshape(xx.shape)
        if mesh[0] is not None: mesh[0].remove()
        if sc[0] is not None: sc[0].remove()
        mesh[0] = ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5, 2.5], cmap=cmap, zorder=1)
        sc[0] = ax.scatter(X[idx, 0], X[idx, 1], c=[cols[t] for t in y[idx]], s=45, edgecolors="white", zorder=3)
        ttl.set_text(f"The model has seen {m} labelled flowers → the boundary it 'learned'")

    save_gif(fig, update, len(sizes), "ml_learns_boundary.gif", CH, fps=1, hold_last=3)


def generate():
    print("Ch01 images:")
    rules_vs_data()
    workflow_cycle_gif()
    ml_learns_boundary_gif()
