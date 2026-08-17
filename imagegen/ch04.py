"""Ch04 — Regression: polynomial degree sweep GIF, gradient descent GIF,
Ridge/Lasso coefficient paths, DummyRegressor baseline card."""
from imagegen.common import *
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, KFold
from sklearn.datasets import load_diabetes


# ── 1. Polynomial degree sweep (train vs CV error filling in) ─────────────
def _poly_sweep():
    rng = np.random.RandomState(3)
    n = 30
    x = np.sort(rng.uniform(0, 2 * np.pi, n))
    y = np.sin(x) + rng.randn(n) * 0.3
    X = x.reshape(-1, 1)
    x_plot = np.linspace(0, 2 * np.pi, 300).reshape(-1, 1)
    degrees = list(range(1, 16))
    kf = KFold(5, shuffle=True, random_state=0)
    train_err, cv_err, preds = [], [], []
    for d in degrees:
        pipe = make_pipeline(StandardScaler(), PolynomialFeatures(d, include_bias=False),
                             StandardScaler(), Ridge(alpha=1e-6))
        pipe.fit(X, y)
        train_err.append(np.mean((pipe.predict(X) - y) ** 2))
        cv_err.append(-cross_val_score(pipe, X, y, cv=kf,
                                       scoring="neg_mean_squared_error").mean())
        preds.append(pipe.predict(x_plot))
    train_err, cv_err = np.array(train_err), np.array(cv_err)
    best = degrees[int(np.argmin(cv_err))]

    fig, (ax_fit, ax_err) = plt.subplots(1, 2, figsize=(11, 4.6),
                                         gridspec_kw={"width_ratios": [1.3, 1]})
    ax_fit.scatter(x, y, color=DARK, s=35, zorder=4, label="noisy data")
    ax_fit.plot(x_plot, np.sin(x_plot), "--", color=MUTED, lw=1.5, label="true sin(x)")
    fit_line, = ax_fit.plot([], [], color=RED, lw=2.5, label="polynomial fit")
    ax_fit.set_xlim(0, 2 * np.pi); ax_fit.set_ylim(-2.2, 2.2)
    ax_fit.set_xlabel("x"); ax_fit.set_ylabel("y")
    ax_fit.legend(loc="lower left", fontsize=9)
    tag = ax_fit.text(0.02, 0.96, "", transform=ax_fit.transAxes, va="top",
                      fontsize=11, fontweight="bold", color="white",
                      bbox=dict(boxstyle="round,pad=0.35", fc=TEAL_DARK, ec="none"))

    ax_err.set_xlim(0.5, 15.5)
    ax_err.set_yscale("log")
    ax_err.set_ylim(0.02, cv_err.max() * 2.5)
    ax_err.set_xticks(degrees)
    ax_err.set_xlabel("polynomial degree"); ax_err.set_ylabel("MSE (log scale)")
    ax_err.set_title("Train vs. cross-validation error")
    ax_err.grid(True, axis="y")
    l_tr, = ax_err.plot([], [], "o-", color=BLUE, lw=2, label="train MSE")
    l_cv, = ax_err.plot([], [], "s-", color=RED, lw=2, label="5-fold CV MSE")
    ax_err.legend(loc="upper left", fontsize=9, bbox_to_anchor=(0.0, 0.88))
    best_line = ax_err.axvline(best, color=GREEN, lw=1.5, ls="--", alpha=0)
    best_txt = ax_err.text(best + 0.3, 0.025, "", color=GREEN,
                           fontsize=9, fontweight="bold", va="bottom")

    def update(i):
        d = degrees[i]
        fit_line.set_data(x_plot.ravel(), preds[i])
        if d <= 2:
            lab, col = f"degree {d}: UNDERFITTING", ORANGE
        elif d <= 7:
            lab, col = f"degree {d}: good fit", GREEN
        else:
            lab, col = f"degree {d}: OVERFITTING", RED
        tag.set_text(lab); tag.get_bbox_patch().set_facecolor(col)
        ax_fit.set_title(f"Polynomial fit, degree {d}")
        l_tr.set_data(degrees[: i + 1], train_err[: i + 1])
        l_cv.set_data(degrees[: i + 1], cv_err[: i + 1])
        if d >= best:
            best_line.set_alpha(0.9)
            best_txt.set_text(f"lowest CV error\n→ degree {best}")

    fig.tight_layout()
    save_gif(fig, update, len(degrees), "poly_degree_sweep.gif", ["ch04"],
             fps=2, hold_last=6)


# ── 2. Gradient descent steps on a loss surface ────────────────────────────
def _gd_steps():
    def loss(w1, w2):
        return 0.5 * w1 ** 2 + 3.0 * w2 ** 2 + 0.8 * w1 * w2

    def grad(w):
        return np.array([w[0] + 0.8 * w[1], 6.0 * w[1] + 0.8 * w[0]])

    lr, w = 0.12, np.array([-3.5, 2.5])
    path = [w.copy()]
    for _ in range(25):
        w = w - lr * grad(w)
        path.append(w.copy())
    path = np.array(path)

    g1, g2 = np.meshgrid(np.linspace(-4.5, 4.5, 300), np.linspace(-3.2, 3.2, 300))
    L = loss(g1, g2)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    levels = np.logspace(-1.5, 1.8, 18)
    ax.contourf(g1, g2, L, levels=levels, cmap="YlGnBu_r", alpha=0.75)
    ax.contour(g1, g2, L, levels=levels, colors="white", linewidths=0.5, alpha=0.6)
    ax.scatter(0, 0, marker="*", s=260, color=GREEN, zorder=5, label="minimum (best weights)")
    ax.scatter(*path[0], s=90, color=BLUE, zorder=5, label="random start")
    line, = ax.plot([], [], "o-", color=DARK, lw=1.8, ms=4, zorder=4)
    pt, = ax.plot([], [], "o", color=RED, ms=10, zorder=6)
    txt = ax.text(0.02, 0.04, "", transform=ax.transAxes, fontsize=10, va="bottom",
                  bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=BORDER))
    ax.set_xlabel("weight w₁"); ax.set_ylabel("weight w₂")
    ax.set_title("Gradient descent: step downhill, repeat")
    ax.legend(loc="upper right", fontsize=9)

    def update(i):
        p = path[: i + 1]
        line.set_data(p[:, 0], p[:, 1])
        pt.set_data([p[-1, 0]], [p[-1, 1]])
        txt.set_text(f"step {i:2d}   loss = {loss(*p[-1]):.3f}")

    fig.tight_layout()
    save_gif(fig, update, len(path), "gradient_descent_steps.gif", ["ch04"],
             fps=3, hold_last=6)


# ── 3. Ridge / Lasso coefficient paths (visible shrinkage) ─────────────────
def _reg_paths():
    d = load_diabetes()
    X = StandardScaler().fit_transform(d.data)
    y = d.target
    alphas_r = np.logspace(-2, 4, 40)
    alphas_l = np.logspace(-2, 1.5, 40)
    cr = np.array([Ridge(alpha=a).fit(X, y).coef_ for a in alphas_r])
    cl = np.array([Lasso(alpha=a, max_iter=50000).fit(X, y).coef_ for a in alphas_l])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, al, c, name in [(axes[0], alphas_r, cr, "Ridge (L2): shrinks smoothly, never exactly 0"),
                            (axes[1], alphas_l, cl, "Lasso (L1): sets coefficients to exactly 0")]:
        for j, f in enumerate(d.feature_names):
            ax.plot(al, c[:, j], lw=2, label=f)
        ax.set_xscale("log"); ax.axhline(0, color=DARK, lw=0.8)
        ax.set_xlabel("alpha (regularization strength) →")
        ax.set_ylabel("coefficient")
        ax.set_title(name, fontsize=11)
        ax.grid(True, axis="y")
    axes[1].legend(fontsize=7, ncol=2, loc="upper right")
    fig.suptitle("Diabetes data: what happens to the 10 coefficients as alpha grows",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "ridge_lasso_paths.png", ["ch04"])


# ── 4. Baseline card: DummyRegressor (mean) vs a real model ────────────────
def _baseline_card():
    rng = np.random.RandomState(1)
    x = np.sort(rng.uniform(0, 10, 40)); y = 2 + 0.8 * x + rng.randn(40) * 1.5
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(x, y, color=DARK, s=30, zorder=3)
    ax.axhline(y.mean(), color=ORANGE, lw=2.5, label="DummyRegressor: always predict the mean  (R² = 0)")
    lr = LinearRegression().fit(x.reshape(-1, 1), y)
    ax.plot(x, lr.predict(x.reshape(-1, 1)), color=TEAL_DARK, lw=2.5,
            label=f"LinearRegression  (R² = {lr.score(x.reshape(-1,1), y):.2f})")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Always beat the baseline first")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    save(fig, "baseline_regression.png", ["ch04"])


def generate():
    print("ch04:")
    _poly_sweep()
    _gd_steps()
    _reg_paths()
    _baseline_card()


if __name__ == "__main__":
    generate()
