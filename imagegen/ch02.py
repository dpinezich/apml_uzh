"""Ch02 — Data Selection, Cleaning & Preparing.

Images:
  preprocess_order.png    two lanes: WRONG (impute/scale on all data, then split) vs RIGHT (split, fit on train)
  leakage_impute.gif/.png animation: mean imputation before vs after the split — the test set 'leaks' into the fill value
  scaling_knn_effect.png  the same 2 features raw vs standardised (concept card for 'why scale')
"""
from imagegen.common import *
from matplotlib.patches import FancyArrowPatch

CH = ["ch02"]


def preprocess_order():
    fig, ax = plt.subplots(figsize=(10, 4.4)); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.4)

    def lane(y, title, col, steps, verdict):
        ax.text(0.15, y + 0.95, title, fontsize=13, fontweight="bold", color=col, va="center")
        x = 0.15
        for i, (txt, c) in enumerate(steps):
            w = 1.9
            ax.add_patch(mpatches.FancyBboxPatch((x, y), w, 0.7, boxstyle="round,pad=0.03", fc=c, ec="none"))
            ax.text(x + w / 2, y + 0.35, txt, ha="center", va="center", fontsize=9.5, color="white", fontweight="bold")
            if i < len(steps) - 1:
                ax.add_patch(FancyArrowPatch((x + w + 0.05, y + 0.35), (x + w + 0.35, y + 0.35),
                                             arrowstyle="-|>", mutation_scale=14, color=MUTED))
            x += w + 0.4
        ax.text(x + 0.05, y + 0.35, verdict, fontsize=10.5, va="center", color=col, fontweight="bold")

    lane(2.7, "✗  Wrong order — test data leaks into the statistics", RED,
         [("compute mean\n(ALL rows)", RED), ("fill NaN\n(ALL rows)", RED), ("fit scaler\n(ALL rows)", RED), ("split", MUTED)],
         "test score\ntoo optimistic")
    lane(0.5, "✓  Right order — everything is fit on the training set only", TEAL_DARK,
         [("split", TEAL_DARK), ("fit imputer\non TRAIN", TEAL_DARK), ("fit scaler\non TRAIN", TEAL_DARK), ("transform\ntrain + test", GREEN)],
         "honest\ntest score")
    ax.text(5, 2.2, "Rule: anything that computes a statistic (mean, median, min/max, std, IQR, categories) → fit on train, transform both",
            ha="center", fontsize=10, color=DARK, style="italic")
    save(fig, "preprocess_order.png", CH)


def leakage_impute_gif():
    rng = np.random.default_rng(1)
    train = rng.normal(50, 8, 14).round(0)
    test = rng.normal(80, 5, 6).round(0)          # test set from a "different" population → shows the leak clearly
    vals = np.r_[train, test]
    n_tr = len(train)
    missing_idx = 4                                # one missing value in train
    mean_all = np.r_[np.delete(train, missing_idx), test].mean()
    mean_tr = np.delete(train, missing_idx).mean()

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.set_xlim(-0.7, len(vals) - 0.3); ax.set_ylim(0, 105)
    ax.set_xlabel("row"); ax.set_ylabel("feature value")
    ax.set_xticks(range(len(vals))); ax.set_xticklabels([str(i) for i in range(len(vals))], fontsize=8)
    bars = ax.bar(range(len(vals)), vals, color=[BLUE] * n_tr + [ORANGE] * len(test), edgecolor="white")
    bars[missing_idx].set_height(0)
    qm = ax.text(missing_idx, 3, "NaN", ha="center", va="bottom", fontsize=10, color=RED, fontweight="bold")
    ax.axvspan(n_tr - 0.5, len(vals) - 0.5, color=ORANGE, alpha=0.08)
    ax.text(n_tr / 2 - 0.5, 100, "TRAIN", ha="center", color=BLUE, fontweight="bold")
    ax.text(n_tr + len(test) / 2 - 0.5, 100, "TEST (locked away)", ha="center", color=ORANGE, fontweight="bold")
    line = ax.axhline(0, color=DARK, lw=1.5, ls="--", visible=False)
    ttl = ax.set_title("")
    msg = ax.text(0.01, 0.80, "", transform=ax.transAxes, fontsize=10.5,
                  bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=BORDER))
    fill = [None]

    steps = [
        ("A dataset with one missing value in the training set", "Task: fill the NaN with the mean.", DARK, None, None),
        ("✗ Wrong: mean over ALL rows (train + test)", f"mean(all rows) = {mean_all:.1f}  ← the test rows pulled it up", RED, mean_all, RED),
        ("✗ Wrong: the filled value carries test-set information", f"row {missing_idx} = {mean_all:.1f} — the model now 'knows' the test distribution", RED, mean_all, RED),
        ("✓ Right: split first, mean over TRAIN rows only", f"mean(train rows) = {mean_tr:.1f}", GREEN, mean_tr, GREEN),
        ("✓ Right: fill train AND test NaNs with the TRAIN mean", f"row {missing_idx} = {mean_tr:.1f} — test set never touched", GREEN, mean_tr, GREEN),
    ]

    def update(i):
        t, m, col, lvl, fc = steps[i]
        ttl.set_text(t); ttl.set_color(col); msg.set_text(m)
        if fill[0] is not None:
            fill[0].remove(); fill[0] = None
        if lvl is None:
            line.set_visible(False); qm.set_visible(True)
        else:
            line.set_ydata([lvl, lvl]); line.set_visible(True); line.set_color(fc)
            fill[0] = ax.bar([missing_idx], [lvl], color=fc, edgecolor="white", hatch="//")
            qm.set_visible(False)

    save_gif(fig, update, len(steps), "leakage_impute.gif", CH, fps=1, hold_last=2)


def generate():
    print("Ch02 images:")
    preprocess_order()
    leakage_impute_gif()
