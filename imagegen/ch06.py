"""Ch06 — Metrics: threshold sweep GIF (ROC + PR + confusion matrix),
accuracy-paradox bars, ROC-vs-PR under imbalance."""
from imagegen.common import *
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (roc_curve, precision_recall_curve, confusion_matrix,
                             roc_auc_score, average_precision_score, recall_score,
                             accuracy_score)


def _imbalanced_data(w=0.95, seed=0, class_sep=1.6):
    X, y = make_classification(n_samples=4000, n_features=10, n_informative=5,
                               weights=[w, 1 - w], flip_y=0.0, class_sep=class_sep,
                               random_state=seed)
    return train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)


# ── 1. Threshold sweep: point moves along ROC and PR, confusion matrix updates ──
def _threshold_gif():
    Xtr, Xte, ytr, yte = _imbalanced_data(w=0.9, class_sep=1.0)
    clf = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    fpr, tpr, _ = roc_curve(yte, p)
    prec, rec, _ = precision_recall_curve(yte, p)
    thresholds = np.round(np.linspace(0.05, 0.95, 19), 2)

    fig, (ax_roc, ax_pr, ax_cm) = plt.subplots(1, 3, figsize=(13, 4.3))
    ax_roc.plot(fpr, tpr, color=BLUE, lw=2)
    ax_roc.plot([0, 1], [0, 1], "--", color=MUTED, lw=1)
    ax_roc.set_xlabel("False Positive Rate"); ax_roc.set_ylabel("True Positive Rate (recall)")
    ax_roc.set_title(f"ROC curve  (AUC = {roc_auc_score(yte, p):.2f})")
    ax_pr.plot(rec, prec, color=RED, lw=2)
    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
    ax_pr.set_title(f"Precision-Recall curve  (AP = {average_precision_score(yte, p):.2f})")
    ax_pr.set_ylim(0, 1.05)
    dot_roc, = ax_roc.plot([], [], "o", color=DARK, ms=11, zorder=5)
    dot_pr, = ax_pr.plot([], [], "o", color=DARK, ms=11, zorder=5)
    ax_cm.set_xticks([0, 1]); ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["pred 0", "pred 1"]); ax_cm.set_yticklabels(["true 0", "true 1"])
    ax_cm.set_xlim(-0.5, 1.5); ax_cm.set_ylim(1.5, -0.5)
    ax_cm.tick_params(length=0)
    for sp in ax_cm.spines.values(): sp.set_visible(False)
    texts = [[ax_cm.text(j, i, "", ha="center", va="center", fontsize=15, fontweight="bold")
              for j in range(2)] for i in range(2)]
    cols = [["#d5f5e3", "#fde2c8"], ["#f9d0cb", "#d5f5e3"]]
    for i in range(2):
        for j in range(2):
            ax_cm.add_patch(mpatches.Rectangle((j - 0.5, i - 0.5), 1, 1, fc=cols[i][j], ec="white", lw=3))
    sup = fig.suptitle("", fontsize=13, fontweight="bold")

    def update(k):
        t = thresholds[k]
        yp = (p >= t).astype(int)
        cm = confusion_matrix(yte, yp)
        tn, fp, fn, tp = cm.ravel()
        r = tp / (tp + fn); pr = tp / max(tp + fp, 1); f = fp / (fp + tn)
        dot_roc.set_data([f], [r]); dot_pr.set_data([r], [pr])
        for (i, j), lab in zip([(0, 0), (0, 1), (1, 0), (1, 1)], ["TN", "FP", "FN", "TP"]):
            texts[i][j].set_text(f"{lab}\n{cm[i, j]}")
        ax_cm.set_title(f"threshold = {t:.2f}\nprecision {pr:.2f}   recall {r:.2f}   acc {accuracy_score(yte, yp):.3f}",
                        fontsize=11)
        sup.set_text("Sliding the decision threshold: same model, very different trade-offs")

    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save_gif(fig, update, len(thresholds), "threshold_sweep.gif", ["ch06"], fps=2, hold_last=4)


# ── 2. Accuracy paradox: dummy vs. real model on 98/2 data ─────────────────
def _accuracy_paradox():
    Xtr, Xte, ytr, yte = _imbalanced_data(w=0.98, seed=1, class_sep=2.0)
    dummy = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
    rows = [("Dummy: always 'healthy'", dummy), ("Logistic Regression\n(class_weight='balanced')", lr)]
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = np.arange(2); w = 0.35
    accs = [accuracy_score(yte, m.predict(Xte)) for _, m in rows]
    recs = [recall_score(yte, m.predict(Xte)) for _, m in rows]
    b1 = ax.bar(xs - w / 2, accs, w, color=MUTED, label="accuracy")
    b2 = ax.bar(xs + w / 2, recs, w, color=RED, label="recall (sick patients found)")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{b.get_height():.2f}",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylim(0, 1.15); ax.set_ylabel("score on test set")
    ax.set_title(f"Accuracy paradox: {100*(1-yte.mean()):.0f}% healthy, {100*yte.mean():.0f}% sick")
    ax.legend(loc="upper center", ncol=2, fontsize=9, frameon=False)
    fig.tight_layout()
    save(fig, "accuracy_paradox.png", ["ch06"])


# ── 3. ROC vs PR under imbalance (why PR curve exists) ────────────────────
def _roc_vs_pr():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for w, col, lab in [(0.5, BLUE, "balanced 50/50"), (0.97, RED, "imbalanced 97/3")]:
        Xtr, Xte, ytr, yte = _imbalanced_data(w=w, seed=2)
        p = LogisticRegression(max_iter=2000).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(yte, p); pr, rc, _ = precision_recall_curve(yte, p)
        axes[0].plot(fpr, tpr, color=col, lw=2, label=f"{lab}  (AUC {roc_auc_score(yte, p):.2f})")
        axes[1].plot(rc, pr, color=col, lw=2, label=f"{lab}  (AP {average_precision_score(yte, p):.2f})")
    axes[0].plot([0, 1], [0, 1], "--", color=MUTED, lw=1)
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC looks fine either way…")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision"); axes[1].set_ylim(0, 1.05)
    axes[1].set_title("…the PR curve exposes the imbalance")
    for ax in axes: ax.legend(fontsize=9, loc="lower left" if ax is axes[1] else "lower right")
    fig.tight_layout()
    save(fig, "roc_vs_pr_imbalance.png", ["ch06"])


def generate():
    print("ch06:")
    _threshold_gif()
    _accuracy_paradox()
    _roc_vs_pr()


if __name__ == "__main__":
    generate()
