"""Ch12 — Capstone (Titanic): opening EDA figure, preprocessing-pipeline diagram,
animated model leaderboard (numbers computed with the same pipeline as the notebook)."""
from imagegen.common import *
import pandas as pd

CSV = ROOT / "0-datasets" / "titanic.csv"


def _load():
    return pd.read_csv(CSV)


def make_survival_by_sex_class():
    df = _load()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    tab = df.groupby(["Pclass", "Sex"])["Survived"].agg(["mean", "size"]).reset_index()
    x = np.arange(3); w = 0.38
    for k, (sex, col) in enumerate([("female", PURPLE), ("male", TEAL_DARK)]):
        sub = tab[tab.Sex == sex]
        bars = ax.bar(x + (k - 0.5) * w, sub["mean"], w, color=col, alpha=0.9, label=sex)
        for b, m, n in zip(bars, sub["mean"], sub["size"]):
            ax.text(b.get_x() + b.get_width() / 2, m + 0.02, f"{m:.0%}\n(n={n})", ha="center", fontsize=9)
    ax.axhline(df.Survived.mean(), color=MUTED, ls="--", lw=1.2)
    ax.text(2.6, df.Survived.mean() + 0.02, f"overall {df.Survived.mean():.0%}", fontsize=9, color=MUTED, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(["1st class", "2nd class", "3rd class"])
    ax.set_ylim(0, 1.12); ax.set_ylabel("survival rate")
    ax.set_title("Survival rate by class and sex"); ax.legend(frameon=False, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    ax = axes[1]
    miss = df.isnull().mean().sort_values(ascending=True)
    miss = miss[miss > 0]
    ax.barh(miss.index, miss.values, color=[RED if v > 0.5 else ORANGE if v > 0.1 else MUTED for v in miss.values])
    for i, v in enumerate(miss.values):
        ax.text(v + 0.01, i, f"{v:.1%}", va="center", fontsize=10)
    ax.set_xlim(0, 1); ax.set_xlabel("fraction missing"); ax.set_title("Missing values (891 rows)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.suptitle("Titanic — “women and children first” is in the data, and it is messy", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "titanic_survival_by_sex_class.png", ["ch12"])


def make_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5.4); ax.axis("off")

    def box(x, y, w, h, text, color, fs=10, tc="white"):
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                             facecolor=color, edgecolor="white", lw=2, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
                fontweight="bold", zorder=3)

    def arrow(x0, y0, x1, y1, color=MUTED):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2, mutation_scale=16), zorder=4)

    box(0.2, 1.7, 1.7, 1.6, "raw X\n(train fold)", BLUE)
    # ColumnTransformer frame
    ax.add_patch(mpatches.FancyBboxPatch((2.6, 0.5), 5.2, 4.1, boxstyle="round,pad=0.1",
                                         facecolor="#f2fbfb", edgecolor=TEAL, lw=2, ls="--", zorder=1))
    ax.text(5.2, 4.32, "ColumnTransformer", ha="center", fontsize=11, color=TEAL_DARK, fontweight="bold")
    box(2.9, 2.6, 2.2, 1.3, "numeric cols\nAge, Fare, SibSp,\nParch, FamilySize", "#dfe6e9", 8.5, DARK)
    box(5.3, 2.6, 2.3, 1.3, "SimpleImputer(median)\n→ StandardScaler", TEAL_DARK, 8.5)
    box(2.9, 0.8, 2.2, 1.3, "categorical cols\nSex, Pclass,\nEmbarked, Title", "#dfe6e9", 8.5, DARK)
    box(5.3, 0.8, 2.3, 1.3, "SimpleImputer(most_frequent)\n→ OneHotEncoder", TEAL_DARK, 8)
    arrow(1.9, 2.5, 2.9, 3.25); arrow(1.9, 2.5, 2.9, 1.45)
    arrow(5.1, 3.25, 5.3, 3.25); arrow(5.1, 1.45, 5.3, 1.45)
    box(8.4, 1.7, 1.6, 1.6, "classifier\n(LogReg / RF / GB)", ORANGE, 9)
    arrow(7.6, 3.25, 8.4, 2.7); arrow(7.6, 1.45, 8.4, 2.3)
    box(10.5, 1.7, 1.3, 1.6, "ŷ", GREEN, 16)
    arrow(10.0, 2.5, 10.5, 2.5)
    ax.add_patch(mpatches.FancyBboxPatch((2.4, 0.15), 7.9, 5.1, boxstyle="round,pad=0.05",
                                         fill=False, edgecolor=ORANGE, lw=1.5, zorder=0))
    ax.text(6.35, 4.95, "Pipeline  →  fit() learns medians / modes / means / categories from the TRAIN fold only",
            ha="center", fontsize=9.5, color=ORANGE, fontweight="bold")
    ax.set_title("Leakage-safe preprocessing: everything that is *learned* from data lives inside the Pipeline",
                 fontsize=12, fontweight="bold", pad=14)
    save(fig, "titanic_pipeline.png", ["ch12"])


def _cv_results():
    from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    df = _load()

    def add_features(d):
        d = d.copy()
        d["FamilySize"] = d["SibSp"] + d["Parch"] + 1
        d["IsAlone"] = (d["FamilySize"] == 1).astype(int)
        d["Title"] = d["Name"].str.extract(r",\s*([^\.]+)\.")[0].str.strip()
        d["Title"] = d["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
        d.loc[~d["Title"].isin(["Mr", "Mrs", "Miss", "Master"]), "Title"] = "Rare"
        return d
    X = add_features(df).drop(columns=["Survived", "PassengerId", "Name", "Ticket", "Cabin"])
    y = df["Survived"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    num = ["Age", "Fare", "SibSp", "Parch", "FamilySize"]
    cat = ["Sex", "Pclass", "Embarked", "Title", "IsAlone"]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat)])
    models = {
        "Baseline\n(majority)": DummyClassifier(strategy="most_frequent"),
        "Logistic\nRegression": LogisticRegression(max_iter=1000, random_state=42),
        "Random\nForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient\nBoosting": GradientBoostingClassifier(random_state=42),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    out = {}
    for name, m in models.items():
        r = cross_validate(Pipeline([("pre", pre), ("clf", m)]), Xtr, ytr, cv=cv, scoring=["accuracy", "f1"])
        out[name] = (r["test_accuracy"].mean(), r["test_f1"].mean())
    return out


def make_leaderboard_gif():
    res = _cv_results()
    names = list(res); acc = np.array([res[n][0] for n in names]); f1 = np.array([res[n][1] for n in names])
    steps_per = 8
    n_frames = steps_per * len(names)
    fig, ax = plt.subplots(figsize=(9, 4.8))

    def update(i):
        ax.clear()
        k, t = divmod(i, steps_per)
        frac = (t + 1) / steps_per
        x = np.arange(len(names)); w = 0.38
        a_show = np.array([acc[j] if j < k else (acc[j] * frac if j == k else 0) for j in range(len(names))])
        f_show = np.array([f1[j] if j < k else (f1[j] * frac if j == k else 0) for j in range(len(names))])
        ax.bar(x - w / 2, a_show, w, color=MUTED, alpha=0.8, label="accuracy")
        ax.bar(x + w / 2, f_show, w, color=TEAL, alpha=0.9, label="F1 (survived)")
        for j in range(len(names)):
            if j < k or (j == k and t == steps_per - 1):
                ax.text(x[j] - w / 2, acc[j] + 0.02, f"{acc[j]:.2f}", ha="center", fontsize=10)
                ax.text(x[j] + w / 2, f1[j] + 0.02, f"{f1[j]:.2f}", ha="center", fontsize=10, color=TEAL_DARK, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(names)
        ax.set_ylim(0, 1.0); ax.set_ylabel("5-fold CV score (training set)")
        ax.set_title("Titanic leaderboard: always beat the baseline first")
        ax.legend(loc="upper left", frameon=False); ax.grid(True, axis="y", alpha=0.3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    update(0); fig.tight_layout()
    save_gif(fig, update, n_frames, "titanic_leaderboard.gif", ["ch12"], fps=4, hold_last=8)


def generate():
    print("── Ch12 ──")
    make_survival_by_sex_class()
    make_pipeline_diagram()
    make_leaderboard_gif()


if __name__ == "__main__":
    generate()
