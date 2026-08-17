# Chapter 12 — Capstone: End-to-End ML Workflow (Titanic)

**Session:** 4 | **Chapter:** 3 of 3 | **Duration:** ~65 min (5 intro + **50 self-work** + 10 debrief)  
**Format:** Guided end-to-end project — minimal slides, maximum coding

---

## Learning Objectives

By the end of this chapter, students will be able to:
- Run a complete, **leakage-safe** ML workflow: split first → row-wise features → `ColumnTransformer` + `Pipeline` → baseline → CV → one test evaluation → interpretation
- Justify design choices (metric, imputation strategy, model choice) and read CV vs test results critically
- Explain what a model learned (permutation importance, error analysis) and ask whether it *should* have (fairness)
- See ML as one workflow, not a list of algorithms

---

## Timing Breakdown

| Block | Content | Time |
|-------|---------|------|
| 1 | Framing: mission, dataset, plan, pipeline diagram, Gradient Boosting in one slide, metric + rules | 5–6 min |
| 2 | **Self-work in the notebook** (no frontal teaching) | **50 min** |
| 3 | Debrief: leaderboard · what the model learned · fairness · recap · what's next | 10 min |
| **Total** | | **~65 min** |

Session 4 total: Ch10 ~45 + Ch11 ~45 + Ch12 ~65 = ~155 min (Ch10/Ch11 blocks carry ~5 min buffer each → fits 150 min if the demos run at pace).

---

## The Project

**Dataset:** Kaggle Titanic CSV — `0-datasets/titanic.csv` (891 rows; `PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked`), loaded with `pd.read_csv('../../0-datasets/titanic.csv')`.  
**Task:** binary classification, **primary metric F1 (class survived)**, accuracy alongside.  
**Baseline:** majority class → 62 % accuracy, F1 = 0.

**Notebook steps and time budget (50 min):**

| Step | Content | Given / TODO | Time |
|---|---|---|---|
| 1 Load & explore | CSV, EDA plots given (sex, class×sex, age), TODO 1 dtypes + missing | ▶ + TODO 1 | 5 |
| 2 Split first | TODO 2 `train_test_split(stratify=y, random_state=42)` | TODO 2 | 3 |
| 3 Feature engineering | `add_features()`: FamilySize, IsAlone, Title from Name (row-wise) | TODO 3a/3b | 7 |
| 4 Pipeline | numeric: median-impute + scale; categorical: mode-impute + one-hot; `ColumnTransformer` | TODO 4 (skeleton) | 8 |
| 5 Baseline + models | Dummy, LogReg, RF, GB in `Pipeline`, `cross_validate` (f1 + accuracy, StratifiedKFold) ; leaderboard plot given; best model chosen from results | TODO 5a/5b | 10 |
| 6 Final evaluation | fit best on train, predict test once, report + confusion matrix | TODO 6 | 7 |
| 7 Interpret | `permutation_importance` on test (raw frame); error analysis of 5 misclassified passengers given | TODO 7 + ▶ | 7 |
| 8 Reflect | incl. fairness question (Sex / Title as proxies) | text | 3 |
| Bonus | A `GridSearchCV` (RF, `clf__` prefix) · B `joblib` persistence · C PCA of the passengers (Ch09) | optional | — |

**Why these choices**
- Imputation/encoding/scaling **inside** the pipeline → the Ch02 rule "impute after split" is enforced by construction (re-fit per CV fold).
- Permutation importance instead of impurity importance: impurity importance over-ranks continuous columns (Age/Fare) and would contradict the EDA.
- Interpretation text in the solutions is **derived from computed values** (best model name, top feature, gaps) — never hard-coded.
- Gradient Boosting is new: introduced with one slide ("sequential shallow trees fitting the residual errors").

---

## Debrief (10 min)

1. **Leaderboard** (`titanic_leaderboard.gif`, computed with the same pipeline): everyone beat the baseline; the three real models are within ~0.02 F1 → features + clean evaluation > algorithm choice. Poll: who won on your CV? test F1?
2. **What the model learned:** Sex/Title on top, then Pclass/Fare; misclassified passengers are the "surprising" cases.
3. **Fairness:** the strongest feature is a protected attribute; Title/Fare/Pclass are proxies — dropping `Sex` alone does not fix it; per-group error rates, proxies, biased targets.
4. Recap table (which chapter showed up where) · what comes next · Kaggle.

---

## Instructor Notes

- **Do not interrupt the 50 minutes.** Walk around. Give a 10-min and a 2-min warning.
- Sticking points: `stratify=y`; imputer outside the pipeline; `clf__` prefix; `permutation_importance` on `X_test_fe` (raw), not on the transformed matrix.
- Reference numbers (seed 42): CV F1 LogReg ≈ 0.76, GB ≈ 0.75, RF ≈ 0.74, baseline 0; test F1 of best ≈ 0.80. Order among the three can flip with another seed — say so.
- Keep the fairness slide technical (checklist), not moralising.

## Materials

- Slides: `01-slides/ch12_slides.md` (images: `titanic_survival_by_sex_class.png`, `titanic_pipeline.png`, `titanic_leaderboard.gif`)
- Exercises / Solutions: `03-exercises/ch12_capstone_exercises.ipynb`, `04-solutions/ch12_capstone_solutions.ipynb` (no separate example notebook — the guided exercise is the example)
- Data: `0-datasets/titanic.csv`
- Image generator: `imagegen/ch12.py`
