# Chapter 05 — Classification Models

**Session:** 2 | **Chapter:** 2 of 3 | **Duration:** 50 min  
**Audience:** Students who completed Ch04 (and KNN in Ch03)  
**Format:** Slides + live demo notebook (interleaved) + exercises

---

## Learning Objectives

By the end of this chapter, students will be able to:
- Distinguish binary from multi-class classification and hard labels from probabilities
- Start with a `DummyClassifier` (majority-class) baseline
- Apply Logistic Regression, KNN (recap), Decision Trees and Random Forests with correct pipelines (scaling where needed)
- Recognize tree depth / k as the model-complexity dial (spiral from Ch03/Ch04)
- Visualize and interpret decision boundaries (`DecisionBoundaryDisplay`)
- Read a confusion matrix and a `classification_report` (rows = truth, positive class = what you look for), and **check the class encoding** (breast cancer target flipped: malignant = 1)

SVM is an appendix slide + bonus task (cut from core for time).

---

## Timing Breakdown (sum 44 min → ~6 min buffer)

| Block | Content | Time |
|-------|---------|------|
| 1 | Classification: the task, predict vs predict_proba, majority baseline | 5 min |
| 2 | Logistic Regression + sigmoid | 6 min |
| 3 | KNN — 2-min recap (Ch03) | 2 min |
| 4 | Decision Trees, depth-sweep GIF, Random Forest | 7 min |
| 5 | Decision boundaries (demo, `DecisionBoundaryDisplay`) | 3 min |
| 6 | Reading a confusion matrix + `classification_report` (mini-slide before the exercise) | 4 min |
| 7 | Quick-check quiz | 2 min |
| 8 | **Exercises** (core) | **12 min** |
| 9 | Debrief | 1 min |
| — | Appendix: SVM (only if ahead of time) | (2 min) |
| **Total** | | **42–44 min** |

Demo notebook `02-examples/ch05_classification_examples.ipynb` (~12 min end-to-end) is interleaved: §1–2 in block 1, §3 in block 6, §4 in block 5, §5–6 in block 4.

---

## Content Outline

### Block 1 — The Task + Baseline (5 min)

Binary vs multi-class (multi-label: mention only). `predict()` → label, `predict_proba()` → probabilities per class (rows sum to 1); the probability is what lets us move the threshold in Ch06.  
Baseline: `DummyClassifier(strategy='most_frequent')` — 63 % on breast cancer by always saying "benign". Plants the accuracy-paradox seed.

**Positive class check (first use of breast cancer):** sklearn encodes 0 = malignant, 1 = benign. We flip once (`y = 1 - target`) so that **1 = malignant = the class we look for**; precision/recall/ROC default to `pos_label=1`.

### Block 2 — Logistic Regression (6 min)

P(y=1|x) = σ(β₀ + β·x), σ(z) = 1/(1+e⁻ᶻ); σ(0) = 0.5 → linear decision boundary. Trained by gradient descent on cross-entropy (mention only). Pipeline with `StandardScaler`, `max_iter=1000`. Multi-class handled natively (softmax).

### Block 3 — KNN recap (2 min)

Majority vote among k nearest; needs scaling; k = complexity dial (k = 1 overfits, huge k → majority class); choose k by CV. Details were in Ch03.

### Block 4 — Trees & Forests (7 min)

Tree = yes/no questions chosen for purity (Gini); `plot_tree`; fully interpretable; `random_state` even for a single tree.  
**GIF `tree_depth_sweep.gif`:** depth 1 → 12 on make_moons; train acc → 1.0, test acc peaks around 5–6, boundary turns to confetti — third appearance of the complexity dial.  
Random Forest: many deep trees on random rows + features, majority vote; low variance; feature importances (model-specific).

### Block 5 — Decision Boundaries (3 min)

Image `decision_boundaries_2d.png` / demo §4 with `DecisionBoundaryDisplay.from_estimator` on two same-scale features (worst radius × worst texture): linear (LogReg) · wiggly (KNN) · axis-parallel boxes (tree) · smoothed boxes (forest). More complex ≠ better.

### Block 6 — Reading a Confusion Matrix (4 min)

Image `confusion_matrix_card.png`: rows = truth, columns = prediction, positive class = 1 = what you look for; FN = the missed case. `ConfusionMatrixDisplay.from_predictions`, `classification_report` = precision / recall / F1 per class (multi-class: one row per class, off-diagonal cells show confusions). Ch06 deepens.

### Block 7 — Quick Check (2 min)

FN cell for a missed cancer; why scaling fixes KNN; which model for a regulator (tree / logistic).

### Block 8 — Exercises (12 min core + bonus)

→ `03-exercises/ch05_classification_exercises.ipynb` — `load_wine`: 178 wines, 13 chemical features, **3 grape cultivars** (multi-class).

| Task | Content | Time |
|------|---------|------|
| 1 | `DummyClassifier` + LogReg pipeline | 3 min |
| 2 | KNN pipeline (k = 7) — why does KNN need the scaler and the forest not? | 2 min |
| 3 | Random forest + `classification_report` | 4 min |
| 4 | Confusion matrix of a deliberately weak depth-2 tree (the forest is perfect on 36 test wines → boring matrix) | 3 min |
| Bonus A | 5-fold CV comparison (StratifiedKFold by default for classifiers) — most consistent model? | |
| Bonus B | SVM (RBF) added to the comparison | |

Debrief: which cultivars does the tree confuse (class_1 → class_0)? Why is the CV std large (36 wines per fold → 1 wine = 2.8 %)?

---

## Instructor Notes

- Address the "logistic *regression*" name head-on.
- KNN is a recap — do not re-teach; Ch03 owns it (incl. animation 02).
- All solution text is computed from variables (e.g. "most frequent confusion: …" is derived from the matrix; if the matrix is perfect it says so).
- SVM: appendix only — show if ahead of schedule or on request. Bonus B keeps it available.
- Spiral: complexity dial (Ch03 k → Ch04 degree/α → Ch05 depth) → Ch06 asks *which metric* to optimize.

---

## Materials

- Slides: `01-slides/ch05_slides.md` (images: `sigmoid_curve.png`, `tree_depth_sweep.gif`, `decision_boundaries_2d.png`, `confusion_matrix_card.png`, `svm_margin.png`; generated by `imagegen/ch05.py`)
- Examples: `02-examples/ch05_classification_examples.ipynb`
- Exercises: `03-exercises/ch05_classification_exercises.ipynb`
- Solutions: `04-solutions/ch05_classification_solutions.ipynb`
