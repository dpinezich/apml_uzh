# Chapter 06 — Metrics & Evaluation

**Session:** 2 | **Chapter:** 3 of 3 | **Duration:** 50 min  
**Audience:** Students who completed Ch04 & Ch05  
**Format:** Slides + live demo notebook (interleaved) + exercises

---

## Learning Objectives

By the end of this chapter, students will be able to:
- Compute and interpret MAE, RMSE (`root_mean_squared_error`), R², and read a residual plot
- Demonstrate the accuracy paradox with a `DummyClassifier` on imbalanced data
- Read a confusion matrix, compute precision / recall / F1 by hand on a small example, and choose the metric that matches the cost of errors
- Move the decision threshold and explain the precision–recall trade-off
- Read ROC / AUC and the PR curve, and know when the PR curve is the better tool (rare positives)
- Handle imbalance: `class_weight='balanced'`, threshold, `StratifiedKFold`, F1/recall/PR-AUC
- Decode `classification_report` macro vs weighted averages
- Compare models with `cross_val_score` using pipelines and the right `scoring`

---

## Timing Breakdown (sum 44 min → ~6 min buffer)

| Block | Content | Time |
|-------|---------|------|
| 1 | Why metrics matter + accuracy paradox demo (Dummy vs LogReg on 98/2 data) | 5 min |
| 2 | Regression metrics + R² + residual plot | 6 min |
| 3 | Confusion matrix + worked spam example (numbers!) | 6 min |
| 4 | Precision / Recall / F1 — when to use which; threshold GIF | 5 min |
| 5 | ROC & AUC; ROC vs PR under imbalance; imbalance toolbox | 6 min |
| 6 | Multi-class averages (1 min) + CV with pipelines and the right scoring | 4 min |
| 7 | Quick-check quiz | 2 min |
| 8 | **Exercises** (core) | **10 min** |
| **Total** | | **44 min** |

Demo notebook `02-examples/ch06_metrics_examples.ipynb` (~12 min end-to-end): Part A in block 2, Part B in block 1, Part C in blocks 3–5, Part D in block 6.

---

## Content Outline

### Block 1 — The Metric Defines Success (5 min)

99 % healthy → "always healthy" = 99 % accuracy, 0 % recall. **Show it:** image `accuracy_paradox.png` / demo Part B on `make_classification(weights=[.98,.02])`: dummy 98 % accuracy / 0 recall; `LogisticRegression(class_weight='balanced')` 90 % accuracy / 94 % recall. The metric is the decision.

### Block 2 — Regression Metrics (6 min)

MAE, MSE, RMSE (`root_mean_squared_error` — `squared=False` was removed in sklearn 1.6), R² (0 = the DummyRegressor, < 0 = broken). Units! RMSE ≥ MAE; the gap flags large errors. Domain-dependent "good" R². Residual plot: random cloud = fine; pattern = missing structure (housing: $500k cap streak).

### Block 3 — Confusion Matrix + Worked Example (6 min)

Image `confusion_matrix_card.png` (positive = 1 = what you look for; our cancer target is flipped so malignant = 1). `tn, fp, fn, tp = confusion_matrix(...).ravel()`.  
Spam matrix TN 90 / FP 5 / FN 3 / TP 2 → accuracy 0.92, **precision 2/7 = 0.29, recall 2/5 = 0.40** — compute with the class. FP (lost email) vs FN (missed cancer): different costs.

### Block 4 — Precision / Recall / F1 + Threshold (5 min)

Table: precision (FP costly, spam), recall (FN costly, screening), F1 (both, imbalance), accuracy (balanced classes). Harmonic mean punishes imbalance (P 1.0, R 0.1 → F1 0.18).  
**GIF `threshold_sweep.gif`:** point moves along ROC and PR while the confusion matrix updates. `(proba >= 0.3).astype(int)`.

### Block 5 — ROC / AUC / PR / Imbalance (6 min)

ROC = TPR vs FPR per threshold; AUC = P(random positive ranked above random negative); threshold-independent → ranking quality, says nothing about the deployed threshold.  
Image `roc_vs_pr_imbalance.png`: ROC looks fine for 97/3 data, PR curve exposes it → rare positives: PR curve, F1/recall, `class_weight='balanced'`, lower threshold, `StratifiedKFold`.

### Block 6 — Multi-class + CV (4 min)

`classification_report`: macro avg (plain mean, imbalance-friendly) vs weighted avg (by support). `scoring='f1_macro'`.  
`cross_val_score(pipeline, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='f1')` — pipeline inside (no leakage), stratified, mean ± std, scoring matches the cost of errors.

### Block 7 — Quick Check (2 min)

Model A (recall .98/precision .20) vs B (.80/.90) for screening vs surgery; AUC 0.99 yet 3 missed cancers → threshold; "accuracy 0.97 on 1 % fraud" → ask for recall/F1 and the dummy.

### Block 8 — Exercises (10 min core + bonus)

→ `03-exercises/ch06_metrics_exercises.ipynb` — models pre-trained; students evaluate and interpret. **Positive class = 1 = malignant (flipped in setup).**

| Task | Content | Time |
|------|---------|------|
| A1 | MAE / RMSE / R² for linear vs forest on diabetes — better? by how much? | 3 min |
| B1 | accuracy / precision / recall / F1 — which matters for screening? | 2 min |
| B2 | confusion matrix — how many cancers missed (FN)? | 2 min |
| B3 | ROC curve + AUC | 3 min |
| Bonus A | threshold 0.5 → 0.3 → 0.1: missed cancers 3 → 1, false alarms 1 → 5 — which would you choose? | |
| Bonus B | residual plot | |
| Bonus C | dummy baseline: 63 % accuracy, 0 % recall | |

---

## Instructor Notes

- Say once, clearly: *positive class = malignant = 1 in every cancer notebook* (sklearn's default is the opposite; the flip is in the setup cell). Every "FN = missed cancer" statement is now literally true.
- Do the spam arithmetic with the class — that is where precision/recall click.
- All conclusions in the notebooks are computed (e.g. threshold table, "missed cancers 3 → 1") — no hard-coded claims.
- Common student error: reporting accuracy on imbalanced data — catch it in the debrief with the dummy comparison.
- Cross-validation was introduced in Ch03 (concept) and practised in Ch04/Ch05; here it is only re-used with the right metric.

---

## Materials

- Slides: `01-slides/ch06_slides.md` (images: `accuracy_paradox.png`, `confusion_matrix_card.png`, `threshold_sweep.gif`, `roc_curve.png`, `roc_vs_pr_imbalance.png`; generated by `imagegen/ch06.py` and `imagegen/ch05.py`)
- Examples: `02-examples/ch06_metrics_examples.ipynb`
- Exercises: `03-exercises/ch06_metrics_exercises.ipynb`
- Solutions: `04-solutions/ch06_metrics_solutions.ipynb`
