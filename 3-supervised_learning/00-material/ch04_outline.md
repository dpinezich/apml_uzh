# Chapter 04 — Regression Models

**Session:** 2 | **Chapter:** 1 of 3 | **Duration:** 50 min  
**Audience:** Students who completed Session 1 (Ch01-03: data prep, train/test split, over/underfitting, KNN, CV as a concept)  
**Format:** Slides + live demo notebook (interleaved) + exercises

---

## Learning Objectives

By the end of this chapter, students will be able to:
- Start every regression task with a `DummyRegressor` baseline and know what R² = 0 means
- Apply linear regression and interpret its coefficients (units!)
- Explain training as loss minimization (MSE) and gradient descent as "step downhill" (intuition only)
- Recognize model complexity (polynomial degree, 1/α, tree depth) as a dial and choose it with **cross-validation**, not training error
- Use `cross_val_score` with `KFold(shuffle=True)` and read mean ± std
- Apply Ridge and Lasso, explain shrinkage vs. selection, and choose α with `RidgeCV` / `GridSearchCV`
- Use Decision Tree and Random Forest regressors and read feature importances

---

## Timing Breakdown (sum 45 min → ~5 min buffer)

| Block | Content | Time |
|-------|---------|------|
| 1 | Regression: the task + baseline (`DummyRegressor`) | 5 min |
| 2 | Linear regression, reading coefficients, MSE loss, gradient-descent hook (GIF) | 8 min |
| 3 | Overfitting revisited: polynomial degree sweep (GIF) → CV in practice (spiral from Ch03) | 7 min |
| 4 | Regularization: Ridge & Lasso, coefficient paths, choosing α with RidgeCV / GridSearchCV | 7 min |
| 5 | Tree & Random Forest regression, model-choice table | 5 min |
| 6 | Quick-check quiz | 2 min |
| 7 | **Exercises** (core) | **10 min** |
| 8 | Debrief | 1 min |
| **Total** | | **45 min** |

The demo notebook (`02-examples/ch04_regression_examples.ipynb`, ~12 min if run end-to-end) is **interleaved**: §1 during block 3, §2–6 during blocks 1–5, §7–8 during block 5.

---

## Content Outline

### Block 1 — Regression: The Task + Baseline (5 min)

**Goal:** predict a continuous number (house price, energy demand, recovery time).

**Metrics (preview, Ch06 details):** MAE (average absolute error, units of y), RMSE (large errors weigh more), R² (share of variance explained; 1 = perfect, 0 = predicting the mean).

**Baseline first:**
```python
from sklearn.dummy import DummyRegressor
DummyRegressor(strategy='mean')      # R² = 0 by definition — every real model must beat it
```

---

### Block 2 — Linear Regression, Loss, Gradient Descent (8 min)

**Model:** ŷ = β₀ + β₁x₁ + … + βₙxₙ  
**Learning:** minimize the MSE loss (ordinary least squares, closed form).

```python
model = LinearRegression().fit(X_train, y_train)
model.coef_, model.intercept_
```

**Reading β:** target in k€, `area_sqm` β = 3.5 → each extra m² adds 3.5 k€ = 3 500 €, all else equal.  
Raw β depends on the feature's unit → standardize (`StandardScaler` in a pipeline) if you want to compare features.  
Scaling does **not** change OLS predictions; it matters for interpretability and for Ridge/Lasso.

**Loss function (moved here from Ch03):** MSE(β) = (1/n) Σ (yᵢ − ŷᵢ)². Training = finding the β at the bottom of the bowl.  
**Gradient descent hook (GIF `gradient_descent_steps.gif`):** w ← w − η ∇L(w). Linear regression is closed-form; logistic regression, boosting, neural nets iterate. Full animation: `0-animations/03_gradient_descent.ipynb`.

---

### Block 3 — Overfitting Revisited → CV in Practice (7 min)

```python
make_pipeline(StandardScaler(), PolynomialFeatures(degree=3), LinearRegression())
```
Scale **before** powering (numerical stability at high degree).

**GIF `poly_degree_sweep.gif`:** degree 1 → 15 on the sine data; train MSE falls monotonically, 5-fold CV MSE has a minimum then explodes.  
Message: *train error always says "more complex"; CV error tells you where to stop.* Full animation: `0-animations/05_polynomial_overfitting.ipynb`.

```python
cv = KFold(n_splits=5, shuffle=True, random_state=42)     # shuffle unless time series
scores = cross_val_score(pipeline, X, y, cv=cv, scoring='r2')
scores.mean(), scores.std()
```
Rules: pass the **pipeline** (scaler re-fit inside each fold → no leakage, Ch02 callback); compare mean ± std — a difference smaller than the std is noise.

---

### Block 4 — Regularization: Ridge & Lasso (7 min)

Penalty added to the loss: Ridge α Σβ² (shrinks smoothly, never exactly 0), Lasso α Σ|β| (sets some β = 0 → feature selection). Needs scaled features.  
Image `ridge_lasso_paths.png`: coefficient paths on the diabetes data as α grows.  
α → 0: plain OLS; α → ∞: all β = 0 = the baseline.

**Choosing α — show it, don't just say it:**
```python
make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 30), cv=5))
GridSearchCV(RandomForestRegressor(), {'max_depth': [3, 5, 10]}, cv=5)
```
Never tune on the test set.

---

### Block 5 — Tree-based Regression (5 min)

Decision tree: split into boxes, predict the mean per box; no scaling; `max_depth` = complexity dial (same role as degree / 1/α).  
Random forest: many trees on random rows + features, average → variance drops; `feature_importances_` (model-specific, not causal).  
Model-choice table (interpretable / non-linear / scaling / outliers) — corrected: OLS needs no scaling for predictions; all squared-error models are hurt by outliers in y.

---

### Block 6 — Quick Check (2 min)

Three questions (train R² 0.99 → ask for CV R²; RidgeCV picks the smallest α → regularization does not help; 0.481 ± 0.085 vs 0.478 ± 0.083 → not distinguishable). Answers via click.

---

### Block 7 — Exercises (10 min core + bonus)

→ `03-exercises/ch04_regression_exercises.ipynb` — diabetes progression (442 × 10). A `score()` helper is given.

| Task | Content | Time |
|------|---------|------|
| 1 | `DummyRegressor` + linear-regression pipeline | 3 min |
| 2 | Ridge vs Lasso (α = 1), which features does Lasso drop? | 3 min |
| 3 | Random forest + feature importances, compare with linear β | 4 min |
| Bonus A | 5-fold CV comparison (`KFold(shuffle=True)`) — result: linear ≈ RF, all within noise | |
| Bonus B | `RidgeCV` picks α | |
| Bonus C | predicted-vs-actual plot | |

Expected outcome / debrief: on this small, near-linear dataset the forest does **not** beat linear regression; CV std (≈0.08) dwarfs the differences.

---

## Instructor Notes

- Ask students to interpret one coefficient in plain language, with units — the 3.5 vs 3 500 confusion is real.
- The demo's Ridge sweep uses α up to 10⁵ on 16k rows on purpose: with α = 1 nothing visible happens — regularization matters when data is scarce (exercise: 353 rows).
- Spiral: overfitting (Ch03, pictures) → quantified here with CV → the same complexity dial returns in Ch05 (k, depth) and Ch06 (which metric to optimize).
- Everything printed in the notebooks is computed from variables — no hard-coded "model X wins" text; safe if numbers shift.
- Emphasize the habit: baseline row first in every results table.

---

## Materials

- Slides: `01-slides/ch04_slides.md` (images: `baseline_regression.png`, `linear_reg_fit.png`, `gradient_descent_steps.gif`, `poly_degree_sweep.gif`, `ridge_lasso_paths.png`; generated by `imagegen/ch04.py`)
- Examples: `02-examples/ch04_regression_examples.ipynb`
- Exercises: `03-exercises/ch04_regression_exercises.ipynb`
- Solutions: `04-solutions/ch04_regression_solutions.ipynb`
- Animations: `0-animations/03_gradient_descent.ipynb`, `0-animations/05_polynomial_overfitting.ipynb`
