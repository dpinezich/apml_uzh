# Chapter 06 — Metrics & Evaluation

**Session:** 2 | **Chapter:** 3 of 3 | **Duration:** 50 min  
**Audience:** Students who completed Ch04 & Ch05  
**Format:** Slides + Examples + Exercises

---

## Learning Objectives

By the end of this chapter, students will be able to:
- Compute and interpret regression metrics (MAE, MSE, RMSE, R²)
- Compute and interpret classification metrics (accuracy, precision, recall, F1, AUC)
- Read and interpret a confusion matrix
- Explain why accuracy alone is misleading for imbalanced datasets
- Use cross-validation to get robust performance estimates
- Compare multiple models fairly

---

## Timing Breakdown

| Block | Content | Time |
|-------|---------|------|
| 1 | Why Metrics Matter | 3 min |
| 2 | Regression Metrics | 8 min |
| 3 | Classification Metrics: Confusion Matrix | 8 min |
| 4 | Precision, Recall, F1 | 8 min |
| 5 | ROC Curve & AUC | 7 min |
| 6 | Model Comparison & Cross-Validation | 6 min |
| 7 | **Exercises** | **10 min** |
| **Total** | | **50 min** |

> **Exercise time: 10 minutes**

---

## Content Outline

### Block 1 — Why Metrics Matter (3 min)

**A metric is how you define "success" for your model.**  
Choosing the wrong metric = optimizing for the wrong thing.

**Classic example:**
- Disease detection: 99% of patients are healthy
- A model that always predicts "healthy" has 99% accuracy
- But it catches 0% of sick patients
- **Accuracy is misleading here!**

**Key insight:** Always think about what you're optimizing for *in the real world*.

---

### Block 2 — Regression Metrics (8 min)

Given predictions ŷ and true values y:

**Mean Absolute Error (MAE):**
- MAE = (1/n) × Σ|yᵢ - ŷᵢ|
- Intuitive: average error in the same units as y
- Less sensitive to outliers

**Mean Squared Error (MSE) & RMSE:**
- MSE = (1/n) × Σ(yᵢ - ŷᵢ)²
- RMSE = √MSE (same units as y)
- Penalizes large errors heavily (squaring)
- Useful when large errors are especially costly

**R² Score (Coefficient of Determination):**
- R² = 1 - (Σ(yᵢ - ŷᵢ)²) / (Σ(yᵢ - ȳ)²)
- Range: (-∞, 1.0]. Best = 1.0
- Interpretation: "fraction of variance in y explained by the model"
- R² = 0 means the model is no better than predicting the mean

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
```

---

### Block 3 — Classification Metrics: The Confusion Matrix (8 min)

**For binary classification:**

```
                Predicted: Negative  |  Predicted: Positive
Actual: Negative    True Negative (TN)  |  False Positive (FP)
Actual: Positive    False Negative (FN)  |  True Positive (TP)
```

**Definitions:**
- **True Positive (TP):** Correctly predicted positive
- **True Negative (TN):** Correctly predicted negative
- **False Positive (FP):** Predicted positive, actually negative (Type I error)
- **False Negative (FN):** Predicted negative, actually positive (Type II error)

**Accuracy:** (TP + TN) / (TP + TN + FP + FN) — misleading with imbalanced data

**Real-world stakes:**
- FP in spam filter: important email goes to spam
- FN in disease detection: sick patient gets no treatment
- The costs are very different!

---

### Block 4 — Precision, Recall, F1 (8 min)

**Precision:** Of all predicted positives, what fraction were actually positive?
- Precision = TP / (TP + FP)
- "When I say positive, how often am I right?"
- Optimize when FP is costly (e.g., spam filter → don't block good emails)

**Recall (Sensitivity):** Of all actual positives, what fraction did we catch?
- Recall = TP / (TP + FN)
- "Of all actual positives, how many did I find?"
- Optimize when FN is costly (e.g., cancer detection → don't miss sick patients)

**F1 Score:** Harmonic mean of precision and recall
- F1 = 2 × (Precision × Recall) / (Precision + Recall)
- Good single metric when both precision and recall matter
- Best = 1.0, worst = 0.0

**Precision-Recall trade-off:**
- Increasing one usually decreases the other
- Controlled by the decision threshold (default 0.5)

---

### Block 5 — ROC Curve & AUC (7 min)

**ROC = Receiver Operating Characteristic Curve**

- X-axis: False Positive Rate = FP / (FP + TN)
- Y-axis: True Positive Rate (= Recall) = TP / (TP + FN)
- Each point on the curve = one decision threshold

**AUC = Area Under the ROC Curve**
- Range: [0, 1]
- AUC = 1.0: perfect classifier
- AUC = 0.5: random guessing (diagonal line)
- AUC = 0.0: perfectly wrong

**Why AUC?** It's threshold-independent — measures the overall discriminative power.

```python
from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 1])
auc = roc_auc_score(y_test, y_proba[:, 1])
```

---

### Block 6 — Model Comparison & Cross-Validation (6 min)

**How to compare multiple models fairly:**

1. Use the same train/test split for all models
2. Use cross-validation for more robust estimates
3. Compare the same metric

**Cross-Validation for model selection:**
```python
from sklearn.model_selection import cross_val_score

models = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'RandomForest': RandomForestRegressor()
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f'{name}: {scores.mean():.3f} ± {scores.std():.3f}')
```

**Look at both mean and std:** A model with lower mean but also lower variance might be preferable in production.

---

### Block 7 — Exercises (10 min)

→ See `03-exercises/ch06_metrics_exercises.ipynb`

---

## Instructor Notes

- The confusion matrix: spend time making sure students can read it
- The precision/recall trade-off: use the cancer detection vs. spam filter examples — these are memorable
- AUC: the "area" interpretation is intuitive — more area = better discrimination
- Cross-validation: this is the professional standard — encourage students to always use it
- Common mistake: students report accuracy on imbalanced data — catch this early

---

## Materials

- Slides: `01-slides/ch06_slides.md`
- Examples: `02-examples/ch06_metrics_examples.ipynb`
- Exercises: `03-exercises/ch06_metrics_exercises.ipynb`
- Solutions: `04-solutions/ch06_metrics_solutions.ipynb`
