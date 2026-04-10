# Chapter 04 — Regression Models

**Session:** 2 | **Chapter:** 1 of 3 | **Duration:** 50 min  
**Audience:** Students who completed Session 1 (Ch01-03)  
**Format:** Slides + Examples + Exercises

---

## Learning Objectives

By the end of this chapter, students will be able to:
- Apply linear regression and understand what it learns
- Understand polynomial regression as an extension
- Apply Ridge and Lasso regularization and explain why they help
- Use Decision Tree and Random Forest regressors
- Choose an appropriate model for a regression task

---

## Timing Breakdown

| Block | Content | Time |
|-------|---------|------|
| 1 | Regression: The Task | 5 min |
| 2 | Linear Regression | 10 min |
| 3 | Polynomial Regression | 7 min |
| 4 | Regularization: Ridge & Lasso | 8 min |
| 5 | Tree-based Regression | 5 min |
| 6 | Live example | 5 min |
| 7 | **Exercises** | **10 min** |
| **Total** | | **50 min** |

> **Exercise time: 10 minutes**

---

## Content Outline

### Block 1 — Regression: The Task (5 min)

**Goal:** Predict a continuous numerical output.

**Examples:**
- Predict house price from size, location, rooms
- Predict temperature from historical weather data
- Predict exam score from study hours

**Key metric (preview — details in Ch06):**
- Mean Absolute Error (MAE): average absolute distance between prediction and truth
- Root Mean Squared Error (RMSE): penalizes large errors more heavily
- R² Score: fraction of variance explained (1.0 = perfect)

---

### Block 2 — Linear Regression (10 min)

**The simplest and most interpretable regression model.**

**Model:** ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ

- β₀ = intercept (bias)
- β₁...βₙ = coefficients (weights)
- Each βᵢ tells us: "if xᵢ increases by 1, ŷ changes by βᵢ"

**Learning:** Find the βs that minimize the sum of squared residuals (Ordinary Least Squares).

**Assumptions:**
- Linear relationship between features and target
- No extreme multicollinearity
- Residuals roughly normally distributed

**Strengths:** Interpretable, fast, strong baseline  
**Weaknesses:** Assumes linearity, sensitive to outliers

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
print(model.coef_)       # feature coefficients
print(model.intercept_)  # bias term
```

---

### Block 3 — Polynomial Regression (7 min)

**Problem:** What if the relationship is non-linear?

**Solution:** Add polynomial features (x², x³, x₁·x₂, ...) and then fit a linear model on them.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

poly_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())
])
```

**Warning:** High degree → overfitting. Polynomial features grow fast (degree 3 with 5 features → many new columns).

**Rule of thumb:** Degree 2 is often enough. Always cross-validate.

---

### Block 4 — Regularization: Ridge & Lasso (8 min)

**Problem:** Linear models with many features can overfit.

**Solution:** Add a penalty term to the loss function that discourages large coefficients.

**Ridge (L2 regularization):**
- Adds penalty: α × Σβᵢ²
- Shrinks all coefficients toward zero (but rarely to exactly zero)
- Good when all features are somewhat relevant

**Lasso (L1 regularization):**
- Adds penalty: α × Σ|βᵢ|
- Can shrink coefficients to *exactly* zero → automatic feature selection
- Good when you suspect many irrelevant features

**α (alpha):** The regularization strength. Larger α → more shrinkage.

```python
from sklearn.linear_model import Ridge, Lasso
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1)
```

**ElasticNet:** Combines Ridge and Lasso (not in exercises, but worth mentioning).

---

### Block 5 — Tree-based Regression (5 min)

**Decision Tree Regressor:**
- Splits the feature space into rectangular regions
- Predicts the mean of training samples in each region
- No need for feature scaling
- Can capture non-linear relationships and interactions
- Prone to overfitting → needs pruning (`max_depth` parameter)

**Random Forest Regressor:**
- Ensemble of decision trees
- Each tree trained on a random subset of data and features
- Final prediction = average of all trees
- Much more robust than a single tree
- Key hyperparameter: `n_estimators` (number of trees)

```python
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
```

---

### Block 6 — Live Example (5 min)

→ See `02-examples/ch04_regression_examples.ipynb`

Show: house price prediction with all 4 models, comparing their predictions visually.

---

### Block 7 — Exercises (10 min)

→ See `03-exercises/ch04_regression_exercises.ipynb`

---

## Instructor Notes

- Linear regression coefficients: make students interpret them in plain language
- Regularization: the "penalty" analogy — you want simple explanations, not overly complex ones
- Random Forest: emphasize it's a "crowd wisdom" model — many weak learners together
- Feature importance from Random Forest is a nice bonus to show

---

## Materials

- Slides: `01-slides/ch04_slides.md`
- Examples: `02-examples/ch04_regression_examples.ipynb`
- Exercises: `03-exercises/ch04_regression_exercises.ipynb`
- Solutions: `04-solutions/ch04_regression_solutions.ipynb`
