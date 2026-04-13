---
layout: cover
title: "Ch03 — Introduction to Supervised Learning"
controls: false
fonts:
  sans: Lato
  mono: JetBrains Mono
  weights: '300,400,700,900'
---

# Introduction to Supervised Learning

**Applied Machine Learning — Session 1, Chapter 3**

---

# Supervised Learning

**Learning from labeled examples to predict new cases.**

```
Input:  { (X₁, y₁), (X₂, y₂), ..., (Xₙ, yₙ) }
                  ↓
           Model Training
                  ↓
Output: f(X_new) → ŷ_new
```

- **X** = features (what we know)
- **y** = label / target (what we want to predict)
- **ŷ** = prediction (what the model says)

---

# Regression vs Classification

| Task | Target | Example |
|------|--------|---------|
| **Regression** | Continuous number | House price: €285,000 |
| **Classification** | Discrete category | Spam / Not Spam |

---

# The Learning Process

```
① Collect labeled data
② Choose a model
③ Train: minimize error between ŷ and y
④ Evaluate on held-out data
⑤ Deploy or iterate
```

**Training error** = error on data the model has already seen  
**Test error** = error on new, unseen data ← **what we actually care about**

---

# Loss Functions

**How we measure "how wrong" the model is:**

Regression:
```
MSE = (1/n) Σ(yᵢ - ŷᵢ)²
```

Classification:
```
Cross-Entropy Loss = -Σ yᵢ · log(ŷᵢ)
```

Training = finding parameters that **minimize** the loss.

---

# Underfitting vs Overfitting

![overfit_curves](./overfit_curves.png)

- **Underfitting:** High bias, high train error AND high test error
- **Overfitting:** Low bias, low train error BUT high test error

---

# The Bias-Variance Tradeoff

![bias_variance](./bias_variance.png)

**Goal:** Find the sweet spot where total error is minimized.

---

# Cross-Validation

**Problem:** Single train/test split gives noisy performance estimates.

![cross_val_folds](./cross_val_folds.png)

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)
```

---

# The sklearn API

**One interface to rule them all:**

```python
# 1. Create
model = SomeModel(hyperparameter=value)

# 2. Train
model.fit(X_train, y_train)

# 3. Predict
y_pred = model.predict(X_test)

# 4. Score
model.score(X_test, y_test)
```

Every sklearn model works exactly this way. ✅

---

# Now: Examples!

→ Open `02-examples/ch03_supervised_intro_examples.ipynb`

We'll see overfitting and underfitting live,  
then fix it with cross-validation.

---

# Key Takeaways

- Supervised learning = learning from (X, y) pairs
- Goal = generalize to **new, unseen data** (not memorize training data)
- Underfitting: model too simple
- Overfitting: model memorizes noise
- Cross-validation: robust performance estimation
- sklearn API: fit → predict → score

---
layout: end
---

# Next: Chapter 4

## Regression Models

> _"Time to meet the algorithms. Starting with the classics."_
