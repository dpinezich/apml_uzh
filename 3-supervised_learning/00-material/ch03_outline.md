# Chapter 03 — Introduction to Supervised Learning

**Session:** 1 | **Chapter:** 3 of 3 | **Duration:** 50 min  
**Audience:** Students who completed Ch01 & Ch02  
**Format:** Slides + Examples + Exercises

---

## Learning Objectives

By the end of this chapter, students will be able to:
- Explain what supervised learning is and when to use it
- Understand the concepts of features, labels, and predictions
- Describe overfitting, underfitting, and the bias-variance tradeoff
- Use scikit-learn's fit/predict/score API
- Perform k-fold cross-validation
- Recognize the difference between training error and generalization error

---

## Timing Breakdown

| Block | Content | Time |
|-------|---------|------|
| 1 | What Is Supervised Learning? | 8 min |
| 2 | The Learning Process | 7 min |
| 3 | Overfitting & Underfitting | 8 min |
| 4 | Cross-Validation | 5 min |
| 5 | The sklearn API | 5 min |
| 6 | Live example walkthrough | 7 min |
| 7 | **Exercises** | **10 min** |
| **Total** | | **50 min** |

> **Exercise time: 10 minutes**

---

## Content Outline

### Block 1 — What Is Supervised Learning? (8 min)

**Definition:** Learning from labeled examples to make predictions on new, unseen data.

**The key components:**
- **Features (X):** The input variables — what we know about each example
- **Label / Target (y):** What we want to predict — the "right answer"
- **Model:** A function learned from data that maps X → y
- **Prediction (ŷ):** The model's answer for a new, unseen X

**Two flavors:**
- **Regression:** y is continuous (house price, temperature, stock price)
- **Classification:** y is discrete (spam/not spam, disease category, iris species)

**When to use supervised learning:**
- You have labeled historical data
- You want to automate decisions or predictions
- You can define a clear "right answer"

---

### Block 2 — The Learning Process (7 min)

The supervised learning loop:

```
1. Collect labeled data: {(X₁, y₁), (X₂, y₂), ..., (Xₙ, yₙ)}
2. Choose a model architecture
3. Train: minimize the difference between ŷ and y
4. Evaluate on held-out data
5. Deploy (or iterate)
```

**Loss functions — how we measure error:**
- Regression: Mean Squared Error (MSE) = average of (yᵢ - ŷᵢ)²
- Classification: Cross-Entropy Loss (log loss)

**Training is optimization:** The algorithm finds model parameters that minimize the loss on training data.

**Key distinction:**
- Training error = error on data the model was trained on (always low)
- Test/generalization error = error on data the model has *never seen* (what we care about)

---

### Block 3 — Overfitting & Underfitting (8 min)

This is the central challenge of machine learning.

**Underfitting (High Bias):**
- Model is too simple to capture the patterns in the data
- Both training error and test error are high
- Solution: more complex model, more features

**Overfitting (High Variance):**
- Model memorizes the training data, including noise
- Training error is low, but test error is high
- Model fails to generalize
- Solution: simpler model, more data, regularization

**The Bias-Variance Tradeoff:**
```
Total Error = Bias² + Variance + Irreducible Noise
```
- More complex model → lower bias, higher variance
- Simpler model → higher bias, lower variance
- Goal: find the sweet spot

**Visual:** Draw the classic U-shaped test error curve vs. model complexity

---

### Block 4 — Cross-Validation (5 min)

**Problem:** With a single train/test split, our estimate of model performance is noisy. It depends on which samples ended up in the test set.

**Solution: k-Fold Cross-Validation**

1. Split data into k equal "folds" (usually k=5 or k=10)
2. For each fold i:
   - Train on all folds except fold i
   - Evaluate on fold i
3. Average the k evaluation scores

**Benefits:**
- More robust estimate of performance
- Uses all data for both training and evaluation
- Default: use `cross_val_score(model, X, y, cv=5)` in sklearn

**When to use:** Always when selecting models or hyperparameters.

---

### Block 5 — The sklearn API (5 min)

Every sklearn model follows the same 3-method interface:

```python
# 1. Create the model
model = SomeModel(hyperparameter=value)

# 2. Train the model
model.fit(X_train, y_train)

# 3. Predict
y_pred = model.predict(X_test)

# 4. Evaluate
score = model.score(X_test, y_test)
```

**Why this matters:** Once you learn this pattern with one model, you know how to use all of them.

---

### Block 6 — Live Example (7 min)

→ See `02-examples/ch03_supervised_intro_examples.ipynb`

Demonstrate with a simple regression problem:
1. Generate data with a known pattern (+ noise)
2. Fit models of increasing complexity
3. Visualize underfitting → good fit → overfitting
4. Show cross-validation in action

---

### Block 7 — Exercises (10 min)

→ See `03-exercises/ch03_supervised_intro_exercises.ipynb`

---

## Instructor Notes

- The bias-variance tradeoff is fundamental — spend time here
- The sklearn API consistency is a big selling point — emphasize it
- Cross-validation: use the analogy of exam preparation — studying 80% then testing on 20% each time
- Emphasize: the goal is always good *generalization*, not memorization

---

## Materials

- Slides: `01-slides/ch03_slides.md`
- Examples: `02-examples/ch03_supervised_intro_examples.ipynb`
- Exercises: `03-exercises/ch03_supervised_intro_exercises.ipynb`
- Solutions: `04-solutions/ch03_supervised_intro_solutions.ipynb`
