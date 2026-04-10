# Chapter 05 — Classification Models

**Session:** 2 | **Chapter:** 2 of 3 | **Duration:** 50 min  
**Audience:** Students who completed Ch04  
**Format:** Slides + Examples + Exercises

---

## Learning Objectives

By the end of this chapter, students will be able to:
- Distinguish between binary and multi-class classification
- Apply Logistic Regression, KNN, Decision Tree, Random Forest, and SVM
- Visualize and interpret decision boundaries
- Choose a classification model based on the problem context
- Handle multi-class classification with sklearn

---

## Timing Breakdown

| Block | Content | Time |
|-------|---------|------|
| 1 | Classification: The Task | 5 min |
| 2 | Logistic Regression | 8 min |
| 3 | K-Nearest Neighbors | 7 min |
| 4 | Decision Trees & Random Forests | 8 min |
| 5 | Support Vector Machines | 5 min |
| 6 | Decision Boundaries (visualization) | 5 min |
| 7 | **Exercises** | **12 min** |
| **Total** | | **50 min** |

> **Exercise time: 12 minutes**

---

## Content Outline

### Block 1 — Classification: The Task (5 min)

**Goal:** Predict a discrete category label.

**Binary Classification:** Two classes (yes/no, spam/ham, disease/healthy)
**Multi-class Classification:** More than two classes (iris species, hand-written digit, language)
**Multi-label Classification:** Multiple labels at once (not covered here)

**What models output:**
- Hard prediction: the class label (0, 1, 2, ...)
- Probability: P(y=1 | X) — often more useful

**Key question:** Which class is the model most confident about?

---

### Block 2 — Logistic Regression (8 min)

**Despite the name: this is a classification model.**

**Idea:** Model the *probability* that a sample belongs to class 1.

```
P(y=1 | X) = sigmoid(β₀ + β₁x₁ + ... + βₙxₙ)
```

**The sigmoid function:** Maps any real number to (0, 1)
- σ(z) = 1 / (1 + e^-z)
- σ(0) = 0.5 → decision boundary
- σ(+∞) = 1, σ(-∞) = 0

**Decision:** If P(y=1) > 0.5 → predict class 1, else class 0

**Multi-class:** Use One-vs-Rest (OvR) or Softmax (native multi-class)

**Strengths:** Fast, interpretable, outputs probabilities, good baseline  
**Weaknesses:** Assumes linear decision boundary

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
proba = model.predict_proba(X_test)  # probabilities!
```

---

### Block 3 — K-Nearest Neighbors (7 min)

**The simplest possible idea:** "Look at the k most similar training examples. Predict the majority class."

**Algorithm:**
1. For a new sample x:
2. Find the k training samples closest to x (Euclidean distance)
3. Predict the most common class among those k neighbors

**Hyperparameter k:**
- k=1 → overfitting (every training point is its own neighbor)
- Large k → smoother boundaries, risk of underfitting
- Rule of thumb: try k = √n (n = number of training samples)

**Important:** KNN requires feature scaling! Distance is scale-sensitive.

**Strengths:** Simple, no training phase, naturally multi-class  
**Weaknesses:** Slow at prediction time for large datasets, sensitive to irrelevant features

```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
```

---

### Block 4 — Decision Trees & Random Forests (8 min)

**Decision Tree Classifier:**
- Learns a hierarchy of yes/no questions about features
- Each internal node: a feature threshold test
- Each leaf node: a class prediction
- Splits chosen to maximize class purity (Gini impurity or Information Gain)

**Reading a decision tree:** Each path from root to leaf = a rule

**Random Forest Classifier:**
- Ensemble of decision trees
- Randomness in: which samples (bagging) and which features each tree sees
- Final prediction: majority vote of all trees
- Robust, handles mixed feature types, gives feature importances
- Almost always better than a single decision tree

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

dt = DecisionTreeClassifier(max_depth=3)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
```

---

### Block 5 — Support Vector Machines (5 min)

**Idea:** Find the hyperplane that maximally separates the two classes.

**The margin:** Distance between the hyperplane and the nearest data points (support vectors).
SVM maximizes this margin.

**Kernels (key concept):**
- Linear kernel: linear boundary
- RBF (Radial Basis Function) kernel: non-linear, circular boundaries
- Polynomial kernel

**C parameter:** Controls the trade-off between a wide margin and classifying all training points correctly.

**When to use:** Works well in high-dimensional spaces, effective with clear margin of separation.

```python
from sklearn.svm import SVC
svm = SVC(kernel='rbf', C=1.0, probability=True)
```

---

### Block 6 — Decision Boundaries (5 min)

Visualizing what each model actually learned:
- Logistic Regression → linear boundary
- KNN → irregular, local boundary
- Decision Tree → rectangular regions
- Random Forest → smoother than DT, still non-linear
- SVM (RBF) → smooth, curved boundary

**Key insight:** More complex boundaries are not always better — they can overfit.

---

### Block 7 — Exercises (12 min)

→ See `03-exercises/ch05_classification_exercises.ipynb`

Task: Classify breast cancer tumors as malignant/benign using multiple models and compare results.

---

## Instructor Notes

- Logistic Regression: the name confuses students — address it head-on
- KNN: the "vote among your neighbors" analogy is very intuitive
- Decision boundaries visualization is very powerful — run it live
- Random Forest feature importance: great for building intuition about "what the model uses"
- SVM: keep it at an intuitive level — the math is complex

---

## Materials

- Slides: `01-slides/ch05_slides.md`
- Examples: `02-examples/ch05_classification_examples.ipynb`
- Exercises: `03-exercises/ch05_classification_exercises.ipynb`
- Solutions: `04-solutions/ch05_classification_solutions.ipynb`
