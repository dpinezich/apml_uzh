---
layout: cover
title: "Ch12 — Capstone: End-to-End ML Workflow"
---

# Chapter 12
## Capstone: End-to-End ML Workflow

**Session 4 | Final Chapter**
Putting it all together

---

## Today's Mission

April 15, 1912.
RMS Titanic strikes an iceberg.
1,502 of 2,224 passengers die.

**Can we predict who survived?**

---

## The Full ML Workflow

```
① Define the Problem
② Load & Explore (EDA)
③ Clean & Preprocess
④ Feature Engineering
⑤ Train Multiple Models
⑥ Evaluate & Compare
⑦ Interpret Results
⑧ Reflect & Conclude
```

---

## The Dataset

| Feature | Type | Notes |
|---------|------|-------|
| `Pclass` | ordinal | 1st, 2nd, 3rd class |
| `Sex` | categorical | gender |
| `Age` | numeric | **many missing** |
| `SibSp` | numeric | siblings/spouses aboard |
| `Parch` | numeric | parents/children aboard |
| `Fare` | numeric | ticket price |
| `Embarked` | categorical | C, Q, S |
| `Cabin` | text | **mostly missing** |

**Target:** `Survived` (0 = No, 1 = Yes)

---

## What Does "Success" Mean?

- We want to **correctly identify survivors** → high recall
- We also want **predictions to be reliable** → high precision
- **F1-score** balances both

A False Negative here = predicted dead, actually survived

---

## Step 1 — Explore

- `df.shape`, `df.dtypes`, `df.isnull().sum()`
- Survival rate by gender, class, age
- **The "women and children first" signal is in the data**

---

## Step 2 — Clean

- `Age` → impute with median **per class** (not global median!)
- `Embarked` → fill with mode (only 2 missing)
- Drop: `Cabin` (too sparse), `Name`, `Ticket`, `PassengerId`
- Encode: `Sex` → 0/1, `Embarked` → one-hot

---

## Step 3 — Feature Engineering

```python
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
```

**Why?** Solo travelers had different survival odds
**Domain knowledge + creativity = better features**

---

## Step 4 — Train Multiple Models

| Model | Idea |
|-------|------|
| Logistic Regression | Linear boundary in probability space |
| Random Forest | Ensemble of decision trees |
| Gradient Boosting | Sequential error correction |

Cross-validate all → compare F1 + accuracy

---

## Step 5 — Interpret

- Which features matter most? (Random Forest importances)
- Confusion matrix: where does the model fail?
- What does Sex importance tell us about history?

---

## What We've Covered

| Chapter | Topic |
|---------|-------|
| Ch01 | Data Science workflow — **this is it** |
| Ch02 | Cleaning and encoding — **most of the work** |
| Ch03–06 | Fit / predict / evaluate / cross-validate |
| Ch07–09 | Clustering and dimensionality reduction |
| Ch10–11 | Reinforcement learning — a different paradigm |

---

## You Can Now

- Load and explore **any** dataset
- Clean and preprocess data correctly
- Train, evaluate, and compare ML models
- Understand clustering and dimensionality reduction
- Know what reinforcement learning is

---

## What Comes Next

- **Deep Learning** — neural networks
- **Hyperparameter tuning** — GridSearchCV, Optuna
- **Model deployment** — serving predictions in production
- **MLOps** — monitoring, retraining, pipelines

---

## Keep Going

- **Kaggle** — real competitions, real data
- **Fast.ai** — practical deep learning
- **"Hands-On ML"** — Aurélien Géron (the book)

---
layout: end
---

# Now — open the notebook

`03-exercises/ch12_capstone_exercises.ipynb`

**35 minutes. Full workflow. You've got this.**
