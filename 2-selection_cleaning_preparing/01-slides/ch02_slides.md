---
layout: cover
title: "Ch02 — Data Selection, Cleaning & Preparing"
---

# Data Selection, Cleaning & Preparing

**Applied Machine Learning — Session 1, Chapter 2**

---

# Why Data Cleaning Matters

> "Data scientists spend 80% of their time cleaning data and 20% complaining about it."
> — Everyone in data science

- Real data is **messy by default**
- A great model on bad data → bad results
- A simple model on clean data → good results

**The quality of your model is bounded by the quality of your data.**

---

# What Can Go Wrong?

- Missing values (`NaN`, empty strings, `-999`)
- Wrong data types (age as string)
- Inconsistent categories (`"Male"`, `"male"`, `"M"`)
- Outliers (age = `999`, salary = `-1`)
- Duplicate rows
- Mixed scales (salary in thousands vs. age in years)

---

# Missing Values — Detection

```python
df.isnull().sum()          # count per column
df.isnull().mean()         # proportion per column

import seaborn as sns
sns.heatmap(df.isnull(), cbar=False)  # visual overview
```

**Rule of thumb:**
- < 5% missing → usually safe to drop rows
- > 50% missing in a column → consider dropping the column

---

# Missing Values — Strategies

| Strategy | When to use |
|---------|------------|
| Drop rows | Few missing, random missingness |
| Drop column | More than 50% missing |
| Fill with **mean** | Numerical, no outliers |
| Fill with **median** | Numerical, with outliers |
| Fill with **mode** | Categorical |
| Fill with constant | Domain knowledge (e.g. "Unknown") |

⚠️ **Always impute AFTER train/test split** — otherwise data leakage!

---

# Outliers

An outlier = a value unusually far from the rest.

**Detection:**
- Visual: boxplot, histogram
- IQR: values outside `[Q1 - 1.5·IQR, Q3 + 1.5·IQR]`
- Z-score: values with `|z| > 3`

**Treatment:**
- Remove (if clearly wrong)
- Cap / clip to boundary
- Log-transform (shrinks large values)
- Keep (if it's real signal!)

---

# Feature Types

| Type | Example | Ready for ML? |
|------|---------|--------------|
| Numerical continuous | Age, salary | After scaling |
| Numerical discrete | # children | Usually yes |
| Categorical nominal | City, color | Need encoding |
| Categorical ordinal | S / M / L / XL | Need ordered encoding |
| Binary | Yes / No | Encode as 0 / 1 |

---

# Encoding Categorical Features

**One-Hot Encoding** (for nominal categories)
```
City:  [NYC, LA, CHI]
→ is_NYC  is_LA  is_CHI
     1       0      0     ← NYC
     0       1      0     ← LA
     0       0      1     ← CHI
```

```python
pd.get_dummies(df, columns=['city'])
# or: sklearn OneHotEncoder
```

⚠️ With 100 cities → 100 new columns. Consider alternatives for high-cardinality features.

---

# Feature Scaling

Many algorithms are **scale-sensitive** (KNN, SVM, Linear models, Neural Nets).

**StandardScaler** → mean=0, std=1
```python
x_scaled = (x - mean) / std
```

**MinMaxScaler** → range [0, 1]
```python
x_scaled = (x - min) / (max - min)
```

**Tree-based models** (Decision Tree, Random Forest) → no scaling needed.

---

# Train / Test Split

**Golden rule:** Never evaluate on data you trained on.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,    # 80% train, 20% test
    random_state=42,  # for reproducibility
    stratify=y        # keep class balance
)
```

**Always set `random_state`** — makes results reproducible.

---

# Data Leakage ⚠️

**What is it?** When information from the test set "leaks" into the training process.

**Example:** Computing the mean on the full dataset, then imputing.  
→ The imputed training values were influenced by test set statistics.

**Fix:** Use sklearn Pipelines — they fit only on train, then apply to test.

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([('imputer', SimpleImputer()), ('scaler', StandardScaler())])
pipe.fit(X_train)        # fit only on train
X_test_clean = pipe.transform(X_test)   # apply to test
```

---

# The Preprocessing Checklist

```
✅ Inspect shape, dtypes, head
✅ Check missing values
✅ Handle missing values (impute or drop)
✅ Detect and treat outliers
✅ Encode categorical features
✅ Scale numerical features
✅ Train / test split (LAST!)
```

---

# Now: Exercises!

→ Open `03-exercises/ch02_data_cleaning_exercises.ipynb`

**You will:**
- Work with a messy dataset
- Apply all today's techniques step by step
- ~10 minutes

---

# Key Takeaways

- Real data is always messy — cleaning is non-negotiable
- Impute missing values (after the split!)
- Encode all categories to numbers
- Scale features when using distance or gradient-based algorithms
- **Always split first, then preprocess**

---
layout: end
---

# Next: Chapter 3

## Introduction to Supervised Learning

> _"Now that our data is clean, let's teach a machine to learn from it."_
