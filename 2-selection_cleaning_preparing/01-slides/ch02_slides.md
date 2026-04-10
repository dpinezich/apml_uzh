---
layout: cover
title: "Ch02 — Data Selection, Cleaning & Preparing"
controls: false
fonts:
  sans: Lato
  mono: JetBrains Mono
  weights: '300,400,700,900'
---

# Data Selection, Cleaning & Preparing

**Applied Machine Learning — Session 1, Chapter 2**

---

# The ML Pipeline

![pipeline_overview](/ch02/pipeline_overview.png)

**Garbage in — garbage out.** This chapter = the data cleaning step.

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
```

![missing_values_heatmap](/ch02/missing_values_heatmap.png)

**Rule of thumb:** < 5% missing → drop rows · > 50% in a column → drop column

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

![outlier_boxplot](/ch02/outlier_boxplot.png)

**Detection:** boxplot · IQR rule (`Q1 − 1.5·IQR` / `Q3 + 1.5·IQR`) · Z-score `|z| > 3`

**Treatment:** Remove · Cap/clip · Log-transform · Keep (if real signal)

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

![onehot_encoding](/ch02/onehot_encoding.png)

```python
pd.get_dummies(df, columns=['city'])   # pandas
# or: sklearn OneHotEncoder
```

⚠️ With 100 cities → 100 new columns. Consider target encoding for high-cardinality features.

---

# Feature Scaling

![feature_scaling](/ch02/feature_scaling.png)

**StandardScaler** → mean=0, std=1 · **MinMaxScaler** → [0, 1]

**Tree-based models** (Decision Tree, Random Forest) → no scaling needed.

---

# Train / Test Split

![train_test_split](/ch02/train_test_split.png)

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

**Always set `random_state`** — makes results reproducible.

---

# Data Leakage ⚠️

![data_leakage](/ch02/data_leakage.png)

**Fix:** Use sklearn Pipelines — `fit()` only on train, `transform()` on both.

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([('imputer', SimpleImputer()), ('scaler', StandardScaler())])
pipe.fit(X_train)
X_test_clean = pipe.transform(X_test)
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
