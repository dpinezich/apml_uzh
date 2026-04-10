# Chapter 02 — Data Selection, Cleaning & Preparing

**Session:** 1 | **Chapter:** 2 of 3 | **Duration:** 50 min  
**Audience:** Students with basic Python knowledge, new to ML  
**Format:** Slides + Examples + Exercises

---

## Learning Objectives

By the end of this chapter, students will be able to:
- Identify and handle missing values in a dataset
- Detect and treat outliers
- Encode categorical features numerically
- Scale numerical features appropriately
- Split data into training and test sets correctly
- Apply a full preprocessing pipeline to a messy dataset

---

## Timing Breakdown

| Block | Content | Time |
|-------|---------|------|
| 1 | What Makes Good Data? | 5 min |
| 2 | Missing Values | 8 min |
| 3 | Outliers | 5 min |
| 4 | Feature Types & Encoding | 7 min |
| 5 | Feature Scaling | 5 min |
| 6 | Train / Test Split | 5 min |
| 7 | Live example walkthrough | 5 min |
| 8 | **Exercises** | **10 min** |
| 9 | Solution discussion | 5 min |
| **Total** | | **50 min** |

> **Exercise time: 10 minutes** (individual or pair work)

---

## Content Outline

### Block 1 — What Makes Good Data? (5 min)

Real-world data problems (show examples, make students grimace):
- Misspelled categories: "Male", "male", "M", "MALE"
- Missing values: NaN, empty strings, -999, "N/A"
- Wrong data types: age stored as string
- Outliers: age = 999, salary = -1
- Duplicate rows
- Mixed scales: one feature 0–1, another 0–1,000,000

**Key message:** "The quality of your model is bounded by the quality of your data."

---

### Block 2 — Missing Values (8 min)

**Detection:**
```python
df.isnull().sum()
df.isnull().mean()  # proportion of missing
```

**Visualization:**
```python
import seaborn as sns
sns.heatmap(df.isnull(), cbar=False)
```

**Strategies:**
1. **Drop rows** — when missingness is random and rare (<5%)
2. **Drop columns** — when >50% is missing
3. **Impute with mean/median** — for numerical features (median more robust to outliers)
4. **Impute with mode** — for categorical features
5. **Impute with constant** — e.g., "Unknown" for categories
6. **Advanced:** model-based imputation (not in this course)

**Pitfall:** Always impute *after* the train/test split! Otherwise you leak test information into training. (This concept — data leakage — will be reinforced in Chapter 06.)

---

### Block 3 — Outliers (5 min)

**What is an outlier?** A value that is unusually far from the rest of the data.

**Detection methods:**
1. **Visual:** boxplot, histogram
2. **IQR method:** values below Q1 - 1.5×IQR or above Q3 + 1.5×IQR
3. **Z-score:** values with |z| > 3

**Treatment options:**
1. **Remove** — if clearly erroneous (e.g., age = 999)
2. **Cap/Clip** — replace with boundary value (winsorization)
3. **Transform** — log transform reduces impact of large values
4. **Keep** — if the outlier is real and important signal

**Context matters:** A salary of $1M is an outlier in a general population but normal in a CEO dataset.

---

### Block 4 — Feature Types & Encoding (7 min)

ML algorithms work with **numbers**. We need to convert everything to numbers.

**Feature types:**
- **Numerical continuous:** age, salary, temperature → ready to use (after scaling)
- **Numerical discrete:** number of children, room count → usually ready to use
- **Categorical nominal:** color, city, country → no natural order
- **Categorical ordinal:** size (S/M/L/XL), rating (1-5) → has natural order
- **Binary:** yes/no, true/false → encode as 0/1

**Encoding strategies:**

| Type | Strategy | Example |
|------|---------|---------|
| Binary | Label encode 0/1 | male→0, female→1 |
| Ordinal | Label encode with order | S→0, M→1, L→2, XL→3 |
| Nominal (few categories) | One-hot encoding | city → [is_NYC, is_LA, is_CHI] |
| Nominal (many categories) | Target encoding / embeddings | (advanced) |

**One-hot encoding warning:** With 100 cities, you get 100 columns. Be careful!

---

### Block 5 — Feature Scaling (5 min)

Many algorithms are sensitive to the scale of features.  
Example: salary (0–100,000) dominates age (0–100) in distance-based models.

**Two main approaches:**

1. **Standardization (Z-score scaling)** — `StandardScaler`  
   - Transforms to mean=0, std=1  
   - Formula: `(x - mean) / std`  
   - Use when: data roughly follows a normal distribution

2. **Normalization (Min-Max scaling)** — `MinMaxScaler`  
   - Transforms to range [0, 1]  
   - Formula: `(x - min) / (max - min)`  
   - Use when: you need values in a fixed range (e.g., for neural networks)

**Rule of thumb:** StandardScaler is the safer default for most algorithms.

**What NOT to scale:** Tree-based models (Decision Trees, Random Forests) don't need scaling.

---

### Block 6 — Train / Test Split (5 min)

**The golden rule:** Never evaluate a model on data it was trained on.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% for testing
    random_state=42,    # reproducibility
    stratify=y          # keep class proportions (for classification)
)
```

**Typical splits:**
- 80/20 or 70/30 (train/test)
- 60/20/20 (train/validation/test) — for hyperparameter tuning

**Why `random_state`?** Reproducibility. Set it and document it.

**Why `stratify`?** In classification, ensure each split has the same class proportions. Without it, you might get a test set with no minority class examples.

---

### Block 7 — Putting It Together (5 min)

A typical preprocessing workflow with sklearn:

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
```

Pipelines prevent data leakage automatically — they fit on train, transform both train and test.

---

### Block 8 — Exercises (10 min)

→ See `03-exercises/ch02_data_cleaning_exercises.ipynb`

**Exercise overview:**
- Load a messy synthetic dataset
- Task 1: Identify and count missing values (2 min)
- Task 2: Handle missing values (2 min)
- Task 3: Detect and remove outliers (2 min)
- Task 4: Encode categorical features (2 min)
- Task 5: Scale numerical features + train/test split (2 min)

---

## Instructor Notes

- The "data leakage" warning is critical — plant the seed now, reinforce in Ch06
- Use the messy dataset example to make students *feel* the pain of bad data
- One-hot encoding can be shown visually — draw the table on the board
- For scaling: the distance-based model example is very intuitive
- Emphasize: there is no single "right" strategy — always think about the context

---

## Materials

- Slides: `01-slides/ch02_slides.md`
- Examples: `02-examples/ch02_data_cleaning_examples.ipynb`
- Exercises: `03-exercises/ch02_data_cleaning_exercises.ipynb`
- Solutions: `04-solutions/ch02_data_cleaning_solutions.ipynb`
