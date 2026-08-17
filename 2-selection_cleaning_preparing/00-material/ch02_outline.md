# Chapter 02 — Data Selection, Cleaning & Preparing

**Session:** 1 | **Chapter:** 2 of 3 | **Duration:** 50 min  
**Audience:** Students with basic Python knowledge, new to ML  
**Format:** Slides + Live demo + Exercises

---

## Learning Objectives

By the end of this chapter, students will be able to:
- Identify and handle missing values, duplicates and impossible values in a dataset
- Detect outliers (boxplot / IQR) and decide whether to clip, remove or keep them
- Encode categorical features (binary, ordinal, one-hot)
- Scale numerical features and know which models need it
- Split data into training and test sets **first**, then fit all preprocessing on the training set only
- Recognise the two kinds of leakage (target leakage vs. train/test leakage)
- Bundle preprocessing in a `ColumnTransformer` + `Pipeline`

---

## Timing Breakdown

| Block | Content | Time |
|-------|---------|------|
| 1 | What can go wrong with data? | 3 min |
| 2 | Missing values (detection, strategies, `SimpleImputer`) | 6 min |
| 3 | Outliers (impossible values vs. real extremes) | 3 min |
| 4 | Feature types & encoding | 5 min |
| 5 | Feature scaling | 3 min |
| 6 | Train/test split + data leakage (KEY, incl. GIF) | 6 min |
| 7 | Quick-check quiz: "which of these steps leak?" | 2 min |
| 8 | Live demo walkthrough | 8 min |
| 9 | **Exercises** (core tasks) | **10 min** |
| **Total** | | **46 min + 4 min buffer** (solution discussion spills into the Ch03 opening if needed) |

---

## Content Outline

### Block 1 — What Can Go Wrong? (3 min)

Misspelled categories, missing values (`NaN`, `-999`, `"N/A"`), wrong dtypes, impossible values (age = 999), duplicates, mixed scales, **and a feature that secretly contains the answer**.

**Key message:** "The quality of your model is bounded by the quality of your data."

### Block 2 — Missing Values (6 min)

Detection: `df.isnull().sum()`, `df.isnull().mean()`, heatmap; also `df.duplicated().sum()`.
Strategies: drop rows (<5 %, random) · drop column (>50 %) · mean / **median** (robust) · mode · constant.
Ask *why* a value is missing (informative missingness).
**Pitfall:** the fill value is a statistic → `SimpleImputer` fit on train only.

### Block 3 — Outliers (3 min)

Two different things: *impossible* value (error → set to NaN and impute) vs. *large but real* value (may be signal → keep or clip). IQR rule with bounds from the training set. Z-score → bonus slide.

### Block 4 — Feature Types & Encoding (5 min)

Numerical continuous / discrete, categorical nominal / ordinal, binary. One-hot for nominal (`OneHotEncoder(handle_unknown='ignore')` learns categories on train; `pd.get_dummies` needs manual column alignment). High cardinality → mention target encoding only.

### Block 5 — Feature Scaling (3 min)

`StandardScaler` (default) vs. `MinMaxScaler`. Needed for distance/gradient-based models (KNN in Ch03!), not for trees. Fit on train.

### Block 6 — Train/Test Split & Leakage (6 min)

`train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)` — `stratify` **only for classification** (raises with a continuous target).
GIF `leakage_impute.gif`: mean over all rows vs. mean over train rows. Image `preprocess_order.png`: wrong lane vs. right lane.
Golden rule: split first → fit on train → transform both. Pipelines make it automatic.

### Block 7 — Quiz (2 min)

Six steps — which leak? (only the pre-split median; dropping a post-outcome column *removes* target leakage).

### Block 8 — Live Demo (8 min)

`02-examples/ch02_data_cleaning_examples.ipynb`: messy student survey → inspection (incl. duplicates) → deterministic fixes → **target leakage (`grade` derived from `score`) → drop** → split → `SimpleImputer` → IQR clip → encoding → scaling → **`ColumnTransformer` + `Pipeline` in 10 lines**. Sections 6–9 can be skimmed if late; section 10 is the one to keep.

### Block 9 — Exercises (10 min core + bonus)

`03-exercises/ch02_data_cleaning_exercises.ipynb` (housing data with duplicate + 9999 m²):
- Task 1 (3 min): missing/duplicates, unique values, lowercase, drop duplicates, 9999 → NaN
- Task 2 (1 min): split (no stratify — regression!)
- Task 3 (3 min): `SimpleImputer` median / most_frequent, fit on train
- Task 4 (3 min): `ColumnTransformer` (scaler + one-hot), fit_transform train / transform test
- Bonus A–D: IQR clip from train · mean vs median · Z-score · correlation heatmap

---

## Instructor Notes

- Say "split first, fit on train, transform both" at least three times.
- Typical exercise errors: `fit_transform` on the test set; `stratify=y` on the price target; forgetting `reset_index` after `drop_duplicates`.
- Differentiation: fast students → Bonus B (mean vs. median imputation, "what if the 9999 were still there?").
- Emphasise there is no single right strategy — context (domain) decides.
- Everything works offline (synthetic data).

---

## Materials

- Slides: `01-slides/ch02_slides.md` (new visuals from `imagegen/ch02.py`: `preprocess_order.png`, `leakage_impute.gif`)
- Examples: `02-examples/ch02_data_cleaning_examples.ipynb`
- Exercises: `03-exercises/ch02_data_cleaning_exercises.ipynb`
- Solutions: `04-solutions/ch02_data_cleaning_solutions.ipynb`
