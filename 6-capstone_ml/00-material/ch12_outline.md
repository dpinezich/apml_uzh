# Chapter 12 — Capstone: End-to-End ML Workflow

**Session:** 4 | **Chapter:** 3 of 3 | **Duration:** 50 min  
**Format:** Guided end-to-end project (minimal slides, maximum coding)

---

## Learning Objectives

By the end of this chapter, students will be able to:
- Execute a complete, professional ML workflow independently
- Make deliberate design choices at each stage with justification
- Critically evaluate model results and communicate findings
- Recognize the connections between all course topics
- See ML as a workflow, not a collection of disconnected algorithms

---

## Timing Breakdown

| Block | Content | Time |
|-------|---------|------|
| 1 | Brief framing + dataset intro | 5 min |
| 2 | **Guided capstone notebook** (step by step) | 35 min |
| 3 | Model comparison + reflection | 5 min |
| 4 | Course wrap-up + Q&A | 5 min |
| **Total** | | **50 min** |

> **Exercise time: 35 minutes** — this is primarily a hands-on session.

---

## The Project

**Dataset:** Titanic Survival Prediction  
**Task:** Binary classification — predict who survived the Titanic disaster  
**Why Titanic?**
- Iconic dataset, real historical stakes
- Mix of numerical and categorical features
- Real messiness (missing values, different types)
- Interesting: survival not just about luck — social class, gender, age matter
- Good challenge: ~80% accuracy is achievable

**Full workflow steps:**

```
① Define the Problem
② Load & Explore Data (EDA)
③ Clean & Preprocess
④ Feature Engineering
⑤ Train Multiple Models
⑥ Evaluate & Compare
⑦ Interpret Results
⑧ Reflect & Conclude
```

---

## Content Outline

### Block 1 — Framing (5 min)

**The problem:**  
April 15, 1912. RMS Titanic sinks. 1,502 out of 2,224 passengers die.  
Can we predict who survived based on passenger characteristics?

**Features available:**
- `PassengerId`: passenger ID (not predictive)
- `Pclass`: ticket class (1st, 2nd, 3rd) — socioeconomic proxy
- `Name`: passenger name (can extract title)
- `Sex`: gender
- `Age`: age in years (many missing!)
- `SibSp`: siblings/spouses aboard
- `Parch`: parents/children aboard
- `Ticket`: ticket number (complex, ignore for now)
- `Fare`: ticket fare
- `Cabin`: cabin number (many missing!)
- `Embarked`: port of embarkation (C, Q, S)

**Target:** `Survived` (0 = No, 1 = Yes)

**Important:** Discuss what "success" means here. We want high recall for actual survivors AND good precision. F1-score is a good target metric.

---

### Block 2 — Guided Capstone (35 min)

→ See `03-exercises/ch12_capstone_exercises.ipynb`

**Step 1: Load and Explore (5 min)**
- Load the dataset
- Check shape, dtypes, missing values
- Visualize survival rate by gender, class, age

**Step 2: Clean & Preprocess (8 min)**
- Handle missing `Age` (impute with median by class)
- Handle missing `Embarked` (fill with mode)
- Drop `Cabin` (too many missing), `Name`, `Ticket`, `PassengerId`
- Encode: `Sex` → binary, `Embarked` → one-hot

**Step 3: Feature Engineering (5 min)**
- Create `FamilySize` = SibSp + Parch + 1
- Create `IsAlone` = 1 if FamilySize == 1
- (Optional: extract Title from Name)

**Step 4: Train & Evaluate Multiple Models (12 min)**
- Logistic Regression (baseline)
- Random Forest
- Gradient Boosting (briefly)
- Cross-validate all, compare F1 and accuracy

**Step 5: Interpret (5 min)**
- Feature importances from Random Forest
- What characteristics determined survival?
- Confusion matrix interpretation

---

### Block 3 — Reflection (5 min)

Discussion questions:
- Which preprocessing step was most important?
- Why does `Sex` have such high importance?
- What does a False Negative mean in this context (predicted "died" when actually survived)?
- If you had more time, what would you try next?

**Course connections:**
- Ch01: This IS the DS workflow we introduced on Day 1
- Ch02: We spent the most time here (cleaning and encoding)
- Ch03-06: We used fit/predict/score, cross-validation, multiple metrics
- Ch07-09: Could use PCA/clustering for further exploration
- Ch10-11: RL was a different paradigm — autonomous decision making

---

### Block 4 — Course Wrap-Up (5 min)

**What you can do now:**
- Load and explore any dataset
- Clean and preprocess data correctly
- Train, evaluate, and compare ML models
- Understand clustering and dimensionality reduction
- Know what reinforcement learning is and how Q-Learning works

**What comes next:**
- Deep Learning (neural networks)
- Advanced feature engineering
- Hyperparameter tuning (GridSearchCV, Optuna)
- Model deployment and serving
- MLOps and production systems

**Recommended next steps:**
- Kaggle competitions (practice on real problems)
- Fast.ai / deeplearning.ai courses
- Read "Hands-On Machine Learning" by Aurélien Géron

---

## Instructor Notes

- The Titanic context creates real engagement — the "women and children first" rule shows in the data
- Let students discover the gender + class insight themselves through visualization
- Resist the urge to code everything for them — let them try, then show the solution
- Feature engineering is where creativity meets domain knowledge — discuss why FamilySize matters
- The wrap-up is emotionally important — students should leave feeling capable and excited

---

## Materials

- Slides: `01-slides/ch12_slides.md`
- Exercises: `03-exercises/ch12_capstone_exercises.ipynb`
- Solutions: `04-solutions/ch12_capstone_solutions.ipynb`
