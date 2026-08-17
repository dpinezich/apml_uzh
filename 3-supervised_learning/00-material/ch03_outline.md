# Chapter 03 — Introduction to Supervised Learning

**Session:** 1 | **Chapter:** 3 of 3 | **Duration:** 50 min  
**Audience:** Students who completed Ch01 & Ch02  
**Format:** Slides + Live demo + Exercises — **red thread: K-Nearest Neighbours**

---

## Learning Objectives

By the end of this chapter, students will be able to:
- Explain supervised learning: features, label, prediction; regression vs. classification
- Apply the protocol split → baseline → fit on train → score on test
- Explain generalisation, underfitting and overfitting and read them off a train/test accuracy plot
- Describe how KNN predicts (distance, k nearest, majority vote) and why it needs scaled features
- Distinguish hyperparameters (chosen) from parameters (learned)
- Use the sklearn API (`Pipeline`, `fit`, `predict`, `score`) with a `DummyClassifier` baseline
- Know that cross-validation exists and what it is for (used in Ch04)

---

## Timing Breakdown

| Block | Content | Time |
|-------|---------|------|
| 1 | What is supervised learning? (X, y, ŷ; regression vs classification) | 3 min |
| 2 | Protocol: train/test, baseline, generalisation | 4 min |
| 3 | Underfitting vs overfitting, bias–variance intuition | 6 min |
| 4 | KNN: algorithm, effect of k (GIF), scaling | 8 min |
| 5 | Hyperparameter vs parameter · sklearn API | 4 min |
| 6 | Cross-validation — concept preview only | 1 min |
| 7 | Quick-check quiz | 2 min |
| 8 | Live demo | 10 min |
| 9 | **Exercises** (core) | **10 min** |
| **Total** | | **48 min** (buffer: skip CV preview / shorten demo to 8 min) |

---

## Content Outline

### Block 1 — What Is Supervised Learning? (3 min)
Features X, label y, prediction ŷ. Regression (continuous) vs. classification (discrete). Today: classification.

### Block 2 — The Protocol (4 min)
① split (Ch02) → ② baseline (`DummyClassifier(strategy='most_frequent')`) → ③ `fit` on train → ④ `score` on test. Training accuracy ≠ generalisation.

### Block 3 — Under-/Overfitting (6 min)
`overfit_curves.png`, `bias_variance.png`. Underfitting: too simple, train & test poor. Overfitting: memorises noise, train ≫ test. Complexity axis for KNN = 1/k.

### Block 4 — KNN (8 min)
`knn_vote.png`: distance → k nearest → majority vote; no training, stores the data.
`knn_boundary_k.gif` (make_moons, k = 1 → 100, train/test accuracy live): k = 1 islands (overfit), moderate k smooth, k = 100 straight line (underfit). Static fallback `knn_k_sweep.png`.
`knn_scaling.png`: unscaled income dominates age → wrong neighbours → `StandardScaler` in the pipeline.

### Block 5 — Hyperparameter vs Parameter · sklearn API (4 min)
`hyperparam_vs_param.png`. `Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(k))])` → `.fit / .predict / .score`. Same interface for every model.

### Block 6 — CV Preview (1 min)
`cross_val_folds.png`: one split = one noisy number; k-fold averages. Used in Ch04 to choose hyperparameters. Not implemented today.

### Block 7 — Quiz (2 min)
100 % train / 72 % test / baseline 65 % → overfitting, increase k, never report training accuracy.

### Block 8 — Live Demo (10 min)
`02-examples/ch03_supervised_intro_examples.ipynb`: make_moons → split → baseline → pipeline KNN k=5 → boundaries k = 1/7/51 → k sweep table + plot (conclusions printed from the numbers) → Penguins with/without scaler (0.77 vs 0.99) → CV taste (3 lines, shuffled StratifiedKFold).

### Block 9 — Exercises (10 min core + bonus)
`03-exercises/ch03_supervised_intro_exercises.ipynb` (make_moons, seed 7): Task 1 split · Task 2 baseline + pipeline KNN · Task 3 k sweep table · Task 4 plot (scaffold given) + interpretation. Bonus: Penguins pipeline with `SimpleImputer` (Ch02) · scaler on/off · CV taste.

---

## Instructor Notes

- ONE algorithm all the way (slide → GIF → demo → exercise → animation notebook) — do not introduce a second one today.
- The exercise must show overfitting at k = 1: make_moons noise 0.3 gives train 1.00 vs test ≈ 0.87.
- Baseline every time — students should ask "better than what?" reflexively by Ch06.
- Loss functions and polynomial overfitting are deliberately NOT here (→ Ch04); if asked, point to the bonus slide.
- Animation notebook: `0-animations/02_knn_decision_boundary.ipynb` (same data as the GIF, interactive in Jupyter).

---

## Materials

- Slides: `01-slides/ch03_slides.md` (new visuals from `imagegen/ch03.py`: `knn_vote.png`, `knn_boundary_k.gif/.png`, `knn_k_sweep.png`, `knn_scaling.png`, `hyperparam_vs_param.png`)
- Examples: `02-examples/ch03_supervised_intro_examples.ipynb`
- Exercises: `03-exercises/ch03_supervised_intro_exercises.ipynb`
- Solutions: `04-solutions/ch03_supervised_intro_solutions.ipynb`
- Animation: `0-animations/02_knn_decision_boundary.ipynb`
