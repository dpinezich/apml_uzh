# Chapter 09 — Dimensionality Reduction

**Session:** 3 | **Chapter:** 3 of 3 | **Duration:** 50 min  
**Format:** Slides + Examples + Exercises

---

## Learning Objectives

By the end of this chapter, students will be able to:
- Explain the curse of dimensionality (distances concentrate) and why it motivates dimensionality reduction
- Explain PCA as "new axes ordered by variance" and read `explained_variance_ratio_` and `components_`
- State that PCA ≠ feature selection (each PC mixes all features)
- Distinguish the two regimes: 2 PCs for a picture (any %) vs ≈95 % for preprocessing
- Use PCA inside a Pipeline (scaler → PCA → model) evaluated with CV, without leakage
- Explain what t-SNE shows and does not show, and why it cannot be used as preprocessing (no `transform`)

---

## Timing Breakdown (planned 45 min)

| Block | Content | Time |
|-------|---------|------|
| 1 | Curse of dimensionality (`curse_dimensionality.gif`) | 5 min |
| 2 | PCA intuition (`pca_rotation.gif`) + code in a Pipeline | 8 min |
| 3 | PCA ≠ feature selection (`pca_loadings.png`) | 3 min |
| 4 | How many components — two regimes (`pca_two_regimes.png`) | 4 min |
| 5 | PCA as preprocessing inside the Pipeline, leakage, baseline | 4 min |
| 6 | t-SNE (`pca_vs_tsne.png`) — can / cannot | 4 min |
| 7 | Quiz (2 questions) | 2 min |
| 8 | Live demo notebook | 5 min |
| 9 | **Exercises** | **10 min** |
| **Total** | | **45 min** |

---

## Content Outline

### Block 1 — Curse of Dimensionality (5 min)
- GIF: histogram of pairwise distances narrows and nearest ÷ farthest neighbour → 1 as d grows (1 → 1000).
- Consequences: distance-based methods (KNN, K-Means, DBSCAN) lose signal; more data needed; overfitting; no plots beyond 3-D.
- Counterpoint (manifold hypothesis): real data lives near low-D structure — what PCA exploits.

### Block 2 — PCA (8 min)
- GIF: rotate a line through the data, project, measure spread; PC1 = max variance (89 % in the demo), PC2 ⊥ PC1, …; dropping low-variance axes = compression. PCA is unsupervised (never sees y).
```python
pca_pipe = make_pipeline(StandardScaler(), PCA(n_components=2))
X_2d = pca_pipe.fit_transform(X)
pca_pipe['pca'].explained_variance_ratio_ / .components_
```
- Scale first — without scaling the largest-valued feature *is* PC1.

### Block 3 — PCA ≠ Feature Selection (3 min)
- `pca_loadings.png` (breast cancer): PC1 = similar positive weights on nearly all 30 features; PC2 contrasts size vs texture. PCs are combinations → harder to interpret; for "which raw features matter" use selection / importances.

### Block 4 — How Many Components (4 min)
- Cumulative variance plot; `PCA(n_components=0.95)`.
- Two regimes: **visualise** with 2 PCs (digits: 22 % — still informative; see `0-material/pca_low_variance_microbiome.ipynb` for 500 features), **preprocess** with ≈90–95 % (digits: 40 of 64).

### Block 5 — PCA as Preprocessing (4 min)
```python
pipe = make_pipeline(StandardScaler(), PCA(0.95), RandomForestClassifier(random_state=42))
cross_val_score(pipe, X, y, cv=5)     # PCA fitted per training fold → no leakage
```
- Compare with the no-PCA pipeline and a DummyClassifier baseline; expect roughly equal or slightly lower accuracy with fewer features — a trade-off, not a free lunch. Helps for very wide / strongly correlated data and distance/linear models.

### Block 6 — t-SNE (4 min)
- Keeps neighbourhoods; not distances between groups, not group sizes, not the number of clusters (perplexity, seed change the picture). Barnes-Hut O(n log n).
- **No `.transform()`** for new data → cannot go into a train/test pipeline → visualisation only. UMAP: similar, faster, has transform (`umap-learn`).

### Block 7 — Quiz (2 min)
- Q1: 18 % variance in 2-D but groups separate → fine for a picture. Q2: PCA fitted before the split → leakage.

### Block 8 — Live Demo (5 min) → `02-examples/ch09_dimensionality_reduction_examples.ipynb`
Digits: scree (80/90/95 % → 21/31/40 PCs) → PCA vs t-SNE side by side → RF with/without PCA vs dummy (StratifiedKFold CV, verdict printed from the numbers).

### Block 9 — Exercises (10 min) → `03-exercises/ch09_dimensionality_reduction_exercises.ipynb`
Breast cancer, **target flipped so malignant = 1** (positive class, as in Ch05/06): Task 1 scree · Task 2 2-D picture · Task 3 dummy vs RF vs RF+PCA(0.95) with CV · Bonus A t-SNE (two seeds) · Bonus B PC1 loadings.

---

## Instructor Notes
- Kernbotschaft: PCA builds new axes; it does not select features.
- Whenever a PCA plot is coloured by class: "labels added afterwards — PCA never saw them".
- Students will over-interpret t-SNE; the `.transform()` argument is the practical one.
- CV appears again (spiral from Ch04/06): the pipeline is what makes CV leakage-free.

---

## Materials
- Slides: `01-slides/ch09_slides.md` (images from `imagegen/ch09.py`: `curse_dimensionality.gif`, `pca_rotation.gif`, `pca_loadings.png`, `pca_two_regimes.png`, `pca_vs_tsne.png`)
- Examples: `02-examples/ch09_dimensionality_reduction_examples.ipynb`
- Exercises / Solutions: `03-exercises/ch09_dimensionality_reduction_exercises.ipynb`, `04-solutions/ch09_dimensionality_reduction_solutions.ipynb`
- Background: `0-material/pca_low_variance_microbiome.ipynb`
