# Chapter 09 — Dimensionality Reduction

**Session:** 3 | **Chapter:** 3 of 3 | **Duration:** 50 min  
**Format:** Slides + Examples + Exercises

---

## Learning Objectives

By the end of this chapter, students will be able to:
- Explain why dimensionality reduction is needed (curse of dimensionality)
- Apply PCA and interpret explained variance
- Choose the number of principal components using a scree plot
- Use PCA as a preprocessing step in a pipeline
- Apply t-SNE for visualization purposes
- Understand when to use PCA vs t-SNE

---

## Timing Breakdown

| Block | Content | Time |
|-------|---------|------|
| 1 | The Curse of Dimensionality | 7 min |
| 2 | Principal Component Analysis (PCA) | 15 min |
| 3 | t-SNE: Non-linear Visualization | 8 min |
| 4 | PCA in a Pipeline (Preprocessing) | 5 min |
| 5 | PCA vs t-SNE vs UMAP | 5 min |
| 6 | **Exercises** | **10 min** |
| **Total** | | **50 min** |

> **Exercise time: 10 minutes**

---

## Content Outline

### Block 1 — The Curse of Dimensionality (7 min)

**What it means:** As the number of features (dimensions) grows, data becomes increasingly sparse.

**Intuition:**
- In 1D: 10 points on a line — they're fairly close together
- In 2D: 10 points in a plane — more space, they spread out
- In 100D: 10 points in a 100D cube — vast empty space separates them

**Practical consequences:**
1. Distance-based algorithms (KNN, K-Means, SVM) struggle — "everything is far from everything"
2. Exponentially more data needed to cover the feature space
3. Overfitting becomes more likely
4. Visualization is impossible beyond 3D

**The blessing of dimensionality (counterpoint):**
- High-dimensional data often lies on a low-dimensional manifold
- e.g., images of faces: billions of pixels, but only ~50 "meaningful" dimensions (expression, lighting, angle)
- This is what dimensionality reduction exploits!

---

### Block 2 — Principal Component Analysis (PCA) (15 min)

**PCA = find the directions of maximum variance in the data.**

**Intuition:**
- Rotate the coordinate system to align with the directions of most spread
- First principal component (PC1): direction of maximum variance
- Second principal component (PC2): direction of maximum remaining variance, orthogonal to PC1
- And so on...
- Discard the last components (least variance) → compression without much information loss

**What PCA produces:**
- `components_`: the principal component directions (eigenvectors of the covariance matrix)
- `explained_variance_ratio_`: fraction of variance captured by each component
- `n_components` controls how many components to keep

**Scree Plot:** Plot of explained variance ratio vs. component number.
- Look for the "elbow" — keep components up to there
- Or choose enough components to explain e.g. 95% of variance

**Important:** PCA is a linear method — it finds straight-line directions.  
**Must scale first!** PCA is scale-sensitive — always apply StandardScaler before PCA.

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pca_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2))
])
X_2d = pca_pipe.fit_transform(X)

# Explained variance
print(pca_pipe['pca'].explained_variance_ratio_)
print('Total variance explained:', pca_pipe['pca'].explained_variance_ratio_.sum())
```

**Two use cases:**
1. **Visualization:** Reduce to 2 or 3 components, plot
2. **Preprocessing:** Reduce to n components that explain 95% variance, then train a classifier

---

### Block 3 — t-SNE: Non-linear Visualization (8 min)

**t-distributed Stochastic Neighbor Embedding (t-SNE)**

**What makes it different from PCA:**
- Non-linear: can "unfold" curved manifolds that PCA cannot
- Preserves local structure: similar points in high-D are nearby in 2D
- Does NOT preserve global structure: distances between clusters are not meaningful

**Algorithm idea (simplified):**
1. Compute pairwise similarities in high-D space (Gaussian kernel)
2. Initialize random 2D positions
3. Iteratively move points to match the high-D similarity structure
4. Optimizes with gradient descent

**Key hyperparameter — perplexity:**
- Controls how many neighbors to consider (~5–50, default 30)
- Smaller perplexity: focuses on local structure
- Larger perplexity: more global view

**t-SNE rules:**
- **Only for visualization** — do NOT use as preprocessing for ML
- **Not deterministic** — run multiple times, results vary
- **Slow** for large datasets (O(n²))
- **Never compare distances between clusters** in the t-SNE plot

```python
from sklearn.manifold import TSNE
X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_scaled)
```

---

### Block 4 — PCA as Preprocessing (5 min)

PCA as a preprocessing step before training a classifier:

```python
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# Keep 95% of variance
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),  # float = fraction of variance to keep
    ('classifier', RandomForestClassifier(random_state=42))
])

pipeline.fit(X_train, y_train)
print('Components used:', pipeline['pca'].n_components_)
print('Accuracy:', pipeline.score(X_test, y_test))
```

**When PCA preprocessing helps:**
- Very high-dimensional data (many more features than samples)
- Features are highly correlated
- Noise reduction is desired

**When it might not help:**
- Model already handles correlated features well (e.g., Random Forest)
- Interpretability of features is important

---

### Block 5 — PCA vs t-SNE vs UMAP (5 min)

| Method | Linear | For ML | For viz | Speed | Global structure |
|--------|:------:|:------:|:-------:|:-----:|:---------------:|
| PCA | ✅ | ✅ | ✅ | Fast | ✅ Preserved |
| t-SNE | ❌ | ❌ | ✅ | Slow | ❌ Lost |
| UMAP | ❌ | ⚠️ | ✅ | Moderate | ⚠️ Partial |

**UMAP** (Uniform Manifold Approximation and Projection):
- Modern alternative to t-SNE
- Faster, better global structure
- Good for both visualization and sometimes preprocessing
- Requires `umap-learn` package (not in sklearn)

---

## Instructor Notes

- The curse of dimensionality: the face example is very memorable
- PCA scree plot: let students choose the cutoff themselves — observe that choices differ
- t-SNE: be very explicit about what you can and cannot conclude from t-SNE plots
- Common misconception: "t-SNE clusters = real clusters" — address this head-on

---

## Materials

- Slides: `01-slides/ch09_slides.md`
- Examples: `02-examples/ch09_dimensionality_reduction_examples.ipynb`
- Exercises: `03-exercises/ch09_dimensionality_reduction_exercises.ipynb`
- Solutions: `04-solutions/ch09_dimensionality_reduction_solutions.ipynb`
