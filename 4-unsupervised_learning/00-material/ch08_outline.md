# Chapter 08 — Clustering Techniques

**Session:** 3 | **Chapter:** 2 of 3 | **Duration:** 50 min  
**Format:** Slides + Examples + Exercises

---

## Learning Objectives

By the end of this chapter, students will be able to:
- Explain the K-Means loop (assign → update) and why it converges to a *local* optimum (`k-means++`, `n_init`)
- Scale features before any distance-based clustering (Pipeline with StandardScaler)
- Use elbow and silhouette to choose k — and explain why they can disagree
- Recognise that K-Means always returns k clusters and check whether structure is real
- Apply DBSCAN and read its noise label as outliers
- (Bonus) Read a dendrogram / cut a hierarchical clustering

---

## Timing Breakdown (planned 43 min + buffer)

| Block | Content | Time |
|-------|---------|------|
| 1 | What is clustering + K-Means algorithm (GIF `kmeans_iterations.gif`) | 7 min |
| 2 | Parameters, `init`/`n_init` — local optima (`kmeans_init.png`) | 3 min |
| 3 | Scale first (`scaling_kmeans.png`, Pipeline idiom) | 3 min |
| 4 | Choosing k: elbow + silhouette (`elbow_silhouette.png`); clustering noise (`kmeans_uniform.png`) | 6 min |
| 5 | DBSCAN (`kmeans_vs_dbscan.png`) + algorithm table | 4 min |
| 6 | Quiz (2 questions) | 2 min |
| 7 | Live demo notebook | 8 min |
| 8 | **Exercises** | **10 min** |
| **Total** | | **43 min** |

> Hierarchical clustering is a **bonus/appendix** slide (`dendrogram.png`) and Bonus B in the exercise — show only if time allows.

---

## Content Outline

### Block 1 — K-Means (7 min)
- Clustering: similar within, dissimilar between; output = new column of arbitrary ids.
- Algorithm: pick k random points → ASSIGN each point to nearest centroid → UPDATE centroids to cluster means → repeat until nothing moves. Minimises inertia = Σ‖x − μ_c(x)‖²; inertia only decreases → converges.
- GIF shows ~6 iterations with the inertia counter; `0-animations/01_kmeans_convergence.ipynb` is the interactive version (`SEED` knob, seed 1 = bad local optimum).

### Block 2 — Parameters (3 min)
```python
KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)
kmeans.labels_ / cluster_centers_ / inertia_
```
- `kmeans_init.png`: random single-run start ends with inertia ~4× worse than k-means++ with n_init=10. Kill the misconception "K-Means finds THE clusters".

### Block 3 — Scale first (3 min)
- Distances are dominated by the feature with the largest numbers → `make_pipeline(StandardScaler(), KMeans(...))`. Applies to K-Means, DBSCAN, hierarchical, KNN.
- Caveat (Ch07 exercise): on digit pixels scaling *hurts* K-Means (near-constant border pixels get amplified) — default, not law.

### Block 4 — Choosing k (6 min)
- Elbow: inertia always decreases with k; look for the flattening (often ambiguous).
- Silhouette per sample s = (b − a)/max(a, b); average it; +1 good, 0 boundary, −1 wrong; pick the max — favours few round clusters.
- **Iris**: elbow ≈ 3, silhouette max at 2 — both defensible (exercise Task 2c).
- `kmeans_uniform.png`: uniform noise → K-Means still returns 3 clusters, elbow plot looks "normal"; silhouette 0.38 vs 0.83 for real blobs. Always plot (PCA 2-D) and check the silhouette *value*.

### Block 5 — DBSCAN (4 min)
- Core point: ≥ `min_samples` neighbours within `eps`; border; noise (−1) = built-in outlier detection.
- No k, arbitrary shapes; price: `eps` tuning (scale first!), and it struggles when clusters have very *different* densities (one eps for all — HDBSCAN/OPTICS fix that, out of scope). Rule of thumb: `min_samples ≈ 2 × n_features`, eps from the k-distance knee.

| Algorithm | Shape | k needed | Outliers | Notes |
|---|---|---|---|---|
| K-Means | round, similar size | yes | pulled by them | fast, first try |
| DBSCAN | any (density) | no | flagged −1 | eps tuning; mixed densities hard |
| Hierarchical (bonus) | any | no (cut) | no | dendrogram, O(n²) |
| GMM (mention) | ellipses, soft | yes | no | probabilities |

### Block 6 — Quiz (2 min)
- Q1 elbow 3 vs silhouette 2 → look at both, choose by usefulness. Q2 age vs income unscaled → income dominates.

### Block 7 — Live Demo (8 min) → `02-examples/ch08_clustering_examples.ipynb`
1. Elbow + silhouette on 4 blobs **and** on uniform noise (both give a "best k" — only the value/plot tells them apart)
2. Scale + Pipeline; K-Means vs DBSCAN on moons (robust colour map, noise grey)
3. Customer segmentation: silhouette-chosen k → **profile table** → names derived from the numbers (never from ids)
4. Bonus: Ward dendrogram with a computed cut

### Block 8 — Exercises (10 min) → `03-exercises/ch08_clustering_exercises.ipynb`
Iris without species: Task 1 elbow · Task 2 silhouette + explain the disagreement · Task 3 Pipeline(StandardScaler, KMeans), petal plot with inverse-transformed centroids · Task 4 ARI vs species framed as *sanity check* (crosstab shows which species mix) · Bonus A DBSCAN (eps 0.5 vs 0.8) · Bonus B hierarchical.

---

## Instructor Notes
- The uniform-noise slide is the most important one: K-Means never says "no structure".
- Whenever species labels appear: "sanity check, not accuracy — clusters ≠ classes".
- Segment names in the demo are computed from the profile table; cluster ids are arbitrary and change with the seed.

---

## Materials
- Slides: `01-slides/ch08_slides.md` (images from `imagegen/ch08.py`: `kmeans_iterations.gif`, `kmeans_init.png`, `scaling_kmeans.png`, `elbow_silhouette.png`, `kmeans_uniform.png`, `kmeans_vs_dbscan.png`, `dendrogram.png`)
- Examples: `02-examples/ch08_clustering_examples.ipynb`
- Exercises / Solutions: `03-exercises/ch08_clustering_exercises.ipynb`, `04-solutions/ch08_clustering_solutions.ipynb`
- Animation: `0-animations/01_kmeans_convergence.ipynb`
