# Chapter 08 — Clustering Techniques

**Session:** 3 | **Chapter:** 2 of 3 | **Duration:** 50 min  
**Format:** Slides + Examples + Exercises

---

## Learning Objectives

By the end of this chapter, students will be able to:
- Apply K-Means clustering and interpret its output
- Use the Elbow Method and Silhouette Score to select k
- Apply hierarchical clustering and read a dendrogram
- Apply DBSCAN and understand its density-based approach
- Choose the right clustering method for a given dataset

---

## Timing Breakdown

| Block | Content | Time |
|-------|---------|------|
| 1 | K-Means Algorithm | 12 min |
| 2 | Choosing k: Elbow & Silhouette | 8 min |
| 3 | Hierarchical Clustering | 8 min |
| 4 | DBSCAN | 7 min |
| 5 | Algorithm Comparison | 5 min |
| 6 | **Exercises** | **10 min** |
| **Total** | | **50 min** |

> **Exercise time: 10 minutes**

---

## Content Outline

### Block 1 — K-Means Algorithm (12 min)

**The most popular clustering algorithm.**

**Idea:** Partition n samples into k clusters, minimizing within-cluster variance.

**Algorithm:**
1. Randomly initialize k cluster centers (centroids)
2. Assign each sample to its nearest centroid
3. Recompute centroids as the mean of all samples in the cluster
4. Repeat steps 2-3 until centroids stop moving (convergence)

**What K-Means optimizes:**
- Inertia = Σᵢ Σₓ∈Cᵢ ||x - μᵢ||²
- (Sum of squared distances from each point to its cluster center)

**Hyperparameters:**
- `k` (n_clusters): number of clusters — the key decision
- `init`: initialization method ('k-means++' is smarter than random)
- `n_init`: number of runs (use 10 to avoid bad local optima)
- `max_iter`: maximum iterations

**Limitations of K-Means:**
- Assumes clusters are spherical (equal variance)
- Sensitive to outliers (they can distort centroids)
- Must specify k in advance
- Can get stuck in local optima

```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_         # cluster assignment for each sample
centroids = kmeans.cluster_centers_  # cluster center coordinates
inertia = kmeans.inertia_       # within-cluster sum of squares
```

---

### Block 2 — Choosing k: Elbow & Silhouette (8 min)

**The central question: how many clusters should we use?**

**Method 1: Elbow Method**
- Run K-Means for k = 1, 2, 3, ..., N
- Plot inertia vs k
- Look for the "elbow" — the point where improvement levels off
- That k is a good choice

**Method 2: Silhouette Score**
- For each sample, compute:
  - a = mean distance to all points in the same cluster
  - b = mean distance to all points in the nearest other cluster
  - Silhouette = (b - a) / max(a, b)
- Range: [-1, 1]
  - +1: sample is far from neighboring clusters → well clustered
  - 0: sample is on the boundary
  - -1: sample is in the wrong cluster
- Choose k that maximizes the average silhouette score

```python
from sklearn.metrics import silhouette_score
score = silhouette_score(X, labels)
```

**Practical advice:** Combine both methods — neither is definitive alone.

---

### Block 3 — Hierarchical Clustering (8 min)

**A different approach: build a tree of clusters.**

**Agglomerative (bottom-up):**
1. Start: every sample is its own cluster (n clusters)
2. Merge the two most similar clusters
3. Repeat until all samples are in one cluster
4. Cut the tree at any level to get k clusters

**Linkage criteria (how to measure cluster similarity):**
- Ward: minimizes total within-cluster variance (most popular)
- Complete: maximum distance between clusters
- Average: mean distance between clusters
- Single: minimum distance between clusters (can create long chains)

**Dendrogram:** Tree visualization of the hierarchical merging process.
- y-axis: distance/dissimilarity at which clusters were merged
- Cut horizontally to choose the number of clusters
- A long vertical line = a natural gap in the data

**Advantage over K-Means:** No need to specify k in advance; dendrogram reveals natural structure.

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

Z = linkage(X, method='ward')
dendrogram(Z)
```

---

### Block 4 — DBSCAN (7 min)

**Density-Based Spatial Clustering of Applications with Noise.**

**Core idea:** Clusters are dense regions separated by sparse regions.

**Key concepts:**
- **Core point:** has at least `min_samples` neighbors within radius `eps`
- **Border point:** within `eps` of a core point but not a core point itself
- **Noise point:** not a core or border point → labeled -1 (outlier!)

**Algorithm:**
1. Find all core points
2. Connect core points that are within `eps` of each other
3. Assign border points to the nearest core point's cluster
4. Label remaining points as noise

**Hyperparameters:**
- `eps`: neighborhood radius (critical to tune — use knee of k-distance graph)
- `min_samples`: minimum points to form a core point (try 2 × n_features)

**Strengths vs K-Means:**
- Does NOT require k in advance
- Finds arbitrarily shaped clusters
- Explicitly identifies outliers (noise)
- Handles clusters of varying density (somewhat)

```python
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)
print(f'Clusters found: {len(set(labels)) - (1 if -1 in labels else 0)}')
print(f'Outliers: {(labels == -1).sum()}')
```

---

### Block 5 — Algorithm Comparison (5 min)

| Algorithm | Shape | Outliers | k needed | Scales |
|-----------|-------|---------|---------|--------|
| K-Means | Spherical | No | Yes | Well |
| Hierarchical | Any | No | No (cut) | Poorly (O(n²)) |
| DBSCAN | Any | Yes (labels them) | No | Moderate |

**Rule of thumb:**
- Unknown k, roughly spherical → K-Means + Elbow
- Want to explore hierarchy → Agglomerative + Dendrogram
- Irregular shapes, outliers expected → DBSCAN

---

## Instructor Notes

- K-Means: animate the iterations if possible — it's very visual
- The elbow method: warn students the "elbow" is often ambiguous
- Silhouette scores: explain intuitively before the formula
- DBSCAN: the eps parameter is the hardest to tune — show the k-distance graph trick
- Comparison: emphasize that no algorithm is universally best

---

## Materials

- Slides: `01-slides/ch08_slides.md`
- Examples: `02-examples/ch08_clustering_examples.ipynb`
- Exercises: `03-exercises/ch08_clustering_exercises.ipynb`
- Solutions: `04-solutions/ch08_clustering_solutions.ipynb`
