---
marp: true
theme: default
paginate: true
---

# Introduction to Unsupervised Learning

**Applied Machine Learning — Session 3, Chapter 1**

---

# The Key Difference

**Supervised Learning:**
```
Input: (X, y)  →  Learn to predict y
```

**Unsupervised Learning:**
```
Input: X only  →  Discover structure in X
```

No labels. No "right answer." Just data.

---

# Why Does Unlabeled Data Exist?

- Labeling is **expensive** (human annotation costs time + money)
- Labeling is **impossible** for future data
- Labels **don't exist yet** (discovery science)
- We don't know what we're looking for

> Most data in the world is unlabeled.  
> Supervised learning is the exception, not the rule.

---

# The Question Changes

| Paradigm | Question |
|----------|---------|
| Supervised | "What is this?" (classify/predict y) |
| Unsupervised | "What is **in** this?" (find patterns) |

**Analogy:** Sorting mail without knowing the rules.  
You find your own groupings — by sender, size, topic.  
Different people might sort differently. Both can be valid.

---

# Three Types of Unsupervised Learning

**1. Clustering** (Ch08)
→ Group similar samples together

**2. Dimensionality Reduction** (Ch09)
→ Compress many features into fewer

**3. Density Estimation**
→ Model the probability distribution of the data

---

# Clustering

**Input:** X (no labels)  
**Output:** Cluster assignments {0, 1, 2, ...}

```
Before                   After K-Means
 • •  ○ ○                [0][0] [1][1]
• • •  ○ ○      →       [0][0][0] [1][1]
 • •    ○               [0][0]    [1]
```

Applications:
- Customer segmentation
- Document topic modeling
- Gene expression grouping

---

# Dimensionality Reduction

**Input:** X with many features  
**Output:** X' with fewer features (preserving information)

```
100 features → 2 features → visualize!
```

Applications:
- Visualization (2D or 3D plots of high-dim data)
- Preprocessing (fewer features → faster, less overfitting)
- Noise reduction

---

# The Evaluation Challenge

**Without labels — how do we know if we did a good job?**

**Internal metrics (no labels needed):**
- Silhouette Score: cluster separation quality
- Inertia: within-cluster compactness

**External validation:**
- Does it make business sense?
- Do domain experts agree?
- Does it help a downstream task?

> **Domain knowledge is essential in unsupervised learning.**

---

# Real-World Applications

| Field | Application | Technique |
|-------|------------|-----------|
| Marketing | Customer segments | Clustering |
| Medicine | Disease subtypes | Clustering |
| NLP | Topic discovery | Clustering |
| Finance | Anomaly detection | Density estimation |
| Vision | Image compression | PCA |
| Any | Data visualization | t-SNE / UMAP |

---

# Let's Explore Together

→ Open `02-examples/ch07_unsupervised_intro_examples.ipynb`

We will:
1. Generate and visualize unlabeled data
2. See what "structure" looks like without labels
3. Preview what clustering will find in Chapter 8

---

# Key Takeaways

- Unsupervised = learning without labels
- Goal: discover structure, patterns, groups
- Three main types: clustering, dim. reduction, density
- Evaluation is hard — domain knowledge matters
- Most real-world data is unlabeled → unsupervised is powerful

---

# Next: Chapter 8

## Clustering Techniques

> _"Let's make the machine find the groups."_
