# Chapter 07 — Introduction to Unsupervised Learning

**Session:** 3 | **Chapter:** 1 of 3 | **Duration:** 50 min  
**Format:** Slides + Examples (conceptual, no formal exercise)

---

## Learning Objectives

By the end of this chapter, students will be able to:
- Explain what unsupervised learning is and when to use it
- Distinguish between clustering, dimensionality reduction, and density estimation
- Understand the challenge of evaluation without labels
- Recognize real-world applications of unsupervised learning
- Perform first visual exploration of unlabeled data

---

## Timing Breakdown

| Block | Content | Time |
|-------|---------|------|
| 1 | What Is Unsupervised Learning? | 10 min |
| 2 | Types of Unsupervised Learning | 8 min |
| 3 | The Evaluation Challenge | 7 min |
| 4 | Real-World Applications | 5 min |
| 5 | Live Example: Exploring Unlabeled Data | 20 min |
| **Total** | | **50 min** |

> **No formal exercises** — this is a conceptual and exploratory chapter.

---

## Content Outline

### Block 1 — What Is Unsupervised Learning? (10 min)

**The key difference from supervised learning:**
- Supervised: we have labels → train to predict them
- Unsupervised: we have no labels → discover structure

**The question changes:**
- Supervised: "What is this?" (predict y from X)
- Unsupervised: "What is in this?" (find patterns in X alone)

**Why does unlabeled data exist?**
- Labeling is expensive (human annotation)
- Labeling is impossible (future data)
- Labels don't exist yet (discovery science)
- We don't know what we're looking for

**The fundamental challenge:** There is no "right answer" to compare to.

**Analogy:** Imagine sorting a pile of mail. Without knowing the rules, you find your own groupings (by size, color, sender). Different people might sort differently — both could be valid.

---

### Block 2 — Types of Unsupervised Learning (8 min)

**1. Clustering** (Chapter 08)
- Group similar samples together
- Input: X only → Output: cluster assignments
- Examples: customer segments, document topics, gene expression groups

**2. Dimensionality Reduction** (Chapter 09)
- Compress many features into fewer dimensions
- Input: X (many features) → Output: X' (fewer features)
- Uses: visualization, preprocessing, noise reduction
- Examples: PCA, t-SNE, UMAP, autoencoders

**3. Density Estimation**
- Model the underlying probability distribution of the data
- Uses: anomaly detection (low density = anomaly), generation
- Examples: Gaussian Mixture Models, Kernel Density Estimation
- (Not covered in exercises, but worth mentioning)

**4. Generative Models** (not in this course)
- Learn to generate new samples from the same distribution
- Examples: VAEs, GANs, diffusion models

---

### Block 3 — The Evaluation Challenge (7 min)

**The core problem:** Without labels, how do we know if we did a good job?

**Internal evaluation metrics (no labels needed):**
- Silhouette Score: measures how well-separated clusters are (-1 to +1, higher = better)
- Inertia/Within-cluster sum of squares: measures cluster compactness (lower = better)
- Davies-Bouldin Index: measures cluster separation (lower = better)

**External evaluation (when some labels are available):**
- Adjusted Rand Index
- Normalized Mutual Information

**Human evaluation:**
- Do the discovered groups make business/domain sense?
- Are the patterns interpretable?

**Key insight:** Domain knowledge is critical in unsupervised learning. Numbers alone are not enough.

---

### Block 4 — Real-World Applications (5 min)

| Application | Type | What it finds |
|------------|------|--------------|
| Customer segmentation | Clustering | Groups of similar buyers |
| Document topic modeling | Clustering | Thematic groups in text |
| Anomaly/fraud detection | Density | Unusual transactions |
| Gene expression analysis | Clustering | Disease subgroups |
| Image compression | Dim. reduction | Compact representation |
| Data visualization | Dim. reduction | 2D/3D view of high-dim data |
| Recommendation pre-processing | Dim. reduction | Latent user/item factors |

---

### Block 5 — Live Example (20 min)

→ See `02-examples/ch07_unsupervised_intro_examples.ipynb`

Key demonstrations:
1. Visualize labeled vs unlabeled data (showing we "forget" the labels)
2. Generate synthetic clusters with `make_blobs` and `make_moons`
3. Show that visual patterns emerge without labels
4. Introduce the concept of "structure" in data
5. First look at what clustering will do in Ch08

---

## Instructor Notes

- Spend time on the motivation — students need to understand WHY unsupervised learning exists
- The "no right answer" challenge is profound — make students feel the difficulty
- Use the sorting mail analogy and invite students to think of other examples
- The applications list often sparks great discussions — let it breathe
- This chapter sets up Ch08 and Ch09 conceptually

---

## Materials

- Slides: `01-slides/ch07_slides.md`
- Examples: `02-examples/ch07_unsupervised_intro_examples.ipynb`
