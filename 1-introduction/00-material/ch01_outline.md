# Chapter 01 — Introduction to Machine Learning & Data Science Workflows

**Session:** 1 | **Chapter:** 1 of 3 | **Duration:** 50 min  
**Audience:** Students with basic Python knowledge, new to ML  
**Format:** Slides + Live examples (no exercises)

---

## Learning Objectives

By the end of this chapter, students will be able to:
- Explain what Machine Learning is and why it matters
- Distinguish between Supervised, Unsupervised, and Reinforcement Learning
- Describe the end-to-end Data Science workflow
- Name the core Python tools used in ML
- Load and do a first exploration of a dataset

---

## Timing Breakdown

| Block | Content | Time |
|-------|---------|------|
| 1 | What is Machine Learning? | 10 min |
| 2 | Types of Learning | 8 min |
| 3 | The Data Science Workflow | 7 min |
| 4 | Tools & Ecosystem | 5 min |
| 5 | Live Example: Exploring a Dataset | 15 min |
| 6 | Q&A and Wrap-up | 5 min |
| **Total** | | **50 min** |

---

## Content Outline

### Block 1 — What is Machine Learning? (10 min)

**Core idea:** Traditional programming gives computers rules; ML lets computers *learn* rules from data.

- Traditional programming: `rules + data → output`
- Machine Learning: `data + output → rules`
- Real-world motivation: spam filters, recommendation systems, image recognition, medical diagnosis
- The fundamental promise: finding patterns too complex for humans to write by hand

**Analogy for students:**  
Teaching a child to recognize a dog. You don't give them a rulebook — you show them examples. ML works the same way.

---

### Block 2 — Types of Learning (8 min)

Three paradigms, three chapters of this course:

**Supervised Learning** (Session 1–2)
- Labeled training data: `(X, y)` pairs
- Goal: learn a mapping from X → y
- Examples: house price prediction, email spam detection, medical diagnosis

**Unsupervised Learning** (Session 3)
- No labels — only features: just `X`
- Goal: find hidden structure in data
- Examples: customer segmentation, anomaly detection, topic modeling

**Reinforcement Learning** (Session 4)
- Agent learns by interacting with an environment
- Reward-based feedback
- Examples: game-playing agents, robotics, recommendation systems

*Quick visual: the ML family tree on the board / slide*

---

### Block 3 — The Data Science Workflow (7 min)

The lifecycle every project follows. This is the backbone of the entire course.

```
1. Define the Problem
        ↓
2. Collect & Select Data
        ↓
3. Explore & Clean Data (EDA)
        ↓
4. Preprocess & Feature Engineering
        ↓
5. Choose & Train a Model
        ↓
6. Evaluate the Model
        ↓
7. Iterate → Interpret → Deploy
```

- Step 1 is the most underrated: garbage goal → garbage model
- Steps 2–4 consume ~80% of a real project's time
- Steps 5–6 are where the algorithms live
- Step 7 is where models create value (or cause harm)

**Key message:** The workflow is circular, not linear. You will always go back.

---

### Block 4 — Tools & Ecosystem (5 min)

| Tool | Purpose | Used in course |
|------|---------|----------------|
| `numpy` | Numerical arrays, math operations | Throughout |
| `pandas` | Data loading, cleaning, manipulation | Throughout |
| `matplotlib` / `seaborn` | Visualization | Throughout |
| `scikit-learn` | ML algorithms, preprocessing, evaluation | Sessions 1–3 |
| `gymnasium` | RL environments | Session 4 |
| `Jupyter Notebook` | Interactive coding environment | Throughout |

*Brief demo: importing libraries, nothing fancy yet*

---

### Block 5 — Live Example: Exploring a Dataset (15 min)

Walk through `ch01_examples.ipynb` — see examples folder.

Topics covered in the notebook:
1. Loading the Iris dataset (sklearn)
2. Inspecting shape, dtypes, basic stats
3. Checking for missing values
4. Visualizing distributions
5. Visualizing relationships between features
6. First intuition: can we predict the species from measurements?

**Teaching goal:** Show that even without algorithms, visualization already reveals a lot.

---

### Block 6 — Q&A and Wrap-up (5 min)

**Summary points:**
- ML = learning patterns from data
- Three paradigms: supervised, unsupervised, reinforcement
- Workflow is a cycle, not a pipeline
- We always start by *understanding the data*

**Teaser for Ch02:** "Next, we'll get our hands dirty with messy real data — because real data is never clean."

---

## Instructor Notes

- Keep it high-level — the goal is inspiration, not information overload
- Encourage questions throughout Block 1 & 2
- If students seem comfortable with pandas, speed through Block 4
- The Iris dataset is iconic — mention its history (R.A. Fisher, 1936)
- Optional: show a cool ML demo (e.g., quickdraw.withgoogle.com) to energize the room

---

## Materials

- Slides: `01-slides/ch01_slides.md`
- Examples: `02-examples/ch01_introduction_examples.ipynb`
- No exercises / solutions for this chapter
