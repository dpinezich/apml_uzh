---
layout: cover
title: "Ch01 — Introduction to Machine Learning"
controls: false
fonts:
  sans: Lato
  mono: JetBrains Mono
  weights: '300,400,700,900'
---

# Introduction to Machine Learning
## & Data Science Workflows

**Applied Machine Learning — Session 1, Chapter 1**

---

# What Is Machine Learning?

**Traditional programming**
Rules + Data → Output

**Machine Learning**
Data + Output → Rules

> _"A computer program is said to learn from experience E with respect to some task T and performance measure P..."_
> — Tom Mitchell, 1997

---

# Why Machine Learning?

- Email spam → millions of patterns, impossible to hand-code
- Image recognition → pixels to meaning
- Medical diagnosis → beyond human attention span
- Recommendations → individual preferences at scale

**Key insight:** When rules are too complex to write — let data write them.

---

# Three Paradigms of ML

![ml_paradigms](./ml_paradigms.png)

---

# Supervised Learning

- Training data has **labels**
- Model learns: X → y
- Two flavors:
  - **Regression** → continuous output (price, temperature)
  - **Classification** → discrete output (spam/not spam, disease/no disease)

---

# Unsupervised Learning

- **No labels** — only raw data
- Model finds hidden structure
- Applications:
  - Customer segmentation
  - Anomaly detection
  - Dimensionality reduction
  - Topic modeling

---

# Reinforcement Learning

- **Agent** interacts with **Environment**
- Takes **Actions** → receives **Rewards**
- Goal: maximize cumulative reward

```
Agent → Action → Environment
  ↑                    ↓
  ←←← Reward + State ←←←
```

---

# The Data Science Workflow

![workflow_cycle](./workflow_cycle.png)

**This is a cycle, not a pipeline.**

---

# Step ① — Define the Problem

- What are we trying to predict/discover?
- Who uses the result?
- What is a "good enough" answer?

> Most ML project failures happen here, not in the model.

---

# Step ② — Data

- Where does it come from?
- Is it representative?
- How much do we have?
- What are the legal/ethical constraints?

**Garbage in → Garbage out.**

---

# Steps ③④ — EDA & Preprocessing

- Explore distributions, relationships, outliers
- Handle missing values
- Encode categorical features
- Scale numerical features
- Split: train / validation / test

> ~80% of project time lives here.

---

# Steps ⑤⑥ — Train & Evaluate

- Choose algorithm(s)
- Fit on training data
- Measure performance on held-out data
- Avoid overfitting

---

# Step ⑦ — Deploy & Iterate

- A model in a notebook helps no one
- Deployment = value creation
- Monitor: data drifts, model degrades
- Retrain, improve, repeat

---

# Why Iteration Matters

**Most projects fail not because of bad algorithms — but because of bad iterations.**

| What goes wrong | How iteration fixes it |
|----------------|----------------------|
| Wrong problem definition | Early feedback from stakeholders |
| Dirty / biased data | EDA catches this before training |
| Model doesn't generalize | Evaluation reveals overfitting |
| Features miss signal | Domain experts improve features |
| Production drift | Monitoring triggers retraining |

> **The cycle is the method. Expect at least 3 full loops.**

---

# The Python ML Ecosystem

| Tool | Role |
|------|------|
| `numpy` | Arrays & math |
| `pandas` | Data wrangling |
| `matplotlib` / `seaborn` | Visualization |
| `scikit-learn` | Algorithms & pipelines |
| `gymnasium` | RL environments |

---

# Now: Examples!

→ Open `ch01_introduction_examples.ipynb`

We will:
1. Load the famous **Iris dataset**
2. Inspect its structure
3. Visualize features
4. Build first intuitions — **before any model**

---

# Key Takeaways

- ML = learning patterns from data automatically
- Three paradigms: supervised / unsupervised / reinforcement
- Workflow is a **cycle** — expect iteration
- Great ML starts with **understanding the data**

---
layout: end
---

# Next: Chapter 2

## Data Selection, Cleaning & Preparing

> _"Real-world data is messy, incomplete, and full of surprises. Let's learn how to tame it."_
