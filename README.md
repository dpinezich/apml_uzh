# Applied Machine Learning (UZH)

A hands-on machine learning course: 4 sessions, 12 chapters, ~10 hours of instruction.

## Prerequisites

- Python basics (variables, loops, functions)
- No prior ML knowledge required

## Setup

```bash
# Python dependencies
pip install -r requirements.txt

# Slide presentation (Slidev)
npm install
```

## Running Slides

```bash
./slides.sh           # list all chapters
./slides.sh 2         # serve Ch02 → http://localhost:3030
./slides.sh 2 pdf     # export Ch02 to PDF
```

Press **S** in the browser for speaker/presenter notes.

## Course Structure

| Session | Chapters | Topic |
|---------|----------|-------|
| **1** | Ch01–Ch03 | Data fundamentals, leakage-safe preprocessing, first model (KNN) |
| **2** | Ch04–Ch06 | Regression, classification, evaluation & metrics |
| **3** | Ch07–Ch09 | Clustering & dimensionality reduction |
| **4** | Ch10–Ch12 | Reinforcement learning & 50-min Titanic capstone |

## Repository Layout

```
1-introduction/          Ch01 — Introduction to ML
2-selection_cleaning_preparing/  Ch02 — Data cleaning & preprocessing
3-supervised_learning/   Ch03–Ch06 — Supervised learning
4-unsupervised_learning/ Ch07–Ch09 — Unsupervised learning
5-reinforcement_learning/ Ch10–Ch11 — Reinforcement learning
6-capstone_ml/           Ch12 — End-to-end Titanic project
0-animations/            Animation notebooks (match the GIFs in the slides)
0-datasets/              Local datasets (Titanic CSV); penguins.csv lives in 1-introduction/03-exercises
0-material/              Cheatsheet, slide summary, PCA/microbiome deep-dive
imagegen/                Per-chapter image & GIF generators (called by generate_images.py)
slidev/                  Slide theme, layouts, generated images
```

Regenerate all slide images and GIF animations with `python generate_images.py`.

Each chapter folder contains:
- `00-material/` — Outline and teaching notes
- `01-slides/` — Slidev presentation
- `02-examples/` — Live-demo notebook
- `03-exercises/` — Student exercises
- `04-solutions/` — Exercise solutions

## Didactic Concept

See [DIDAKTIK.md](DIDAKTIK.md) for the full pedagogical plan, session timing, and instructor notes.
