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
| **1** | Ch01–Ch03 | Data fundamentals & first models |
| **2** | Ch04–Ch06 | Regression, classification, evaluation |
| **3** | Ch07–Ch09 | Clustering & dimensionality reduction |
| **4** | Ch10–Ch12 | Reinforcement learning & capstone project |

## Repository Layout

```
1-introduction/          Ch01 — Introduction to ML
2-selection_cleaning_preparing/  Ch02 — Data cleaning & preprocessing
3-supervised_learning/   Ch03–Ch06 — Supervised learning
4-unsupervised_learning/ Ch07–Ch09 — Unsupervised learning
5-reinforcement_learning/ Ch10–Ch11 — Reinforcement learning
6-capstone_ml/           Ch12 — End-to-end Titanic project
0-animations/            Supplementary animation notebooks
slidev/                  Slide theme, layouts, generated images
```

Each chapter folder contains:
- `00-material/` — Outline and teaching notes
- `01-slides/` — Slidev presentation
- `02-examples/` — Live-demo notebook
- `03-exercises/` — Student exercises
- `04-solutions/` — Exercise solutions

## Didactic Concept

See [DIDAKTIK.md](DIDAKTIK.md) for the full pedagogical plan, session timing, and instructor notes.
