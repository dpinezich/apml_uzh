# Didaktischer Plan — Applied Machine Learning (UZH)

**Kurs:** Applied Machine Learning  
**Zielgruppe:** Studierende mit Python-Grundkenntnissen, kein ML-Vorwissen nötig  
**Umfang:** 4 Sessions × 3 Kapitel × 50 min = 12 Kapitel (~10 Stunden)  
**Format:** Slides + Live-Demos + Übungen  
**Letztes Update:** April 2026

---

## Didaktische Leitprinzipien

| Prinzip | Umsetzung |
|---------|-----------|
| **Spiral-Curriculum** | Konzepte werden eingeführt, dann in späteren Kapiteln vertieft (z.B. Train/Test Split in Ch02 → Cross-Validation in Ch06) |
| **Hands-on first** | Jede theoretische Einheit endet mit Übungen, die das Gelernte sofort anwenden |
| **Motivierende Einstiege** | Jedes Kapitel startet mit einem realen Anwendungsfall oder einer provokanten Frage |
| **Visualisieren vor Formalisieren** | Intuition durch Plots und Analogien aufbauen, bevor Formeln eingeführt werden |
| **Sichtbare Fehler zeigen** | Bewusst fehlerhafte Ansätze demonstrieren (Data Leakage, Overfitting) um kritisches Denken zu fördern |

---

## Gesamtübersicht der Sessions

| Session | Thema | Kapitel | Exercises | Kumulatives Thema |
|---------|-------|---------|-----------|-------------------|
| **1** | Daten & Grundlagen | Ch01 → Ch02 → Ch03 | Ch02: 10 min, Ch03: 10 min | Vom Rohdatensatz zum ersten Modell |
| **2** | Supervised Learning | Ch04 → Ch05 → Ch06 | je 10–12 min | Regression, Klassifikation, Evaluation |
| **3** | Unsupervised Learning | Ch07 → Ch08 → Ch09 | Ch08+Ch09: je 10 min | Clustering & Dimensionsreduktion |
| **4** | RL + Capstone | Ch10 → Ch11 → Ch12 | Ch11: 10 min, Ch12: **35 min** | Reinforcement Learning + Titanic Projekt |

---

## Session 1 — Daten verstehen & erste Modelle

**Dauer:** ~150 min (3 × 50 min)  
**Ziel:** Studierende können eine Datenmenge laden, bereinigen und ihr erstes Modell trainieren.  
**Roter Faden:** *"Daten sind das Fundament — ohne saubere Daten kein gutes Modell."*

### Timing Session 1

| Zeit | Kapitel | Block | Methode | Hinweis |
|------|---------|-------|---------|---------|
| 0–50 min | **[Ch01 — Einführung](1-introduction/01-slides/ch01_slides.md)** | Überblick ML, Typen, Workflow, Tools | Slides + Live-Demo | Optionale Übung (Penguins EDA) als Bonus |
| 50–100 min | **[Ch02 — Daten selektieren & vorbereiten](2-selection_cleaning_preparing/01-slides/ch02_slides.md)** | Missing Values, Outlier, Encoding, Scaling, Split | Slides + Übungen | ⚠️ Data Leakage ist das zentrale Konzept |
| 100–150 min | **[Ch03 — Supervised Learning Intro](3-supervised_learning/01-slides/ch03_slides.md)** | Bias/Variance, Overfitting, KNN | Slides + Übungen | Erstes echtes Modell → motivierend! |

### Didaktische Hinweise Session 1

- **Einstieg Ch01:** Frage an die Klasse: *"Wer hat heute schon ML benutzt?"* → Alle haben es (Spotify, Netflix, Google Maps). Sofort zeigen dass ML überall ist.
- **Ch02 Kernbotschaft:** Den Satz *"Impute after split!"* mehrfach wiederholen und begründen. Das ist der häufigste Anfängerfehler.
- **Ch02 → Ch03 Übergang:** Pipeline aus Ch02 als Input für Ch03 nutzen — die sauberen Daten aus den Übungen direkt ins erste Modell stecken.
- **Differenzierung:** Wer schnell fertig ist mit Übungen → Bonus-Aufgabe: andere Imputation-Strategie testen und Ergebnis vergleichen.

**Materialien Session 1:**
- Slides: [`ch01_slides.md`](1-introduction/01-slides/ch01_slides.md) · [`ch02_slides.md`](2-selection_cleaning_preparing/01-slides/ch02_slides.md) · [`ch03_slides.md`](3-supervised_learning/01-slides/ch03_slides.md)
- Outline/Lehrplan: [`ch01_outline.md`](1-introduction/00-material/ch01_outline.md) · [`ch02_outline.md`](2-selection_cleaning_preparing/00-material/ch02_outline.md) · [`ch03_outline.md`](3-supervised_learning/00-material/ch03_outline.md)
- Übungen: [`ch01_introduction_exercises.ipynb`](1-introduction/03-exercises/) (Bonus) · [`ch02_data_cleaning_exercises.ipynb`](2-selection_cleaning_preparing/03-exercises/) · [`ch03_supervised_intro_exercises.ipynb`](3-supervised_learning/03-exercises/)
- Animationen: [`02_knn_decision_boundary.ipynb`](0-animations/02_knn_decision_boundary.ipynb) · [`05_polynomial_overfitting.ipynb`](0-animations/05_polynomial_overfitting.ipynb)

---

## Session 2 — Supervised Learning: Regression, Klassifikation, Evaluation

**Dauer:** ~150 min (3 × 50 min)  
**Ziel:** Studierende verstehen Regression und Klassifikation als zwei Seiten von Supervised Learning und können Modelle korrekt evaluieren.  
**Roter Faden:** *"Welches Modell für welches Problem — und wie weiss ich ob es gut ist?"*

### Timing Session 2

| Zeit | Kapitel | Block | Methode | Hinweis |
|------|---------|-------|---------|---------|
| 0–50 min | **[Ch04 — Regression](3-supervised_learning/01-slides/ch04_slides.md)** | Linear Regression, Polynomiale Regression, Overfitting | Slides + Übungen | Visualisierung der Regressionsgerade live |
| 50–100 min | **[Ch05 — Klassifikation](3-supervised_learning/01-slides/ch05_slides.md)** | Logistic Regression, Decision Trees, SVM | Slides + Übungen | Decision Boundary visualisieren |
| 100–150 min | **[Ch06 — Evaluation & Metriken](3-supervised_learning/01-slides/ch06_slides.md)** | Accuracy, Precision, Recall, F1, ROC-AUC, Cross-Validation | Slides + Übungen | ⚠️ "Accuracy Paradox" zeigen |

### Didaktische Hinweise Session 2

- **Ch04 Einstieg:** Hauspreisvorhersage — jede\*r kennt Immobilienpreise. *"Wie würdet ihr den Preis schätzen?"* → Intuitive Regression.
- **Ch05 Einstieg:** Spam-E-Mail Beispiel. *"Was macht ein Spam-Filter?"* → Binäre Klassifikation.
- **Ch06 Kernbotschaft:** "Accuracy ist oft eine Lüge." → Beispiel Krebs-Screening: 1% krank, 99% gesund → 99% Accuracy mit dem Modell "immer negativ" — trotzdem wertlos. Dieses Beispiel prägt sich ein.
- **Cross-Validation:** Hier noch nicht implementieren — Konzept einführen, in Session 3 vertiefen.
- **Pace:** Ch04 und Ch05 können zügig durch wenn Grundkonzept aus Ch03 sitzt. Zeit für Ch06 schützen — Metriken sind erfahrungsgemäss der am stärksten unterschätzte Teil.

**Materialien Session 2:**
- Slides: [`ch04_slides.md`](3-supervised_learning/01-slides/ch04_slides.md) · [`ch05_slides.md`](3-supervised_learning/01-slides/ch05_slides.md) · [`ch06_slides.md`](3-supervised_learning/01-slides/ch06_slides.md)
- Outline/Lehrplan: [`ch04_outline.md`](3-supervised_learning/00-material/ch04_outline.md) · [`ch05_outline.md`](3-supervised_learning/00-material/ch05_outline.md) · [`ch06_outline.md`](3-supervised_learning/00-material/ch06_outline.md)
- Übungen: [`ch04_regression_exercises.ipynb`](3-supervised_learning/03-exercises/) · [`ch05_classification_exercises.ipynb`](3-supervised_learning/03-exercises/) · [`ch06_metrics_exercises.ipynb`](3-supervised_learning/03-exercises/)
- Animationen: [`03_gradient_descent.ipynb`](0-animations/03_gradient_descent.ipynb)

---

## Session 3 — Unsupervised Learning: Clustering & Dimensionsreduktion

**Dauer:** ~150 min (3 × 50 min)  
**Ziel:** Studierende können ungelabelte Daten mit Clustering und PCA explorieren.  
**Roter Faden:** *"Was lernt ein Algorithmus, wenn es keine richtigen Antworten gibt?"*

### Timing Session 3

| Zeit | Kapitel | Block | Methode | Hinweis |
|------|---------|-------|---------|---------|
| 0–50 min | **[Ch07 — Unsupervised Learning Intro](4-unsupervised_learning/01-slides/ch07_slides.md)** | Konzepte, Anwendungen, kein Label-Problem | Slides + Diskussion | Optionale Übung (PCA + K-Means auf Digits) als Bonus |
| 50–100 min | **[Ch08 — Clustering (K-Means)](4-unsupervised_learning/01-slides/ch08_slides.md)** | K-Means, Elbow-Method, Silhouette Score | Slides + Übungen | K-Means Animation zeigen (siehe [`animations/`](0-animations/)) |
| 100–150 min | **[Ch09 — Dimensionsreduktion (PCA)](4-unsupervised_learning/01-slides/ch09_slides.md)** | Curse of Dimensionality, PCA, t-SNE | Slides + Übungen | PCA auf Iris visuell erklären |

### Didaktische Hinweise Session 3

- **Ch07 Einstieg:** *"Was haben Kundensegmentierung, Genexpression und Spracherkennung gemeinsam?"* → Alle nutzen Unsupervised Learning. Macht den Paradigmenwechsel klar: keine Labels.
- **Ch08 Animation:** Die K-Means Konvergenzanimation aus dem [`animations/`](0-animations/01_kmeans_convergence.ipynb) Ordner live zeigen — sehr wirkungsvoll um den Algorithmus zu verstehen.
- **Ch09 Kernbotschaft:** PCA ist Dimensionsreduktion, nicht Merkmals-Selektion. Den Unterschied klar machen.
- **Verbindung zu Session 2:** *"In Session 2 hatten wir Labels, jetzt nicht mehr. Wie evaluieren wir also unsere Modelle?"* → Silhouette Score, Elbow Method.

**Materialien Session 3:**
- Slides: [`ch07_slides.md`](4-unsupervised_learning/01-slides/ch07_slides.md) · [`ch08_slides.md`](4-unsupervised_learning/01-slides/ch08_slides.md) · [`ch09_slides.md`](4-unsupervised_learning/01-slides/ch09_slides.md)
- Outline/Lehrplan: [`ch07_outline.md`](4-unsupervised_learning/00-material/ch07_outline.md) · [`ch08_outline.md`](4-unsupervised_learning/00-material/ch08_outline.md) · [`ch09_outline.md`](4-unsupervised_learning/00-material/ch09_outline.md)
- Übungen: [`ch07_unsupervised_intro_exercises.ipynb`](4-unsupervised_learning/03-exercises/) (Bonus) · [`ch08_clustering_exercises.ipynb`](4-unsupervised_learning/03-exercises/) · [`ch09_dimensionality_reduction_exercises.ipynb`](4-unsupervised_learning/03-exercises/)
- Animationen: [`01_kmeans_convergence.ipynb`](0-animations/01_kmeans_convergence.ipynb)

---

## Session 4 — Reinforcement Learning & Capstone-Projekt

**Dauer:** ~150 min (3 × 50 min)  
**Ziel:** Studierende verstehen das RL-Paradigma und können ein vollständiges ML-Projekt selbstständig durchführen.  
**Roter Faden:** *"Lernen durch Feedback — und alles zusammenbringen."*

### Timing Session 4

| Zeit | Kapitel | Block | Methode | Hinweis |
|------|---------|-------|---------|---------|
| 0–50 min | **[Ch10 — RL Einführung](5-reinforcement_learning/01-slides/ch10_slides.md)** | Agent, Environment, Reward, Policy, Q-Values | Slides + Demo | Optionale Übung (Reward Shaping) als Bonus |
| 50–100 min | **[Ch11 — Q-Learning](5-reinforcement_learning/01-slides/ch11_slides.md)** | Q-Table, Bellman Equation, Epsilon-Greedy | Slides + Übungen | RL-Agent Animation zeigen |
| 100–150 min | **[Ch12 — Capstone: Titanic](6-capstone_ml/01-slides/ch12_slides.md)** | End-to-End Projekt: von Rohdaten zum Modell | Geführtes Projekt | **35 min Selbstarbeit** — alles aus Sessions 1-4 |

### Didaktische Hinweise Session 4

- **Ch10 Einstieg:** *"Wie lernt ein Hund, einen Trick zu machen?"* → Belohnung → Verstärkendes Lernen. Dann: AlphaGo, Roboter, Trading-Algorithmen.
- **Ch11 Bellman Equation:** Sehr abstrakt — mit GridWorld-Beispiel visuell einführen. Die RL-Agent Animation nutzen.
- **Ch12 Capstone:** Das ist der Höhepunkt des Kurses. **Nicht unterbrechen während der 35 Min Übungsphase.** Herumgehen, helfen, aber keine Frontalphase einschalten. (Kein separates Example-Notebook — die geführte Übung IST das Beispiel.)
- **Ch12 Debriefing (letzte 10 min):** Gemeinsam Ergebnisse vergleichen — wer hat welche Accuracy erreicht? Was hat geholfen? Verbindung zu allen vorherigen Kapiteln ziehen.

**Materialien Session 4:**
- Slides: [`ch10_slides.md`](5-reinforcement_learning/01-slides/ch10_slides.md) · [`ch11_slides.md`](5-reinforcement_learning/01-slides/ch11_slides.md) · [`ch12_slides.md`](6-capstone_ml/01-slides/ch12_slides.md)
- Outline/Lehrplan: [`ch10_outline.md`](5-reinforcement_learning/00-material/ch10_outline.md) · [`ch11_outline.md`](5-reinforcement_learning/00-material/ch11_outline.md) · [`ch12_outline.md`](6-capstone_ml/00-material/ch12_outline.md)
- Übungen: [`ch10_rl_intro_exercises.ipynb`](5-reinforcement_learning/03-exercises/) (Bonus) · [`ch11_rl_algorithms_exercises.ipynb`](5-reinforcement_learning/03-exercises/) · [`ch12_capstone_exercises.ipynb`](6-capstone_ml/03-exercises/)
- Animationen: [`04_rl_agent_learning.ipynb`](0-animations/04_rl_agent_learning.ipynb)

---

## Kapitelübersicht auf einen Blick

| Ch | Titel | Session | Format | Übung | Slides | Outline |
|----|-------|---------|--------|-------|--------|---------|
| 01 | Einführung in ML | S1 | Slides + Demo | (Bonus) | [↗](1-introduction/01-slides/ch01_slides.md) | [↗](1-introduction/00-material/ch01_outline.md) |
| 02 | Daten auswählen & vorbereiten | S1 | Slides + Übung | 10 min | [↗](2-selection_cleaning_preparing/01-slides/ch02_slides.md) | [↗](2-selection_cleaning_preparing/00-material/ch02_outline.md) |
| 03 | Supervised Learning Intro | S1 | Slides + Übung | 10 min | [↗](3-supervised_learning/01-slides/ch03_slides.md) | [↗](3-supervised_learning/00-material/ch03_outline.md) |
| 04 | Regression | S2 | Slides + Übung | 10 min | [↗](3-supervised_learning/01-slides/ch04_slides.md) | [↗](3-supervised_learning/00-material/ch04_outline.md) |
| 05 | Klassifikation | S2 | Slides + Übung | 12 min | [↗](3-supervised_learning/01-slides/ch05_slides.md) | [↗](3-supervised_learning/00-material/ch05_outline.md) |
| 06 | Evaluation & Metriken | S2 | Slides + Übung | 10 min | [↗](3-supervised_learning/01-slides/ch06_slides.md) | [↗](3-supervised_learning/00-material/ch06_outline.md) |
| 07 | Unsupervised Learning Intro | S3 | Slides + Diskussion | (Bonus) | [↗](4-unsupervised_learning/01-slides/ch07_slides.md) | [↗](4-unsupervised_learning/00-material/ch07_outline.md) |
| 08 | Clustering (K-Means) | S3 | Slides + Übung | 10 min | [↗](4-unsupervised_learning/01-slides/ch08_slides.md) | [↗](4-unsupervised_learning/00-material/ch08_outline.md) |
| 09 | Dimensionsreduktion (PCA) | S3 | Slides + Übung | 10 min | [↗](4-unsupervised_learning/01-slides/ch09_slides.md) | [↗](4-unsupervised_learning/00-material/ch09_outline.md) |
| 10 | Reinforcement Learning Intro | S4 | Slides + Demo | (Bonus) | [↗](5-reinforcement_learning/01-slides/ch10_slides.md) | [↗](5-reinforcement_learning/00-material/ch10_outline.md) |
| 11 | Q-Learning | S4 | Slides + Übung | 10 min | [↗](5-reinforcement_learning/01-slides/ch11_slides.md) | [↗](5-reinforcement_learning/00-material/ch11_outline.md) |
| 12 | Capstone: Titanic | S4 | Guided Project | **35 min** | [↗](6-capstone_ml/01-slides/ch12_slides.md) | [↗](6-capstone_ml/00-material/ch12_outline.md) |

---

## Konzeptionelle Progressionskarte

```
Session 1                 Session 2                 Session 3                 Session 4
─────────────────────────────────────────────────────────────────────────────────────────
Was ist ML?          →   Regression              →   Was ohne Labels?    →   Agent & Reward
  │                          │                           │                      │
Daten vorbereiten    →   Klassifikation          →   K-Means             →   Q-Learning
  │                          │                           │                      │
Erstes Modell        →   Evaluieren & Metriken  →   PCA / t-SNE         →   Titanic Projekt
─────────────────────────────────────────────────────────────────────────────────────────
         └── DATEN ──────────────────────────────────────────────────────────┘
              Train/Test Split  ←→  Imputation  ←→  Encoding  ←→  Scaling
```

**Konzept-Querverbindungen (für Lehrgespräche nutzen):**

| Konzept | Eingeführt | Vertieft | Angewendet |
|---------|-----------|---------|-----------|
| Train/Test Split | Ch02 | Ch06 (Cross-Val) | Ch12 |
| Overfitting | Ch03 | Ch04 (Poly Reg) | Ch06 |
| Feature Encoding | Ch02 | Ch05 | Ch12 |
| Pipeline | Ch02 | Ch04 | Ch12 |
| Evaluation | Ch03 | Ch06 | Ch12 |

---

## Häufige Missverständnisse (Lehrerhinweise)

| Missverständnis | Kapitel | Gegenmassnahme |
|----------------|---------|----------------|
| "Mehr Daten → immer besser" | Ch02 | Qualität vor Quantität zeigen; verseuchte Daten verschlechtern alles |
| "Höhere Accuracy = besseres Modell" | Ch06 | Accuracy Paradox mit Krebs-Screening-Beispiel |
| "Man imputed vor dem Split" | Ch02 | Data Leakage Pipeline live durchführen und Fehler sichtbar machen |
| "Unsupervised learning braucht kein Evaluation" | Ch08 | Silhouette Score, Elbow Method einführen |
| "RL ist gleich wie supervised" | Ch10 | Feedback-Loop vs. feste Labels — grundlegend verschieden |
| "K-Means findet immer die 'richtigen' Cluster" | Ch08 | Verschiedene Initialisierungen zeigen (K-Means++ vs. random) |

---

## Verwendete Datensätze

| Datensatz | Herkunft | Verwendet in |
|-----------|---------|-------------|
| Iris | `sklearn.datasets.load_iris()` | Ch01, Ch03, Ch07, Ch08 |
| Penguins | `seaborn.load_dataset('penguins')` | Ch01 (Exercises) |
| California Housing | `sklearn.datasets.fetch_california_housing()` | Ch04, Ch06 |
| Breast Cancer | `sklearn.datasets.load_breast_cancer()` | Ch05, Ch06, Ch09 |
| Diabetes | `sklearn.datasets.load_diabetes()` | Ch04 (Exercises), Ch06 |
| Wine | `sklearn.datasets.load_wine()` | Ch05 (Exercises) |
| Digits | `sklearn.datasets.load_digits()` | Ch07, Ch09 |
| Titanic | `seaborn.load_dataset('titanic')` | Ch12 |
| Synthetische Daten | `make_blobs`, `make_moons`, eigene | Ch02, Ch03, Ch07, Ch08 |

---

## Vorbereitung pro Session — Checkliste

```
Vor jeder Session:
✅ Jupyter Notebook Server starten
✅ Slidev bereit: `./slides.sh <N>` startet die Slides
✅ Lösungs-Notebooks bereit (aber nicht geöffnet)
✅ Kurze Review der Übungsaufgaben der letzten Session

Session 1:
✅ seaborn + sklearn installiert (requirements.txt)
✅ Iris-Datensatz erreichbar (sklearn, offline verfügbar)

Session 4 (Capstone):
✅ 35 Minuten ungestörte Arbeitszeit einplanen
✅ Debriefing vorbereiten: "Was hat eurem Modell am meisten gebracht?"
```

---

## Slides starten (Slidev)

Alle Slides sind im **Slidev** Markdown-Format. Voraussetzung: `npm install` im Projektverzeichnis.

```bash
# Einzelnes Kapitel starten (mit Hot-Reload):
./slides.sh 2          # → Ch02 auf http://localhost:3030

# Alle Kapitel als Listing:
./slides.sh

# Vite-Cache leeren und starten:
./slides.sh 2 fresh

# Als PDF exportieren:
./slides.sh 2 pdf      # → exports/02_slides.pdf

# Als statische Seite bauen:
./slides.sh 2 build    # → exports/ch02/
```

**Speaker Notes** sind in den Slides eingebaut — im Presenter-Modus mit `S` öffnen.

**slidev/ Verzeichnisstruktur:**
```
slidev/
├── style.css               ← Globales Stylesheet (Brand-Farben, Layout)
├── vite.config.ts          ← Asset-Resolution Plugin
├── setup/
│   └── main.ts             ← Runtime-Setup
├── layouts/
│   ├── cover.vue           ← Titel-Slides
│   ├── default.vue         ← Content-Slides (mit Teal-Akzent)
│   └── end.vue             ← Abschluss-/Übergangs-Slides
└── public/                 ← Generierte PNG-Bilder für alle Kapitel
```

**Bilder** bei Bedarf neu generieren:
```bash
python generate_images.py   # → Bilder in slidev/public/ und Kapitel-Ordner
```
