# Didaktischer Plan — Applied Machine Learning (UZH)

**Kurs:** Applied Machine Learning  
**Zielgruppe:** Studierende mit Python-Grundkenntnissen, kein ML-Vorwissen nötig  
**Umfang:** 4 Sessions × 3 Kapitel × 50 min = 12 Kapitel (~10 Stunden)  
**Format:** Slides + Live-Demos + Übungen  
**Letztes Update:** August 2026 (Review-Überarbeitung: Leakage-Reihenfolge, KNN-Kapitel, Baselines, Quiz-Slides, GIF-Animationen, Capstone 50 min)

---

## Didaktische Leitprinzipien

| Prinzip | Umsetzung |
|---------|-----------|
| **Spiral-Curriculum** | Konzepte werden eingeführt, dann in späteren Kapiteln vertieft (z.B. Train/Test Split in Ch02 → Cross-Validation in Ch06) |
| **Hands-on first** | Jede theoretische Einheit endet mit Übungen, die das Gelernte sofort anwenden |
| **Motivierende Einstiege** | Jedes Kapitel startet mit einem realen Anwendungsfall oder einer provokanten Frage |
| **Visualisieren vor Formalisieren** | Intuition durch Plots und Analogien aufbauen, bevor Formeln eingeführt werden |
| **Sichtbare Fehler zeigen** | Bewusst fehlerhafte Ansätze demonstrieren (Data Leakage, Overfitting, Reward Hacking) um kritisches Denken zu fördern |
| **Baseline zuerst** | In jedem Kapitel mit Modellen zuerst `DummyClassifier`/`DummyRegressor` — jede Zahl braucht einen Vergleichswert |
| **Check for Understanding** | Jedes Kapitel hat eine Quiz-Slide (Frage → Handzeichen → Antwort per Klick) vor Demo/Übung |
| **Animieren statt behaupten** | Kernalgorithmen als GIF in den Slides (KNN-k, Polynomgrad, Gradient Descent, Threshold, K-Means, PCA-Rotation, Q-Table) — jeweils mit passendem Notebook in `0-animations/` |

---

## Gesamtübersicht der Sessions

| Session | Thema | Kapitel | Exercises | Kumulatives Thema |
|---------|-------|---------|-----------|-------------------|
| **1** | Daten & Grundlagen | Ch01 → Ch02 → Ch03 | Ch02: 10 min, Ch03: 10 min (Ch01: Bonus) | Vom Rohdatensatz zum ersten Modell (KNN) |
| **2** | Supervised Learning | Ch04 → Ch05 → Ch06 | Ch04: 10, Ch05: 12, Ch06: 10 min (+ Bonus) | Regression, Klassifikation, Evaluation |
| **3** | Unsupervised Learning | Ch07 → Ch08 → Ch09 | Ch08+Ch09: je 10 min (Ch07: Hausaufgabe) | Clustering & Dimensionsreduktion |
| **4** | RL + Capstone | Ch10 → Ch11 → Ch12 | Ch11: 10 min, Ch12: **50 min** (Ch10: Bonus) | Reinforcement Learning + Titanic Projekt (~155 min) |

---

## Session 1 — Daten verstehen & erste Modelle

**Dauer:** ~150 min (3 × 50 min)  
**Ziel:** Studierende können eine Datenmenge laden, bereinigen und ihr erstes Modell trainieren.  
**Roter Faden:** *"Daten sind das Fundament — ohne saubere Daten kein gutes Modell."*

### Timing Session 1

| Zeit | Kapitel | Block | Methode | Hinweis |
|------|---------|-------|---------|---------|
| 0–50 min | **[Ch01 — Einführung](1-introduction/01-slides/ch01_slides.md)** | Überblick ML, Paradigmen, Workflow (animiert), Baseline-Idee, Tools | Slides + Live-Demo + Quiz | Optionale Übung (Penguins EDA, offline CSV) als Bonus |
| 50–100 min | **[Ch02 — Daten selektieren & vorbereiten](2-selection_cleaning_preparing/01-slides/ch02_slides.md)** | Missing Values, Duplikate, Outlier (IQR), Encoding, Scaling, **Split zuerst**, `ColumnTransformer` + `Pipeline` | Slides + Übungen + Quiz | ⚠️ Data Leakage ist das zentrale Konzept — Übung ist jetzt in der richtigen Reihenfolge (Split → Impute → Encode/Scale) |
| 100–150 min | **[Ch03 — Supervised Learning Intro](3-supervised_learning/01-slides/ch03_slides.md)** | Supervised-Protokoll, Baseline, Over-/Underfitting, **KNN als roter Faden** (Vote → k → Scaling), Hyperparameter vs. Parameter, CV nur als Konzept-Preview | Slides + GIF + Demo + Übungen | Erstes echtes Modell → motivierend! Ein Algorithmus von Slide bis Übung |

### Didaktische Hinweise Session 1

- **Einstieg Ch01:** Frage an die Klasse: *"Wer hat heute schon ML benutzt?"* → Alle haben es (Spotify, Netflix, Google Maps). Sofort zeigen dass ML überall ist.
- **Ch02 Kernbotschaft:** Den Satz *"Impute after split!"* mehrfach wiederholen und begründen. Das ist der häufigste Anfängerfehler.
- **Ch02 Target Leakage:** Im Beispiel ist `grade` aus `score` abgeleitet — bewusst als zweiter Leakage-Typ (Feature verrät Target) neben Train/Test-Leakage besprechen.
- **Ch02 → Ch03 Übergang:** Ch02 endet mit `ColumnTransformer` + `Pipeline`; Ch03 hängt an dieselbe Pipeline einfach einen `KNeighborsClassifier` — *„die Pipeline IST das Modell"*.
- **Ch03 KNN-GIF:** `knn_boundary_k.gif` loopen lassen (k = 1 → 100): k = 1 Inseln (Overfitting), k ≈ 5–30 glatt, k = 100 fast Gerade (Underfitting). Rechts läuft die Train/Test-Kurve live mit — das ist die Bias/Variance-Kurve.
- **Baseline:** Ab Ch03 in jedem Kapitel zuerst `DummyClassifier` — „91 % klingt gut, bis man weiss, dass 90 % die Baseline ist".
- **Differenzierung:** Ch02 Bonus B (Mean vs. Median Imputation vergleichen), Ch03 Bonus (Penguins mit `SimpleImputer` + Scaler an/aus: 0.77 vs. 0.99).

**Materialien Session 1:**
- Slides: [`ch01_slides.md`](1-introduction/01-slides/ch01_slides.md) · [`ch02_slides.md`](2-selection_cleaning_preparing/01-slides/ch02_slides.md) · [`ch03_slides.md`](3-supervised_learning/01-slides/ch03_slides.md)
- Outline/Lehrplan: [`ch01_outline.md`](1-introduction/00-material/ch01_outline.md) · [`ch02_outline.md`](2-selection_cleaning_preparing/00-material/ch02_outline.md) · [`ch03_outline.md`](3-supervised_learning/00-material/ch03_outline.md)
- Übungen: [`ch01_introduction_exercises.ipynb`](1-introduction/03-exercises/) (Bonus) · [`ch02_data_cleaning_exercises.ipynb`](2-selection_cleaning_preparing/03-exercises/) · [`ch03_supervised_intro_exercises.ipynb`](3-supervised_learning/03-exercises/)
- Animationen: [`02_knn_decision_boundary.ipynb`](0-animations/02_knn_decision_boundary.ipynb) (identisch zum Slide-GIF)

---

## Session 2 — Supervised Learning: Regression, Klassifikation, Evaluation

**Dauer:** ~150 min (3 × 50 min)  
**Ziel:** Studierende verstehen Regression und Klassifikation als zwei Seiten von Supervised Learning und können Modelle korrekt evaluieren.  
**Roter Faden:** *"Welches Modell für welches Problem — und wie weiss ich ob es gut ist?"*

### Timing Session 2

| Zeit | Kapitel | Block | Methode | Hinweis |
|------|---------|-------|---------|---------|
| 0–50 min | **[Ch04 — Regression](3-supervised_learning/01-slides/ch04_slides.md)** | Baseline, Linear Regression, MSE/GD-Intuition (GIF), Polynom-Overfitting + **CV in der Praxis** (GIF), Ridge/Lasso (RidgeCV/GridSearchCV), Trees/RF | Slides + Übungen + Quiz | Polynomgrad-Sweep-GIF: Train- vs. CV-Fehler füllen sich live |
| 50–100 min | **[Ch05 — Klassifikation](3-supervised_learning/01-slides/ch05_slides.md)** | Baseline, Logistic Regression, KNN-Recap (2 min), Trees/RF (Depth-GIF), Decision Boundaries, **Confusion-Matrix lesen** (Mini-Slide vor der Übung); SVM nur Bonus | Slides + Übungen + Quiz | Positive Klasse explizit: Breast Cancer → malignant = 1 |
| 100–150 min | **[Ch06 — Evaluation & Metriken](3-supervised_learning/01-slides/ch06_slides.md)** | Accuracy-Paradox (echte Demo), Regressionsmetriken, Precision/Recall/F1 (Rechenbeispiel), Threshold-GIF, ROC vs. PR, Imbalance-Toolbox (`class_weight`, `StratifiedKFold`), CV mit Pipeline | Slides + Übungen + Quiz | ⚠️ Accuracy-Paradox jetzt real: Dummy 98 % / Recall 0 |

### Didaktische Hinweise Session 2

- **Ch04 Einstieg:** Hauspreisvorhersage — jede\*r kennt Immobilienpreise. *"Wie würdet ihr den Preis schätzen?"* → Intuitive Regression.
- **Ch05 Einstieg:** Spam-E-Mail Beispiel. *"Was macht ein Spam-Filter?"* → Binäre Klassifikation.
- **Ch06 Kernbotschaft:** "Accuracy ist oft eine Lüge." → Demo mit `make_classification(weights=[.98,.02])`: DummyClassifier 98 % Accuracy, Recall 0 — dann `class_weight='balanced'`. Dieses Beispiel prägt sich ein, weil man es sieht.
- **Positive Klasse:** Breast Cancer hat im Original 1 = benign. Wir flippen `y = 1 - target` (malignant = 1) und machen daraus die Lektion *„Was ist die positive Klasse? Immer prüfen!"* — sonst sind Precision/Recall/FN falsch herum.
- **Cross-Validation:** Konzept Ch03 → Praxis Ch04 (`KFold(shuffle=True)`, `RidgeCV`) → Ch05/06 mit passendem Scoring (`f1`, `recall`) und Pipeline im CV.
- **Ch06 Threshold-GIF:** Punkt wandert entlang ROC- und PR-Kurve, Confusion Matrix aktualisiert sich — Threshold ist eine Business-Entscheidung, nicht 0.5.
- **Pace:** Ch04 und Ch05 können zügig durch wenn Grundkonzept aus Ch03 sitzt. Zeit für Ch06 schützen — Metriken sind erfahrungsgemäss der am stärksten unterschätzte Teil.

**Materialien Session 2:**
- Slides: [`ch04_slides.md`](3-supervised_learning/01-slides/ch04_slides.md) · [`ch05_slides.md`](3-supervised_learning/01-slides/ch05_slides.md) · [`ch06_slides.md`](3-supervised_learning/01-slides/ch06_slides.md)
- Outline/Lehrplan: [`ch04_outline.md`](3-supervised_learning/00-material/ch04_outline.md) · [`ch05_outline.md`](3-supervised_learning/00-material/ch05_outline.md) · [`ch06_outline.md`](3-supervised_learning/00-material/ch06_outline.md)
- Übungen: [`ch04_regression_exercises.ipynb`](3-supervised_learning/03-exercises/) · [`ch05_classification_exercises.ipynb`](3-supervised_learning/03-exercises/) · [`ch06_metrics_exercises.ipynb`](3-supervised_learning/03-exercises/)
- Animationen: [`03_gradient_descent.ipynb`](0-animations/03_gradient_descent.ipynb) · [`05_polynomial_overfitting.ipynb`](0-animations/05_polynomial_overfitting.ipynb) (beide Ch04)

---

## Session 3 — Unsupervised Learning: Clustering & Dimensionsreduktion

**Dauer:** ~150 min (3 × 50 min)  
**Ziel:** Studierende können ungelabelte Daten mit Clustering und PCA explorieren.  
**Roter Faden:** *"Was lernt ein Algorithmus, wenn es keine richtigen Antworten gibt?"*

### Timing Session 3

| Zeit | Kapitel | Block | Methode | Hinweis |
|------|---------|-------|---------|---------|
| 0–50 min | **[Ch07 — Unsupervised Learning Intro](4-unsupervised_learning/01-slides/ch07_slides.md)** | Drei Fragen (Gruppen, Struktur, Ausreisser), „Cluster ≠ Klassen", Anomalie-Erkennung (IsolationForest), GMM-Erwähnung | Slides + Diskussion + Demo (12 min) + Quiz | Übung optional / Hausaufgabe (~15 min) |
| 50–100 min | **[Ch08 — Clustering (K-Means)](4-unsupervised_learning/01-slides/ch08_slides.md)** | K-Means (GIF), Init/`n_init`, **Scale first**, Elbow + Silhouette, „Clustering funktioniert immer" (Uniform-Noise), DBSCAN; Hierarchical nur Bonus | Slides + Übungen + Quiz | K-Means-GIF im Slide; Elbow (3) vs. Silhouette (2) auf Iris ist ein Lehrmoment |
| 100–150 min | **[Ch09 — Dimensionsreduktion (PCA)](4-unsupervised_learning/01-slides/ch09_slides.md)** | Curse of Dimensionality (GIF), PCA (Rotations-GIF), **Loadings: PCA ≠ Feature-Selektion**, zwei Regime (2 PCs vs. 95 %), PCA in Pipeline (Leakage), t-SNE (kein `transform()`) | Slides + Übungen + Quiz | PCA-Rotations-GIF: Achse dreht, erklärte Varianz läuft mit |

### Didaktische Hinweise Session 3

- **Ch07 Einstieg:** *"Was haben Kundensegmentierung, Genexpression und Spracherkennung gemeinsam?"* → Alle nutzen Unsupervised Learning. Macht den Paradigmenwechsel klar: keine Labels.
- **Ch08 Animation:** `kmeans_iterations.gif` im Slide (assign → update, Inertia-Zähler); im Notebook [`01_kmeans_convergence.ipynb`](0-animations/01_kmeans_convergence.ipynb) mit `SEED`-Knopf (SEED 1/20 = schlechtes lokales Optimum → warum `n_init` und k-means++).
- **Ch08 Kernbotschaft:** K-Means liefert *immer* k Cluster — Uniform-Noise-Demo zeigen (Silhouette 0.38 vs. 0.83). Silhouette-Wert prüfen, bevor man Cluster interpretiert. Elbow (3) vs. Silhouette (2) auf Iris ist ein Lehrmoment, kein Fehler.
- **Ch09 Kernbotschaft:** PCA ist Dimensionsreduktion, nicht Merkmals-Selektion — jetzt mit Loadings-Slide (PC1 = gewichtete Kombination aller Features). Zwei Regime: 2 PCs zum Anschauen, `n_components=0.95` zum Vorverarbeiten.
- **Verbindung zu Session 2:** *"In Session 2 hatten wir Labels, jetzt nicht mehr. Wie evaluieren wir also unsere Modelle?"* → Silhouette Score, Elbow Method; Labels höchstens als Sanity-Check (ARI), nie als Ziel. Cross-Validation kommt in Ch09 zurück (PCA in Pipeline, `StratifiedKFold`).

**Materialien Session 3:**
- Slides: [`ch07_slides.md`](4-unsupervised_learning/01-slides/ch07_slides.md) · [`ch08_slides.md`](4-unsupervised_learning/01-slides/ch08_slides.md) · [`ch09_slides.md`](4-unsupervised_learning/01-slides/ch09_slides.md)
- Outline/Lehrplan: [`ch07_outline.md`](4-unsupervised_learning/00-material/ch07_outline.md) · [`ch08_outline.md`](4-unsupervised_learning/00-material/ch08_outline.md) · [`ch09_outline.md`](4-unsupervised_learning/00-material/ch09_outline.md)
- Übungen: [`ch07_unsupervised_intro_exercises.ipynb`](4-unsupervised_learning/03-exercises/) (Bonus) · [`ch08_clustering_exercises.ipynb`](4-unsupervised_learning/03-exercises/) · [`ch09_dimensionality_reduction_exercises.ipynb`](4-unsupervised_learning/03-exercises/)
- Animationen: [`01_kmeans_convergence.ipynb`](0-animations/01_kmeans_convergence.ipynb) (identisch zum Slide-GIF, mit `SEED`-Knopf)

---

## Session 4 — Reinforcement Learning & Capstone-Projekt

**Dauer:** ~155 min (45 + 45 + 65)  
**Ziel:** Studierende verstehen das RL-Paradigma und können ein vollständiges ML-Projekt selbstständig durchführen.  
**Roter Faden:** *"Lernen durch Feedback — und alles zusammenbringen."*

### Timing Session 4

| Zeit | Kapitel | Block | Methode | Hinweis |
|------|---------|-------|---------|---------|
| 0–45 min | **[Ch10 — RL Einführung](5-reinforcement_learning/01-slides/ch10_slides.md)** | Agent, Environment, Reward, Policy, Credit Assignment, GridWorld, deterministisch vs. slippery, ε-Decay, Q-Table | Slides + Demo + Quiz | Übung (Reward Shaping mit gelerntem Agenten) als Bonus |
| 45–90 min | **[Ch11 — Q-Learning](5-reinforcement_learning/01-slides/ch11_slides.md)** | Q-Table-GIF, TD-Update (aus der Bellman-Optimalitätsgleichung), ε-Greedy, FrozenLake (gymnasium, slippery ≈ 70 %); SARSA/PG nur Further Reading | Slides + Übungen (10 min) + Quiz | Q-Table-GIF + Pfad-GIF (Episode 1 / 50 / 3000) |
| 90–155 min | **[Ch12 — Capstone: Titanic](6-capstone_ml/01-slides/ch12_slides.md)** | End-to-End: Split zuerst → Features (Title) → `ColumnTransformer`+`Pipeline` → Baseline → `cross_validate` → Test → `permutation_importance` → Error Analysis → Fairness | Geführtes Projekt (5 Intro + **50 min Selbstarbeit** + 10 Debrief) | Kaggle-CSV `0-datasets/titanic.csv` |

### Didaktische Hinweise Session 4

- **Ch10 Einstieg:** *"Wie lernt ein Hund, einen Trick zu machen?"* → Belohnung → Verstärkendes Lernen. Dann: AlphaGo, Roboter, Trading-Algorithmen.
- **Ch11 TD-Update:** Nicht „die Bellman-Gleichung" nennen — es ist das TD-Update, abgeleitet aus der Bellman-Optimalitätsgleichung. Zwei Slides: Idee, dann ein durchgerechneter Schritt mit den Zahlen aus der Übung. Q-Table-GIF zeigen (Heatmap füllt sich, Greedy-Pfeile drehen sich).
- **Ch11 Hyperparameter** überall gleich: α = 0.1, γ = 0.99, ε 1 → 0.01 (×0.999). Trainings-Erfolgsrate ≠ Greedy-Erfolgsrate — separat evaluieren.
- **Ch11 SARSA (Bonus):** Wenn SARSA schlechter abschneidet, ist das der Lehrmoment (on-policy = konservativ) — Text im Notebook ist ergebnisabhängig.
- **Ch12 Capstone:** Das ist der Höhepunkt des Kurses. **Nicht unterbrechen während der 50 Min Übungsphase.** Herumgehen, helfen, aber keine Frontalphase einschalten. (Kein separates Example-Notebook — die geführte Übung IST das Beispiel.) Kernregel sichtbar: Split zuerst, Imputation/Encoding **in** der Pipeline.
- **Ch12 Debriefing (letzte 10 min):** Leaderboard-GIF, Ergebnisse vergleichen (F1, nicht nur Accuracy) — LogReg ≈ GB ≈ RF, Reihenfolge kann kippen; was hat geholfen (Title-Feature)? `permutation_importance`: Sex, Title, Pclass → Fairness-Frage: „Sensible Spalte löschen = fair?" (Proxies). Verbindung zu allen vorherigen Kapiteln ziehen.

**Materialien Session 4:**
- Slides: [`ch10_slides.md`](5-reinforcement_learning/01-slides/ch10_slides.md) · [`ch11_slides.md`](5-reinforcement_learning/01-slides/ch11_slides.md) · [`ch12_slides.md`](6-capstone_ml/01-slides/ch12_slides.md)
- Outline/Lehrplan: [`ch10_outline.md`](5-reinforcement_learning/00-material/ch10_outline.md) · [`ch11_outline.md`](5-reinforcement_learning/00-material/ch11_outline.md) · [`ch12_outline.md`](6-capstone_ml/00-material/ch12_outline.md)
- Übungen: [`ch10_rl_intro_exercises.ipynb`](5-reinforcement_learning/03-exercises/) (Bonus) · [`ch11_rl_algorithms_exercises.ipynb`](5-reinforcement_learning/03-exercises/) · [`ch12_capstone_exercises.ipynb`](6-capstone_ml/03-exercises/)
- Animationen: [`04_rl_agent_learning.ipynb`](0-animations/04_rl_agent_learning.ipynb) (gleiche GridWorld/Hyperparameter wie Ch10/11)

---

## Kapitelübersicht auf einen Blick

| Ch | Titel | Session | Format | Übung | Slides | Outline |
|----|-------|---------|--------|-------|--------|---------|
| 01 | Einführung in ML | S1 | Slides + Demo + Quiz | (Bonus) | [↗](1-introduction/01-slides/ch01_slides.md) | [↗](1-introduction/00-material/ch01_outline.md) |
| 02 | Daten auswählen & vorbereiten | S1 | Slides + Übung | 10 min | [↗](2-selection_cleaning_preparing/01-slides/ch02_slides.md) | [↗](2-selection_cleaning_preparing/00-material/ch02_outline.md) |
| 03 | Supervised Learning Intro (KNN) | S1 | Slides + GIF + Übung | 10 min | [↗](3-supervised_learning/01-slides/ch03_slides.md) | [↗](3-supervised_learning/00-material/ch03_outline.md) |
| 04 | Regression | S2 | Slides + Übung | 10 min | [↗](3-supervised_learning/01-slides/ch04_slides.md) | [↗](3-supervised_learning/00-material/ch04_outline.md) |
| 05 | Klassifikation (SVM Bonus) | S2 | Slides + Übung | 12 min | [↗](3-supervised_learning/01-slides/ch05_slides.md) | [↗](3-supervised_learning/00-material/ch05_outline.md) |
| 06 | Evaluation & Metriken | S2 | Slides + Übung | 10 min | [↗](3-supervised_learning/01-slides/ch06_slides.md) | [↗](3-supervised_learning/00-material/ch06_outline.md) |
| 07 | Unsupervised Learning Intro | S3 | Slides + Diskussion + Demo | (Hausaufgabe) | [↗](4-unsupervised_learning/01-slides/ch07_slides.md) | [↗](4-unsupervised_learning/00-material/ch07_outline.md) |
| 08 | Clustering (K-Means, DBSCAN) | S3 | Slides + GIF + Übung | 10 min | [↗](4-unsupervised_learning/01-slides/ch08_slides.md) | [↗](4-unsupervised_learning/00-material/ch08_outline.md) |
| 09 | Dimensionsreduktion (PCA) | S3 | Slides + Übung | 10 min | [↗](4-unsupervised_learning/01-slides/ch09_slides.md) | [↗](4-unsupervised_learning/00-material/ch09_outline.md) |
| 10 | Reinforcement Learning Intro | S4 | Slides + Demo + Quiz | (Bonus) | [↗](5-reinforcement_learning/01-slides/ch10_slides.md) | [↗](5-reinforcement_learning/00-material/ch10_outline.md) |
| 11 | Q-Learning | S4 | Slides + Übung | 10 min | [↗](5-reinforcement_learning/01-slides/ch11_slides.md) | [↗](5-reinforcement_learning/00-material/ch11_outline.md) |
| 12 | Capstone: Titanic | S4 | Guided Project | **50 min** | [↗](6-capstone_ml/01-slides/ch12_slides.md) | [↗](6-capstone_ml/00-material/ch12_outline.md) |

---

## Konzeptionelle Progressionskarte

```
Session 1                 Session 2                 Session 3                 Session 4
─────────────────────────────────────────────────────────────────────────────────────────
Was ist ML?          →   Regression              →   Was ohne Labels?    →   Agent & Reward
  │                          │                           │                      │
Daten vorbereiten    →   Klassifikation          →   K-Means             →   Q-Learning
  │                          │                           │                      │
Erstes Modell (KNN)  →   Evaluieren & Metriken  →   PCA / t-SNE         →   Titanic Projekt
─────────────────────────────────────────────────────────────────────────────────────────
         └── DATEN ──────────────────────────────────────────────────────────┘
              Train/Test Split  ←→  Imputation  ←→  Encoding  ←→  Scaling
```

**Konzept-Querverbindungen (für Lehrgespräche nutzen):**

| Konzept | Eingeführt | Vertieft | Angewendet |
|---------|-----------|---------|-----------|
| Train/Test Split | Ch02 | Ch03 (Protokoll) | Ch12 (stratify, Split zuerst) |
| Cross-Validation | Ch03 (Konzept) | Ch04 (Praxis, KFold shuffle, RidgeCV) | Ch05/06 (Scoring), Ch09 (PCA-Pipeline), Ch12 |
| Baseline | Ch03 (Dummy) | Ch04–Ch06 | Ch09, Ch12 |
| Overfitting | Ch03 (KNN k=1) | Ch04 (Polynomgrad-GIF), Ch05 (Tree Depth) | Ch06, Ch12 |
| Feature Encoding / Imputation | Ch02 (SimpleImputer, OneHot) | Ch03 (Penguins-Bonus) | Ch12 (in ColumnTransformer) |
| Pipeline | Ch02 (ColumnTransformer) | Ch03 (mit KNN), Ch04, Ch08 (KMeans), Ch09 (PCA) | Ch12 |
| Leakage (Train/Test + Target) | Ch02 (`grade`) | Ch09 (PCA fit on train) | Ch12 |
| Evaluation / Metriken | Ch03 (Accuracy vs. Baseline) | Ch05 (Confusion Matrix), Ch06 | Ch12 (F1, permutation_importance) |
| Positive Klasse | Ch05 (Breast Cancer geflippt) | Ch06 | Ch12 (Survived = 1) |

---

## Häufige Missverständnisse (Lehrerhinweise)

| Missverständnis | Kapitel | Gegenmassnahme |
|----------------|---------|----------------|
| "Mehr Daten → immer besser" | Ch02 | Qualität vor Quantität zeigen; verseuchte Daten verschlechtern alles |
| "Höhere Accuracy = besseres Modell" | Ch06 | Accuracy Paradox mit Krebs-Screening-Beispiel |
| "Man imputed vor dem Split" | Ch02 | Data Leakage Pipeline live durchführen und Fehler sichtbar machen |
| "Unsupervised learning braucht kein Evaluation" | Ch08 | Silhouette Score, Elbow Method einführen |
| "RL ist gleich wie supervised" | Ch10 | Feedback-Loop vs. feste Labels — grundlegend verschieden |
| "K-Means findet immer die 'richtigen' Cluster" | Ch08 | `kmeans_init.png` (random vs. k-means++), Animation mit `SEED`-Knopf, Uniform-Noise-Slide |
| "Trainings-Accuracy = Modellgüte" | Ch03 | KNN k=1: Train 100 %, Test 87 % — nur Held-out zählt |
| "Feature enthält heimlich das Target" (Target Leakage) | Ch02 | `grade` aus `score` abgeleitet → groupby zeigen, droppen |
| "Positive Klasse = 1 ist immer 'das Gesuchte'" | Ch05/06 | Breast Cancer: 1 = benign im Original → flippen, Recall neu lesen |
| "t-SNE-Cluster = echte Cluster / t-SNE als Features" | Ch09 | Zwei Seeds vergleichen; kein `transform()` → nur Visualisierung |
| "Niedrige erklärte Varianz = PCA gescheitert" | Ch09 | Zwei Regime (2 PCs für Plot vs. 95 % fürs Preprocessing); Microbiome-Notebook |
| "Trainings-Erfolgsrate = Qualität der Policy" | Ch11 | Greedy-Policy separat evaluieren (ε = 0) |
| "Sensible Spalte löschen = fair" | Ch12 | Proxies (Title, Fare, Pclass) — Reflexionsfrage im Debrief |

---

## Verwendete Datensätze

| Datensatz | Herkunft | Verwendet in |
|-----------|---------|-------------|
| Iris | `sklearn.datasets.load_iris()` | Ch01, Ch07 (petal), Ch08 |
| Penguins | `1-introduction/03-exercises/penguins.csv` (offline; seaborn-Fallback) | Ch01 (Bonus), Ch03 (Demo/Bonus) |
| California Housing | `sklearn.datasets.fetch_california_housing()` | Ch04, Ch06 |
| Breast Cancer | `sklearn.datasets.load_breast_cancer()` — **Target geflippt: malignant = 1** | Ch05, Ch06, Ch09 |
| Diabetes | `sklearn.datasets.load_diabetes()` | Ch04 (Exercises), Ch06 |
| Wine | `sklearn.datasets.load_wine()` (3 Rebsorten) | Ch05 (Exercises) |
| Digits | `sklearn.datasets.load_digits()` | Ch07 (Teaser/Übung), Ch09 |
| Titanic | `0-datasets/titanic.csv` (Kaggle train.csv, 891 Zeilen) | Ch12 |
| FrozenLake | `gymnasium.make('FrozenLake-v1', is_slippery=...)` (GridWorld-Fallback) | Ch11 |
| Synthetische Daten | `make_blobs`, `make_moons` (Ch03, Ch05), `make_classification(weights=[.98,.02])` (Ch06), eigene | Ch02, Ch03, Ch05, Ch06, Ch07, Ch08 |

---

## Vorbereitung pro Session — Checkliste

```
Vor jeder Session:
✅ Jupyter Notebook Server starten
✅ Slidev bereit: `./slides.sh <N>` startet die Slides
✅ Lösungs-Notebooks bereit (aber nicht geöffnet)
✅ Kurze Review der Übungsaufgaben der letzten Session

Session 1:
✅ seaborn + sklearn installiert (requirements.txt, sklearn ≥ 1.4)
✅ Iris-Datensatz erreichbar (sklearn, offline verfügbar)
✅ Penguins-CSV vorhanden (1-introduction/03-exercises/penguins.csv)

Session 4 (RL + Capstone):
✅ gymnasium installiert (FrozenLake); sonst GridWorld-Fallback
✅ 0-datasets/titanic.csv vorhanden
✅ 50 Minuten ungestörte Arbeitszeit einplanen
✅ Debriefing vorbereiten: "Was hat eurem Modell am meisten gebracht?" + Fairness-Frage
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

**Bilder & GIF-Animationen** bei Bedarf neu generieren:
```bash
python generate_images.py   # → PNGs + GIFs in slidev/public/ und Kapitel-Ordner
```
Pro Kapitel liegt ein Modul `imagegen/chNN.py` (statische Bilder + GIFs via `imagegen.common.save_gif`).
GIFs zeigen im PDF-Export nur das erste Frame — daneben liegt jeweils ein `<name>.png` mit dem letzten Frame als statischer Ersatz.
