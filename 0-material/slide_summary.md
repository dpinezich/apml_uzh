# Applied Machine Learning — Zusammenfassung aller Slidesets

Diese Datei fasst die 12 Kapitel des APML-Kurses zusammen und erklärt die wichtigsten Fakten,
Konzepte und Begriffe pro Slideset. Reihenfolge entspricht der Vorlesung
(4 Sessions × 3 Kapitel).

---

## Session 1 — Grundlagen & Datenaufbereitung

### Kapitel 1 — Introduction to Machine Learning & Data Science Workflows

**Ziel:** Was ist ML, welche Paradigmen gibt es, wie sieht der Data-Science-Workflow aus.

**Kernfakten:**
- **Klassische Programmierung:** Regeln + Daten → Output. **ML:** Daten + Output → Regeln.
- Mitchell (1997): Lernen aus Erfahrung E bzgl. Aufgabe T mit Performance-Maß P.
- **Drei Paradigmen:**
  - *Supervised Learning* — gelabelte Daten, lerne X → y; entweder **Regression** (kontinuierlich) oder **Classification** (diskret).
  - *Unsupervised Learning* — keine Labels, finde verborgene Struktur (Segmentierung, Anomalien, Topic Modeling).
  - *Reinforcement Learning* — Agent ↔ Umwelt: Action → Reward → State; Ziel: kumulativen Reward maximieren.
- **Data-Science-Workflow ist ein Zyklus, keine Pipeline:** ① Problem definieren ② Daten ③ EDA ④ Preprocessing ⑤ Training ⑥ Evaluation ⑦ Deployment & Iterate.
- **80% der Projektzeit** liegen in Schritten 2–4 (Daten + EDA + Preprocessing).
- Die meisten ML-Projekte scheitern an unsauberer Problemdefinition oder schlechten Daten — nicht am Algorithmus.
- **Python-ML-Ökosystem:** numpy, pandas, matplotlib/seaborn, scikit-learn, gymnasium (RL).

**Take-away:** ML lernt Muster automatisch aus Daten; der Workflow ist iterativ und datengetrieben.

---

### Kapitel 2 — Data Selection, Cleaning & Preparing

**Ziel:** Reale Daten aufräumen, ohne Data Leakage zu produzieren.

**Kernfakten:**
- **Garbage in → Garbage out.** Datenqualität begrenzt die Modellqualität.
- **Typische Probleme:** fehlende Werte (NaN, leerer String, `-999`), falsche Datentypen, inkonsistente Kategorien (`"Male"` vs. `"male"` vs. `"M"`), Outlier, Duplikate, gemischte Skalen.
- **Missing Values:**
  - Detektion: `df.isnull().sum()` / `.mean()`.
  - Faustregel: <5% fehlend → Zeilen droppen; >50% in einer Spalte → Spalte droppen.
  - Imputation: Mean (numerisch, ohne Outlier), Median (numerisch, mit Outlier), Mode (kategorial), Konstante (Domain).
- **Outlier:** Boxplot, IQR-Regel (`Q1−1.5·IQR` / `Q3+1.5·IQR`), Z-Score `|z|>3`. Behandlung: entfernen, kappen, log-transformieren oder behalten.
- **Feature-Typen:** numerisch (kont./diskret), kategorial (nominal/ordinal), binär. Encoding via One-Hot oder ordinal; bei vielen Kategorien Target-Encoding erwägen.
- **Skalierung:** `StandardScaler` (Mean=0, Std=1), `MinMaxScaler` (0–1). Tree-basierte Modelle brauchen **keine** Skalierung.
- **Train/Test-Split** mit `random_state` für Reproduzierbarkeit; `stratify=y` bei Klassen-Imbalance.
- **⚠️ Data Leakage:** Imputation und Skalierung **immer NACH dem Split** — `fit()` nur auf Train, `transform()` auf beide. Lösung: `sklearn.pipeline.Pipeline`.
- **Goldene Regel:** Alles, was Statistiken berechnet (Mean, Median, IQR, Scaler-Parameter), wird ausschließlich auf Trainingsdaten gefittet.

**Take-away:** Zuerst splitten, dann preprocessen — Pipelines verhindern Leakage automatisch.

---

## Session 2 — Supervised Learning

### Kapitel 3 — Introduction to Supervised Learning

**Ziel:** Grundbegriffe (Loss, Generalisierung, Bias-Variance, Cross-Validation) und die sklearn-API.

**Kernfakten:**
- Eingabe: gelabelte Paare (X, y); Ausgabe: Funktion f(X) → ŷ.
- **Regression vs. Classification:** kontinuierliche Zahl (z.B. Hauspreis) vs. diskrete Kategorie (Spam/Ham).
- **Train Error vs. Test Error:** uns interessiert der Test-Fehler — nur er misst Generalisierung.
- **Loss Functions:** Regression → MSE = (1/n)Σ(yᵢ−ŷᵢ)²; Classification → Cross-Entropy −Σ yᵢ log(ŷᵢ). Training = Loss minimieren.
- **Underfitting:** Modell zu einfach → hoher Train-Fehler UND hoher Test-Fehler.
- **Overfitting:** Modell zu komplex → niedriger Train-Fehler, hoher Test-Fehler (memoriert das Rauschen).
- **Bias-Variance-Tradeoff:** U-Kurve — der Sweet-Spot minimiert die Summe.
- **Cross-Validation** (z.B. `cv=5`) liefert robustere Performance-Schätzungen als ein einzelner Split.
- **sklearn-API:** `model = SomeModel(...)` → `model.fit(X_train, y_train)` → `model.predict(X_test)` → `model.score(...)`. Jedes sklearn-Modell folgt diesem Schema.

**Take-away:** Ziel ist Generalisierung; Cross-Validation und das Verständnis von Bias/Variance verhindern Selbstbetrug.

---

### Kapitel 4 — Regression Models

**Ziel:** Klassische Regressionsalgorithmen verstehen und vergleichen.

**Kernfakten:**
- **Bewertungsmetriken:** MAE (mittlerer absoluter Fehler), RMSE (bestraft große Fehler stärker), R² (erklärte Varianz).
- **Linear Regression:** ŷ = β₀ + β₁x₁ + … + βₙxₙ; Lernen via OLS (Sum of Squared Residuals minimieren). Schnell, interpretierbar, gute Baseline; nimmt linearen Zusammenhang an.
- **Coefficient-Interpretation:** Bei `area_sqm` mit Koeffizient 3.5 → +1 m² ≈ +€3.500 Preis (ceteris paribus).
- **Polynomial Regression:** Hinzufügen von x², x³, … via `PolynomialFeatures`; höherer Grad → Overfitting-Gefahr → immer cross-validieren.
- **Regularisierung:**
  - *Ridge (L2):* Strafe = α·Σβᵢ² → schrumpft alle Koeffizienten gegen 0.
  - *Lasso (L1):* Strafe = α·Σ|βᵢ| → setzt Koeffizienten exakt auf 0 → automatische Feature-Selection.
  - α steuert die Stärke der Regularisierung.
- **Decision Tree Regression:** Splittet Feature-Raum, sagt Mittelwert pro Region voraus. Kein Scaling nötig, fängt Nichtlinearität ein, neigt aber zum Overfitting → `max_depth` begrenzen.
- **Random Forest:** Ensemble vieler Bäume auf zufälligen Subsamples → mittelt Vorhersagen. Reduziert Varianz, liefert `feature_importances_`. Fast immer besser als ein einzelner Baum.
- **Faustregel:** Mit Linear/Ridge starten als Baseline; Tree-Modelle bei vermuteter Nichtlinearität.

**Take-away:** Klein anfangen (linear), dann komplexer werden (Regularisierung → Trees → Forest); immer mit CV vergleichen.

---

### Kapitel 5 — Classification Models

**Ziel:** Diskrete Vorhersagen — Algorithmen und ihre Decision Boundaries.

**Kernfakten:**
- **Output:** harter Klassen-Label (`predict`) oder Wahrscheinlichkeiten (`predict_proba`).
- **Logistic Regression:** trotz Name **Klassifikation**! P(y=1|X) = σ(βᵀx) mit Sigmoid σ(z)=1/(1+e^−z); Decision Boundary linear; Schwelle standardmäßig bei 0.5.
- **K-Nearest Neighbors (KNN):** Mehrheitsentscheid unter den k nächsten Nachbarn. **Skalierung erforderlich** (distanzbasiert!). k=1 → Overfitting; großes k → Underfitting; k≈√n als Daumenwert; immer cross-validieren. Langsam bei großen Datasets.
- **Decision Tree Classifier:** Folge von Ja/Nein-Fragen; vollständig interpretierbar; visualisierbar mit `plot_tree`.
- **Random Forest Classifier:** Ensemble vieler Bäume → reduziert Varianz, robust gegen Outlier, liefert Feature Importances.
- **Support Vector Machine (SVM):** Hyperplane mit maximalem Margin; `C` steuert Margin-Breite (kleines C → mehr Regularisierung); Kernel-Trick (`kernel='rbf'`) ermöglicht nichtlineare Boundaries.
- Jeder Algorithmus zeichnet eine andere Decision Boundary — Visualisierung im Notebook lohnt.

**Take-away:** Logistic als Baseline; KNN nur skaliert; Trees für Interpretierbarkeit; Forest/SVM für Robustheit/Power.

---

### Kapitel 6 — Metrics & Evaluation

**Ziel:** Die richtige Metrik wählen — Accuracy ist oft irreführend.

**Kernfakten:**
- **Accuracy-Paradoxon:** Bei 99% gesunden Patienten erreicht "alles gesund" 99% Accuracy bei 0% Recall — nutzlos.
- **Regressionsmetriken:** MAE (gleiche Einheit wie y), MSE (y²), RMSE (=√MSE, bestraft große Fehler), R² (erklärte Varianz; <0 = schlechter als Mittelwert).
- **R² ist domänenabhängig:** Finance R²=0.1 oft beeindruckend, Physik R²=0.99 erwartet.
- **Confusion Matrix:** TP, TN, FP, FN.
  - **FP** = positiv vorhergesagt, eigentlich negativ (Spam-Filter: gute Mail im Spam-Ordner).
  - **FN** = negativ vorhergesagt, eigentlich positiv (Krankheit übersehen).
- **Precision** = TP / (TP+FP) — "Wie oft liege ich richtig, wenn ich positiv sage?"
- **Recall** = TP / (TP+FN) — "Wie viele echte Positiven habe ich gefunden?"
- **F1** = 2·Precision·Recall / (Precision+Recall) — balancierte Zusammenfassung.
- **Wann was?** Precision: FP teuer (Spam-Filter). Recall: FN teuer (Krankheits-Screening). F1: beides wichtig. Accuracy: nur bei balancierten Klassen.
- **Threshold-Tuning:** Standard 0.5 ist nicht heilig — Schwelle nach Business-Kontext setzen.
- **ROC-Kurve & AUC:** TPR vs. FPR über alle Schwellen. AUC=1 perfekt, AUC=0.5 zufällig. AUC ist threshold-unabhängig.
- **Modellvergleich** mittels `cross_val_score(..., scoring='f1')` — Mean **und** Std beachten.

**Take-away:** Wähle die Metrik, die deine reale Fehlerkosten widerspiegelt — und vergleiche Modelle immer per Cross-Validation.

---

## Session 3 — Unsupervised Learning

### Kapitel 7 — Introduction to Unsupervised Learning

**Ziel:** Lernen ohne Labels — warum, was, wie evaluieren.

**Kernfakten:**
- Kein Label, keine "richtige Antwort" — nur Daten.
- Labeling ist teuer, manchmal unmöglich oder noch nicht definiert. **Die meisten realen Daten sind ungelabelt.**
- **Frage ändert sich:** Supervised "Was ist das?" → Unsupervised "Was ist *darin*?"
- **Drei Typen:**
  - *Clustering* (Ch08) — ähnliche Samples gruppieren.
  - *Dimensionality Reduction* (Ch09) — viele Features auf wenige reduzieren.
  - *Density Estimation* — Wahrscheinlichkeitsverteilung der Daten modellieren (z.B. Anomaly Detection).
- **Evaluations-Herausforderung:**
  - *Interne Metriken:* Silhouette Score, Inertia (kein Label nötig).
  - *Externe Validierung:* Domänen-Expertise, Business-Sinn, Downstream-Task.
- **5-Schritte-Eval:** Visualisieren (PCA/t-SNE) → interne Metriken → externe Validierung (ARI, falls Labels vorhanden) → Downstream-Task → Domain-Experte.
- **Anwendungen:** Marketing (Segmente), Medizin (Subtypen), NLP (Topics), Finanzen (Anomalien), Vision (Kompression), Visualisierung (t-SNE/UMAP).

**Take-away:** Es gibt keine einzige "richtige" Lösung — mehrere valide Gruppierungen können koexistieren; Domänen-Wissen ist essenziell.

---

### Kapitel 8 — Clustering Techniques

**Ziel:** Drei Clustering-Algorithmen verstehen und vergleichen.

**Kernfakten:**
- Ziel: ähnlich innerhalb, unähnlich zwischen Clustern.
- **K-Means** (4 Schritte):
  1. k zufällige Centroids wählen.
  2. Jeden Punkt dem nächsten Centroid zuordnen.
  3. Centroid auf Cluster-Mittelwert verschieben.
  4. Bis Konvergenz wiederholen.
  - Optimiert: Summe der quadrierten Distanzen (Inertia).
  - Wichtige Parameter: `n_clusters`, `init='k-means++'`, `n_init=10`, `random_state`.
- **k wählen:**
  - *Elbow Method:* Inertia vs. k plotten, Knick suchen (oft mehrdeutig).
  - *Silhouette Score:* (b−a)/max(a,b); +1 gut, 0 Grenze, −1 falsch — höchsten Wert wählen.
- **K-Means-Limits:** sphärische Cluster nötig; k muss vorgegeben werden; sensitiv für Outlier; lokale Minima (deshalb `n_init>1`).
- **Hierarchical Clustering (Agglomerativ):** Bottom-up Merging → Dendrogramm. Kein k im Voraus nötig — auf gewünschter Höhe schneiden. Lange vertikale Linien = natürliche Lücken.
- **DBSCAN:** Dichtebasiert; Cluster = dichte Regionen, getrennt durch dünne. Findet **beliebige Formen** und markiert **Outlier** (Label = −1). Parameter: `eps`, `min_samples`.
- **Vergleich:** K-Means schnell aber sphärisch; Hierarchical liefert Hierarchie aber langsam; DBSCAN flexibel und outlier-robust.

**Take-away:** Unbekanntes k & sphärische Daten → K-Means; Hierarchie nötig → Agglomerative; Outlier oder unregelmäßige Formen → DBSCAN.

---

### Kapitel 9 — Dimensionality Reduction

**Ziel:** Hochdimensionale Daten verstehen, komprimieren und visualisieren.

**Kernfakten:**
- **Curse of Dimensionality:** Mit wachsenden Dimensionen werden Daten dünn → distanzbasierte Algorithmen versagen, exponentiell mehr Daten nötig, Visualisierung unmöglich, Overfitting steigt.
- **Blessing of Dimensionality:** Hochdim. Daten haben oft niedrige *intrinsische* Dimension (Beispiel Gesichter: Mio. Pixel, ~50 sinnvolle Dimensionen).
- **PCA (Principal Component Analysis):**
  - Findet Richtungen maximaler Varianz: PC1 die meiste, PC2 die zweitmeiste (orthogonal zu PC1) usw.
  - **⚠️ Vor PCA immer `StandardScaler`** — PCA ist skalensensitiv. Pipeline verwenden.
  - `explained_variance_ratio_` zeigt, wie viel Varianz pro Komponente erhalten bleibt.
  - **Komponenten wählen** über Scree-Plot der kumulierten Varianz; Faustregel: ≥95%. Mit `PCA(n_components=0.95)` automatisch.
  - Doppelnutzen: Visualisierung (2D-Projektion) UND Preprocessing (schneller, weniger Overfitting).
- **t-SNE (t-Distributed Stochastic Neighbor Embedding):**
  - Nichtlinear, bewahrt **lokale** Nachbarschaft. Parameter `perplexity` (5–50).
  - **Nur für Visualisierung — niemals als Preprocessing.**
  - **⚠️ Cluster-Größe und Distanzen in t-SNE sind NICHT aussagekräftig.** Anzahl der "Cluster" hängt von Perplexity ab.
  - Erlaubt: "Diese Punkte sind ähnlich." Verboten: "Cluster A liegt 3× weiter weg als B."
- **PCA vs. t-SNE vs. UMAP:** PCA linear & schnell (für ML-Prep + Visualisierung); t-SNE & UMAP nichtlinear (nur Visualisierung). Default: PCA fürs Preprocessing, t-SNE/UMAP fürs Auge.

**Take-away:** Erst skalieren, dann PCA; t-SNE-Plots niemals überinterpretieren.

---

## Session 4 — Reinforcement Learning & Capstone

### Kapitel 10 — Introduction to Reinforcement Learning

**Ziel:** Das dritte Paradigma — Lernen durch Trial & Error mit Belohnungen.

**Kernfakten:**
- **Signal-Unterschied:** Supervised hat (X,y), Unsupervised nur X, **RL hat Rewards aus Trial & Error**.
- Agent ↔ Environment: state → action → reward → next state. Ziel: kumulativen Reward maximieren.
- **Alltagsanalogien:** Hund mit Leckerli, Fahrradfahren, Videospiel.
- **Kernkomponenten:** State (s), Action (a), Reward (r), Policy π (s → a), Episode.
- **Markov Decision Process (MDP):** (S, A, P, R, γ) mit Transition P(s'|s,a), Reward R(s,a), Discount γ.
- **Markov-Eigenschaft:** "Die Zukunft hängt nur vom aktuellen Zustand ab" — Geschichte irrelevant.
- **Discount Factor γ:** γ=0 nur sofortige Rewards (kurzsichtig), γ=1 alle gleich, γ≈0.9 typisch (€100 heute > €100 nächstes Jahr).
- **Exploration vs. Exploitation:** Lieblingsrestaurant (sicher) vs. neues Restaurant (riskant, evtl. besser).
- **ε-Greedy:** mit Wkt. ε zufällig (explore), sonst beste bekannte Aktion (exploit).
- **Epsilon Decay:** Start hoch (~0.9), Decay-Rate 0.99–0.999, Min ~0.01. Phase früh = explore, spät = exploit.
- **Q(s, a):** erwarteter Gesamt-Reward nach Aktion a in Zustand s. Optimale Policy: `a* = argmax_a Q(s,a)`.
- **Deterministisch vs. stochastisch:** Grid World deterministisch; FrozenLake (Ch11) stochastisch (Agent rutscht).
- **Meilensteine:** DeepMind Atari (2013), AlphaGo (2016), AlphaStar (2019), AlphaCode (2022).

**Take-away:** RL = Agent lernt durch Belohnungen; ε-Greedy balanciert das Explore/Exploit-Dilemma.

---

### Kapitel 11 — Basic RL Algorithms

**Ziel:** Q-Learning verstehen und auf FrozenLake anwenden.

**Kernfakten:**
- **Model-Based vs. Model-Free:** Model-free (kein Umweltmodell) ist praktisch häufiger — heute Fokus.
- **Value-based** (Q lernen, z.B. Q-Learning) vs. **Policy-based** (π direkt lernen, z.B. Policy Gradient).
- **Q-Learning:** Tabelle Q(s,a). Optimale Policy = `argmax_a Q(s,a)`.
- **Bellman-Update (Herzstück):**
  ```
  Q(s,a) ← Q(s,a) + α · [ r + γ · max_a' Q(s',a') − Q(s,a) ]
                          └────────── TD-Error ──────────┘
  ```
  - α = Lernrate; r = Reward; γ·max Q(s',·) = diskontierter Future-Value; TD-Error = "wie falsch lag die Schätzung?".
- **GPS-Analogie für TD-Error:** Vorhersage 30 min, Stau +10 → Update auf 40 min.
- **Algorithmus:** Q init = 0; pro Episode: ε-greedy Aktion → Schritt → Bellman-Update → ε decay.
- **Hyperparameter (typisch):** α=0.1–0.5, γ=0.95–0.99, ε start 0.9–1.0 mit Decay 0.99–0.999, min 0.01, 1.000–10.000 Episoden.
- **SARSA (On-Policy-Alternative):** Update mit der *tatsächlich* genommenen Folge-Aktion statt mit `max`. Sicherer in riskanten Umgebungen; Q-Learning ist off-policy.
- **Policy Gradient (REINFORCE):** Trajektorien sammeln, Wahrscheinlichkeiten gut gelaufener Aktionen erhöhen. Moderne Erweiterungen: PPO, A3C, SAC.
- **FrozenLake-Umgebung:** 4×4-Grid (S Start, F Frozen, H Hole, G Goal), 16 States, 4 Aktionen (L=0, D=1, R=2, U=3); Reward +1 nur am Ziel; Eis ist rutschig.
- **Warum FrozenLake schwierig ist:** Sparse Rewards (nur 1 am Ziel), Stochastizität (Aktion kann fehlschlagen), Credit Assignment (welche frühere Aktion war nötig?).

**Take-away:** Q-Learning lernt Q(s,a) per Bellman-TD-Update; ε-greedy + Decay sorgt für sauberen Übergang von Exploration zu Exploitation.

---

### Kapitel 12 — Capstone: End-to-End ML Workflow (Titanic)

**Ziel:** Den kompletten ML-Workflow an einem realen Datensatz durchspielen.

**Kernfakten:**
- **Setting:** RMS Titanic (April 1912) — 1.502 von 2.224 Passagieren starben. Vorhersage: wer überlebt?
- **Workflow** (wie in Ch01): ① Define ② Explore ③ Clean ④ Engineer ⑤ Train ⑥ Evaluate ⑦ Interpret → Loop.
- **Dataset (seaborn `titanic`):** Pclass (ordinal), Sex, Age (viele NaN), SibSp, Parch, Fare, Embarked, Cabin (meist fehlend); Target = `Survived`.
- **Erfolgsmetrik:** F1-Score balanciert Precision & Recall. FN = "tot vorhergesagt, aber überlebt".
- **Cleaning:**
  - Age **per Klasse** mit Median imputieren (nicht globaler Median!).
  - Embarked mit Mode (nur 2 fehlend).
  - Drop: Cabin (zu sparse), Name, Ticket, PassengerId.
  - Encoden: Sex → 0/1, Embarked → One-Hot.
- **Feature Engineering:**
  - `FamilySize = SibSp + Parch + 1`.
  - `IsAlone = (FamilySize == 1)`.
  - Hintergrund: Solo-Reisende hatten andere Überlebenschancen → Domain Knowledge + Kreativität.
- **Modelle:** Logistic Regression, Random Forest, Gradient Boosting → alle cross-validieren, F1 + Accuracy vergleichen.
- **Typische Resultate:** Logistic ~80%/F1 0.75 · Random Forest ~82%/0.78 · Gradient Boosting ~83%/0.79. Baseline (Mehrheitsklasse) ~62%.
- **Wichtigste Erkenntnis:** Modelle liegen nah beieinander → **Feature Engineering schlägt Algorithmen-Wahl**. Sex und Pclass sind durchgehend Top-Features ("women and children first" steckt in den Daten).
- **Course Recap:** Ch01 Workflow · Ch02 Cleaning · Ch03–06 Supervised · Ch07–09 Unsupervised · Ch10–11 RL · Ch12 alles zusammen.
- **Was kommt danach:** Deep Learning, Hyperparameter-Tuning (GridSearchCV/Optuna), Deployment, MLOps. Ressourcen: Kaggle, Fast.ai, Géron "Hands-On ML".

**Take-away:** Der gesamte Kurs in einem Projekt — saubere Daten + clevere Features schlagen jeden Fancy-Algorithmus.

---

## Querschnitts-Merksätze

- **Cycle, not pipeline** — ML-Projekte sind iterativ.
- **Split first, then preprocess** — sonst Data Leakage.
- **Fit on train, transform on both** — Pipelines erzwingen das automatisch.
- **Tree-Modelle brauchen kein Scaling** — distanz- und gradientenbasierte Modelle schon.
- **Accuracy lügt bei Imbalance** — Precision/Recall/F1/AUC nutzen.
- **Cross-Validation immer** — Mean **und** Std vergleichen.
- **t-SNE-Plots niemals quantitativ interpretieren** — nur lokale Nachbarschaft ist verlässlich.
- **Feature Engineering > Algorithmus-Wahl** — meist gilt: bessere Features schlagen besseres Modell.
