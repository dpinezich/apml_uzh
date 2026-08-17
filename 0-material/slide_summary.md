# Applied Machine Learning — Zusammenfassung aller Slidesets

Diese Datei fasst die 12 Kapitel des APML-Kurses zusammen und erklärt die wichtigsten Fakten,
Konzepte und Begriffe pro Slideset. Reihenfolge entspricht der Vorlesung
(4 Sessions × 3 Kapitel). Pro Kapitel: Ziel · Kernfakten · Quiz · Bonus/Appendix · Take-away.

**Roter Faden durch alle Decks:** Split first → Baseline (`DummyClassifier`/`DummyRegressor`) → Modell als `Pipeline` →
Cross-Validation → einmal auf dem Test-Set evaluieren. Jedes Kapitel hat einen Quiz-Slide und endet mit Key Takeaways.

---

## Session 1 — Grundlagen, Datenaufbereitung & erstes Modell (Ch01–03)

### Kapitel 1 — Introduction to Machine Learning & Data Science Workflows

**Ziel:** Was ist ML, welche drei Paradigmen gibt es, wie sieht der Data-Science-Workflow aus (~50 min, kein Pflicht-Exercise).

**Kernfakten:**
- **Klassische Programmierung:** Regeln + Daten → Output. **ML:** Daten + Output → Regeln ("let data write the rules").
- Mitchell (1997): Lernen aus Erfahrung E bzgl. Aufgabe T mit Performance-Maß P (E = Daten → Ch02, T = Tasks → Ch03, P = Metriken → Ch06).
- **Drei Paradigmen** (= Kurskarte): Supervised (Sessions 1–2), Unsupervised (Session 3), Reinforcement (Session 4).
  - *Supervised* — gelabelte Daten, X → y; **Regression** (kontinuierlich) oder **Classification** (diskret).
  - *Unsupervised* — keine Labels; Segmentierung, Anomalien, Dimensionsreduktion, Topic Modeling.
  - *Reinforcement* — Agent ↔ Umwelt: Action → Reward; Ziel: kumulativen Reward maximieren.
- **Workflow ist ein Zyklus, keine Pipeline** (GIF): ① Problem definieren ② Daten ③ EDA ④ Preprocessing ⑤ Training ⑥ Evaluation ⑦ Deploy/Monitor → Iterate ("mind. 3 volle Loops").
- **~80% der Projektzeit** in Schritten 2–4. Die meisten Fehlschläge passieren bei der Problemdefinition, nicht beim Modell.
- Schritt ⑥ führt bereits zwei Schlüsselwörter ein: **held-out data** und **Baseline**.
- **Python-Ökosystem:** numpy, pandas, matplotlib/seaborn, scikit-learn, gymnasium (RL).
- **Live-Demo (~12 min):** Iris laden, Pairplot ("welche 2 Features trennen die Arten?" → petal length/width), Baseline + erstes Modell; GIF `ml_learns_boundary`. Nur 30 Test-Blumen → 1 Blume = 3.3% Accuracy → 100% nicht überinterpretieren.

**Quiz:** Strom-Nachfrage aus 5 Jahren → Supervised (Regression) · News nach Themen ohne Vorgabe → Unsupervised · lernender Thermostat → RL · Kreditkartenbetrug → Supervised (Classification).

**Bonus:** Penguins-EDA-Notebook (`03-exercises/ch01_introduction_exercises.ipynb`, Offline-CSV) für schnelle Studierende / Hausaufgabe.

**Take-away:** ML lernt Muster aus Daten; Workflow ist iterativ; Modelle auf **ungesehenen Daten gegen eine Baseline** beurteilen.

---

### Kapitel 2 — Data Selection, Cleaning & Preparing

**Ziel:** Reale Daten aufräumen, ohne Data Leakage — "Split first. Fit on train. Transform both." (mind. 3× sagen).

**Kernfakten:**
- **Garbage in → Garbage out.** Typische Probleme: NaN/leere Strings/`-999`, falsche Typen, inkonsistente Kategorien (`"Male"/"male"/"M"`), unmögliche Werte (age 999), Duplikate, gemischte Skalen, **Target Leakage** (Feature, das heimlich die Antwort ist — z.B. Entlassungsbrief erwähnt Wiederaufnahme).
- **Missing Values:** `df.isnull().sum()/.mean()`, `df.duplicated().sum()`, Heatmap (zufällig oder blockweise fehlend? *Warum* fehlt es?).
  - Strategien: <5% zufällig fehlend → Zeilen droppen; >50% → Spalte droppen; Mean (numerisch, ohne Outlier), Median (mit Outlier), Mode (kategorial), Konstante (`"Unknown"`).
  - ⚠️ Der Füllwert ist eine **Statistik** → nur auf Train berechnen (`SimpleImputer(strategy='median')`).
- **Outlier:** Boxplot, IQR-Regel (`Q1−1.5·IQR` / `Q3+1.5·IQR`, Grenzen aus Train). Behandlung: entfernen (klarer Fehler), kappen, log-transformieren, **behalten** (echtes Signal). Unmöglicher Wert (age 999) → NaN + imputieren; großer Wert (200 m²) evtl. real.
- **Feature-Typen & Encoding:** nominal → **One-Hot**; ordinal → geordnete Integers; binär → 0/1. Zip code ist nominal, obwohl numerisch. `pd.get_dummies` (quick & dirty) vs. `OneHotEncoder(handle_unknown='ignore')` (lernt Kategorien auf Train, unbekannte Test-Kategorie → alle 0). 100 Städte → 100 Spalten; High Cardinality → Target Encoding/Embeddings (nicht im Kurs).
- **Skalierung:** `StandardScaler` (Mean 0, Std 1), `MinMaxScaler` ([0,1]). Nötig für distanz-/gradientenbasierte Modelle (KNN, lineare/logistische Regression, SVM, NN); **Tree-Modelle nicht.**
- **Train/Test-Split:** `train_test_split(test_size=0.2, random_state=42, stratify=y)` — `stratify=y` **nur bei Klassifikation** (bei kontinuierlichem Target → ValueError). Test-Set bleibt bis ganz zum Schluss weggeschlossen.
- **Data Leakage (#1 Anfängerfehler, GIF `leakage_impute`):** Mean über alle Zeilen trägt Test-Info ins Training → zu optimistischer Score.
- **Richtige Reihenfolge:** deterministische Fixes (Text, Duplikate, unmögliche Werte) *vor* dem Split; alles, was eine Statistik berechnet (Mean, Median, Min/Max, Std, IQR, Kategorienliste) → `fit` auf Train, `transform` auf beide. `Pipeline([SimpleImputer, StandardScaler])` + `ColumnTransformer` (numerisch vs. kategorial) machen das automatisch.
- **Demo (~8 min):** messy Studierenden-Umfrage → Duplikate, Text-Fixes, Target Leakage (`grade`), Split, SimpleImputer, IQR, Encoding, Scaling → ColumnTransformer + Pipeline in 10 Zeilen. **Exercises (~10 min):** messy Housing-Daten (Fixes → Split → Impute → ColumnTransformer).

**Quiz:** Welche Schritte leaken? `str.lower()` safe · `fillna(median)` vor Split **leakt** · Split safe · scaler.fit(train)/transform(test) safe · drop_duplicates vor Split safe · Spalte `final_invoice_amount` droppen = *entfernt* Target Leakage.

**Bonus:** Z-Score-Regel (`|z|>3`, ≈0.3% einer Normalverteilung; empfindlich gegen die Outlier, die sie sucht → IQR meist robuster; Exercise Bonus C).

**Take-away:** Zuerst splitten, dann imputieren/kappen/encoden/skalieren — fit on train, transform both; Pipeline + ColumnTransformer erzwingen das.

---

### Kapitel 3 — Introduction to Supervised Learning

**Ziel:** Protokoll (Split → Baseline → Train → Test), Generalisierung, Over-/Underfitting, **ein** Algorithmus durchgehend (KNN), sklearn-API.

**Kernfakten:**
- X = Features, y = Label/Target, ŷ = Vorhersage. Regression (Hauspreis €285 000) vs. Classification (Spam, Iris-Art).
- **Protokoll:** ① Split (Ch02) ② **Baseline** "immer Mehrheitsklasse" — die Latte ③ `fit(X_train, y_train)` ④ `score(X_test, y_test)` ← die Zahl, die zählt. Trainings-Accuracy = Wiederholen des Gesehenen; Test-Accuracy = Generalisierung. Ab jetzt in jedem Kapitel eine `DummyClassifier`-Baseline (90/10-Daten → Dummy schon 90%).
- **Underfitting:** zu einfach → Train- **und** Test-Fehler hoch (High Bias). **Overfitting:** memoriert Rauschen → Train niedrig, Test hoch (High Variance). **Bias-Variance-Tradeoff:** Sweet Spot mit minimalem Test-Fehler (Formel nicht nötig).
- **KNN:** kein Training, keine Formel — Distanz zu allen Trainingspunkten, **k nächste**, Mehrheitsvotum. Gespeichert wird das ganze Trainingsset (→ langsam bei großen Daten).
- **Effekt von k** (GIF `knn_boundary_k`, make_moons, k = 1 → 100): k=1 → jede Insel für sich, Train 100%, Test tiefer = Overfitting; k≈5–30 glatt; k=100 fast Gerade = Underfitting. Bias-Variance-Kurve live.
- **KNN braucht Skalierung:** Feature mit größtem Range dominiert die Distanz → `StandardScaler` **in der Pipeline**. Demo Penguins: 0.77 vs. 0.99 Test-Accuracy.
- **Hyperparameter vs. Parameter:** Hyperparameter = Knopf, den *du* vor dem Training setzt (k, degree, depth); Parameter = was der Algorithmus lernt (Slope, Weights). KNN hat keine Parameter.
- **sklearn-API:** `Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=5))])` → `.fit / .predict / .score` — die Pipeline **ist** das Modell, Leakage per Konstruktion unmöglich; jedes sklearn-Modell funktioniert so.
- **Cross-Validation nur als Preview** (k-fold-Bild): ein Split = eine verrauschte Zahl; CV mittelt und dient der Hyperparameterwahl → **implementiert in Ch04**.
- **Demo (~10 min):** two moons → Baseline → Pipeline(scaler, KNN) → Boundaries k = 1/7/51 → k-Sweep (Train vs. Test) → Penguins mit/ohne Scaler → CV-Kostprobe. **Exercises (~10 min):** Split → Baseline + KNN → k-Sweep → Plot; Bonus: Penguins-Pipeline mit `SimpleImputer`, Scaler an/aus, CV.

**Quiz:** KNN Train 100% / Test 72% / Baseline 65% → Overfitting; k **erhöhen**; die 100% nicht berichten — Test-Accuracy nennen und dass sie nur 7 Punkte über der Baseline liegt.

**Bonus (Further Reading):** Loss Functions (MSE/Cross-Entropy → Ch04/05), Distanzmetriken (`metric=`), `weights='distance'`, `KNeighborsRegressor`.

**Take-away:** Split → Baseline → fit on train → score on test; Train/Test-Gap verrät Over-/Underfitting; KNN = Distanz + Vote, k = Hyperparameter, scale first.

---

## Session 2 — Supervised Learning: Regression, Klassifikation, Metriken (Ch04–06)

### Kapitel 4 — Regression Models

**Ziel:** Baseline, lineare Regression, Loss/Gradient Descent, Overfitting quantifiziert mit CV, Regularisierung, Trees/Forests.

**Kernfakten:**
- **Metriken (Details Ch06):** MAE (gleiche Einheit wie y), RMSE (große Fehler zählen mehr), R² (erklärte Varianz; 1 perfekt, 0 = "Mittelwert vorhersagen").
- **Baseline first:** `DummyRegressor(strategy='mean')` → R² = 0 per Definition; auf California Housing ~$90k daneben — die Zahl, die zu schlagen ist. Jede Ergebnistabelle beginnt mit einer Dummy-Zeile.
- **Linear Regression:** ŷ = β₀ + β₁x₁ + … + βₙxₙ; OLS minimiert die quadrierten Residuen; `coef_`, `intercept_`. Warum quadriert? glatt, eindeutig, geschlossene Lösung.
- **β lesen:** Target in k€, `area_sqm` β = 3.5 → +3.5 k€ = +3 500 € pro m², ceteris paribus. Rohe Koeffizienten hängen von Einheiten ab → für "welches Feature zählt am meisten?" erst standardisieren (β = Änderung pro Standardabweichung). Skalierung ändert OLS-Vorhersagen nicht — Ridge/Lasso brauchen sie.
- **Loss Function:** MSE(β) = (1/n)Σ(yᵢ−ŷᵢ)²; Training = β finden, das den Loss minimiert. Linear Regression: geschlossene Formel; fast alles andere (LogReg, NN, Boosting): iterativ → **Gradient Descent** `w ← w − η·∇L(w)` (GIF; η zu klein = langsam, zu groß = Overshoot; Skalierung hilft, weil die Schüssel sonst gestreckt ist).
- **Polynomial Regression:** `make_pipeline(StandardScaler(), PolynomialFeatures(degree=3), LinearRegression())` — x **vor** dem Potenzieren skalieren. GIF `poly_degree_sweep`: Train-MSE sinkt immer (→ würde Grad 15 wählen), **CV-MSE hat ein Minimum** (~Grad 3–4).
- **CV in der Praxis:** `KFold(n_splits=5, shuffle=True, random_state=42)` + `cross_val_score(pipeline, X, y, cv=cv, scoring='r2')` → **mean ± std** berichten; jede Probe genau einmal Test; die ganze Pipeline geht rein (Scaler pro Fold neu gefittet). `shuffle=True` außer bei Zeitreihen. 0.48±0.08 vs. 0.47±0.08 → nicht unterscheidbar.
- **Regularisierung:** viele korrelierte Features + wenig Daten → riesige, sich aufhebende Koeffizienten. **Ridge (L2)** MSE + α·Σβᵢ² schrumpft alle β glatt; **Lasso (L1)** MSE + α·Σ|βᵢ| setzt manche β **exakt 0** (automatische Feature-Selektion). α→0 = OLS, α→∞ = alle β = 0 (= Baseline). Braucht skalierte Features. Diabetes-Daten: um α≈1 fallen bereits Features weg; auf 20k-Zeilen California ist α=1 unsichtbar — Regularisierung zählt, wenn Daten knapp sind.
- **α wählen:** `RidgeCV(alphas=np.logspace(-3,3,30), cv=5)` → `model[-1].alpha_`; allgemein `GridSearchCV(..., cv=5)`. **Nie Hyperparameter auf dem Test-Set wählen.**
- **Decision Tree Regressor:** Feature-Raum in Boxen, Vorhersage = Mittelwert der Box; `max_depth` ist sein Regularisierungsknopf; kein Scaling; nichtlinear. Depth 1 = Stufenfunktion, Depth 20 = ein Blatt pro Haus.
- **Random Forest:** viele Bäume auf zufälligen Zeilen **und** Features → Mittelwert; `feature_importances_`; fast immer besser als ein Baum, weniger interpretierbar. Demo: RF schlägt Linear auf California (R² 0.78 vs. 0.60); auf Diabetes (Exercise) **nicht**.
- **Modellwahl-Tabelle:** Linear (interpretierbar, linear, Scaling nur zum β-Vergleich) · Ridge/Lasso (Scaling nötig) · Tree/Forest (nichtlinear, kein Scaling, Outlier in X egal, in y nicht). Faustregel: Baseline → Linear/Ridge → Forest bei Nichtlinearität → einfachstes "gut genug"-Modell.
- **Exercises (~10 min):** Diabetes (442 Patienten, 10 Features) — Baseline + Linear · Ridge vs. Lasso (welche Features fliegen raus?) · RF + Importances · Bonus CV/RidgeCV/predicted-vs-actual. Debrief: RF gewinnt hier nicht (kleines n, fast lineares Target).

**Quiz:** "Grad-12-Polynom, Train R² 0.99, ship it" → nach **CV/Test-R²** fragen · RidgeCV wählt α=0.001 aus [0.001…1000] → Regularisierung hilft nicht (oder Grid tiefer) · 0.481±0.085 vs. 0.478±0.083 → **nicht unterscheidbar**.

**Take-away:** Baseline first (R² = 0); Modellkomplexität (degree, 1/α, depth) ist ein Regler — Train-Fehler sinkt immer, **CV-Fehler sagt, wo Schluss ist**; Ridge schrumpft, Lasso selektiert; α per RidgeCV/GridSearchCV.

---

### Kapitel 5 — Classification Models

**Ziel:** Diskrete Vorhersagen — Baseline, Logistic Regression, KNN-Recap, Trees, Forest, Decision Boundaries, Confusion Matrix lesen.

**Kernfakten:**
- Binär (Spam, malignant/benign, Churn) vs. Multi-Class (Iris, Digits, 3 Rebsorten). `predict` = hartes Label, `predict_proba` = Wahrscheinlichkeiten (Zeile summiert zu 1) — oft nützlicher (Ch06: Schwelle verschieben).
- **Baseline:** `DummyClassifier(strategy='most_frequent')`. Breast Cancer: 63% benign → Dummy 63% Accuracy. 90% Accuracy auf 95/5-Daten = schlechter als Dummy.
- **Logistic Regression — Klassifikator trotz Name:** P(y=1|x) = σ(β₀+β₁x₁+…), σ(z) = 1/(1+e⁻ᶻ); `make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))`; **lineare Decision Boundary** (wo Score = 0, σ(0)=0.5); β = Effekt auf Log-Odds; trainiert per Gradient Descent auf **Cross-Entropy** (nur Erwähnung); Multi-Class via Softmax automatisch. `predict()` = "P > 0.5" — die 0.5 ist eine *Wahl*.
- **KNN — 2-Minuten-Recap (Ch03):** braucht Scaling, kein Training, langsam bei Big Data; k = Komplexitätsregler; k per CV wählen wie degree/α in Ch04. (Wine: proline ~1000 vs. hue ~1.)
- **Decision Tree Classifier:** Ja/Nein-Fragen, gewählt für möglichst reine Gruppen (**Gini**); `DecisionTreeClassifier(max_depth=3, random_state=42)`, `plot_tree`; vollständig interpretierbar; kein Scaling; `random_state` zählt auch bei einem Baum (Tie-Breaks).
- **Komplexitätsregler zum dritten Mal — Tree Depth** (GIF `tree_depth_sweep`): Train → 1.0, Test peakt früh (~5–6) und fällt, Boundary wird Konfetti → Grund für Forests.
- **Random Forest Classifier:** viele tiefe Bäume auf zufälligen Zeilen + Feature-Subsets → Mehrheitsvotum; mittelt das Rauschen weg; kein Scaling; `feature_importances_` (bevorzugen Features mit vielen Werten — "was dieses Modell nutzte", nicht Kausalität).
- **Decision Boundaries** (`DecisionBoundaryDisplay`): linear (LogReg) · lokal/wackelig (KNN) · achsenparallele Boxen (Tree) · geglättete Boxen (Forest). Komplexere Boundary ≠ besser; einfache Modelle extrapolieren vernünftiger.
- **Confusion Matrix lesen (Vorbereitung Ch06):** Zeilen = Wahrheit, Spalten = Vorhersage, Diagonale = richtig; **positive Klasse = 1 = das Gesuchte — prüfen!** sklearn Breast Cancer hat 1 = benign → **im Notebook geflippt (`y = 1 − target`, malignant = 1)**. `ConfusionMatrixDisplay.from_predictions`, `classification_report` (Precision/Recall/F1 pro Klasse).
- **Exercises (~12 min):** `load_wine` (178 Weine, 13 Messwerte, **3 Rebsorten**) — Baseline + LogReg-Pipeline · KNN-Pipeline · RF + classification_report · Confusion Matrix eines *schwachen* Baums (depth 2, welche Klassen werden verwechselt? class_1 → class_0) · Bonus 5-fold CV, SVM. CV-Std groß, weil 178 Samples → ein Wein = 2.8%.

**Quiz:** Malignant, Modell sagt benign → **False Negative**, der teure Fehler · KNN 0.72 → 0.95 nach StandardScaler → ein Feature dominierte die Distanz · Regulator muss jede Entscheidung verstehen → **Decision Tree** (oder LogReg).

**Appendix/Bonus:** **SVM** — breitester Margin zu den Support Vectors; `make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0))`; kleines C = breiter Margin (mehr Regularisierung); rbf = nichtlinear; braucht Scaling; langsam ab ~50k Zeilen. Nicht Kern (Exercise Bonus B).

**Take-away:** Baseline first; LogReg = lineare Boundary + Wahrscheinlichkeiten; KNN lokal (skalieren); Tree = lesbare Regeln, Forest = robust; gleiche Daten → sehr verschiedene Boundaries; Confusion Matrix: Zeilen = Wahrheit, positive Klasse = das Gesuchte.

---

### Kapitel 6 — Metrics & Evaluation

**Ziel:** Die Metrik definiert den Erfolg — Accuracy ist oft das falsche Lineal. Positive Klasse überall = 1 = **malignant** (geflippt).

**Kernfakten:**
- **Accuracy-Paradoxon:** 99% gesund → "immer gesund" = 99% Accuracy, 0% Recall. Live (Demo Teil B, 98/2 synthetisch, `make_classification`): `DummyClassifier` 98% Accuracy / 0% Recall vs. `LogisticRegression(class_weight='balanced')` 90% Accuracy / 94% Recall. Die Metrik *ist* die Entscheidung.
- **Regressionsmetriken:** MAE (Einheit y), MSE (y²), RMSE = √MSE (Einheit y), R² = 1 − SS_res/SS_tot. `root_mean_squared_error` ersetzt `squared=False` (sklearn ≥ 1.4). **RMSE ≥ MAE immer**; große Lücke = wenige große Fehler. Fehler 1,1,1,1 vs. 0,0,0,4: gleiche MAE, RMSE 1 vs. 2.
- **R²:** 1 perfekt, 0 = Mittelwert (= DummyRegressor), <0 = kaputt. Domänenabhängig: Finance 0.1 beeindruckend, Physik 0.99 erwartet, Hauspreise 0.6–0.8 typisch. **Residual-Plot**: Fehler sollen zufälliges Rauschen um 0 sein (California: $500k-Cap als diagonaler Streifen = Datenproblem).
- **Confusion Matrix:** `tn, fp, fn, tp = confusion_matrix(...).ravel()` (nur binär, diese Reihenfolge!).
- **Spam-Beispiel:** TN 90, FP 5, FN 3, TP 2 → Accuracy 0.92 (klingt toll), Precision 2/7 = 0.29, Recall 2/5 = 0.40. Verlorene echte Mail (FP) schlimmer als ein Spam im Posteingang → Spam-Filter optimiert **Precision**; Krebs-Screening → FN schlimmer → **Recall**.
- **Wann was:** Precision = TP/(TP+FP) "wenn ich positiv sage, stimmt's?" (FP teuer, Spam) · Recall = TP/(TP+FN) "alle Positiven gefunden?" (FN teuer, Screening) · F1 = 2PR/(P+R) harmonisch (beides, Imbalance: Fraud, Defekte; P=1.0/R=0.1 → F1 0.18) · Accuracy nur bei balancierten Klassen und gleich teuren Fehlern.
- **Threshold ist deine Wahl** (GIF `threshold_sweep`): `(predict_proba[:,1] >= 0.3).astype(int)`; tiefe Schwelle → Recall ↑, Precision ↓. Jeder Punkt = eine Schwelle → das *sind* ROC/PR-Kurven.
- **ROC & AUC:** TPR (Recall) vs. FPR über alle Schwellen; AUC 1.0 perfekt, 0.5 zufällig = "P(zufälliger Positiver scort höher als zufälliger Negativer)"; threshold-unabhängig → vergleicht Ranking-Qualität. AUC 0.99 mit 3 verpassten Krebsen sind trotzdem 3 verpasste Krebse.
- **ROC vs. PR bei Imbalance:** ROC teilt durch die große Klasse → sieht gut aus, auch wenn Positive selten sind → dann **PR-Kurve**. Toolbox: `class_weight='balanced'`, Schwelle senken, `StratifiedKFold`, Recall/F1/PR-AUC statt Accuracy.
- **Multi-Class `classification_report`:** macro avg = Mittel über Klassen (gleiches Gewicht) → bei Imbalance **macro F1**, `scoring='f1_macro'`; weighted avg = nach Support.
- **CV mit der richtigen Metrik:** Pipelines (Scaler drin) + `StratifiedKFold(5, shuffle=True, random_state=42)` + `cross_val_score(..., scoring='f1')` (oder recall/roc_auc/f1_macro) → mean ± std. 0.963±0.024 vs. 0.949±0.026 → nicht überzeugend verschieden.
- **Exercises (~10 min, Modelle vortrainiert — evaluieren & interpretieren):** A1 MAE/RMSE/R² zweier Diabetes-Modelle · B1 Accuracy/Precision/Recall/F1 Krebs-Modell · B2 Confusion Matrix (wie viele Krebse verpasst?) · B3 ROC + AUC · Bonus Threshold-Tuning (Schwelle 0.5 → 0.1: verpasste Krebse 3 → 1, Fehlalarme 1 → 5), Residual-Plot, Dummy.

**Quiz:** Screening: Modell A (Recall 0.98/Precision 0.20) für Erst-Screening, B (0.80/0.90) wenn die Aktion selbst teuer ist · AUC 0.99 + 3 von 42 verpasst bei 0.5 → kein Widerspruch, Schwelle senken · "Accuracy 0.97" bei 1% Fraud → Recall/F1? Dummy? (0.99 → 0.97 ist *schlechter* als nichts).

**Take-away:** Accuracy lügt bei Imbalance — immer mit Dummy vergleichen; Metrik nach Fehlerkosten wählen; 0.5 ist eine Entscheidung, kein Gesetz; seltene Positive → PR-Kurve/class_weight/stratified/F1; CV mit Pipeline + richtigem `scoring`, mean ± std.

---

## Session 3 — Unsupervised Learning (Ch07–09)

### Kapitel 7 — Introduction to Unsupervised Learning

**Ziel:** Lernen ohne `y`-Spalte — warum, drei Fragen, Evaluations-Herausforderung (~41 min Inhalt; Exercise optional/Hausaufgabe).

**Kernfakten:**
- Kein Label, keine "richtige Antwort". Frage wechselt von "Was ist das?" zu "Was ist *darin*?".
- **Warum ungelabelte Daten:** Labeln teuer, für Zukunftsdaten unmöglich, Labels existieren noch nicht (Discovery), man weiß nicht, wonach man sucht. **Die meisten Daten der Welt sind ungelabelt — Supervised ist die Ausnahme.** Post-Sortier-Analogie: verschiedene Sortierungen können beide valide sein.
- **GIF `find_groups`:** Blobs / Moons / Circles — die Augen machen Unsupervised Learning; Teaser: K-Means schafft Blobs, scheitert an Moons/Circles.
- **Drei Fragen:** *Clustering* (welche Samples gehören zusammen? → Gruppen-ID, Ch08) · *Dimensionality Reduction* (Info in weniger Features? → neue Koordinaten, Ch09) · *Anomaly Detection* (welche Samples sind ungewöhnlich? → Outlier-Flag, heute 1 Zeile). Density Estimation = Familie dahinter; GMM = "soft K-Means" (nur Erwähnung); generative Modelle out of scope.
- **Ein-Zeiler:** `KMeans(n_clusters=3, random_state=42).fit_predict(X)` · `PCA(n_components=2).fit_transform(X)` · `IsolationForest(random_state=42).fit_predict(X) == -1` — gleiche sklearn-Grammatik, **kein y**. Output = eine Spalte, die wir erfinden.
- **Evaluations-Herausforderung** (Bild `two_valid_groupings`): interne Metriken (Silhouette, Inertia — kompakt & getrennt?) · externe Checks (ARI, falls Labels — Sanity-Check, nicht Ziel) · Domain Sense (kann jemand jede Gruppe erklären? Downstream-Task?). Silhouette/Inertia ersetzen die Accuracy, sagen aber nie "korrekt". **Clusters ≠ Classes.**
- **Anwendungen:** Marketing-Segmente, Krankheits-Subtypen (Genexpression), Topics, Fraud/Intrusion/Sensorfehler (Anomalie ≠ Fraud — Kandidaten für Menschen), Bildkompression/Eigenfaces (PCA), 100-D → 2-D (PCA/t-SNE/UMAP), Recommender (latente Faktoren).
- **Demo (~12 min):** Iris ohne Labels · drei Strukturarten · IsolationForest auf Kundendaten ("automatisch blocken?" → nein, untersuchen) · 30-s-Teaser Digits (64 Features, Ch09).

**Quiz:** "K-Means mit k=3 → wir haben 3 Kundentypen" → K-Means liefert k Gruppen, **egal wie die Daten aussehen**; Kompaktheit/Trennung prüfen und Business-Sinn · 10 000 ungelabelte + 200 gelabelte Röntgenbilder → Unsupervised nutzt alle 10 200, Supervised nur 200; Kombination (PCA/Cluster als Features, semi-supervised) üblich.

**Optional/Hausaufgabe:** `03-exercises/ch07_unsupervised_intro_exercises.ipynb` (PCA + K-Means auf Digits, ~15 min).

**Take-away:** Unsupervised = Normalzustand realer Daten; drei Fragen (Clustering, Dim.-Reduktion, Anomalien); Evaluation = interne Metriken + Domänenurteil; Clusters ≠ Classes.

---

### Kapitel 8 — Clustering Techniques

**Ziel:** K-Means richtig (Init, Scaling, k wählen, "Clustering funktioniert immer") und DBSCAN; Hierarchical = Bonus.

**Kernfakten:**
- Ziel: ähnlich innerhalb, unähnlich zwischen Clustern; Output = neue Spalte `labels` (0/1/2 = willkürliche IDs, keine Klassen). "Ähnlich wonach?" → Distanz → Scaling zählt.
- **K-Means-Algorithmus** (GIF `kmeans_iterations`): ① k zufällige Centroids ② ASSIGN zum nächsten Centroid ③ UPDATE Centroid = Mittelwert ④ wiederholen bis nichts sich bewegt. Minimiert **Inertia** (Summe quadrierter Distanzen zum eigenen Centroid) — kann nur sinken → konvergiert immer.
- **Parameter:** `KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)`; `labels_`, `cluster_centers_`, `inertia_`. sklearn ≥ 1.4: `n_init='auto'` (=1) → explizit 10 setzen.
- **Warum init/n_init:** K-Means findet nur ein **lokales** Optimum; schlechter Start → 4× schlechtere Inertia. `k-means++` streut, `n_init=10` behält den besten Lauf. Misconception: "K-Means findet DIE Cluster".
- **Scale first:** `make_pipeline(StandardScaler(), KMeans(...)).fit_predict(X)` — gilt für K-Means, DBSCAN, Hierarchical, KNN. Caveat: auf Digits-Pixeln verstärkt Scaling fast konstante Randpixel → K-Means wird schlechter → "scale by default, but check".
- **k wählen:** *Elbow* (Inertia vs. k — sinkt immer, k=n → 0; Knick oft mehrdeutig) · *Silhouette* = (b−a)/max(a,b), +1 getrennt, 0 Grenze, −1 falsch — Max wählen, bevorzugt wenige runde Cluster. Iris: Elbow sagt 3, Silhouette 2 — beides "richtig".
- **⚠️ Clustering "funktioniert" immer** (wichtigster Slide, Bild `kmeans_uniform`): K-Means sagt nie "keine Struktur". Uniformes Rauschen: Silhouette ≈ 0.38 vs. 0.83 bei echten Blobs. Faustregel: < 0.25 keine substanzielle Struktur, 0.5+ vernünftig. Vor dem Vertrauen: PCA-2D-Plot (Ch09), Silhouette, Sinn?
- **DBSCAN:** `DBSCAN(eps=0.25, min_samples=5).fit_predict(X_scaled)`; Core Point = ≥ min_samples Nachbarn innerhalb eps; Ketten von Core Points = Cluster; Rest = **Noise (−1)** → eingebaute Outlier-Erkennung. Kein k, beliebige Formen. Preis: eps schwer zu tunen (skalieren!), Probleme bei sehr verschiedenen Dichten (HDBSCAN/OPTICS out of scope). Faustregel: min_samples ≈ 2·n_features, eps aus k-distance-Knie.
- **Welcher Algorithmus:** K-Means (rund, ähnlich groß, k nötig, von Outliern gezogen, schnell, erster Versuch) · DBSCAN (beliebig, kein k, Outlier = −1) · Hierarchical *(Bonus)* (Dendrogramm, O(n²), kleine Daten) · GMM *(Erwähnung)* (Ellipsen, soft). Workflow: K-Means → Silhouette + Plot → DBSCAN bei seltsamen Formen/Outliern.
- **Demo (~8 min):** Elbow + Silhouette auf Blobs und **auf reinem Rauschen** · K-Means vs. DBSCAN auf Moons · Kundensegmentierung: scale → K-Means → **Profile** → Namen aus den Zahlen (nie aus der Cluster-ID). **Exercises (~10 min):** Iris ohne Labels — Elbow → Silhouette → k entscheiden (widersprechen sich!) → Pipeline → Plot → Sanity-Check gegen Species; Bonus DBSCAN, Hierarchical.

**Quiz:** Elbow k=3, Silhouette k=2 → kein Orakel: beide plotten, nach Nützlichkeit entscheiden, berichten dass 3 eine große Gruppe teilt · age (18–80) + income (20 000–200 000) unskaliert → clustert fast nur nach income (~1000× größere Differenzen) → StandardScaler-Pipeline.

**Bonus/Appendix:** **Hierarchical Clustering** — `linkage(X_scaled, method='ward')`, `dendrogram(Z)`, `fcluster(Z, t=3, criterion='maxclust')`; lange vertikale Äste = natürliche Lücken; Ward minimiert Within-Cluster-Varianz (K-Means-Cousin); O(n²) Speicher; Linkages ward/complete/average/single.

**Take-away:** K-Means = assign → update → repeat, lokales Optimum (k-means++, n_init); scale first; k = Elbow + Silhouette + **Plot** + Domänensinn; K-Means liefert immer k Cluster — Struktur prüfen; DBSCAN: Dichte, kein k, Outlier = −1; Clusters ≠ Classes.

---

### Kapitel 9 — Dimensionality Reduction

**Ziel:** Curse of Dimensionality, PCA (neue Achsen ≠ Feature-Selektion, zwei Regimes, in der Pipeline), t-SNE nur zur Visualisierung.

**Kernfakten:**
- **Curse of Dimensionality** (GIF `curse_dimensionality`): mit wachsendem d werden zufällige Punkte **gleich weit** voneinander entfernt (Histogramm der Paardistanzen wird schmaler, nearest/farthest → 1) → distanzbasierte Methoden (KNN, K-Means, DBSCAN) verlieren ihr Signal; mehr Daten nötig; Overfitting; kein Plot > 3-D. Gegenpunkt (Manifold-Hypothese): reale Hoch-D-Daten liegen nahe einer Niedrig-D-Struktur (Gesichter: Mio. Pixel, wenige Dutzend Richtungen) — das nutzt PCA.
- **PCA** (GIF `pca_rotation`): PC1 = Richtung maximaler Varianz (89% im Beispiel), PC2 = max. Restvarianz ⊥ PC1, …; Rotation verliert nichts, **Droppen** der varianzarmen Achsen ist die Kompression. PCA sieht y nie — unsupervised.
- **Code, immer mit Scaling:** `make_pipeline(StandardScaler(), PCA(n_components=2)).fit_transform(X)`; `explained_variance_ratio_` (z.B. [0.44, 0.19]), `.sum()`, `components_` (Loadings). Ohne Scaling *ist* das Feature mit den größten Zahlen PC1 (Breast Cancer: mean/worst area).
- **PCA ≠ Feature Selection** (Kernbotschaft, Bild `pca_loadings`): jede PC = gewichtete Summe **aller** Features; PCA wählt nicht "die besten 2 Spalten", sondern baut 2 neue Achsen → schwerer interpretierbar; für "welche Roh-Features zählen?" Feature-Selektion/Importances nutzen. Breast Cancer PC1 ≈ "Gesamtgröße/Irregularität", PC2 kontrastiert Größe vs. Textur/Fraktal.
- **Zwei Regimes** (Bild `pca_two_regimes`): **Visualisieren** = 2 PCs, der %-Wert ist, was er ist (22% bei Digits) — Bild beurteilen, nicht die Zahl · **Preprocessing** = genug PCs für ≈ 90–95% → `PCA(n_components=0.95)` (Float = Varianzanteil; Digits: 40 von 64). Misreading "nur 22% → PCA gescheitert" ist falsch. Cumsum-Scree-Plot. Farben im Plot = Labels *nachträglich*.
- **PCA als Preprocessing in der Pipeline:** `make_pipeline(StandardScaler(), PCA(0.95), RandomForestClassifier())` in `cross_val_score` → PCA nur auf Trainingsteil jedes Folds gefittet (kein Leakage, gleiche Regel wie Imputation/Scaling); `pipe['pca'].n_components_`. Hilft bei vielen korrelierten Features / Speed / Denoising; Tree-Modell auf 30 sauberen Features wird oft **nicht** besser (Digits ≈ 0.98 → 0.965 mit 40/64) → immer mit No-PCA-Pipeline und **Dummy-Baseline** vergleichen; "Trade-off, kein Free Lunch".
- **t-SNE:** `TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_scaled)`; erhält **Nachbarn** (Punkte nahe in 64-D bleiben nahe), nicht Distanzen zwischen Gruppen oder Gruppengrößen; Perplexity ~5–50 = "wie viele Nachbarn zählen"; Barnes-Hut O(n log n), ok bis ~10k.
- **t-SNE — erlaubt/verboten:** ✅ "diese Samples sitzen zusammen → ähnlich"; ✅ "isolierte Insel → evtl. eigene Gruppe, verifizieren" · ❌ "A ist 3× weiter von B als von C" · ❌ "5 Cluster im Plot → 5 Gruppen" (Perplexity/Seed ändern das Bild) · ❌ t-SNE-Koordinaten als Features: **kein `.transform()`** für neue Daten → passt in keine Train/Test-Pipeline; kein Silhouette auf t-SNE. **Nur Visualisierung.** UMAP: ähnlich, schneller, hat `transform` (`pip install umap-learn`).
- **Demo (~5 min):** Digits Scree-Plot (PCs für 80/90/95%) · 2-D PCA vs. t-SNE · PCA in Klassifikations-Pipeline mit Baseline. **Exercises (~10 min):** Breast Cancer (30 Features, malignant = 1 geflippt) — Scree → 2-D → PCA-Pipeline vs. No-PCA vs. Dummy; Bonus t-SNE, PC1-Loadings. Erwartet: 10 PCs für 95%, RF ≈ 0.95–0.96 mit/ohne PCA, Dummy ≈ 0.63.

**Quiz:** 2-D-PCA von 500 Features zeigt 18% Varianz, Gruppen trennen sich → fürs Bild fein (18% normal), fürs Preprocessing ≈ 90–95% behalten · PCA auf allen Daten gefittet, dann Split, 97% → Leakage (Komponenten sahen Test-Zeilen) → PCA in die Pipeline.

**Take-away:** Hoch-D → Distanzen verlieren Bedeutung; PCA = neue Achsen aus allen Features, nach Varianz sortiert, scale first, unsupervised; 2 PCs für Bilder (egal welcher %), ≈ 95% für Preprocessing — in der Pipeline; PCA ≠ Feature Selection; t-SNE nur Nachbarn, keine Distanzen, kein `.transform()` → nur Visualisierung.

---

## Session 4 — Reinforcement Learning & Capstone (Ch10–12)

Session-Timing: Ch10 ~45 min (Slides + Demo, kein Exercise) · Ch11 ~45 min (inkl. 10 min Exercise) · Ch12 65 min (5 Intro + 50 Selbstarbeit + 10 Debrief).

### Kapitel 10 — Introduction to Reinforcement Learning

**Ziel:** Das dritte Paradigma — Vokabular auf einer GridWorld, MDP, γ, Explore/Exploit, Q(s,a) (~39 min Inhalt; Exercise = Bonus).

**Kernfakten:**
- **Paradigmen-Tabelle:** Supervised = (X, y) · Unsupervised = X · **RL = gar kein Datensatz, nur Rewards für ausprobierte Aktionen** — der Agent erzeugt seine Daten selbst. Loop: Agent handelt, Umwelt antwortet mit neuem Zustand + Reward. **Ziel: Gesamt-Reward einer Episode maximieren, nicht den nächsten.**
- **Analogien:** Hund (Kommando / sit-bark-run / Leckerli), Fahrrad, Videospiel, Schach (Reward erst am **Ende**) → wer sagt dem Schach-Agenten, welcher der 40 Züge der Fehler war? Niemand → **Credit Assignment Problem** = Kernschwierigkeit.
- **Vokabular auf GridWorld (4×4, 16 Zellen, in Demo/Ch11/Exercise/Animation):** State s (Zelle 0–15) · Action a (← ↓ → ↑) · Reward r (**+1 Ziel, −1 Loch, −0.01 pro Schritt**) · Episode (Start → Loch oder Ziel) · Return (Summe der Rewards; kürzester Pfad 5 Schritte → 0.95) · Policy π (State → Action, Tabelle mit 16 Einträgen). Der −0.01 unterscheidet 6-Schritt- von 60-Schritt-Pfaden. Random Walker erreicht das Ziel in ~1–2% der Episoden.
- **MDP (S, A, P, R, γ):** P(s'|s,a) Übergangswahrscheinlichkeiten ("wo lande ich?"), R Reward-Funktion, γ Discount. **Markov-Eigenschaft:** Zukunft hängt nur vom aktuellen State ab (GPS-Analogie). Schach Markov, Poker nicht (versteckte Info).
- **Deterministisch vs. stochastisch:** GridWorld deterministisch; FrozenLake (Ch11, `is_slippery=True`): beabsichtigte Richtung mit p = **1/3**, sonst seitlich rutschen. Neben einem Loch lieber gegen die Wand laufen.
- **Discount γ:** 0 = kurzsichtig, → 1 = weitsichtig, typisch 0.9–0.99, **wir: 0.99**. Zwei Jobs: Ungeduld (€100 heute > nächstes Jahr) und Wert *fließt rückwärts* vom Ziel zu früheren States (→ löst Credit Assignment in Ch11).
- **Explore vs. Exploit** (Restaurant-Analogie): nur exploit → evtl. für immer mittelmäßige Policy; nur explore → Gelerntes nie genutzt.
- **ε-Greedy + Decay:** `if rng.random() < epsilon: random_action() else: argmax(Q[state])`; nach jeder Episode `epsilon = max(epsilon*0.999, 0.01)`; Start **ε = 1**, Ziel **0.01**; zu schnell = hört auf zu explorieren bevor gelernt, zu langsam = verschwendet Episoden. Bei ~Episode 700 ε≈0.5, ~2300 ε≈0.1. Gleiche Zahlen wie Ch11-Code (α=0.1, γ=0.99).
- **Q(s,a)** = erwarteter diskontierter Gesamt-Reward, wenn ich a in s nehme und danach gut handle; Q-Tabelle 16×4; State 14 neben Ziel: Q(14,→) ≈ 1, Q(0,↓) ≈ 0.9 (Wert schrumpft mit Distanz). **Kennst du Q, ist die Policy trivial: `π(s) = argmax_a Q(s,a)`** → Ch11 lernt die Tabelle.
- **Anwendungen:** Games, Robotik, Recommender (Clicks/Watch Time), Rechenzentrums-Kühlung, **RLHF** (LLM-Feintuning: Reward = Modell menschlicher Präferenzen, Action = nächstes Token). Meilensteine: Atari aus Pixeln 2013 · AlphaGo 2016 · AlphaStar 2019 · RLHF für ChatGPT 2022.
- **Demo (~8 min):** GridWorld bauen (16 States, 4 Aktionen) → Random Agent ≈ 1–2% → handgeschriebene Policy 100%, Return 0.95 → Sanity-Check für jeden State; Teaser-GIF: Q-Learning-Agent nach 1/50/3000 Episoden.

**Quiz:** Staubsauger-Roboter (+1 pro m², −5 Treppe): State = Position + Schmutzkarte (+ Akku), Action = Richtung/Saugen, Episode = ein Putzlauf · ε = 0 im ganzen Training → exploriert nie, bleibt bei schlechter Policy hängen · γ = 0: State 14 → (sofort +1), State 0 indifferent (überall −0.01, "sieht" das Ziel nicht → kein Credit Assignment).

**Bonus:** `03-exercises/ch10_rl_intro_exercises.ipynb` — Reward Shaping mit lernendem Agenten (Rewards ändern, Agent farmt Schritt-Rewards oder springt in Löcher); nach Ch11 oder zu Hause.

**Take-away:** RL = Lernen aus Rewards durch Trial & Error; State → Action → Reward + neuer State; MDP (S,A,P,R,γ); ε-greedy mit Decay; Q(s,a) = Langzeitwert, Policy = argmax; Reward-Design bestimmt, was man bekommt.

---

### Kapitel 11 — Q-Learning

**Ziel:** **Ein** Algorithmus richtig: tabellarisches Q-Learning (Q-Table + ε-greedy + TD-Update) auf GridWorld und Gymnasium-FrozenLake (~34 min + 10 min Exercise).

**Kernfakten:**
- **Von Ch10 zu Ch11:** Q kennen → Policy = argmax. Jetzt: Q **lernen** ohne Karte, P oder R zu kennen: Q = 0 überall, ε-greedy handeln, nach *jedem Schritt* Q(s,a) Richtung Beobachtung schieben. GIF `qtable_learning`: Wissen erscheint zuerst neben dem Ziel und breitet sich rückwärts aus.
- **Q-Table:** `np.zeros((16, 4))`; Policy = argmax pro Zeile; Ties → Action 0 (Left) → untrainierter Greedy-Agent rennt ewig gegen die linke Wand → deshalb ε startet bei 1.
- **TD-Update (Herzstück):**
  ```
  target  = r + γ · max_a' Q(s', a')          (Episode zu Ende in s' → target = r)
  Q(s,a) ← Q(s,a) + α · ( target − Q(s,a) )
                          └──── TD error ────┘
  ```
  Korrekt benannt: **Temporal-Difference-Update, abgeleitet aus der Bellman-Optimalitätsgleichung** Q*(s,a) = E[r + γ max Q*(s',a')] — die Gleichung ist der Fixpunkt, das Update der Weg dorthin.
- **Durchgerechneter Schritt** (α=0.1, γ=0.99, Q=0): State 14 → ins Ziel: target 1.0, TD-Fehler 1.0, **Q(14,→) = 0.10**. Nächste Episode 13 → 14 (r = −0.01): target = −0.01 + 0.99·0.10 = 0.089 → **Q(13,→) = 0.0089**. Wert **fließt rückwärts**, eine Zelle pro erfolgreichem Besuch; kleine Zahlen, aber die *Ordnung* zählt für argmax. ≥ 6 erfolgreiche Durchläufe bis State 0 ≠ 0. (Exercise Task 3 prüft genau 0.1 und 0.0089.)
- **Voller Algorithmus:** Q, ε = zeros, 1.0 → pro Episode: reset → pro Schritt: ε-greedy → `env_step(state, action)` → `target = reward + (0 if done else gamma·max Q[next])` → `Q[s,a] += alpha·(target − Q[s,a])` → bei done break → `epsilon = max(epsilon*0.999, 0.01)`. Gymnasium gibt 5-Tupel (obs, r, terminated, truncated, info) — alte 4-Wert-API ist weg.
- **Hyperparameter (unsere / typisch):** α **0.1** (0.05–0.5) · γ **0.99** (0.9–0.99) · ε **1 → 0.01**, ×0.999/Episode · Episoden **3 000–5 000** (10³–10⁵). Bar-Chart (echt, slippery, 5000 Ep.): α bis 0.5 ok, α ≥ 0.9 kollabiert (jeder Rutscher überschreibt die Schätzung); deterministische Welten tolerieren großes α, stochastische nicht.
- **FrozenLake (Gymnasium):** Map `SFFF/FHFH/FFFH/HFFG`, S=0, G=15; H → Episode endet, Reward 0; **Reward nur am Ziel (+1) → sparse**; 16 States, 4 Aktionen (←0 ↓1 →2 ↑3); `is_slippery=True` p=1/3. **Gleicher Code, zwei Welten:** deterministisch ~100%, slippery ~70% — auf Glatteis gewinnt selbst die *optimale* Policy nur ≈ 3 von 4 Episoden → **die Umgebung setzt die Decke, nicht der Algorithmus** ("was ist der bestmögliche Score?" fragen).
- **Warum RL schwer ist:** Sparse Rewards (Random Exploration muss erst ins Ziel stolpern) · Stochastizität (viele Samples, kleines α) · Credit Assignment (γ propagiert rückwärts). **Warum es trotzdem geht:** tabellarisches Q-Learning konvergiert beweisbar zu Q*, wenn jedes (s,a) unendlich oft besucht wird und α passend sinkt; praktisch: genug Episoden + ε-Decay.
- **Demo (~8 min):** FrozenLake in `reset/step` wrappen → Q-Learning in ~25 Zeilen auf festem Eis → gleicher Code auf Glatteis → Q-Table-Heatmaps + Greedy-Pfeile (Pfeile "in die Wand" neben Löchern = sicherer Zug). Fallback GridWorld ohne gymnasium. **Exercise (~10 min):** Tasks 1–4 Q-Table → ε-greedy → TD-Update → drei Zeilen im Trainingsloop, mit ▶-Check-Zellen; Bonus A slippery FrozenLake (warum nicht 100%?), Bonus B SARSA (eine Zeile ändern). Typische Bugs: `0 if done` vergessen, Q[next_state] statt Q[state] updaten, max über falsche Achse.

**Quiz:** α = 1 deterministisch fein (Target exakt), slippery: Q springt und beruhigt sich nie (0% bei α=1) · γ = 0: nur Einträge, deren Schritt *im* Ziel landet (Q(14,→)), nichts fließt zurück · ε noch 0.05 → Trainings-Erfolgsrate ≠ Erfolgsrate der gelernten Policy → **Greedy-Policy separat evaluieren** (`evaluate_greedy`; train ≠ test).

**Further Reading (nicht fürs Capstone nötig):** SARSA (on-policy, Target `r + γ Q(s',a')` mit tatsächlich nächster Aktion, vorsichtiger an Klippen; Bonus B) · DQN (NN statt Tabelle, Atari 2013) · Policy Gradient/PPO (π direkt, kontinuierliche Aktionen, Workhorse hinter RLHF) · Model-based (P und R lernen, planen; AlphaZero) · Off- vs. On-Policy (Q-Learning lernt greedy, verhält sich ε-greedy).

**Take-away:** Q-Learning = Q-Table + ε-greedy + TD-Update über viele Episoden; Target r + γ·max Q(s',·), TD-Fehler = Target − Schätzung, Schritt α; Wert fließt rückwärts (Credit Assignment); Umgebung setzt die Decke; **Greedy-Policy evaluieren, nicht den Trainingslauf**.

---

### Kapitel 12 — Capstone: End-to-End ML Workflow (Titanic)

**Ziel:** Den kompletten Workflow selbst bauen — 5 min Intro, **50 min Selbstarbeit** (niemand redet vorne), 10 min Debrief.

**Kernfakten:**
- **Setting:** 15. April 1912, 1 502 von 2 224 Menschen sterben. "Women and children first" steckt in den Daten (97% vs. 14%); Daten messy (Cabin 77% fehlend, Age 20%).
- **Dataset (Kaggle, `0-datasets/titanic.csv`, 891 Zeilen):** `Pclass` (ordinal 1/2/3, sozioökonomischer Proxy) · `Sex` · `Age` (20% fehlend → **in der Pipeline** imputieren) · `SibSp`, `Parch` (→ `FamilySize`) · `Fare` · `Embarked` (C/Q/S, 2 fehlend) · `Name` (→ `Title`: Mr/Mrs/Miss/Master/Rare) · `Cabin`, `Ticket`, `PassengerId` → **drop**. Target `Survived` (1 = ja), 38% überlebt → **Mehrheits-Baseline 62% Accuracy, F1 = 0**.
- **Plan (mit Kapitel-Zuordnung):** ① Load & Explore (EDA gegeben, Ch01) ② **Split first** `train_test_split(stratify=y)`, Test weggeschlossen (Ch02) ③ Feature Engineering `FamilySize`, `IsAlone`, `Title` (zeilenweise, kein Fitting; Ch02) ④ Preprocessing-Pipeline: `SimpleImputer` + `StandardScaler` / `OneHotEncoder` im `ColumnTransformer` (Ch02) ⑤ Baseline + 3 Modelle: `DummyClassifier`, LogReg, RandomForest, GradientBoosting — 5-fold CV via `cross_validate` (Ch03–06) ⑥ Finale Test-Evaluation: bestes CV-Modell, **ein** Blick aufs Test-Set, Report + Confusion Matrix (Ch06) ⑦ Interpretieren & reflektieren: Permutation Importance, Error Analysis, Fairness (Ch06) · Bonus `GridSearchCV`, `joblib`, PCA der Passagiere (Ch09).
- **Leakage-sicheres Preprocessing = Kern des Tages:** `cross_validate(pipe, X_train, y_train, cv=5)` fittet Imputer, Scaler, Encoder **in jedem Fold** nur auf dem Trainingsteil. Age-Median vor dem Split → Test-Alter beeinflussen den Median (klein hier, riesig bei Target Encoding).
- **Ein neues Modell — Gradient Boosting (ein Slide):** RF = viele tiefe Bäume **parallel** (Bootstrap) → Mittel; **GB = viele *flache* Bäume in Sequenz, jeder fittet die Fehler der vorherigen**; starker Default auf Tabellendaten, hyperparameter-sensitiver als Forest; `GradientBoostingClassifier(random_state=42)`, gleiche API.
- **Erfolgsmetrik:** Überlebende finden (Recall Klasse 1) und recht haben, wenn "überlebt" (Precision) → **F1 primär**, Accuracy daneben. FN = "gestorben" vorhergesagt, tatsächlich überlebt. Spielregeln: der Reihe nach, TODO-Zellen selbst, ▶-Zellen gegeben, Check-Zellen nutzen, Bonus nur wenn fertig.
- **Typische Stolpersteine:** `stratify=y` vergessen, Imputer außerhalb der Pipeline, `clf__`-Prefix im Bonus-Grid, `permutation_importance` auf dem Roh-Frame (`X_test_fe`), nicht auf der transformierten Matrix.
- **Debrief 1 — Leaderboard (GIF, Train-CV, seed 42):** Baseline 0.62 Acc / 0 F1; die drei echten Modelle innerhalb ~0.02 F1; erwartet LogReg ≈ GB ≥ RF — Reihenfolge kann mit anderem Seed kippen; Punkt: sie liegen nah beieinander.
- **Debrief 2 — Was hat das Modell gelernt:** Permutation Importance: **`Sex` und `Title` oben, dann `Pclass`/`Fare`**; impurity-basierte Importances (Ch05) hätten `Age`/`Fare` höher gerankt (kontinuierliche Spalten bekommen unfairen Bonus) → Permutation = ehrliche Version. Error Analysis: die Fehlklassifizierten sind die "überraschenden" Fälle (3.-Klasse-Männer, die überlebten; 1.-Klasse-Frauen, die starben). **Modelle nah beieinander → Features + saubere Evaluation zählen mehr als der Algorithmus.**
- **Debrief 3 — Fairness:** bestes Signal `Sex`, `Title` ist Proxy dafür (+ Alter, Status). Geschichte vorhersagen ok; dasselbe Rezept heute (Kredite, Triage, Versicherung) = Entscheidung nach geschütztem Attribut, direkt oder via Proxy. Vor Deployment: Fehlerraten **pro Gruppe**, Proxies, ob das Target selbst Diskriminierung kodiert. **`Sex` löschen würde es NICHT fixen** — `Title`, `Fare`, `Pclass` tragen das Signal.
- **Course Recap:** Ch01 Workflow ("this is it") · Ch02 Split first + Pipeline (Kern des Tages) · Ch03–06 fit/predict, Baseline, CV, F1, Confusion Matrix · Ch07–09 (Bonus) PCA · Ch10–11 Lernen aus Rewards.
- **Was danach:** Hyperparameter-Tuning (`GridSearchCV`/`RandomizedSearchCV`/Optuna — Bonus A), Deep Learning, Deployment & MLOps (`joblib` — Bonus B, Monitoring, Retraining), Praxis: Kaggle (genau dieses Dataset), fast.ai, Géron *Hands-On ML*.

**Take-away:** "Split first. Beat the baseline. Evaluate once. Ask what the model learned." — saubere Daten + Features + ehrliche Evaluation schlagen die Algorithmenwahl.

---

## Querschnitts-Merksätze

- **Cycle, not pipeline** — ML-Projekte sind iterativ (mind. 3 Loops).
- **Split first, then preprocess** — sonst Data Leakage; deterministische Fixes vor dem Split, alles mit Statistik danach.
- **Fit on train, transform on both** — `Pipeline` + `ColumnTransformer` erzwingen das; die Pipeline *ist* das Modell (auch in CV, auch mit PCA/K-Means).
- **Baseline first** — `DummyClassifier`/`DummyRegressor` in jeder Ergebnistabelle (R² = 0, Mehrheitsklasse).
- **Der Komplexitätsregler** — k (KNN), degree, 1/α, depth: Train-Fehler sinkt immer, **CV-Fehler** sagt, wo Schluss ist; Hyperparameter nie auf dem Test-Set wählen.
- **Tree-Modelle brauchen kein Scaling** — distanz- und gradientenbasierte Modelle schon (KNN, LogReg, Ridge/Lasso, SVM, K-Means, DBSCAN, PCA).
- **Accuracy lügt bei Imbalance** — Precision/Recall/F1/PR-Kurve; positive Klasse = 1 = das Gesuchte (Breast Cancer geflippt: malignant = 1); Threshold ist eine Entscheidung.
- **Cross-Validation mit Pipeline und richtigem `scoring`** — `KFold`/`StratifiedKFold(shuffle=True)`; Mean **und** Std vergleichen.
- **Clustering funktioniert immer** — Silhouette + Plot + Domänensinn; Clusters ≠ Classes.
- **PCA ≠ Feature Selection; t-SNE nur Visualisierung** — kein `.transform()`, keine Distanzen.
- **RL: Greedy-Policy evaluieren, nicht den Trainingslauf** — train ≠ test, auch hier; die Umgebung setzt die Decke.
- **Feature Engineering + saubere Evaluation > Algorithmus-Wahl** — und "sensible Spalte löschen" macht ein Modell nicht fair (Proxies).
