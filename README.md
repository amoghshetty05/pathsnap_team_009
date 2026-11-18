# PathSnap – Distributed Map-Matching & Route Reconstruction

UE24CS352A Machine Learning hackathon · PES University RR Campus · Team **PT-009**

**Status: Phase 1.** This phase covers data understanding, preprocessing, candidate generation, feature engineering and the comparison of six models. Phase 2 will add Bayesian hyperparameter optimisation, HMM/Viterbi route reconstruction, the FastAPI backend and the evaluator UI.

| Member | SRN | Phase 1 components |
|---|---|---|
| Amith Prasad S G | PES1UG24AM033 | Data loading, projection, per-observation labelling, candidate generation (`configs/`, `data_processing/` except EDA) |
| Amruta Sai | PES1UG24AM035 | Exploratory data analysis and Folium maps (`data_processing/eda.py`, `visualisations/`, `results/eda`, `results/maps`) |
| Amogh B Shetty | PES1UG24AM034 | Feature engineering, evaluation metrics, model definitions (`features/`, `evaluation/`, `models/model_zoo.py`) |
| Ananthakrishna M A | PES1UG24AM036 | Model training and comparison, results, README (`models/train_compare.py`, `models/artifacts/`, `results/models`) |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Put the provided data at `data/raw/train/<trip>/…` and `data/raw/student_test/<trip>/…`. The data is not committed.

## Run (from the repo root, in this order)

```bash
python -m data_processing.build_dataset   # per-observation labels, candidates, features -> data/processed/
python -m data_processing.eda             # EDA tables, plots, Folium maps -> results/eda, results/maps
python -m models.train_compare            # 6 models + baseline, 5-fold GroupKFold CV, student-test -> results/models
```
All randomness is seeded (`SEED = 42`). The evaluator-testing partition is never read.

## Approach

* **Raw data.** Each trip has its own road network. Arcs are directed: both directions of a two-way street are separate arcs with identical geometry. `.route` lists the arcs driven in order (e.g. 149 arcs for 1,240 GPS points), so it does not label individual observations.
* **Distances.** Each trip uses a local equirectangular projection around its GPS centroid, giving metres. It is checked against haversine in `results/eda/projection_check.csv`.
* **Per-observation labels.** Observations are aligned to route positions with a monotone dynamic programme (`data_processing/labelling.py`), which handles loops and repeated streets. Observations more than 60 m from their aligned arc are marked unreliable and excluded from training and scoring.
* **Candidates.** All arcs within 50 m of an observation, up to 12 nearest. The radius was chosen from a sweep on train only (`results/eda/candidate_radius_sweep.csv`).
* **Features.** 16 trip-independent geometric and kinematic features (`features/features.py`). Arc ids are not used, since they mean nothing across trips.
* **Task.** Binary classification of each (observation, candidate) row. Scores are normalised within each observation's candidate set, and the top-scoring candidate is the prediction.
* **Protocol.** 5-fold GroupKFold by trip on the 80 train trips, using every 3rd observation (1 Hz neighbours are near-duplicates). Final fit on all train rows, then evaluation on every observation of the 10 student-test trips.

## Data summary (`results/eda/`)

| Partition | Trips | Observations | Unreliable labels | Candidate miss rate | Mean candidates |
|---|---|---|---|---|---|
| train | 80 | 208,417 | 10.72% | 0.91% | 7.38 |
| student_test | 10 | 19,160 | 0.13% | 0.08% | 7.68 |

Main data-quality findings: train trips 00000054 and 00000033 sit more than 100 m away from their reference routes. Some trips have very large time gaps (up to 153,064 s). Trips are spread across many countries, and road networks range from 299 to 155,003 nodes. Full details are in `results/eda/data_quality_issues.csv`.

## Model comparison (`results/models/model_comparison.csv`)

| Model | Family | CV macro F1 | CV Top-1 | Test Top-1 | Test macro F1 | Test log loss | Route continuity | Inference ms/obs |
|---|---|---|---|---|---|---|---|---|
| NearestCandidate (baseline) | Rule-based | 0.7430 | 0.5553 | 0.5390 | 0.7352 | N/A | 0.4284 | 0.000 |
| LogisticRegression | Linear / probabilistic | 0.9318 | 0.8783 | 0.8716 | 0.9264 | 0.1068 | 0.7277 | 0.001 |
| GaussianNB | Linear / probabilistic | 0.9003 | 0.8244 | 0.8280 | 0.9013 | 0.1952 | 0.6356 | 0.011 |
| KNN | Instance-based | 0.9312 | 0.8773 | 0.8689 | 0.9249 | 0.1123 | 0.7464 | 4.191 |
| RandomForest | Ensemble (bagging) | 0.9386 | 0.8898 | 0.8976 | 0.9413 | 0.0655 | 0.7647 | 0.052 |
| LightGBM | Boosting | 0.9398 | 0.8920 | 0.8951 | 0.9399 | 0.0647 | 0.7676 | 0.034 |
| MLP | Neural network | 0.9360 | 0.8856 | 0.8861 | 0.9348 | 0.0729 | 0.7677 | 0.003 |

* **Top-1 accuracy:** the share of reliable observations whose top-scored candidate is the reference arc.
* **Macro F1:** computed at candidate level, with the selected candidate = 1 and the others = 0.
* **Log loss:** binary, on the normalised candidate probabilities.
* **Route continuity:** the share of consecutive predicted arcs that are connected in the network.

LightGBM is the strongest model on CV macro F1 and log loss, and Random Forest is within noise of it.
