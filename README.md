# PathSnap – Distributed Map-Matching & Route Reconstruction

UE24CS352A Machine Learning hackathon · PES University RR Campus · Team **PT-009**

| Member | SRN | Phase 1 | Phase 2 |
|---|---|---|---|
| Amith Prasad S G | PES1UG24AM033 | Data loading, projection, labelling, candidates (`configs/`, `data_processing/`) | Bayesian hyperparameter optimisation (`optimisation/`) |
| Amruta Sai | PES1UG24AM035 | EDA and Folium maps (`data_processing/eda.py`, `visualisations/`) | HMM / Viterbi decoder and decoder tuning (`hmm_decoder/`) |
| Amogh B Shetty | PES1UG24AM034 | Features, metrics, model definitions (`features/`, `evaluation/`, `models/model_zoo.py`) | Inference pipeline and FastAPI backend (`hmm_decoder/pipeline.py`, `api/`) |
| Ananthakrishna M A | PES1UG24AM036 | Model training and comparison (`models/train_compare.py`) | Evaluator UI, deployment, README (`ui/`) |

PathSnap matches noisy GPS observations to a road network. A classifier scores each candidate road segment and an HMM decoded with Viterbi turns the scores into a connected route. A FastAPI backend (Laptop 1) serves the pipeline to an evaluator UI (Laptop 2).

## Pipeline

```
.track/.nodes/.arcs ─► local metric projection ─► candidate arcs (≤12 within 50 m)
      ─► 16 trip-independent features ─► LightGBM P(candidate is true arc)
      ─► HMM (states = candidate arcs) + Viterbi ─► gap-filled drivable route
```

## Repository layout

| Folder | Contents |
|---|---|
| `configs/` | Central configuration (paths, radii, seeds) |
| `data_processing/` | File reading, geometry/projection, per-observation label alignment, candidate generation, EDA, dataset build |
| `features/` | Feature engineering for (observation, candidate) pairs |
| `models/` | Model zoo, training + 5-fold GroupKFold comparison, saved models (`models/artifacts/`) |
| `evaluation/` | Metric definitions (Top-1, macro F1, log loss, candidate miss rate, route continuity, route P/R/F1) |
| `optimisation/` | Bayesian optimisation (Optuna TPE), trial log, tuned-vs-untuned comparison |
| `hmm_decoder/` | HMM lattice, Viterbi, route reconstruction, end-to-end pipeline, decoder tuning/evaluation |
| `visualisations/` | Folium map generation |
| `api/` | FastAPI backend (Laptop 1) |
| `ui/` | Evaluator UI (Laptop 2) |
| `results/` | All CSVs, logs and plots behind the reported numbers; `results/maps/` has the interactive Folium HTML maps |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Put the provided data at `data/raw/train/<trip>/…` and `data/raw/student_test/<trip>/…`. The data is not committed.

## Reproducing the results (run from the repo root, in this order)

```bash
python -m data_processing.build_dataset   # labels, candidates, features -> data/processed/
python -m data_processing.eda             # EDA tables, plots, Folium maps -> results/eda, results/maps
python -m models.train_compare            # 6 models + baseline, 5-fold GroupKFold, student test -> results/models
python -m optimisation.bayes_opt          # Optuna TPE (50 trials) on LightGBM -> results/optimisation, final model
python -m hmm_decoder.tune_decoder        # decoder tuning on TRAIN OOF, classifier vs HMM -> results/hmm, maps
```
All randomness is seeded (`SEED = 42`). The evaluator-testing partition is never read by any of these scripts.

## Deployment (two laptops)

**Laptop 1: API.** Copy the evaluator trips to `data/raw/evaluator_test/<trip>/` (or set `PATHSNAP_EVAL_DIR`).
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Every request is logged to the console: client IP, trip, accuracy and timing. Show this log to the evaluator.

**Laptop 2: UI.** Both laptops must be on the same network.
```bash
python ui/serve.py --port 5500
# open http://localhost:5500/?api=http://<laptop-1-ip>:8000
```
The UI lists the evaluation trips. Selecting one POSTs `{"trip_id": ...}` to `/predict` and draws the GPS points and the predicted route, along with the metrics. The map tiles need internet access.

API endpoints: `GET /health`, `GET /trips`, `POST /predict`. The prediction pipeline loads only `.track/.nodes/.arcs`. The `.route` file is read afterwards, on the server, only to compute accuracy, and the reference route is never sent to the UI.

## Key design decisions

* **Labels per observation.** `.route` lists the arcs driven, not one label per GPS point. Each observation is aligned to a route position with a monotone dynamic programme (`data_processing/labelling.py`). Observations more than 60 m from their aligned arc are marked unreliable and excluded from training and scoring. These are 10.7 % of train observations, mostly from trips 00000054 and 00000033, where the GPS and the reference route disagree by more than 100 m.
* **Distances.** Each trip uses a local equirectangular projection around its centroid, checked against haversine in `results/eda/projection_check.csv`.
* **Candidates.** All arcs within 50 m, up to 12 per observation. The candidate miss rate on train is 1.10 % at 50 m vs 0.98 % at 100 m with 27 % more candidates (`results/eda/candidate_radius_sweep.csv`).
* **Features.** Trip-independent geometry and kinematics (distance, rank, heading difference over two baselines, distance to the same arc at neighbouring observations, arc length/degree, speed, time gap). Arc ids are not features, because every trip has its own road network.
* **Evaluation protocol.** 5-fold GroupKFold by trip on the 80 train trips (every 3rd observation, since 1 Hz neighbours are near-duplicates), then all observations of the 10 student-test trips.

## Results (all values produced by the scripts above)

### Model comparison (`results/models/model_comparison.csv`)

| Model | Family | CV macro F1 | Test Top-1 | Test macro F1 | Test log loss | Route continuity | Inference ms/obs |
|---|---|---|---|---|---|---|---|
| NearestCandidate (baseline) | Rule-based | 0.7430 | 0.5390 | 0.7352 | N/A | 0.4284 | 0.000 |
| LogisticRegression | Linear / probabilistic | 0.9318 | 0.8716 | 0.9264 | 0.1068 | 0.7277 | 0.001 |
| GaussianNB | Linear / probabilistic | 0.9003 | 0.8280 | 0.9013 | 0.1952 | 0.6356 | 0.011 |
| KNN | Instance-based | 0.9312 | 0.8689 | 0.9249 | 0.1123 | 0.7464 | 4.191 |
| RandomForest | Ensemble (bagging) | 0.9386 | 0.8976 | 0.9413 | 0.0655 | 0.7647 | 0.052 |
| LightGBM | Boosting | 0.9398 | 0.8951 | 0.9399 | 0.0647 | 0.7676 | 0.034 |
| MLP | Neural network | 0.9360 | 0.8861 | 0.9348 | 0.0729 | 0.7677 | 0.003 |

LightGBM was selected. It has the best CV macro F1 and the best log loss (calibrated probabilities matter for the HMM emissions). Its inference is faster and its artifact smaller than Random Forest's, which is within noise on accuracy.

### Bayesian optimisation (`results/optimisation/`)

| Model | CV macro F1 | Test Top-1 | Test macro F1 | Test log loss | Inference ms/obs |
|---|---|---|---|---|---|
| LightGBM (untuned) | 0.9398 | 0.8951 | 0.9399 | 0.0647 | 0.036 |
| LightGBM (tuned) | 0.9404 | 0.8976 | 0.9414 | 0.0644 | 0.057 |

Best parameters: `{"n_estimators": 350, "learning_rate": 0.08299575156189523, "num_leaves": 63, "min_child_samples": 27, "subsample": 0.8328999032452618, "colsample_bytree": 0.5510034437341257, "reg_lambda": 0.5706494482259403, "reg_alpha": 0.007060938615615895}`

### HMM / Viterbi (`results/hmm/`)

The decoder has α (emission weight) = 2.0 and β = 0.5 m. These were chosen on train out-of-fold predictions. Only α·β affects the Viterbi argmax.

| Partition | Method | Top-1 | Route continuity (per obs) | Reconstructed route continuity | Reconstructed route F1 |
|---|---|---|---|---|---|
| train_oof | classifier only | 0.8931 | 0.7590 | 0.9815 | 0.8395 |
| train_oof | classifier + HMM | 0.9431 | 0.8980 | 0.9990 | 0.9755 |
| student_test | classifier only | 0.8976 | 0.7789 | 0.9919 | 0.8785 |
| student_test | classifier + HMM | 0.9363 | 0.9125 | 0.9990 | 0.9895 |

## Known limitations

* The reference routes are not error-free. Two training trips are offset from their routes by more than 100 m, which is why the label-reliability filter exists.
* Directed arcs on two-way streets share geometry, so at standstill (no reliable heading) the direction is ambiguous. The HMM resolves most of these cases using connectivity.
* Student-test trip 00000077 is the weakest (83 % Top-1). Its dense parallel streets keep the classifier ambiguous.
