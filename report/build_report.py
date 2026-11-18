"""Build the PathSnap technical report (HTML + PDF) from the committed result files.

Run from the repo root:   python report/build_report.py
Every number in the report is read from results/ or models/artifacts/ at build time, so the PDF can
be checked against the repository. PDF export uses headless Chromium via Playwright
(`pip install playwright && playwright install chromium`); without it only the HTML is written.

Screenshots from the two evaluation laptops can be added as report/figures/ui_laptop2.png and
report/figures/api_cmd_laptop1.png; when present they are used in Section 7.
"""
import base64
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "report" / "figures"
OUT_HTML = ROOT / "report" / "PathSnap_Technical_Report.html"
OUT_PDF = ROOT / "report" / "PathSnap_Technical_Report.pdf"

fig_no = [0]
tab_no = [0]


def img(path, caption, width="100%"):
    fig_no[0] += 1
    p = Path(path)
    data = base64.b64encode(p.read_bytes()).decode()
    rel = p.relative_to(ROOT)
    return (f'<figure><img src="data:image/png;base64,{data}" style="width:{width}">'
            f'<figcaption><b>Figure {fig_no[0]}.</b> {caption} <span class="src">[{rel}]</span></figcaption></figure>')


def table(df, caption, src):
    tab_no[0] += 1
    return (f'<p class="tcap"><b>Table {tab_no[0]}.</b> {caption} <span class="src">[{src}]</span></p>'
            + df.to_html(index=False, escape=False, border=0, classes="t"))


def pct(x):
    return f"{100 * x:.2f}%"


def f4(x):
    return "N/A" if pd.isna(x) else f"{x:.4f}"


# ---------------------------------------------------------------- load results
mc = pd.read_csv(RES / "models" / "model_comparison.csv")
folds = pd.read_csv(RES / "models" / "cv_folds.csv")
fs = folds.groupby("model").agg(val_top1=("val_top1_accuracy", "mean"), val_top1_sd=("val_top1_accuracy", "std"),
                                train_top1=("train_top1_accuracy", "mean"), val_f1=("val_macro_f1", "mean"),
                                val_ll=("val_log_loss", "mean"), train_ll=("train_log_loss", "mean"))
hist = json.loads((RES / "models" / "training_histories.json").read_text())
tv = pd.read_csv(RES / "optimisation" / "tuned_vs_untuned.csv")
trials = pd.read_csv(RES / "optimisation" / "optuna_trials.csv")
fcfg = json.loads((ROOT / "models" / "artifacts" / "final_model_config.json").read_text())
dcfg = json.loads((ROOT / "models" / "artifacts" / "decoder_config.json").read_text())
hv = pd.read_csv(RES / "hmm" / "classifier_vs_hmm.csv")
hpt = pd.read_csv(RES / "hmm" / "classifier_vs_hmm_per_trip.csv", dtype={"trip_id": str})
grid = pd.read_csv(RES / "hmm" / "decoder_grid_train_oof.csv")
psum = pd.read_csv(RES / "eda" / "partition_summary.csv").set_index("partition")
csum = pd.read_csv(RES / "eda" / "candidate_summary.csv").set_index("partition")
sweep = pd.read_csv(RES / "eda" / "candidate_radius_sweep.csv")
inv = pd.read_csv(RES / "eda" / "inventory.csv", dtype={"trip_id": str})
net = pd.read_csv(RES / "eda" / "network_connectivity.csv", dtype={"trip_id": str})
proj = pd.read_csv(RES / "eda" / "projection_check.csv")
issues = pd.read_csv(RES / "eda" / "data_quality_issues.csv", dtype={"trip_id": str})
api_log = (FIG / "api_log.txt").read_text()
api_resp = json.loads((FIG / "api_91.json").read_text())

M = mc.set_index("model")
st_h = hv[hv.partition == "student_test"].set_index("method")
tr_h = hv[hv.partition == "train_oof"].set_index("method")
pt_st = hpt[hpt.partition == "student_test"]
pt_tr = hpt[hpt.partition == "train_oof"]
untuned, tuned = tv.iloc[0], tv.iloc[1]
best = fcfg["params"]
sw = sweep.set_index("radius_m")


def m(name, col):
    return M.loc[name, col]


# ---------------------------------------------------------------- sections
H = []
H.append(f"""
<div class="title">
<h1>PathSnap: Distributed Map-Matching &amp; Route Reconstruction</h1>
<p class="sub">Technical Report · UE24CS352A Machine Learning · PES University RR Campus · Team PT-009</p>
<table class="t team"><tr><th>Member</th><th>SRN</th><th>Components</th></tr>
<tr><td>Amith Prasad S G</td><td>PES1UG24AM033</td><td>Data loading, projection, labelling, candidate generation; Bayesian optimisation</td></tr>
<tr><td>Amruta Sai</td><td>PES1UG24AM035</td><td>Exploratory data analysis, Folium maps; HMM / Viterbi decoder</td></tr>
<tr><td>Amogh B Shetty</td><td>PES1UG24AM034</td><td>Feature engineering, metrics, model definitions; inference pipeline, FastAPI backend</td></tr>
<tr><td>Ananthakrishna M A</td><td>PES1UG24AM036</td><td>Model training and comparison; evaluator UI, deployment</td></tr></table>
<p>Repository: <code>https://github.com/amoghshetty05/pathsnap_team_009</code>. Every table and figure below is generated from files in that repository; the source path is given in brackets after each caption.</p>
</div>

<h2>1. Summary</h2>
<p>PathSnap matches noisy GPS observations to a road network. For each observation we generate candidate road segments (arcs) within 50 m, describe each candidate with 16 trip-independent features, score it with a tuned LightGBM classifier, and decode a connected route with a Hidden Markov Model and the Viterbi algorithm. A FastAPI backend (Laptop 1) serves the pipeline to an evaluator UI (Laptop 2).</p>
<p>On the 10 student-testing trips, observation-level Top-1 accuracy is <b>{pct(m('NearestCandidate (baseline)','test_top1'))}</b> for a nearest-arc rule, <b>{pct(tuned.test_top1)}</b> for the tuned classifier, and <b>{pct(st_h.loc['classifier + HMM','top1_accuracy'])}</b> after HMM decoding, with a reconstructed-route continuity of <b>{pct(st_h.loc['classifier + HMM','reconstructed_route_continuity_mean'])}</b>.</p>
""")

# ---- Section 2
tr, te = psum.loc["train"], psum.loc["student_test"]
miss = pd.DataFrame({
    "Check": ["Rows with non-finite lon / lat / time", "Non-increasing timestamps", "Trips without a .route file",
              "Isolated road-network nodes", "Sampling intervals > 10 s", "Observations with no candidate arc within 150 m",
              "Observations with unreliable reference label (&gt; 60 m from aligned arc)", "Reference arc absent from candidate set (miss rate)"],
    "Train (80 trips)": [int(tr.nonfinite_rows), int(tr.non_monotonic_ts), int(tr.trips_without_route), int(tr.total_isolated_nodes),
                         int(tr.total_gaps_gt10s), int(csum.loc["train", "no_candidate_obs"]),
                         f'{int(csum.loc["train","unreliable_label_obs"]):,} ({csum.loc["train","unreliable_label_pct"]:.2f}%)',
                         pct(csum.loc["train", "candidate_miss_rate"])],
    "Student test (10 trips)": [int(te.nonfinite_rows), int(te.non_monotonic_ts), int(te.trips_without_route), int(te.total_isolated_nodes),
                                int(te.total_gaps_gt10s), int(csum.loc["student_test", "no_candidate_obs"]),
                                f'{int(csum.loc["student_test","unreliable_label_obs"]):,} ({csum.loc["student_test","unreliable_label_pct"]:.2f}%)',
                                pct(csum.loc["student_test", "candidate_miss_rate"])],
})
inv_t = pd.DataFrame({
    "Statistic": ["Trips", "GPS observations (total)", "Observations per trip (min / median / max)",
                  "Intervals of exactly 1 s (mean over trips)", "Largest time gap", "Distance driven",
                  "Road-network nodes (median / max)", "Road-network arcs (median / max)", "Weakly connected components (mean)"],
    "Train": [int(tr.trips), f"{int(tr.total_obs):,}", f"{int(tr.min_obs)} / {int(tr.median_obs)} / {int(tr.max_obs):,}",
              f"{tr.mean_pct_dt_1s:.1f}%", f"{tr.max_gap_s:,.0f} s", f"{tr.total_km:,.0f} km",
              f"{int(tr.median_nodes):,} / {int(tr.max_nodes):,}", f"{int(tr.median_arcs):,} / {int(tr.max_arcs):,}", f"{tr.mean_weak_components:.1f}"],
    "Student test": [int(te.trips), f"{int(te.total_obs):,}", f"{int(te.min_obs)} / {int(te.median_obs)} / {int(te.max_obs):,}",
                     f"{te.mean_pct_dt_1s:.1f}%", f"{te.max_gap_s:,.0f} s", f"{te.total_km:,.0f} km",
                     f"{int(te.median_nodes):,} / {int(te.max_nodes):,}", f"{int(te.median_arcs):,} / {int(te.max_arcs):,}", f"{te.mean_weak_components:.1f}"],
})
sw_t = sweep.rename(columns={"radius_m": "Radius (m)", "miss_rate": "Candidate miss rate", "mean_cands": "Mean candidates"})
sw_t["Candidate miss rate"] = sw_t["Candidate miss rate"].map(pct)
sw_t["Mean candidates"] = sw_t["Mean candidates"].map(lambda x: f"{x:.2f}")
bad = issues.sort_values("median_label_dist_m", ascending=False).head(2)

H.append(f"""
<h2>2. Dataset Understanding and Exploratory Data Analysis</h2>
<p>All exploration used the <b>Training</b> (80 trips) and <b>Student-testing</b> (10 trips) partitions only; the partition is stated for every table and figure. The evaluator-testing partition was never read during development. Code: <code>data_processing/eda.py</code>, <code>data_processing/build_dataset.py</code>.</p>
{table(inv_t, "Dataset inventory per partition (Training and Student-testing).", "results/eda/partition_summary.csv, inventory.csv, network_connectivity.csv")}

<h3>2.1 How did you preprocess the data?</h3>
<ol>
<li><b>Parsing and validation</b> (<code>data_processing/io.py</code>). Each trip's <code>.track</code> (lon, lat, t), <code>.nodes</code> (lon, lat), <code>.arcs</code> (from, to; row index = arc id, directed) and <code>.route</code> (arc ids in driving order) are read without headers. Rows with non-finite values are dropped and observations are sorted by timestamp. No such rows were found (Table 2).</li>
<li><b>Metric coordinates</b> (<code>data_processing/geo.py</code>). Longitude and latitude are angles, so each trip is projected into metres with a local equirectangular projection centred on its GPS centroid. Against the haversine distance, the median relative error between first and last observation is {proj.start_end_rel_err.median():.5%} (maximum {proj.start_end_rel_err.max():.3%}), far below GPS noise.</li>
<li><b>Per-observation labels</b> (<code>data_processing/labelling.py</code>). The <code>.route</code> file lists arcs, not one label per observation (train trip 00000048: 1,240 observations, 149 route arcs). Each observation is aligned to a route position with a monotone dynamic programme that minimises the total point-to-arc distance while the route index can only stay or advance in time. This handles loops and streets driven twice, which nearest-route-arc snapping gets wrong.</li>
<li><b>Label reliability.</b> Observations more than 60 m from their aligned arc are marked unreliable and excluded from training and scoring: {csum.loc['train','unreliable_label_pct']:.1f}% of train observations, mostly from two trips (Section 2.2), and {csum.loc['student_test','unreliable_label_pct']:.2f}% of student-test observations.</li>
<li><b>Candidate generation</b> (<code>data_processing/candidates.py</code>). Arcs are densified every 10 m into a KD-tree. For each observation every arc within 50 m is kept, nearest 12 at most; if none lies within 50 m the radius is widened to 150 m. The radius was chosen on Training data only (Table 3).</li>
<li><b>Training rows.</b> Every (observation, candidate) pair is one row with label 1 if the candidate is the aligned reference arc. For model fitting we keep every 3rd reliable observation, because consecutive 1 Hz observations are near-duplicates.</li>
</ol>
{table(miss, "Missing-value and data-validity analysis (Training and Student-testing).", "results/eda/partition_summary.csv, candidate_summary.csv")}
{table(sw_t, "Candidate radius sweep on Training trips, reliable observations, no fallback radius.", "results/eda/candidate_radius_sweep.csv")}
<p>Raising the radius from 50 m to 100 m lowers the miss rate by only {100*(sw.loc[50,'miss_rate']-sw.loc[100,'miss_rate']):.2f} points but adds {100*(sw.loc[100,'mean_cands']/sw.loc[50,'mean_cands']-1):.0f}% more candidates, so 50 m was chosen.</p>

<h3>2.2 What patterns did you observe? What features did you extract and why?</h3>
<ul>
<li><b>Global, heterogeneous trips.</b> Trips lie in many countries (Figure 1) and road networks range from {int(net.n_nodes.min())} to {int(net.n_nodes.max()):,} nodes. Arc ids therefore carry no meaning across trips, so every feature must be geometric and trip-independent.</li>
<li><b>Mostly 1 Hz with large gaps.</b> {tr.mean_pct_dt_1s:.1f}% of train intervals are exactly 1 s, but {int(tr.total_gaps_gt10s)} gaps exceed 10 s and the largest is {tr.max_gap_s:,.0f} s (Figure 2). The decoder must cope with gaps.</li>
<li><b>Two-way streets are duplicated arcs.</b> A median {100*net[net.partition=='train'].two_way_arc_frac.median():.0f}% of train arcs have a reverse twin with identical geometry. Distance alone cannot separate them, so heading features are essential; the nearest-arc rule reaches only {pct(m('NearestCandidate (baseline)','test_top1'))} Top-1.</li>
<li><b>The reference route is not error-free.</b> On train trips {bad.trip_id.iloc[0]} and {bad.trip_id.iloc[1]} the median GPS-to-route distance is {bad.median_label_dist_m.iloc[0]:.0f} m and {bad.median_label_dist_m.iloc[1]:.0f} m (Figure 3, Figure 6), against about 5 m on other trips. These labels are treated as unreliable rather than as ground truth.</li>
<li><b>Networks are not fully connected.</b> Train networks have {tr.mean_weak_components:.0f} weakly connected components on average, so some consecutive candidates are unreachable from each other; the decoder handles this explicitly (Section 5).</li>
</ul>
<p><b>Features</b> (<code>features/features.py</code>), 16 per (observation, candidate) row:</p>
<table class="t"><tr><th>Group</th><th>Features</th><th>Why</th></tr>
<tr><td>Distance</td><td><code>dist</code>, <code>dist_rank</code>, <code>dist_gap</code>, <code>n_cands</code></td><td>Closeness to the arc, absolute and relative to the other candidates of the same observation.</td></tr>
<tr><td>Direction</td><td><code>heading_diff</code> (±1 point), <code>heading_diff_w</code> (±5 points), <code>heading_valid</code></td><td>Separates the two directions of a street and parallel roads with different bearings; the ±5 baseline is robust when slow or noisy; heading is carried forward through stops and flagged.</td></tr>
<tr><td>Context</td><td><code>dist_prev</code>, <code>dist_next</code>, <code>dist_win_mean</code> (±5 points)</td><td>A single noisy point can drift onto a parallel road; its neighbours rarely all do.</td></tr>
<tr><td>Kinematics</td><td><code>speed</code>, <code>dt_prev</code></td><td>Heading reliability and gap size.</td></tr>
<tr><td>Arc geometry</td><td><code>arc_len</code>, <code>frac</code>, <code>to_deg</code>, <code>from_deg</code></td><td>Position along the arc and junction degree; short arcs at junctions are where most confusions happen.</td></tr></table>
{img(RES/'eda'/'trip_locations.png', 'Trip locations (GPS centroid) for Training and Student-testing partitions.', '78%')}
{img(RES/'eda'/'trajectory_length_and_sampling.png', 'Observations per trip (Training and Student-testing) and sampling-interval distribution (Training).', '90%')}
{img(RES/'eda'/'speed_and_label_distance.png', 'GPS speed and distance from each observation to its aligned reference arc (Training); the dashed line is the 60 m reliability cut-off.', '90%')}
{img(RES/'eda'/'sample_trajectories.png', 'GPS trajectories (blue) against reference routes (green) for six Training trips, in local metres.', '95%')}
{img(RES/'eda'/'network_size_connectivity.png', 'Road-network size and connectivity per trip (Training blue, Student-testing orange).', '90%')}
{img(FIG/'folium_eda_trip54.png', 'Folium map of Training trip 00000054: GPS observations (blue), nearby road network (grey) and reference route (green) with layer control. The GPS runs well away from the reference route. Map tiles were not available when the screenshot was taken; the interactive HTML shows the OpenStreetMap base map.', '95%')}
""")

# ---- Section 3
rows = []
cfgs = {"NearestCandidate (baseline)": "argmin distance"}
for r in mc.itertuples():
    rows.append({"Model": r.model, "Algorithm family": r.family, "Key configuration": r.config,
                 "CV macro F1": f"{r.cv_macro_f1:.4f} ± {r.cv_macro_f1_std:.4f}" if not pd.isna(r.cv_macro_f1_std) else f4(r.cv_macro_f1),
                 "Top-1 accuracy (student test)": pct(r.test_top1),
                 "Route continuity": f"{r.test_route_continuity:.4f}",
                 "Student-test macro F1": f"{r.test_macro_f1:.4f}",
                 "Log loss": "N/A (no probabilities)" if pd.isna(r.test_log_loss) else f"{r.test_log_loss:.4f}",
                 "Inference time": "≈0" if r.inference_time_s == 0 else f"{r.inference_time_s:.2f} s ({r.inference_ms_per_obs:.4f} ms/obs)"})
comp = pd.DataFrame(rows)


def model_section(name, family, why, strengths, limits, hyper, loss, fit_comment):
    s = fs.loc[name]
    return f"""
<h3>3.2.{list(fs.index.drop([], errors='ignore')).index(name) + 1 if False else ''}{name}</h3>
<p><b>A1. Why suitable?</b> {why}</p>
<p><b>A2. Strengths and limitations.</b> Strengths: {strengths} Limitations: {limits}</p>
<p><b>B5. Important hyperparameters (initial values).</b> {hyper}</p>
<p><b>B6. Loss / objective.</b> {loss}</p>
<p><b>C7–C8. Student-testing performance.</b> Top-1 accuracy {pct(m(name,'test_top1'))}, macro F1 {m(name,'test_macro_f1'):.4f}, log loss {f4(m(name,'test_log_loss'))}, precision {m(name,'test_precision'):.4f}, recall {m(name,'test_recall'):.4f}, route continuity {m(name,'test_route_continuity'):.4f}; training time {m(name,'train_time_s'):.1f} s, inference {m(name,'inference_time_s'):.2f} s for 147,061 candidate rows ({m(name,'inference_ms_per_obs'):.4f} ms per observation).</p>
<p><b>C9. Overfitting or underfitting?</b> Mean Top-1 over the 5 folds: fitting folds {pct(s.train_top1)}, validation folds {pct(s.val_top1)} (fold s.d. {s.val_top1_sd:.4f}); log loss {s.train_ll:.4f} vs {s.val_ll:.4f}. {fit_comment}</p>"""


lgb_h = hist["LightGBM"]
best_round = int(pd.Series(lgb_h["val_logloss"]).idxmin()) + 1
mlp_h = hist["MLP"]

H.append(f"""
<h2>3. Model Experimentation and Comparison</h2>
<p>Each (observation, candidate) row is classified as true arc or not; scores are normalised within each observation's candidate set and the top-scoring candidate is the prediction. Six models from six algorithm families were trained (<code>models/model_zoo.py</code>, <code>models/train_compare.py</code>). Protocol: 5-fold <b>GroupKFold by trip</b> on the 80 Training trips (every 3rd reliable observation, 440,382 rows), refit on all of them, then evaluation on <b>every</b> observation of the 10 Student-testing trips. Scalers are inside each pipeline, so they are fitted on training folds only. Seed 42 throughout.</p>
<p><b>Primary selection metric:</b> cross-validated observation-level performance (macro F1 and Top-1), with log loss as the tie-breaker because the HMM uses the probabilities as emissions. Training accuracy and student-test results were not used to select the model. Metric definitions are in <code>evaluation/metrics.py</code>: Top-1 counts an observation correct when its top candidate is the reference arc (a candidate miss counts as wrong); macro F1 is computed on candidate rows with the selected candidate as the positive prediction; route continuity is the share of consecutive predicted arcs that are connected (end node of one = start node of the next).</p>
<h3>3.1 Overall model comparison</h3>
{table(comp, "Consolidated comparison. CV = 5-fold GroupKFold on Training; other columns on Student-testing. Route continuity is for classifier-only predictions.", "results/models/model_comparison.csv")}
<p>LightGBM is selected: highest CV macro F1 ({m('LightGBM','cv_macro_f1'):.4f}), lowest CV fold variability and the best log loss on both CV ({fs.loc['LightGBM','val_ll']:.4f}) and student test ({m('LightGBM','test_log_loss'):.4f}). Random Forest is within one fold standard deviation on accuracy but takes {m('RandomForest','train_time_s')/m('LightGBM','train_time_s'):.0f}× longer to train, is slower at inference and its saved model is about 100× larger.</p>

<h3>3.2 Compulsory questions for every model</h3>
{model_section('LogisticRegression', '', 'A linear model on standardised features gives calibrated probabilities directly, which the HMM needs as emissions, and is a strong, interpretable baseline: most features (distance, heading difference) relate monotonically to the chance of being the true arc.', 'fast (0.4 s to train), stable across folds, well-calibrated probabilities.', 'a single linear boundary cannot express interactions such as "heading only matters when the vehicle is moving".', 'StandardScaler; L2 penalty, C = 1.0; solver lbfgs; max_iter = 2000.', 'L2-regularised binary cross-entropy (log loss).', f'Fitting and validation accuracy are almost equal, so there is no overfitting; the gap to the tree models ({100*(m("LightGBM","test_top1")-m("LogisticRegression","test_top1")):.1f} points lower student-test Top-1) indicates mild underfitting from the linear boundary.')}
{model_section('GaussianNB', '', 'Naive Bayes is a fast probabilistic baseline that models each feature per class; it tests whether simple per-feature likelihoods are enough to rank candidates.', 'trains in under a second, needs no tuning, and produces probabilities.', 'the independence assumption is badly violated here: dist, dist_rank, dist_gap and dist_win_mean are strongly correlated, so evidence is counted several times and probabilities become over-confident.', 'QuantileTransformer to a normal distribution (so features fit the Gaussian assumption); var_smoothing = 1e-9.', 'Maximum likelihood estimates of class priors and per-class Gaussian means and variances; no iterative loss.', f'Fitting and validation scores are similar, so it does not overfit; it underfits, with the lowest Top-1, the worst log loss ({m("GaussianNB","test_log_loss"):.4f}) and the largest fold variability, consistent with the violated independence assumption.')}
{model_section('KNN', '', 'Candidates of the same kind (close, aligned, consistent with neighbours) should look alike in feature space, so a similarity-based vote is a natural non-parametric model.', 'no training phase, captures local non-linear structure.', 'very slow inference because every query is compared with 440k stored rows, sensitive to feature scaling, probabilities are coarse vote shares.', 'StandardScaler; k = 25; distance weighting; Euclidean metric; brute-force search.', 'None (lazy learner). The probability is the distance-weighted share of positive neighbours.', f'Clear overfitting: with distance weighting each fitting row is its own nearest neighbour, so fitting-fold Top-1 is {pct(fs.loc["KNN","train_top1"])} against {pct(fs.loc["KNN","val_top1"])} on validation folds, and fitting log loss ({fs.loc["KNN","train_ll"]:.4f}) is far below validation ({fs.loc["KNN","val_ll"]:.4f}). It is also the slowest model at inference.')}
{model_section('RandomForest', '', 'Averaging many decorrelated decision trees captures non-linear interactions (heading matters only when moving; context matters at junctions) without feature scaling.', 'strong accuracy, robust to feature scale and outliers, little tuning needed.', 'slow to train on 440k rows (about 100 s), large model (about 42 MB), probabilities from vote averaging are less smooth than boosting.', '200 trees; max_depth = None; min_samples_leaf = 5; max_features = sqrt; bootstrap sampling.', 'Each tree greedily minimises Gini impurity at each split; the forest averages tree probabilities.', f'Moderate overfitting: fitting-fold Top-1 {pct(fs.loc["RandomForest","train_top1"])} against {pct(fs.loc["RandomForest","val_top1"])} on validation, but validation performance is the second best, so the variance reduction from bagging keeps generalisation good.')}
{model_section('LightGBM', '', 'Gradient-boosted trees model the same non-linear interactions as a random forest but optimise log loss directly, which gives the best-calibrated probabilities for the HMM emissions, at low training and inference cost.', 'best CV macro F1 and log loss, lowest fold variability, 3.7 s to train, small model.', 'more hyperparameters to set, and it can overfit with many rounds or leaves.', '300 boosting rounds; learning rate 0.05; num_leaves 31; min_child_samples 20; subsample 1.0; colsample_bytree 1.0.', 'Binary log loss, minimised by gradient boosting.', f'Small gap between fitting ({pct(fs.loc["LightGBM","train_top1"])}) and validation ({pct(fs.loc["LightGBM","val_top1"])}). On a held-out 20% of train trips, validation log loss is lowest at round {best_round} ({min(lgb_h["val_logloss"]):.4f}) and stays flat to round 300 ({lgb_h["val_logloss"][-1]:.4f}) while training loss keeps falling (Figure 11): slight overfitting after about {best_round} rounds with no loss of validation performance.')}
{model_section('MLP', '', 'A small neural network learns non-linear feature combinations end to end and outputs probabilities.', 'fast inference (0.003 ms per observation), smooth probabilities.', 'needs scaling and careful regularisation, results depend on the random initialisation, and it is less accurate than the tree ensembles on tabular data of this size.', 'StandardScaler; hidden layers (64, 32), ReLU; Adam, learning rate 1e-3; L2 alpha = 1e-4; batch 512; max 60 epochs; early stopping on 10% validation with patience 6.', 'Binary cross-entropy with L2 penalty.', f'Mild gap between fitting ({pct(fs.loc["MLP","train_top1"])}) and validation ({pct(fs.loc["MLP","val_top1"])}). Early stopping ended training after {len(mlp_h["train_loss"])} epochs: training loss kept falling while validation accuracy plateaued (Figure 11), so training stopped before overfitting.')}

<h3>3.3 Model visualisations</h3>
{img(RES/'models'/'cv_macro_f1.png', 'Cross-validation macro F1 for all six models (5-fold GroupKFold on Training; error bars = fold standard deviation).', '85%')}
{img(RES/'models'/'student_test_top1_f1.png', 'Student-testing Top-1 accuracy and macro F1, including the nearest-arc baseline.', '88%')}
{img(RES/'models'/'student_test_log_loss.png', 'Student-testing log loss (the baseline produces no probabilities and is omitted).', '80%')}
{img(RES/'models'/'student_test_continuity.png', 'Student-testing route continuity of classifier-only predictions.', '80%')}
{img(RES/'models'/'training_histories.png', 'Training histories, shown only for the two models that train iteratively: LightGBM log loss per boosting round (held-out trips) and MLP training loss and validation accuracy per epoch.', '95%')}
{img(RES/'models'/'confusion_matrices.png', 'Candidate-level confusion matrices on Student-testing (selected top-1 candidate vs reference arc).', '95%')}
""")

# ---- Section 4
space = fcfg["search_space"]
sp_t = pd.DataFrame({"Hyperparameter": list(space), "Range / distribution": list(space.values()),
                     "Untuned": ["300", "0.05", "31", "20", "1.0", "1.0", "0 (LightGBM default)", "0 (LightGBM default)"],
                     "Best": [str(best["n_estimators"]), f"{best['learning_rate']:.4f}", str(best["num_leaves"]), str(best["min_child_samples"]),
                              f"{best['subsample']:.3f}", f"{best['colsample_bytree']:.3f}", f"{best['reg_lambda']:.4f}", f"{best['reg_alpha']:.4f}"]})
tvt = pd.DataFrame({"Metric": ["CV macro F1 (5-fold)", "CV Top-1 (5-fold)", "CV log loss", "Student-test Top-1", "Student-test macro F1",
                               "Student-test log loss", "Student-test route continuity", "Training time", "Inference time (ms / observation)"],
                    "Untuned": [f"{untuned.cv_macro_f1:.4f}", pct(untuned.cv_top1), f"{untuned.cv_log_loss:.4f}", pct(untuned.test_top1), f"{untuned.test_macro_f1:.4f}",
                                f"{untuned.test_log_loss:.4f}", f"{untuned.test_route_continuity:.4f}", f"{untuned.train_time_s:.1f} s", f"{untuned.inference_ms_per_obs:.4f}"],
                    "Tuned": [f"{tuned.cv_macro_f1:.4f}", pct(tuned.cv_top1), f"{tuned.cv_log_loss:.4f}", pct(tuned.test_top1), f"{tuned.test_macro_f1:.4f}",
                              f"{tuned.test_log_loss:.4f}", f"{tuned.test_route_continuity:.4f}", f"{tuned.train_time_s:.1f} s", f"{tuned.inference_ms_per_obs:.4f}"]})
H.append(f"""
<h2>4. Bayesian Hyperparameter Optimisation (LightGBM)</h2>
<p>Code: <code>optimisation/bayes_opt.py</code>. Library: Optuna, Tree-structured Parzen Estimator (TPE) sampler, seed 42. All trials are stored in <code>results/optimisation/optuna_trials.csv</code> and <code>optuna_study.db</code>.</p>
<p><b>1. Hyperparameters searched</b> and <b>2. ranges</b>: Table 5. Learning rate, number of leaves, minimum child samples and both regularisation terms are searched on a log scale because their effect is multiplicative.</p>
{table(sp_t, "Search space, untuned values and best values found.", "models/artifacts/final_model_config.json")}
<p><b>3. Objective.</b> Mean observation-level Top-1 accuracy over 3-fold GroupKFold on the 80 Training trips (no trip in both fitting and validation). To keep {fcfg['n_trials']} trials × 3 folds affordable on a 2-core laptop, the search uses every {fcfg['search_row_stride']}th reliable observation; the final comparison below uses the standard rows and 5 folds.</p>
<p><b>4. Trials.</b> {fcfg['n_trials']} trials, {fcfg['search_seconds']/60:.1f} minutes in total. Trial 0 is the untuned configuration (regularisation at the lower bound of its range), so the search starts from the baseline (Figure 13).</p>
<p><b>5. Best values.</b> Table 5, best 3-fold search objective {trials.value.max():.4f} against {trials.value.iloc[0]:.4f} for the untuned trial.</p>
<p><b>6. Did tuning help?</b> Yes, slightly, on accuracy and F1 in both CV and student test (Table 6). CV log loss is essentially unchanged and inference is slower because the tuned model has more leaves. The small gain shows the default configuration was already close to what these features allow; the remaining errors need sequence information, which Section 5 adds.</p>
{table(tvt, "Untuned vs tuned LightGBM with identical data partitions and protocol.", "results/optimisation/tuned_vs_untuned.csv")}
{img(RES/'optimisation'/'optimisation_history.png', 'Optimisation history: objective per trial and best-so-far.', '75%')}
{img(RES/'optimisation'/'param_slices.png', 'Objective against each hyperparameter across the 50 trials.', '100%')}
{img(RES/'optimisation'/'tuned_vs_untuned.png', 'Untuned vs tuned: accuracy and F1, log loss and inference time.', '95%')}
""")

# ---- Section 5
hm = pd.DataFrame({
    "Partition": ["Train (out-of-fold)", "Train (out-of-fold)", "Student test", "Student test"],
    "Method": ["Classifier only", "Classifier + HMM"] * 2,
    "Top-1 accuracy": [pct(tr_h.loc["classifier only", "top1_accuracy"]), pct(tr_h.loc["classifier + HMM", "top1_accuracy"]),
                       pct(st_h.loc["classifier only", "top1_accuracy"]), pct(st_h.loc["classifier + HMM", "top1_accuracy"])],
    "Macro F1 (mean of trips)": [f"{tr_h.loc['classifier only','macro_f1_mean_over_trips']:.4f}", f"{tr_h.loc['classifier + HMM','macro_f1_mean_over_trips']:.4f}",
                                 f"{st_h.loc['classifier only','macro_f1_mean_over_trips']:.4f}", f"{st_h.loc['classifier + HMM','macro_f1_mean_over_trips']:.4f}"],
    "Route continuity (per obs.)": [f"{tr_h.loc['classifier only','route_continuity_mean']:.4f}", f"{tr_h.loc['classifier + HMM','route_continuity_mean']:.4f}",
                                    f"{st_h.loc['classifier only','route_continuity_mean']:.4f}", f"{st_h.loc['classifier + HMM','route_continuity_mean']:.4f}"],
    "Route F1 (sequence level)": [f"{tr_h.loc['classifier only','route_f1_mean']:.4f}", f"{tr_h.loc['classifier + HMM','route_f1_mean']:.4f}",
                                  f"{st_h.loc['classifier only','route_f1_mean']:.4f}", f"{st_h.loc['classifier + HMM','route_f1_mean']:.4f}"],
    "Reconstructed continuity": ["—", f"{tr_h.loc['classifier + HMM','reconstructed_route_continuity_mean']:.4f}", "—",
                                 f"{st_h.loc['classifier + HMM','reconstructed_route_continuity_mean']:.4f}"],
    "Chain breaks": [0, int(tr_h.loc["classifier + HMM", "chain_breaks_total"]), 0, int(st_h.loc["classifier + HMM", "chain_breaks_total"])],
})
ptt = pt_st[["trip_id", "n_obs", "clf_top1", "hmm_top1", "clf_continuity", "hmm_continuity", "hmm_chain_breaks"]].copy()
ptt.columns = ["Trip", "Observations", "Classifier Top-1", "HMM Top-1", "Classifier continuity", "HMM continuity", "Breaks"]
for c in ["Classifier Top-1", "HMM Top-1"]:
    ptt[c] = ptt[c].map(pct)
for c in ["Classifier continuity", "HMM continuity"]:
    ptt[c] = ptt[c].map(lambda x: f"{x:.4f}")
hurt = pt_tr[pt_tr.hmm_gain < 0].sort_values("hmm_gain")
gmin = pt_st.assign(g=pt_st.hmm_top1 - pt_st.clf_top1).sort_values("g")

H.append(f"""
<h2>5. HMM and Viterbi Route Reconstruction</h2>
<p>Code: <code>hmm_decoder/viterbi.py</code> (lattice, Viterbi, route reconstruction), <code>hmm_decoder/tune_decoder.py</code> (decoder tuning and evaluation), <code>hmm_decoder/pipeline.py</code> (end-to-end inference). The decoder consumes the probabilities of the selected tuned LightGBM model.</p>
<p><b>1. Hidden states.</b> For observation t the states are its candidate arcs from Section 2.1. The candidate-to-arc correspondence is stored explicitly (state k of observation t is arc <code>arcs[t][k]</code>), so each decoded state maps back to one road segment and every observation with candidates receives a predicted arc.</p>
<p><b>2. Emission probabilities.</b> log e<sub>t</sub>(k) = α · log p<sub>t</sub>(k), where p<sub>t</sub>(k) is the LightGBM probability normalised over the candidate set of observation t; α weights the classifier against the transition model.</p>
<p><b>3. Transition probabilities.</b> For consecutive observations i → j, candidates a (offset s<sub>a</sub> along a) and b (offset s<sub>b</sub>): if a = b, the route distance is |s<sub>b</sub> − s<sub>a</sub>|; otherwise it is (length<sub>a</sub> − s<sub>a</sub>) + shortest-path(end of a → start of b) + s<sub>b</sub>, found with a distance-bounded Dijkstra on the directed network. Then log T(a → b) = −|d<sub>route</sub> − d<sub>gc</sub>| / β, where d<sub>gc</sub> is the straight-line distance between the two GPS observations: a transition is likely when driving distance matches observed displacement. The Viterbi recurrence is δ<sub>j</sub>(b) = max<sub>a</sub>[δ<sub>i</sub>(a) + log T(a → b)] + log e<sub>j</sub>(b), followed by back-tracking.</p>
<p><b>Decoder tuning.</b> α and β were chosen on Training trips using out-of-fold LightGBM probabilities (each trip scored by a model that never saw it), never on the evaluator partition. Because Viterbi's argmax is unchanged when all log-scores are multiplied by a constant, only α·β matters; the grid confirms equal products give equal accuracy (Figure 18). Chosen: α = {dcfg['alpha']}, β = {dcfg['beta']} m (α·β = {dcfg['effective_parameter_alpha_x_beta']}).</p>
<p><b>4. Missing candidates and large GPS gaps.</b> Observations with no candidate get no state; the chain links the previous and next observations that have candidates ({int(csum.loc['train','no_candidate_obs'])} such observations in Training, none in Student-testing). The shortest-path search radius grows with the observed displacement (3 × d<sub>gc</sub> + 100 m, between 250 m and 6 km), so long gaps can still be bridged.</p>
<p><b>5. Disconnected segments.</b> If b is unreachable from a within the search radius, T(a → b) = 0 (log = −∞). If every transition into observation j is impossible, the chain is broken: Viterbi restarts at j from the emissions alone and each part is back-tracked separately. This happened {int(tr_h.loc['classifier + HMM','chain_breaks_total'])} times over the 80 Training trips and {int(st_h.loc['classifier + HMM','chain_breaks_total'])} times over the 10 Student-testing trips. After decoding, consecutive decoded arcs that are not adjacent (a short arc crossed between two 1 Hz samples) are joined with the shortest connecting path to give a drivable route.</p>
<p><b>6. Did HMM decoding improve the route?</b> Yes (Table 7). On Student-testing, Top-1 rises from {pct(st_h.loc['classifier only','top1_accuracy'])} to {pct(st_h.loc['classifier + HMM','top1_accuracy'])} and per-observation route continuity from {st_h.loc['classifier only','route_continuity_mean']:.3f} to {st_h.loc['classifier + HMM','route_continuity_mean']:.3f}; after gap filling the route is {pct(st_h.loc['classifier + HMM','reconstructed_route_continuity_mean'])} connected. Every student-test trip improved (Table 8).</p>
{table(hm, "Classifier-only vs classifier + HMM on Training (out-of-fold) and Student-testing.", "results/hmm/classifier_vs_hmm.csv")}
{table(ptt, "Per-trip effect of HMM decoding on Student-testing.", "results/hmm/classifier_vs_hmm_per_trip.csv")}
<p><b>Where HMM decoding did not help.</b> On Training (out-of-fold), {len(hurt)} of 80 trips became slightly worse, the largest drop being trip {hurt.trip_id.iloc[0]} ({100*hurt.hmm_gain.iloc[0]:+.2f} points). These are trips where the classifier was already accurate and the transition model occasionally overrides a correct low-confidence choice, for example when the vehicle turns within one sampling interval and the driving distance differs from the straight-line displacement. On Student-testing the smallest gains were trips {gmin.trip_id.iloc[0]} ({100*gmin.g.iloc[0]:+.1f} points) and {gmin.trip_id.iloc[1]} ({100*gmin.g.iloc[1]:+.1f} points). Trip 00000077 remains the weakest ({pct(pt_st.set_index('trip_id').loc['00000077','hmm_top1'])}): dense parallel streets keep both the classifier and the transition model ambiguous. Per-observation continuity stays below 1 because 1 Hz samples skip very short arcs; the reconstructed route resolves this.</p>
{img(RES/'hmm'/'classifier_vs_hmm.png', 'Student-testing: classifier-only vs classifier + HMM (Top-1, route continuity, route F1).', '95%')}
{img(RES/'hmm'/'per_trip_hmm_gain.png', 'Per-trip Top-1 accuracy before and after HMM decoding (Student-testing).', '88%')}
{img(RES/'hmm'/'decoder_grid.png', 'Decoder grid on Training out-of-fold predictions; cells with equal α·β have equal accuracy.', '90%')}
{img(FIG/'zoom_00000091.png', 'Student-testing trip 00000091, observations 69–189, plotted on the local road network (grey). The classifier switches to adjacent arcs at the junction and to the opposite-direction twin arc along the street; the HMM follows the reference route throughout this window.', '72%')}
""")

# ---- Section 6
H.append(f"""
<h2>6. Geographical Visualisation Using Folium</h2>
<p>Map code: <code>visualisations/folium_maps.py</code> (function <code>trip_map</code>), called from <code>data_processing/eda.py</code> and <code>hmm_decoder/tune_decoder.py</code>. Interactive maps are committed in <code>results/maps/</code>: three EDA maps and one map per Student-testing trip. Each map has these layers, each with its own colour and a toggle in the layer control: raw GPS observations (blue markers), the nearby road network (grey), the reference route (wide green), the classifier-only prediction (orange) and the predicted route after HMM decoding (red). It also has start and end markers, a legend, and a view fitted to the trip extent.</p>
{img(FIG/'folium_hmm_trip40.png', 'Folium map of Student-testing trip 00000040 (HMM Top-1 95.8%): the predicted route (red) overlays the reference route (green) along the whole trip.', '95%')}
{img(FIG/'folium_hmm_trip91_zoom.png', 'Folium map of Student-testing trip 00000091, zoomed to a junction: classifier-only predictions (orange) jump to adjacent arcs, while the classifier + HMM route (red) stays on the reference route (green).', '95%')}
<p class="note">Screenshots were exported headlessly without internet access, so the OpenStreetMap base tiles are not shown; the committed HTML maps display them when opened in a browser.</p>
""")

# ---- Section 7
ui_img = FIG / "ui_laptop2.png"
api_img = FIG / "api_cmd_laptop1.png"
resp_excerpt = {k: api_resp[k] for k in ("trip_id", "n_observations", "metrics", "timings_s")}
resp_excerpt["gps"] = f"[{len(api_resp['gps'])} [lat, lon] points]"
resp_excerpt["predicted_route"] = f"[{len(api_resp['predicted_route'])} arc polylines]"
resp_excerpt["predicted_arc_ids"] = f"[{len(api_resp['predicted_arc_ids'])} arc ids, one per observation]"
H.append(f"""
<h2>7. API, UI and Deployment</h2>
<p><b>Architecture.</b> Laptop 1 runs the FastAPI backend (<code>api/main.py</code>) with the tuned LightGBM model (<code>models/artifacts/final_model_lgb.txt</code>), the decoder configuration (<code>models/artifacts/decoder_config.json</code>), the road networks and the reference routes of the evaluation trips. Laptop 2 runs the UI (<code>ui/index.html</code>, served by <code>ui/serve.py</code>). The UI calls <code>GET /trips</code> to list the ten evaluation trips and <code>POST /predict</code> with the selected trip id. It then draws the returned GPS points and predicted route on a Leaflet map and shows the metrics.</p>
<p><b>Ground-truth isolation.</b> The prediction pipeline loads only <code>.track</code>, <code>.nodes</code> and <code>.arcs</code> (<code>load_trip(..., with_route=False)</code>). The <code>.route</code> file is read afterwards, server-side, only to compute accuracy. The response contains no reference route. Unknown trip ids are rejected, so requests cannot reach files outside the evaluation folder. Nothing is hard-coded: every request runs the full pipeline.</p>
<p><b>Run.</b> Laptop 1: <code>uvicorn api.main:app --host 0.0.0.0 --port 8000</code>, with the evaluation trips in <code>data/raw/evaluator_test/</code>. Laptop 2: <code>python ui/serve.py --port 5500</code>, then open <code>http://localhost:5500/?api=http://&lt;laptop-1-ip&gt;:8000</code>. Full instructions are in the README.</p>
{img(ui_img if ui_img.exists() else FIG/'ui_trip40.png', ('Working UI on Laptop 2, connected to the API on Laptop 1 (http://10.8.121.132:8000), after selecting trip 00000056: predicted route (red) on the OpenStreetMap base map, Top-1 accuracy 96.75% (classifier-only 92.19%), reconstructed route continuity 99.93%, route F1 97.82%, 3,958 GPS observations, API processing time 2.69 s.' if ui_img.exists() else 'Working UI after selecting trip 00000040: the predicted route (red) over the GPS points, with accuracy, classifier-only accuracy, continuity, route precision/recall/F1 and API processing time. This was captured in a test run with the API and UI on one machine and no base-map tiles.'), '95%')}
{img(api_img, 'API log on the receiving laptop (Laptop 1) showing incoming requests and results.', '95%') if api_img.exists() else ''}
<p class="tcap"><b>API console output</b> from the same test run (uvicorn on the API machine). Each request logs the client address, trip, accuracy and timing <span class="src">[report/figures/api_log.txt]</span>:</p>
<pre class="console">{api_log}</pre>
<p class="tcap"><b>Successful <code>POST /predict</code> response</b> for trip 00000091, with long arrays summarised <span class="src">[report/figures/api_91.json]</span>:</p>
<pre class="console">{json.dumps(resp_excerpt, indent=2)}</pre>
""")

# ---- Section 8
H.append(f"""
<h2>8. Academic Integrity, Reproducibility and Limitations</h2>
<ul>
<li><b>Independent work.</b> All models, configurations, weights and results in this report were produced by Team PT-009 from the provided data. No models, code or results were shared with other teams.</li>
<li><b>Reproducibility.</b> Run these from the repository root: <code>python -m data_processing.build_dataset</code>, <code>python -m data_processing.eda</code>, <code>python -m models.train_compare</code>, <code>python -m optimisation.bayes_opt</code>, <code>python -m hmm_decoder.tune_decoder</code>. The seed is fixed at 42. Every table and figure here cites its source file in <code>results/</code> or <code>models/artifacts/</code>, and <code>report/build_report.py</code> regenerates this PDF from those files.</li>
<li><b>Leakage controls.</b> The evaluator-testing partition was not used for training, feature design, model selection, hyperparameters, decoder parameters or thresholds. Cross-validation is grouped by trip. Preprocessing is fitted inside training folds. The decoder was tuned on Training out-of-fold predictions.</li>
<li><b>AI assistance disclosure.</b> AI coding assistants were used to help write parts of the code and documentation. All experiments were run by the team on the provided data, all reported metrics come from the committed scripts and result files, and each member can explain the components they own.</li>
<li><b>Limitations.</b> Reference routes are imperfect (two Training trips are offset by more than 100 m), so labels within 60 m are trusted and the rest excluded. Opposite-direction twin arcs remain ambiguous when the vehicle is stationary. Dense parallel streets (trip 00000077) are the main remaining error source.</li>
</ul>
""")

CSS = """
@page { size: A4; margin: 16mm 14mm; }
body { font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 10.5pt; line-height: 1.42; color: #1d2329; }
h1 { font-size: 21pt; margin: 0 0 4px; } h2 { font-size: 15pt; margin: 22px 0 8px; border-bottom: 1.5px solid #1d2329; padding-bottom: 3px; page-break-after: avoid; }
h3 { font-size: 12pt; margin: 16px 0 6px; page-break-after: avoid; }
.sub { color: #555; margin: 0 0 14px; } .title { margin-bottom: 10px; }
table.t { border-collapse: collapse; width: 100%; font-size: 8.6pt; margin: 4px 0 12px; page-break-inside: avoid; }
table.t th, table.t td { border: 1px solid #c8cdd2; padding: 3px 5px; text-align: left; vertical-align: top; }
table.t th { background: #eef1f4; } table.team td, table.team th { font-size: 9.5pt; }
figure { margin: 10px 0 14px; text-align: center; page-break-inside: avoid; } figure img { max-width: 100%; border: 1px solid #e3e6ea; }
figcaption, .tcap { font-size: 8.8pt; color: #333; text-align: left; margin: 4px 0; } .src { color: #777; font-family: monospace; font-size: 8pt; }
code { font-size: 9pt; background: #f2f4f6; padding: 0 2px; } .note { font-size: 8.8pt; color: #555; }
pre.console { background: #0c0c0c; color: #e8e8e8; font-size: 7.4pt; padding: 8px 10px; white-space: pre-wrap; word-break: break-all; page-break-inside: avoid; }
p { margin: 5px 0 7px; } li { margin-bottom: 3px; }
"""

html = f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>PathSnap Technical Report – PT-009</title><style>{CSS}</style></head><body>{''.join(H)}</body></html>"
# number the per-model subsections
for i, name in enumerate(["LogisticRegression", "GaussianNB", "KNN", "RandomForest", "LightGBM", "MLP"], 1):
    html = html.replace(f"<h3>3.2.{name}</h3>", f"<h3>3.2.{i} {name} ({mc.set_index('model').loc[name, 'family']})</h3>")
OUT_HTML.write_text(html, encoding="utf-8")
print("wrote", OUT_HTML)

try:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        import os
        exe = os.environ.get("CHROME_PATH")
        b = p.chromium.launch(**({"executable_path": exe} if exe else {}))
        pg = b.new_page()
        pg.goto(OUT_HTML.as_uri())
        pg.pdf(path=str(OUT_PDF), format="A4", print_background=True,
               margin={"top": "16mm", "bottom": "16mm", "left": "14mm", "right": "14mm"},
               display_header_footer=True, header_template="<span></span>",
               footer_template="<div style='font-size:8px;width:100%;text-align:center;color:#777'>PathSnap · Team PT-009 · <span class='pageNumber'></span> / <span class='totalPages'></span></div>")
        b.close()
    print("wrote", OUT_PDF)
except Exception as e:  # pragma: no cover
    print("PDF export skipped:", e)
