"""Bayesian hyperparameter optimisation of the selected classifier (LightGBM) with Optuna (TPE).

Run:  python -m optimisation.bayes_opt          (after models.train_compare)

Search
    Objective: mean observation-level Top-1 accuracy over 3-fold GroupKFold on the 80 TRAIN trips
    (grouped by trip), using the same strided reliable training rows as the model comparison.
    Search rows: every SEARCH_STRIDE-th reliable observation (cost control); see constant below.
    Sampler: TPE (seed 42), N_TRIALS trials (the first is the untuned configuration).

After the search
    * untuned (default config from model_zoo) and tuned models are both scored with the same
      5-fold GroupKFold CV and on the student-test partition;
    * the tuned model is refitted on all strided TRAIN rows and saved as models/artifacts/final_model.joblib;
    * out-of-fold probabilities for every TRAIN observation (5-fold GroupKFold) and student-test
      probabilities are saved for decoder tuning (hmm_decoder/tune_decoder.py).

Outputs: results/optimisation/*, models/artifacts/final_model.joblib, models/artifacts/final_model_config.json
"""
import json
import time

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

from configs import config as C
from evaluation.metrics import candidate_metrics, route_continuity, select_top1
from features.features import FEATURES
from models.train_compare import load_arcs, load_tables, n_reliable, strided_train_rows

OUT = C.RESULTS_DIR / "optimisation"
N_TRIALS = 50
SEARCH_FOLDS = 3
# The search uses every 6th reliable observation (half the comparison rows) to bound the cost of
# 50 trials x 3 folds on a 2-core laptop. Final untuned-vs-tuned scoring uses the standard rows.
SEARCH_STRIDE = 6

UNTUNED = dict(n_estimators=300, learning_rate=0.05, num_leaves=31)

SEARCH_SPACE = {
    "n_estimators": "int [100, 600], step 50",
    "learning_rate": "float [0.01, 0.2], log-uniform",
    "num_leaves": "int [15, 255], log-uniform",
    "min_child_samples": "int [5, 200], log-uniform",
    "subsample": "float [0.5, 1.0] (bagging_freq=1)",
    "colsample_bytree": "float [0.5, 1.0]",
    "reg_lambda": "float [1e-3, 10], log-uniform",
    "reg_alpha": "float [1e-3, 10], log-uniform",
}


def make_lgbm(params):
    p = dict(random_state=C.SEED, n_jobs=2, verbose=-1)
    p.update(params)
    if p.get("subsample", 1.0) < 1.0:
        p["subsample_freq"] = 1
    return lgb.LGBMClassifier(**p)


def cv_score(params, rows, ptr, folds, stride=C.TRAIN_POINT_STRIDE):
    X, y, g = rows[FEATURES].to_numpy(), rows.y.to_numpy(), rows.trip_id.to_numpy()
    res = []
    for fi, vi in GroupKFold(folds).split(X, y, g):
        est = make_lgbm(params).fit(X[fi], y[fi])
        v = rows.iloc[vi].copy()
        v["p"] = est.predict_proba(X[vi])[:, 1]
        res.append(candidate_metrics(v, n_reliable(ptr, v.trip_id.unique(), stride)))
    return pd.DataFrame(res)


def evaluate_on_test(est, te, pte, arcs_te):
    t0 = time.time()
    p = est.predict_proba(te[FEATURES].to_numpy())[:, 1]
    inf_s = time.time() - t0
    te_all = te.copy(); te_all["p"] = p
    m = candidate_metrics(te_all[te_all.label_ok], n_reliable(pte, te.trip_id.unique()))
    sel = select_top1(te_all, "p")
    ch = te_all.loc[sel].sort_values(["trip_id", "point"])
    m["route_continuity"] = float(np.mean([route_continuity(g.arc.to_numpy(), arcs_te[t])
                                           for t, g in ch.groupby("trip_id")]))
    m["inference_time_s"] = inf_s
    m["inference_ms_per_obs"] = 1000 * inf_s / te_all[["trip_id", "point"]].drop_duplicates().shape[0]
    return m, p


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tr, te, ptr, pte = load_tables()
    rows = strided_train_rows(tr)
    search_rows = rows[rows.point % SEARCH_STRIDE == 0].reset_index(drop=True)
    arcs_te = load_arcs(C.STUDENT_TEST_DIR, te.trip_id.unique())

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 200, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        }
        t0 = time.time()
        r = cv_score(params, search_rows, ptr, SEARCH_FOLDS, stride=SEARCH_STRIDE)
        trial.set_user_attr("cv_macro_f1", float(r.macro_f1.mean()))
        trial.set_user_attr("cv_log_loss", float(r.log_loss.mean()))
        trial.set_user_attr("seconds", time.time() - t0)
        print(f"trial {trial.number}: top1={r.top1_accuracy.mean():.4f} ({time.time() - t0:.0f}s)", flush=True)
        return float(r.top1_accuracy.mean())

    study = optuna.create_study(direction="maximize", study_name="pathsnap_lgbm",
                                sampler=optuna.samplers.TPESampler(seed=C.SEED),
                                storage=f"sqlite:///{OUT / 'optuna_study.db'}", load_if_exists=False)
    # enqueue the untuned configuration so the search starts from the baseline
    study.enqueue_trial({"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31,
                         "min_child_samples": 20, "subsample": 1.0, "colsample_bytree": 1.0,
                         "reg_lambda": 1e-3, "reg_alpha": 1e-3})
    t_search = time.time()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=N_TRIALS)
    search_s = time.time() - t_search
    trials = study.trials_dataframe()
    trials.to_csv(OUT / "optuna_trials.csv", index=False)
    best = study.best_params
    print("best params:", best, "best CV top1:", study.best_value)

    # ---- untuned vs tuned with identical protocol ----
    comp = []
    final = None
    for name, params in [("untuned", UNTUNED), ("tuned", best)]:
        cv = cv_score(params, rows, ptr, C.CV_FOLDS)
        t0 = time.time()
        est = make_lgbm(params).fit(rows[FEATURES].to_numpy(), rows.y.to_numpy())
        fit_s = time.time() - t0
        m, p = evaluate_on_test(est, te, pte, arcs_te)
        comp.append({"model": f"LightGBM ({name})", "cv_macro_f1": cv.macro_f1.mean(),
                     "cv_macro_f1_std": cv.macro_f1.std(), "cv_top1": cv.top1_accuracy.mean(),
                     "cv_log_loss": cv.log_loss.mean(), "test_top1": m["top1_accuracy"],
                     "test_macro_f1": m["macro_f1"], "test_log_loss": m["log_loss"],
                     "test_route_continuity": m["route_continuity"], "train_time_s": fit_s,
                     "inference_time_s": m["inference_time_s"],
                     "inference_ms_per_obs": m["inference_ms_per_obs"]})
        if name == "tuned":
            final = est
    comp = pd.DataFrame(comp)
    comp.to_csv(OUT / "tuned_vs_untuned.csv", index=False)
    print(comp.to_string(index=False))

    joblib.dump(final, C.ARTIFACT_DIR / "final_model.joblib", compress=3)
    # native LightGBM text model: loads on any machine regardless of scikit-learn version
    final.booster_.save_model(str(C.ARTIFACT_DIR / "final_model_lgb.txt"))
    with open(C.ARTIFACT_DIR / "final_model_config.json", "w") as f:
        json.dump({"algorithm": "LightGBM", "params": best, "features": FEATURES,
                   "objective": "mean 3-fold GroupKFold observation-level top-1 accuracy",
                   "search_row_stride": SEARCH_STRIDE,
                   "n_trials": N_TRIALS, "search_folds": SEARCH_FOLDS, "search_space": SEARCH_SPACE,
                   "search_seconds": search_s, "seed": C.SEED}, f, indent=2)

    # ---- out-of-fold predictions on every TRAIN observation for decoder tuning ----
    X_all = tr[FEATURES].to_numpy()
    oof = np.zeros(len(tr))
    for fi_trips, vi in _trip_folds(tr):
        fit_rows = rows[rows.trip_id.isin(fi_trips)]
        est = make_lgbm(best).fit(fit_rows[FEATURES].to_numpy(), fit_rows.y.to_numpy())
        oof[vi] = est.predict_proba(X_all[vi])[:, 1]
    tr[["trip_id", "point", "arc", "dist", "frac", "y", "label_ok"]].assign(p=oof).to_parquet(
        C.PROCESSED_DIR / "train_oof_predictions.parquet", index=False)
    te[["trip_id", "point", "arc", "dist", "frac", "y", "label_ok"]].assign(
        p=final.predict_proba(te[FEATURES].to_numpy())[:, 1]).to_parquet(
        C.PROCESSED_DIR / "student_test_predictions.parquet", index=False)

    plots(trials, comp)


def _trip_folds(tr):
    trips = np.array(sorted(tr.trip_id.unique()))
    for fi, vi in GroupKFold(C.CV_FOLDS).split(trips, groups=trips):
        val = set(trips[vi])
        yield set(trips[fi]), np.where(tr.trip_id.isin(val).to_numpy())[0]


def plots(trials, comp):
    t = trials[trials.state == "COMPLETE"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t.number, t.value, "o", alpha=0.6, label="trial")
    ax.plot(t.number, t.value.cummax(), "-", c="tab:red", label="best so far")
    ax.set(xlabel="trial", ylabel="CV top-1 accuracy (3-fold GroupKFold)", title="Bayesian optimisation history (TPE)")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "optimisation_history.png", dpi=130); plt.close(fig)

    pcols = [c for c in t.columns if c.startswith("params_")]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for ax, c in zip(axes.ravel(), pcols):
        ax.scatter(t[c], t.value, s=14)
        ax.set(xlabel=c.replace("params_", ""), ylabel="CV top-1")
        if c.replace("params_", "") in ("learning_rate", "num_leaves", "min_child_samples", "reg_lambda", "reg_alpha"):
            ax.set_xscale("log")
    fig.suptitle("Objective vs hyperparameter (each point = one trial)")
    fig.tight_layout(); fig.savefig(OUT / "param_slices.png", dpi=130); plt.close(fig)

    metrics = ["cv_macro_f1", "test_top1", "test_macro_f1"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].bar(metrics, comp.iloc[0][metrics], width=-0.4, align="edge", label="untuned")
    axes[0].bar(metrics, comp.iloc[1][metrics], width=0.4, align="edge", label="tuned")
    lo = comp[metrics].to_numpy().min()
    axes[0].set(ylim=(lo - 0.02, min(1, comp[metrics].to_numpy().max() + 0.01)), title="Accuracy / F1")
    axes[0].legend(); axes[0].tick_params(axis="x", rotation=15)
    axes[1].bar(["untuned", "tuned"], comp.test_log_loss, color=["tab:blue", "tab:orange"])
    axes[1].set(title="Student-test log loss")
    axes[2].bar(["untuned", "tuned"], comp.inference_ms_per_obs, color=["tab:blue", "tab:orange"])
    axes[2].set(title="Inference time (ms / observation)")
    fig.tight_layout(); fig.savefig(OUT / "tuned_vs_untuned.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
