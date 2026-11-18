"""Train and compare candidate classifiers.

Run:  python -m models.train_compare            (after data_processing.build_dataset)

Protocol
  * Cross-validation: 5-fold GroupKFold on the 80 TRAIN trips (a trip never appears in both the
    fitting and validation folds). Training rows use every TRAIN_POINT_STRIDE-th reliable
    observation (consecutive 1 Hz points are near-duplicates); validation uses the same stride.
  * Final fit: all strided reliable TRAIN rows.
  * Student-test evaluation: every observation of the 10 STUDENT-TEST trips.
  * A rule-based nearest-candidate baseline is reported for reference (not a trained model).

Outputs: results/models/*.csv|png, models/artifacts/<name>.joblib
"""
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import GroupKFold

from configs import config as C
from evaluation.metrics import candidate_metrics, route_continuity, select_top1
from features.features import FEATURES
from models.model_zoo import make_models

OUT = C.RESULTS_DIR / "models"


def load_tables():
    tr = pd.read_parquet(C.PROCESSED_DIR / "train_candidates.parquet")
    te = pd.read_parquet(C.PROCESSED_DIR / "student_test_candidates.parquet")
    ptr = pd.read_parquet(C.PROCESSED_DIR / "train_points.parquet")
    pte = pd.read_parquet(C.PROCESSED_DIR / "student_test_points.parquet")
    return tr, te, ptr, pte


def strided_train_rows(tr):
    keep = tr.label_ok & (tr.point % C.TRAIN_POINT_STRIDE == 0)
    return tr[keep].reset_index(drop=True)


def n_reliable(points, trips, stride=1):
    p = points[points.trip_id.isin(trips) & points.label_ok]
    if stride > 1:
        p = p[p.point % stride == 0]
    return len(p)


def load_arcs(pdir, trip_ids):
    return {t: np.loadtxt(pdir / t / f"{t}.arcs", dtype=np.int64, ndmin=2) for t in trip_ids}


def continuity_of(df, col, arcs_by_trip):
    sel = select_top1(df, col)
    chosen = df.loc[sel, ["trip_id", "point", "arc"]].sort_values(["trip_id", "point"])
    vals = [route_continuity(g.arc.to_numpy(), arcs_by_trip[t]) for t, g in chosen.groupby("trip_id")]
    return float(np.mean(vals))


def cross_validate(name, factory, train_rows, ptr):
    gkf = GroupKFold(n_splits=C.CV_FOLDS)
    folds = []
    X, y, g = train_rows[FEATURES].to_numpy(), train_rows.y.to_numpy(), train_rows.trip_id.to_numpy()
    for k, (fi, vi) in enumerate(gkf.split(X, y, g)):
        est = factory()
        t0 = time.time()
        est.fit(X[fi], y[fi])
        fit_s = time.time() - t0
        val = train_rows.iloc[vi].copy()
        t0 = time.time()
        val["p"] = est.predict_proba(X[vi])[:, 1]
        pred_s = time.time() - t0
        m = candidate_metrics(val, n_reliable(ptr, val.trip_id.unique(), C.TRAIN_POINT_STRIDE))
        # training-fold score for over/underfitting diagnosis
        # (scored on a fixed 15% sample of the fitting-fold observations to bound cost)
        trn = train_rows.iloc[fi]
        pts_key = trn.trip_id + ":" + trn.point.astype(str)
        uniq = pts_key.unique()
        keep = set(np.random.default_rng(C.SEED).choice(uniq, size=max(1, int(0.15 * len(uniq))), replace=False))
        mask = pts_key.isin(keep).to_numpy()
        trn = trn[mask].copy()
        trn["p"] = est.predict_proba(X[fi][mask])[:, 1]
        mt = candidate_metrics(trn, len(keep))
        folds.append({"model": name, "fold": k, **{f"val_{a}": b for a, b in m.items()},
                      **{f"train_{a}": b for a, b in mt.items()}, "fit_s": fit_s, "pred_s": pred_s})
        print(f"  {name} fold {k}: val top1={m['top1_accuracy']:.4f} F1={m['macro_f1']:.4f} "
              f"train top1={mt['top1_accuracy']:.4f} ({fit_s:.0f}s)", flush=True)
    return folds


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    C.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(C.SEED)
    tr, te, ptr, pte = load_tables()
    train_rows = strided_train_rows(tr)
    te_ok = te[te.label_ok].copy()
    n_te = n_reliable(pte, te.trip_id.unique())
    arcs_te = load_arcs(C.STUDENT_TEST_DIR, te.trip_id.unique())
    print(f"train rows (strided, reliable): {len(train_rows)} from {train_rows.trip_id.nunique()} trips; "
          f"student-test rows: {len(te)}", flush=True)

    all_folds, summary, histories = [], [], {}

    # ---- rule-based baseline: nearest candidate ----
    te_ok["p"] = 1.0 / (1.0 + te_ok.dist)
    te_all = te.copy(); te_all["p"] = 1.0 / (1.0 + te_all.dist)
    tr_b = train_rows.copy(); tr_b["p"] = 1.0 / (1.0 + tr_b.dist)
    gk = GroupKFold(C.CV_FOLDS)
    cvb = []
    for _, vi in gk.split(tr_b, groups=tr_b.trip_id):
        v = tr_b.iloc[vi]
        cvb.append(candidate_metrics(v, n_reliable(ptr, v.trip_id.unique(), C.TRAIN_POINT_STRIDE)))
    mb = candidate_metrics(te_ok, n_te)
    summary.append({"model": "NearestCandidate (baseline)", "family": "Rule-based",
                    "config": "argmin perpendicular distance; score=1/(1+d)",
                    "cv_macro_f1": np.mean([c["macro_f1"] for c in cvb]),
                    "cv_macro_f1_std": np.std([c["macro_f1"] for c in cvb]),
                    "cv_top1": np.mean([c["top1_accuracy"] for c in cvb]),
                    "cv_log_loss": np.nan,
                    "test_top1": mb["top1_accuracy"], "test_macro_f1": mb["macro_f1"],
                    "test_precision": mb["precision"], "test_recall": mb["recall"],
                    "test_log_loss": np.nan,
                    "test_route_continuity": continuity_of(te_all, "p", arcs_te),
                    "train_time_s": 0.0, "inference_time_s": 0.0, "inference_ms_per_obs": 0.0})

    test_preds = {}
    for name, (family, cfg, factory) in make_models().items():
        print(f"== {name}", flush=True)
        folds = cross_validate(name, factory, train_rows, ptr)
        all_folds += folds
        est = factory()
        t0 = time.time()
        est.fit(train_rows[FEATURES].to_numpy(), train_rows.y.to_numpy())
        train_s = time.time() - t0
        t0 = time.time()
        p_all = est.predict_proba(te[FEATURES].to_numpy())[:, 1]
        inf_s = time.time() - t0
        te_all = te.copy(); te_all["p"] = p_all
        ok = te_all[te_all.label_ok].copy()
        m = candidate_metrics(ok, n_te)
        test_preds[name] = ok
        fd = pd.DataFrame(folds)
        summary.append({"model": name, "family": family, "config": cfg,
                        "cv_macro_f1": fd.val_macro_f1.mean(), "cv_macro_f1_std": fd.val_macro_f1.std(),
                        "cv_top1": fd.val_top1_accuracy.mean(), "cv_log_loss": fd.val_log_loss.mean(),
                        "cv_train_top1": fd.train_top1_accuracy.mean(),
                        "test_top1": m["top1_accuracy"], "test_macro_f1": m["macro_f1"],
                        "test_precision": m["precision"], "test_recall": m["recall"],
                        "test_log_loss": m["log_loss"],
                        "test_route_continuity": continuity_of(te_all, "p", arcs_te),
                        "train_time_s": train_s, "inference_time_s": inf_s,
                        "inference_ms_per_obs": 1000 * inf_s / te_all[["trip_id", "point"]].drop_duplicates().shape[0]})
        print(f"   student-test top1={m['top1_accuracy']:.4f} F1={m['macro_f1']:.4f} "
              f"logloss={m['log_loss']:.4f}", flush=True)
        joblib.dump(est, C.ARTIFACT_DIR / f"{name}.joblib", compress=3)
        if name == "MLP":
            mlp = est[-1]
            histories["MLP"] = {"train_loss": list(map(float, mlp.loss_curve_)),
                                "val_accuracy": list(map(float, mlp.validation_scores_))}
        if name == "LightGBM":
            histories["LightGBM"] = lightgbm_history(train_rows)

    summ = pd.DataFrame(summary)
    summ.to_csv(OUT / "model_comparison.csv", index=False)
    pd.DataFrame(all_folds).to_csv(OUT / "cv_folds.csv", index=False)
    with open(OUT / "training_histories.json", "w") as f:
        json.dump(histories, f)
    make_plots(summ, pd.DataFrame(all_folds), test_preds, histories)
    print(summ[["model", "cv_macro_f1", "cv_top1", "test_top1", "test_macro_f1", "test_log_loss",
                "test_route_continuity", "inference_ms_per_obs"]].to_string(index=False))


def lightgbm_history(train_rows):
    """Boosting-round training/validation log-loss on a group-held-out 20% of train trips."""
    import lightgbm as lgb
    trips = np.array(sorted(train_rows.trip_id.unique()))
    rng = np.random.default_rng(C.SEED)
    val_trips = set(rng.choice(trips, size=len(trips) // 5, replace=False))
    m = train_rows.trip_id.isin(val_trips)
    est = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                             random_state=C.SEED, n_jobs=2, verbose=-1)
    est.fit(train_rows.loc[~m, FEATURES], train_rows.loc[~m, "y"],
            eval_set=[(train_rows.loc[~m, FEATURES], train_rows.loc[~m, "y"]),
                      (train_rows.loc[m, FEATURES], train_rows.loc[m, "y"])],
            eval_names=["train", "valid"], eval_metric="binary_logloss")
    r = est.evals_result_
    return {"train_logloss": r["train"]["binary_logloss"], "val_logloss": r["valid"]["binary_logloss"]}


def make_plots(summ, folds, test_preds, histories):
    trained = summ[~summ.model.str.contains("baseline")]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(trained.model, trained.cv_macro_f1, yerr=trained.cv_macro_f1_std, capsize=4, color="tab:blue")
    ax.set(ylabel="CV macro F1 (5-fold GroupKFold)", title="Cross-validation macro F1 by model",
           ylim=(max(0, trained.cv_macro_f1.min() - 0.05), 1))
    for i, v in enumerate(trained.cv_macro_f1):
        ax.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=8)
    plt.xticks(rotation=20); fig.tight_layout(); fig.savefig(OUT / "cv_macro_f1.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(summ)); w = 0.38
    ax.bar(x - w / 2, summ.test_top1, w, label="Top-1 accuracy")
    ax.bar(x + w / 2, summ.test_macro_f1, w, label="Macro F1")
    ax.set_xticks(x, summ.model, rotation=20)
    ax.set(ylim=(max(0, min(summ.test_top1.min(), summ.test_macro_f1.min()) - 0.05), 1),
           title="Student-test performance (observation level)")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "student_test_top1_f1.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(trained.model, trained.test_log_loss, color="tab:red")
    ax.set(ylabel="log loss (lower is better)", title="Student-test log loss")
    plt.xticks(rotation=20); fig.tight_layout(); fig.savefig(OUT / "student_test_log_loss.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(summ.model, summ.test_route_continuity, color="tab:green")
    ax.set(ylabel="route continuity", title="Student-test route continuity (classifier only)", ylim=(0, 1))
    plt.xticks(rotation=20); fig.tight_layout(); fig.savefig(OUT / "student_test_continuity.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, (name, df) in zip(axes.ravel(), test_preds.items()):
        sel = select_top1(df, "p").astype(int)
        cm = confusion_matrix(df.y, sel)
        ConfusionMatrixDisplay(cm, display_labels=["not chosen", "true arc"]).plot(ax=ax, colorbar=False, values_format="d")
        ax.set(title=name, xlabel="predicted (top-1 selected)", ylabel="reference")
    fig.suptitle("Candidate-level confusion matrices (student test)")
    fig.tight_layout(); fig.savefig(OUT / "confusion_matrices.png", dpi=130); plt.close(fig)

    if histories:
        plot_histories(histories)


def plot_histories(histories):
    """Epoch-wise curves only for models that actually produce training histories."""
    fig, axes = plt.subplots(1, len(histories), figsize=(6 * len(histories), 4))
    axes = np.atleast_1d(axes)
    for ax, (name, h) in zip(axes, histories.items()):
        if name == "MLP":
            l1 = ax.plot(h["train_loss"], c="tab:blue", label="training loss (log loss)")
            ax2 = ax.twinx()
            l2 = ax2.plot(h["val_accuracy"], c="tab:orange", label="validation accuracy (10% hold-out)")
            ax2.set_ylabel("validation accuracy (row level)")
            ax.set(xlabel="epoch", ylabel="training loss", title="MLP training history (early stopping)")
            ax.legend(l1 + l2, [l.get_label() for l in l1 + l2], loc="center right")
        else:
            ax.plot(h["train_logloss"], label="train")
            ax.plot(h["val_logloss"], label="validation (held-out trips)")
            ax.set(xlabel="boosting round", ylabel="binary log loss", title="LightGBM training history")
            ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "training_histories.png", dpi=130); plt.close(fig)

if __name__ == "__main__":
    main()
