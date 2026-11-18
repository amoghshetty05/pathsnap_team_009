"""Tune the HMM decoder parameters (alpha, beta) and evaluate classifier-only vs classifier + HMM.

Run:  python -m hmm_decoder.tune_decoder        (after optimisation.bayes_opt)

* Decoder parameters are chosen on TRAIN trips using OUT-OF-FOLD classifier probabilities (the
  classifier never saw the trip it is scoring), so the decoder is tuned on realistic inputs.
* The chosen parameters are then applied unchanged to the STUDENT-TEST trips.
* The evaluator partition is never touched.

Outputs: results/hmm/*.csv|png, results/maps/hmm_student_test_<trip>.html,
         models/artifacts/decoder_config.json
"""
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from configs import config as C
from data_processing.io import load_trip
from evaluation.metrics import candidate_metrics, route_continuity, route_overlap
from hmm_decoder.pipeline import rows_to_lists
from hmm_decoder.viterbi import RoadGraph, build_lattice, decode_full, reconstruct_route
from visualisations.folium_maps import trip_map

OUT = C.RESULTS_DIR / "hmm"
# Note: Viterbi's argmax is unchanged when every log-score is multiplied by a constant, so
# (alpha, beta) only matter through their product alpha*beta. A 2-D grid makes this visible
# (equal products give equal accuracy); the finer beta values refine the optimum.
ALPHAS = [0.5, 1.0, 2.0]
BETAS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0]


def prepare(partition, pdir, preds, points):
    """Build lattices once per trip (they do not depend on alpha/beta)."""
    out = {}
    for tid, g in preds.groupby("trip_id"):
        t0 = time.time()
        trip = load_trip(pdir, tid)
        n = len(trip.xy)
        arcs, frac, prob = rows_to_lists(g, n)
        clf = np.full(n, -1, np.int64)
        for i in range(n):
            if len(arcs[i]):
                clf[i] = arcs[i][int(np.argmax(prob[i]))]
        graph = RoadGraph(trip)
        lat = build_lattice(trip, arcs, frac, prob, graph)
        pt = points[points.trip_id == tid].sort_values("point")
        out[tid] = dict(trip=trip, lat=lat, graph=graph, clf=clf, label=pt.label_arc.to_numpy(),
                        ok=pt.label_ok.to_numpy(), rows=g)
        print(f"[{partition}] lattice {tid}: {n} obs, {time.time() - t0:.1f}s", flush=True)
    return out


def score(prepared, alpha=None, beta=None, detail=False):
    """Observation-level accuracy pooled over trips; optionally per-trip details."""
    correct = total = 0
    per_trip = []
    for tid, d in prepared.items():
        if alpha is None:
            pred, breaks = d["clf"], 0
        else:
            pred, breaks = decode_full(d["lat"], len(d["label"]), alpha, beta)
        ok = d["ok"]
        c = int((pred[ok] == d["label"][ok]).sum())
        correct += c; total += int(ok.sum())
        if detail:
            trip = d["trip"]
            row = {"trip_id": tid, "n_obs": len(pred), "n_reliable": int(ok.sum()),
                   "top1_accuracy": c / max(1, ok.sum()), "route_continuity": route_continuity(pred, trip.arcs),
                   "chain_breaks": breaks, **route_overlap(pred, trip.route, trip.seg_len)}
            route = reconstruct_route(pred, d["graph"])
            row["reconstructed_route_continuity"] = route_continuity(route, trip.arcs)
            row.update({f"reconstructed_{k}": v for k, v in
                        route_overlap(route, trip.route, trip.seg_len).items()})
            # candidate-level macro F1: the decoded arc is the selected candidate
            rows = d["rows"][d["rows"].label_ok].copy()
            rows["s"] = (rows.arc.to_numpy() == pred[rows.point.to_numpy()]).astype(float)
            m = candidate_metrics(rows, int(ok.sum()), col="s")
            row["macro_f1"] = m["macro_f1"]
            row["_pred"] = pred
            per_trip.append(row)
    return correct / max(1, total), per_trip


def summarise(per_trip, name):
    df = pd.DataFrame(per_trip)
    w = df.n_reliable / df.n_reliable.sum()
    return {"method": name,
            "top1_accuracy": float((df.top1_accuracy * w).sum()),
            "macro_f1_mean_over_trips": float(df.macro_f1.mean()),
            "route_continuity_mean": float(df.route_continuity.mean()),
            "route_precision_mean": float(df.route_precision.mean()),
            "route_recall_mean": float(df.route_recall.mean()),
            "route_f1_mean": float(df.route_f1.mean()),
            "reconstructed_route_continuity_mean": float(df.reconstructed_route_continuity.mean()),
            "reconstructed_route_f1_mean": float(df.reconstructed_route_f1.mean()),
            "chain_breaks_total": int(df.chain_breaks.sum())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (C.RESULTS_DIR / "maps").mkdir(parents=True, exist_ok=True)
    pts_tr = pd.read_parquet(C.PROCESSED_DIR / "train_points.parquet")
    pts_te = pd.read_parquet(C.PROCESSED_DIR / "student_test_points.parquet")
    oof = pd.read_parquet(C.PROCESSED_DIR / "train_oof_predictions.parquet")
    pte = pd.read_parquet(C.PROCESSED_DIR / "student_test_predictions.parquet")

    t0 = time.time()
    train = prepare("train", C.TRAIN_DIR, oof, pts_tr)
    lattice_s = time.time() - t0

    grid = []
    base_acc, _ = score(train)
    for a in ALPHAS:
        for b in BETAS:
            acc, _ = score(train, a, b)
            grid.append({"alpha": a, "beta": b, "train_oof_top1": acc})
            print(f"alpha={a} beta={b}: {acc:.4f}", flush=True)
    grid = pd.DataFrame(grid)
    grid.to_csv(OUT / "decoder_grid_train_oof.csv", index=False)
    grid["alpha_x_beta"] = grid.alpha * grid.beta
    best = grid.sort_values(["train_oof_top1", "alpha"], ascending=[False, False]).iloc[0]
    alpha, beta = float(best.alpha), float(best.beta)
    cfg = {"alpha": alpha, "beta": beta, "selected_on": "TRAIN out-of-fold predictions",
           "train_oof_top1_classifier_only": base_acc, "train_oof_top1_hmm": float(best.train_oof_top1),
           "effective_parameter_alpha_x_beta": float(best.alpha * best.beta),
           "grid": {"alpha": ALPHAS, "beta": BETAS}, "lattice_build_seconds_train": lattice_s}
    (C.ARTIFACT_DIR / "decoder_config.json").write_text(json.dumps(cfg, indent=2))
    print("chosen", cfg)

    piv = grid.pivot(index="alpha", columns="beta", values="train_oof_top1")
    fig, ax = plt.subplots(figsize=(9, 3.6))
    im = ax.imshow(piv.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(piv.columns)), piv.columns); ax.set_yticks(range(len(piv.index)), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.3f}", ha="center", va="center", color="w", fontsize=8)
    ax.set(xlabel="beta (m)", ylabel="alpha (emission weight)",
           title=f"Decoder grid, TRAIN OOF top-1 (classifier only = {base_acc:.3f})")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(OUT / "decoder_grid.png", dpi=130); plt.close(fig)

    # ---- comparisons on TRAIN (OOF) and STUDENT-TEST ----
    test = prepare("student_test", C.STUDENT_TEST_DIR, pte, pts_te)
    rows, per_trip_rows = [], []
    for part, prep in (("train_oof", train), ("student_test", test)):
        _, clf = score(prep, detail=True)
        _, hmm = score(prep, alpha, beta, detail=True)
        rows.append({"partition": part, **summarise(clf, "classifier only")})
        rows.append({"partition": part, **summarise(hmm, "classifier + HMM")})
        for c, h in zip(clf, hmm):
            per_trip_rows.append({"partition": part, "trip_id": c["trip_id"], "n_obs": c["n_obs"],
                                  "clf_top1": c["top1_accuracy"], "hmm_top1": h["top1_accuracy"],
                                  "clf_continuity": c["route_continuity"], "hmm_continuity": h["route_continuity"],
                                  "clf_route_f1": c["route_f1"], "hmm_route_f1": h["route_f1"],
                                  "hmm_reconstructed_continuity": h["reconstructed_route_continuity"],
                                  "hmm_reconstructed_route_f1": h["reconstructed_route_f1"],
                                  "hmm_chain_breaks": h["chain_breaks"]})
        if part == "student_test":
            for c, h in zip(clf, hmm):
                trip = prep[c["trip_id"]]["trip"]
                m = trip_map(trip, {"reference": trip.route, "classifier": c["_pred"],
                                    "hmm": reconstruct_route(h["_pred"], prep[c["trip_id"]]["graph"])},
                             title=f"Student-test trip {trip.trip_id}: HMM top-1 {h['top1_accuracy']:.3f}")
                m.save(str(C.RESULTS_DIR / "maps" / f"hmm_student_test_{trip.trip_id}.html"))
    comp = pd.DataFrame(rows)
    comp.to_csv(OUT / "classifier_vs_hmm.csv", index=False)
    pt = pd.DataFrame(per_trip_rows)
    pt["hmm_gain"] = pt.hmm_top1 - pt.clf_top1
    pt.to_csv(OUT / "classifier_vs_hmm_per_trip.csv", index=False)
    print(comp.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    st = comp[comp.partition == "student_test"]
    for ax, col, title in zip(axes, ["top1_accuracy", "route_continuity_mean", "route_f1_mean"],
                              ["Top-1 accuracy", "Route continuity", "Route F1 (length-weighted)"]):
        ax.bar(st.method, st[col], color=["tab:orange", "tab:red"])
        ax.set(title=f"Student test: {title}", ylim=(max(0, st[col].min() - 0.1), 1.0))
        for i, v in enumerate(st[col]):
            ax.text(i, v + 0.005, f"{v:.3f}", ha="center")
    fig.tight_layout(); fig.savefig(OUT / "classifier_vs_hmm.png", dpi=130); plt.close(fig)

    s = pt[pt.partition == "student_test"]
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(s))
    ax.bar(x - 0.2, s.clf_top1, 0.4, label="classifier only")
    ax.bar(x + 0.2, s.hmm_top1, 0.4, label="classifier + HMM")
    ax.set_xticks(x, s.trip_id, rotation=30)
    ax.set(ylabel="top-1 accuracy", title="Per-trip effect of HMM decoding (student test)",
           ylim=(max(0, min(s.clf_top1.min(), s.hmm_top1.min()) - 0.05), 1))
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "per_trip_hmm_gain.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
