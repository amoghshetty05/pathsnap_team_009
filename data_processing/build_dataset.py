"""Build the candidate/feature tables for the TRAIN and STUDENT-TEST partitions.

Run:  python -m data_processing.build_dataset

Outputs (data/processed/):
    <partition>_candidates.parquet   one row per (observation, candidate) with features and label
    <partition>_points.parquet       one row per observation: label arc, label distance, reliability,
                                      whether the reference arc is in the candidate set
Also writes results/eda/candidate_radius_sweep.csv (TRAIN only) used to choose the radius.

The evaluator-testing partition is never read here.
"""
import time

import numpy as np
import pandas as pd

from configs import config as C
from data_processing.candidates import CandidateIndex
from data_processing.io import list_trips, load_trip
from data_processing.labelling import align_track_to_route
from features.features import build_features


def process_partition(name, pdir, radius_sweep=None):
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    rows, points, sweep = [], [], []
    for tid in list_trips(pdir):
        t0 = time.time()
        trip = load_trip(pdir, tid)
        label, _, label_dist = align_track_to_route(trip)
        ok = label_dist <= C.LABEL_MAX_DIST_M
        index = CandidateIndex(trip)
        cands = index.query(trip.xy)
        in_set = np.array([label[i] in cands[i][0] for i in range(len(cands))])
        ncand = np.array([len(c[0]) for c in cands])
        points.append(pd.DataFrame({
            "trip_id": tid, "point": np.arange(len(label)), "t": trip.times,
            "label_arc": label, "label_dist": label_dist, "label_ok": ok,
            "label_in_cands": in_set, "n_cands": ncand,
        }))
        rows.append(build_features(trip, cands, label, ok))
        if radius_sweep:
            for r in radius_sweep:
                c_r = index.query(trip.xy[ok], radius=r, fallback=0)
                lab_ok = label[ok]
                miss = np.mean([lab_ok[i] not in c_r[i][0] for i in range(len(c_r))])
                k = np.mean([len(c[0]) for c in c_r])
                sweep.append({"trip_id": tid, "radius_m": r, "miss_rate": miss, "mean_cands": k})
        print(f"[{name}] {tid}: {len(trip.xy)} pts, {len(rows[-1])} cand rows, "
              f"miss={1 - in_set[ok].mean():.4f}, {time.time() - t0:.1f}s", flush=True)

    cand = pd.concat(rows, ignore_index=True)
    pts = pd.concat(points, ignore_index=True)
    cand.to_parquet(C.PROCESSED_DIR / f"{name}_candidates.parquet", index=False)
    pts.to_parquet(C.PROCESSED_DIR / f"{name}_points.parquet", index=False)
    if sweep:
        out = C.RESULTS_DIR / "eda"
        out.mkdir(parents=True, exist_ok=True)
        s = pd.DataFrame(sweep)
        s.to_csv(out / "candidate_radius_sweep_per_trip.csv", index=False)
        s.groupby("radius_m")[["miss_rate", "mean_cands"]].mean().to_csv(out / "candidate_radius_sweep.csv")
    return cand, pts


def summarise(name, pts):
    ok = pts[pts.label_ok]
    return {
        "partition": name,
        "trips": pts.trip_id.nunique(),
        "observations": len(pts),
        "unreliable_label_obs": int((~pts.label_ok).sum()),
        "unreliable_label_pct": round(100 * (~pts.label_ok).mean(), 3),
        "no_candidate_obs": int((pts.n_cands == 0).sum()),
        "candidate_miss_rate": round(1 - ok.label_in_cands.mean(), 5),
        "mean_candidates": round(pts.n_cands.mean(), 2),
    }


if __name__ == "__main__":
    np.random.seed(C.SEED)
    _, ptr = process_partition("train", C.TRAIN_DIR, radius_sweep=[20, 30, 40, 50, 75, 100])
    _, pte = process_partition("student_test", C.STUDENT_TEST_DIR)
    summ = pd.DataFrame([summarise("train", ptr), summarise("student_test", pte)])
    summ.to_csv(C.RESULTS_DIR / "eda" / "candidate_summary.csv", index=False)
    print(summ.to_string(index=False))
