"""End-to-end inference pipeline used by the decoder experiments and the API.

    raw trip folder -> candidates -> features -> classifier probabilities -> HMM/Viterbi -> route

The pipeline never reads the .route file. Evaluation against the reference route is a separate
step (evaluate_against_reference) that the API performs server-side after prediction.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np

from configs import config as C
from data_processing.candidates import CandidateIndex
from data_processing.io import load_trip
from data_processing.labelling import align_track_to_route
from evaluation.metrics import route_continuity, route_overlap
from features.features import FEATURES, build_features
from hmm_decoder.viterbi import RoadGraph, build_lattice, decode_full, reconstruct_route

DECODER_CONFIG = C.ARTIFACT_DIR / "decoder_config.json"
FINAL_MODEL = C.ARTIFACT_DIR / "final_model_lgb.txt"   # native LightGBM model (version-robust)


def rows_to_lists(df, n_obs, col="p"):
    """Candidate table of one trip -> per-observation lists (arcs, frac, prob), sorted by distance."""
    arcs = [np.empty(0, np.int64)] * n_obs
    frac = [np.empty(0)] * n_obs
    prob = [np.empty(0)] * n_obs
    df = df.sort_values(["point", "dist"])
    for pt, g in df.groupby("point", sort=False):
        arcs[pt] = g.arc.to_numpy(np.int64)
        frac[pt] = g.frac.to_numpy(float)
        prob[pt] = g[col].to_numpy(float)
    return arcs, frac, prob


class _BoosterModel:
    """Wraps a native LightGBM Booster with a predict_proba interface."""
    def __init__(self, path):
        import lightgbm as lgb
        self.booster = lgb.Booster(model_file=str(path))

    def predict_proba(self, X):
        p = self.booster.predict(X)
        return np.column_stack([1 - p, p])


def load_model(path: Path):
    path = Path(path)
    return _BoosterModel(path) if path.suffix == ".txt" else joblib.load(path)


class PathSnapPipeline:
    def __init__(self, model_path: Path = FINAL_MODEL, decoder_config: Path = DECODER_CONFIG):
        self.model = load_model(model_path)
        cfg = json.loads(Path(decoder_config).read_text()) if Path(decoder_config).exists() else {}
        self.alpha = float(cfg.get("alpha", 1.0))
        self.beta = float(cfg.get("beta", 10.0))

    def predict_trip(self, partition_dir: Path, trip_id: str):
        timings = {}
        t0 = time.time()
        trip = load_trip(partition_dir, trip_id, with_route=False)   # ground truth is not loaded here
        timings["load_s"] = time.time() - t0

        t0 = time.time()
        cands = CandidateIndex(trip).query(trip.xy)
        feats = build_features(trip, cands)
        timings["candidates_features_s"] = time.time() - t0

        t0 = time.time()
        feats["p"] = self.model.predict_proba(feats[FEATURES].to_numpy())[:, 1] if len(feats) else []
        timings["classifier_s"] = time.time() - t0

        n = len(trip.xy)
        arcs, frac, prob = rows_to_lists(feats, n)
        clf = np.full(n, -1, np.int64)
        for i in range(n):
            if len(arcs[i]):
                clf[i] = arcs[i][int(np.argmax(prob[i]))]

        t0 = time.time()
        graph = RoadGraph(trip)
        lat = build_lattice(trip, arcs, frac, prob, graph)
        hmm, breaks = decode_full(lat, n, self.alpha, self.beta)
        route = reconstruct_route(hmm, graph)
        timings["hmm_s"] = time.time() - t0
        return {"trip": trip, "classifier_arcs": clf, "hmm_arcs": hmm, "route": route, "breaks": breaks,
                "n_no_candidates": int(sum(len(a) == 0 for a in arcs)), "timings": timings}


def evaluate_against_reference(partition_dir: Path, trip_id: str, pred: dict):
    """Server-side scoring of a prediction against the trip's reference route."""
    ref = load_trip(partition_dir, trip_id, with_route=True)
    label, _, label_dist = align_track_to_route(ref)
    ok = label_dist <= C.LABEL_MAX_DIST_M
    out = {"n_observations": int(len(label)), "n_reliable_observations": int(ok.sum())}
    for key in ("classifier_arcs", "hmm_arcs"):
        p = pred[key]
        name = key.replace("_arcs", "")
        out[f"{name}_top1_accuracy"] = float(np.mean(p[ok] == label[ok])) if ok.any() else None
        out[f"{name}_route_continuity"] = route_continuity(p, ref.arcs)
        ov = route_overlap(p, ref.route, ref.seg_len)
        out.update({f"{name}_{k}": float(v) for k, v in ov.items()})
    # the gap-filled, drivable route built from the HMM output
    out["reconstructed_route_continuity"] = route_continuity(pred["route"], ref.arcs)
    out.update({f"reconstructed_{k}": float(v) for k, v in
                route_overlap(pred["route"], ref.route, ref.seg_len).items()})
    return out
