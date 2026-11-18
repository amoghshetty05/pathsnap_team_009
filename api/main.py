"""PathSnap FastAPI backend (Laptop 1).

Run (from the repo root):
    PATHSNAP_EVAL_DIR=data/raw/evaluator_test uvicorn api.main:app --host 0.0.0.0 --port 8000

Endpoints
    GET  /health              status, loaded model and decoder parameters
    GET  /trips               trip ids available in PATHSNAP_EVAL_DIR (the evaluation trips)
    POST /predict {trip_id}   full pipeline for one trip + server-side scoring

The prediction pipeline loads only .track/.nodes/.arcs. The .route file (reference) is read only
afterwards, inside evaluate_against_reference, to compute accuracy. The reference route is never
returned to the UI: the response contains GPS points, the predicted route and the metrics.
"""
import logging
import os
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from configs import config as C
from data_processing.io import list_trips
from hmm_decoder.pipeline import PathSnapPipeline, evaluate_against_reference
from visualisations.folium_maps import arcs_to_lines

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap.api")

_default = C.DATA_DIR / "evaluator_test"
EVAL_DIR = Path(os.environ.get("PATHSNAP_EVAL_DIR", _default if _default.exists() else C.STUDENT_TEST_DIR))
if not EVAL_DIR.is_absolute():
    EVAL_DIR = C.ROOT / EVAL_DIR

app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

PIPELINE = PathSnapPipeline()
log.info("Model loaded: %s | decoder alpha=%.3f beta=%.3f | evaluation dir: %s",
         "LightGBM (tuned)" if hasattr(PIPELINE.model, "booster") else type(PIPELINE.model).__name__, PIPELINE.alpha, PIPELINE.beta, EVAL_DIR)


class PredictRequest(BaseModel):
    trip_id: str


def _available():
    return list_trips(EVAL_DIR) if EVAL_DIR.exists() else []


@app.get("/health")
def health():
    return {"status": "ok", "model": "LightGBM (tuned)" if hasattr(PIPELINE.model, "booster") else type(PIPELINE.model).__name__, "alpha": PIPELINE.alpha,
            "beta": PIPELINE.beta, "evaluation_dir": str(EVAL_DIR), "n_trips": len(_available())}


@app.get("/trips")
def trips(request: Request):
    t = _available()
    log.info("GET /trips from %s -> %d trips", request.client.host if request.client else "?", len(t))
    return {"trips": t}


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    client = request.client.host if request.client else "?"
    if req.trip_id not in _available():
        log.warning("POST /predict from %s: unknown trip_id=%r", client, req.trip_id)
        raise HTTPException(404, f"unknown trip_id {req.trip_id!r}")
    log.info("POST /predict from %s: trip_id=%s - running pipeline", client, req.trip_id)
    t0 = time.time()
    pred = PIPELINE.predict_trip(EVAL_DIR, req.trip_id)
    pred_s = time.time() - t0
    metrics = evaluate_against_reference(EVAL_DIR, req.trip_id, pred)
    trip = pred["trip"]

    route_arcs = np.asarray(pred["route"], dtype=np.int64)
    step = max(1, len(trip.track) // 3000)  # keep the payload light for very long trips
    resp = {
        "trip_id": req.trip_id,
        "n_observations": int(len(trip.track)),
        "gps": [[float(lat), float(lon)] for lon, lat in trip.track[::step, :2]],
        "predicted_route": arcs_to_lines(trip, route_arcs),
        "predicted_arc_ids": [int(a) for a in pred["hmm_arcs"]],
        "metrics": {
            "top1_accuracy": metrics["hmm_top1_accuracy"],
            "classifier_only_top1_accuracy": metrics["classifier_top1_accuracy"],
            "route_continuity": metrics["hmm_route_continuity"],
            "classifier_only_route_continuity": metrics["classifier_route_continuity"],
            "reconstructed_route_continuity": metrics["reconstructed_route_continuity"],
            "route_precision": metrics["reconstructed_route_precision"],
            "route_recall": metrics["reconstructed_route_recall"],
            "route_f1": metrics["reconstructed_route_f1"],
            "n_reliable_observations": metrics["n_reliable_observations"],
            "observations_without_candidates": pred["n_no_candidates"],
            "hmm_chain_breaks": pred["breaks"],
        },
        "timings_s": {**{k: round(v, 3) for k, v in pred["timings"].items()},
                      "pipeline_total": round(pred_s, 3), "request_total": round(time.time() - t0, 3)},
    }
    log.info("trip_id=%s done: obs=%d top1=%.4f (classifier-only %.4f) continuity=%.4f "
             "reconstructed_continuity=%.4f route_f1=%.4f breaks=%d time=%.2fs", req.trip_id,
             resp["n_observations"], resp["metrics"]["top1_accuracy"],
             resp["metrics"]["classifier_only_top1_accuracy"], resp["metrics"]["route_continuity"],
             resp["metrics"]["reconstructed_route_continuity"],
             resp["metrics"]["route_f1"], pred["breaks"], pred_s)
    return resp
