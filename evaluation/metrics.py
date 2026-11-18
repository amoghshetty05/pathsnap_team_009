"""Evaluation metrics for candidate classification and route quality.

Definitions (all computed only on observations whose reference label is reliable, see
configs.config.LABEL_MAX_DIST_M):

Observation-level Top-1 Accuracy
    For each GPS observation, the candidate arc with the highest score is compared with the
    reference arc. Observations whose reference arc is not in the candidate set (candidate miss) or
    that have no candidates count as wrong. Denominator = all reliable observations.

Macro F1 / Precision / Recall (candidate level)
    Every (observation, candidate) row is a binary decision: 1 if the candidate is the one selected
    for that observation (top-1), else 0. Ground truth is 1 for the reference arc. Macro F1 is the
    unweighted mean of the F1 for both classes. Precision/recall are for the positive class.

Log Loss
    Binary cross-entropy of the per-row probability (normalised within each observation's
    candidate set) against the 0/1 row label.

Candidate Miss Rate
    Share of reliable observations whose reference arc is absent from the candidate set.

Route Continuity
    The per-observation predicted arcs are collapsed (consecutive duplicates removed). Continuity
    is the share of consecutive collapsed arc pairs (a, b) where a's end node is b's start node.

Route (sequence-level) Precision / Recall / F1
    Length-weighted overlap between the set of predicted arcs and the reference route arcs.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score

EPS = 1e-6


def normalise_scores(df, col="p"):
    s = df.groupby(["trip_id", "point"])[col].transform("sum").to_numpy()
    return np.where(s > 0, df[col].to_numpy() / np.where(s > 0, s, 1), 0.0)


def select_top1(df, col="p"):
    """Boolean mask of the top-scoring candidate per observation (ties -> nearest, i.e. first)."""
    order = df.sort_values(["trip_id", "point", col, "dist"], ascending=[True, True, False, True])
    first = ~order.duplicated(["trip_id", "point"])
    return first.reindex(df.index).to_numpy()


def candidate_metrics(df, n_reliable_points, col="p"):
    """df: candidate rows of reliable observations with columns trip_id, point, y, dist, <col>."""
    sel = select_top1(df, col)
    y = df["y"].to_numpy()
    correct = int((sel & (y == 1)).sum())
    pn = np.clip(normalise_scores(df, col), EPS, 1 - EPS)
    return {
        "top1_accuracy": correct / max(1, n_reliable_points),
        "macro_f1": f1_score(y, sel.astype(int), average="macro"),
        "precision": precision_score(y, sel.astype(int), zero_division=0),
        "recall": recall_score(y, sel.astype(int), zero_division=0),
        "log_loss": log_loss(y, pn, labels=[0, 1]),
    }


def route_continuity(arc_seq, arcs):
    """arc_seq: per-observation arc ids (None/-1 for gaps). arcs: (A,2) from/to node array."""
    s = [int(a) for a in arc_seq if a is not None and a >= 0]
    c = [a for i, a in enumerate(s) if i == 0 or a != s[i - 1]]
    if len(c) < 2:
        return 1.0
    c = np.asarray(c)
    return float(np.mean(arcs[c[:-1], 1] == arcs[c[1:], 0]))


def route_overlap(pred_arcs, ref_route, seg_len):
    p = set(int(a) for a in pred_arcs if a is not None and a >= 0)
    r = set(int(a) for a in ref_route)
    if not p or not r:
        return {"route_precision": 0.0, "route_recall": 0.0, "route_f1": 0.0}
    inter = sum(seg_len[a] for a in p & r)
    prec = inter / max(EPS, sum(seg_len[a] for a in p))
    rec = inter / max(EPS, sum(seg_len[a] for a in r))
    f1 = 2 * prec * rec / max(EPS, prec + rec)
    return {"route_precision": prec, "route_recall": rec, "route_f1": f1}


def per_observation_top1(df, col="p"):
    """Series indexed by (trip_id, point) -> selected arc id."""
    sel = select_top1(df, col)
    s = df.loc[sel, ["trip_id", "point", "arc"]]
    return s.set_index(["trip_id", "point"])["arc"]


def observation_accuracy(pred, label, ok):
    """pred/label: per-observation arc arrays (-1 = no prediction). ok: reliability mask."""
    pred, label, ok = map(np.asarray, (pred, label, ok))
    return float(np.mean(pred[ok] == label[ok])) if ok.any() else float("nan")
