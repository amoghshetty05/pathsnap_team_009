"""Feature engineering for (GPS observation, candidate arc) pairs.

The road network is different for every trip (trips are in different countries), so arc ids carry
no meaning across trips. All features are therefore relative/geometric and transfer between trips.

Point-level kinematics
    speed            displacement / dt between neighbouring observations (m/s)
    dt_prev          time since previous observation (s), large values = GPS gap
    heading_valid    1 if the vehicle moved enough for a reliable GPS heading

Candidate-level geometry
    dist             perpendicular distance from the GPS point to the arc (m)
    dist_rank        rank of the candidate by distance within the point's candidate set
    dist_gap         dist - min dist in the candidate set
    n_cands          size of the candidate set
    heading_diff     |GPS heading - arc bearing| in [0, 180] using a +-1 point baseline
    heading_diff_w   same with a +-CONTEXT_WINDOW baseline (robust at low speed / noise)
    arc_len          arc length (m)
    frac             position of the projected foot along the arc (0 = start, 1 = end)
    dist_prev/next   distance from observation t-1 / t+1 to the same arc
    dist_win_mean    mean distance from observations t-W..t+W to the same arc
    from_deg/to_deg  number of arcs leaving the arc's end node / entering its start node
"""
import numpy as np
import pandas as pd

from configs import config as C
from data_processing.geo import angle_diff, point_segment, segment_bearing

FEATURES = [
    "dist", "dist_rank", "dist_gap", "n_cands",
    "heading_diff", "heading_diff_w", "heading_valid", "speed", "dt_prev",
    "arc_len", "frac", "dist_prev", "dist_next", "dist_win_mean",
    "to_deg", "from_deg",
]


def _track_headings(xy, w):
    T = len(xy)
    lo = np.clip(np.arange(T) - w, 0, T - 1)
    hi = np.clip(np.arange(T) + w, 0, T - 1)
    disp = np.linalg.norm(xy[hi] - xy[lo], axis=1)
    hdg = segment_bearing(xy[lo], xy[hi])
    return hdg, disp


def point_kinematics(trip):
    xy, t = trip.xy, trip.times
    T = len(xy)
    h1, disp1 = _track_headings(xy, 1)
    hw, dispw = _track_headings(xy, C.CONTEXT_WINDOW)
    dt_prev = np.r_[0.0, np.diff(t)]
    step = np.r_[0.0, np.linalg.norm(np.diff(xy, axis=0), axis=1)]
    speed = np.where(dt_prev > 0, step / np.where(dt_prev > 0, dt_prev, 1), 0.0)
    if T > 1:
        speed[0] = speed[1]
    valid = (disp1 > 2.0).astype(float)
    # carry the last reliable heading forward through stops
    h1 = h1.copy()
    last = h1[0]
    for i in range(T):
        if valid[i]:
            last = h1[i]
        else:
            h1[i] = last
    return dict(h1=h1, hw=hw, valid=valid, dispw=dispw, speed=speed, dt_prev=dt_prev)


def build_features(trip, cands, labels=None, label_ok=None, point_idx=None):
    """Return a DataFrame with one row per (observation, candidate).

    cands: output of CandidateIndex.query for all observations of the trip.
    labels / label_ok: optional per-observation reference arc and reliability flag.
    point_idx: optional subset of observation indices to emit rows for.
    """
    T = len(trip.xy)
    kin = point_kinematics(trip)
    out_deg = np.bincount(trip.arcs[:, 0], minlength=len(trip.nodes))
    in_deg = np.bincount(trip.arcs[:, 1], minlength=len(trip.nodes))
    if point_idx is None:
        point_idx = np.arange(T)

    pts, arcs, dist, frac, rank, ncand = [], [], [], [], [], []
    for i in point_idx:
        a, d, f = cands[i]
        if len(a) == 0:
            continue
        pts.append(np.full(len(a), i))
        arcs.append(a); dist.append(d); frac.append(f)
        rank.append(np.arange(len(a))); ncand.append(np.full(len(a), len(a)))
    if not pts:
        return pd.DataFrame(columns=["trip_id", "point", "arc"] + FEATURES)
    pts = np.concatenate(pts); arcs = np.concatenate(arcs)
    dist = np.concatenate(dist); frac = np.concatenate(frac)
    rank = np.concatenate(rank); ncand = np.concatenate(ncand)

    min_d = pd.Series(dist).groupby(pts).transform("min").to_numpy()
    sa, sb = trip.seg_a[arcs], trip.seg_b[arcs]

    def dist_at(offset):
        j = np.clip(pts + offset, 0, T - 1)
        d, _ = point_segment(trip.xy[j], sa, sb)
        return d

    W = C.CONTEXT_WINDOW
    win = np.mean([dist_at(o) for o in range(-W, W + 1)], axis=0)

    df = pd.DataFrame({
        "trip_id": trip.trip_id,
        "point": pts,
        "arc": arcs,
        "dist": dist,
        "dist_rank": rank,
        "dist_gap": dist - min_d,
        "n_cands": ncand,
        "heading_diff": np.where(kin["valid"][pts] > 0,
                                 angle_diff(kin["h1"][pts], trip.seg_bearing[arcs]), 90.0),
        "heading_diff_w": np.where(kin["dispw"][pts] > 5.0,
                                   angle_diff(kin["hw"][pts], trip.seg_bearing[arcs]), 90.0),
        "heading_valid": kin["valid"][pts],
        "speed": kin["speed"][pts],
        "dt_prev": kin["dt_prev"][pts],
        "arc_len": trip.seg_len[arcs],
        "frac": frac,
        "dist_prev": dist_at(-1),
        "dist_next": dist_at(1),
        "dist_win_mean": win,
        "to_deg": out_deg[trip.arcs[arcs, 1]],
        "from_deg": in_deg[trip.arcs[arcs, 0]],
    })
    if labels is not None:
        df["label_arc"] = labels[pts]
        df["y"] = (arcs == labels[pts]).astype(np.int8)
        df["label_ok"] = label_ok[pts] if label_ok is not None else True
    return df
