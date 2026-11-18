"""Candidate road-segment generation.

Every arc is densified into sample points every DENSIFY_STEP_M metres and indexed in a KD-tree.
For a GPS point we query samples within (radius + step/2), take the unique arcs, compute the exact
point-to-segment distance and keep the MAX_CANDIDATES nearest arcs within the radius. If nothing is
within the radius we retry with the fallback radius; if still nothing, the observation has no
candidates (treated as a gap by the decoder).

Both directions of a two-way street are separate directed arcs with identical geometry; heading
features let the classifier tell them apart.
"""
import numpy as np
from scipy.spatial import cKDTree

from configs import config as C
from data_processing.geo import point_segment


class CandidateIndex:
    def __init__(self, trip, step: float = C.DENSIFY_STEP_M):
        self.trip = trip
        self.step = step
        n = np.maximum(1, np.ceil(trip.seg_len / step).astype(np.int64))
        arc_ids = np.repeat(np.arange(len(n)), n + 1)
        offs = np.concatenate([np.arange(k + 1) / k for k in n])
        a = trip.seg_a[arc_ids]
        b = trip.seg_b[arc_ids]
        pts = a + offs[:, None] * (b - a)
        self.sample_arc = arc_ids
        self.tree = cKDTree(pts)

    def query(self, xy: np.ndarray, radius: float = C.CANDIDATE_RADIUS_M,
              fallback: float = C.CANDIDATE_FALLBACK_RADIUS_M, k: int = C.MAX_CANDIDATES):
        """Returns list (per point) of (arc_ids, dist, frac) arrays sorted by distance."""
        trip = self.trip
        out = []
        hits = self.tree.query_ball_point(xy, r=radius + self.step / 2)
        for i, h in enumerate(hits):
            res = self._finish(xy[i], h, radius, k)
            if res is None and fallback and fallback > radius:
                h2 = self.tree.query_ball_point(xy[i], r=fallback + self.step / 2)
                res = self._finish(xy[i], h2, fallback, k)
            if res is None:
                res = (np.empty(0, np.int64), np.empty(0), np.empty(0))
            out.append(res)
        return out

    def _finish(self, p, hits, radius, k):
        if not hits:
            return None
        arcs = np.unique(self.sample_arc[hits])
        d, f = point_segment(p[None, :], self.trip.seg_a[arcs], self.trip.seg_b[arcs])
        keep = d <= radius
        if not keep.any():
            return None
        arcs, d, f = arcs[keep], d[keep], f[keep]
        order = np.argsort(d, kind="stable")[:k]
        return arcs[order], d[order], f[order]
