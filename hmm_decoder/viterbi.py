"""HMM map-matching with Viterbi decoding.

Hidden states
    For observation t, the states are its candidate arcs (from data_processing.candidates). The
    candidate -> arc correspondence is kept explicitly (Lattice.arcs[t][k] is the arc of state k).

Emission
    log e_t(k) = alpha * log p_t(k), where p_t(k) is the classifier probability normalised over the
    candidate set of observation t. alpha controls how much the decoder trusts the classifier.

Transition (Newson & Krumm style, using road-network connectivity)
    For consecutive observations i -> j with candidates a (offset s_a along a) and b (offset s_b):
        same arc:     d_route = |s_b - s_a|
        different:    d_route = (len_a - s_a) + shortest_path(end(a) -> start(b)) + s_b
    log T(a -> b) = -|d_route - d_gc| / beta, where d_gc is the straight-line distance between the
    two GPS observations. If b is not reachable from a within the search cutoff the transition is
    impossible (-inf).

Gaps / missing candidates
    Observations with no candidates get no state; the chain links the previous and next observations
    that have candidates and the cutoff grows with the elapsed distance (bounded by MAX_CUTOFF_M).
    If every transition into observation j is impossible (disconnected network, very long gap, bad
    candidates), the chain is broken: Viterbi restarts at j from the emissions alone and each part
    is back-tracked separately (HMM break handling).
"""
import heapq
from dataclasses import dataclass, field

import numpy as np

MIN_CUTOFF_M = 250.0
MAX_CUTOFF_M = 6000.0
NEG_INF = -np.inf


class RoadGraph:
    """Directed road graph with cached, distance-bounded Dijkstra from node sources."""

    def __init__(self, trip):
        self.arcs = trip.arcs
        self.seg_len = trip.seg_len
        n = len(trip.nodes)
        order = np.argsort(trip.arcs[:, 0], kind="stable")
        self.out_ptr = np.searchsorted(trip.arcs[order, 0], np.arange(n + 1))
        self.out_arcs = order
        self._cache = {}

    def dist_from(self, src: int, cutoff: float) -> dict:
        hit = self._cache.get(src)
        if hit is not None and hit[0] >= cutoff:
            return hit[1]
        dist = {src: 0.0}
        heap = [(0.0, src)]
        arcs, L, ptr, out = self.arcs, self.seg_len, self.out_ptr, self.out_arcs
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, np.inf):
                continue
            for e in out[ptr[u]:ptr[u + 1]]:
                v = int(arcs[e, 1])
                nd = d + L[e]
                if nd <= cutoff and nd < dist.get(v, np.inf):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        self._cache[src] = (cutoff, dist)
        return dist

    def path_arcs(self, src: int, dst: int, cutoff: float):
        """Arc ids of the shortest directed path src -> dst (None if not found within cutoff)."""
        if src == dst:
            return []
        dist, pred = {src: 0.0}, {}
        heap = [(0.0, src)]
        arcs, L, ptr, out = self.arcs, self.seg_len, self.out_ptr, self.out_arcs
        while heap:
            d, u = heapq.heappop(heap)
            if u == dst:
                break
            if d > dist.get(u, np.inf):
                continue
            for e in out[ptr[u]:ptr[u + 1]]:
                v = int(arcs[e, 1])
                nd = d + L[e]
                if nd <= cutoff and nd < dist.get(v, np.inf):
                    dist[v] = nd
                    pred[v] = int(e)
                    heapq.heappush(heap, (nd, v))
        if dst not in pred:
            return None
        path, v = [], dst
        while v != src:
            e = pred[v]
            path.append(e)
            v = int(arcs[e, 0])
        return path[::-1]


def reconstruct_route(arc_seq, graph: RoadGraph, cutoff: float = 2000.0):
    """Collapse the per-observation arcs and fill gaps between consecutive arcs with the shortest
    connecting path, giving a drivable arc sequence. Unbridgeable gaps are left as-is."""
    seq = [int(a) for a in arc_seq if a is not None and a >= 0]
    seq = [a for i, a in enumerate(seq) if i == 0 or a != seq[i - 1]]
    if not seq:
        return []
    out = [seq[0]]
    for b in seq[1:]:
        a = out[-1]
        if graph.arcs[a, 1] != graph.arcs[b, 0]:
            mid = graph.path_arcs(int(graph.arcs[a, 1]), int(graph.arcs[b, 0]), cutoff)
            if mid:
                out.extend(mid)
        out.append(b)
    return [a for i, a in enumerate(out) if i == 0 or a != out[i - 1]]


@dataclass
class Lattice:
    obs: list                    # observation indices that have candidates
    arcs: list                   # per obs: arc ids (states)
    logp: list                   # per obs: log normalised classifier probability
    route_dist: list = field(default_factory=list)  # per step j>0: (K_prev, K_j) route distances (inf = unreachable)
    gc_dist: list = field(default_factory=list)     # per step j>0: straight-line distance


def build_lattice(trip, cand_arcs, cand_frac, probs, graph=None, eps=1e-9):
    """cand_arcs/cand_frac/probs: per-observation arrays (empty for no-candidate observations)."""
    graph = graph or RoadGraph(trip)
    obs = [i for i in range(len(cand_arcs)) if len(cand_arcs[i]) > 0]
    lat = Lattice(obs=obs, arcs=[], logp=[])
    for i in obs:
        p = np.asarray(probs[i], float)
        p = p / p.sum() if p.sum() > 0 else np.full(len(p), 1 / len(p))
        lat.arcs.append(np.asarray(cand_arcs[i], np.int64))
        lat.logp.append(np.log(p + eps))
    L = trip.seg_len
    for j in range(1, len(obs)):
        i0, i1 = obs[j - 1], obs[j]
        a, b = lat.arcs[j - 1], lat.arcs[j]
        sa = np.asarray(cand_frac[i0]) * L[a]
        sb = np.asarray(cand_frac[i1]) * L[b]
        gc = float(np.linalg.norm(trip.xy[i1] - trip.xy[i0]))
        cutoff = min(MAX_CUTOFF_M, max(MIN_CUTOFF_M, 3.0 * gc + 100.0))
        D = np.full((len(a), len(b)), np.inf)
        start_b = trip.arcs[b, 0]
        for ka, arc_a in enumerate(a):
            same = b == arc_a
            if same.any():
                D[ka, same] = np.abs(sb[same] - sa[ka])
            dmap = graph.dist_from(int(trip.arcs[arc_a, 1]), cutoff)
            tail = L[arc_a] - sa[ka]
            for kb in np.where(~same)[0]:
                sp = dmap.get(int(start_b[kb]))
                if sp is not None:
                    D[ka, kb] = tail + sp + sb[kb]
        lat.route_dist.append(D)
        lat.gc_dist.append(gc)
    return lat


def viterbi(lat: Lattice, alpha: float = 1.0, beta: float = 10.0):
    """Returns (chosen arc per lattice observation, number of chain breaks)."""
    n = len(lat.obs)
    if n == 0:
        return np.empty(0, np.int64), 0
    deltas = [alpha * lat.logp[0]]
    backs = [None]
    breaks = 0
    for j in range(1, n):
        D = lat.route_dist[j - 1]
        trans = np.where(np.isfinite(D), -np.abs(D - lat.gc_dist[j - 1]) / beta, NEG_INF)
        score = deltas[-1][:, None] + trans
        best_prev = np.argmax(score, axis=0)
        best = score[best_prev, np.arange(score.shape[1])]
        if not np.isfinite(best).any():
            breaks += 1
            deltas.append(alpha * lat.logp[j])
            backs.append(None)  # chain break: start a new segment here
        else:
            deltas.append(best + alpha * lat.logp[j])
            backs.append(best_prev)
    states = np.empty(n, np.int64)
    k = int(np.argmax(deltas[-1]))
    for j in range(n - 1, -1, -1):
        states[j] = k
        if j == 0:
            break
        if backs[j] is None:
            k = int(np.argmax(deltas[j - 1]))
        else:
            k = int(backs[j][k])
            if not np.isfinite(deltas[j - 1][k]):
                k = int(np.argmax(deltas[j - 1]))
    chosen = np.array([lat.arcs[j][states[j]] for j in range(n)], dtype=np.int64)
    return chosen, breaks


def decode_full(lat: Lattice, n_obs: int, alpha: float, beta: float):
    """Per-observation arc ids for the whole track (-1 where there were no candidates)."""
    chosen, breaks = viterbi(lat, alpha, beta)
    out = np.full(n_obs, -1, dtype=np.int64)
    out[lat.obs] = chosen
    return out, breaks
