"""Derive a reference arc for every GPS observation from the trip-level .route file.

The .route file lists the arcs travelled in order (e.g. 149 arcs for 1240 GPS points), so it does
not label observations directly. We align observations to route positions with a monotone dynamic
programme: observation t is assigned route index j(t) with j(t) non-decreasing in t, minimising
sum_t dist(gps_t, route_arc_{j(t)}). Monotonicity stops points from being matched to a later or
earlier pass along the same street, which nearest-route-arc snapping would get wrong on loops.

    C[t, j] = d[t, j] + min_{k <= j} C[t-1, k]

Skipping route arcs is allowed (short arcs can be crossed between two 1 Hz samples).
"""
import numpy as np

from data_processing.geo import point_segment


def align_track_to_route(trip, chunk: int = 2048):
    route = trip.route
    if route is None or len(route) == 0:
        raise ValueError(f"trip {trip.trip_id} has no reference route")
    a = trip.seg_a[route]
    b = trip.seg_b[route]
    T, R = len(trip.xy), len(route)
    bp_dtype = np.int16 if R < 32_000 else np.int32
    back = np.empty((T, R), dtype=bp_dtype)

    prev = None
    for start in range(0, T, chunk):
        stop = min(T, start + chunk)
        d, _ = point_segment(trip.xy[start:stop, None, :], a[None], b[None])
        for i in range(stop - start):
            t = start + i
            if prev is None:
                cur = d[i].copy()
                back[t] = np.arange(R)
            else:
                # prefix minimum of the previous cost row and where it was attained
                acc = np.minimum.accumulate(prev)
                changed = np.r_[True, acc[1:] < acc[:-1]]
                run_idx = np.maximum.accumulate(np.where(changed, np.arange(R), 0))
                cur = d[i] + acc
                back[t] = run_idx
            prev = cur

    idx = np.empty(T, dtype=np.int64)
    idx[-1] = int(np.argmin(prev))
    for t in range(T - 1, 0, -1):
        idx[t - 1] = back[t, idx[t]]

    label_arc = route[idx]
    label_dist, _ = point_segment(trip.xy, trip.seg_a[label_arc], trip.seg_b[label_arc])
    return label_arc, idx, label_dist
