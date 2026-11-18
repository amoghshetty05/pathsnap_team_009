"""Reading raw PathSnap trip folders.

Each trip folder <id>/ contains:
    <id>.track  lon lat t          (GPS observations, ~1 Hz)
    <id>.nodes  lon lat            (row index = node id)
    <id>.arcs   from_node to_node  (row index = arc id, directed)
    <id>.route  arc_id             (reference route, chronological; optional)
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from data_processing.geo import LocalProjection, segment_bearing


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray            # (T, 3) lon, lat, t
    nodes: np.ndarray            # (N, 2) lon, lat
    arcs: np.ndarray             # (A, 2) from, to
    route: Optional[np.ndarray]  # (R,) arc ids or None
    proj: LocalProjection = field(repr=False)
    # projected geometry (metres)
    xy: np.ndarray = field(repr=False, default=None)        # (T, 2)
    node_xy: np.ndarray = field(repr=False, default=None)   # (N, 2)
    seg_a: np.ndarray = field(repr=False, default=None)     # (A, 2)
    seg_b: np.ndarray = field(repr=False, default=None)     # (A, 2)
    seg_len: np.ndarray = field(repr=False, default=None)   # (A,)
    seg_bearing: np.ndarray = field(repr=False, default=None)  # (A,) degrees

    @property
    def times(self) -> np.ndarray:
        return self.track[:, 2]


def _read(path: Path, dtype=float, ncols=None) -> np.ndarray:
    arr = np.loadtxt(path, dtype=dtype, ndmin=2)
    if ncols is not None and arr.shape[1] != ncols:
        raise ValueError(f"{path}: expected {ncols} columns, got {arr.shape[1]}")
    return arr


def list_trips(partition_dir: Path) -> list:
    return sorted(p.name for p in Path(partition_dir).iterdir() if p.is_dir())


def load_trip(partition_dir: Path, trip_id: str, with_route: bool = True) -> Trip:
    d = Path(partition_dir) / trip_id
    track = _read(d / f"{trip_id}.track", float, 3)
    nodes = _read(d / f"{trip_id}.nodes", float, 2)
    arcs = _read(d / f"{trip_id}.arcs", np.int64, 2)
    route = None
    rpath = d / f"{trip_id}.route"
    if with_route and rpath.exists():
        route = np.loadtxt(rpath, dtype=np.int64, ndmin=1)

    # Drop rows with non-finite values and enforce time ordering.
    ok = np.isfinite(track).all(axis=1)
    track = track[ok]
    track = track[np.argsort(track[:, 2], kind="stable")]

    proj = LocalProjection.from_lonlat(track[:, 0], track[:, 1])
    trip = Trip(trip_id, track, nodes, arcs, route, proj)
    trip.xy = proj.forward(track[:, 0], track[:, 1])
    trip.node_xy = proj.forward(nodes[:, 0], nodes[:, 1])
    trip.seg_a = trip.node_xy[arcs[:, 0]]
    trip.seg_b = trip.node_xy[arcs[:, 1]]
    trip.seg_len = np.linalg.norm(trip.seg_b - trip.seg_a, axis=1)
    trip.seg_bearing = segment_bearing(trip.seg_a, trip.seg_b)
    return trip
