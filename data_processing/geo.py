"""Geometry helpers.

Longitude/latitude are angular, so distances in degrees are meaningless. Each trip covers at most
a few tens of kilometres, so we use a local equirectangular projection centred on the trip's GPS
centroid: x = R*cos(lat0)*dlon, y = R*dlat (radians). Over <50 km the distortion versus the
haversine distance is well below GPS noise (checked in data_processing/eda.py).
"""
from dataclasses import dataclass

import numpy as np

EARTH_R = 6_371_008.8  # mean Earth radius (m)


@dataclass
class LocalProjection:
    lon0: float
    lat0: float

    @classmethod
    def from_lonlat(cls, lon, lat):
        return cls(float(np.mean(lon)), float(np.mean(lat)))

    def forward(self, lon, lat) -> np.ndarray:
        lon = np.asarray(lon, float)
        lat = np.asarray(lat, float)
        x = np.radians(lon - self.lon0) * EARTH_R * np.cos(np.radians(self.lat0))
        y = np.radians(lat - self.lat0) * EARTH_R
        return np.column_stack([x, y])

    def inverse(self, xy) -> np.ndarray:
        xy = np.asarray(xy, float)
        lat = self.lat0 + np.degrees(xy[:, 1] / EARTH_R)
        lon = self.lon0 + np.degrees(xy[:, 0] / (EARTH_R * np.cos(np.radians(self.lat0))))
        return np.column_stack([lon, lat])


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def segment_bearing(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compass bearing in degrees [0, 360) of vector a->b in projected (x east, y north) metres."""
    d = b - a
    return np.degrees(np.arctan2(d[..., 0], d[..., 1])) % 360.0


def angle_diff(a, b):
    """Smallest absolute difference between two bearings, in [0, 180]."""
    d = np.abs((np.asarray(a) - np.asarray(b)) % 360.0)
    return np.minimum(d, 360.0 - d)


def point_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Distance from points p to segments a-b (broadcasting) and the clamped projection parameter.

    Returns (dist, frac) where frac in [0, 1] is the position of the foot point along a->b.
    """
    ab = b - a
    ap = p - a
    denom = np.einsum("...i,...i->...", ab, ab)
    frac = np.where(denom > 0, np.einsum("...i,...i->...", ap, ab) / np.where(denom > 0, denom, 1), 0.0)
    frac = np.clip(frac, 0.0, 1.0)
    foot = a + frac[..., None] * ab
    dist = np.linalg.norm(p - foot, axis=-1)
    return dist, frac
