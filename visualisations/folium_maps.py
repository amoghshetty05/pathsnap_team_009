"""Folium map generation: GPS observations, nearby road network, reference route and predicted route.

Used by the EDA script, the HMM evaluation script and the API (which returns the predicted route
geometry; the UI draws it with Leaflet).
"""
import folium
import numpy as np

from data_processing.geo import point_segment

COLORS = {"gps": "#1f77b4", "network": "#9e9e9e", "reference": "#2ca02c",
          "classifier": "#ff7f0e", "hmm": "#d62728"}


def _nearby_arcs(trip, margin_m=150.0, max_arcs=6000):
    """Arcs whose midpoint is within margin_m of any GPS point (keeps maps light)."""
    from scipy.spatial import cKDTree
    mid = (trip.seg_a + trip.seg_b) / 2
    tree = cKDTree(trip.xy)
    d, _ = tree.query(mid, distance_upper_bound=margin_m + 1)
    idx = np.where(np.isfinite(d))[0]
    if len(idx) > max_arcs:
        idx = idx[np.argsort(d[idx])[:max_arcs]]
    return idx


def arcs_to_lines(trip, arc_ids):
    """List of [[lat, lon], [lat, lon]] segments for the given arcs."""
    a = trip.nodes[trip.arcs[arc_ids, 0]]
    b = trip.nodes[trip.arcs[arc_ids, 1]]
    return [[[float(p[1]), float(p[0])], [float(q[1]), float(q[0])]] for p, q in zip(a, b)]


def collapse(seq):
    """Remove consecutive duplicates / None from a per-observation arc sequence."""
    out = []
    for a in seq:
        if a is None or (isinstance(a, float) and np.isnan(a)):
            continue
        a = int(a)
        if not out or out[-1] != a:
            out.append(a)
    return out


def trip_map(trip, layers=None, gps_every=1, title=None, show_network=True):
    """layers: dict name -> list of arc ids (e.g. {'reference': route, 'hmm': pred_arcs})."""
    lonlat = trip.track[:, :2]
    center = [float(lonlat[:, 1].mean()), float(lonlat[:, 0].mean())]
    m = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap", control_scale=True)

    if show_network:
        fg = folium.FeatureGroup(name="Road network (nearby)", show=True)
        for line in arcs_to_lines(trip, _nearby_arcs(trip)):
            folium.PolyLine(line, color=COLORS["network"], weight=1, opacity=0.5).add_to(fg)
        fg.add_to(m)

    styles = {"reference": dict(weight=7, opacity=0.55, label="Reference route"),
              "classifier": dict(weight=3, opacity=0.9, label="Classifier-only prediction"),
              "hmm": dict(weight=4, opacity=0.9, label="Predicted route (classifier + HMM)")}
    for name, arc_ids in (layers or {}).items():
        st = styles.get(name, dict(weight=3, opacity=0.9, label=name))
        fg = folium.FeatureGroup(name=st["label"], show=True)
        for line in arcs_to_lines(trip, np.asarray(collapse(arc_ids), dtype=np.int64)):
            folium.PolyLine(line, color=COLORS.get(name, "#9467bd"), weight=st["weight"],
                            opacity=st["opacity"]).add_to(fg)
        fg.add_to(m)

    fg = folium.FeatureGroup(name="Raw GPS observations", show=True)
    for lon, lat in lonlat[::gps_every]:
        folium.CircleMarker([float(lat), float(lon)], radius=2, color=COLORS["gps"],
                            fill=True, fill_opacity=0.8, weight=0).add_to(fg)
    fg.add_to(m)
    folium.Marker([float(lonlat[0, 1]), float(lonlat[0, 0])], tooltip="Start",
                  icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([float(lonlat[-1, 1]), float(lonlat[-1, 0])], tooltip="End",
                  icon=folium.Icon(color="red", icon="stop")).add_to(m)

    legend = "".join(
        f'<div><span style="display:inline-block;width:14px;height:4px;background:{COLORS[k]};'
        f'margin-right:6px;vertical-align:middle"></span>{v}</div>'
        for k, v in [("gps", "GPS observations"), ("network", "Road network"),
                     ("reference", "Reference route"), ("classifier", "Classifier-only"),
                     ("hmm", "Classifier + HMM")]
        if k in ("gps", "network") or k in (layers or {}))
    html = (f'<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;'
            f'padding:8px 10px;border:1px solid #999;font-size:12px">'
            f'<b>{title or "Trip " + trip.trip_id}</b>{legend}</div>')
    m.get_root().html.add_child(folium.Element(html))
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[float(lonlat[:, 1].min()), float(lonlat[:, 0].min())],
                  [float(lonlat[:, 1].max()), float(lonlat[:, 0].max())]])
    return m
