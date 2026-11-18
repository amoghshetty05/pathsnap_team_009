"""Exploratory data analysis for TRAIN and STUDENT-TEST partitions.

Run:  python -m data_processing.eda        (after data_processing.build_dataset)

Writes to results/eda/:
    inventory.csv                 per-trip file sizes, row counts, time span, gaps, extent
    network_connectivity.csv      per-trip node/arc counts, isolated nodes, weakly/strongly connected components
    data_quality_issues.csv       trips with large gaps / poor label agreement
    partition_summary.csv         aggregate statistics per partition
    projection_check.csv          local projection error vs haversine
    *.png                         plots
results/maps/eda_trip_<id>.html   Folium maps (GPS + network + reference route)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from configs import config as C
from data_processing.geo import haversine
from data_processing.io import list_trips, load_trip
from visualisations.folium_maps import trip_map

OUT = C.RESULTS_DIR / "eda"
MAPS = C.RESULTS_DIR / "maps"


def trip_stats(partition, pdir, tid):
    d = pdir / tid
    trip = load_trip(pdir, tid)
    raw = np.loadtxt(d / f"{tid}.track", ndmin=2)
    t = trip.times
    dt = np.diff(t)
    lon, lat = trip.track[:, 0], trip.track[:, 1]
    step = haversine(lon[:-1], lat[:-1], lon[1:], lat[1:])
    speed = np.divide(step, dt, out=np.zeros_like(step), where=dt > 0)

    A = len(trip.arcs)
    N = len(trip.nodes)
    g = coo_matrix((np.ones(A), (trip.arcs[:, 0], trip.arcs[:, 1])), shape=(N, N)).tocsr()
    n_weak, weak_lab = connected_components(g, directed=True, connection="weak")
    n_strong, strong_lab = connected_components(g, directed=True, connection="strong")
    deg = np.bincount(trip.arcs.ravel(), minlength=N)
    route_ok = None
    if trip.route is not None:
        r = trip.route
        route_ok = float(np.mean(trip.arcs[r[:-1], 1] == trip.arcs[r[1:], 0])) if len(r) > 1 else 1.0

    inv = {
        "partition": partition, "trip_id": tid,
        "track_bytes": (d / f"{tid}.track").stat().st_size,
        "nodes_bytes": (d / f"{tid}.nodes").stat().st_size,
        "arcs_bytes": (d / f"{tid}.arcs").stat().st_size,
        "route_exists": (d / f"{tid}.route").exists(),
        "n_obs": len(raw), "n_nonfinite": int((~np.isfinite(raw)).any(axis=1).sum()),
        "non_monotonic_ts": int((np.diff(raw[:, 2]) <= 0).sum()),
        "duration_s": float(t[-1] - t[0]),
        "pct_dt_1s": float(np.mean(dt == 1) * 100) if len(dt) else np.nan,
        "max_gap_s": float(dt.max()) if len(dt) else 0.0,
        "n_gaps_gt10s": int((dt > 10).sum()),
        "path_length_km": float(step.sum() / 1000),
        "median_speed_mps": float(np.median(speed)) if len(speed) else 0.0,
        "p99_speed_mps": float(np.percentile(speed, 99)) if len(speed) else 0.0,
        "lon_min": lon.min(), "lon_max": lon.max(), "lat_min": lat.min(), "lat_max": lat.max(),
        "lon_center": lon.mean(), "lat_center": lat.mean(),
        "n_route_arcs": len(trip.route) if trip.route is not None else 0,
        "route_consecutive_connected": route_ok,
    }
    net = {
        "partition": partition, "trip_id": tid, "n_nodes": N, "n_arcs": A,
        "isolated_nodes": int((deg == 0).sum()),
        "weak_components": int(n_weak),
        "largest_weak_component_frac": float(np.bincount(weak_lab).max() / N),
        "strong_components": int(n_strong),
        "largest_strong_component_frac": float(np.bincount(strong_lab).max() / N),
        "two_way_arc_frac": float(np.mean(pd.MultiIndex.from_arrays(
            [trip.arcs[:, 1], trip.arcs[:, 0]]).isin(pd.MultiIndex.from_arrays(
                [trip.arcs[:, 0], trip.arcs[:, 1]])))),
        "mean_arc_len_m": float(trip.seg_len.mean()),
    }
    # projection check: local projected distance vs haversine for consecutive points
    proj_d = np.linalg.norm(np.diff(trip.xy, axis=0), axis=1)
    far = np.linalg.norm(trip.xy[-1] - trip.xy[0])
    far_h = haversine(lon[0], lat[0], lon[-1], lat[-1])
    pc = {"trip_id": tid, "max_abs_err_consecutive_m": float(np.max(np.abs(proj_d - step))) if len(step) else 0,
          "start_end_rel_err": float(abs(far - far_h) / far_h) if far_h > 0 else 0.0}
    return inv, net, pc, speed, dt


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    MAPS.mkdir(parents=True, exist_ok=True)
    inv, net, pcs, speeds, dts = [], [], [], {}, {}
    for part, pdir in [("train", C.TRAIN_DIR), ("student_test", C.STUDENT_TEST_DIR)]:
        speeds[part], dts[part] = [], []
        for tid in list_trips(pdir):
            a, b, pc, sp, dt = trip_stats(part, pdir, tid)
            inv.append(a); net.append(b); pcs.append(pc)
            speeds[part].append(sp); dts[part].append(dt)
    inv = pd.DataFrame(inv); net = pd.DataFrame(net)
    inv.to_csv(OUT / "inventory.csv", index=False)
    net.to_csv(OUT / "network_connectivity.csv", index=False)
    pd.DataFrame(pcs).to_csv(OUT / "projection_check.csv", index=False)

    # label agreement from build_dataset
    pts = pd.concat([pd.read_parquet(C.PROCESSED_DIR / f"{p}_points.parquet").assign(partition=p)
                     for p in ("train", "student_test")])
    lab = pts.groupby(["partition", "trip_id"]).agg(
        median_label_dist_m=("label_dist", "median"),
        unreliable_label_pct=("label_ok", lambda s: 100 * (1 - s.mean())),
        candidate_miss_pct=("label_in_cands", lambda s: 100 * (1 - s.mean()))).reset_index()
    inv = inv.merge(lab, on=["partition", "trip_id"])

    summ = inv.groupby("partition").agg(
        trips=("trip_id", "count"), total_obs=("n_obs", "sum"), min_obs=("n_obs", "min"),
        median_obs=("n_obs", "median"), max_obs=("n_obs", "max"),
        nonfinite_rows=("n_nonfinite", "sum"), non_monotonic_ts=("non_monotonic_ts", "sum"),
        mean_pct_dt_1s=("pct_dt_1s", "mean"), total_gaps_gt10s=("n_gaps_gt10s", "sum"),
        max_gap_s=("max_gap_s", "max"), total_km=("path_length_km", "sum"),
        trips_without_route=("route_exists", lambda s: int((~s).sum())))
    summ = summ.join(net.groupby("partition").agg(
        median_nodes=("n_nodes", "median"), max_nodes=("n_nodes", "max"),
        median_arcs=("n_arcs", "median"), max_arcs=("n_arcs", "max"),
        total_isolated_nodes=("isolated_nodes", "sum"),
        mean_weak_components=("weak_components", "mean")))
    summ.to_csv(OUT / "partition_summary.csv")
    print(summ.T.to_string())

    issues = inv[(inv.max_gap_s > 60) | (inv.unreliable_label_pct > 5) | (inv.candidate_miss_pct > 2)]
    issues = issues[["partition", "trip_id", "n_obs", "max_gap_s", "n_gaps_gt10s",
                     "median_label_dist_m", "unreliable_label_pct", "candidate_miss_pct"]]
    issues.to_csv(OUT / "data_quality_issues.csv", index=False)
    print("\nData-quality issues:\n", issues.to_string(index=False))

    # ---- plots ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for part in ("train", "student_test"):
        ax[0].hist(inv[inv.partition == part].n_obs, bins=30, alpha=0.6, label=part)
    ax[0].set(xlabel="GPS observations per trip", ylabel="trips", title="Trajectory length")
    ax[0].legend()
    allg = np.concatenate(dts["train"])
    ax[1].hist(np.clip(allg, 0, 30), bins=30, color="tab:purple")
    ax[1].set(yscale="log", xlabel="sampling interval (s), clipped at 30", ylabel="count",
              title="Sampling intervals (train)")
    fig.tight_layout(); fig.savefig(OUT / "trajectory_length_and_sampling.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for part, mk in (("train", "o"), ("student_test", "^")):
        s = inv[inv.partition == part]
        ax.scatter(s.lon_center, s.lat_center, marker=mk, label=part, alpha=0.8)
    ax.set(xlabel="longitude", ylabel="latitude", title="Trip locations (GPS centroid)",
           xlim=(-180, 180), ylim=(-60, 75))
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "trip_locations.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(np.clip(np.concatenate(speeds["train"]), 0, 50), bins=50, color="tab:green")
    ax[0].set(xlabel="speed (m/s), clipped at 50", ylabel="count", title="GPS speed (train)")
    tr = pts[pts.partition == "train"]
    ax[1].hist(np.clip(tr.label_dist, 0, 200), bins=60, color="tab:red")
    ax[1].axvline(C.LABEL_MAX_DIST_M, ls="--", c="k", label=f"{C.LABEL_MAX_DIST_M:.0f} m reliability cut-off")
    ax[1].set(yscale="log", xlabel="distance GPS -> aligned reference arc (m)", title="GPS vs reference route (train)")
    ax[1].legend()
    fig.tight_layout(); fig.savefig(OUT / "speed_and_label_distance.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].scatter(net.n_nodes, net.n_arcs, c=(net.partition == "train").map({True: "tab:blue", False: "tab:orange"}))
    ax[0].set(xscale="log", yscale="log", xlabel="nodes", ylabel="arcs", title="Road-network size per trip")
    ax[1].hist(net.largest_strong_component_frac, bins=20, color="tab:gray")
    ax[1].set(xlabel="fraction of nodes in largest strongly connected component", ylabel="trips",
              title="Network connectivity")
    fig.tight_layout(); fig.savefig(OUT / "network_size_connectivity.png", dpi=130); plt.close(fig)

    sweep = pd.read_csv(OUT / "candidate_radius_sweep.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sweep.radius_m, 100 * sweep.miss_rate, "o-", label="candidate miss rate (%)")
    ax2 = ax.twinx()
    ax2.plot(sweep.radius_m, sweep.mean_cands, "s--", c="tab:orange", label="mean candidates")
    ax.axvline(C.CANDIDATE_RADIUS_M, c="k", ls=":")
    ax.set(xlabel="search radius (m)", ylabel="miss rate (%)", title="Candidate radius sweep (train)")
    ax2.set_ylabel("mean candidates per observation")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.88))
    fig.tight_layout(); fig.savefig(OUT / "candidate_radius_sweep.png", dpi=130); plt.close(fig)

    # sample GPS trajectory plots in local metres
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, tid in zip(axes.ravel(), list_trips(C.TRAIN_DIR)[:6]):
        trip = load_trip(C.TRAIN_DIR, tid)
        r = trip.route
        for a in r:
            ax.plot(*np.c_[trip.seg_a[a], trip.seg_b[a]], c="tab:green", lw=2, alpha=0.6)
        ax.plot(trip.xy[:, 0], trip.xy[:, 1], ".", ms=1.5, c="tab:blue")
        ax.set(title=f"train {tid}", aspect="equal"); ax.tick_params(labelsize=7)
    fig.suptitle("GPS (blue) vs reference route (green), local metres")
    fig.tight_layout(); fig.savefig(OUT / "sample_trajectories.png", dpi=130); plt.close(fig)

    # Folium maps: a typical train trip, the worst-agreement train trip, and one student-test trip
    worst = lab[lab.partition == "train"].sort_values("median_label_dist_m").trip_id.iloc[-1]
    for part, pdir, tid in [("train", C.TRAIN_DIR, "00000048"), ("train", C.TRAIN_DIR, worst),
                            ("student_test", C.STUDENT_TEST_DIR, list_trips(C.STUDENT_TEST_DIR)[0])]:
        trip = load_trip(pdir, tid)
        m = trip_map(trip, {"reference": trip.route}, gps_every=2,
                     title=f"EDA - {part} trip {tid}")
        m.save(str(MAPS / f"eda_{part}_trip_{tid}.html"))
    print("EDA written to", OUT)


if __name__ == "__main__":
    main()
