"""Sectoral relatedness between the origin and destination of a commute.

Relatedness reads the employment mix of both ends of a trip through the product
space, so an origin-destination pair scores highly when the capabilities housed at
the origin are close to those the destination draws on.
"""

import numpy as np
import pandas as pd


def employment_shares(employment):
    """Share of each zone's workers in every sector."""
    pivot = employment.pivot_table(index="geomid", columns="naics", values="workers",
                                   aggfunc="sum", fill_value=0).astype(float)
    totals = pivot.sum(axis=1).replace(0, np.nan)
    return pivot.div(totals, axis=0).fillna(0.0)


def dominant_sector(shares):
    """The sector each zone employs the most workers in."""
    return shares.idxmax(axis=1)


def relatedness_for_flows(flows, shares, proximity):
    """Relatedness and cosine similarity for every origin-destination pair."""
    sectors = shares.columns.intersection(proximity.columns)
    share_values = shares[sectors].values
    phi = proximity.loc[sectors, sectors].values

    position = {zone: i for i, zone in enumerate(shares.index)}
    origin_index = flows["home_geomid"].map(position)
    destination_index = flows["work_geomid"].map(position)
    usable = origin_index.notna() & destination_index.notna()

    relatedness = np.full(len(flows), np.nan)
    cosine = np.full(len(flows), np.nan)

    if usable.any():
        origins = share_values[origin_index[usable].astype(int).values]
        destinations = share_values[destination_index[usable].astype(int).values]

        relatedness[usable.values] = (origins @ phi * destinations).sum(axis=1)

        norms = np.linalg.norm(origins, axis=1) * np.linalg.norm(destinations, axis=1)
        norms[norms == 0] = 1e-12
        cosine[usable.values] = (origins * destinations).sum(axis=1) / norms

    return (pd.Series(relatedness, index=flows.index, name="sr"),
            pd.Series(cosine, index=flows.index, name="cosine"))


def flow_weighted_median(values, weights):
    """Median of values, weighting each by its flow."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, cumulative[-1] / 2.0)])


def sector_profiles(flows, dominant):
    """Per-sector commuting profile, weighted by flows leaving that sector."""
    frame = flows[flows["flows"] > 0].copy()
    frame["home_sector"] = frame["home_geomid"].map(dominant)
    frame["work_sector"] = frame["work_geomid"].map(dominant)
    frame = frame.dropna(subset=["home_sector", "work_sector"])
    frame["same"] = (frame["home_sector"] == frame["work_sector"]).astype(float)

    rows = []
    for sector, group in frame.groupby("home_sector"):
        weights = group["flows"].values.astype(float)
        if weights.sum() < 1:
            continue
        rows.append({
            "sector": sector,
            "total_flows": weights.sum(),
            "within_pct": (group["same"] * weights).sum() / weights.sum() * 100,
            "med_dist_km": flow_weighted_median(
                group["distance_home_to_work"].values / 1000, weights),
            "med_eci": flow_weighted_median(group["eci"].values, weights),
            "med_sr": flow_weighted_median(group["sr"].values, weights),
        })
    return pd.DataFrame(rows)


DECILE_LABELS = [f"D{i + 1}" for i in range(10)]
INFORMALITY_LABELS = ["Low inf.", "High inf."]


def flow_weighted_cut(distances, flows, n_bins, labels):
    """Bin distances so that each bin carries roughly the same total flow."""
    distances = np.asarray(distances, dtype=float)
    flows = np.asarray(flows, dtype=float)
    order = np.argsort(distances)
    sorted_distance, sorted_flow = distances[order], flows[order]
    cumulative = np.cumsum(sorted_flow)

    edges = [-np.inf]
    for fraction in np.linspace(0, 1, n_bins + 1)[1:-1]:
        position = min(int(np.searchsorted(cumulative, fraction * cumulative[-1],
                                           side="right")), len(sorted_distance) - 1)
        edges.append(float(sorted_distance[position]))
    edges.append(np.inf)
    edges = sorted(set(edges))

    return pd.cut(distances, bins=edges, labels=labels[:len(edges) - 1],
                  include_lowest=True)


def weighted_median_ci(values, weights, n_boot=500, seed=42):
    """Flow-weighted median with a bootstrap confidence interval."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    usable = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[usable], weights[usable]
    if len(values) < 2:
        return np.nan, np.nan, np.nan

    median = flow_weighted_median(values, weights)
    rng = np.random.default_rng(seed)
    draws = np.array([flow_weighted_median(values[idx], weights[idx])
                      for idx in (rng.integers(0, len(values), size=len(values))
                                  for _ in range(n_boot))])
    return median, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def weighted_linear_fit(x, y, weights=None):
    """Weighted least squares, returning the correlation and a line to draw."""
    from scipy import stats as sp_stats

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    weights = np.ones(len(x)) if weights is None else np.asarray(weights, dtype=float)
    usable = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights) & (weights > 0)
    x, y, weights = x[usable], y[usable], weights[usable]
    if len(x) < 3:
        return None

    weights = weights / weights.sum()
    x_mean, y_mean = weights @ x, weights @ y
    covariance = weights @ ((x - x_mean) * (y - y_mean))
    var_x = weights @ (x - x_mean) ** 2
    var_y = weights @ (y - y_mean) ** 2

    r = covariance / np.sqrt(max(1e-30, var_x * var_y))
    slope = covariance / max(1e-30, var_x)
    n = int(usable.sum())
    t_statistic = r * np.sqrt(n - 2) / np.sqrt(max(1e-15, 1 - r ** 2))

    line_x = np.array([x.min(), x.max()])
    return dict(r=r, p=2 * sp_stats.t.sf(abs(t_statistic), df=n - 2),
                x_fit=line_x, y_fit=y_mean - slope * x_mean + slope * line_x, n=n)


def origin_decile_table(flows):
    """Relatedness per origin zone and commuting-distance decile."""
    frame = flows.dropna(subset=["sr", "home_informality", "flows",
                                 "distance_home_to_work"]).copy()
    frame = frame[frame["flows"] > 0].reset_index(drop=True)
    frame["decile"] = flow_weighted_cut(frame["distance_home_to_work"].values,
                                        frame["flows"].values, 10, DECILE_LABELS)
    frame["weighted_sr"] = frame["sr"] * frame["flows"]

    table = (frame.groupby(["home_geomid", "decile"], observed=True)
             .agg(weighted_sr=("weighted_sr", "sum"),
                  total_flows=("flows", "sum"),
                  mean_sr=("sr", "mean"))
             .reset_index())
    table["fwm_sr"] = table["weighted_sr"] / table["total_flows"].clip(lower=1e-9)
    table = table.drop(columns="weighted_sr")

    per_origin = frame.groupby("home_geomid").agg(
        home_informality=("home_informality", "first")).reset_index()
    median_informality = per_origin["home_informality"].median()
    per_origin["inf_bin"] = np.where(
        per_origin["home_informality"] <= median_informality,
        INFORMALITY_LABELS[0], INFORMALITY_LABELS[1])

    table = table.merge(per_origin, on="home_geomid", how="left")

    decile_km = {}
    for label in DECILE_LABELS:
        group = frame[frame["decile"] == label]
        if len(group) >= 2:
            decile_km[label] = flow_weighted_median(
                group["distance_home_to_work"].values, group["flows"].values) / 1000
    return table, decile_km
