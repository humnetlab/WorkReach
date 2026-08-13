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
