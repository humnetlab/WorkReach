"""Economic complexity quantities derived from the zone by sector employment tables."""

import numpy as np
import pandas as pd
import networkx as nx


def load_employment(path):
    """Read a zone by sector employment table."""
    return pd.read_csv(path, dtype={"geomid": str})


def compute_binary_rca(employment):
    """Binary RCA matrix from a long table of geomid, naics and workers."""
    pivot = employment.pivot_table(index="geomid", columns="naics", values="workers",
                                   aggfunc="sum", fill_value=0).astype(float)
    row_sums = pivot.sum(axis=1)
    col_sums = pivot.sum(axis=0)
    total = pivot.values.sum()
    rca = pivot.div(row_sums, axis=0).div(col_sums / total, axis=1)
    return (rca >= 1.0).astype(int)


def compute_pci(matrix):
    """Product complexity from the second eigenvector of the sector projection."""
    zone_diversity = matrix.sum(axis=1).values.astype(float)
    sector_ubiquity = matrix.sum(axis=0).values.astype(float)
    zone_diversity[zone_diversity == 0] = 1.0
    sector_ubiquity[sector_ubiquity == 0] = 1.0

    values = matrix.values.astype(float)
    normalised = values / zone_diversity[:, None] / sector_ubiquity[None, :]
    projection = normalised.T @ values

    _, eigenvectors = np.linalg.eigh(projection)
    pci = eigenvectors[:, -2].real
    pci = (pci - pci.mean()) / (pci.std() + 1e-10)
    if np.corrcoef(pci, matrix.sum(axis=0).values)[0, 1] > 0:
        pci = -pci
    return pd.Series(pci, index=matrix.columns, name="pci")


def compute_eci(matrix):
    """Zone complexity from the same projection, taken on the zone side."""
    zone_diversity = matrix.sum(axis=1).values.astype(float)
    sector_ubiquity = matrix.sum(axis=0).values.astype(float)
    zone_diversity[zone_diversity == 0] = 1.0
    sector_ubiquity[sector_ubiquity == 0] = 1.0

    values = matrix.values.astype(float)
    normalised = values / zone_diversity[:, None] / sector_ubiquity[None, :]
    projection = normalised @ values.T

    _, eigenvectors = np.linalg.eigh(projection)
    eci = eigenvectors[:, -2].real
    eci = (eci - eci.mean()) / (eci.std() + 1e-10)
    if np.corrcoef(eci, matrix.sum(axis=1).values)[0, 1] < 0:
        eci = -eci
    return pd.Series(eci, index=matrix.index, name="eci")


def compute_proximity(matrix):
    """Minimum conditional probability that two sectors co-occur in a zone."""
    values = matrix.values.astype(float)
    ubiquity = values.sum(axis=0)
    ubiquity[ubiquity == 0] = 1.0

    n_sectors = values.shape[1]
    phi = np.zeros((n_sectors, n_sectors))
    for i in range(n_sectors):
        for j in range(n_sectors):
            co_occurrence = (values[:, i] * values[:, j]).sum()
            phi[i, j] = co_occurrence / max(ubiquity[i], ubiquity[j])
    return pd.DataFrame(phi, index=matrix.columns, columns=matrix.columns)


def product_space_graph(proximity, threshold=0.20):
    """Sectors linked wherever proximity exceeds the threshold."""
    graph = nx.Graph()
    graph.add_nodes_from(proximity.columns)
    for i, first in enumerate(proximity.columns):
        for second in list(proximity.columns)[i + 1:]:
            if proximity.loc[first, second] > threshold:
                graph.add_edge(first, second, weight=proximity.loc[first, second])
    return graph


def nodf(matrix):
    """Nestedness measured by overlap and decreasing fill."""
    values = matrix.values.astype(int) if hasattr(matrix, "values") else matrix.astype(int)
    n_rows, n_cols = values.shape
    row_sums = values.sum(axis=1)
    col_sums = values.sum(axis=0)

    def paired(fills, take):
        total, pairs = 0.0, 0
        for i in range(len(fills)):
            for j in range(i + 1, len(fills)):
                if fills[i] == fills[j]:
                    continue
                high, low = (i, j) if fills[i] > fills[j] else (j, i)
                if fills[low] == 0:
                    continue
                total += (take(high) & take(low)).sum() / fills[low]
                pairs += 1
        return total / pairs if pairs else 0.0

    rows = paired(row_sums, lambda i: values[i])
    columns = paired(col_sums, lambda i: values[:, i])
    return 100.0 * (rows + columns) / 2.0


def nodf_null(matrix, n_iterations=999, seed=0):
    """NODF of matrices randomised at the observed fill."""
    values = matrix.values.astype(int) if hasattr(matrix, "values") else matrix.astype(int)
    rng = np.random.default_rng(seed)
    fill = values.mean()
    return np.array([nodf((rng.random(values.shape) < fill).astype(int))
                     for _ in range(n_iterations)])


def mst_with_threshold(proximity, threshold):
    """Maximum spanning tree of the proximity graph, plus every edge above a threshold."""
    sectors = list(proximity.columns)
    complete = nx.Graph()
    for i, first in enumerate(sectors):
        for second in sectors[i + 1:]:
            weight = proximity.loc[first, second]
            if weight > 0:
                complete.add_edge(first, second, weight=weight, neg_weight=-weight)

    tree = nx.minimum_spanning_tree(complete, weight="neg_weight")
    edges = set((min(u, v), max(u, v)) for u, v in tree.edges())
    for i, first in enumerate(sectors):
        for second in sectors[i + 1:]:
            if proximity.loc[first, second] > threshold:
                edges.add((min(first, second), max(first, second)))

    graph = nx.Graph()
    graph.add_nodes_from(sectors)
    for first, second in edges:
        graph.add_edge(first, second, weight=proximity.loc[first, second])
    return graph


PRODUCT_SPACE_THRESHOLDS = {"Bay Area": 0.30, "Los Angeles": 0.30,
                            "Mexico City": 0.20, "Rio de Janeiro": 0.20}


def zone_components(matrix):
    """Diversity and mean ubiquity of the sectors each zone is specialised in."""
    ubiquity = matrix.sum(axis=0)
    diversity = matrix.sum(axis=1)

    mean_ubiquity = matrix.apply(
        lambda row: ubiquity[row == 1].mean() if row.sum() else np.nan, axis=1)

    return pd.DataFrame({"geomid": matrix.index,
                         "diversity": diversity.values,
                         "mean_ubiquity": mean_ubiquity.values})
