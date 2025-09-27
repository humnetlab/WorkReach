import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Tuple
from haversine import haversine, Unit

def load_and_prepare_data(
    mzn_path: str,
    flows_path: str,
    city_name: str = "Unknown City",
    expansion: bool = True
) -> Tuple[pd.DataFrame, gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Load and prepare all necessary data for modeling."""
    mzn = gpd.read_file(mzn_path)
    mzn = mzn.loc[mzn["population"] > 0]
    mzn = mzn.loc[~((mzn.eci.isna()) | (mzn.informality_rate.isna()))]

    aligned_data = pd.read_csv(flows_path, dtype={'home_geomid': str, 'work_geomid': str})
    aligned_data = aligned_data.drop(columns="eci")


    aligned_data = pd.merge(
        aligned_data,
        mzn[["geomid","informality_rate"]],
        left_on="home_geomid",
        right_on="geomid",
        how="inner"
    ).drop(columns="geomid")

    aligned_data = pd.merge(
        aligned_data,
        mzn[["geomid","eci"]],
        left_on="work_geomid",
        right_on="geomid",
        how="inner"
    ).drop(columns="geomid")


    aligned_data.loc[aligned_data["home_geomid"] == aligned_data["work_geomid"], "flows"] = 0


    aligned_data["informality_rate"] = aligned_data["informality_rate"].fillna(aligned_data["informality_rate"].max())
    aligned_data["eci"] = aligned_data["eci"].fillna(aligned_data["eci"].min())


    valid_geomids = set(aligned_data["home_geomid"]).intersection(set(aligned_data["work_geomid"]))
    aligned_data = aligned_data[
        aligned_data["home_geomid"].isin(valid_geomids) &
        aligned_data["work_geomid"].isin(valid_geomids)
    ]
    aligned_data["distance_home_to_work"] = aligned_data["distance_home_to_work"].fillna(
        aligned_data["distance_home_to_work"].replace(0, np.nan).min())


    flows_data = aligned_data.copy()


    if expansion:
        print("Applying expansion factors...")
        exp_fact = calculate_expansion_factors(aligned_data, mzn)
        flows_data = flows_data.merge(
            exp_fact[["home_geomid", "efactor"]],
            on="home_geomid",
            how="left"
        )
        flows_data["flows"] = np.round(flows_data["flows"] * flows_data["efactor"]).fillna(0).astype(int)
    else:
        print("Using original flows without expansion...")
        flows_data["flows"] = np.round(flows_data["flows"]).fillna(0).astype(int)


    mzn["radius"] = mzn.geometry.length / (2 * np.pi)
    median_radius = mzn["radius"].median()/2

    mean_eci = mzn["eci"].mean()
    mean_informality = mzn["informality_rate"].mean()


    distance_matrix = flows_data.pivot(
        index='home_geomid', columns='work_geomid', values='distance_home_to_work'
    ).fillna(median_radius)

    eci_matrix = flows_data.pivot(
        index='home_geomid', columns='work_geomid', values='eci'
    ).fillna(mean_eci)

    flows_matrix = flows_data.pivot(
        index='home_geomid', columns='work_geomid', values='flows'
    ).fillna(0)

    home_matrix = flows_data.pivot(
        index='home_geomid', columns='work_geomid', values='population'
    ).dropna()

    informality_matrix = flows_data.pivot(
        index='home_geomid', columns='work_geomid', values='informality_rate'
    ).fillna(mean_informality)


    all_indices = sorted(list(set(distance_matrix.index) &
                            set(eci_matrix.index) &
                            set(flows_matrix.index) &
                            set(home_matrix.index) &
                            set(informality_matrix.index)))

    all_columns = sorted(list(set(distance_matrix.columns) &
                             set(eci_matrix.columns) &
                             set(flows_matrix.columns) &
                             set(home_matrix.columns) &
                             set(informality_matrix.columns)))

    distance_matrix = distance_matrix.loc[all_indices, all_columns]
    eci_matrix = eci_matrix.loc[all_indices, all_columns]
    flows_matrix = flows_matrix.loc[all_indices, all_columns]
    home_matrix = home_matrix.loc[all_indices, all_columns]
    informality_matrix = informality_matrix.loc[all_indices, all_columns]

    return flows_data, mzn, distance_matrix, eci_matrix, flows_matrix, home_matrix, informality_matrix, city_name

def calculate_expansion_factors(df: pd.DataFrame, mzn: gpd.GeoDataFrame) -> pd.DataFrame:
    """Calculate expansion factors to scale CDR/LBS data to match population."""

    exp_fact = (df.groupby("home_geomid")["flows"]
                .sum()
                .reset_index()
                .merge(mzn[["geomid","population"]],
                      how="left",
                      left_on="home_geomid",
                      right_on="geomid")
                .drop(columns="geomid")
                .rename(columns={"flows":"users"}))


    exp_fact = exp_fact.query("users > 10")


    exp_fact["efactor"] = exp_fact["population"] / exp_fact["users"]

    return exp_fact

def preprocess_data(
    home_matrix: pd.DataFrame,
    distance_matrix: pd.DataFrame,
    eci_matrix: pd.DataFrame,
    flows_matrix: pd.DataFrame,
    informality_matrix: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
    """Preprocess matrices for modeling."""

    distance = distance_matrix.values
    eci = eci_matrix.values
    flows = flows_matrix.values
    informality = informality_matrix.values


    home_population = home_matrix.max(axis=1).astype(int).values
    work_population = home_matrix[home_matrix.columns].max(axis=1).astype(int).values


    distance_min = np.min(distance)
    distance_max = np.max(distance)
    distance_diff = distance_max - distance_min
    distance_log = ((distance - distance_min) / distance_diff)


    eci = np.exp(eci)
    eci_min = np.min(eci)
    eci_max = np.max(eci)
    eci_diff = eci_max - eci_min
    eci = ((eci - eci_min) / eci_diff)

    informality = np.exp(informality)
    informality_min = np.min(informality)
    informality_max = np.max(informality)
    informality_diff = informality_max - informality_min
    informality = ((informality - informality_min) / informality_diff)

    return distance, distance_log, eci, flows, informality, home_population, work_population, distance_min, distance_diff, eci_min, eci_diff

def prepare_flow_map_data(
    flow_df: pd.DataFrame,
    gdf: gpd.GeoDataFrame
) -> pd.DataFrame:
    """Prepare data for flow map visualization."""

    gdf['centroid'] = gdf.geometry.centroid
    centroid_df = gdf[["geomid", "centroid"]].copy()


    home_centroids = centroid_df.rename(
        columns={"geomid": "home_geomid", "centroid": "home_centroid"}
    )
    flow_merged = flow_df.merge(home_centroids, on="home_geomid", how="left")


    work_centroids = centroid_df.rename(
        columns={"geomid": "work_geomid", "centroid": "work_centroid"}
    )
    flow_merged = flow_merged.merge(work_centroids, on="work_geomid", how="left")


    flow_merged["home_lon"] = flow_merged["home_centroid"].apply(
        lambda p: p.x if p is not None else np.nan
    )
    flow_merged["home_lat"] = flow_merged["home_centroid"].apply(
        lambda p: p.y if p is not None else np.nan
    )
    flow_merged["work_lon"] = flow_merged["work_centroid"].apply(
        lambda p: p.x if p is not None else np.nan
    )
    flow_merged["work_lat"] = flow_merged["work_centroid"].apply(
        lambda p: p.y if p is not None else np.nan
    )


    flow_merged = flow_merged.drop(columns=["home_centroid", "work_centroid"])
    flow_merged = flow_merged.dropna(subset=["home_lon", "work_lon"])

    return flow_merged.query('flows > 1')

def create_dataset_summary_table(
    aligned_data: pd.DataFrame,
    mzn: gpd.GeoDataFrame,
    city_name: str = "Dataset"
) -> pd.DataFrame:
    """Create a summary table with dataset statistics."""

    unique_home = aligned_data['home_geomid'].unique()
    unique_work = aligned_data['work_geomid'].unique()
    all_geoms = np.unique(np.concatenate([unique_home, unique_work]))


    entries = len(aligned_data)


    total_population = mzn.query('population > 0').loc[mzn['geomid'].isin(all_geoms), 'population'].sum()


    flow_min = aligned_data.query('flows > 0')['flows'].min()
    flow_max = aligned_data['flows'].max()


    dist_min = aligned_data['distance_home_to_work'].min() / 1000
    dist_max = aligned_data['distance_home_to_work'].max() / 1000


    pop_min = mzn.query('population > 0').loc[mzn['geomid'].isin(all_geoms), 'population'].min()
    pop_max = mzn.loc[mzn['geomid'].isin(all_geoms), 'population'].max()


    city_projections = {
        "Rio de Janeiro": 31983,
        "Bay Area": 26910,
        "Los Angeles": 26911,
        "Mexico City": 32614
    }


    temp_gdf = mzn.loc[mzn['geomid'].isin(all_geoms), ['geomid', 'geometry']].copy()


    epsg_code = city_projections.get(city_name)

    if epsg_code:

        temp_gdf = temp_gdf.to_crs(epsg=epsg_code)
    else:

        temp_gdf = temp_gdf.to_crs(epsg=3857)


    areas_km2 = temp_gdf.geometry.area / 10**6
    area_avg = areas_km2.mean()


    summary = {
        "Dataset": city_name,
        "Entries": entries,
        "Municipalities": len(all_geoms),
        "Median Area (km²)": round(areas_km2.median(), 2),
        "Avg Area (km²)": round(area_avg, 2),
        "Total Area (km²)": round(areas_km2.sum(), 2),
        "Flow Min": flow_min,
        "Flow Max": flow_max,
        "Distance Min (km)": round(dist_min, 4),
        "Distance Max (km)": round(dist_max, 4),
        "Population Min": int(pop_min),
        "Population Max": int(pop_max),
        "Total Population": int(total_population)
    }

    return pd.DataFrame([summary])
