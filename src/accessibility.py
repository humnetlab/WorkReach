import numpy as np
import pandas as pd
from typing import Optional
from models import compute_utilities

def compute_distance_weighted_accessibility(
    predicted_flows,
    distance_matrix,
    distance_km=False
):
    predicted_flows_rounded = np.round(predicted_flows)

    distance_div = distance_matrix.values.copy()
    if not distance_km:
        distance_div = distance_div / 1000

    distance_div[distance_div < 0.1] = 0.1

    inverse_distance = 1 / distance_div

    weighted_flows = predicted_flows_rounded * inverse_distance

    total_outflows = predicted_flows_rounded.sum(axis=1)
    total_outflows[total_outflows == 0] = 1


    accessibility = weighted_flows.sum(axis=1) / total_outflows

    return accessibility

def compute_accessibility_surplus(
    distance_log,
    eci,
    informality,
    home_population,
    beta_distance,
    beta_eci,
    beta_informality,
    threshold=None,
    k=1.0,
    use_transition_weight=False
):
    """Compute accessibility surplus for the utility-based model."""

    if use_transition_weight and threshold is not None:
        transition_weight = 1 / (1 + np.exp(-k * (distance_log - threshold)))
        utilities = (beta_distance * distance_log +
                    transition_weight * (beta_eci * eci) +
                    transition_weight * (beta_informality * informality))
    else:
        utilities = (beta_distance * distance_log +
                    beta_eci * eci +
                    beta_informality * informality)



    max_u = np.max(utilities, axis=1, keepdims=True)
    exp_utilities = np.exp(utilities - max_u)
    accessibility = np.log(np.sum(exp_utilities, axis=1)) + max_u.flatten()

    return accessibility

def process_accessibility_data(all_city_data, all_models):
    """Process accessibility data for all cities."""
    all_accessibility_dfs = {}

    for city_name, city_data in all_city_data.items():
        print(f"\nCalculating accessibility for {city_name}...")
        city_models = all_models[city_name]


        group_map = {"Low": 0, "Medium": 1, "High": 2}


        aligned_data = city_data["aligned_data"]
        if 'informality_level' not in aligned_data.columns:

            q = aligned_data['informality_rate'].quantile([0.33, 0.66]).values



            eps = 1e-9
            if q[1] <= q[0]:
                q[1] = q[0] + eps

            bins = [-np.inf, q[0], q[1], np.inf]

            aligned_data['informality_level'] = pd.cut(
                aligned_data['informality_rate'],
                bins=bins,
                labels=['Low','Medium','High'],
                include_lowest=True
            )


        home_groups = aligned_data.groupby("home_geomid")["informality_level"].first().map(group_map)


        results = []


        for model_name, (params, predictions, _) in city_models.items():

            distance_weighted = compute_distance_weighted_accessibility(
                predictions, city_data["distance_matrix"], distance_km=False
            )


            for i, geomid in enumerate(city_data["distance_matrix"].index):

                group_value = home_groups.get(geomid, None)


                group_label = None
                if group_value is not None:
                    group_label = {v: k for k, v in group_map.items()}.get(group_value, "Unknown")
                else:

                    group_df = aligned_data[aligned_data["home_geomid"] == geomid]
                    if not group_df.empty:
                        group_label = group_df["informality_level"].iloc[0]
                    else:
                        group_label = "Unknown"

                record = {
                    "City": city_name,
                    "Model": model_name,
                    "geomid": geomid,
                    "home_group": group_value,
                    "group_label": group_label,
                    "distance_weighted_accessibility": distance_weighted[i] if i < len(distance_weighted) else None
                }


                if model_name == "Utility" and i < len(distance_weighted):
                    try:

                        beta_distance, beta_eci, beta_informality, threshold, k = params


                        surplus_accessibility = compute_accessibility_surplus(
                            city_data["distance_log"],
                            city_data["eci"],
                            city_data["informality"],
                            city_data["home_population"],
                            beta_distance, beta_eci, beta_informality,
                            threshold, k, use_transition_weight=True
                        )

                        record["surplus_accessibility"] = surplus_accessibility[i] if i < len(surplus_accessibility) else None
                    except Exception as e:
                        print(f"  Could not calculate surplus accessibility for {city_name}, {model_name}: {e}")

                results.append(record)


        accessibility_df = pd.DataFrame(results)
        all_accessibility_dfs[city_name] = accessibility_df


        utility_accessibility = accessibility_df[accessibility_df["Model"] == "Utility"]
        print(f"\n{city_name} - Utility Model Accessibility Summary by Informality Level:")
        for group in ["Low", "Medium", "High"]:
            group_data = utility_accessibility[utility_accessibility["group_label"] == group]
            if not group_data.empty:
                dist_weighted = group_data["distance_weighted_accessibility"].median()
                surplus = group_data["surplus_accessibility"].median() if "surplus_accessibility" in group_data.columns else "N/A"
                print(f"  {group} Informality: Distance-Weighted = {dist_weighted:.4f}, Surplus = {surplus}")


    combined_accessibility_df = pd.concat(all_accessibility_dfs.values())
    return combined_accessibility_df, all_accessibility_dfs
