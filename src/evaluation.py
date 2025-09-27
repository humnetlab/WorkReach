import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from typing import Dict, Tuple

def common_part_of_commuters(values1: np.ndarray, values2: np.ndarray) -> float:
    """Compute the common part of commuters (Sørensen-Dice coefficient) for two pairs of fluxes."""
    return 2.0 * np.sum(np.minimum(values1, values2)) / (np.sum(values1) + np.sum(values2))

def calculate_metrics(
    flows: np.ndarray,
    predicted_flows: np.ndarray,
    model_name: str,
    num_params: int
) -> Dict[str, float]:
    """Calculate comprehensive evaluation metrics for a model."""

    predicted_flows_rounded = np.round(predicted_flows)

    flows_flat = flows.flatten()
    preds_flat = np.round(predicted_flows_rounded).flatten()


    valid_mask = (~np.isnan(flows_flat) & ~np.isnan(preds_flat))
    flows_flat = flows_flat[valid_mask]
    preds_flat = preds_flat[valid_mask]


    numerator = 2.0 * np.sum(np.minimum(flows_flat, preds_flat))
    denominator = np.sum(flows_flat) + np.sum(preds_flat)
    cpc = numerator / denominator


    corr = pearsonr(flows_flat, preds_flat)[0]


    eps = 1e-12
    log_corr = pearsonr(
        np.log(flows_flat + eps),
        np.log(preds_flat + eps)
    )[0]


    abs_error = np.abs(flows_flat - preds_flat)
    mean_abs_error = np.mean(abs_error)
    median_abs_error = np.median(abs_error)


    nonzero_mask = flows_flat > 0
    if np.any(nonzero_mask):
        rel_error = np.abs((flows_flat[nonzero_mask] - preds_flat[nonzero_mask]) /
                          flows_flat[nonzero_mask])
        mean_rel_error = np.mean(rel_error) * 100
        median_rel_error = np.median(rel_error) * 100
    else:
        mean_rel_error = np.nan
        median_rel_error = np.nan


    log_ratio = np.abs(np.log((preds_flat + eps) / (flows_flat + eps)))
    mean_log_ratio = np.mean(log_ratio)
    median_log_ratio = np.median(log_ratio)


    log_likelihood = np.sum(flows_flat * np.log(preds_flat + eps))


    n = flows.size
    aic = 2 * num_params - 2 * log_likelihood


    bic = -2 * log_likelihood + num_params * np.log(n)


    error_distributions = {
        "abs_error_dist": abs_error.flatten(),
        "rel_error_dist": rel_error if np.any(nonzero_mask) else np.array([]),
        "log_ratio_dist": log_ratio.flatten()
    }

    return {
        "Model": model_name,
        "CPC": cpc,
        "Correlation": corr,
        "Log Correlation": log_corr,
        "MAE": mean_abs_error,
        "Median_AE": median_abs_error,
        "MAPE": mean_rel_error,
        "Median_APE": median_rel_error,
        "Mean_Log_Ratio": mean_log_ratio,
        "Median_Log_Ratio": median_log_ratio,
        "Log-likelihood": log_likelihood,
        "AIC": aic,
        "BIC": bic,
        "Parameters": num_params,
        "Error_Distributions": error_distributions
    }

def compare_models(models_dict: Dict[str, Tuple[np.ndarray, np.ndarray, int]], flows: np.ndarray) -> Tuple[pd.DataFrame, Dict]:
    """Compare different models using several metrics."""
    metrics_list = []
    error_distributions = {}

    for model_name, (params, predictions, num_params) in models_dict.items():
        metrics = calculate_metrics(flows, predictions, model_name, num_params)


        error_distributions[model_name] = metrics.pop("Error_Distributions")

        metrics_list.append(metrics)


    metrics_df = pd.DataFrame(metrics_list)
    metrics_df = metrics_df.sort_values("CPC", ascending=False)

    return metrics_df, error_distributions

def create_utility_model_parameters_table(
    optimized_params_utility: np.ndarray,
    city_name: str = "Unknown City"
) -> pd.DataFrame:
    """Create a wide-format table of parameters for the utility model with substitution rates."""

    beta_distance, beta_eci, beta_informality, log_threshold, k = optimized_params_utility


    threshold_meters = np.exp(log_threshold)


    substitution_rate_eci_distance = beta_eci / beta_distance
    substitution_rate_informality_distance = beta_informality / beta_distance
    substitution_rate_informality_eci = beta_informality / beta_eci


    param_dict = {
        'City': city_name,
        'beta_distance': beta_distance,
        'beta_eci': beta_eci,
        'beta_informality': beta_informality,
        'threshold (log)': log_threshold,
        'threshold [m]': threshold_meters,
        'k': k,
        'ECI/distance': substitution_rate_eci_distance,
        'informality/distance': substitution_rate_informality_distance,
        'informality/ECI': substitution_rate_informality_eci
    }


    return pd.DataFrame([param_dict])
