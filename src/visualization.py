import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from typing import Dict, Tuple
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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

def plot_utility_parameters(utility_params_df):
    param_cols = ['beta_distance', 'beta_eci', 'beta_informality']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    beta_cols = ['beta_distance', 'beta_eci', 'beta_informality']
    beta_df = utility_params_df[beta_cols].copy()

    beta_df.plot(kind='bar', width=0.7, ax=axes[0], color=['#3498db', '#2ecc71', '#e74c3c'])
    axes[0].set_title('Model Coefficients', fontsize=14)
    axes[0].set_ylabel('Coefficient Value', fontsize=12)
    axes[0].set_xlabel('')
    axes[0].set_xticklabels(utility_params_df['City'], rotation=45, ha='right')
    axes[0].legend(labels=['Distance', 'ECI', 'Informality'], fontsize=10)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    ratio_cols = ['ECI/distance', 'informality/distance', 'informality/ECI']
    ratio_df = utility_params_df[ratio_cols].copy()
    ratio_df = ratio_df

    ratio_df.plot(kind='bar', width=0.7, ax=axes[1], color=['#9b59b6', '#f1c40f', '#1abc9c'])
    axes[1].set_title('Substitution Rates', fontsize=14)
    axes[1].set_ylabel('Substitution Rate Value', fontsize=12)
    axes[1].set_xlabel('')
    axes[1].set_xticklabels(utility_params_df['City'], rotation=45, ha='right')
    axes[1].legend(fontsize=10)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    fig.suptitle('Utility Model Parameters Across Cities', fontsize=16, y=1.05)

    plt.tight_layout()

    return fig


def create_accessibility_boxplots(combined_accessibility_df, city_order, figsize=(16, 10)):
    """Create boxplots showing accessibility metrics across cities."""

    utility_data = combined_accessibility_df[combined_accessibility_df["Model"] == "Utility"].copy()

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Accessibility Analysis Across Cities', fontsize=16, y=0.95)

    city_colors = {
        "Bay Area": "#2ca02c",
        "Los Angeles": "#ff7f0e",
        "Mexico City": "#d62728",
        "Rio de Janeiro": "#1f77b4"
    }

    ax1 = axes[0, 0]
    if 'distance_weighted_accessibility' in utility_data.columns:
        sns.boxplot(data=utility_data, x='City', y='distance_weighted_accessibility',
                   order=city_order, palette=[city_colors[city] for city in city_order], ax=ax1)
        ax1.set_title('Distance-Weighted Accessibility')
        ax1.set_xlabel('')
        ax1.tick_params(axis='x', rotation=45)

    ax2 = axes[0, 1]
    if 'surplus_accessibility' in utility_data.columns:
        sns.boxplot(data=utility_data, x='City', y='surplus_accessibility',
                   order=city_order, palette=[city_colors[city] for city in city_order], ax=ax2)
        ax2.set_title('Consumer Surplus Accessibility')
        ax2.set_xlabel('')
        ax2.tick_params(axis='x', rotation=45)

    ax3 = axes[1, 0]
    if 'distance_weighted_accessibility' in utility_data.columns and 'group_label' in utility_data.columns:
        sns.boxplot(data=utility_data, x='group_label', y='distance_weighted_accessibility',
                   hue='City', hue_order=city_order, ax=ax3)
        ax3.set_title('Distance-Weighted by Informality Level')
        ax3.set_xlabel('Informality Level')
        ax3.legend(title='City', bbox_to_anchor=(1.05, 1), loc='upper left')

    ax4 = axes[1, 1]
    if 'surplus_accessibility' in utility_data.columns and 'group_label' in utility_data.columns:
        sns.boxplot(data=utility_data, x='group_label', y='surplus_accessibility',
                   hue='City', hue_order=city_order, ax=ax4)
        ax4.set_title('Consumer Surplus by Informality Level')
        ax4.set_xlabel('Informality Level')
        ax4.legend(title='City', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    return fig

def create_maps_with_histograms(all_city_data, city_order, figsize=(24, 18)):
    """Create maps with accompanying histograms for key variables."""

    variables = ['eci', 'informality_rate', 'population']
    var_labels = ['Economic Complexity Index', 'Informality Rate', 'Population']

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(len(variables), len(city_order) * 2,
                  width_ratios=[3, 1] * len(city_order),
                  hspace=0.3, wspace=0.15)

    cmaps = ['viridis', 'plasma', 'cividis']

    for var_idx, (variable, var_label, cmap) in enumerate(zip(variables, var_labels, cmaps)):
        for city_idx, city in enumerate(city_order):

            ax_map = fig.add_subplot(gs[var_idx, city_idx * 2])

            ax_hist = fig.add_subplot(gs[var_idx, city_idx * 2 + 1])

            try:
                mzn = all_city_data[city]["mzn"]

                if variable == 'eci':
                    plot_data = np.exp(mzn[variable])
                    hist_data = plot_data
                elif variable == 'population':
                    plot_data = mzn[variable]
                    hist_data = np.log10(plot_data + 1)
                else:
                    plot_data = mzn[variable]
                    hist_data = plot_data

                mzn_plot = mzn.copy()
                mzn_plot['plot_var'] = plot_data

                mzn_plot = mzn_plot[np.isfinite(mzn_plot['plot_var'])]

                if len(mzn_plot) > 0:
                    mzn_plot.plot(column='plot_var', cmap=cmap, ax=ax_map,
                                 linewidth=0.1, edgecolor='white')
                    ax_map.set_axis_off()

                    ax_hist.hist(hist_data.dropna(), bins=30, alpha=0.7,
                               color=plt.cm.get_cmap(cmap)(0.6), edgecolor='black', linewidth=0.5)
                    ax_hist.set_ylabel('Frequency')

                    if variable == 'population':
                        ax_hist.set_xlabel('Log10(Population + 1)')
                    else:
                        ax_hist.set_xlabel(var_label)

                    if var_idx == 0:
                        ax_map.set_title(city, fontsize=14, fontweight='bold')

                else:
                    ax_map.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax_map.transAxes)
                    ax_hist.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax_hist.transAxes)

            except Exception as e:
                print(f"Error plotting {variable} for {city}: {e}")
                ax_map.text(0.5, 0.5, f'Error: {str(e)[:20]}...',
                           ha='center', va='center', transform=ax_map.transAxes)
                ax_hist.text(0.5, 0.5, 'Error', ha='center', va='center', transform=ax_hist.transAxes)


        fig.text(0.02, 0.85 - var_idx * 0.28, var_label,
                fontsize=14, fontweight='bold', rotation=90, va='center')

    return fig
