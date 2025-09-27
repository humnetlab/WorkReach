import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from typing import Dict, Tuple
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


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


def create_accessibility_boxplots(combined_accessibility_df, city_order, figsize=(24, 12)):
    """Create boxplots showing accessibility metrics across cities with sophisticated styling."""
    
    sns.set_theme(context="talk", style="whitegrid")
    plt.rcParams.update({
        "axes.titlesize": 32,
        "axes.labelsize": 30,
        "xtick.labelsize": 30,
        "ytick.labelsize": 30,
        "figure.figsize": figsize
    })
    
    PAL = ["#4C72B0", "#55A868", "#C44E52"]   # low/med/high
    
    def add_box(ax, df, ycol, city, ylabel=False):
        order = ["Low", "High"]
        sns.boxplot(
            data=df, x="group_label", y=ycol, order=order,
            palette=PAL, linewidth=2, saturation=.85, showfliers=False, ax=ax)
        
        for i, grp in enumerate(order):
            med = df.loc[df.group_label == grp, ycol].median()
            ax.text(i, med, f"{med:.2f}", ha="center", va="center",
                    color="white", fontweight="bold", fontsize=26,
                    bbox=dict(boxstyle="round,pad=0.15", fc="0.25", alpha=.8))
        
        ax.set_title(city, pad=4)
        ax.set_xlabel("")
        ax.set_ylabel("Accessibility" if ylabel else "")
        ax.grid(axis="y", linestyle="--", alpha=.65)
        ax.set_facecolor("#F8F9FA")
    
    fig, axes = plt.subplots(
        2, 4,
        figsize=figsize,
        sharex=True,
        sharey='row',
        gridspec_kw={"hspace": 0.45, "wspace": 0.35}
    )
    
    utility_data = combined_accessibility_df[
        combined_accessibility_df["Model"] == "Utility"
    ].copy()
    
    for col, city in enumerate(city_order):
        subset = utility_data.query("City == @city")
        
        try:
            if 'distance_weighted_accessibility' in utility_data.columns:
                add_box(
                    axes[0, col], subset,
                    ycol="distance_weighted_accessibility",
                    city=city,
                    ylabel=True if col == 0 else False)
            
            if 'surplus_accessibility' in utility_data.columns:
                add_box(
                    axes[1, col], subset,
                    ycol="surplus_accessibility",
                    city="",
                    ylabel=True if col == 0 else False)
                    
        except Exception as e:
            print(f"Error plotting accessibility for {city}: {e}")
            axes[0, col].text(0.5, 0.5, f'Error: {str(e)[:20]}...', 
                             ha='center', va='center', transform=axes[0, col].transAxes)
            axes[1, col].text(0.5, 0.5, f'Error: {str(e)[:20]}...', 
                             ha='center', va='center', transform=axes[1, col].transAxes)
    
    fig.text(0.1, 0.97, "a)                         Distance-Weighted Accessibility",
             fontsize=32, fontweight="bold", va="top")
    fig.text(0.1, 0.49, "b)                         Consumer-Surplus Accessibility",
             fontsize=32, fontweight="bold", va="top")
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    return fig

def create_maps_with_histograms(all_city_data, city_order, figsize=(42, 26)):
    """Create maps with accompanying histograms for key variables using sophisticated styling."""
    
    sns.set_theme(context='talk', style='whitegrid')
    plt.rcParams.update({
        'axes.titlesize': 40, 'axes.labelsize': 38,
        'xtick.labelsize': 34, 'ytick.labelsize': 34,
        'legend.fontsize': 40, 'figure.figsize': figsize
    })
    
    CITY_COLORS = dict(zip(city_order, sns.color_palette("tab10", 4)))
    
    def tidy_ticks(ax, x_vals, y_vals, fs=22):
        ax.set_xticks([round(np.nanmin(x_vals), 2), round(np.nanmax(x_vals), 2)])
        ax.set_yticks([0, round(np.max(y_vals), 2)])
        ax.tick_params(axis='both', labelsize=fs)
    
    def city_panel(ax, fig, gdf, value_col, cmap, city_name, bins=15, hist_bottom=0.34):
        vmin, vmax = gdf[value_col].min(), gdf[value_col].max()
        
        gdf.plot(
            column=value_col, cmap=cmap, ax=ax,
            vmin=vmin, vmax=vmax, linewidth=0.001, edgecolor='none',
            rasterized=True,
            missing_kwds=dict(color="lightgrey", label="NA")
        )
        ax.set_axis_off()
        
        cax = inset_axes(ax, width="5%", height="70%", loc='center left',
                         bbox_to_anchor=(1.02, 0., 1, 1),
                         bbox_transform=ax.transAxes, borderpad=0)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []
        cb = plt.colorbar(sm, cax=cax)
        cb.ax.tick_params(labelsize=32)
        cb.outline.set_linewidth(0)
        cb.set_label(value_col.upper() if value_col.lower() == "eci"
                     else value_col.replace('_', ' ').title(), fontsize=36)
        cb.solids.set_edgecolor("none")
        cb.solids.set_linewidth(0)
        cb.solids.set_rasterized(True)
        
        ax_pos = ax.get_position()
        hist_height = 0.09
        hist_width = ax_pos.width
        hist_left = ax_pos.x0
        
        ax_hist = fig.add_axes([hist_left, hist_bottom, hist_width, hist_height])
        vals = gdf[value_col].dropna()
        
        counts, edges = np.histogram(vals, bins=bins)
        bin_centers = 0.5 * (edges[:-1] + edges[1:])
        colormap = plt.get_cmap(cmap)
        
        for i in range(len(counts)):
            color = colormap((bin_centers[i] - vmin) / (vmax - vmin))
            ax_hist.bar(bin_centers[i], counts[i],
                        width=edges[1] - edges[0],
                        color=color, edgecolor='black', align='center', rasterized=True)
        
        for patch in ax_hist.patches:
            patch.set_rasterized(True)
        
        tidy_ticks(ax_hist, edges, counts, fs=32)
        ax_hist.set_xlabel('')
        ax_hist.set_ylabel('Count', labelpad=2, fontsize=34)
        ax_hist.tick_params(axis='both', labelsize=30)
        sns.despine(ax=ax_hist)
    
    fig, axes = plt.subplots(2, 4, figsize=figsize,
                             gridspec_kw={"hspace": 0.6, "wspace": 0.25})
    axes = axes.reshape(2, 4)
    
    cmap_eci, cmap_inf = "viridis", "magma_r"
    hist_y_top = 0.50
    hist_y_bot = 0.03
    
    # ECI
    for col, city in enumerate(city_order):
        try:
            mzn = all_city_data[city]["mzn"]
            city_panel(axes[0, col], fig, mzn, "eci", cmap_eci, city, hist_bottom=hist_y_top)
        except Exception as e:
            print(f"Error plotting ECI for {city}: {e}")
            axes[0, col].text(0.5, 0.5, f'Error: {str(e)[:20]}...', 
                             ha='center', va='center', transform=axes[0, col].transAxes)
    
    # Informality
    for col, city in enumerate(city_order):
        try:
            mzn = all_city_data[city]["mzn"]
            city_panel(axes[1, col], fig, mzn, "informality_rate", cmap_inf, city, hist_bottom=hist_y_bot)
        except Exception as e:
            print(f"Error plotting informality for {city}: {e}")
            axes[1, col].text(0.5, 0.5, f'Error: {str(e)[:20]}...', 
                             ha='center', va='center', transform=axes[1, col].transAxes)
    

    for c, city in enumerate(city_order):
        fig.text(0.17 + c * 0.219, 0.97, city, fontsize=50, fontweight="bold", ha="center", va="top")
    
    for r, (row_label, metric_name) in enumerate([("a)", "ECI"), ("b)", "Informality Rate")]):
        ax_pos = axes[r, 0].get_position()
        row_top_y = ax_pos.y0 + ax_pos.height + 0.02
        fig.text(0.04, row_top_y, f"{row_label}  {metric_name}",
                 fontsize=44, fontweight='bold', va='bottom')
    
    return fig