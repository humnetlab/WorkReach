import sys

sys.path.append("..")

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import stats
from scipy.stats import gaussian_kde, lognorm, pearsonr
import seaborn as sns
import networkx as nx
import matplotlib.patheffects as PathEffects
import matplotlib.colors as mcolors
from matplotlib.colors import to_rgba
from matplotlib.collections import LineCollection

from evaluation import common_part_of_commuters
import mapclassify
from pysal.lib import weights
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

from visualization import project_for_mapping, add_scale_bar, ECI_LABEL

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"]})

DIST_COLORS = ["#2E86AB", "#A23B72", "#F18F01"]

def plot_spatial_analysis(gdf_city, variable_col, city_name, variable_display_name):
    """Three-panel plot: variable choropleth, spatial lag, Moran scatter."""
    gdf_valid = gdf_city.dropna(subset=[variable_col]).copy()
    gdf_valid = gdf_valid[gdf_valid.geometry.is_valid & ~gdf_valid.geometry.is_empty]
    gdf_valid = gdf_valid.reset_index(drop=True)

    w = weights.KNN.from_dataframe(gdf_valid, k=8)
    w.transform = "R"

    data_col = gdf_valid[variable_col].values
    data_mean = data_col.mean()
    data_std_dev = data_col.std()
    if data_std_dev == 0:
        var_std = np.zeros_like(data_col)
    else:
        var_std = (data_col - data_mean) / data_std_dev

    var_lag = weights.lag_spatial(w, data_col)
    var_lag_std = weights.lag_spatial(w, var_std)

    gdf_valid[f"{variable_col}_lag"] = var_lag
    gdf_valid[f"{variable_col}_std"] = var_std
    gdf_valid[f"{variable_col}_lag_std"] = var_lag_std

    sns.set_theme(context="talk", style="whitegrid")
    plt.rcParams.update({
        'font.size': 28, 'axes.titlesize': 32, 'axes.labelsize': 30,
        'xtick.labelsize': 28, 'ytick.labelsize': 28,
        'legend.fontsize': 28, 'figure.titlesize': 36
    })

    gdf_plot = project_for_mapping(gdf_valid, city_name)

    fig = plt.figure(figsize=(38, 10))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.5)

    # --- Panel A: Variable map ---
    ax1 = fig.add_subplot(gs[0, 0])
    plot_data_A = gdf_plot[[variable_col, 'geometry']].dropna(subset=[variable_col])
    q1 = mapclassify.Quantiles(plot_data_A[variable_col], k=10)
    bounds1 = q1.bins
    colors1 = plt.cm.viridis(np.linspace(0, 1, 10))
    cmap1 = mpl.colors.ListedColormap(colors1)
    norm1 = mpl.colors.BoundaryNorm(bounds1, cmap1.N)
    sm1 = plt.cm.ScalarMappable(cmap=cmap1, norm=norm1)
    sm1.set_array([])
    plot_data_A.plot(column=variable_col, cmap="viridis", scheme="quantiles", k=10,
                     edgecolor="none", linewidth=0.1, alpha=0.85, legend=False, ax=ax1)
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="6%", pad=0.25)
    cbar1 = plt.colorbar(sm1, cax=cax1, spacing='uniform')
    tick_idx = np.linspace(0, len(bounds1)-2, 4, dtype=int)
    tick_pos = [(bounds1[i] + bounds1[i+1])/2 for i in tick_idx]
    tick_lab = [f'{bounds1[i]:.2f}-\n{bounds1[i+1]:.2f}' for i in tick_idx]
    cbar1.set_ticks(tick_pos); cbar1.set_ticklabels(tick_lab, fontsize=23, ha='left')
    cbar1.set_label(variable_display_name, fontsize=28, labelpad=30)
    cbar1.outline.set_linewidth(0)
    ax1.set_axis_off()
    add_scale_bar(ax1, city_name, font_size=24, below=False)

    # --- Panel B: Spatial lag map ---
    ax2 = fig.add_subplot(gs[0, 1])
    lag_col = f"{variable_col}_lag"
    plot_data_B = gdf_plot[[lag_col, 'geometry']].dropna(subset=[lag_col])
    q2 = mapclassify.Quantiles(plot_data_B[lag_col], k=10)
    bounds2 = q2.bins
    colors2 = plt.cm.viridis(np.linspace(0, 1, 10))
    cmap2 = mpl.colors.ListedColormap(colors2)
    norm2 = mpl.colors.BoundaryNorm(bounds2, cmap2.N)
    sm2 = plt.cm.ScalarMappable(cmap=cmap2, norm=norm2)
    sm2.set_array([])
    plot_data_B.plot(column=lag_col, cmap="viridis", scheme="quantiles", k=10,
                     edgecolor="none", linewidth=0.1, alpha=0.85, legend=False, ax=ax2)
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="6%", pad=0.25)
    cbar2 = plt.colorbar(sm2, cax=cax2, spacing='uniform')
    tick_idx2 = np.linspace(0, len(bounds2)-2, 4, dtype=int)
    tick_pos2 = [(bounds2[i] + bounds2[i+1])/2 for i in tick_idx2]
    tick_lab2 = [f'{bounds2[i]:.2f}-\n{bounds2[i+1]:.2f}' for i in tick_idx2]
    cbar2.set_ticks(tick_pos2); cbar2.set_ticklabels(tick_lab2, fontsize=23, ha='left')
    cbar2.set_label(f"{variable_display_name} (Lag)", fontsize=28, labelpad=30)
    cbar2.outline.set_linewidth(0)
    ax2.set_axis_off()
    add_scale_bar(ax2, city_name, font_size=24, below=False)

    # --- Panel C: Moran scatter ---
    ax3 = fig.add_subplot(gs[0, 2])
    x_data = gdf_valid[f"{variable_col}_std"].dropna()
    y_data = gdf_valid[f"{variable_col}_lag_std"].dropna()

    max_abs = max(abs(x_data.min()), abs(x_data.max()),
                  abs(y_data.min()), abs(y_data.max()))
    lim = max_abs * 1.2
    xlim, ylim = [-lim, lim], [-lim, lim]

    ax3.fill_between([0, xlim[1]], 0, ylim[1], alpha=0.15, color='lightblue', zorder=0)
    ax3.fill_between([xlim[0], 0], 0, ylim[1], alpha=0.15, color='lightcoral', zorder=0)
    ax3.fill_between([xlim[0], 0], ylim[0], 0, alpha=0.15, color='lightblue', zorder=0)
    ax3.fill_between([0, xlim[1]], ylim[0], 0, alpha=0.15, color='lightcoral', zorder=0)

    ax3.scatter(x_data, y_data, alpha=0.7, s=50, c='#2E86AB', edgecolors='none', zorder=5)
    slope, intercept, *_ = stats.linregress(x_data, y_data)
    line_x = np.linspace(xlim[0], xlim[1], 100)
    ax3.plot(line_x, slope * line_x + intercept, color='#C5282F', lw=4, alpha=0.9, zorder=4)
    ax3.axvline(0, color='black', alpha=0.4, lw=2, zorder=3)
    ax3.axhline(0, color='black', alpha=0.4, lw=2, zorder=3)
    ax3.set_xlim(xlim); ax3.set_ylim(ylim)
    ax3.set_aspect('equal')
    ax3.set_xlabel(variable_display_name, fontsize=30, fontweight='bold')
    ax3.set_ylabel(f"{variable_display_name} (Lag)", fontsize=30, fontweight='bold')
    ax3.grid(True, alpha=0.3, lw=0.5, zorder=1)
    ax3.set_facecolor('white')

    for label, xp, yp, c, edge in [('HH', 0.95, 0.95, 'lightblue', 'navy'),
                                     ('LH', 0.05, 0.95, 'lightcoral', 'darkred'),
                                     ('LL', 0.05, 0.05, 'lightblue', 'navy'),
                                     ('HL', 0.95, 0.05, 'lightcoral', 'darkred')]:
        ax3.text(xp, yp, label, transform=ax3.transAxes, fontsize=34, fontweight='bold',
                 ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor=c, alpha=0.9,
                           edgecolor=edge, linewidth=2), zorder=6)
    ax3.tick_params(axis='both', which='major', labelsize=26, width=2)

    common_title_y = 0.890
    for ax, title in [(ax1, f"a) {variable_display_name}"),
                      (ax2, f"b) {variable_display_name} (spatial lag)"),
                      (ax3, f"c) Moran plot, Moran's I: {slope:.3f}")]:
        pos = ax.get_position()
        fig.text(pos.x0, common_title_y, title, fontsize=32, fontweight='bold',
                 ha='left', va='bottom')

    fig.suptitle(f"Spatial Autocorrelation of {variable_display_name} - {city_name}",
                 fontsize=36, fontweight='bold', y=1.00)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    return fig, slope


def plot_city_distributions(city, prep, flows_df, gdf):
    """Plot distributions of distance, ECI, informality after scaling."""
    distance_vals = flows_df.query("distance_home_to_work > 0")["distance_home_to_work"].values
    eci_vals = gdf["eci"].dropna().values
    informality_vals = gdf.query("informality_rate > 0")["informality_rate"].dropna().values

    d_scaled = (distance_vals - distance_vals.min()) / (distance_vals.max() - distance_vals.min())
    eci_scaled = (eci_vals - eci_vals.min()) / (eci_vals.max() - eci_vals.min())
    inf_scaled = (informality_vals - informality_vals.min()) / (informality_vals.max() - informality_vals.min())

    datasets = [d_scaled, eci_scaled, inf_scaled]
    kde_funcs = [gaussian_kde(d) for d in datasets]
    x = np.linspace(0, 1, 200)

    lognorm_params = []
    for data in datasets:
        data_adj = data + 1e-10
        s, loc, scale = lognorm.fit(data_adj, floc=0)
        lognorm_params.append((np.log(scale), s))

    plt.style.use('default')
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"]})
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    for i, (ax, kde, color, title, xlabel, (mu, sigma)) in enumerate(zip(
        axes, kde_funcs, DIST_COLORS,
        ["Distance", "Economic complexity", "Informality rate"],
        ["Scaled distance", r"Scaled $\mathrm{ECI}^{\mathrm{emp}}$", "Scaled informality"],
        lognorm_params
    )):
        ax.plot(x, kde(x), color=color, linewidth=4, alpha=0.9)
        ax.fill_between(x, kde(x), alpha=0.2, color=color)
        ax.set_title(title, fontsize=28, fontweight='bold', pad=25)
        ax.text(-0.10, 1.06, f"{chr(97 + i)})", transform=ax.transAxes,
                fontsize=30, fontweight='bold', va='bottom', ha='left')
        ax.set_xlabel(xlabel, fontsize=24, fontweight='semibold')
        if i == 0:
            ax.set_ylabel("Density", fontsize=24, fontweight='semibold')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.tick_params(axis='both', which='major', labelsize=20, width=1.5, length=6)
        ax.set_xlim(-0.05, 1)
        ax.set_ylim(0, None)
        # ax.text(0.95, 0.95, f'$\\mu$ = {mu:.3f}\n$\\sigma$ = {sigma:.3f}',
        #         transform=ax.transAxes, fontsize=28, ha='right', va='top',
        #         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9,
        #                   edgecolor=color))

    fig.suptitle(city, fontsize=32, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.subplots_adjust(top=0.80, bottom=0.15)
    return fig


def plot_transition_weights(city_order, params_by_city, scaling_by_city):
    """Transition weight against distance for each city."""
    mpl.rcParams.update({
        "text.usetex": False, "font.size": 18,
        "axes.titlesize": 22, "axes.labelsize": 20,
        "xtick.labelsize": 16, "ytick.labelsize": 16,
        "legend.fontsize": 18, "figure.titlesize": 24
    })

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharey=True)
    d = np.linspace(0, 1, 500)

    for i, (ax, city) in enumerate(zip(axes.flatten(), city_order)):
        tau, k = params_by_city[city][3], params_by_city[city][4]
        diff, dmin = scaling_by_city[city]

        tau_km = (tau * diff + dmin) / 1000
        d_km = (d * diff + dmin) / 1000
        w = 1.0 / (1.0 + np.exp(-k * (d - tau)))

        ax.plot(d_km, w, color="#9b59b6", lw=3)
        ax.axvline(tau_km, color="gray", linestyle="--")
        ax.annotate(f"$\\tau$ = {tau_km:.3f}\nk = {k:.3f}",
                    xy=(0.65, 0.25), xycoords="axes fraction", fontsize=22,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              alpha=0.8, edgecolor="gray"))

        ax.set_title(city, fontweight="bold", pad=15)
        ax.text(-0.14, 1.06, f"{chr(97 + i)})", transform=ax.transAxes,
                fontsize=26, fontweight="bold", va="bottom", ha="left")
        ax.set_xlabel(r"Distance $d_{ij}$ [km]")
        ax.set_ylabel(r"$w_{ij}(\tau, k)$")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xlim(dmin / 1000, (diff + dmin) / 1000)

        secax = ax.secondary_xaxis("top")
        secax.set_xlabel(r"Standardized distance $d_{ij}$")
        secax.set_xticks([(p * diff + dmin) / 1000 for p in [0, 0.25, 0.5, 0.75, 1.0]])
        secax.set_xticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"])
        secax.tick_params(labelsize=14)
        secax.set_xlim(ax.get_xlim())

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.suptitle(r"Transition function: $w_{ij}(\tau, k)$", y=0.975)
    return fig


PAL = ['#3498db', '#2ecc71']
QUAD_COLORS = {
    'high_high': '#c8e6c9', 'low_high': '#b3e5fc',
    'high_low': '#ffe0b2', 'low_low': '#ffcdd2',
}

def add_scatter(ax, df, ylabel=False):
    colors = [PAL[0] if g == "Low" else PAL[1] for g in df['group_label']]
    ax.scatter(df['distance_weighted_accessibility'],
               df['surplus_accessibility'], c=colors, alpha=0.5, s=30, zorder=2)

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    med_x = df['distance_weighted_accessibility'].median()
    med_y = df['surplus_accessibility'].median()

    ax.fill_between([med_x, x_max], med_y, y_max, color=QUAD_COLORS['high_high'], alpha=0.7, zorder=0)
    ax.fill_between([x_min, med_x], med_y, y_max, color=QUAD_COLORS['low_high'], alpha=0.7, zorder=0)
    ax.fill_between([med_x, x_max], y_min, med_y, color=QUAD_COLORS['high_low'], alpha=0.7, zorder=0)
    ax.fill_between([x_min, med_x], y_min, med_y, color=QUAD_COLORS['low_low'], alpha=0.7, zorder=0)
    ax.axvline(med_x, color='gray', ls='--', alpha=0.7, zorder=1)
    ax.axhline(med_y, color='gray', ls='--', alpha=0.7, zorder=1)
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)

    rho = df['distance_weighted_accessibility'].corr(df['surplus_accessibility'])
    ax.annotate(f"$r = {rho:.2f}$", xy=(0.67, 0.95), xycoords='axes fraction',
                fontsize=23, bbox=dict(boxstyle="round", fc="w"))
    ax.set_xlabel("DW")
    if ylabel:
        ax.set_ylabel("CS")
    ax.grid(True, ls="--", alpha=.3)
    ax.set_facecolor("#F8F9FA")


def plot_accessibility_scatters(combined_accessibility_df, city_order):
    """Distance-weighted against consumer surplus accessibility, by city."""
    fig, axes = plt.subplots(1, 4, figsize=(24, 8), gridspec_kw={"wspace": 0.35})

    for idx, city in enumerate(city_order):
        subset = combined_accessibility_df.query("City == @city and Model == 'Utility'")
        add_scatter(axes[idx], subset, ylabel=(idx == 0))
        axes[idx].set_title(city, fontsize=32)
        axes[idx].text(-0.12, 1.06, f"{chr(97 + idx)})", transform=axes[idx].transAxes,
                       fontsize=30, fontweight="bold", va="bottom", ha="left")

    quad_legend = [
        Rectangle((0, 0), 1, 1, facecolor=QUAD_COLORS['high_high'], label='High DW, high CS'),
        Rectangle((0, 0), 1, 1, facecolor=QUAD_COLORS['high_low'], label='High DW, low CS'),
        Rectangle((0, 0), 1, 1, facecolor=QUAD_COLORS['low_high'], label='Low DW, high CS'),
        Rectangle((0, 0), 1, 1, facecolor=QUAD_COLORS['low_low'], label='Low DW, low CS'),
    ]
    inf_legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=PAL[0], markersize=22,
               label='Low informality'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=PAL[1], markersize=22,
               label='High informality'),
    ]

    fig.subplots_adjust(bottom=0.35)
    fig.legend(handles=quad_legend, title="Accessibility quadrants", loc='lower center',
               bbox_to_anchor=(0.30, -0.06), ncol=2, fontsize=28, title_fontsize=30, frameon=True)
    fig.legend(handles=inf_legend, title="Informality", loc='lower center',
               bbox_to_anchor=(0.72, -0.04), ncol=2, fontsize=28, title_fontsize=30, frameon=True)
    return fig


CITY_COLORS = {"Bay Area": "#2166ac", "Los Angeles": "#d6604d",
               "Mexico City": "#b35806", "Rio de Janeiro": "#4dac26"}

def _format_p(p):
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"

def plot_eci_informality_scatter(zone_data, city_order):
    """Zone-level ECI against residential informality, by city."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()

    for i, (ax, city) in enumerate(zip(axes, city_order)):
        gdf = zone_data[city].dropna(subset=["eci", "informality_rate"])
        eci = gdf["eci"].values
        informality = gdf["informality_rate"].values
        r, p = pearsonr(eci, informality)

        ax.scatter(eci, informality, alpha=0.35, s=14,
                   color=CITY_COLORS[city], rasterized=True)
        slope, intercept, *_ = stats.linregress(eci, informality)
        x_line = np.linspace(eci.min(), eci.max(), 200)
        ax.plot(x_line, slope * x_line + intercept, color="k", lw=1.5, ls="--")

        ax.set_xlabel(r"$\mathrm{ECI}^{\mathrm{emp}}$", fontsize=22)
        ax.set_ylabel("Informality rate", fontsize=22)
        ax.set_title(city, fontsize=22, fontweight="bold")
        ax.tick_params(labelsize=20)
        ax.text(-0.24, 1.08, f"{chr(97 + i)})", transform=ax.transAxes,
                fontsize=26, fontweight="bold", va="bottom", ha="left")

        side = "right" if r < 0 else "left"
        ax.text(0.97 if r < 0 else 0.03, 0.97,
                f"$r$ = {r:.2f} ($p$ {_format_p(p)})", transform=ax.transAxes,
                ha=side, va="top", fontsize=17,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9))

    fig.suptitle(r"$\mathrm{ECI}^{\mathrm{emp}}$ against informality rate", fontsize=24, y=1.01)
    fig.tight_layout()
    return fig


MODEL_COLORS = {"WorkReach": "#9b59b6", "Gravity": "#f39c12",
                "Radiation Ext": "#1abc9c", "BMS Plausible": "#34495e"}


def _row_letter(ax, index, x=-0.30, y=1.02):
    ax.text(x, y, f"{chr(97 + index)})", transform=ax.transAxes,
            fontsize=28, fontweight="bold", va="bottom", ha="left")


def plot_error_correlation_heatmaps(corr_tables, city_order):
    """Pairwise correlation between the residuals of every pair of models."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))

    for i, city in enumerate(city_order):
        ax = axes[i // 2, i % 2]
        corr = corr_tables[city]
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r",
                    vmin=0.5, vmax=1.0, square=True, mask=mask, cbar=False,
                    linewidths=0.5, ax=ax, annot_kws={"size": 16})
        ax.set_title(city, fontsize=22, fontweight="bold")
        ax.tick_params(axis="both", labelsize=16)
        _row_letter(ax, i, x=-0.22, y=1.06)

    plt.tight_layout(rect=[0, 0, 0.86, 1])

    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    mappable = plt.cm.ScalarMappable(cmap="RdBu_r",
                                     norm=plt.Normalize(vmin=0.5, vmax=1.0))
    mappable.set_array([])
    cbar = fig.colorbar(mappable, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label("$r$", fontsize=22)
    return fig


def plot_error_correlation_scatter(residuals, city_order, benchmarks):
    """WorkReach residuals against those of each benchmark, one row per city."""
    fig, axes = plt.subplots(len(city_order), len(benchmarks), figsize=(15, 18))

    for i, city in enumerate(city_order):
        workreach_residual = residuals[city]["WorkReach"]

        for j, benchmark in enumerate(benchmarks):
            ax = axes[i, j]
            benchmark_residual = residuals[city][benchmark]
            r = np.corrcoef(workreach_residual, benchmark_residual)[0, 1]

            if len(workreach_residual) > 10000:
                idx = np.random.default_rng(42).choice(len(workreach_residual),
                                                       10000, replace=False)
                x_plot, y_plot = benchmark_residual[idx], workreach_residual[idx]
            else:
                x_plot, y_plot = benchmark_residual, workreach_residual

            ax.scatter(x_plot, y_plot, s=3, alpha=0.05,
                       color=MODEL_COLORS.get(benchmark, "gray"), rasterized=True)

            low = min(x_plot.min(), y_plot.min())
            high = max(x_plot.max(), y_plot.max())
            ax.plot([low, high], [low, high], "k--", alpha=0.4, linewidth=0.8)
            ax.set_xlim(low, high)
            ax.set_ylim(low, high)
            ax.set_aspect("equal")
            ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
            ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)

            ax.text(0.05, 0.95, f"$r$ = {r:.3f}", transform=ax.transAxes,
                    fontsize=22, va="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            ax.tick_params(axis="both", which="major", labelsize=22)

            if i == 0:
                ax.set_title(benchmark,
                             fontsize=26, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{city}\nWorkReach residual", fontsize=22)
                _row_letter(ax, i)
            if i == len(city_order) - 1:
                ax.set_xlabel(f"{benchmark} residual",
                              fontsize=22)

    plt.tight_layout()
    return fig


def plot_oos_scatter(oos_predictions, city_flows, city_order, model_order):
    """Observed against out-of-sample predicted flows, stitched across folds."""
    fig, axes = plt.subplots(len(city_order), len(model_order),
                             figsize=(16, 16), sharex=True, sharey=True)

    for i, city in enumerate(city_order):
        observed = city_flows[city].flatten()

        for j, model in enumerate(model_order):
            ax = axes[i, j]
            predicted = np.round(oos_predictions[city][model]).flatten()

            mask = ~np.isnan(observed) & ~np.isnan(predicted)
            obs_plot, pred_plot = observed[mask], predicted[mask]

            r = np.corrcoef(obs_plot, pred_plot)[0, 1]
            cpc = common_part_of_commuters(obs_plot, pred_plot)

            ax.scatter(obs_plot, pred_plot, s=10, alpha=0.01,
                       color=MODEL_COLORS.get(model, "blue"), rasterized=True)

            max_val = max(np.max(obs_plot), np.max(pred_plot))
            ax.plot([0, max_val], [0, max_val], "r--", alpha=0.88)

            ax.text(0.05, 0.95, f"$r$ = {r:.2f}\nCPC = {cpc:.2f}",
                    transform=ax.transAxes, fontsize=16, verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

            if i == 0:
                ax.set_title(model, fontsize=22)
            if j == 0:
                ax.set_ylabel(f"{city}\nPredicted flows", fontsize=20)
                _row_letter(ax, i, x=-0.34)
            if i == len(city_order) - 1:
                ax.set_xlabel("Observed flows", fontsize=20)

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(1, max_val * 1.1)
            ax.set_ylim(1, max_val * 1.1)
            ax.tick_params(axis="both", labelsize=18)

    plt.tight_layout()
    return fig


def plot_train_test_performance(results_df, city_order, model_names):
    """Fold-averaged train and test performance for every model and city."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharey="row")

    for col, city in enumerate(city_order):
        city_df = results_df[results_df["City"] == city]

        for row, metric in enumerate(["CPC", "r"]):
            ax = axes[row, col]
            means = city_df.groupby("Model").agg(
                train_mean=(f"Train {metric}", "mean"),
                train_std=(f"Train {metric}", "std"),
                test_mean=(f"Test {metric}", "mean"),
                test_std=(f"Test {metric}", "std"),
            ).loc[model_names]

            x = np.arange(len(model_names))
            width = 0.35
            ax.bar(x - width / 2, means["train_mean"], width,
                   yerr=means["train_std"], capsize=3,
                   label="Train", color="steelblue", alpha=0.8)
            ax.bar(x + width / 2, means["test_mean"], width,
                   yerr=means["test_std"], capsize=3,
                   label="Test", color="coral", alpha=0.8)

            ax.set_xticks(x)
            ax.set_xticklabels([m.replace(" ", "\n") for m in model_names],
                               fontsize=18, rotation=45)
            ax.tick_params(axis="y", labelsize=18)
            if col == 0:
                ax.set_ylabel(metric, fontsize=20)
            if row == 0:
                ax.set_title(city, fontsize=22, fontweight="bold")
                ax.text(-0.10, 1.16, f"{chr(97 + col)})", transform=ax.transAxes,
                        fontsize=28, fontweight="bold", va="bottom", ha="left")
            if row == 0 and col == 0:
                ax.legend(fontsize=18, loc="upper left")

    plt.tight_layout()
    return fig


FLOW_COLORS = [
    (0.0, "#d4e5ff"),
    (0.4, "#4361ee"),
    (0.6, "#7209b7"),
    (0.8, "#d00000"),
    (1.0, "#ffbe0b"),
]


def flow_colormap():
    """Light blue through blue, purple and red to yellow."""
    return mcolors.LinearSegmentedColormap.from_list("flow", FLOW_COLORS)


def plot_flow_map(flow_df, gdf_indexed, ref_min, ref_max, city,
                  figsize=(12, 12), background="#EEEEEE"):
    """Origin-destination flows drawn between zone centroids, on a metric CRS."""
    cmap = flow_colormap()
    norm = mcolors.LogNorm(vmin=max(ref_min, 1), vmax=ref_max)

    fig, ax = plt.subplots(figsize=figsize)
    gdf_indexed.plot(ax=ax, color=background, edgecolor="#888888",
                     linewidth=0.4, alpha=0.6)

    positive = flow_df.loc[flow_df.flows > 0].sort_values("flows")
    flows = positive["flows"].to_numpy(dtype=float)
    centroids = gdf_indexed.geometry.centroid
    origins = centroids.loc[positive.home_geomid]
    destinations = centroids.loc[positive.work_geomid]
    segments = np.stack([np.column_stack([origins.x, origins.y]),
                         np.column_stack([destinations.x, destinations.y])], axis=1)

    log_flows = np.log10(flows)
    scaled = (log_flows - log_flows.min()) / (log_flows.max() - log_flows.min())
    curved = scaled ** 2

    colors = [to_rgba(cmap(norm(f)), alpha=a)
              for f, a in zip(flows, 0.1 + 0.7 * curved)]
    ax.add_collection(LineCollection(segments, colors=colors,
                                     linewidths=0.05 + 2.5 * curved,
                                     zorder=5, rasterized=True))

    ax.set_axis_off()

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cax = inset_axes(ax, width="55%", height="3.3%", loc="lower left", borderpad=0.9)
    colorbar = plt.colorbar(mappable, cax=cax, orientation="horizontal")
    colorbar.set_label("$T_{od}$", fontsize=38)
    colorbar.ax.tick_params(labelsize=36)

    add_scale_bar(ax, city, font_size=30, offset=0.93)
    plt.tight_layout()
    ax.set_position([0.05, 0.15, 0.9, 0.75])
    cax.set_position([0.1, 0.05, 0.8, 0.04])
    return fig


def assemble_unified_figure(city, panels):
    """Lay the observed map and the four model maps out as a single figure."""
    fig = plt.figure(figsize=(18, 12))
    grid = GridSpec(2, 6, figure=fig, height_ratios=[1, 1], hspace=0.05, wspace=0.05)
    positions = [grid[0, 0:2], grid[0, 2:4], grid[0, 4:6], grid[1, 1:3], grid[1, 3:5]]

    for index, (position, (name, image)) in enumerate(zip(positions, panels.items())):
        ax = fig.add_subplot(position)
        ax.imshow(image, aspect="auto")
        ax.axis("off")
        ax.text(0.02, 0.98, f"{chr(97 + index)})", transform=ax.transAxes,
                fontsize=20, fontweight="bold", ha="left", va="top")
        ax.set_title(name, fontsize=16, fontweight="bold", pad=10)

    fig.suptitle(f"{city} - Flow Maps Comparison", fontsize=24, fontweight="bold", y=0.95)
    return fig


def plot_accessibility_vs_informality(zone_data, z_scores, city_order, row_info):
    """Each accessibility measure against the informality rate of the home zone."""
    fig, axes = plt.subplots(3, 4, figsize=(24, 18),
                             gridspec_kw={"hspace": 0.25, "wspace": 0.12})

    for c, city in enumerate(city_order):
        fig.text(0.20 + c * 0.21, 0.955, city, fontsize=38,
                 fontweight="bold", ha="center", va="top")

    for r, (metric, _) in enumerate(row_info):
        for c, city in enumerate(city_order):
            ax = axes[r, c]
            city_zones = zone_data[city].merge(
                z_scores[z_scores.City == city][["geomid", metric]],
                on="geomid", how="left")

            ax.scatter(city_zones["informality_rate"], city_zones[metric], alpha=0.5)
            if r == 2:
                ax.set_xlabel("Informality rate", fontsize=30)
            else:
                ax.set_xticklabels([])
            if c == 0:
                ax.set_ylabel("Accessibility\n($z$-score)", fontsize=30)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=28)

            valid = city_zones.dropna(subset=[metric])
            if len(valid) > 2:
                r_value, _ = pearsonr(valid["informality_rate"], valid[metric])
                ax.text(0.95, 0.95, f"$r = {r_value:.2f}$", transform=ax.transAxes,
                        fontsize=28, va="top", ha="right",
                        bbox=dict(facecolor="white", edgecolor="black"))

    for r, (_, row_label) in enumerate(row_info):
        position = axes[r, 0].get_position()
        fig.text(0.07, position.y0 + position.height, row_label,
                 fontsize=34, fontweight="bold", va="bottom")

    plt.tight_layout(rect=[0.06, 0.01, 0.98, 0.91])
    return fig


def plot_rca_matrices(rca_matrices, city_order):
    """Binary RCA matrices sorted by diversity and ubiquity, showing the staircase."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    binary = mcolors.ListedColormap(["#f7f7f7", "#2166ac"])

    for i, (ax, city) in enumerate(zip(axes.flatten(), city_order)):
        matrix = rca_matrices[city]
        rows = matrix.sum(axis=1).sort_values(ascending=False).index
        columns = matrix.sum(axis=0).sort_values(ascending=False).index

        ax.imshow(matrix.loc[rows, columns].values, aspect="auto", cmap=binary,
                  interpolation="nearest", vmin=0, vmax=1)
        ax.set_xticks(range(len(columns)))
        ax.set_xticklabels(columns, rotation=45, ha="right", fontsize=13)
        ax.set_yticks([])
        ax.set_xlabel("Sector (sorted by ubiquity)", fontsize=16)
        ax.set_ylabel("Zone (sorted by diversity)", fontsize=16)
        ax.set_title(city, fontsize=17, fontweight="bold")
        ax.text(-0.08, 1.08, f"{chr(97 + i)})", transform=ax.transAxes,
                fontsize=24, fontweight="bold", va="bottom", ha="left")

    fig.suptitle("Zone-sector network", fontsize=20, y=1.01, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_product_space_grid(city_order, proximity_data, pci_data, employment_data,
                            graphs):
    """Product space of each city, with sectors coloured by complexity."""
    sns.set_theme(context="talk", style="white")
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                         "axes.titlesize": 22, "axes.labelsize": 20,
                         "xtick.labelsize": 16, "ytick.labelsize": 16})

    fig, axes = plt.subplots(2, 2, figsize=(24, 22))

    for i, (ax, city) in enumerate(zip(axes.flat, city_order)):
        proximity = proximity_data[city]
        pci = pci_data[city]
        workers = employment_data[city].groupby("naics")["workers"].sum()
        sectors = list(proximity.columns)

        full = nx.Graph()
        full.add_nodes_from(sectors)
        for a, first in enumerate(sectors):
            for second in sectors[a + 1:]:
                if proximity.loc[first, second] > 0:
                    full.add_edge(first, second, weight=proximity.loc[first, second])
        positions = nx.spring_layout(full, weight="weight", seed=42, k=0.4)

        values = np.array([pci.get(s, 0.0) for s in sectors])
        norm = mpl.colors.Normalize(vmin=values.min(), vmax=values.max())
        sizes = np.array([np.sqrt(workers.get(s, 0)) * 4 + 120 for s in sectors])

        graph = graphs[city]
        nx.draw_networkx_edges(graph, positions, ax=ax,
                               width=[graph[u][v]["weight"] * 5 for u, v in graph.edges()],
                               edge_color="slategray", alpha=0.35)
        nx.draw_networkx_nodes(graph, positions, ax=ax,
                               node_color=plt.cm.RdYlBu_r(norm(values)),
                               node_size=sizes, edgecolors="black", linewidths=0.6)
        for sector, (x, y) in positions.items():
            ax.text(x, y, sector, ha="center", va="center", fontsize=20, weight="bold",
                    path_effects=[PathEffects.withStroke(linewidth=3, foreground="white")])

        ax.set_title(city, loc="center", fontsize=26, weight="bold")
        ax.set_axis_off()
        ax.text(-0.02, 1.02, f"{chr(97 + i)})", transform=ax.transAxes,
                fontsize=30, fontweight="bold", va="bottom", ha="left")

        mappable = mpl.cm.ScalarMappable(norm=norm, cmap="RdYlBu_r")
        mappable.set_array([])
        colorbar = fig.colorbar(mappable, ax=ax, shrink=0.7, pad=0.02, aspect=25)
        colorbar.set_label(r"$\mathrm{PCI}^{\mathrm{emp}}$", fontsize=24)
        colorbar.ax.tick_params(labelsize=20)

    plt.tight_layout()
    return fig


def plot_eci_components(zone_frames, components, city_order):
    """Zone complexity against the diversity and ubiquity it is built from."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.6))

    for column, city in enumerate(city_order):
        zones = zone_frames[city][["geomid", "eci"]].dropna()
        zones["geomid"] = zones["geomid"].astype(str)
        merged = zones.merge(components[city], on="geomid", how="inner").dropna()
        colour = CITY_COLORS[city]

        for row, (values, label) in enumerate((
                (merged["diversity"], "Diversity"),
                (merged["mean_ubiquity"], "Mean ubiquity"))):
            ax = axes[row, column]
            ax.scatter(merged["eci"], values, s=18, alpha=0.55, color=colour,
                       linewidths=0, rasterized=True)
            ax.set_xlabel(ECI_LABEL, fontsize=18)
            ax.set_ylabel(label if column == 0 else "", fontsize=18)
            ax.tick_params(labelsize=15)
            r, _ = pearsonr(merged["eci"], values)
            ax.annotate(f"$r$ = {r:.2f}", xy=(0.05, 0.88), xycoords="axes fraction",
                        fontsize=16)
            if row == 0:
                ax.set_title(city, fontsize=20, fontweight="bold")
                ax.text(-0.12, 1.16, f"{chr(97 + column)})", transform=ax.transAxes,
                        fontsize=24, fontweight="bold", va="bottom", ha="left")

    fig.tight_layout()
    return fig


INCOME_AXES = {
    "Bay Area": ("Mean income (USD k/yr)", 1_000),
    "Los Angeles": ("Mean income (USD k/yr)", 1_000),
    "Mexico City": ("Flow-weighted wealth index", 1),
    "Rio de Janeiro": ("Flow-weighted income\n(BRL k/mo)", 1_000),
}


def plot_eci_income_scatter(zone_frames, income_data, city_order):
    """Zone complexity against the income or wealth proxy of the same zone."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()

    for i, (ax, city) in enumerate(zip(axes, city_order)):
        label, divisor = INCOME_AXES[city]
        zones = zone_frames[city][["geomid", "eci"]].copy()
        zones["geomid"] = zones["geomid"].astype(str)
        merged = zones.merge(income_data[city], on="geomid", how="inner").dropna(
            subset=["eci", "mean_wage"])

        eci = merged["eci"].values
        values = merged["mean_wage"].values / divisor
        r, p = pearsonr(eci, values)

        ax.scatter(eci, values, alpha=0.35, s=14, color=CITY_COLORS[city], rasterized=True)
        slope, intercept, *_ = stats.linregress(eci, values)
        line = np.linspace(eci.min(), eci.max(), 200)
        ax.plot(line, slope * line + intercept, color="k", lw=1.5, ls="--")

        ax.set_xlabel(ECI_LABEL, fontsize=22)
        ax.set_ylabel(label, fontsize=22)
        ax.set_title(city, fontsize=22, fontweight="bold")
        ax.tick_params(labelsize=20)
        ax.text(-0.30, 1.10, f"{chr(97 + i)})", transform=ax.transAxes,
                fontsize=26, fontweight="bold", va="bottom", ha="left")

        side = "right" if r < 0 else "left"
        ax.text(0.97 if r < 0 else 0.03, 0.97, f"$r$ = {r:.2f} ($p$ {_format_p(p)})",
                transform=ax.transAxes, ha=side, va="top", fontsize=17,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9))

    fig.suptitle(ECI_LABEL + " against the income or wealth proxy", fontsize=24, y=1.01)
    fig.tight_layout()
    return fig


SECTOR_MARKERS = {"Bay Area": "o", "Los Angeles": "s",
                  "Mexico City": "^", "Rio de Janeiro": "D"}

SECTOR_COLORS = {"Bay Area": "#2166ac", "Los Angeles": "#4393c3",
                 "Mexico City": "#d6604d", "Rio de Janeiro": "#b2182b"}


def plot_sector_crosscity(profiles, city_order):
    """Share of each sector's outflows that stay within the same sector."""
    combined = pd.concat([frame.assign(city=city) for city, frame in profiles.items()])
    order = combined.groupby("sector")["within_pct"].mean().sort_values().index.tolist()
    position = {sector: i for i, sector in enumerate(order)}

    fig, ax = plt.subplots(figsize=(12, max(7, len(order) * 0.7 + 3)))

    for i in range(len(order)):
        ax.axhspan(i - 0.5, i + 0.5, color="#f4f4f4" if i % 2 == 0 else "white", zorder=0)

    largest = np.log1p(combined["total_flows"].max())
    for city in city_order:
        subset = combined[combined["city"] == city].copy()
        subset = subset[subset["sector"].isin(position)]
        subset["y"] = subset["sector"].map(position)
        ax.scatter(subset["within_pct"], subset["y"],
                   s=np.log1p(subset["total_flows"]) / largest * 280 + 35,
                   color=SECTOR_COLORS[city], marker=SECTOR_MARKERS[city],
                   alpha=0.82, edgecolors="white", linewidths=0.4,
                   zorder=3, label=city)

    for sector, mean_value in combined.groupby("sector")["within_pct"].mean().items():
        if sector in position:
            y = position[sector]
            ax.plot([mean_value, mean_value], [y - 0.35, y + 0.35],
                    color="#222222", lw=2.0, solid_capstyle="round", zorder=4)
    ax.plot([], [], color="#222222", lw=2.5, label="Cross-city mean")

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=16)
    ax.set_xlabel("Within dominant sector (%)", fontsize=18)
    ax.set_xlim(left=-1)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=16)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, title="City", loc="lower center",
               bbox_to_anchor=(0.5, -0.08), ncol=5, fontsize=15,
               frameon=True, framealpha=0.9, title_fontsize=15)
    plt.tight_layout()
    return fig
