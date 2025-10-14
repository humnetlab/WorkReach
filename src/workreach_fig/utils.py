import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.path import Path
from matplotlib.textpath import TextToPath
from matplotlib.font_manager import FontProperties
from matplotlib_scalebar.scalebar import ScaleBar
import fontawesome as fa
import math


def get_marker(symbol, font_path="/usr/share/fonts-font-awesome/fonts/FontAwesome.otf"):
    """
    Convert a FontAwesome symbol to a matplotlib marker path.
    
    Parameters
    ----------
    symbol : str
        FontAwesome character/symbol
    font_path : str
        Path to FontAwesome font file
        
    Returns
    -------
    Path
        Matplotlib Path object for the symbol
    """
    fp = FontProperties(fname=font_path)
    v, codes = TextToPath().get_text_path(fp, symbol)
    v = np.array(v)
    mean = np.mean([np.max(v, axis=0), np.min(v, axis=0)], axis=0)
    return Path(v - mean, codes, closed=False)


def haversine_m(lon1, lat1, lon2, lat2):
    """
    Calculate great circle distance in meters between two points.
    
    Parameters
    ----------
    lon1, lat1, lon2, lat2 : float
        Coordinates in decimal degrees
        
    Returns
    -------
    float
        Distance in meters
    """
    R = 6_371_000.0
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = math.sin(dlat/2.0)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2.0)**2
    return 2.0 * R * math.asin(math.sqrt(a))


def get_zoom_bounds(centroid_x, centroid_y, side_length_km, 
                    km_per_lon=105, km_per_lat=111):
    """
    Calculate bounding box centered at given coordinates.
    
    Parameters
    ----------
    centroid_x : float
        Center longitude
    centroid_y : float
        Center latitude
    side_length_km : float
        Full side length of bounding box in kilometers
    km_per_lon : float
        Kilometers per degree longitude (default for Mexico City)
    km_per_lat : float
        Kilometers per degree latitude
        
    Returns
    -------
    list
        [min_lon, min_lat, max_lon, max_lat]
    """
    half_side = side_length_km / 2.0
    lat_offset = half_side / km_per_lat
    lon_offset = half_side / km_per_lon
    return [
        centroid_x - lon_offset,
        centroid_y - lat_offset,
        centroid_x + lon_offset,
        centroid_y + lat_offset
    ]


def draw_arrow(ax, x_start, y_start, x_end, y_end, width, color, zorder=1):
    """
    Draw a fancy curved arrow between two points.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axis to draw on
    x_start, y_start : float
        Start coordinates
    x_end, y_end : float
        End coordinates
    width : float
        Width of arrow (proportional to utility)
    color : str
        Arrow color
    zorder : int
        Drawing order
        
    Returns
    -------
    tuple
        (midx, midy) - midpoint coordinates of the arrow
    """
    arrow = FancyArrowPatch(
        (x_start, y_start), 
        (x_end, y_end),
        connectionstyle="arc3,rad=0.1", 
        arrowstyle=f"simple,head_width={5*width},head_length={7*width}", 
        linewidth=5,
        color="#000000",
        zorder=zorder
    )
    ax.add_patch(arrow)
    
    t = 0.6
    rad = 0.1
    dx = x_end - x_start
    dy = y_end - y_start
    midx = (1-t)*x_start + t*x_end + rad*dy
    midy = (1-t)*y_start + t*y_end - rad*dx
    
    return midx, midy


def process_utility_data(gdf, close_regime_radius, close_job_idx, far_job_idx, 
                         plot_size, random_state=42):
    """
    Process geodataframe to compute utilities and select job locations.
    
    Parameters
    ----------
    gdf : GeoDataFrame
        Input geodataframe with 'eci' and 'eci_level' columns
    close_regime_radius : float
        Radius in meters for close regime
    close_job_idx : int
        Index for selecting close job (sorted by utility)
    far_job_idx : int
        Index for selecting far job (sorted by utility)
    plot_size : float
        Plot size in km for filtering far jobs
    random_state : int
        Random state for reproducible home selection
        
    Returns
    -------
    tuple
        (processed_gdf, home_coords, job_locations, zoom_bounds)
    """
    zoom_gdf = gdf.copy()
    
    if 'centroid' not in zoom_gdf.columns:
        zoom_gdf['centroid'] = zoom_gdf.geometry.centroid
    
    centroid_x = zoom_gdf['centroid'].x.mean()
    centroid_y = zoom_gdf['centroid'].y.mean()
    
    home_candidates = zoom_gdf[zoom_gdf['eci_level'].isin(['Medium', 'High'])]
    if len(home_candidates) == 0:
        home_candidates = zoom_gdf
    
    d2c = home_candidates['centroid'].apply(
        lambda p: haversine_m(p.x, p.y, centroid_x, centroid_y)
    )
    home_idx = d2c.nsmallest(3).sample(1, random_state=random_state).index[0]
    
    zoom_gdf['location_type'] = 'regular'
    zoom_gdf.loc[home_idx, 'location_type'] = 'home'
    
    home_pt = zoom_gdf.at[home_idx, 'centroid']
    home_lon, home_lat = float(home_pt.x), float(home_pt.y)
    home_coords = (home_lon, home_lat)
    
    zoom_gdf['distance_to_home'] = zoom_gdf['centroid'].apply(
        lambda p: haversine_m(p.x, p.y, home_lon, home_lat)
    )
    
    zoom_gdf['regime'] = np.where(
        zoom_gdf['distance_to_home'] <= close_regime_radius, 
        'close', 
        'far'
    )
    
    eci_min = zoom_gdf['eci'].min()
    eci_max = zoom_gdf['eci'].max()
    eci_norm = (zoom_gdf['eci'] - eci_min) / (eci_max - eci_min + 1e-12)
    
    zoom_gdf['utility'] = np.where(
        zoom_gdf['regime'] == 'close',
        0.9 - 0.7 * (zoom_gdf['distance_to_home'] / close_regime_radius),
        0.7 * eci_norm + 0.3 * (1.0 - np.minimum(1.0, zoom_gdf['distance_to_home'] / 10_000.0))
    )
    
    zoom_gdf.loc[home_idx, 'utility'] = np.nan
    zoom_gdf['utility'] = np.clip(zoom_gdf['utility'], 0.0, 1.0).round(2)
    
    close_jobs = zoom_gdf[
        (zoom_gdf['regime'] == 'close') & 
        (zoom_gdf.index != home_idx)
    ].sort_values('utility', ascending=False).iloc[close_job_idx:close_job_idx+1]
    
    far_jobs = pd.concat([
        zoom_gdf[
            (zoom_gdf['regime'] == 'far') & 
            (zoom_gdf["distance_to_home"] < plot_size * 500)
        ].sort_values('utility', ascending=False).head(1),
        zoom_gdf[
            (zoom_gdf['regime'] == 'far') & 
            (zoom_gdf["distance_to_home"] < plot_size * 500)
        ].sort_values('utility', ascending=False).iloc[far_job_idx:far_job_idx+1]
    ])
    
    job_locations = pd.concat([close_jobs, far_jobs])
    
    zoom_gdf.loc[job_locations.index, 'location_type'] = np.where(
        job_locations['regime'] == 'close',
        'job_close',
        'job_far'
    )
    
    zoom_bounds = get_zoom_bounds(centroid_x, centroid_y, plot_size)
    
    return zoom_gdf, home_coords, job_locations, zoom_bounds


def plot_utility_map(zoom_gdf, home_coords, job_locations, zoom_bounds, 
                     close_regime_radius, column='utility', cmap='viridis',
                     figsize=(15, 15), save_path=None, dpi=300):
    """
    Create the utility map visualization.
    
    Parameters
    ----------
    zoom_gdf : GeoDataFrame
        Processed geodataframe with utility values
    home_coords : tuple
        (lon, lat) of home location
    job_locations : DataFrame
        Selected job locations
    zoom_bounds : list
        [min_lon, min_lat, max_lon, max_lat]
    close_regime_radius : float
        Radius in meters for close regime circle
    column : str
        Column to plot ('utility' or 'eci')
    cmap : str
        Colormap name
    figsize : tuple
        Figure size
    save_path : str or None
        Path to save figure (if None, doesn't save)
    dpi : int
        DPI for saved figure
        
    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axis objects
    """
    sns.set_theme(style="ticks", context="talk")
    fig, ax = plt.subplots(figsize=figsize)
    
    # Normalize values for coloring
    vmin = zoom_gdf[column].min()
    vmax = zoom_gdf[column].max()
    
    zoom_gdf.plot(
        column=column,
        ax=ax,
        cmap=cmap,
        edgecolor='white',
        linewidth=0.5,
        alpha=0.8,
        vmin=vmin,
        vmax=vmax,
        legend=True,
        legend_kwds={
            'label': 'Utility' if column == 'utility' else 'Economic Complexity Index',
            'orientation': 'vertical',
            'shrink': 0.5
        }
    )
    
    ax.set_xlim(zoom_bounds[0] - 0.005, zoom_bounds[2] + 0.005)
    ax.set_ylim(zoom_bounds[1] - 0.005, zoom_bounds[3] + 0.005)
    
    lat_radius = close_regime_radius / 111_000  # Convert meters to degrees
    lon_radius = close_regime_radius / 105_000
    avg_radius = (lat_radius + lon_radius) / 2
    close_circle = Circle(
        home_coords, avg_radius, fill=False,
        edgecolor='#FFFFFF', linestyle='--', linewidth=5
    )
    ax.add_patch(close_circle)
    
    home_icon = fa.icons['home']
    ax.scatter(
        home_coords[0], home_coords[1], 
        s=2000,
        c="white", 
        marker=get_marker(home_icon),
        edgecolors="black", linewidth=1, zorder=100
    )
    
    job_icon = fa.icons['building']
    for idx, row in job_locations.iterrows():
        x, y = row.centroid.x, row.centroid.y
        ax.scatter(
            x, y, 
            s=1000,
            c="white", 
            marker=get_marker(job_icon),
            edgecolors="black", linewidth=1, zorder=100
        )
    
    for idx, job in job_locations.iterrows():
        x_start, y_start = home_coords
        x_end, y_end = job.centroid.x, job.centroid.y
        color = '#4CAF50' if job['regime'] == 'close' else '#9C27B0'
        width = job['utility'] * 1.5 * 0.001
        
        midx, midy = draw_arrow(ax, x_start, y_start, x_end, y_end, width, color, zorder=4)
        
        offset = (60, 60) if job['regime'] == 'close' else (60, -60)
        ax.annotate(
            f'Utility: {job["utility"]}\nECI: {job["eci"]:.2f}\nDistance: {int(job["distance_to_home"])} m',
            xy=(x_end, y_end),
            xytext=offset,
            textcoords="offset points",
            ha='center',
            va='center',
            fontsize=16,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.8),
            zorder=6
        )
    
    scalebar = ScaleBar(
        105,
        "km",
        fixed_value=5,
        location='lower right',
        pad=0.5,
    )
    ax.add_artist(scalebar)
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        if save_path.endswith('.png'):
            pdf_path = save_path.replace('.png', '.pdf')
            plt.savefig(pdf_path, dpi=dpi, bbox_inches='tight')
    
    return fig, ax