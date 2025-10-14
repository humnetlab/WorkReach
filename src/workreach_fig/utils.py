import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.path import Path
from matplotlib.textpath import TextToPath
from matplotlib.font_manager import FontProperties
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import fontawesome as fa
import math


def get_marker(symbol, font_path="/usr/share/fonts-font-awesome/fonts/FontAwesome.otf"):
    fp = FontProperties(fname=font_path)
    v, codes = TextToPath().get_text_path(fp, symbol)
    v = np.array(v)
    mean = np.mean([np.max(v, axis=0), np.min(v, axis=0)], axis=0)
    return Path(v - mean, codes, closed=False)


def haversine_m(lon1, lat1, lon2, lat2):
    R = 6_371_000.0
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = math.sin(dlat/2.0)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2.0)**2
    return 2.0 * R * math.asin(math.sqrt(a))


def get_zoom_bounds(centroid_x, centroid_y, side_length_km, 
                    km_per_lon=105, km_per_lat=111):
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
    sns.set_theme(style="ticks", context="talk")
    fig, ax = plt.subplots(figsize=figsize)
    
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
    
    lat_radius = close_regime_radius / 111_000
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

################## 3D Stacked Map  ##################
def polygon_to_3dcoords(polygon, z=0):
    if polygon is None or polygon.is_empty:
        return []
    
    def ring_to_3d(ring):
        return [(x, y, z) for x, y in ring.coords]
    
    coords_3d = []
    if polygon.geom_type == 'Polygon':
        coords_3d.append(ring_to_3d(polygon.exterior))
        for interior in polygon.interiors:
            coords_3d.append(ring_to_3d(interior))
    elif polygon.geom_type == 'MultiPolygon':
        for poly in polygon.geoms:
            coords_3d.append(ring_to_3d(poly.exterior))
            for interior in poly.interiors:
                coords_3d.append(ring_to_3d(interior))
    
    return coords_3d


def create_plane(bounds, z_level, color="white", alpha=0.3):
    xmin, ymin, xmax, ymax = bounds
    verts_3d = [
        (xmin, ymin, z_level),
        (xmax, ymin, z_level),
        (xmax, ymax, z_level),
        (xmin, ymax, z_level)
    ]
    poly = Poly3DCollection(
        [verts_3d], 
        facecolors=color, 
        alpha=alpha, 
        edgecolor="black"
    )
    return poly


def plot_geodf_3d(ax, geodf, z_level, column=None, colormap='Blues', 
                  alpha=0.8, edgecolor='black', linewidth=0.2, zpos=0):
    if column is not None and column in geodf.columns:
        vals = geodf[column].dropna()
        vmin, vmax = vals.min(), vals.max()
        norm = plt.Normalize(vmin, vmax)
        cmap = plt.cm.get_cmap(colormap)
    else:
        cmap = plt.cm.get_cmap(colormap)
        norm = None
    
    for idx, row in geodf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        
        if norm is not None and column in geodf.columns and not np.isnan(row[column]):
            color = cmap(norm(row[column]))
        else:
            color = cmap(0.5)
        
        # Convert geometry to 3D
        rings_3d = polygon_to_3dcoords(geom, z=z_level)
        for ring in rings_3d:
            if len(ring) >= 3:
                poly_3d = Poly3DCollection(
                    [ring], 
                    facecolors=color, 
                    edgecolors=edgecolor,
                    alpha=alpha, 
                    linewidth=linewidth
                )
                poly_3d.set_sort_zpos(zpos)
                ax.add_collection3d(poly_3d)


def create_3d_stacked_map(gdf, z_levels, columns, colormaps, 
                          title="3D Stacked Map Visualization",
                          figsize=(20, 20), alpha_min=0.1, alpha_max=0.7,
                          elev=30, azim=-60, save_path=None, dpi=300):

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    bounds = gdf.total_bounds
    padding = (bounds[2] - bounds[0]) * 0.05
    map_bounds = [
        bounds[0] - padding, 
        bounds[1] - padding,
        bounds[2] + padding, 
        bounds[3] + padding
    ]
    
    min_z, max_z = min(z_levels), max(z_levels)
    if max_z > min_z:
        alpha_values = [
            alpha_min + (alpha_max - alpha_min) * ((z - min_z)/(max_z - min_z)) 
            for z in z_levels
        ]
    else:
        alpha_values = [0.7] * len(z_levels)
    
    z_sorted = sorted(
        list(zip(z_levels, columns, colormaps, alpha_values)), 
        key=lambda x: x[0]
    )
    
    for i, (z, col, cmap_name, plane_alpha) in enumerate(z_sorted):
        plane = create_plane(map_bounds, z_level=z, color="white", alpha=plane_alpha)
        plane.set_sort_zpos(i * 2)
        ax.add_collection3d(plane)
        
        plot_geodf_3d(
            ax=ax, 
            geodf=gdf.dropna(subset=[col]), 
            z_level=z,
            column=col, 
            colormap=cmap_name, 
            alpha=1.0,
            edgecolor='lightgrey', 
            linewidth=0.2,
            zpos=i*2 + 1
        )
    
    ax.view_init(elev=elev, azim=azim)
    
    ax.set_xlim(map_bounds[0], map_bounds[2])
    ax.set_ylim(map_bounds[1], map_bounds[3])
    ax.set_zlim(0, max(z_levels)*1.2 if max_z > 0 else 1)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    x_range = map_bounds[2] - map_bounds[0]
    y_range = map_bounds[3] - map_bounds[1]
    z_range = max_z * 1.2 if max_z > 0 else 1
    ax.set_box_aspect((x_range, y_range, z_range*0.3))
    
    ax.set_axis_off()
    if title:
        plt.title(title, fontsize=14, pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        if save_path.endswith('.png'):
            pdf_path = save_path.replace('.png', '.pdf')
            plt.savefig(pdf_path, dpi=dpi, bbox_inches='tight')
    
    return fig, ax