# simulation/probability_map.py
"""
Download thermal probability GeoTIFF from GEE and expose
it as numpy arrays ready for the thermal field simulation.

Usage:
    from simulation.probability_map import build_probability_map

    pmap = build_probability_map(
        roi_path = '../data/roi/val_de_ruz.geojson',
        years    = [2021, 2022, 2023],
        month    = 5,
        out_path = '../data/probability_maps/val_de_ruz_2023_05.tif',
    )

    # pmap keys:
    #   p          – raw probability array  [0, 1]      shape (H, W)
    #   p_spawn    – spawn distribution     sums to 1   shape (H, W)
    #   transform  – affine transform (pixel → UTM m)
    #   resolution – pixel size in metres
    #   bounds     – dict  x_min/x_max/y_min/y_max
"""

import os
import urllib.request
import numpy as np
import rasterio
from rasterio.transform import rowcol, xy as rc_xy
from scipy.ndimage import sobel

def apply_sobel(pmap):
    p = pmap['p']
    sx = sobel(p, axis=1)   # horizontal gradient
    sy = sobel(p, axis=0)   # vertical gradient
    magnitude = np.hypot(sx, sy)
    # normalise to [0, 1]
    magnitude /= magnitude.max()
    return magnitude

import ee
from gee import config
from gee.utils import (
    load_roi_from_geojson,
    detect_hemisphere,
    get_optimal_aspect_direction,
)
from gee.core import (
    calculate_slope, calculate_aspect,
    normalize_slope, normalize_aspect,
    get_landsat_composite, get_sentinel_composite,
    calculate_lst, normalize_to_unit_scale, calculate_ndvi,
    weighted_overlay, apply_water_mask,
)

# ── physical ranges ────────────────────────────────────────────────────────
P_THRESHOLD = 0.10   # cells below this → 0


def _build_gee_image(roi, years, month):
    """
    Build thermal_probability ee.Image averaged over multiple years.

    Parameters
    ----------
    roi   : ee.Geometry
    years : list of int  e.g. [2021, 2022, 2023]
    month : int
    """
    hemisphere     = detect_hemisphere(roi)
    optimal_aspect = get_optimal_aspect_direction(hemisphere)

    # Static layers — DEM-derived, same for every year
    dem         = ee.Image(config.DEM_COLLECTION).clip(roi)
    slope_norm  = normalize_slope(calculate_slope(dem))
    aspect_norm = normalize_aspect(calculate_aspect(dem), optimal_aspect)

    # Build one probability image per year then average
    yearly_images = []
    for y in years:
        print(f'  processing year {y}...')
        landsat  = get_landsat_composite(roi, y, month)
        sentinel = get_sentinel_composite(roi, y, month)

        lst_norm  = normalize_to_unit_scale(calculate_lst(landsat), roi)
        ndvi_norm = normalize_to_unit_scale(calculate_ndvi(sentinel), roi)
        dryness   = ee.Image.constant(1).subtract(ndvi_norm).rename('dryness_normalized')

        layers = {
            'slope':       slope_norm,
            'aspect':      aspect_norm,
            'temperature': lst_norm,
            'dryness':     dryness,
        }
        prob = weighted_overlay(layers, config.WEIGHTS)
        yearly_images.append(prob)

    # Pixel-wise mean across years
    averaged = ee.ImageCollection(yearly_images).mean().rename('thermal_probability')
    return apply_water_mask(averaged, roi)


def download_tif(roi_path, years, month, out_path,
                 crs='EPSG:32632', scale=30, gee_project='mybootcamp-dac1e'):
    """
    Download thermal probability GeoTIFF from GEE, averaged over multiple years.

    Parameters
    ----------
    roi_path    : path to GeoJSON file
    years       : list of int  e.g. [2021, 2022, 2023]
    month       : int
    out_path    : where to save the .tif
    crs         : UTM CRS string  (default: zone 32N, covers Switzerland)
    scale       : resolution in metres
    gee_project : GEE project id
    """
    ee.Initialize(project=gee_project)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    roi   = load_roi_from_geojson(roi_path)
    print(f'Building month={month} average over years {years}...')
    image = _build_gee_image(roi, years, month)

    url = image.getDownloadURL({
        'name':   'thermal_probability',
        'scale':  scale,
        'region': roi,
        'crs':    crs,
        'format': 'GEO_TIFF',
    })

    print(f'Downloading → {out_path}')
    urllib.request.urlretrieve(url, out_path)
    print('✓ Done')
    return out_path


def load_tif(tif_path, p_threshold=P_THRESHOLD):
    """
    Load a GeoTIFF into a dict of numpy arrays.

    Parameters
    ----------
    tif_path    : path to .tif file
    p_threshold : cells below this are zeroed out

    Returns
    -------
    dict with keys: p, p_spawn, transform, resolution, bounds
                    transform, resolution, bounds
    """
    with rasterio.open(tif_path) as src:
        raw       = src.read(1).astype(np.float32)
        transform = src.transform
        res       = abs(transform.a)
        h, w      = raw.shape
        x_min, y_max = transform * (0, 0)
        x_max, y_min = transform * (w, h)

    # clean
    raw = np.clip(np.nan_to_num(raw, nan=0.0), 0.0, 1.0)
    raw[raw < p_threshold] = 0.0

    n_active = int((raw > 0).sum())
    print(f'[ProbabilityMap] active cells: {n_active}/{raw.size} '
          f'({100*n_active/raw.size:.1f}%)  resolution: {res:.0f}m')

    # spawn distribution
    total   = raw.sum()
    
    p_spawn = raw / total if total > 0 else raw.copy()
    
    sobel_raw = apply_sobel({'p': raw})
    sobel_clip = np.clip(sobel_raw, 0.0, 1.0)
    sobel_clip[sobel_clip < p_threshold] = 0.0
    sobel_spawn = sobel_clip / sobel_clip.sum() if sobel_clip.sum() > 0 else sobel_clip.copy()

    return dict(
        p         = raw,
        p_spawn   = p_spawn,
        sobel_p   = sobel_raw,
        sobel_spawn = sobel_spawn,
        transform = transform,
        resolution= res,
        bounds    = dict(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max),
    )


def build_probability_map(roi_path, years, month, out_path,
                          p_threshold=P_THRESHOLD,
                          crs='EPSG:32632', scale=30,
                          gee_project='mybootcamp-dac1e'):
    """
    Download (if needed) and load the probability map in one call.
    Skips the download if out_path already exists.
    """
    if not os.path.exists(out_path):
        download_tif(roi_path, years, month, out_path, crs, scale, gee_project)
    else:
        print(f'[ProbabilityMap] using cached {out_path}')

    return load_tif(out_path, p_threshold)


# ── spatial helpers ────────────────────────────────────────────────────────

def query(pmap, x, y, key='p'):
    """Return pmap[key] at UTM position (x, y)."""
    t = pmap['transform']
    row, col = rowcol(t, x, y)
    h, w = pmap[key].shape
    row = int(np.clip(row, 0, h - 1))
    col = int(np.clip(col, 0, w - 1))
    return float(pmap[key][row, col])


def sample_spawn(pmap, rng=None):
    """
    Draw one (x, y) spawn location proportional to p_spawn.
    Returns (x, y) in UTM metres with sub-pixel jitter.
    """
    if rng is None:
        rng = np.random.default_rng()

    flat     = pmap['p_spawn'].ravel()
    idx      = rng.choice(flat.size, p=flat)
    row, col = divmod(idx, pmap['p_spawn'].shape[1])
    x, y     = rc_xy(pmap['transform'], int(row), int(col))
    r        = pmap['resolution'] / 2
    return float(x) + rng.uniform(-r, r), float(y) + rng.uniform(-r, r)


def sample_spawn_params(pmap, rng=None):
    """
    Sample a spawn location from the probability map.
    Physical thermal parameters (w_max, T_life, R_base) are
    determined at spawn time by ThermalState, not here.

    Returns dict: x0, y0, p_val
    """
    if rng is None:
        rng = np.random.default_rng()

    x0, y0 = sample_spawn(pmap, rng)

    return dict(
        x0    = x0,
        y0    = y0,
        p_val = query(pmap, x0, y0, 'p'),
    )


# ── quick visual check ─────────────────────────────────────────────────────
def plot_sources(pmap, sources, save_path=None, map_key='p'):
    """
    Overlay boolean source mask on the probability map.

    Parameters
    ----------
    pmap    : dict from load_tif()
    sources : 2D bool array (H, W), same shape as pmap['p']
              True where a thermal spawns this tick
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    b   = pmap['bounds']
    ext = [b['x_min'], b['x_max'], b['y_min'], b['y_max']]
    res = pmap['resolution']

    # convert active pixel indices to UTM coords (cell centres)
    rows, cols = np.where(sources)
    xs, ys = [], []
    for r, c in zip(rows, cols):
        x, y = rc_xy(pmap['transform'], int(r), int(c))
        xs.append(x); ys.append(y)

    fig, ax = plt.subplots(figsize=(9, 7))

    # background: probability map
    im = ax.imshow(pmap[map_key], origin='upper', extent=ext,
                   cmap='Greens', vmin=0, vmax=1, alpha=0.85)
    plt.colorbar(im, ax=ax, label='Spawn probability P', shrink=0.8)

    # sources: red dots
    if xs:
        ax.scatter(xs, ys, c='red', s=18, linewidths=0,
                   alpha=0.8, label=f'Active sources ({len(xs)})')

    ax.set_title(f'Thermal spawn sources  [n={len(xs)}]')
    ax.set_xlabel('Easting [m]')
    ax.set_ylabel('Northing [m]')
    ax.legend(fontsize=9, loc='upper right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'✓ saved {save_path}')
    plt.show()

def plot(pmap, n_samples=50, save_path=None):
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    pts = np.array([sample_spawn(pmap, rng) for _ in range(n_samples)])
    b   = pmap['bounds']
    ext = [b['x_min'], b['x_max'], b['y_min'], b['y_max']]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(pmap['p'], origin='upper', extent=ext,
                   cmap='RdYlGn', vmin=0, vmax=1)
    ax.scatter(pts[:, 0], pts[:, 1], c='blue', s=12, alpha=0.6, label='sampled spawns')
    ax.set_title('Thermal Spawn Probability P(x,y)')
    ax.set_xlabel('Easting [m]')
    ax.set_ylabel('Northing [m]')
    ax.legend(fontsize=8)
    plt.colorbar(im, ax=ax, label='P')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'✓ saved {save_path}')
    plt.show()


# ── standalone entry point ─────────────────────────────────────────────────

if __name__ == '__main__':
    pmap = build_probability_map(
        roi_path    = '../data/roi/val_de_ruz.geojson',
        years       = [2021, 2022, 2023],
        month       = 5,
        out_path    = '../data/probability_maps/val_de_ruz_may_avg.tif',
        gee_project = 'mybootcamp-dac1e',
    )
    plot(pmap, save_path='../data/probability_maps/val_de_ruz_may_avg.png')