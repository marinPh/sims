"""Core Google Earth Engine processing functions for thermal mapping."""
import sys
sys.path.insert(0, '../thermal_survey')
sys.path.insert(0, '../gee')
import ee
from gee import config
import os
from typing import Dict

# ============================================================================
# STATIC LAYER FUNCTIONS
# ============================================================================

def calculate_slope(dem: ee.Image) -> ee.Image:
    """
    Calculate slope from DEM.

    Args:
        dem: Digital Elevation Model (ee.Image)

    Returns:
        ee.Image with slope in degrees (0-90)
    """
    slope = ee.Terrain.slope(dem)
    return slope.rename('slope')


def calculate_aspect(dem: ee.Image) -> ee.Image:
    """
    Calculate aspect (slope direction) from DEM.

    Args:
        dem: Digital Elevation Model (ee.Image)

    Returns:
        ee.Image with aspect in degrees (0-360)
        0 = North, 90 = East, 180 = South, 270 = West
    """
    aspect = ee.Terrain.aspect(dem)
    return aspect.rename('aspect')


def normalize_slope(slope_image: ee.Image) -> ee.Image:
    """
    Normalize slope to 0-1 scale using sigmoid function.

    Optimal slopes (20-45°) map to values near 1.
    Flat terrain (0°) and very steep (>60°) map to values near 0.

    Args:
        slope_image: ee.Image with slope in degrees

    Returns:
        ee.Image with normalized slope (0-1)
    """
    # Sigmoid centered at 30° (midpoint of optimal range)
    center = (config.SLOPE_OPTIMAL_MIN + config.SLOPE_OPTIMAL_MAX) / 2
    width = config.SLOPE_SIGMOID_WIDTH

    # Distance from optimal midpoint
    distance_from_optimal = slope_image.subtract(center).abs()

    # Sigmoid that peaks at optimal_mid and decreases on both sides
    normalized = ee.Image.constant(1).divide(
        ee.Image.constant(1).add(
            distance_from_optimal.divide(width).exp()
        )
    )

    return normalized.rename('slope_normalized')


def normalize_aspect(aspect_image: ee.Image, optimal_direction: Dict[str, any]) -> ee.Image:
    """
    Normalize aspect to 0-1 scale using cosine transform.

    Slopes facing the optimal direction (e.g., south in NH) map to 1.
    Slopes facing opposite direction map to 0.

    Args:
        aspect_image: ee.Image with aspect in degrees (0-360)
        optimal_direction: Dict with 'center' key (optimal aspect in degrees)

    Returns:
        ee.Image with normalized aspect (0-1)
    """
    optimal_aspect = optimal_direction['center']

    # Calculate angular difference from optimal
    # Convert to radians for cosine
    diff = aspect_image.subtract(optimal_aspect)
    diff_radians = diff.multiply(3.14159265359 / 180.0)

    # Cosine transform: cos(diff) maps 0° diff to 1, 180° diff to -1
    # Then scale to 0-1: (cos(diff) + 1) / 2
    cosine = diff_radians.cos()
    normalized = cosine.add(1).divide(2)

    return normalized.rename('aspect_normalized')


def calculate_ndvi(sentinel_image: ee.Image) -> ee.Image:
    """
    Calculate Normalized Difference Vegetation Index from Sentinel-2.

    NDVI = (NIR - Red) / (NIR + Red)

    Args:
        sentinel_image: ee.Image with B8 (NIR) and B4 (Red) bands

    Returns:
        ee.Image with NDVI band (-1 to 1)
        High values (>0.6) = dense vegetation
        Low values (<0.2) = barren land, rock
    """
    nir = sentinel_image.select('B8')
    red = sentinel_image.select('B4')

    ndvi = nir.subtract(red).divide(nir.add(red))

    return ndvi.rename('NDVI')


def normalize_to_unit_scale(image: ee.Image, roi: ee.Geometry) -> ee.Image:
    """
    Normalize image to 0-1 scale using min-max scaling within ROI.

    normalized = (value - min) / (max - min)

    Args:
        image: ee.Image to normalize
        roi: Region for computing min/max statistics

    Returns:
        ee.Image with values scaled to 0-1
    """
    # Get min/max from ROI (use coarser scale for speed)
    stats = image.reduceRegion(
        reducer=ee.Reducer.minMax(),
        geometry=roi,
        scale=config.NORMALIZATION_SCALE,
        maxPixels=1e9,
        bestEffort=True  # Allow GEE to use larger scale if needed
    )

    band_name = image.bandNames().get(0)
    min_val = ee.Number(stats.get(ee.String(band_name).cat('_min')))
    max_val = ee.Number(stats.get(ee.String(band_name).cat('_max')))

    # Normalize: (value - min) / (max - min)
    normalized = image.subtract(min_val).divide(max_val.subtract(min_val))

    return normalized.rename(ee.String(band_name).cat('_normalized'))


def weighted_overlay(layers_dict: Dict[str, ee.Image], weights_dict: Dict[str, float]) -> ee.Image:
    """
    Combine normalized layers using weighted sum.

    Args:
        layers_dict: Dict mapping layer name to ee.Image (all normalized to 0-1)
        weights_dict: Dict mapping layer name to weight (must sum to 1.0)

    Returns:
        ee.Image with single 'thermal_probability' band (0-1)
    """
    # Start with zero image
    result = ee.Image.constant(0)

    # Add each weighted layer
    for layer_name, weight in weights_dict.items():
        layer = layers_dict[layer_name]
        weighted = layer.multiply(weight)
        result = result.add(weighted)

    return result.rename('thermal_probability')


def apply_water_mask(image: ee.Image, roi: ee.Geometry) -> ee.Image:
    """
    Mask out water bodies from thermal probability image.

    Uses JRC Global Surface Water dataset.
    Masks pixels with water occurrence > threshold.

    Args:
        image: ee.Image to mask (typically thermal_probability)
        roi: Region of interest (for clipping water dataset)

    Returns:
        ee.Image with water pixels masked (set to null)
    """
    # Load JRC Global Surface Water
    water = ee.Image(config.WATER_DATASET).select('occurrence')

    # Create mask: 1 where water occurrence <= threshold, 0 where > threshold
    water_mask = water.lte(config.WATER_OCCURRENCE_THRESHOLD)

    # Apply mask (pixels with 0 mask become null)
    masked = image.updateMask(water_mask)

    return masked


# ============================================================================
# DYNAMIC MONTHLY PROCESSING
# ============================================================================

def get_landsat_composite(roi: ee.Geometry, year: int, month: int) -> ee.Image:
    """
    Get cloud-masked Landsat 8/9 median composite for a month.

    Args:
        roi: Region of interest
        year: Year (e.g., 2023)
        month: Month number (1-12)

    Returns:
        ee.Image with Landsat bands (median composite)
    """
    # Generate date range
    from gee.utils import generate_month_list
    date_info = generate_month_list(year, [month])[0]
    start_date = date_info['start']
    end_date = date_info['end']

    # Filter Landsat collection
    collection = ee.ImageCollection(config.LANDSAT_COLLECTION) \
        .filterBounds(roi) \
        .filterDate(start_date, end_date)

    # Cloud mask function
    def mask_landsat_clouds(image):
        qa = image.select('QA_PIXEL')
        # Bits 3 and 4 are cloud and cloud shadow
        cloud_mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
        return image.updateMask(cloud_mask)

    # Apply cloud mask and compute median
    masked = collection.map(mask_landsat_clouds)
    composite = masked.median().clip(roi)

    return composite


def calculate_lst(landsat_image: ee.Image) -> ee.Image:
    """
    Calculate Land Surface Temperature from Landsat thermal band.

    Args:
        landsat_image: Landsat 8/9 image with ST_B10 band

    Returns:
        ee.Image with LST in Celsius
    """
    # Landsat Collection 2 provides temperature in Kelvin * 0.00341802 + 149.0
    # ST_B10 is already in Kelvin, just need to convert to Celsius
    lst_kelvin = landsat_image.select('ST_B10').multiply(0.00341802).add(149.0)
    lst_celsius = lst_kelvin.subtract(273.15)

    return lst_celsius.rename('LST')


def get_sentinel_composite(roi: ee.Geometry, year: int, month: int) -> ee.Image:
    """
    Get cloud-masked Sentinel-2 median composite for a month.

    Args:
        roi: Region of interest
        year: Year (e.g., 2023)
        month: Month number (1-12)

    Returns:
        ee.Image with Sentinel-2 bands (median composite)
    """
    # Generate date range
    from gee.utils import generate_month_list
    date_info = generate_month_list(year, [month])[0]
    start_date = date_info['start']
    end_date = date_info['end']

    # Filter Sentinel-2 collection
    collection = ee.ImageCollection(config.SENTINEL_COLLECTION) \
        .filterBounds(roi) \
        .filterDate(start_date, end_date)

    # Cloud mask function
    def mask_sentinel_clouds(image):
        qa = image.select('QA60')
        # Bits 10 and 11 are clouds and cirrus
        cloud_mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
        return image.updateMask(cloud_mask)

    # Apply cloud mask and compute median
    masked = collection.map(mask_sentinel_clouds)
    composite = masked.median().clip(roi)

    return composite


def process_single_month(roi: ee.Geometry, year: int, month: int,
                        static_layers: Dict[str, ee.Image],
                        weights: Dict[str, float]) -> ee.Image:
    """
    Process a single month to generate thermal probability map.

    Args:
        roi: Region of interest
        year: Year (e.g., 2023)
        month: Month number (1-12)
        static_layers: Dict with 'slope' and 'aspect' normalized images
        weights: Dict with weights for slope, aspect, temperature, dryness

    Returns:
        ee.Image with thermal_probability band (0-1)
    """
    # Get dynamic composites
    landsat = get_landsat_composite(roi, year, month)
    sentinel = get_sentinel_composite(roi, year, month)

    # Calculate LST and normalize
    lst = calculate_lst(landsat)
    lst_norm = normalize_to_unit_scale(lst, roi)

    # Calculate NDVI and dryness
    ndvi = calculate_ndvi(sentinel)
    ndvi_norm = normalize_to_unit_scale(ndvi, roi)
    dryness = ee.Image.constant(1).subtract(ndvi_norm).rename('dryness_normalized')

    # Combine all layers
    layers = {
        'slope': static_layers['slope'],
        'aspect': static_layers['aspect'],
        'temperature': lst_norm,
        'dryness': dryness
    }

    # Weighted overlay
    probability = weighted_overlay(layers, weights)

    # Apply water mask
    masked_probability = apply_water_mask(probability, roi)

    # Set month property for time-series
    return masked_probability.set('month', month).set('year', year)


def generate_monthly_series(roi: ee.Geometry, year: int, months: list,
                           static_layers: Dict[str, ee.Image],
                           weights: Dict[str, float]) -> ee.ImageCollection:
    """
    Generate time-series of thermal probability maps.

    Args:
        roi: Region of interest
        year: Year (e.g., 2023)
        months: List of month numbers (e.g., [5, 6, 7, 8, 9])
        static_layers: Dict with 'slope' and 'aspect' normalized images
        weights: Dict with weights for slope, aspect, temperature, dryness

    Returns:
        ee.ImageCollection with monthly thermal probability maps
    """
    # Process each month
    monthly_images = []
    for month in months:
        print(f"Processing month {month}...")
        monthly_prob = process_single_month(roi, year, month, static_layers, weights)
        monthly_images.append(monthly_prob)

    # Convert to ImageCollection
    return ee.ImageCollection(monthly_images)


# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_thermal_map_to_png(image: ee.Image, roi: ee.Geometry,
                               filepath: str, width: int = 1920,
                               palette: list = None) -> str:
    """
    Export thermal probability map as PNG using GEE getThumbURL.

    Args:
        image: ee.Image with thermal_probability band
        roi: Region of interest for bounds
        filepath: Output filepath (e.g., 'exports/png/may_2023.png')
        width: Image width in pixels (height auto-calculated)
        palette: Color palette list (defaults to config.COLOR_PALETTE)

    Returns:
        str: Filepath where PNG was saved
    """
    import urllib.request
    import os

    if palette is None:
        palette = config.COLOR_PALETTE

    # Get visualization parameters
    vis_params = {
        'min': 0,
        'max': 1,
        'palette': palette,
        'dimensions': width,
        'region': roi,
        'format': 'png'
    }

    # Get thumbnail URL
    url = image.getThumbURL(vis_params)

    # Create directory if needed
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Download PNG
    urllib.request.urlretrieve(url, filepath)

    return filepath


def export_monthly_series_to_png(monthly_collection: ee.ImageCollection,
                                  roi: ee.Geometry, year: int,
                                  output_dir: str = 'exports/png',
                                  region_name: str = 'region') -> list:
    """
    Export all monthly thermal maps as PNG files.

    Args:
        monthly_collection: ee.ImageCollection with monthly thermal_probability
        roi: Region of interest
        year: Year for filename
        output_dir: Output directory path
        region_name: Name for filename (e.g., 'val_de_ruz')

    Returns:
        list: List of exported filepaths
    """
    month_names = {
        1: 'january', 2: 'february', 3: 'march', 4: 'april',
        5: 'may', 6: 'june', 7: 'july', 8: 'august',
        9: 'september', 10: 'october', 11: 'november', 12: 'december'
    }

    monthly_list = monthly_collection.toList(monthly_collection.size())
    size = monthly_collection.size().getInfo()

    exported_files = []

    for i in range(size):
        monthly_img = ee.Image(monthly_list.get(i))
        month = monthly_img.get('month').getInfo()
        month_name = month_names.get(month, f'month{month}')

        filename = f'{region_name}_{year}_{month:02d}_{month_name}.png'
        filepath = os.path.join(output_dir, filename)

        print(f"Exporting {month_name.capitalize()} {year}...")
        export_thermal_map_to_png(monthly_img, roi, filepath)
        exported_files.append(filepath)
        print(f"  ✓ Saved: {filepath}")

    return exported_files
