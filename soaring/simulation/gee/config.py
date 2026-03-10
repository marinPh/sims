# config.py
"""Configuration parameters for thermal soaring probability mapping."""

# Weighted overlay configuration
WEIGHTS = {
    'slope': 0.25,
    'aspect': 0.25,
    'temperature': 0.25,
    'dryness': 0.25,
}

# Season configuration
SEASON_MONTHS = [5, 6, 7, 8]  # May through September (Northern Hemisphere)
DEFAULT_YEAR = 2023

# Performance tuning
NORMALIZATION_SCALE = 100  # Scale (meters) for min/max computation (higher = faster, lower = more accurate)

# DEM source
DEM_COLLECTION = 'USGS/SRTMGL1_003'  # SRTM 30m

# Satellite collections
LANDSAT_COLLECTION = 'LANDSAT/LC08/C02/T1_L2'  # Landsat 8 Collection 2
SENTINEL_COLLECTION = 'COPERNICUS/S2_SR_HARMONIZED'  # Sentinel-2 Surface Reflectance

# Water mask dataset
WATER_DATASET = 'JRC/GSW1_4/GlobalSurfaceWater'
WATER_OCCURRENCE_THRESHOLD = 50  # Mask pixels with >50% water occurrence

# Normalization parameters
SLOPE_OPTIMAL_MIN = 20  # Degrees - lower bound of optimal slope
SLOPE_OPTIMAL_MAX = 45  # Degrees - upper bound of optimal slope
SLOPE_SIGMOID_WIDTH = 15  # Degrees - width of sigmoid transition

# Cloud masking
COVERAGE_THRESHOLD = 0.3  # Minimum 30% valid pixels after cloud masking

# Temporal interpolation
INTERPOLATION_WEIGHTS = {
    'current': 0.5,
    'previous': 0.25,
    'next': 0.25,
}

# Visualization
COLOR_PALETTE = ['0000FF', 'FFFF00', 'FF0000']  # Blue -> Yellow -> Red
