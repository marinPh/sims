# utils.py
"""Utility functions for thermal mapping system."""

import calendar
from typing import List, Dict
import ee
import geopandas as gpd

def generate_month_list(year: int, month_numbers: List[int]) -> List[Dict[str, any]]:
    """
    Generate list of date ranges for GEE filtering.

    Args:
        year: Year (e.g., 2023)
        month_numbers: List of month numbers (1-12)

    Returns:
        List of dicts with 'month', 'start', 'end' keys
        Example: [{'month': 5, 'start': '2023-05-01', 'end': '2023-05-31'}]
    """
    result = []

    for month in month_numbers:
        # Get last day of month (handles leap years automatically)
        last_day = calendar.monthrange(year, month)[1]

        result.append({
            'month': month,
            'start': f'{year}-{month:02d}-01',
            'end': f'{year}-{month:02d}-{last_day:02d}'
        })

    return result


def detect_hemisphere(roi_geometry: ee.Geometry) -> str:
    """
    Detect hemisphere from ROI centroid latitude.

    Args:
        roi_geometry: Earth Engine Geometry object

    Returns:
        'north' or 'south'

    Note:
        If ROI crosses equator, defaults to 'north' with warning
    """
    bounds = roi_geometry.bounds().getInfo()['coordinates'][0]
    lats = [coord[1] for coord in bounds]

    min_lat, max_lat = min(lats), max(lats)

    # Check for equator crossing
    if min_lat < 0 and max_lat > 0:
        print("⚠️  ROI crosses equator - using northern hemisphere aspect optimization")
        print("    Consider splitting analysis into two regions for better accuracy")
        return 'north'

    # Use centroid latitude
    centroid_lat = (min_lat + max_lat) / 2
    return 'north' if centroid_lat >= 0 else 'south'


def get_optimal_aspect_direction(hemisphere: str) -> Dict[str, any]:
    """
    Get optimal aspect direction for solar heating based on hemisphere.

    Args:
        hemisphere: 'north' or 'south'

    Returns:
        Dict with 'center' (degrees) and 'range' (tuple)

    Note:
        Northern hemisphere: south-facing slopes (180°, range 135-225°)
        Southern hemisphere: north-facing slopes (0°, range 315-45°)
    """
    if hemisphere == 'north':
        return {
            'center': 180,  # South
            'range': (135, 225)  # SW to SE
        }
    else:  # south
        return {
            'center': 0,  # North (also 360)
            'range': (315, 45)  # NW to NE
        }


def load_roi_from_geojson(filepath: str) -> ee.Geometry:
    """
    Load GeoJSON file and convert to Earth Engine Geometry.

    Args:
        filepath: Path to GeoJSON file

    Returns:
        ee.Geometry object

    Raises:
        ValueError: If file not found, invalid GeoJSON, or coordinates out of bounds
    """
    try:
        # Load with geopandas
        gdf = gpd.read_file(filepath)

        # Validate bounds
        bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
        if not (-180 <= bounds[0] <= 180 and -180 <= bounds[2] <= 180):
            raise ValueError(f"Longitude out of bounds: {bounds[0]}, {bounds[2]}")
        if not (-90 <= bounds[1] <= 90 and -90 <= bounds[3] <= 90):
            raise ValueError(f"Latitude out of bounds: {bounds[1]}, {bounds[3]}")

        # Check area and warn if large
        area_km2 = gdf.to_crs('EPSG:6933').area.sum() / 1e6  # Equal Area projection
        if area_km2 > 1_000_000:
            print(f"⚠️  Large ROI ({area_km2:,.0f} km²) - processing may be slow")

        # Convert to ee.Geometry
        geojson = gdf.geometry.iloc[0].__geo_interface__
        return ee.Geometry(geojson)

    except FileNotFoundError:
        raise ValueError(f"GeoJSON file not found: {filepath}")
    except Exception as e:
        raise ValueError(f"Invalid GeoJSON: {e}")
