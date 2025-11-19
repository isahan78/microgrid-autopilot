"""
Weather forecast API integration using Open-Meteo.

Free API, no key required. Provides GHI, DNI, temperature forecasts.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import requests
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import get_config


# Load configuration
config = get_config()

# Cache directory
PROJECT_DIR = Path(__file__).parent.parent
CACHE_DIR = PROJECT_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Open-Meteo API endpoint
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_weather_forecast(
    latitude: float = None,
    longitude: float = None,
    forecast_days: int = 2,
    use_cache: bool = True,
    cache_hours: int = 1
) -> Optional[pd.DataFrame]:
    """
    Fetch weather forecast from Open-Meteo API.

    Parameters:
    -----------
    latitude : float
        Location latitude (default from config)
    longitude : float
        Location longitude (default from config)
    forecast_days : int
        Number of days to forecast (1-16)
    use_cache : bool
        Whether to use cached data if available
    cache_hours : int
        Cache validity in hours

    Returns:
    --------
    pd.DataFrame with columns: timestamp, ghi, dni, temperature
    """
    # Get location from config if not provided
    weather_config = config.get('weather', default={}) or {}
    if latitude is None:
        latitude = weather_config.get('latitude', 32.65)  # Default: San Diego
    if longitude is None:
        longitude = weather_config.get('longitude', -117.15)

    # Check cache
    cache_file = CACHE_DIR / f"weather_forecast_{latitude}_{longitude}.json"

    if use_cache and cache_file.exists():
        cache_age = datetime.now().timestamp() - cache_file.stat().st_mtime
        if cache_age < cache_hours * 3600:
            print(f"  Using cached weather data ({cache_age/60:.0f} min old)")
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            return _parse_api_response(cached_data)

    # API parameters
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'hourly': [
            'temperature_2m',
            'shortwave_radiation',  # GHI
            'direct_normal_irradiance',  # DNI
            'cloudcover',
            'weathercode'
        ],
        'timezone': 'auto',
        'forecast_days': forecast_days
    }

    try:
        print(f"  Fetching weather forecast from Open-Meteo...")
        print(f"  Location: {latitude}, {longitude}")

        response = requests.get(OPEN_METEO_URL, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        # Cache the response
        with open(cache_file, 'w') as f:
            json.dump(data, f)

        print(f"  Received {len(data.get('hourly', {}).get('time', []))} hourly forecasts")

        return _parse_api_response(data)

    except requests.exceptions.RequestException as e:
        print(f"  Error fetching weather data: {e}")
        return None


def _parse_api_response(data: dict) -> pd.DataFrame:
    """Parse Open-Meteo API response into DataFrame."""
    hourly = data.get('hourly', {})

    if not hourly:
        return None

    df = pd.DataFrame({
        'timestamp': pd.to_datetime(hourly.get('time', [])),
        'temperature': hourly.get('temperature_2m', []),
        'ghi': hourly.get('shortwave_radiation', []),  # W/m²
        'dni': hourly.get('direct_normal_irradiance', []),  # W/m²
        'cloudcover': hourly.get('cloudcover', []),
        'weathercode': hourly.get('weathercode', [])
    })

    return df


def resample_to_15min(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample hourly data to 15-minute intervals.

    Uses linear interpolation for smooth transitions.
    """
    if df is None or df.empty:
        return df

    # Set timestamp as index
    df = df.set_index('timestamp')

    # Resample to 15 minutes with interpolation
    df_15min = df.resample('15min').interpolate(method='linear')

    # Reset index
    df_15min = df_15min.reset_index()

    return df_15min


def get_weather_for_optimization(
    start_time: datetime = None,
    hours: int = 48,
    latitude: float = None,
    longitude: float = None
) -> pd.DataFrame:
    """
    Get weather forecast formatted for optimization.

    Returns 15-minute interval data ready for the optimization pipeline.

    Parameters:
    -----------
    start_time : datetime
        Start time for forecast (default: now)
    hours : int
        Number of hours to forecast
    latitude : float
        Location latitude
    longitude : float
        Location longitude

    Returns:
    --------
    pd.DataFrame with columns: timestamp, ghi, dni, temperature
    """
    # Fetch forecast
    forecast_days = (hours // 24) + 1
    df = fetch_weather_forecast(
        latitude=latitude,
        longitude=longitude,
        forecast_days=min(forecast_days, 16)  # API limit
    )

    if df is None:
        return None

    # Resample to 15 minutes
    df = resample_to_15min(df)

    # Filter to requested time range
    if start_time is None:
        start_time = datetime.now().replace(minute=0, second=0, microsecond=0)

    end_time = start_time + timedelta(hours=hours)

    mask = (df['timestamp'] >= start_time) & (df['timestamp'] < end_time)
    df = df[mask].copy()

    # Select and rename columns
    df = df[['timestamp', 'ghi', 'dni', 'temperature']].copy()

    return df


def save_weather_forecast(df: pd.DataFrame, output_path: Path = None):
    """Save weather forecast to CSV."""
    if output_path is None:
        output_path = PROJECT_DIR / config.paths.get('data_processed', 'data_processed') / 'weather.csv'

    df.to_csv(output_path, index=False)
    print(f"  Weather forecast saved to {output_path}")


def main():
    """Fetch and save weather forecast."""
    print("=" * 50)
    print("MICROGRID AUTOPILOT - Weather Forecast API")
    print("=" * 50)

    # Get configuration
    weather_config = config.get('weather', default={})
    latitude = weather_config.get('latitude', 32.65) if weather_config else 32.65
    longitude = weather_config.get('longitude', -117.15) if weather_config else -117.15

    print(f"\nLocation: {latitude}, {longitude}")

    # Fetch forecast
    print("\nFetching weather forecast...")
    df = get_weather_for_optimization(
        hours=48,
        latitude=latitude,
        longitude=longitude
    )

    if df is None:
        print("Failed to fetch weather forecast")
        return

    print(f"\nForecast data:")
    print(f"  Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Records: {len(df)}")
    print(f"  GHI range: {df['ghi'].min():.0f} - {df['ghi'].max():.0f} W/m²")
    print(f"  Temperature range: {df['temperature'].min():.1f} - {df['temperature'].max():.1f} °C")

    # Save to processed data directory
    save_weather_forecast(df)

    print("\n" + "=" * 50)
    print("Weather forecast updated successfully")
    print("=" * 50)

    return df


if __name__ == "__main__":
    main()
