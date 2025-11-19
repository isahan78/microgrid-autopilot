"""
Fetch real historical data for ML model training.

Downloads historical weather from Open-Meteo archive API,
calculates PV output using physics model, and generates
realistic load profiles.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import requests
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import get_config
from utils.pv_model import calculate_pv_output

config = get_config()

PROJECT_DIR = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_DIR / config.paths.get('data_processed', 'data_processed')

# Open-Meteo Historical Archive API (free, no key needed)
ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_historical_weather(
    start_date: str,
    end_date: str,
    latitude: float = None,
    longitude: float = None
) -> pd.DataFrame:
    """
    Fetch historical weather data from Open-Meteo Archive API.

    Parameters:
    -----------
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    latitude : float
        Location latitude (from config if not provided)
    longitude : float
        Location longitude (from config if not provided)

    Returns:
    --------
    pd.DataFrame with timestamp, ghi, dni, temperature
    """
    weather_config = config.get('weather', default={}) or {}

    if latitude is None:
        latitude = weather_config.get('latitude', 32.65)
    if longitude is None:
        longitude = weather_config.get('longitude', -117.15)

    print(f"Fetching historical weather data...")
    print(f"  Location: {latitude}, {longitude}")
    print(f"  Period: {start_date} to {end_date}")

    params = {
        'latitude': latitude,
        'longitude': longitude,
        'start_date': start_date,
        'end_date': end_date,
        'hourly': [
            'temperature_2m',
            'shortwave_radiation',  # GHI
            'direct_normal_irradiance',  # DNI
            'cloudcover'
        ],
        'timezone': 'auto'
    }

    try:
        response = requests.get(ARCHIVE_API_URL, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching historical weather: {e}")
        return None

    # Parse response
    hourly = data.get('hourly', {})
    timestamps = hourly.get('time', [])

    if not timestamps:
        print("No data returned from API")
        return None

    weather_df = pd.DataFrame({
        'timestamp': pd.to_datetime(timestamps),
        'ghi': hourly.get('shortwave_radiation', [0] * len(timestamps)),
        'dni': hourly.get('direct_normal_irradiance', [0] * len(timestamps)),
        'temperature': hourly.get('temperature_2m', [20] * len(timestamps)),
        'cloudcover': hourly.get('cloudcover', [0] * len(timestamps))
    })

    # Handle missing values
    weather_df = weather_df.fillna(0)

    # Resample to 15-minute intervals
    weather_df = weather_df.set_index('timestamp')
    weather_df = weather_df.resample('15min').interpolate(method='linear')
    weather_df = weather_df.reset_index()

    print(f"  Retrieved {len(weather_df)} records (15-min intervals)")

    return weather_df


def generate_pv_from_weather(weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate PV output from historical weather using physics model.

    This gives realistic PV data based on actual weather conditions.
    """
    print("Generating PV output from weather data...")

    pv_output = calculate_pv_output(
        ghi=weather_df['ghi'].values,
        dni=weather_df['dni'].values,
        temperature=weather_df['temperature'].values,
        timestamps=weather_df['timestamp']
    )

    # Add small random variation to simulate real-world conditions
    # (inverter efficiency variations, soiling, etc.)
    variation = np.random.normal(1.0, 0.02, len(pv_output))
    pv_output = pv_output * variation
    pv_output = np.clip(pv_output, 0, None)

    pv_df = pd.DataFrame({
        'timestamp': weather_df['timestamp'],
        'pv_power_mw': pv_output
    })

    print(f"  Peak PV: {pv_df['pv_power_mw'].max():.2f} MW")
    print(f"  Total generation: {pv_df['pv_power_mw'].sum() * 0.25:.1f} MWh")

    return pv_df


def generate_realistic_load(
    timestamps: pd.DatetimeIndex,
    base_load: float = None
) -> pd.DataFrame:
    """
    Generate realistic load profile with proper patterns and variation.

    Uses configured load profile with realistic daily/weekly patterns
    and random variation that mimics real building behavior.
    """
    print("Generating realistic load profile...")

    load_config = config.get('load_profile', default={}) or {}

    if base_load is None:
        base_load = load_config.get('base_load_mw', 3.0)

    hourly_pattern = load_config.get('hourly_pattern', {})
    weekly_pattern = load_config.get('weekly_pattern', {})

    load_values = []

    # Track previous values for autocorrelation (realistic behavior)
    prev_load = base_load

    for i, ts in enumerate(timestamps):
        hour = ts.hour
        day_of_week = ts.dayofweek
        minute = ts.minute

        # Get hourly multiplier
        hour_mult = hourly_pattern.get(hour, hourly_pattern.get(str(hour), 1.0))

        # Interpolate for sub-hourly
        if minute > 0:
            next_hour = (hour + 1) % 24
            next_mult = hourly_pattern.get(next_hour, hourly_pattern.get(str(next_hour), 1.0))
            hour_mult = hour_mult + (next_mult - hour_mult) * (minute / 60)

        # Get weekly multiplier
        week_mult = weekly_pattern.get(day_of_week, weekly_pattern.get(str(day_of_week), 1.0))

        # Base load calculation
        load = base_load * hour_mult * week_mult

        # Add realistic variations:
        # 1. Daily random factor (consistent throughout day)
        day_of_year = ts.dayofyear
        np.random.seed(day_of_year + ts.year * 1000)
        daily_factor = np.random.normal(1.0, 0.05)

        # 2. Short-term autocorrelated noise (load doesn't change abruptly)
        np.random.seed(i)
        noise = np.random.normal(0, 0.03 * base_load)

        # Autocorrelation with previous value
        target_load = load * daily_factor + noise
        actual_load = 0.7 * target_load + 0.3 * prev_load

        # 3. Occasional spikes (equipment cycling)
        if np.random.random() < 0.02:  # 2% chance
            spike = np.random.uniform(0.1, 0.3) * base_load
            actual_load += spike

        actual_load = max(0.3 * base_load, actual_load)  # Minimum load
        load_values.append(actual_load)
        prev_load = actual_load

    load_df = pd.DataFrame({
        'timestamp': timestamps,
        'load_mw': load_values
    })

    print(f"  Average load: {load_df['load_mw'].mean():.2f} MW")
    print(f"  Peak load: {load_df['load_mw'].max():.2f} MW")
    print(f"  Min load: {load_df['load_mw'].min():.2f} MW")

    return load_df


def fetch_and_prepare_training_data(
    days: int = 90,
    end_date: str = None
) -> dict:
    """
    Fetch historical data and prepare for ML training.

    Parameters:
    -----------
    days : int
        Number of days of historical data to fetch
    end_date : str
        End date (defaults to yesterday)

    Returns:
    --------
    dict with weather_df, pv_df, load_df
    """
    print("=" * 60)
    print("FETCHING REAL HISTORICAL DATA FOR ML TRAINING")
    print("=" * 60)

    # Calculate date range
    if end_date is None:
        end = datetime.now() - timedelta(days=1)  # Yesterday
    else:
        end = datetime.strptime(end_date, '%Y-%m-%d')

    start = end - timedelta(days=days)

    start_str = start.strftime('%Y-%m-%d')
    end_str = end.strftime('%Y-%m-%d')

    # Fetch weather
    weather_df = fetch_historical_weather(start_str, end_str)

    if weather_df is None:
        print("Failed to fetch weather data!")
        return None

    # Generate PV from weather
    pv_df = generate_pv_from_weather(weather_df)

    # Generate load profile
    load_df = generate_realistic_load(weather_df['timestamp'])

    # Save to data_processed
    print("\nSaving training data...")

    weather_out = weather_df[['timestamp', 'ghi', 'dni', 'temperature']].copy()
    weather_out.to_csv(PROCESSED_DIR / "weather.csv", index=False)

    pv_df.to_csv(PROCESSED_DIR / "pv.csv", index=False)
    load_df.to_csv(PROCESSED_DIR / "load.csv", index=False)

    print(f"  weather.csv: {len(weather_out)} records")
    print(f"  pv.csv: {len(pv_df)} records")
    print(f"  load.csv: {len(load_df)} records")

    # Summary
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    print(f"Period: {start_str} to {end_str} ({days} days)")
    print(f"Records: {len(weather_df)} (15-minute intervals)")
    print(f"Location: {config.get('weather', default={}).get('latitude', 32.65)}, "
          f"{config.get('weather', default={}).get('longitude', -117.15)}")
    print()
    print("Weather:")
    print(f"  Avg Temperature: {weather_df['temperature'].mean():.1f}°C")
    print(f"  Avg GHI: {weather_df['ghi'].mean():.0f} W/m²")
    print()
    print("PV Generation:")
    print(f"  Total: {pv_df['pv_power_mw'].sum() * 0.25:.1f} MWh")
    print(f"  Peak: {pv_df['pv_power_mw'].max():.2f} MW")
    print(f"  Capacity Factor: {(pv_df['pv_power_mw'].mean() / config.get('pv_system', default={}).get('capacity_mw', 5.0) * 100):.1f}%")
    print()
    print("Load:")
    print(f"  Total: {load_df['load_mw'].sum() * 0.25:.1f} MWh")
    print(f"  Average: {load_df['load_mw'].mean():.2f} MW")
    print(f"  Peak: {load_df['load_mw'].max():.2f} MW")
    print("=" * 60)

    return {
        'weather': weather_df,
        'pv': pv_df,
        'load': load_df
    }


def main():
    """Main function to fetch historical data."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Fetch real historical data for ML training'
    )
    parser.add_argument(
        '--days', type=int, default=90,
        help='Number of days of historical data (default: 90)'
    )
    parser.add_argument(
        '--end-date', type=str, default=None,
        help='End date in YYYY-MM-DD format (default: yesterday)'
    )

    args = parser.parse_args()

    result = fetch_and_prepare_training_data(
        days=args.days,
        end_date=args.end_date
    )

    if result:
        print("\nTraining data ready! Now run:")
        print("  python forecasting/pv_forecast.py --retrain")
        print("  python forecasting/load_forecast.py --retrain")


if __name__ == "__main__":
    main()
