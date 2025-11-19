"""
Generate synthetic training data for ML models.

Creates realistic PV and load data based on physics models and profiles
for training XGBoost models when historical data is not available.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import get_config
from utils.pv_model import calculate_pv_output

config = get_config()

PROJECT_DIR = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_DIR / config.paths.get('data_processed', 'data_processed')


def generate_synthetic_weather(days=30, start_date=None):
    """
    Generate synthetic weather data with realistic patterns.

    Parameters:
    -----------
    days : int
        Number of days to generate
    start_date : datetime
        Starting date (defaults to 30 days ago)

    Returns:
    --------
    pd.DataFrame with timestamp, ghi, dni, temperature
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=days)

    # Generate 15-minute timestamps
    timestamps = pd.date_range(start=start_date, periods=days * 96, freq='15min')

    weather_data = []

    for ts in timestamps:
        hour = ts.hour + ts.minute / 60
        day_of_year = ts.dayofyear

        # Solar angle approximation
        solar_noon = 12.5
        day_length = 14  # hours
        sunrise = solar_noon - day_length / 2
        sunset = solar_noon + day_length / 2

        if sunrise <= hour <= sunset:
            # Bell curve for solar radiation
            solar_progress = (hour - sunrise) / day_length
            solar_factor = np.sin(solar_progress * np.pi)

            # Base GHI with seasonal variation
            max_ghi = 900 + 100 * np.sin(2 * np.pi * (day_of_year - 172) / 365)

            # Add cloud variation
            cloud_factor = 0.8 + 0.2 * np.random.random()

            ghi = max_ghi * solar_factor * cloud_factor
            dni = ghi * (0.7 + 0.2 * np.random.random())
        else:
            ghi = 0
            dni = 0

        # Temperature with daily cycle
        base_temp = 20 + 5 * np.sin(2 * np.pi * (day_of_year - 172) / 365)
        daily_variation = 8 * np.sin(2 * np.pi * (hour - 6) / 24)
        noise = np.random.normal(0, 1)
        temperature = base_temp + daily_variation + noise

        weather_data.append({
            'timestamp': ts,
            'ghi': max(0, ghi),
            'dni': max(0, dni),
            'temperature': temperature
        })

    return pd.DataFrame(weather_data)


def generate_synthetic_pv(weather_df):
    """
    Generate synthetic PV output based on weather data.

    Uses physics model with added noise to simulate real-world variation.
    """
    # Calculate base PV output using physics model
    pv_output = calculate_pv_output(
        ghi=weather_df['ghi'].values,
        dni=weather_df['dni'].values,
        temperature=weather_df['temperature'].values,
        timestamps=weather_df['timestamp']
    )

    # Add realistic variation (inverter efficiency, soiling, etc.)
    variation = np.random.normal(1.0, 0.05, len(pv_output))
    pv_output = pv_output * variation

    # Ensure non-negative
    pv_output = np.clip(pv_output, 0, None)

    return pd.DataFrame({
        'timestamp': weather_df['timestamp'],
        'pv_power_mw': pv_output
    })


def generate_synthetic_load(timestamps):
    """
    Generate synthetic load data based on configured profile.

    Uses load profile from config with added realistic variation.
    """
    load_config = config.get('load_profile', default={}) or {}
    base_load = load_config.get('base_load_mw', 3.0)
    hourly_pattern = load_config.get('hourly_pattern', {})
    weekly_pattern = load_config.get('weekly_pattern', {})

    load_values = []

    for ts in timestamps:
        hour = ts.hour
        day_of_week = ts.dayofweek

        # Get multipliers
        hour_mult = hourly_pattern.get(hour, hourly_pattern.get(str(hour), 1.0))
        week_mult = weekly_pattern.get(day_of_week, weekly_pattern.get(str(day_of_week), 1.0))

        # Base load with patterns
        load = base_load * hour_mult * week_mult

        # Add realistic variation
        # - Random daily variation
        daily_factor = 0.95 + 0.1 * np.random.random()
        # - Short-term noise
        noise = np.random.normal(0, 0.1)

        load = load * daily_factor + noise
        load_values.append(max(0.5, load))

    return pd.DataFrame({
        'timestamp': timestamps,
        'load_mw': load_values
    })


def main(days=30):
    """Generate synthetic training data."""
    print("=" * 50)
    print("Generating Synthetic Training Data")
    print("=" * 50)

    print(f"\nGenerating {days} days of data...")

    # Generate weather
    print("  Generating weather data...")
    weather_df = generate_synthetic_weather(days=days)

    # Generate PV
    print("  Generating PV data...")
    pv_df = generate_synthetic_pv(weather_df)

    # Generate load
    print("  Generating load data...")
    load_df = generate_synthetic_load(weather_df['timestamp'])

    # Save to data_processed
    print("\nSaving to data_processed/...")
    weather_df.to_csv(PROCESSED_DIR / "weather.csv", index=False)
    pv_df.to_csv(PROCESSED_DIR / "pv.csv", index=False)
    load_df.to_csv(PROCESSED_DIR / "load.csv", index=False)

    print(f"  weather.csv: {len(weather_df)} records")
    print(f"  pv.csv: {len(pv_df)} records")
    print(f"  load.csv: {len(load_df)} records")

    # Summary stats
    print("\nData Summary:")
    print(f"  Date range: {weather_df['timestamp'].min()} to {weather_df['timestamp'].max()}")
    print(f"  Peak PV: {pv_df['pv_power_mw'].max():.2f} MW")
    print(f"  Avg Load: {load_df['load_mw'].mean():.2f} MW")
    print(f"  Peak Load: {load_df['load_mw'].max():.2f} MW")

    print("\n" + "=" * 50)
    print("Training data generated successfully!")
    print("=" * 50)

    return weather_df, pv_df, load_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=30, help='Days of data to generate')
    args = parser.parse_args()
    main(days=args.days)
