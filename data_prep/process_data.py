"""
Data processing module for Microgrid Autopilot.

Processes raw PV, load, tariff, weather, and carbon data.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# Paths
RAW_DIR = Path(__file__).parent.parent / "data_raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data_processed"


def process_pv_data():
    """
    Process PV data:
    - Parse timestamps
    - Extract 2006-07-15 to 2006-07-16 (48-hour window)
    - Resample to 15 minutes
    - Save as timestamp, pv_power_mw
    """
    print("Processing PV data...")

    # Read raw PV data
    df = pd.read_csv(RAW_DIR / "pv_raw.csv")

    # Parse timestamps (format: MM/DD/YY HH:MM)
    df['timestamp'] = pd.to_datetime(df['LocalTime'], format='%m/%d/%y %H:%M')
    df = df.rename(columns={'Power(MW)': 'pv_power_mw'})
    df = df[['timestamp', 'pv_power_mw']]

    # Set timestamp as index for resampling
    df = df.set_index('timestamp')

    # Extract 48-hour window: 2006-07-15 to 2006-07-16
    start_time = pd.Timestamp('2006-07-15 00:00:00')
    end_time = pd.Timestamp('2006-07-16 23:59:59')
    df = df[start_time:end_time]

    # Resample to 15 minutes (mean aggregation for 5-min data)
    df = df.resample('15min').mean()

    # Reset index
    df = df.reset_index()

    # Ensure we have exactly 192 records (48 hours * 4 intervals/hour)
    df = df.head(192)

    # Save processed data
    df.to_csv(PROCESSED_DIR / "pv.csv", index=False)
    print(f"  Saved {len(df)} records to pv.csv")

    return df


def process_load_data(pv_timestamps):
    """
    Process load data:
    - Extract any 48-hour window
    - Resample hourly to 15 minutes
    - Align timestamps to PV window
    - Save as timestamp, load_mw
    """
    print("Processing load data...")

    # Read raw load data
    df = pd.read_csv(RAW_DIR / "load_raw.csv")

    # Parse timestamps
    df['timestamp'] = pd.to_datetime(df['utc_timestamp'])

    # Use DE (Germany) load as our load source - it's well populated
    # Scale down to microgrid size (divide by 10000 to get ~4-5 MW)
    load_col = 'DE_load_actual_entsoe_transparency'
    df = df[['timestamp', load_col]].dropna()
    df = df.rename(columns={load_col: 'load_mw'})

    # Scale to microgrid size (original is ~40,000 MW, scale to ~4 MW)
    df['load_mw'] = df['load_mw'] / 10000

    # Set timestamp as index
    df = df.set_index('timestamp')

    # Extract first 48-hour window with data
    start_time = df.index.min()
    end_time = start_time + pd.Timedelta(hours=48)
    df = df[start_time:end_time]

    # Resample to 15 minutes (interpolate hourly to 15-min)
    df = df.resample('15min').interpolate(method='linear')

    # Reset index and truncate to match PV length
    df = df.reset_index()
    n_records = len(pv_timestamps)
    df = df.head(n_records)

    # Align timestamps to PV window
    df['timestamp'] = pv_timestamps.values

    # Save processed data
    df.to_csv(PROCESSED_DIR / "load.csv", index=False)
    print(f"  Saved {len(df)} records to load.csv")

    return df


def process_tariff_data(pv_timestamps):
    """
    Create synthetic TOU tariff schedule:
    - 00-16: $0.12/kWh
    - 17-21: $0.30/kWh (peak)
    - 22-24: $0.15/kWh
    """
    print("Processing tariff data...")

    # Create tariff for each timestamp
    tariffs = []
    for ts in pv_timestamps:
        hour = ts.hour
        if 0 <= hour < 17:
            price = 0.12
        elif 17 <= hour < 22:
            price = 0.30
        else:
            price = 0.15
        tariffs.append({'timestamp': ts, 'price_per_kwh': price})

    df = pd.DataFrame(tariffs)

    # Save processed data
    df.to_csv(PROCESSED_DIR / "tariff.csv", index=False)
    print(f"  Saved {len(df)} records to tariff.csv")

    return df


def process_carbon_data(pv_timestamps):
    """
    Process carbon data:
    - Read monthly carbon intensity
    - Repeat for entire window
    - Save as timestamp, carbon_intensity
    """
    print("Processing carbon data...")

    # Read raw carbon data
    carbon_df = pd.read_csv(RAW_DIR / "carbon_raw.csv")

    # Create a mapping from month to carbon intensity
    carbon_map = dict(zip(carbon_df['month'], carbon_df['carbon_intensity_kg_per_mwh']))

    # Create carbon intensity for each timestamp
    records = []
    for ts in pv_timestamps:
        month = ts.month
        intensity = carbon_map.get(month, 400)  # default value
        records.append({'timestamp': ts, 'carbon_intensity': intensity})

    df = pd.DataFrame(records)

    # Save processed data
    df.to_csv(PROCESSED_DIR / "carbon.csv", index=False)
    print(f"  Saved {len(df)} records to carbon.csv")

    return df


def process_weather_data(pv_timestamps):
    """
    Process weather data:
    - Extract GHI, DNI, temperature
    - Resample to 15 minutes
    - Reassign timestamps to PV window
    - Save as timestamp, ghi, dni, temperature
    """
    print("Processing weather data...")

    # Read raw weather data (skip metadata rows)
    df = pd.read_csv(RAW_DIR / "weather_raw.csv", skiprows=2)

    # Create timestamp from year, month, day, hour, minute columns
    df['timestamp'] = pd.to_datetime(
        df[['Year', 'Month', 'Day', 'Hour', 'Minute']].astype(int).astype(str).agg('-'.join, axis=1),
        format='%Y-%m-%d-%H-%M'
    )

    # Extract relevant columns
    df = df[['timestamp', 'GHI', 'DNI', 'Temperature']]
    df = df.rename(columns={'GHI': 'ghi', 'DNI': 'dni', 'Temperature': 'temperature'})

    # Set timestamp as index
    df = df.set_index('timestamp')

    # Extract 48-hour window
    start_time = df.index.min()
    end_time = start_time + pd.Timedelta(hours=48)
    df = df[start_time:end_time]

    # Resample to 15 minutes (interpolate 30-min data)
    df = df.resample('15min').interpolate(method='linear')

    # Reset index and truncate to match PV length
    df = df.reset_index()
    n_records = len(pv_timestamps)
    df = df.head(n_records)

    # Reassign timestamps to PV window
    df['timestamp'] = pv_timestamps.values

    # Save processed data
    df.to_csv(PROCESSED_DIR / "weather.csv", index=False)
    print(f"  Saved {len(df)} records to weather.csv")

    return df


def main():
    """Run all data processing steps."""
    print("=" * 50)
    print("MICROGRID AUTOPILOT - Data Processing")
    print("=" * 50)

    # Ensure output directory exists
    PROCESSED_DIR.mkdir(exist_ok=True)

    # Process PV data first (defines the timestamp window)
    pv_df = process_pv_data()
    pv_timestamps = pv_df['timestamp']

    # Process other data aligned to PV timestamps
    load_df = process_load_data(pv_timestamps)
    tariff_df = process_tariff_data(pv_timestamps)
    carbon_df = process_carbon_data(pv_timestamps)
    weather_df = process_weather_data(pv_timestamps)

    print("\n" + "=" * 50)
    print("Data processing complete!")
    print("=" * 50)

    # Summary
    print("\nProcessed data summary:")
    print(f"  PV:      {len(pv_df)} records")
    print(f"  Load:    {len(load_df)} records")
    print(f"  Tariff:  {len(tariff_df)} records")
    print(f"  Carbon:  {len(carbon_df)} records")
    print(f"  Weather: {len(weather_df)} records")
    print(f"\nTime window: {pv_timestamps.iloc[0]} to {pv_timestamps.iloc[-1]}")


if __name__ == "__main__":
    main()
