"""
Physics-based PV generation model.

Calculates solar power output from weather data without requiring
historical training data. Uses standard PV performance equations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import get_config

config = get_config()


def calculate_pv_output(
    ghi: np.ndarray,
    dni: np.ndarray,
    temperature: np.ndarray,
    timestamps: pd.DatetimeIndex = None,
    system_capacity_mw: float = None,
    panel_efficiency: float = None,
    temperature_coefficient: float = None,
    inverter_efficiency: float = None,
    system_losses: float = None
) -> np.ndarray:
    """
    Calculate PV power output using physics-based model.

    Parameters:
    -----------
    ghi : np.ndarray
        Global Horizontal Irradiance in W/m²
    dni : np.ndarray
        Direct Normal Irradiance in W/m²
    temperature : np.ndarray
        Ambient temperature in °C
    timestamps : pd.DatetimeIndex
        Timestamps for the data (optional, for time-based adjustments)
    system_capacity_mw : float
        Rated system capacity in MW (from config if not provided)
    panel_efficiency : float
        Panel efficiency at STC (0-1)
    temperature_coefficient : float
        Power temperature coefficient (%/°C, typically -0.4 to -0.5)
    inverter_efficiency : float
        Inverter efficiency (0-1)
    system_losses : float
        Other system losses (soiling, mismatch, wiring, etc.) (0-1)

    Returns:
    --------
    np.ndarray : PV power output in MW
    """
    # Get PV system parameters from config
    pv_config = config.get('pv_system', default={}) or {}

    if system_capacity_mw is None:
        system_capacity_mw = pv_config.get('capacity_mw', 5.0)
    if panel_efficiency is None:
        panel_efficiency = pv_config.get('panel_efficiency', 0.20)
    if temperature_coefficient is None:
        temperature_coefficient = pv_config.get('temperature_coefficient', -0.004)
    if inverter_efficiency is None:
        inverter_efficiency = pv_config.get('inverter_efficiency', 0.96)
    if system_losses is None:
        system_losses = pv_config.get('system_losses', 0.14)

    # Standard Test Conditions
    STC_IRRADIANCE = 1000  # W/m²
    STC_TEMPERATURE = 25  # °C
    NOCT = 45  # Nominal Operating Cell Temperature

    # Convert to numpy arrays
    ghi = np.array(ghi)
    dni = np.array(dni)
    temperature = np.array(temperature)

    # Calculate effective irradiance
    # Simple model: use GHI as primary, with DNI contribution
    # More sophisticated models would include tilt/azimuth calculations
    effective_irradiance = ghi + 0.1 * dni  # Simplified POA calculation

    # Calculate cell temperature (simplified NOCT model)
    cell_temperature = temperature + (NOCT - 20) * (effective_irradiance / 800)

    # Temperature derating
    temperature_factor = 1 + temperature_coefficient * (cell_temperature - STC_TEMPERATURE)
    temperature_factor = np.clip(temperature_factor, 0.7, 1.1)  # Reasonable bounds

    # Calculate DC power output
    # P_dc = (G / G_stc) * P_rated * temp_factor
    dc_power = (effective_irradiance / STC_IRRADIANCE) * system_capacity_mw * temperature_factor

    # Apply efficiencies and losses
    ac_power = dc_power * inverter_efficiency * (1 - system_losses)

    # Ensure non-negative output
    ac_power = np.clip(ac_power, 0, system_capacity_mw)

    # Night time: zero output when GHI is very low
    ac_power = np.where(ghi < 10, 0, ac_power)

    return ac_power


def generate_pv_forecast(weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate PV forecast from weather data.

    Parameters:
    -----------
    weather_df : pd.DataFrame
        Weather data with columns: timestamp, ghi, dni, temperature

    Returns:
    --------
    pd.DataFrame with timestamp and forecast_pv_mw columns
    """
    pv_output = calculate_pv_output(
        ghi=weather_df['ghi'].values,
        dni=weather_df['dni'].values,
        temperature=weather_df['temperature'].values,
        timestamps=weather_df['timestamp']
    )

    result = pd.DataFrame({
        'timestamp': weather_df['timestamp'],
        'forecast_pv_mw': pv_output
    })

    return result


def main():
    """Test the PV model with sample data."""
    print("=" * 50)
    print("Physics-Based PV Model Test")
    print("=" * 50)

    # Create sample data
    hours = pd.date_range(start='2024-01-01', periods=24, freq='h')

    # Typical daily GHI pattern
    ghi = np.array([0, 0, 0, 0, 0, 50, 200, 400, 600, 750, 850, 900,
                    900, 850, 750, 600, 400, 200, 50, 0, 0, 0, 0, 0])
    dni = ghi * 0.8  # Approximate DNI
    temperature = np.array([15, 14, 14, 13, 13, 14, 16, 18, 20, 22, 24, 26,
                            27, 28, 28, 27, 25, 23, 20, 18, 17, 16, 15, 15])

    weather_df = pd.DataFrame({
        'timestamp': hours,
        'ghi': ghi,
        'dni': dni,
        'temperature': temperature
    })

    # Generate forecast
    result = generate_pv_forecast(weather_df)

    print("\nPV Output:")
    for i, row in result.iterrows():
        if row['forecast_pv_mw'] > 0:
            print(f"  {row['timestamp'].strftime('%H:%M')}: {row['forecast_pv_mw']:.2f} MW")

    print(f"\nPeak output: {result['forecast_pv_mw'].max():.2f} MW")
    print(f"Daily generation: {result['forecast_pv_mw'].sum():.2f} MWh")

    return result


if __name__ == "__main__":
    main()
