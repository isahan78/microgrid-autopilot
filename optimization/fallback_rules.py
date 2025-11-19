"""
Fallback rules module for Microgrid Autopilot.

Rule-based charging/discharging logic when optimization fails.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# Battery parameters (same as MPC solver)
BATTERY_CAPACITY_MWH = 5.0
BATTERY_POWER_MW = 2.0
CHARGE_EFFICIENCY = 0.95
DISCHARGE_EFFICIENCY = 0.95
SOC_MIN = 0.2
SOC_MAX = 0.9
SOC_INITIAL = 0.5


def apply_fallback_rules(data):
    """
    Apply simple rule-based battery control:
    - Charge during cheap hours (price < 0.15)
    - Discharge during expensive hours (price >= 0.25)
    - Use excess PV to charge battery
    - Discharge battery to meet load when PV is insufficient
    """
    results = []
    soc = SOC_INITIAL * BATTERY_CAPACITY_MWH  # Current SOC in MWh
    dt = 0.25  # 15 minutes

    for idx, row in data.iterrows():
        pv = row['forecast_pv_mw']
        load = row['forecast_load_mw']
        price = row['price_per_kwh']
        carbon = row['carbon_intensity']

        # Initialize battery actions
        charge = 0.0
        discharge = 0.0

        # Net power (positive = excess, negative = deficit)
        net_power = pv - load

        if net_power > 0:
            # Excess PV: charge battery
            available_charge_capacity = (SOC_MAX * BATTERY_CAPACITY_MWH - soc) / (CHARGE_EFFICIENCY * dt)
            charge = min(net_power, BATTERY_POWER_MW, available_charge_capacity)
            charge = max(0, charge)
        else:
            # Deficit: consider discharging battery
            deficit = -net_power

            # Discharge during peak hours or when there's a deficit
            if price >= 0.25:  # Peak pricing
                available_discharge = (soc - SOC_MIN * BATTERY_CAPACITY_MWH) * DISCHARGE_EFFICIENCY / dt
                discharge = min(deficit, BATTERY_POWER_MW, available_discharge)
                discharge = max(0, discharge)
            elif price < 0.15:  # Off-peak: charge if we have deficit but cheap power
                # Still meet load from grid, but also charge battery
                available_charge_capacity = (SOC_MAX * BATTERY_CAPACITY_MWH - soc) / (CHARGE_EFFICIENCY * dt)
                charge = min(BATTERY_POWER_MW * 0.5, available_charge_capacity)  # Charge at 50% rate
                charge = max(0, charge)
            else:
                # Mid-peak: discharge to meet deficit
                available_discharge = (soc - SOC_MIN * BATTERY_CAPACITY_MWH) * DISCHARGE_EFFICIENCY / dt
                discharge = min(deficit, BATTERY_POWER_MW, available_discharge)
                discharge = max(0, discharge)

        # Update SOC
        soc_change = charge * CHARGE_EFFICIENCY * dt - discharge / DISCHARGE_EFFICIENCY * dt
        soc = soc + soc_change
        soc = np.clip(soc, SOC_MIN * BATTERY_CAPACITY_MWH, SOC_MAX * BATTERY_CAPACITY_MWH)

        # Calculate grid flows
        net_with_battery = pv + discharge - charge - load
        if net_with_battery >= 0:
            grid_export = net_with_battery
            grid_import = 0
        else:
            grid_import = -net_with_battery
            grid_export = 0

        results.append({
            'timestamp': row['timestamp'],
            'pv_forecast_mw': pv,
            'load_forecast_mw': load,
            'battery_charge_mw': charge,
            'battery_discharge_mw': discharge,
            'grid_import_mw': grid_import,
            'grid_export_mw': grid_export,
            'soc_mwh': soc,
            'soc_percent': soc / BATTERY_CAPACITY_MWH * 100,
            'price_per_kwh': price,
            'carbon_intensity': carbon
        })

    return pd.DataFrame(results)


def main():
    """Test fallback rules with sample data."""
    print("=" * 50)
    print("MICROGRID AUTOPILOT - Fallback Rules Test")
    print("=" * 50)

    # Load data
    PROCESSED_DIR = Path(__file__).parent.parent / "data_processed"

    pv_df = pd.read_csv(PROCESSED_DIR / "forecast_pv.csv", parse_dates=['timestamp'])
    load_df = pd.read_csv(PROCESSED_DIR / "forecast_load.csv", parse_dates=['timestamp'])
    tariff_df = pd.read_csv(PROCESSED_DIR / "tariff.csv", parse_dates=['timestamp'])
    carbon_df = pd.read_csv(PROCESSED_DIR / "carbon.csv", parse_dates=['timestamp'])

    # Merge data
    data = pv_df[['timestamp', 'forecast_pv_mw']].merge(
        load_df[['timestamp', 'forecast_load_mw']], on='timestamp'
    ).merge(
        tariff_df, on='timestamp'
    ).merge(
        carbon_df, on='timestamp'
    )

    # Apply rules
    results = apply_fallback_rules(data)

    print(f"\nProcessed {len(results)} time steps")
    print(f"Final SOC: {results['soc_percent'].iloc[-1]:.1f}%")

    return results


if __name__ == "__main__":
    main()
