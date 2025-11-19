"""
Battery simulation module for Microgrid Autopilot.

Computes SOC trajectory and charge/discharge time series.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# Battery parameters
BATTERY_CAPACITY_MWH = 5.0
BATTERY_POWER_MW = 2.0
CHARGE_EFFICIENCY = 0.95
DISCHARGE_EFFICIENCY = 0.95
SOC_MIN = 0.2
SOC_MAX = 0.9


class BatterySimulator:
    """Battery simulation class."""

    def __init__(self, capacity_mwh=BATTERY_CAPACITY_MWH,
                 max_power_mw=BATTERY_POWER_MW,
                 charge_eff=CHARGE_EFFICIENCY,
                 discharge_eff=DISCHARGE_EFFICIENCY,
                 soc_min=SOC_MIN, soc_max=SOC_MAX):
        """Initialize battery simulator."""
        self.capacity_mwh = capacity_mwh
        self.max_power_mw = max_power_mw
        self.charge_eff = charge_eff
        self.discharge_eff = discharge_eff
        self.soc_min = soc_min
        self.soc_max = soc_max

    def simulate(self, schedule_df, initial_soc=0.5):
        """
        Simulate battery operation based on charge/discharge schedule.

        Args:
            schedule_df: DataFrame with battery_charge_mw, battery_discharge_mw columns
            initial_soc: Initial state of charge (0-1)

        Returns:
            DataFrame with SOC trajectory and energy flows
        """
        results = []
        soc = initial_soc * self.capacity_mwh
        dt = 0.25  # 15 minutes

        for idx, row in schedule_df.iterrows():
            # Get scheduled actions
            charge_cmd = row.get('battery_charge_mw', 0)
            discharge_cmd = row.get('battery_discharge_mw', 0)

            # Apply power limits
            charge_cmd = np.clip(charge_cmd, 0, self.max_power_mw)
            discharge_cmd = np.clip(discharge_cmd, 0, self.max_power_mw)

            # Apply SOC limits
            max_charge = (self.soc_max * self.capacity_mwh - soc) / (self.charge_eff * dt)
            max_discharge = (soc - self.soc_min * self.capacity_mwh) * self.discharge_eff / dt

            actual_charge = min(charge_cmd, max_charge)
            actual_discharge = min(discharge_cmd, max_discharge)

            # Ensure non-negative
            actual_charge = max(0, actual_charge)
            actual_discharge = max(0, actual_discharge)

            # Update SOC
            energy_in = actual_charge * self.charge_eff * dt
            energy_out = actual_discharge / self.discharge_eff * dt
            soc = soc + energy_in - energy_out

            # Record results
            results.append({
                'timestamp': row.get('timestamp', idx),
                'charge_command_mw': charge_cmd,
                'discharge_command_mw': discharge_cmd,
                'actual_charge_mw': actual_charge,
                'actual_discharge_mw': actual_discharge,
                'energy_charged_mwh': energy_in,
                'energy_discharged_mwh': energy_out,
                'soc_mwh': soc,
                'soc_percent': soc / self.capacity_mwh * 100,
                'available_charge_mw': max_charge,
                'available_discharge_mw': max_discharge
            })

        return pd.DataFrame(results)

    def get_state(self, soc_mwh):
        """Get battery state information."""
        soc_percent = soc_mwh / self.capacity_mwh * 100
        available_energy = soc_mwh - self.soc_min * self.capacity_mwh
        available_capacity = self.soc_max * self.capacity_mwh - soc_mwh

        return {
            'soc_mwh': soc_mwh,
            'soc_percent': soc_percent,
            'available_energy_mwh': available_energy,
            'available_capacity_mwh': available_capacity,
            'max_charge_power_mw': self.max_power_mw,
            'max_discharge_power_mw': self.max_power_mw
        }


def main():
    """Run battery simulation."""
    print("=" * 50)
    print("MICROGRID AUTOPILOT - Battery Simulation")
    print("=" * 50)

    # Load optimization results
    PROCESSED_DIR = Path(__file__).parent.parent / "data_processed"
    schedule_df = pd.read_csv(
        PROCESSED_DIR / "optimization_results.csv",
        parse_dates=['timestamp']
    )

    print(f"\nLoaded {len(schedule_df)} scheduled time steps")

    # Create simulator
    simulator = BatterySimulator()

    # Run simulation
    print("\nRunning battery simulation...")
    results = simulator.simulate(schedule_df)

    # Save results
    results.to_csv(PROCESSED_DIR / "battery_simulation.csv", index=False)
    print(f"Results saved to battery_simulation.csv")

    # Summary statistics
    print("\n" + "=" * 50)
    print("BATTERY SIMULATION SUMMARY")
    print("=" * 50)
    print(f"  Total Energy Charged:     {results['energy_charged_mwh'].sum():.2f} MWh")
    print(f"  Total Energy Discharged:  {results['energy_discharged_mwh'].sum():.2f} MWh")
    print(f"  Round-trip Efficiency:    {results['energy_discharged_mwh'].sum() / results['energy_charged_mwh'].sum() * 100:.1f}%")
    print(f"  Initial SOC:              {results['soc_percent'].iloc[0]:.1f}%")
    print(f"  Final SOC:                {results['soc_percent'].iloc[-1]:.1f}%")
    print(f"  Min SOC:                  {results['soc_percent'].min():.1f}%")
    print(f"  Max SOC:                  {results['soc_percent'].max():.1f}%")
    print("=" * 50)

    return results


if __name__ == "__main__":
    main()
