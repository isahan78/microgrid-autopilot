"""
Rolling Horizon MPC Scheduler for Microgrid Autopilot.

Runs optimization periodically, updating forecasts and executing
only the first time step's decisions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import time
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import get_config
from utils.weather_api import fetch_weather_forecast, resample_to_15min
from optimization.mpc_solver import (
    build_optimization_model, solve_optimization,
    extract_results, calculate_kpis
)

# Load configuration
config = get_config()

# Paths
PROJECT_DIR = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_DIR / config.paths.get('data_processed', 'data_processed')
RESULTS_DIR = PROJECT_DIR / config.paths.get('results', 'results')
RESULTS_DIR.mkdir(exist_ok=True)


class RollingMPCScheduler:
    """
    Rolling Horizon MPC Scheduler.

    Runs optimization at regular intervals, updating forecasts
    and tracking actual vs planned operations.
    """

    def __init__(
        self,
        interval_minutes: int = 15,
        horizon_hours: int = 48,
        initial_soc: float = None
    ):
        """
        Initialize the scheduler.

        Parameters:
        -----------
        interval_minutes : int
            How often to run optimization (default: 15 min)
        horizon_hours : int
            Optimization lookahead (default: 48 hours)
        initial_soc : float
            Initial battery SOC (0-1), default from config
        """
        self.interval_minutes = interval_minutes
        self.horizon_hours = horizon_hours

        # Get battery parameters
        battery = config.battery
        self.soc_initial = initial_soc or battery.get('soc_initial', 0.5)
        self.battery_capacity = battery.get('capacity_mwh', 5.0)

        # State tracking
        self.current_soc = self.soc_initial
        self.execution_history = []
        self.optimization_count = 0

        # Results storage
        self.results_log = []

    def get_current_data(self) -> pd.DataFrame:
        """
        Get current forecast data for optimization.

        In production, this would fetch:
        - Latest weather forecast
        - Current load forecast
        - Real-time PV output
        - Current battery SOC
        """
        # Fetch latest weather
        weather_df = fetch_weather_forecast(
            forecast_days=min(self.horizon_hours // 24 + 1, 16)
        )

        if weather_df is None:
            print("  Warning: Using cached weather data")
            weather_df = pd.read_csv(
                PROCESSED_DIR / "weather.csv",
                parse_dates=['timestamp']
            )
        else:
            weather_df = resample_to_15min(weather_df)

        # Load current forecasts
        pv_df = pd.read_csv(
            PROCESSED_DIR / "forecast_pv.csv",
            parse_dates=['timestamp']
        )
        load_df = pd.read_csv(
            PROCESSED_DIR / "forecast_load.csv",
            parse_dates=['timestamp']
        )
        tariff_df = pd.read_csv(
            PROCESSED_DIR / "tariff.csv",
            parse_dates=['timestamp']
        )
        carbon_df = pd.read_csv(
            PROCESSED_DIR / "carbon.csv",
            parse_dates=['timestamp']
        )

        # Merge all data
        data = pv_df[['timestamp', 'forecast_pv_mw']].merge(
            load_df[['timestamp', 'forecast_load_mw']], on='timestamp'
        ).merge(
            tariff_df, on='timestamp'
        ).merge(
            carbon_df, on='timestamp'
        )

        # Apply time-varying carbon intensity
        if config.carbon.get('use_time_varying', False):
            hourly_multipliers = config.carbon.get('hourly_multipliers', {})
            if hourly_multipliers:
                data['hour'] = data['timestamp'].dt.hour
                data['carbon_multiplier'] = data['hour'].map(
                    lambda h: hourly_multipliers.get(h, hourly_multipliers.get(str(h), 1.0))
                )
                data['carbon_intensity'] = data['carbon_intensity'] * data['carbon_multiplier']
                data = data.drop(columns=['hour', 'carbon_multiplier'])

        return data

    def run_single_optimization(self) -> dict:
        """
        Run a single optimization cycle.

        Returns:
        --------
        dict with optimization results and control actions
        """
        self.optimization_count += 1
        timestamp = datetime.now()

        print(f"\n{'='*50}")
        print(f"ROLLING MPC - Optimization #{self.optimization_count}")
        print(f"Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Current SOC: {self.current_soc*100:.1f}%")
        print(f"{'='*50}")

        # Get current data
        print("\nFetching current data...")
        data = self.get_current_data()
        horizon = min(len(data), self.horizon_hours * 4)  # 4 intervals per hour

        print(f"  Horizon: {horizon} steps ({horizon * 15 / 60:.1f} hours)")

        # Update initial SOC in config for this run
        battery = config.battery
        original_soc = battery.get('soc_initial', 0.5)
        battery['soc_initial'] = self.current_soc

        # Build and solve optimization
        print("\nBuilding optimization model...")
        model = build_optimization_model(data, horizon)

        print("Solving optimization...")
        solved_model = solve_optimization(model)

        # Restore original config
        battery['soc_initial'] = original_soc

        if solved_model is None:
            print("  Optimization failed!")
            return None

        # Extract results
        results_df = extract_results(solved_model, data, horizon)
        kpis = calculate_kpis(results_df)

        # Get first time step's control actions
        first_step = results_df.iloc[0]
        control_actions = {
            'timestamp': timestamp,
            'battery_charge_mw': first_step['battery_charge_mw'],
            'battery_discharge_mw': first_step['battery_discharge_mw'],
            'grid_import_mw': first_step['grid_import_mw'],
            'grid_export_mw': first_step['grid_export_mw'],
            'soc_after': first_step['soc_percent'] / 100,
            'pv_forecast_mw': first_step['pv_forecast_mw'],
            'load_forecast_mw': first_step['load_forecast_mw'],
            'price_per_kwh': first_step['price_per_kwh']
        }

        # Print control actions
        print(f"\n{'='*50}")
        print("CONTROL ACTIONS (Next {0} min)".format(self.interval_minutes))
        print(f"{'='*50}")
        print(f"  Battery Charge:    {control_actions['battery_charge_mw']:.3f} MW")
        print(f"  Battery Discharge: {control_actions['battery_discharge_mw']:.3f} MW")
        print(f"  Grid Import:       {control_actions['grid_import_mw']:.3f} MW")
        print(f"  Grid Export:       {control_actions['grid_export_mw']:.3f} MW")
        print(f"  SOC After:         {control_actions['soc_after']*100:.1f}%")
        print(f"  Electricity Price: ${control_actions['price_per_kwh']:.3f}/kWh")

        # Print horizon KPIs
        print(f"\n{'='*50}")
        print("HORIZON KPIs")
        print(f"{'='*50}")
        print(f"  Projected Cost:    ${kpis['net_cost_usd']:.2f}")
        print(f"  Peak Demand:       {kpis['peak_demand_mw']:.2f} MW")
        print(f"  Self-Sufficiency:  {kpis['self_sufficiency_rate']:.1f}%")

        # Log results
        result_entry = {
            'optimization_num': self.optimization_count,
            'timestamp': timestamp.isoformat(),
            'soc_before': self.current_soc,
            'soc_after': control_actions['soc_after'],
            **control_actions,
            'horizon_cost': kpis['net_cost_usd'],
            'horizon_peak': kpis['peak_demand_mw']
        }
        self.results_log.append(result_entry)

        # Update current SOC for next iteration
        self.current_soc = control_actions['soc_after']

        return {
            'control_actions': control_actions,
            'kpis': kpis,
            'results_df': results_df
        }

    def run_continuous(self, duration_hours: float = 1, simulate_time: bool = True):
        """
        Run continuous rolling horizon optimization.

        Parameters:
        -----------
        duration_hours : float
            How long to run the scheduler
        simulate_time : bool
            If True, simulate time passing (for testing)
            If False, actually wait between optimizations
        """
        print("=" * 60)
        print("ROLLING HORIZON MPC SCHEDULER")
        print("=" * 60)
        print(f"Interval: {self.interval_minutes} minutes")
        print(f"Horizon: {self.horizon_hours} hours")
        print(f"Duration: {duration_hours} hours")
        print(f"Initial SOC: {self.current_soc*100:.1f}%")
        print("=" * 60)

        num_iterations = int(duration_hours * 60 / self.interval_minutes)

        for i in range(num_iterations):
            # Run optimization
            result = self.run_single_optimization()

            if result is None:
                print("Optimization failed, waiting for next cycle...")

            # Wait for next interval
            if not simulate_time and i < num_iterations - 1:
                wait_seconds = self.interval_minutes * 60
                print(f"\nWaiting {self.interval_minutes} minutes for next optimization...")
                time.sleep(wait_seconds)
            elif simulate_time:
                # Simulate passage of time
                pass

        # Save results log
        self.save_results()

        print("\n" + "=" * 60)
        print("ROLLING MPC COMPLETE")
        print("=" * 60)
        print(f"Total optimizations: {self.optimization_count}")
        print(f"Final SOC: {self.current_soc*100:.1f}%")

    def save_results(self):
        """Save optimization results to file."""
        if not self.results_log:
            return

        results_df = pd.DataFrame(self.results_log)
        output_path = RESULTS_DIR / f"rolling_mpc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")

        # Also save as JSON for detailed analysis
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(self.results_log, f, indent=2, default=str)


def main():
    """Run rolling horizon MPC demonstration."""
    import argparse

    parser = argparse.ArgumentParser(description='Rolling Horizon MPC Scheduler')
    parser.add_argument('--interval', type=int, default=15,
                        help='Optimization interval in minutes')
    parser.add_argument('--horizon', type=int, default=48,
                        help='Optimization horizon in hours')
    parser.add_argument('--duration', type=float, default=1,
                        help='Run duration in hours')
    parser.add_argument('--soc', type=float, default=None,
                        help='Initial SOC (0-1)')
    parser.add_argument('--realtime', action='store_true',
                        help='Run in real-time (wait between optimizations)')

    args = parser.parse_args()

    # Create scheduler
    scheduler = RollingMPCScheduler(
        interval_minutes=args.interval,
        horizon_hours=args.horizon,
        initial_soc=args.soc
    )

    # Run scheduler
    scheduler.run_continuous(
        duration_hours=args.duration,
        simulate_time=not args.realtime
    )


if __name__ == "__main__":
    main()
