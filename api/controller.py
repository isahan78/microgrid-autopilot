"""
API controller module for Microgrid Autopilot.

Handles business logic for API endpoints.
"""

import pandas as pd
import sys
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_absolute_error

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_prep.process_data import main as process_data_main
from forecasting.pv_forecast import main as pv_forecast_main
from forecasting.load_forecast import main as load_forecast_main
from optimization.mpc_solver import main as mpc_main, calculate_kpis
from simulation.power_flow import (
    simulate_power_flow, calculate_summary_metrics, compare_with_baseline
)
from simulation.battery_sim import BatterySimulator

from api.schemas import (
    ForecastResponse, TimeSeriesPoint, OptimizeResponse, OptimizationResult,
    KPIs, SimulateResponse, SimulationResult, SimulationMetrics,
    BaselineComparison, PipelineStatus, RunPipelineResponse
)


PROCESSED_DIR = Path(__file__).parent.parent / "data_processed"


class MicrogridController:
    """Controller for microgrid operations."""

    def __init__(self):
        """Initialize controller."""
        self.last_optimization_time = None
        self.last_data_time = None

    def run_forecast(self) -> ForecastResponse:
        """Run PV and load forecasting."""
        # Run forecasting
        pv_df = pv_forecast_main()
        load_df = load_forecast_main()

        # Convert to response format
        pv_points = [
            TimeSeriesPoint(timestamp=row['timestamp'], value=row['forecast_pv_mw'])
            for _, row in pv_df.iterrows()
        ]
        load_points = [
            TimeSeriesPoint(timestamp=row['timestamp'], value=row['forecast_load_mw'])
            for _, row in load_df.iterrows()
        ]

        # Calculate MAE
        pv_valid = pv_df.dropna()
        load_valid = load_df.dropna()
        pv_mae = mean_absolute_error(pv_valid['actual_pv_mw'], pv_valid['forecast_pv_mw'])
        load_mae = mean_absolute_error(load_valid['actual_load_mw'], load_valid['forecast_load_mw'])

        return ForecastResponse(
            pv_forecast=pv_points,
            load_forecast=load_points,
            pv_mae=pv_mae,
            load_mae=load_mae
        )

    def run_optimization(self) -> OptimizeResponse:
        """Run MPC optimization."""
        # Run optimization
        results_df, kpis = mpc_main()

        self.last_optimization_time = datetime.now()

        # Convert to response format
        results = [
            OptimizationResult(
                timestamp=row['timestamp'],
                pv_forecast_mw=row['pv_forecast_mw'],
                load_forecast_mw=row['load_forecast_mw'],
                battery_charge_mw=row['battery_charge_mw'],
                battery_discharge_mw=row['battery_discharge_mw'],
                grid_import_mw=row['grid_import_mw'],
                grid_export_mw=row['grid_export_mw'],
                soc_percent=row['soc_percent'],
                price_per_kwh=row['price_per_kwh']
            )
            for _, row in results_df.iterrows()
        ]

        kpis_response = KPIs(
            total_pv_mwh=kpis['total_pv_mwh'],
            total_load_mwh=kpis['total_load_mwh'],
            total_import_mwh=kpis['total_import_mwh'],
            total_export_mwh=kpis['total_export_mwh'],
            net_cost_usd=kpis['net_cost_usd'],
            total_carbon_kg=kpis['total_carbon_kg'],
            self_consumption_rate=kpis['self_consumption_rate'],
            peak_demand_mw=kpis['peak_demand_mw']
        )

        return OptimizeResponse(results=results, kpis=kpis_response)

    def run_simulation(self) -> SimulateResponse:
        """Run power flow simulation."""
        # Load optimization results
        opt_results = pd.read_csv(
            PROCESSED_DIR / "optimization_results.csv",
            parse_dates=['timestamp']
        )

        # Simulate power flow
        power_flow = simulate_power_flow(opt_results)
        metrics = calculate_summary_metrics(power_flow)
        comparison = compare_with_baseline(power_flow)

        # Save results
        power_flow.to_csv(PROCESSED_DIR / "power_flow.csv", index=False)

        # Convert to response format
        results = [
            SimulationResult(
                timestamp=row['timestamp'],
                pv_mw=row['pv_mw'],
                load_mw=row['load_mw'],
                battery_charge_mw=row['battery_charge_mw'],
                battery_discharge_mw=row['battery_discharge_mw'],
                grid_import_mw=row['grid_import_mw'],
                grid_export_mw=row['grid_export_mw'],
                soc_percent=row['soc_percent'],
                net_cost_usd=row['net_cost_usd'],
                carbon_emissions_kg=row['carbon_emissions_kg']
            )
            for _, row in power_flow.iterrows()
        ]

        metrics_response = SimulationMetrics(
            total_pv_mwh=metrics['total_pv_mwh'],
            total_load_mwh=metrics['total_load_mwh'],
            total_import_mwh=metrics['total_import_mwh'],
            total_export_mwh=metrics['total_export_mwh'],
            net_cost_usd=metrics['net_cost_usd'],
            total_carbon_kg=metrics['total_carbon_kg'],
            self_consumption_rate_pct=metrics['self_consumption_rate_pct'],
            self_sufficiency_pct=metrics['self_sufficiency_pct'],
            peak_import_mw=metrics['peak_import_mw']
        )

        comparison_response = BaselineComparison(
            baseline_cost_usd=comparison['baseline_cost_usd'],
            optimized_cost_usd=comparison['optimized_cost_usd'],
            cost_savings_usd=comparison['cost_savings_usd'],
            cost_savings_pct=comparison['cost_savings_pct'],
            carbon_reduction_kg=comparison['carbon_reduction_kg'],
            carbon_reduction_pct=comparison['carbon_reduction_pct']
        )

        return SimulateResponse(
            results=results,
            metrics=metrics_response,
            comparison=comparison_response
        )

    def run_full_pipeline(self, process_data=True, run_forecast=True,
                         run_optimization=True, run_simulation=True) -> RunPipelineResponse:
        """Run the complete pipeline."""
        steps = []
        kpis = None
        comparison = None

        try:
            # Step 1: Process data
            if process_data:
                try:
                    process_data_main()
                    steps.append(PipelineStatus(
                        step="data_processing",
                        status="success",
                        message="Data processing completed"
                    ))
                    self.last_data_time = datetime.now()
                except Exception as e:
                    steps.append(PipelineStatus(
                        step="data_processing",
                        status="error",
                        message=str(e)
                    ))
                    return RunPipelineResponse(status="error", steps=steps)

            # Step 2: Forecasting
            if run_forecast:
                try:
                    pv_forecast_main()
                    load_forecast_main()
                    steps.append(PipelineStatus(
                        step="forecasting",
                        status="success",
                        message="Forecasting completed"
                    ))
                except Exception as e:
                    steps.append(PipelineStatus(
                        step="forecasting",
                        status="error",
                        message=str(e)
                    ))
                    return RunPipelineResponse(status="error", steps=steps)

            # Step 3: Optimization
            if run_optimization:
                try:
                    results_df, kpis_dict = mpc_main()
                    kpis = KPIs(
                        total_pv_mwh=kpis_dict['total_pv_mwh'],
                        total_load_mwh=kpis_dict['total_load_mwh'],
                        total_import_mwh=kpis_dict['total_import_mwh'],
                        total_export_mwh=kpis_dict['total_export_mwh'],
                        net_cost_usd=kpis_dict['net_cost_usd'],
                        total_carbon_kg=kpis_dict['total_carbon_kg'],
                        self_consumption_rate=kpis_dict['self_consumption_rate'],
                        peak_demand_mw=kpis_dict['peak_demand_mw']
                    )
                    steps.append(PipelineStatus(
                        step="optimization",
                        status="success",
                        message="Optimization completed"
                    ))
                    self.last_optimization_time = datetime.now()
                except Exception as e:
                    steps.append(PipelineStatus(
                        step="optimization",
                        status="error",
                        message=str(e)
                    ))
                    return RunPipelineResponse(status="error", steps=steps, kpis=kpis)

            # Step 4: Simulation
            if run_simulation:
                try:
                    sim_response = self.run_simulation()
                    comparison = sim_response.comparison
                    steps.append(PipelineStatus(
                        step="simulation",
                        status="success",
                        message="Simulation completed"
                    ))
                except Exception as e:
                    steps.append(PipelineStatus(
                        step="simulation",
                        status="error",
                        message=str(e)
                    ))
                    return RunPipelineResponse(
                        status="error", steps=steps, kpis=kpis
                    )

            return RunPipelineResponse(
                status="success",
                steps=steps,
                kpis=kpis,
                comparison=comparison
            )

        except Exception as e:
            steps.append(PipelineStatus(
                step="unknown",
                status="error",
                message=str(e)
            ))
            return RunPipelineResponse(status="error", steps=steps)


# Global controller instance
controller = MicrogridController()
