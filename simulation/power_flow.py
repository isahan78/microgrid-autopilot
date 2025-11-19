"""
Power flow simulation module for Microgrid Autopilot.

Computes grid import/export, cost, carbon, and peak demand.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# Paths
PROCESSED_DIR = Path(__file__).parent.parent / "data_processed"


def simulate_power_flow(optimization_results):
    """
    Simulate power flow based on optimization results.

    net_power = pv + battery_discharge - battery_charge - load

    Outputs:
    - Grid import/export
    - Cost
    - Carbon emissions
    - Peak demand
    """
    results = []
    dt = 0.25  # 15 minutes

    for idx, row in optimization_results.iterrows():
        pv = row['pv_forecast_mw']
        load = row['load_forecast_mw']
        charge = row['battery_charge_mw']
        discharge = row['battery_discharge_mw']
        price = row['price_per_kwh']
        carbon_intensity = row['carbon_intensity']

        # Calculate net power
        net_power = pv + discharge - charge - load

        # Determine grid flows
        if net_power >= 0:
            grid_import = 0
            grid_export = net_power
        else:
            grid_import = -net_power
            grid_export = 0

        # Calculate costs
        import_cost = grid_import * price * 1000 * dt  # Convert to $/MWh
        export_revenue = grid_export * price * 1000 * 0.5 * dt  # 50% of import price
        net_cost = import_cost - export_revenue

        # Calculate carbon emissions
        carbon_emissions = grid_import * carbon_intensity * dt  # kg CO2

        results.append({
            'timestamp': row['timestamp'],
            'pv_mw': pv,
            'load_mw': load,
            'battery_charge_mw': charge,
            'battery_discharge_mw': discharge,
            'net_power_mw': net_power,
            'grid_import_mw': grid_import,
            'grid_export_mw': grid_export,
            'price_per_kwh': price,
            'carbon_intensity': carbon_intensity,
            'import_cost_usd': import_cost,
            'export_revenue_usd': export_revenue,
            'net_cost_usd': net_cost,
            'carbon_emissions_kg': carbon_emissions,
            'soc_percent': row.get('soc_percent', 50)
        })

    return pd.DataFrame(results)


def calculate_summary_metrics(power_flow_df):
    """Calculate summary metrics from power flow results."""
    dt = 0.25

    # Energy totals
    total_pv = power_flow_df['pv_mw'].sum() * dt
    total_load = power_flow_df['load_mw'].sum() * dt
    total_import = power_flow_df['grid_import_mw'].sum() * dt
    total_export = power_flow_df['grid_export_mw'].sum() * dt
    total_charge = power_flow_df['battery_charge_mw'].sum() * dt
    total_discharge = power_flow_df['battery_discharge_mw'].sum() * dt

    # Cost totals
    total_import_cost = power_flow_df['import_cost_usd'].sum()
    total_export_revenue = power_flow_df['export_revenue_usd'].sum()
    net_cost = power_flow_df['net_cost_usd'].sum()

    # Carbon totals
    total_carbon = power_flow_df['carbon_emissions_kg'].sum()

    # Peak demand
    peak_import = power_flow_df['grid_import_mw'].max()
    peak_export = power_flow_df['grid_export_mw'].max()

    # Self-sufficiency metrics
    self_consumption = total_pv - total_export
    self_consumption_rate = (self_consumption / total_pv * 100) if total_pv > 0 else 0
    self_sufficiency = ((total_load - total_import) / total_load * 100) if total_load > 0 else 0

    # Average prices
    avg_import_price = (total_import_cost / total_import / 1000) if total_import > 0 else 0

    return {
        'total_pv_mwh': total_pv,
        'total_load_mwh': total_load,
        'total_import_mwh': total_import,
        'total_export_mwh': total_export,
        'total_charge_mwh': total_charge,
        'total_discharge_mwh': total_discharge,
        'total_import_cost_usd': total_import_cost,
        'total_export_revenue_usd': total_export_revenue,
        'net_cost_usd': net_cost,
        'total_carbon_kg': total_carbon,
        'peak_import_mw': peak_import,
        'peak_export_mw': peak_export,
        'self_consumption_rate_pct': self_consumption_rate,
        'self_sufficiency_pct': self_sufficiency,
        'avg_import_price_per_kwh': avg_import_price
    }


def compare_with_baseline(power_flow_df):
    """
    Compare optimized results with baseline (no battery).

    Baseline: All excess PV exported, all deficit imported from grid.
    """
    dt = 0.25

    baseline_import = 0
    baseline_export = 0
    baseline_cost = 0
    baseline_carbon = 0

    for idx, row in power_flow_df.iterrows():
        pv = row['pv_mw']
        load = row['load_mw']
        price = row['price_per_kwh']
        carbon_intensity = row['carbon_intensity']

        net = pv - load
        if net >= 0:
            baseline_export += net * dt
            baseline_cost -= net * price * 1000 * 0.5 * dt
        else:
            baseline_import += (-net) * dt
            baseline_cost += (-net) * price * 1000 * dt
            baseline_carbon += (-net) * carbon_intensity * dt

    # Optimized totals
    opt_import = power_flow_df['grid_import_mw'].sum() * dt
    opt_cost = power_flow_df['net_cost_usd'].sum()
    opt_carbon = power_flow_df['carbon_emissions_kg'].sum()

    return {
        'baseline_import_mwh': baseline_import,
        'baseline_export_mwh': baseline_export,
        'baseline_cost_usd': baseline_cost,
        'baseline_carbon_kg': baseline_carbon,
        'optimized_import_mwh': opt_import,
        'optimized_cost_usd': opt_cost,
        'optimized_carbon_kg': opt_carbon,
        'cost_savings_usd': baseline_cost - opt_cost,
        'cost_savings_pct': ((baseline_cost - opt_cost) / baseline_cost * 100) if baseline_cost > 0 else 0,
        'carbon_reduction_kg': baseline_carbon - opt_carbon,
        'carbon_reduction_pct': ((baseline_carbon - opt_carbon) / baseline_carbon * 100) if baseline_carbon > 0 else 0
    }


def main():
    """Run power flow simulation."""
    print("=" * 50)
    print("MICROGRID AUTOPILOT - Power Flow Simulation")
    print("=" * 50)

    # Load optimization results
    opt_results = pd.read_csv(
        PROCESSED_DIR / "optimization_results.csv",
        parse_dates=['timestamp']
    )
    print(f"\nLoaded {len(opt_results)} optimization results")

    # Simulate power flow
    print("\nSimulating power flow...")
    power_flow = simulate_power_flow(opt_results)

    # Save results
    power_flow.to_csv(PROCESSED_DIR / "power_flow.csv", index=False)
    print(f"Results saved to power_flow.csv")

    # Calculate metrics
    metrics = calculate_summary_metrics(power_flow)
    comparison = compare_with_baseline(power_flow)

    # Print summary
    print("\n" + "=" * 50)
    print("POWER FLOW SUMMARY")
    print("=" * 50)
    print(f"  Total PV Generation:    {metrics['total_pv_mwh']:.2f} MWh")
    print(f"  Total Load:             {metrics['total_load_mwh']:.2f} MWh")
    print(f"  Grid Import:            {metrics['total_import_mwh']:.2f} MWh")
    print(f"  Grid Export:            {metrics['total_export_mwh']:.2f} MWh")
    print(f"  Battery Charge:         {metrics['total_charge_mwh']:.2f} MWh")
    print(f"  Battery Discharge:      {metrics['total_discharge_mwh']:.2f} MWh")

    print("\n  COST ANALYSIS")
    print(f"  Import Cost:            ${metrics['total_import_cost_usd']:.2f}")
    print(f"  Export Revenue:         ${metrics['total_export_revenue_usd']:.2f}")
    print(f"  Net Cost:               ${metrics['net_cost_usd']:.2f}")

    print("\n  ENVIRONMENTAL")
    print(f"  Carbon Emissions:       {metrics['total_carbon_kg']:.2f} kg CO2")

    print("\n  PERFORMANCE")
    print(f"  Peak Import:            {metrics['peak_import_mw']:.2f} MW")
    print(f"  Self-Consumption:       {metrics['self_consumption_rate_pct']:.1f}%")
    print(f"  Self-Sufficiency:       {metrics['self_sufficiency_pct']:.1f}%")

    print("\n" + "=" * 50)
    print("COMPARISON WITH BASELINE (No Battery)")
    print("=" * 50)
    print(f"  Baseline Cost:          ${comparison['baseline_cost_usd']:.2f}")
    print(f"  Optimized Cost:         ${comparison['optimized_cost_usd']:.2f}")
    print(f"  Cost Savings:           ${comparison['cost_savings_usd']:.2f} ({comparison['cost_savings_pct']:.1f}%)")
    print(f"  Carbon Reduction:       {comparison['carbon_reduction_kg']:.2f} kg ({comparison['carbon_reduction_pct']:.1f}%)")
    print("=" * 50)

    return power_flow, metrics, comparison


if __name__ == "__main__":
    main()
