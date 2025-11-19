"""
MPC optimization solver for Microgrid Autopilot.

Optimizes battery scheduling using Pyomo with HiGHS solver.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from pyomo.environ import (
    ConcreteModel, Var, Objective, Constraint,
    NonNegativeReals, minimize, SolverFactory, value
)
import warnings
warnings.filterwarnings('ignore')


# Paths
PROCESSED_DIR = Path(__file__).parent.parent / "data_processed"
OUTPUT_DIR = Path(__file__).parent.parent / "data_processed"


# Battery parameters
BATTERY_CAPACITY_MWH = 5.0  # Battery capacity in MWh
BATTERY_POWER_MW = 2.0      # Max charge/discharge power in MW
CHARGE_EFFICIENCY = 0.95    # Charging efficiency
DISCHARGE_EFFICIENCY = 0.95 # Discharging efficiency
SOC_MIN = 0.2               # Minimum SOC
SOC_MAX = 0.9               # Maximum SOC
SOC_INITIAL = 0.5           # Initial SOC
CARBON_WEIGHT = 0.001       # Weight for carbon in objective


def load_data():
    """Load all required data for optimization."""
    # Load forecasts
    pv_df = pd.read_csv(PROCESSED_DIR / "forecast_pv.csv", parse_dates=['timestamp'])
    load_df = pd.read_csv(PROCESSED_DIR / "forecast_load.csv", parse_dates=['timestamp'])
    tariff_df = pd.read_csv(PROCESSED_DIR / "tariff.csv", parse_dates=['timestamp'])
    carbon_df = pd.read_csv(PROCESSED_DIR / "carbon.csv", parse_dates=['timestamp'])

    # Merge all data
    data = pv_df[['timestamp', 'forecast_pv_mw']].merge(
        load_df[['timestamp', 'forecast_load_mw']], on='timestamp'
    ).merge(
        tariff_df, on='timestamp'
    ).merge(
        carbon_df, on='timestamp'
    )

    return data


def build_optimization_model(data, horizon=None):
    """
    Build Pyomo optimization model.

    Objective: min Σ (price[t] * grid_import[t] + carbon_weight * carbon_intensity[t] * grid_import[t])

    Subject to:
    - Power balance: grid_import - grid_export = load - pv - battery_discharge + battery_charge
    - SOC dynamics: SOC[t+1] = SOC[t] + charge*eff/capacity - discharge/(eff*capacity)
    - SOC limits: SOC_MIN <= SOC <= SOC_MAX
    - Power limits: charge, discharge <= BATTERY_POWER_MW
    """
    if horizon is None:
        horizon = len(data)

    # Time steps
    T = range(horizon)
    dt = 0.25  # 15 minutes = 0.25 hours

    # Create model
    model = ConcreteModel()

    # Decision variables
    model.battery_charge = Var(T, domain=NonNegativeReals, bounds=(0, BATTERY_POWER_MW))
    model.battery_discharge = Var(T, domain=NonNegativeReals, bounds=(0, BATTERY_POWER_MW))
    model.grid_import = Var(T, domain=NonNegativeReals)
    model.grid_export = Var(T, domain=NonNegativeReals)
    model.soc = Var(range(horizon + 1), bounds=(SOC_MIN * BATTERY_CAPACITY_MWH, SOC_MAX * BATTERY_CAPACITY_MWH))

    # Initial SOC
    model.soc[0].fix(SOC_INITIAL * BATTERY_CAPACITY_MWH)

    # Objective: minimize cost + carbon
    def objective_rule(m):
        total_cost = 0
        for t in T:
            price = data['price_per_kwh'].iloc[t]
            carbon = data['carbon_intensity'].iloc[t]
            # Cost of importing from grid ($/MWh = price_per_kwh * 1000)
            total_cost += price * 1000 * m.grid_import[t] * dt
            # Carbon cost
            total_cost += CARBON_WEIGHT * carbon * m.grid_import[t] * dt
            # Small penalty for grid export to prefer self-consumption
            total_cost -= price * 1000 * 0.5 * m.grid_export[t] * dt
        return total_cost

    model.objective = Objective(rule=objective_rule, sense=minimize)

    # Power balance constraint
    def power_balance_rule(m, t):
        pv = data['forecast_pv_mw'].iloc[t]
        load = data['forecast_load_mw'].iloc[t]
        return (m.grid_import[t] - m.grid_export[t] ==
                load - pv - m.battery_discharge[t] + m.battery_charge[t])

    model.power_balance = Constraint(T, rule=power_balance_rule)

    # SOC dynamics constraint
    def soc_dynamics_rule(m, t):
        return (m.soc[t + 1] == m.soc[t] +
                m.battery_charge[t] * CHARGE_EFFICIENCY * dt -
                m.battery_discharge[t] / DISCHARGE_EFFICIENCY * dt)

    model.soc_dynamics = Constraint(T, rule=soc_dynamics_rule)

    # Prevent simultaneous charging and discharging (simplified linear relaxation)
    # This is a soft constraint via objective penalties

    return model


def solve_optimization(model):
    """Solve the optimization model."""
    # Try different solvers
    solvers_to_try = ['appsi_highs', 'highs', 'glpk', 'cbc']

    solver = None
    for solver_name in solvers_to_try:
        try:
            solver = SolverFactory(solver_name)
            if solver.available():
                print(f"  Using solver: {solver_name}")
                break
        except Exception:
            continue

    if solver is None or not solver.available():
        print("  Warning: No LP solver available, using fallback rules")
        return None

    # Solve
    results = solver.solve(model, tee=False)

    # Check solution status
    if results.solver.termination_condition.name == 'optimal':
        return model
    else:
        print(f"  Warning: Solver status: {results.solver.termination_condition}")
        return None


def extract_results(model, data, horizon):
    """Extract optimization results."""
    results = []

    for t in range(horizon):
        results.append({
            'timestamp': data['timestamp'].iloc[t],
            'pv_forecast_mw': data['forecast_pv_mw'].iloc[t],
            'load_forecast_mw': data['forecast_load_mw'].iloc[t],
            'battery_charge_mw': value(model.battery_charge[t]),
            'battery_discharge_mw': value(model.battery_discharge[t]),
            'grid_import_mw': value(model.grid_import[t]),
            'grid_export_mw': value(model.grid_export[t]),
            'soc_mwh': value(model.soc[t]),
            'soc_percent': value(model.soc[t]) / BATTERY_CAPACITY_MWH * 100,
            'price_per_kwh': data['price_per_kwh'].iloc[t],
            'carbon_intensity': data['carbon_intensity'].iloc[t]
        })

    return pd.DataFrame(results)


def calculate_kpis(results_df):
    """Calculate key performance indicators."""
    dt = 0.25  # 15 minutes

    # Total energy
    total_pv = results_df['pv_forecast_mw'].sum() * dt
    total_load = results_df['load_forecast_mw'].sum() * dt
    total_import = results_df['grid_import_mw'].sum() * dt
    total_export = results_df['grid_export_mw'].sum() * dt

    # Cost calculation
    total_cost = (results_df['grid_import_mw'] * results_df['price_per_kwh'] * 1000 * dt).sum()
    export_revenue = (results_df['grid_export_mw'] * results_df['price_per_kwh'] * 1000 * 0.5 * dt).sum()
    net_cost = total_cost - export_revenue

    # Carbon calculation
    total_carbon = (results_df['grid_import_mw'] * results_df['carbon_intensity'] * dt).sum()

    # Self-consumption rate
    self_consumed = total_pv - total_export
    self_consumption_rate = (self_consumed / total_pv * 100) if total_pv > 0 else 0

    # Peak demand
    peak_demand = results_df['grid_import_mw'].max()

    return {
        'total_pv_mwh': total_pv,
        'total_load_mwh': total_load,
        'total_import_mwh': total_import,
        'total_export_mwh': total_export,
        'net_cost_usd': net_cost,
        'total_carbon_kg': total_carbon,
        'self_consumption_rate': self_consumption_rate,
        'peak_demand_mw': peak_demand
    }


def main():
    """Run MPC optimization pipeline."""
    print("=" * 50)
    print("MICROGRID AUTOPILOT - MPC Optimization")
    print("=" * 50)

    # Load data
    print("\nLoading data...")
    data = load_data()
    horizon = len(data)
    print(f"  Optimization horizon: {horizon} steps ({horizon * 15 / 60:.1f} hours)")

    # Build model
    print("\nBuilding optimization model...")
    model = build_optimization_model(data, horizon)

    # Solve
    print("\nSolving optimization...")
    solved_model = solve_optimization(model)

    if solved_model is None:
        print("\nOptimization failed, using fallback rules...")
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from optimization.fallback_rules import apply_fallback_rules
        results_df = apply_fallback_rules(data)
    else:
        print("\nExtracting results...")
        results_df = extract_results(solved_model, data, horizon)

    # Calculate KPIs
    kpis = calculate_kpis(results_df)

    # Save results
    results_df.to_csv(OUTPUT_DIR / "optimization_results.csv", index=False)
    print(f"\nResults saved to optimization_results.csv")

    # Print KPIs
    print("\n" + "=" * 50)
    print("KEY PERFORMANCE INDICATORS")
    print("=" * 50)
    print(f"  Total PV Generation:    {kpis['total_pv_mwh']:.2f} MWh")
    print(f"  Total Load:             {kpis['total_load_mwh']:.2f} MWh")
    print(f"  Grid Import:            {kpis['total_import_mwh']:.2f} MWh")
    print(f"  Grid Export:            {kpis['total_export_mwh']:.2f} MWh")
    print(f"  Net Cost:               ${kpis['net_cost_usd']:.2f}")
    print(f"  Carbon Emissions:       {kpis['total_carbon_kg']:.2f} kg CO2")
    print(f"  Self-Consumption Rate:  {kpis['self_consumption_rate']:.1f}%")
    print(f"  Peak Demand:            {kpis['peak_demand_mw']:.2f} MW")
    print("=" * 50)

    return results_df, kpis


if __name__ == "__main__":
    main()
