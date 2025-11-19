"""
MPC optimization solver for Microgrid Autopilot.

Optimizes battery scheduling using Pyomo with HiGHS solver.
Includes demand charge modeling and configurable parameters.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from pyomo.environ import (
    ConcreteModel, Var, Objective, Constraint,
    NonNegativeReals, minimize, SolverFactory, value
)
import warnings
warnings.filterwarnings('ignore')

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import get_config


# Load configuration
config = get_config()

# Paths
PROJECT_DIR = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_DIR / config.paths.get('data_processed', 'data_processed')
OUTPUT_DIR = PROCESSED_DIR


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

    # Apply time-varying carbon intensity if enabled
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


def build_optimization_model(data, horizon=None):
    """
    Build Pyomo optimization model with demand charge.

    Objective: min Σ (energy_cost[t] + carbon_cost[t]) + demand_charge_cost

    Subject to:
    - Power balance: grid_import - grid_export = load - pv - battery_discharge + battery_charge
    - SOC dynamics: SOC[t+1] = SOC[t] + charge*eff/capacity - discharge/(eff*capacity)
    - SOC limits: SOC_MIN <= SOC <= SOC_MAX
    - Power limits: charge, discharge <= BATTERY_POWER_MW
    - Peak demand tracking for demand charge
    """
    if horizon is None:
        horizon = len(data)

    # Get battery parameters from config
    battery = config.battery
    BATTERY_CAPACITY_MWH = battery.get('capacity_mwh', 5.0)
    BATTERY_POWER_MW = battery.get('max_power_mw', 2.0)
    CHARGE_EFFICIENCY = battery.get('charge_efficiency', 0.95)
    DISCHARGE_EFFICIENCY = battery.get('discharge_efficiency', 0.95)
    SOC_MIN = battery.get('soc_min', 0.2)
    SOC_MAX = battery.get('soc_max', 0.9)
    SOC_INITIAL = battery.get('soc_initial', 0.5)
    CYCLE_COST = battery.get('cycle_cost_per_kwh', 0.02)  # $/kWh degradation cost

    # Get optimization parameters
    opt_config = config.optimization
    CARBON_WEIGHT = opt_config.get('carbon_weight', 0.001)
    EXPORT_PRICE_RATIO = opt_config.get('export_price_ratio', 0.5)

    # Demand charge parameters
    demand_charge_config = config.demand_charge
    DEMAND_CHARGE_ENABLED = demand_charge_config.get('enabled', True)
    DEMAND_CHARGE_RATE = demand_charge_config.get('rate_per_kw', 15.0)  # $/kW-month
    BILLING_PERIOD_DAYS = demand_charge_config.get('billing_period_days', 30)
    PEAK_WINDOW_START = demand_charge_config.get('peak_window_start', 12)
    PEAK_WINDOW_END = demand_charge_config.get('peak_window_end', 20)

    # Time steps
    T = range(horizon)
    dt = 0.25  # 15 minutes = 0.25 hours

    # Identify peak demand window hours
    data['hour'] = data['timestamp'].dt.hour
    peak_window_mask = (data['hour'] >= PEAK_WINDOW_START) & (data['hour'] < PEAK_WINDOW_END)
    peak_indices = [t for t in T if peak_window_mask.iloc[t]]

    # Create model
    model = ConcreteModel()

    # Decision variables
    model.battery_charge = Var(T, domain=NonNegativeReals, bounds=(0, BATTERY_POWER_MW))
    model.battery_discharge = Var(T, domain=NonNegativeReals, bounds=(0, BATTERY_POWER_MW))
    model.grid_import = Var(T, domain=NonNegativeReals)
    model.grid_export = Var(T, domain=NonNegativeReals)
    model.soc = Var(range(horizon + 1), bounds=(SOC_MIN * BATTERY_CAPACITY_MWH, SOC_MAX * BATTERY_CAPACITY_MWH))

    # Peak demand variable for demand charge
    if DEMAND_CHARGE_ENABLED and peak_indices:
        max_import = max(data['forecast_load_mw'].max(), 10)  # Upper bound estimate
        model.peak_demand = Var(domain=NonNegativeReals, bounds=(0, max_import))

    # Initial SOC
    model.soc[0].fix(SOC_INITIAL * BATTERY_CAPACITY_MWH)

    # Objective: minimize cost + carbon + demand charge
    def objective_rule(m):
        total_cost = 0

        # Energy costs
        for t in T:
            price = data['price_per_kwh'].iloc[t]
            carbon = data['carbon_intensity'].iloc[t]

            # Cost of importing from grid ($/MWh = price_per_kwh * 1000)
            total_cost += price * 1000 * m.grid_import[t] * dt

            # Carbon cost
            total_cost += CARBON_WEIGHT * carbon * m.grid_import[t] * dt

            # Revenue from export (at reduced rate)
            total_cost -= price * 1000 * EXPORT_PRICE_RATIO * m.grid_export[t] * dt

        # Demand charge cost (prorated for optimization horizon)
        if DEMAND_CHARGE_ENABLED and peak_indices:
            # Convert peak MW to kW and prorate for optimization period
            horizon_days = horizon * dt / 24
            proration_factor = horizon_days / BILLING_PERIOD_DAYS
            demand_cost = DEMAND_CHARGE_RATE * 1000 * m.peak_demand * proration_factor
            total_cost += demand_cost

        # Battery degradation cost (based on energy throughput)
        # Cost per MWh = cycle_cost_per_kwh * 1000
        for t in T:
            degradation_cost = CYCLE_COST * 1000 * (m.battery_charge[t] + m.battery_discharge[t]) * dt
            total_cost += degradation_cost

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

    # Peak demand constraints (only for hours in demand charge window)
    if DEMAND_CHARGE_ENABLED and peak_indices:
        def peak_demand_rule(m, t):
            return m.peak_demand >= m.grid_import[t]

        model.peak_demand_constraint = Constraint(peak_indices, rule=peak_demand_rule)

    return model


def solve_optimization(model):
    """Solve the optimization model with timeout and better error handling."""
    # Get solver timeout from config
    timeout = config.optimization.get('solver_timeout_seconds', 300)

    # Try different solvers in order of preference
    solvers_to_try = ['appsi_highs', 'highs', 'glpk', 'cbc']

    solver = None
    solver_name = None

    for name in solvers_to_try:
        try:
            solver = SolverFactory(name)
            if solver.available():
                solver_name = name
                print(f"  Using solver: {name}")
                break
        except Exception:
            continue

    if solver is None or not solver.available():
        print("  Warning: No LP solver available, using fallback rules")
        return None

    # Set solver options
    try:
        if 'highs' in solver_name:
            solver.options['time_limit'] = timeout
        elif solver_name == 'glpk':
            solver.options['tmlim'] = timeout
        elif solver_name == 'cbc':
            solver.options['seconds'] = timeout
    except Exception:
        pass  # Some solvers don't support all options

    # Solve
    try:
        results = solver.solve(model, tee=False)

        # Check solution status
        if results.solver.termination_condition.name == 'optimal':
            return model
        elif results.solver.termination_condition.name == 'maxTimeLimit':
            print(f"  Warning: Solver hit time limit ({timeout}s)")
            return model  # Return potentially suboptimal solution
        else:
            print(f"  Warning: Solver status: {results.solver.termination_condition}")
            return None

    except Exception as e:
        print(f"  Error solving optimization: {e}")
        return None


def extract_results(model, data, horizon):
    """Extract optimization results including demand charge info."""
    # Get config values for reference
    battery = config.battery
    BATTERY_CAPACITY_MWH = battery.get('capacity_mwh', 5.0)

    demand_charge_config = config.demand_charge
    DEMAND_CHARGE_ENABLED = demand_charge_config.get('enabled', True)

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

    results_df = pd.DataFrame(results)

    # Add peak demand info
    if DEMAND_CHARGE_ENABLED and hasattr(model, 'peak_demand'):
        results_df['peak_demand_mw'] = value(model.peak_demand)
    else:
        results_df['peak_demand_mw'] = results_df['grid_import_mw'].max()

    return results_df


def calculate_kpis(results_df):
    """Calculate key performance indicators including demand charge."""
    dt = 0.25  # 15 minutes

    # Get config values
    demand_charge_config = config.demand_charge
    DEMAND_CHARGE_ENABLED = demand_charge_config.get('enabled', True)
    DEMAND_CHARGE_RATE = demand_charge_config.get('rate_per_kw', 15.0)
    BILLING_PERIOD_DAYS = demand_charge_config.get('billing_period_days', 30)
    EXPORT_PRICE_RATIO = config.optimization.get('export_price_ratio', 0.5)
    CYCLE_COST = config.battery.get('cycle_cost_per_kwh', 0.02)

    # Total energy
    total_pv = results_df['pv_forecast_mw'].sum() * dt
    total_load = results_df['load_forecast_mw'].sum() * dt
    total_import = results_df['grid_import_mw'].sum() * dt
    total_export = results_df['grid_export_mw'].sum() * dt

    # Battery throughput
    total_charge = results_df['battery_charge_mw'].sum() * dt
    total_discharge = results_df['battery_discharge_mw'].sum() * dt
    battery_throughput = total_charge + total_discharge

    # Energy cost calculation
    energy_cost = (results_df['grid_import_mw'] * results_df['price_per_kwh'] * 1000 * dt).sum()
    export_revenue = (results_df['grid_export_mw'] * results_df['price_per_kwh'] * 1000 * EXPORT_PRICE_RATIO * dt).sum()

    # Demand charge calculation
    peak_demand_mw = results_df['peak_demand_mw'].iloc[0] if 'peak_demand_mw' in results_df.columns else results_df['grid_import_mw'].max()
    horizon_hours = len(results_df) * dt
    horizon_days = horizon_hours / 24
    proration_factor = horizon_days / BILLING_PERIOD_DAYS

    if DEMAND_CHARGE_ENABLED:
        demand_charge = DEMAND_CHARGE_RATE * peak_demand_mw * 1000 * proration_factor  # kW * $/kW
    else:
        demand_charge = 0

    # Battery degradation cost
    degradation_cost = CYCLE_COST * battery_throughput * 1000  # $/kWh * MWh * 1000

    net_cost = energy_cost - export_revenue + demand_charge + degradation_cost

    # Carbon calculation
    total_carbon = (results_df['grid_import_mw'] * results_df['carbon_intensity'] * dt).sum()

    # Self-consumption rate
    self_consumed = total_pv - total_export
    self_consumption_rate = (self_consumed / total_pv * 100) if total_pv > 0 else 0

    # Self-sufficiency (how much load is met by local generation)
    self_sufficiency = ((total_load - total_import) / total_load * 100) if total_load > 0 else 0

    return {
        'total_pv_mwh': total_pv,
        'total_load_mwh': total_load,
        'total_import_mwh': total_import,
        'total_export_mwh': total_export,
        'battery_throughput_mwh': battery_throughput,
        'energy_cost_usd': energy_cost,
        'export_revenue_usd': export_revenue,
        'demand_charge_usd': demand_charge,
        'degradation_cost_usd': degradation_cost,
        'net_cost_usd': net_cost,
        'total_carbon_kg': total_carbon,
        'self_consumption_rate': self_consumption_rate,
        'self_sufficiency_rate': self_sufficiency,
        'peak_demand_mw': peak_demand_mw
    }


def main():
    """Run MPC optimization pipeline."""
    print("=" * 50)
    print("MICROGRID AUTOPILOT - MPC Optimization")
    print("=" * 50)

    # Print configuration
    battery = config.battery
    demand_charge = config.demand_charge
    print(f"\nConfiguration:")
    print(f"  Battery: {battery.get('capacity_mwh')} MWh, {battery.get('max_power_mw')} MW")
    print(f"  SOC Limits: {battery.get('soc_min')*100:.0f}% - {battery.get('soc_max')*100:.0f}%")
    print(f"  Demand Charge: {'Enabled' if demand_charge.get('enabled') else 'Disabled'}")
    if demand_charge.get('enabled'):
        print(f"    Rate: ${demand_charge.get('rate_per_kw')}/kW-month")
        print(f"    Peak Window: {demand_charge.get('peak_window_start')}:00 - {demand_charge.get('peak_window_end')}:00")

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
    print(f"  Battery Throughput:     {kpis['battery_throughput_mwh']:.2f} MWh")
    print(f"\n  Energy Cost:            ${kpis['energy_cost_usd']:.2f}")
    print(f"  Export Revenue:         ${kpis['export_revenue_usd']:.2f}")
    print(f"  Demand Charge:          ${kpis['demand_charge_usd']:.2f}")
    print(f"  Degradation Cost:       ${kpis['degradation_cost_usd']:.2f}")
    print(f"  Net Cost:               ${kpis['net_cost_usd']:.2f}")
    print(f"\n  Carbon Emissions:       {kpis['total_carbon_kg']:.2f} kg CO2")
    print(f"  Self-Consumption Rate:  {kpis['self_consumption_rate']:.1f}%")
    print(f"  Self-Sufficiency Rate:  {kpis['self_sufficiency_rate']:.1f}%")
    print(f"  Peak Demand:            {kpis['peak_demand_mw']:.2f} MW")
    print("=" * 50)

    return results_df, kpis


if __name__ == "__main__":
    main()
