"""
Tests for the MPC optimization solver.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from optimization.mpc_solver import (
    build_optimization_model, solve_optimization, extract_results,
    calculate_kpis, BATTERY_CAPACITY_MWH, SOC_MIN, SOC_MAX
)
from optimization.fallback_rules import apply_fallback_rules
from simulation.battery_sim import BatterySimulator
from simulation.power_flow import simulate_power_flow, calculate_summary_metrics


PROCESSED_DIR = Path(__file__).parent.parent / "data_processed"


class TestMPCSolver:
    """Tests for MPC optimization solver."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        n = 96  # 24 hours at 15-min intervals
        timestamps = pd.date_range('2006-07-15', periods=n, freq='15T')

        # Create realistic daily patterns
        hours = np.array([t.hour + t.minute/60 for t in timestamps])

        # PV: bell curve during day
        pv = 5 * np.exp(-((hours - 12) ** 2) / 20)
        pv = np.clip(pv, 0, None)

        # Load: higher in morning and evening
        load = 3 + 1.5 * np.sin((hours - 6) * np.pi / 12) + np.random.uniform(-0.2, 0.2, n)
        load = np.clip(load, 1, 6)

        # Tariff: TOU pricing
        price = np.where((hours >= 17) & (hours < 22), 0.30,
                        np.where((hours >= 22) | (hours < 17), 0.12, 0.15))

        data = pd.DataFrame({
            'timestamp': timestamps,
            'forecast_pv_mw': pv,
            'forecast_load_mw': load,
            'price_per_kwh': price,
            'carbon_intensity': np.ones(n) * 400
        })

        return data

    def test_build_optimization_model(self, sample_data):
        """Test model building."""
        model = build_optimization_model(sample_data)

        # Check model components exist
        assert hasattr(model, 'battery_charge')
        assert hasattr(model, 'battery_discharge')
        assert hasattr(model, 'grid_import')
        assert hasattr(model, 'grid_export')
        assert hasattr(model, 'soc')
        assert hasattr(model, 'objective')
        assert hasattr(model, 'power_balance')
        assert hasattr(model, 'soc_dynamics')

    def test_soc_constraints(self, sample_data):
        """Test SOC constraints are properly set."""
        model = build_optimization_model(sample_data)

        # Check SOC bounds
        for t in range(len(sample_data) + 1):
            bounds = model.soc[t].bounds
            assert bounds[0] == SOC_MIN * BATTERY_CAPACITY_MWH
            assert bounds[1] == SOC_MAX * BATTERY_CAPACITY_MWH

    def test_calculate_kpis(self, sample_data):
        """Test KPI calculation."""
        # Apply fallback rules to get results
        results = apply_fallback_rules(sample_data)

        kpis = calculate_kpis(results)

        # Check KPIs exist and are reasonable
        assert 'total_pv_mwh' in kpis
        assert 'total_load_mwh' in kpis
        assert 'net_cost_usd' in kpis
        assert 'total_carbon_kg' in kpis

        assert kpis['total_pv_mwh'] >= 0
        assert kpis['total_load_mwh'] >= 0
        assert kpis['total_carbon_kg'] >= 0


class TestFallbackRules:
    """Tests for fallback rule-based control."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        n = 96
        timestamps = pd.date_range('2006-07-15', periods=n, freq='15T')
        hours = np.array([t.hour + t.minute/60 for t in timestamps])

        data = pd.DataFrame({
            'timestamp': timestamps,
            'forecast_pv_mw': 5 * np.exp(-((hours - 12) ** 2) / 20),
            'forecast_load_mw': 3 + np.sin((hours - 6) * np.pi / 12),
            'price_per_kwh': np.where((hours >= 17) & (hours < 22), 0.30, 0.12),
            'carbon_intensity': np.ones(n) * 400
        })

        return data

    def test_apply_fallback_rules(self, sample_data):
        """Test fallback rules application."""
        results = apply_fallback_rules(sample_data)

        # Check output columns
        assert 'battery_charge_mw' in results.columns
        assert 'battery_discharge_mw' in results.columns
        assert 'grid_import_mw' in results.columns
        assert 'grid_export_mw' in results.columns
        assert 'soc_mwh' in results.columns

        # Check constraints are satisfied
        assert results['battery_charge_mw'].min() >= 0
        assert results['battery_discharge_mw'].min() >= 0
        assert results['grid_import_mw'].min() >= 0
        assert results['grid_export_mw'].min() >= 0

        # Check SOC bounds
        soc_percent = results['soc_percent']
        assert soc_percent.min() >= SOC_MIN * 100 - 0.1  # Small tolerance
        assert soc_percent.max() <= SOC_MAX * 100 + 0.1

    def test_power_balance(self, sample_data):
        """Test power balance is maintained."""
        results = apply_fallback_rules(sample_data)

        for idx, row in results.iterrows():
            # Power balance: pv + discharge + import = load + charge + export
            supply = (row['pv_forecast_mw'] + row['battery_discharge_mw'] +
                     row['grid_import_mw'])
            demand = (row['load_forecast_mw'] + row['battery_charge_mw'] +
                     row['grid_export_mw'])

            assert abs(supply - demand) < 0.01, f"Power balance violated at {idx}"


class TestBatterySimulator:
    """Tests for battery simulator."""

    def test_simulator_initialization(self):
        """Test simulator initialization."""
        sim = BatterySimulator()

        assert sim.capacity_mwh == BATTERY_CAPACITY_MWH
        assert sim.soc_min == SOC_MIN
        assert sim.soc_max == SOC_MAX

    def test_simulator_soc_limits(self):
        """Test SOC limits are enforced."""
        sim = BatterySimulator()

        # Create schedule that would exceed limits
        n = 10
        schedule = pd.DataFrame({
            'timestamp': pd.date_range('2006-07-15', periods=n, freq='15T'),
            'battery_charge_mw': [2] * n,  # Constant charging
            'battery_discharge_mw': [0] * n
        })

        results = sim.simulate(schedule, initial_soc=0.8)

        # SOC should not exceed max
        assert results['soc_percent'].max() <= SOC_MAX * 100 + 0.1


class TestPowerFlow:
    """Tests for power flow simulation."""

    @pytest.fixture
    def sample_optimization_results(self):
        """Create sample optimization results."""
        n = 96
        timestamps = pd.date_range('2006-07-15', periods=n, freq='15T')

        return pd.DataFrame({
            'timestamp': timestamps,
            'pv_forecast_mw': np.random.uniform(0, 8, n),
            'load_forecast_mw': np.random.uniform(2, 5, n),
            'battery_charge_mw': np.random.uniform(0, 1, n),
            'battery_discharge_mw': np.random.uniform(0, 1, n),
            'price_per_kwh': np.random.choice([0.12, 0.30], n),
            'carbon_intensity': np.ones(n) * 400,
            'soc_percent': np.random.uniform(20, 90, n)
        })

    def test_simulate_power_flow(self, sample_optimization_results):
        """Test power flow simulation."""
        results = simulate_power_flow(sample_optimization_results)

        # Check output columns
        assert 'grid_import_mw' in results.columns
        assert 'grid_export_mw' in results.columns
        assert 'net_cost_usd' in results.columns
        assert 'carbon_emissions_kg' in results.columns

        # Check values are non-negative
        assert results['grid_import_mw'].min() >= 0
        assert results['grid_export_mw'].min() >= 0
        assert results['carbon_emissions_kg'].min() >= 0

    def test_calculate_summary_metrics(self, sample_optimization_results):
        """Test summary metrics calculation."""
        power_flow = simulate_power_flow(sample_optimization_results)
        metrics = calculate_summary_metrics(power_flow)

        # Check metrics exist
        assert 'total_pv_mwh' in metrics
        assert 'total_load_mwh' in metrics
        assert 'net_cost_usd' in metrics
        assert 'self_consumption_rate_pct' in metrics

        # Check values are reasonable
        assert metrics['total_pv_mwh'] >= 0
        assert metrics['total_load_mwh'] >= 0
        assert 0 <= metrics['self_consumption_rate_pct'] <= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
