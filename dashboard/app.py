"""
Streamlit dashboard for Microgrid Autopilot.

Visualizes forecasts, battery SOC, grid flows, and KPIs.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Paths
PROCESSED_DIR = Path(__file__).parent.parent / "data_processed"


def load_data():
    """Load all processed data for visualization."""
    data = {}

    try:
        # Load forecast data
        if (PROCESSED_DIR / "forecast_pv.csv").exists():
            data['pv_forecast'] = pd.read_csv(
                PROCESSED_DIR / "forecast_pv.csv",
                parse_dates=['timestamp']
            )

        if (PROCESSED_DIR / "forecast_load.csv").exists():
            data['load_forecast'] = pd.read_csv(
                PROCESSED_DIR / "forecast_load.csv",
                parse_dates=['timestamp']
            )

        # Load optimization results
        if (PROCESSED_DIR / "optimization_results.csv").exists():
            data['optimization'] = pd.read_csv(
                PROCESSED_DIR / "optimization_results.csv",
                parse_dates=['timestamp']
            )

        # Load power flow results
        if (PROCESSED_DIR / "power_flow.csv").exists():
            data['power_flow'] = pd.read_csv(
                PROCESSED_DIR / "power_flow.csv",
                parse_dates=['timestamp']
            )

        # Load tariff data
        if (PROCESSED_DIR / "tariff.csv").exists():
            data['tariff'] = pd.read_csv(
                PROCESSED_DIR / "tariff.csv",
                parse_dates=['timestamp']
            )

    except Exception as e:
        st.error(f"Error loading data: {e}")

    return data


def plot_pv_forecast(data):
    """Plot PV forecast vs actual."""
    if 'pv_forecast' not in data:
        st.warning("PV forecast data not available")
        return

    df = data['pv_forecast']

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['actual_pv_mw'],
        name='Actual', mode='lines',
        line=dict(color='orange', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['forecast_pv_mw'],
        name='Forecast', mode='lines',
        line=dict(color='blue', width=2, dash='dash')
    ))

    fig.update_layout(
        title='PV Generation: Forecast vs Actual',
        xaxis_title='Time',
        yaxis_title='Power (MW)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_load_forecast(data):
    """Plot load forecast vs actual."""
    if 'load_forecast' not in data:
        st.warning("Load forecast data not available")
        return

    df = data['load_forecast']

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['actual_load_mw'],
        name='Actual', mode='lines',
        line=dict(color='red', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['forecast_load_mw'],
        name='Forecast', mode='lines',
        line=dict(color='purple', width=2, dash='dash')
    ))

    fig.update_layout(
        title='Load Demand: Forecast vs Actual',
        xaxis_title='Time',
        yaxis_title='Power (MW)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_battery_soc(data):
    """Plot battery SOC over time."""
    if 'optimization' not in data:
        st.warning("Optimization data not available")
        return

    df = data['optimization']

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['soc_percent'],
        name='SOC', mode='lines',
        fill='tozeroy',
        line=dict(color='green', width=2)
    ))

    # Add SOC limits
    fig.add_hline(y=90, line_dash="dash", line_color="red",
                  annotation_text="Max SOC (90%)")
    fig.add_hline(y=20, line_dash="dash", line_color="red",
                  annotation_text="Min SOC (20%)")

    fig.update_layout(
        title='Battery State of Charge',
        xaxis_title='Time',
        yaxis_title='SOC (%)',
        yaxis=dict(range=[0, 100]),
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_grid_flows(data):
    """Plot grid import/export."""
    if 'power_flow' not in data:
        st.warning("Power flow data not available")
        return

    df = data['power_flow']

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['grid_import_mw'],
        name='Grid Import', mode='lines',
        fill='tozeroy',
        line=dict(color='red', width=1)
    ))
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=-df['grid_export_mw'],
        name='Grid Export', mode='lines',
        fill='tozeroy',
        line=dict(color='blue', width=1)
    ))

    fig.update_layout(
        title='Grid Import/Export',
        xaxis_title='Time',
        yaxis_title='Power (MW)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_tariff_overlay(data):
    """Plot tariff prices over time."""
    if 'tariff' not in data:
        st.warning("Tariff data not available")
        return

    df = data['tariff']

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['price_per_kwh'],
        name='Price', mode='lines+markers',
        line=dict(color='green', width=2),
        marker=dict(size=4)
    ))

    fig.update_layout(
        title='Electricity Tariff (TOU)',
        xaxis_title='Time',
        yaxis_title='Price ($/kWh)',
        height=300
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_power_balance(data):
    """Plot power balance stacked area chart."""
    if 'power_flow' not in data:
        st.warning("Power flow data not available")
        return

    df = data['power_flow']

    fig = go.Figure()

    # Add traces for each power component
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['pv_mw'],
        name='PV Generation', mode='lines',
        stackgroup='positive',
        line=dict(color='orange')
    ))
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['battery_discharge_mw'],
        name='Battery Discharge', mode='lines',
        stackgroup='positive',
        line=dict(color='green')
    ))
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['grid_import_mw'],
        name='Grid Import', mode='lines',
        stackgroup='positive',
        line=dict(color='gray')
    ))

    # Load as negative
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=-df['load_mw'],
        name='Load', mode='lines',
        line=dict(color='red', width=3)
    ))

    fig.update_layout(
        title='Power Balance',
        xaxis_title='Time',
        yaxis_title='Power (MW)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


def calculate_kpis(data):
    """Calculate and display KPIs."""
    if 'power_flow' not in data:
        return

    df = data['power_flow']
    dt = 0.25  # 15 minutes

    # Calculate metrics
    total_pv = df['pv_mw'].sum() * dt
    total_load = df['load_mw'].sum() * dt
    total_import = df['grid_import_mw'].sum() * dt
    total_export = df['grid_export_mw'].sum() * dt
    net_cost = df['net_cost_usd'].sum()
    total_carbon = df['carbon_emissions_kg'].sum()

    self_consumption = total_pv - total_export
    self_consumption_rate = (self_consumption / total_pv * 100) if total_pv > 0 else 0
    self_sufficiency = ((total_load - total_import) / total_load * 100) if total_load > 0 else 0

    # Display KPIs in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total PV Generation", f"{total_pv:.1f} MWh")
        st.metric("Total Load", f"{total_load:.1f} MWh")

    with col2:
        st.metric("Grid Import", f"{total_import:.1f} MWh")
        st.metric("Grid Export", f"{total_export:.1f} MWh")

    with col3:
        st.metric("Net Cost", f"${net_cost:.2f}")
        st.metric("Carbon Emissions", f"{total_carbon:.1f} kg")

    with col4:
        st.metric("Self-Consumption", f"{self_consumption_rate:.1f}%")
        st.metric("Self-Sufficiency", f"{self_sufficiency:.1f}%")


def main():
    """Main dashboard application."""
    st.set_page_config(
        page_title="Microgrid Autopilot",
        page_icon="⚡",
        layout="wide"
    )

    st.title("⚡ Microgrid Autopilot Dashboard")
    st.markdown("---")

    # Load data
    data = load_data()

    if not data:
        st.warning("No data available. Please run the pipeline first.")
        st.code("python data_prep/process_data.py\n"
                "python forecasting/pv_forecast.py\n"
                "python forecasting/load_forecast.py\n"
                "python optimization/mpc_solver.py\n"
                "python simulation/power_flow.py")
        return

    # KPIs Section
    st.header("Key Performance Indicators")
    calculate_kpis(data)
    st.markdown("---")

    # Forecasts Section
    st.header("Forecasts")
    col1, col2 = st.columns(2)
    with col1:
        plot_pv_forecast(data)
    with col2:
        plot_load_forecast(data)

    st.markdown("---")

    # Battery and Grid Section
    st.header("Battery & Grid Operations")
    col1, col2 = st.columns(2)
    with col1:
        plot_battery_soc(data)
    with col2:
        plot_grid_flows(data)

    st.markdown("---")

    # Power Balance Section
    st.header("Power Balance")
    plot_power_balance(data)

    # Tariff Section
    st.header("Tariff Schedule")
    plot_tariff_overlay(data)

    # Footer
    st.markdown("---")
    st.caption("Microgrid Autopilot v1.0 - Intelligent Energy Management System")


if __name__ == "__main__":
    main()
