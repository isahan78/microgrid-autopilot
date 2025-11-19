"""
Streamlit dashboard for Microgrid Autopilot.

Visualizes forecasts, battery SOC, grid flows, and KPIs.
Includes demand charge analysis and cost breakdown.
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

from utils.config import get_config

# Load configuration
config = get_config()

# Paths
PROJECT_DIR = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_DIR / config.paths.get('data_processed', 'data_processed')


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

        # Load carbon data
        if (PROCESSED_DIR / "carbon.csv").exists():
            data['carbon'] = pd.read_csv(
                PROCESSED_DIR / "carbon.csv",
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

    # Get SOC limits from config
    soc_min = config.battery.get('soc_min', 0.2) * 100
    soc_max = config.battery.get('soc_max', 0.9) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['soc_percent'],
        name='SOC', mode='lines',
        fill='tozeroy',
        line=dict(color='green', width=2)
    ))

    # Add SOC limits
    fig.add_hline(y=soc_max, line_dash="dash", line_color="red",
                  annotation_text=f"Max SOC ({soc_max:.0f}%)")
    fig.add_hline(y=soc_min, line_dash="dash", line_color="red",
                  annotation_text=f"Min SOC ({soc_min:.0f}%)")

    fig.update_layout(
        title='Battery State of Charge',
        xaxis_title='Time',
        yaxis_title='SOC (%)',
        yaxis=dict(range=[0, 100]),
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_grid_flows(data):
    """Plot grid import/export with peak demand highlight."""
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

    # Add peak demand line
    peak_demand = df['grid_import_mw'].max()
    fig.add_hline(y=peak_demand, line_dash="dot", line_color="darkred",
                  annotation_text=f"Peak: {peak_demand:.2f} MW")

    fig.update_layout(
        title='Grid Import/Export',
        xaxis_title='Time',
        yaxis_title='Power (MW)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_tariff_and_carbon(data):
    """Plot tariff prices and carbon intensity over time."""
    if 'tariff' not in data:
        st.warning("Tariff data not available")
        return

    df = data['tariff']

    # Create subplot with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Tariff prices
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'], y=df['price_per_kwh'],
            name='Electricity Price', mode='lines',
            line=dict(color='green', width=2)
        ),
        secondary_y=False
    )

    # Carbon intensity if available
    if 'optimization' in data:
        opt_df = data['optimization']
        if 'carbon_intensity' in opt_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=opt_df['timestamp'], y=opt_df['carbon_intensity'],
                    name='Carbon Intensity', mode='lines',
                    line=dict(color='gray', width=2, dash='dash')
                ),
                secondary_y=True
            )

    fig.update_layout(
        title='Electricity Tariff & Carbon Intensity',
        height=350,
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    fig.update_xaxes(title_text='Time')
    fig.update_yaxes(title_text='Price ($/kWh)', secondary_y=False)
    fig.update_yaxes(title_text='Carbon (g/kWh)', secondary_y=True)

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


def plot_cost_breakdown(energy_cost, export_revenue, demand_charge):
    """Plot cost breakdown pie chart."""
    labels = ['Energy Cost', 'Demand Charge']
    values = [energy_cost - export_revenue, demand_charge]
    colors = ['#636EFA', '#EF553B']

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker_colors=colors,
        textinfo='label+percent',
        textposition='outside'
    )])

    fig.update_layout(
        title='Cost Breakdown',
        height=300,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)


def calculate_kpis(data):
    """Calculate and display KPIs with demand charge breakdown."""
    if 'power_flow' not in data:
        return

    df = data['power_flow']
    dt = 0.25  # 15 minutes

    # Get config values
    demand_charge_config = config.demand_charge
    DEMAND_CHARGE_ENABLED = demand_charge_config.get('enabled', True)
    DEMAND_CHARGE_RATE = demand_charge_config.get('rate_per_kw', 15.0)
    BILLING_PERIOD_DAYS = demand_charge_config.get('billing_period_days', 30)
    EXPORT_PRICE_RATIO = config.optimization.get('export_price_ratio', 0.5)

    # Calculate metrics
    total_pv = df['pv_mw'].sum() * dt
    total_load = df['load_mw'].sum() * dt
    total_import = df['grid_import_mw'].sum() * dt
    total_export = df['grid_export_mw'].sum() * dt
    total_carbon = df['carbon_emissions_kg'].sum()

    # Cost calculations
    if 'price_per_kwh' in df.columns:
        energy_cost = (df['grid_import_mw'] * df['price_per_kwh'] * 1000 * dt).sum()
        export_revenue = (df['grid_export_mw'] * df['price_per_kwh'] * 1000 * EXPORT_PRICE_RATIO * dt).sum()
    else:
        energy_cost = df['net_cost_usd'].sum()
        export_revenue = 0

    # Demand charge calculation
    peak_demand = df['grid_import_mw'].max()
    horizon_hours = len(df) * dt
    horizon_days = horizon_hours / 24
    proration_factor = horizon_days / BILLING_PERIOD_DAYS

    if DEMAND_CHARGE_ENABLED:
        demand_charge = DEMAND_CHARGE_RATE * peak_demand * 1000 * proration_factor
    else:
        demand_charge = 0

    net_cost = energy_cost - export_revenue + demand_charge

    # Self-consumption and sufficiency
    self_consumption = total_pv - total_export
    self_consumption_rate = (self_consumption / total_pv * 100) if total_pv > 0 else 0
    self_sufficiency = ((total_load - total_import) / total_load * 100) if total_load > 0 else 0

    # Display KPIs in columns
    st.subheader("Energy Metrics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total PV Generation", f"{total_pv:.1f} MWh")
        st.metric("Total Load", f"{total_load:.1f} MWh")

    with col2:
        st.metric("Grid Import", f"{total_import:.1f} MWh")
        st.metric("Grid Export", f"{total_export:.1f} MWh")

    with col3:
        st.metric("Self-Consumption", f"{self_consumption_rate:.1f}%")
        st.metric("Self-Sufficiency", f"{self_sufficiency:.1f}%")

    with col4:
        st.metric("Carbon Emissions", f"{total_carbon:.0f} kg")
        st.metric("Peak Demand", f"{peak_demand:.2f} MW")

    st.markdown("---")

    # Cost breakdown section
    st.subheader("Cost Analysis")
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.metric("Energy Cost", f"${energy_cost:.2f}")
        st.metric("Export Revenue", f"${export_revenue:.2f}", delta=f"-${export_revenue:.2f}")

    with col2:
        if DEMAND_CHARGE_ENABLED:
            st.metric(
                "Demand Charge",
                f"${demand_charge:.2f}",
                help=f"${DEMAND_CHARGE_RATE}/kW-month × {peak_demand*1000:.0f} kW × {proration_factor:.2f}"
            )
        else:
            st.metric("Demand Charge", "Disabled")
        st.metric("Net Cost", f"${net_cost:.2f}")

    with col3:
        plot_cost_breakdown(energy_cost, export_revenue, demand_charge)

    return {
        'total_pv': total_pv,
        'total_load': total_load,
        'total_import': total_import,
        'total_export': total_export,
        'energy_cost': energy_cost,
        'export_revenue': export_revenue,
        'demand_charge': demand_charge,
        'net_cost': net_cost,
        'peak_demand': peak_demand,
        'carbon': total_carbon,
        'self_consumption_rate': self_consumption_rate,
        'self_sufficiency': self_sufficiency
    }


def show_configuration():
    """Display current configuration in sidebar."""
    st.sidebar.header("Configuration")

    battery = config.battery
    st.sidebar.subheader("Battery")
    st.sidebar.text(f"Capacity: {battery.get('capacity_mwh')} MWh")
    st.sidebar.text(f"Max Power: {battery.get('max_power_mw')} MW")
    st.sidebar.text(f"SOC Range: {battery.get('soc_min')*100:.0f}% - {battery.get('soc_max')*100:.0f}%")

    demand = config.demand_charge
    st.sidebar.subheader("Demand Charge")
    st.sidebar.text(f"Enabled: {demand.get('enabled')}")
    st.sidebar.text(f"Rate: ${demand.get('rate_per_kw')}/kW-month")
    st.sidebar.text(f"Peak Window: {demand.get('peak_window_start')}:00 - {demand.get('peak_window_end')}:00")


def main():
    """Main dashboard application."""
    st.set_page_config(
        page_title="Microgrid Autopilot",
        page_icon="",
        layout="wide"
    )

    st.title("Microgrid Autopilot Dashboard")
    st.markdown("---")

    # Show configuration in sidebar
    show_configuration()

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
    kpis = calculate_kpis(data)
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

    # Tariff and Carbon Section
    st.header("Tariff & Carbon Intensity")
    plot_tariff_and_carbon(data)

    # Footer
    st.markdown("---")
    st.caption("Microgrid Autopilot v2.0 - Intelligent Energy Management System with Demand Charge Optimization")


if __name__ == "__main__":
    main()
