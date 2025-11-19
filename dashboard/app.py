"""
Streamlit dashboard for Microgrid Autopilot.

Visualizes forecasts, battery SOC, grid flows, and KPIs.
Supports both historical data and live API forecasting.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime, timedelta
import sys
import joblib

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import get_config
from utils.weather_api import fetch_weather_forecast, resample_to_15min
from utils.pv_model import calculate_pv_output

# Load configuration
config = get_config()

# Paths
PROJECT_DIR = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_DIR / config.paths.get('data_processed', 'data_processed')
MODELS_DIR = PROJECT_DIR / config.paths.get('models', 'models')


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


def generate_pv_forecast_hybrid(weather_df: pd.DataFrame) -> np.ndarray:
    """
    Generate PV forecast using hybrid approach (ML if available, else physics).
    """
    pv_config = config.get('pv_system', default={}) or {}
    model_type = pv_config.get('model_type', 'hybrid')

    # Try ML model first if hybrid or ml mode
    if model_type in ['ml', 'hybrid']:
        ml_model_path = MODELS_DIR / "pv_model.joblib"
        if ml_model_path.exists():
            try:
                model = joblib.load(ml_model_path)
                # ML model needs specific features - for now use physics
                # Full ML integration would require feature engineering
                st.sidebar.info("Using physics-based PV model")
            except Exception:
                pass

    # Use physics-based model
    pv_output = calculate_pv_output(
        ghi=weather_df['ghi'].values,
        dni=weather_df['dni'].values,
        temperature=weather_df['temperature'].values,
        timestamps=weather_df['timestamp']
    )

    return pv_output


def generate_load_forecast_hybrid(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """
    Generate load forecast using hybrid approach (ML if available, else profile).
    """
    load_config = config.get('load_profile', default={}) or {}
    model_type = load_config.get('model_type', 'hybrid')

    # Try ML model first if hybrid or ml mode
    if model_type in ['ml', 'hybrid']:
        ml_model_path = MODELS_DIR / "load_model.joblib"
        if ml_model_path.exists():
            try:
                model = joblib.load(ml_model_path)
                # ML model needs specific features - for now use profile
                st.sidebar.info("Using profile-based load model")
            except Exception:
                pass

    # Use profile-based model from config
    base_load = load_config.get('base_load_mw', 3.0)
    hourly_pattern = load_config.get('hourly_pattern', {})
    weekly_pattern = load_config.get('weekly_pattern', {})
    noise_std = load_config.get('noise_std', 0.05)

    load_values = []
    for ts in timestamps:
        hour = ts.hour
        day_of_week = ts.dayofweek

        # Get hourly multiplier
        hour_mult = hourly_pattern.get(hour, hourly_pattern.get(str(hour), 1.0))

        # Get weekly multiplier
        week_mult = weekly_pattern.get(day_of_week, weekly_pattern.get(str(day_of_week), 1.0))

        load = base_load * hour_mult * week_mult
        load_values.append(load)

    load_array = np.array(load_values)

    # Add random variation
    if noise_std > 0:
        noise = np.random.normal(0, noise_std * base_load, len(load_array))
        load_array = load_array + noise

    # Ensure positive values
    load_array = np.clip(load_array, 0.5, None)

    return load_array


def run_live_forecast(horizon_hours=48):
    """
    Run live forecasting with real-time weather data.

    Uses hybrid models: ML if trained, otherwise physics/profile-based.
    Returns optimization results based on current weather conditions.
    """
    from optimization.mpc_solver import (
        build_optimization_model, solve_optimization,
        extract_results, calculate_kpis
    )

    # Fetch live weather
    weather_df = fetch_weather_forecast(
        forecast_days=min(horizon_hours // 24 + 1, 16)
    )

    if weather_df is None:
        st.error("Failed to fetch weather data")
        return None

    weather_df = resample_to_15min(weather_df)

    # Generate forecasts
    forecast_df = weather_df.copy()

    # PV forecast (hybrid: ML or physics)
    forecast_df['forecast_pv_mw'] = generate_pv_forecast_hybrid(weather_df)

    # Load forecast (hybrid: ML or profile)
    forecast_df['forecast_load_mw'] = generate_load_forecast_hybrid(weather_df['timestamp'])

    # Generate tariff data (TOU pricing from config)
    tariff_config = config.get('tariff', default={}) or {}

    def get_price(hour):
        peak = tariff_config.get('peak', {})
        off_peak = tariff_config.get('off_peak', {})
        mid_peak = tariff_config.get('mid_peak', {})

        peak_hours = peak.get('hours', [17, 22])
        off_peak_hours = off_peak.get('hours', [0, 17])

        if peak_hours[0] <= hour < peak_hours[1]:
            return peak.get('price_per_kwh', 0.30)
        elif off_peak_hours[0] <= hour < off_peak_hours[1]:
            return off_peak.get('price_per_kwh', 0.12)
        else:
            return mid_peak.get('price_per_kwh', 0.15)

    forecast_df['price_per_kwh'] = forecast_df['timestamp'].dt.hour.apply(get_price)

    # Carbon intensity with time-varying multipliers
    carbon_config = config.get('carbon', default={}) or {}
    base_intensity = carbon_config.get('default_intensity', 410)
    hourly_multipliers = carbon_config.get('hourly_multipliers', {})

    def get_carbon(hour):
        multiplier = hourly_multipliers.get(hour, hourly_multipliers.get(str(hour), 1.0))
        return base_intensity * multiplier

    forecast_df['carbon_intensity'] = forecast_df['timestamp'].dt.hour.apply(get_carbon)

    # Prepare data for optimization
    data = forecast_df[['timestamp', 'forecast_pv_mw', 'forecast_load_mw',
                        'price_per_kwh', 'carbon_intensity']].copy()

    horizon = min(len(data), horizon_hours * 4)  # 4 intervals per hour

    # Build and solve optimization
    model = build_optimization_model(data, horizon)
    solved_model = solve_optimization(model)

    if solved_model is None:
        st.error("Optimization failed")
        return None

    # Extract results
    results_df = extract_results(solved_model, data, horizon)
    kpis = calculate_kpis(results_df)

    # Track which models were used
    pv_model_type = config.get('pv_system', default={}).get('model_type', 'hybrid')
    load_model_type = config.get('load_profile', default={}).get('model_type', 'hybrid')

    return {
        'weather': weather_df,
        'results': results_df,
        'kpis': kpis,
        'data': data,
        'models_used': {
            'pv': pv_model_type,
            'load': load_model_type
        }
    }


def create_live_power_flow(results_df):
    """Create power flow dataframe from optimization results."""
    power_flow = pd.DataFrame({
        'timestamp': results_df['timestamp'],
        'pv_mw': results_df['pv_forecast_mw'],
        'load_mw': results_df['load_forecast_mw'],
        'battery_charge_mw': results_df['battery_charge_mw'],
        'battery_discharge_mw': results_df['battery_discharge_mw'],
        'grid_import_mw': results_df['grid_import_mw'],
        'grid_export_mw': results_df['grid_export_mw'],
        'price_per_kwh': results_df['price_per_kwh'],
        'carbon_intensity': results_df['carbon_intensity']
    })

    # Calculate carbon emissions
    dt = 0.25  # 15 min intervals
    power_flow['carbon_emissions_kg'] = (
        power_flow['grid_import_mw'] * power_flow['carbon_intensity'] * dt
    )

    return power_flow


def plot_pv_forecast(data, live_mode=False):
    """Plot PV forecast vs actual."""
    if live_mode:
        if 'results' not in data:
            st.warning("Live results not available")
            return
        df = data['results']

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['pv_forecast_mw'],
            name='PV Forecast', mode='lines',
            fill='tozeroy',
            line=dict(color='orange', width=2)
        ))
        title = 'PV Generation Forecast (Live Weather)'
    else:
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
        title = 'PV Generation: Forecast vs Actual'

    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title='Power (MW)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_load_forecast(data, live_mode=False):
    """Plot load forecast vs actual."""
    if live_mode:
        if 'results' not in data:
            st.warning("Live results not available")
            return
        df = data['results']

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['load_forecast_mw'],
            name='Load Forecast', mode='lines',
            fill='tozeroy',
            line=dict(color='red', width=2)
        ))
        title = 'Load Demand Forecast (Live)'
    else:
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
        title = 'Load Demand: Forecast vs Actual'

    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title='Power (MW)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_battery_soc(data, live_mode=False):
    """Plot battery SOC over time."""
    if live_mode:
        if 'results' not in data:
            st.warning("Live results not available")
            return
        df = data['results']
    else:
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


def plot_grid_flows(data, live_mode=False):
    """Plot grid import/export with peak demand highlight."""
    if live_mode:
        if 'results' not in data:
            st.warning("Live results not available")
            return
        df = data['results']
    else:
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


def plot_weather_conditions(weather_df):
    """Plot weather conditions from live API."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=('Solar Irradiance', 'Temperature'))

    # GHI and DNI
    fig.add_trace(
        go.Scatter(x=weather_df['timestamp'], y=weather_df['ghi'],
                   name='GHI', line=dict(color='orange')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=weather_df['timestamp'], y=weather_df['dni'],
                   name='DNI', line=dict(color='red', dash='dash')),
        row=1, col=1
    )

    # Temperature
    fig.add_trace(
        go.Scatter(x=weather_df['timestamp'], y=weather_df['temperature'],
                   name='Temperature', line=dict(color='blue')),
        row=2, col=1
    )

    fig.update_layout(height=500, showlegend=True)
    fig.update_yaxes(title_text="W/m²", row=1, col=1)
    fig.update_yaxes(title_text="°C", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


def plot_tariff_and_carbon(data, live_mode=False):
    """Plot tariff prices and carbon intensity over time."""
    if live_mode:
        if 'results' not in data:
            st.warning("Live results not available")
            return
        df = data['results']
    else:
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

    # Carbon intensity
    if 'carbon_intensity' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'], y=df['carbon_intensity'],
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


def plot_power_balance(data, live_mode=False):
    """Plot power balance stacked area chart."""
    if live_mode:
        if 'results' not in data:
            st.warning("Live results not available")
            return
        df = data['results']
        pv_col = 'pv_forecast_mw'
        load_col = 'load_forecast_mw'
    else:
        if 'power_flow' not in data:
            st.warning("Power flow data not available")
            return
        df = data['power_flow']
        pv_col = 'pv_mw'
        load_col = 'load_mw'

    fig = go.Figure()

    # Add traces for each power component
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df[pv_col],
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
        x=df['timestamp'], y=-df[load_col],
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


def calculate_kpis(data, live_mode=False):
    """Calculate and display KPIs with demand charge breakdown."""
    if live_mode:
        if 'results' not in data:
            return
        df = data['results']
        pv_col = 'pv_forecast_mw'
        load_col = 'load_forecast_mw'
    else:
        if 'power_flow' not in data:
            return
        df = data['power_flow']
        pv_col = 'pv_mw'
        load_col = 'load_mw'

    dt = 0.25  # 15 minutes

    # Get config values
    demand_charge_config = config.demand_charge
    DEMAND_CHARGE_ENABLED = demand_charge_config.get('enabled', True)
    DEMAND_CHARGE_RATE = demand_charge_config.get('rate_per_kw', 15.0)
    BILLING_PERIOD_DAYS = demand_charge_config.get('billing_period_days', 30)
    EXPORT_PRICE_RATIO = config.optimization.get('export_price_ratio', 0.5)

    # Calculate metrics
    total_pv = df[pv_col].sum() * dt
    total_load = df[load_col].sum() * dt
    total_import = df['grid_import_mw'].sum() * dt
    total_export = df['grid_export_mw'].sum() * dt

    # Carbon emissions
    if 'carbon_intensity' in df.columns:
        total_carbon = (df['grid_import_mw'] * df['carbon_intensity'] * dt).sum()
    else:
        total_carbon = 0

    # Cost calculations
    if 'price_per_kwh' in df.columns:
        energy_cost = (df['grid_import_mw'] * df['price_per_kwh'] * 1000 * dt).sum()
        export_revenue = (df['grid_export_mw'] * df['price_per_kwh'] * 1000 * EXPORT_PRICE_RATIO * dt).sum()
    else:
        energy_cost = 0
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
                help=f"${DEMAND_CHARGE_RATE}/kW-month x {peak_demand*1000:.0f} kW x {proration_factor:.2f}"
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

    weather = config.get('weather', default={}) or {}
    st.sidebar.subheader("Location")
    st.sidebar.text(f"Lat: {weather.get('latitude', 32.65)}")
    st.sidebar.text(f"Lon: {weather.get('longitude', -117.15)}")


def main():
    """Main dashboard application."""
    st.set_page_config(
        page_title="Microgrid Autopilot",
        page_icon="",
        layout="wide"
    )

    st.title("Microgrid Autopilot Dashboard")

    # Mode selection
    mode = st.radio(
        "Data Source",
        ["Historical Data", "Live Forecast"],
        horizontal=True,
        help="Historical uses pre-processed data. Live fetches real-time weather and runs optimization."
    )

    st.markdown("---")

    # Show configuration in sidebar
    show_configuration()

    live_mode = mode == "Live Forecast"

    if live_mode:
        # Live mode with API forecasting
        st.sidebar.markdown("---")
        horizon = st.sidebar.slider("Forecast Horizon (hours)", 12, 48, 48)

        if st.sidebar.button("Run Live Forecast", type="primary"):
            with st.spinner("Fetching weather and running optimization..."):
                live_data = run_live_forecast(horizon_hours=horizon)
                if live_data:
                    st.session_state['live_data'] = live_data
                    st.success(f"Live forecast complete! Horizon: {horizon} hours")

        # Check if we have live data
        if 'live_data' not in st.session_state:
            st.info("Click 'Run Live Forecast' in the sidebar to fetch real-time weather and run optimization.")
            return

        data = st.session_state['live_data']

        # Show weather conditions
        st.header("Live Weather Conditions")
        plot_weather_conditions(data['weather'])
        st.markdown("---")

        # KPIs Section
        st.header("Key Performance Indicators")
        kpis = calculate_kpis(data, live_mode=True)
        st.markdown("---")

        # Forecasts Section
        st.header("Forecasts")
        col1, col2 = st.columns(2)
        with col1:
            plot_pv_forecast(data, live_mode=True)
        with col2:
            plot_load_forecast(data, live_mode=True)

        st.markdown("---")

        # Battery and Grid Section
        st.header("Battery & Grid Operations")
        col1, col2 = st.columns(2)
        with col1:
            plot_battery_soc(data, live_mode=True)
        with col2:
            plot_grid_flows(data, live_mode=True)

        st.markdown("---")

        # Power Balance Section
        st.header("Power Balance")
        plot_power_balance(data, live_mode=True)

        # Tariff and Carbon Section
        st.header("Tariff & Carbon Intensity")
        plot_tariff_and_carbon(data, live_mode=True)

    else:
        # Historical mode - load from CSV files
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
    st.caption("Microgrid Autopilot v3.0 - Real-Time Energy Management with Live Weather Forecasting")


if __name__ == "__main__":
    main()
