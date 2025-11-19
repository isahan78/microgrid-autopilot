# Microgrid Autopilot

An intelligent control system for optimizing PV + Battery + Load microgrids. This system forecasts solar generation and load demand, optimizes battery scheduling to minimize cost and carbon emissions, and provides real-time visualization through an interactive dashboard.

## Features

- **Solar PV Forecasting**: XGBoost-based model using weather data, time features, and historical patterns
- **Load Demand Forecasting**: XGBoost-based model with temperature and temporal features
- **Battery Optimization**: MPC (Model Predictive Control) optimization with Pyomo or rule-based fallback
- **Power Flow Simulation**: Calculates grid import/export, costs, and carbon emissions
- **REST API**: FastAPI endpoints for programmatic control
- **Interactive Dashboard**: Streamlit-based visualization of all KPIs and time series

## Project Structure

```
microgrid_autopilot/
├── data_raw/                    # Raw input data
│   ├── pv_raw.csv              # Solar generation data
│   ├── load_raw.csv            # Load demand data
│   ├── tariff_raw.csv          # Electricity tariff data
│   ├── weather_raw.csv         # Weather data (GHI, DNI, temperature)
│   └── carbon_raw.csv          # Carbon intensity data
├── data_processed/              # Processed data and results
│   ├── pv.csv                  # Processed PV data
│   ├── load.csv                # Processed load data
│   ├── tariff.csv              # TOU tariff schedule
│   ├── weather.csv             # Processed weather data
│   ├── carbon.csv              # Carbon intensity
│   ├── forecast_pv.csv         # PV forecast results
│   ├── forecast_load.csv       # Load forecast results
│   ├── optimization_results.csv # Battery schedule
│   └── power_flow.csv          # Power flow simulation
├── data_prep/
│   └── process_data.py         # Data processing pipeline
├── forecasting/
│   ├── pv_forecast.py          # PV forecasting model
│   └── load_forecast.py        # Load forecasting model
├── optimization/
│   ├── mpc_solver.py           # MPC optimization solver
│   └── fallback_rules.py       # Rule-based fallback control
├── simulation/
│   ├── battery_sim.py          # Battery SOC simulation
│   └── power_flow.py           # Power flow calculations
├── api/
│   ├── main.py                 # FastAPI application
│   ├── controller.py           # Business logic
│   └── schemas.py              # Pydantic models
├── dashboard/
│   └── app.py                  # Streamlit dashboard
├── tests/
│   ├── test_data_pipeline.py   # Data processing tests
│   ├── test_forecasting.py     # Forecasting tests
│   └── test_mpc.py             # Optimization tests
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. Clone or navigate to the project directory:
```bash
cd microgrid-autopilot
```

2. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) Install HiGHS solver for full MPC optimization:
```bash
pip install highspy
```

## Usage

### Run the Complete Pipeline

Execute all steps sequentially:

```bash
# Activate virtual environment
source venv/bin/activate

# 1. Process raw data
python data_prep/process_data.py

# 2. Generate PV forecast
python forecasting/pv_forecast.py

# 3. Generate load forecast
python forecasting/load_forecast.py

# 4. Run optimization
python optimization/mpc_solver.py

# 5. Run power flow simulation
python simulation/power_flow.py
```

### Launch Dashboard

```bash
streamlit run dashboard/app.py
```

Open http://localhost:8501 in your browser.

### Start API Server

```bash
uvicorn api.main:app --reload
```

API available at http://localhost:8000

API documentation at http://localhost:8000/docs

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Health check |
| `/forecast` | POST | Generate PV and load forecasts |
| `/optimize` | POST | Run battery optimization |
| `/simulate` | POST | Run power flow simulation |
| `/run` | POST | Execute complete pipeline |

### Example API Usage

```python
import requests

# Run complete pipeline
response = requests.post("http://localhost:8000/run")
result = response.json()

print(f"Status: {result['status']}")
print(f"Net Cost: ${result['kpis']['net_cost_usd']:.2f}")
print(f"Cost Savings: ${result['comparison']['cost_savings_usd']:.2f}")
```

## System Parameters

### Battery Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Capacity | 5.0 MWh | Total battery capacity |
| Max Power | 2.0 MW | Max charge/discharge rate |
| Charge Efficiency | 95% | Charging efficiency |
| Discharge Efficiency | 95% | Discharging efficiency |
| Min SOC | 20% | Minimum state of charge |
| Max SOC | 90% | Maximum state of charge |
| Initial SOC | 50% | Starting state of charge |

### TOU Tariff Schedule

| Hours | Price ($/kWh) | Period |
|-------|---------------|--------|
| 00:00 - 17:00 | $0.12 | Off-peak |
| 17:00 - 22:00 | $0.30 | Peak |
| 22:00 - 24:00 | $0.15 | Mid-peak |

## Performance Results

### Model Performance

| Model | R² Score | MAE | RMSE |
|-------|----------|-----|------|
| PV Forecast | 0.91 | 0.10 MW | 0.55 MW |
| Load Forecast | 0.97 | 0.02 MW | 0.12 MW |

### Sample 48-Hour Results

| Metric | Value |
|--------|-------|
| Total PV Generation | 128.56 MWh |
| Total Load | 233.12 MWh |
| Grid Import | 128.98 MWh |
| Grid Export | 25.27 MWh |
| Net Cost | $21,867.21 |
| Carbon Emissions | 52,879.89 kg CO2 |
| Self-Consumption Rate | 80.3% |
| Self-Sufficiency | 44.7% |
| **Cost Savings vs Baseline** | **$1,300.26 (5.6%)** |
| **Carbon Reduction** | **352.82 kg (0.7%)** |

## Dashboard Visualizations

The Streamlit dashboard provides:

- **KPI Summary**: Total generation, load, import/export, cost, carbon
- **PV Forecast vs Actual**: Time series comparison
- **Load Forecast vs Actual**: Time series comparison
- **Battery SOC**: State of charge over time with min/max limits
- **Grid Import/Export**: Power flows with the grid
- **Power Balance**: Stacked area chart of all power sources
- **Tariff Schedule**: TOU pricing visualization

## Optimization Approach

### MPC Optimization

The system uses Model Predictive Control (MPC) to optimize battery scheduling:

**Objective Function:**
```
minimize Σ (price[t] × grid_import[t] + carbon_weight × carbon_intensity[t] × grid_import[t])
```

**Subject to:**
- Power balance: `grid_import - grid_export = load - pv - battery_discharge + battery_charge`
- SOC dynamics: `SOC[t+1] = SOC[t] + charge × efficiency - discharge / efficiency`
- SOC limits: `0.2 ≤ SOC ≤ 0.9`
- Power limits: `charge, discharge ≤ 2 MW`

### Fallback Rules

When no LP solver is available, the system uses rule-based control:
- Charge battery with excess PV
- Discharge during peak pricing (≥ $0.25/kWh)
- Charge from grid during off-peak (< $0.15/kWh)

## Data Sources

The system uses real-world data formats:

- **PV Data**: NREL System Advisor Model (SAM) output
- **Load Data**: ENTSO-E European power system data (scaled)
- **Weather Data**: NSRDB (National Solar Radiation Database)
- **Tariff**: Synthetic TOU based on typical utility rates
- **Carbon**: Monthly average grid carbon intensity

## Testing

Run the test suite:

```bash
# Install pytest
pip install pytest

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_mpc.py -v
```

## Dependencies

Core dependencies:
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `xgboost` - Machine learning models
- `scikit-learn` - ML utilities
- `pyomo` - Optimization modeling
- `fastapi` - REST API
- `uvicorn` - ASGI server
- `streamlit` - Dashboard
- `plotly` - Interactive visualizations
- `pydantic` - Data validation

## Troubleshooting

### No LP Solver Available

If you see "No LP solver available, using fallback rules":

```bash
pip install highspy
```

Or install GLPK:
```bash
# macOS
brew install glpk

# Ubuntu
sudo apt-get install glpk-utils
```

### Module Not Found Errors

Ensure you're running from the project root:
```bash
cd /path/to/microgrid-autopilot
source venv/bin/activate
```

### Dashboard Not Loading

Check that all pipeline steps have been run and data files exist in `data_processed/`.

## Future Enhancements

- [ ] Real-time data integration
- [ ] Multi-day rolling horizon optimization
- [ ] Demand response integration
- [ ] EV charging optimization
- [ ] Weather forecast API integration
- [ ] Database persistence
- [ ] User authentication
- [ ] Configurable battery parameters via UI

## License

MIT License

## Acknowledgments

- NREL for PV modeling tools and NSRDB data
- ENTSO-E for European power system data
- XGBoost team for the gradient boosting library
- Pyomo team for optimization modeling

---

**Microgrid Autopilot v1.0** - Intelligent Energy Management System
