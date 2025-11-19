# Microgrid Autopilot

**Intelligent energy management system for solar + battery microgrids using real-time optimization**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Overview

Microgrid Autopilot is a production-ready system that optimizes battery dispatch in solar+storage microgrids to minimize costs and carbon emissions. It combines machine learning forecasting with Model Predictive Control (MPC) for real-time decision making.

### Key Features

- **Live Weather Integration**: Fetches real-time weather from Open-Meteo API
- **Hybrid Forecasting**: ML models (XGBoost) with physics-based fallback
- **MPC Optimization**: Battery dispatch optimization with demand charge awareness
- **Real-Time Dashboard**: Streamlit interface with live forecasting
- **Production Ready**: Docker deployment, health checks, logging
- **Zero Historical Data Required**: Physics models work out-of-the-box

### System Architecture

```
┌─────────────────┐
│  Live Weather   │
│   (Open-Meteo)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│  PV Forecast    │      │ Load Forecast│
│  (Hybrid ML)    │      │  (Hybrid ML) │
└────────┬────────┘      └──────┬───────┘
         │                      │
         └──────────┬───────────┘
                    ▼
         ┌──────────────────┐
         │  MPC Optimizer   │
         │   (Pyomo+HiGHS)  │
         └─────────┬────────┘
                   ▼
         ┌──────────────────┐
         │ Battery Dispatch │
         │   + Dashboard    │
         └──────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.10+
- Docker (optional, for containerized deployment)

### Installation

\`\`\`bash
# Clone repository
git clone https://github.com/isahan78/microgrid-autopilot.git
cd microgrid-autopilot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
\`\`\`

### Configuration

Edit \`config.yaml\` to match your system:

\`\`\`yaml
# Battery configuration
battery:
  capacity_mwh: 10.0        # Battery size
  max_power_mw: 4.0         # Max charge/discharge rate

# PV system
pv_system:
  capacity_mw: 10.0         # Solar array size
  model_type: "hybrid"      # physics, ml, or hybrid

# Load profile
load_profile:
  base_load_mw: 3.0         # Average load
  model_type: "hybrid"      # profile, ml, or hybrid

# Location (for weather)
weather:
  latitude: 32.65           # Your location
  longitude: -117.15
\`\`\`

### Run Dashboard

\`\`\`bash
# Option 1: Local
streamlit run dashboard/app.py

# Option 2: Docker
docker-compose up -d

# Access at http://localhost:8501
\`\`\`

## Usage

### Live Forecasting

1. Open dashboard: http://localhost:8501
2. Select **"Live Forecast"** mode
3. Click **"Run Live Forecast"**
4. View optimized battery schedule and KPIs

### Training ML Models

\`\`\`bash
# Fetch 90 days of real historical weather data
python utils/fetch_historical_data.py --days 90

# Train models
python forecasting/pv_forecast.py --retrain
python forecasting/load_forecast.py --retrain

# Models automatically used in hybrid mode
\`\`\`

## Model Performance

Trained on 90 days of real San Diego weather data:

| Model | R² Score | MAE | Training Data |
|-------|----------|-----|---------------|
| PV Forecast | 0.9993 | 0.012 MW | 8,733 records |
| Load Forecast | 0.9932 | 0.050 MW | 8,733 records |

## System Performance

Latest live forecast (48-hour horizon):

| Metric | Value |
|--------|-------|
| PV Generation | 49.7 MWh |
| Total Load | 133.2 MWh |
| Self-Sufficiency | 38.5% |
| Grid Import | 81.9 MWh |
| Energy Cost | $13,428 |
| Demand Charge | $6,942 |
| **Net Cost** | **$20,370** |
| Carbon Emissions | 33,598 kg CO2 |

## Docker Deployment

\`\`\`bash
# Build and run
docker-compose up -d

# Services:
# - dashboard (port 8501): Streamlit interface
# - api (port 8000): FastAPI backend
# - scheduler: Rolling horizon MPC
\`\`\`

## API Reference

### Health Check
\`\`\`bash
GET /health

Response:
{
  "status": "healthy",
  "version": "3.0.0",
  "components": {
    "pv_model": "ok",
    "load_model": "ok"
  }
}
\`\`\`

### System Status
\`\`\`bash
GET /status
\`\`\`

## Configuration Reference

### Model Selection

\`\`\`yaml
pv_system:
  model_type: "hybrid"  # Options: physics, ml, hybrid

load_profile:
  model_type: "hybrid"  # Options: profile, ml, hybrid
\`\`\`

- **physics/profile**: No training needed, works immediately
- **ml**: Requires trained models (historical data)
- **hybrid**: Uses ML if available, fallback to physics/profile

### Optimization Parameters

\`\`\`yaml
optimization:
  horizon_hours: 48
  demand_charge:
    enabled: true
    rate_per_kw: 15.0            # $/kW-month
    peak_window_start: 12        # Hour
    peak_window_end: 20
\`\`\`

## Project Structure

\`\`\`
microgrid-autopilot/
├── api/                    # FastAPI backend
├── dashboard/             # Streamlit interface
├── forecasting/           # ML models
├── optimization/          # MPC solver
├── utils/                 # Utilities
│   ├── weather_api.py    # Weather integration
│   ├── pv_model.py       # Physics-based PV
│   ├── logger.py         # Logging
│   └── fetch_historical_data.py
├── models/                # Trained ML models
├── config.yaml           # Configuration
└── docker-compose.yml
\`\`\`

## Real Data Integration

### Your Own System

\`\`\`bash
# Replace synthetic data with your measurements
# Save as: data_processed/pv.csv, load.csv, weather.csv
# Then retrain:
python forecasting/pv_forecast.py --retrain
python forecasting/load_forecast.py --retrain
\`\`\`

### Public Datasets

- **NREL NSRDB**: Solar data (use \`nsrdb_download.py\`)
- **Pecan Street**: Building loads
- **OpenEI**: Commercial profiles

## Troubleshooting

**Models Not Found:**
\`\`\`bash
python utils/fetch_historical_data.py --days 90
python forecasting/pv_forecast.py --retrain
\`\`\`

**Optimization Fails:**
- Check HiGHS solver: \`pip install highspy\`
- Verify config.yaml values
- Check logs in \`logs/\` directory

## Contributing

Contributions welcome! Fork, create feature branch, and open PR.

## License

MIT License - see LICENSE file

## Acknowledgments

- Weather: Open-Meteo Archive API
- Optimization: Pyomo + HiGHS
- ML: XGBoost, scikit-learn
- Dashboard: Streamlit, Plotly

---

**Status**: Production Ready ✅

Last Updated: November 2025
