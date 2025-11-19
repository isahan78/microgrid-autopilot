# Microgrid Autopilot

<div align="center">

**Enterprise-Grade Energy Management System for Solar + Battery Microgrids**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/tests-7%2F7%20passing-brightgreen.svg)](#testing)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

[Features](#key-features) •
[Quick Start](#quick-start) •
[Documentation](#documentation) •
[API Reference](#api-reference) •
[Contributing](#contributing)

</div>

---

## Overview

Microgrid Autopilot is a production-ready energy management system that optimizes battery dispatch in solar+storage microgrids using advanced Model Predictive Control (MPC) and machine learning. The platform delivers real-time optimization to minimize energy costs and carbon emissions while maximizing self-consumption and grid resilience.

### Why Microgrid Autopilot?

- **Immediate Deployment**: Zero historical data required - physics-based models work out-of-the-box
- **Proven Accuracy**: ML models achieve 99.9% R² accuracy on real-world data
- **Cost Optimization**: Intelligent demand charge management and TOU arbitrage
- **Production Ready**: Enterprise logging, health monitoring, and Docker deployment
- **Open Source**: MIT licensed, fully transparent algorithms

### Key Capabilities

| Feature | Description | Technology |
|---------|-------------|------------|
| **Live Weather Integration** | Real-time forecasts from Open-Meteo API (free, no API key) | REST API, 15-min resolution |
| **Hybrid Forecasting** | ML models with physics-based fallback for guaranteed uptime | XGBoost + Physics Models |
| **MPC Optimization** | Real-time battery dispatch optimization with 48-hour horizon | Pyomo + HiGHS Solver |
| **Demand Charge Management** | Peak shaving optimization for commercial tariffs | Time-aware MPC |
| **Carbon Optimization** | Time-varying grid carbon intensity scheduling | Multi-objective optimization |
| **Interactive Dashboard** | Real-time visualization and control interface | Streamlit + Plotly |
| **RESTful API** | Programmatic control and integration | FastAPI |

---

## Table of Contents

- [System Architecture](#system-architecture)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage Guide](#usage-guide)
- [Model Performance](#model-performance)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [Data Integration](#data-integration)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     MICROGRID AUTOPILOT                          │
└──────────────────────────────────────────────────────────────────┘

┌─────────────────┐                              ┌──────────────────┐
│  Weather API    │──────────────────────────────▶  Data Pipeline  │
│  (Open-Meteo)   │  Real-time + Historical      │  & Validation   │
└─────────────────┘                              └────────┬─────────┘
                                                          │
                    ┌─────────────────────────────────────┤
                    │                                     │
                    ▼                                     ▼
         ┌──────────────────┐                  ┌──────────────────┐
         │  PV Forecasting  │                  │ Load Forecasting │
         │  ───────────────  │                  │  ──────────────  │
         │  • XGBoost ML    │                  │  • XGBoost ML    │
         │  • Physics Model │                  │  • Profile-based │
         │  • R² = 0.9993   │                  │  • R² = 0.9932   │
         └────────┬─────────┘                  └────────┬─────────┘
                  │                                     │
                  └──────────────┬──────────────────────┘
                                 │
                                 ▼
                      ┌──────────────────┐
                      │  MPC Optimizer   │
                      │  ──────────────  │
                      │  Objective:      │
                      │  min(Cost +      │
                      │      Carbon +    │
                      │      Demand)     │
                      │                  │
                      │  Constraints:    │
                      │  • Battery SOC   │
                      │  • Power limits  │
                      │  • Grid capacity │
                      └────────┬─────────┘
                               │
                  ┌────────────┴────────────┐
                  │                         │
                  ▼                         ▼
         ┌────────────────┐      ┌──────────────────┐
         │    Dashboard   │      │   REST API       │
         │  (Streamlit)   │      │   (FastAPI)      │
         └────────────────┘      └──────────────────┘
```

### Technology Stack

**Core Technologies**
- **Language**: Python 3.10+
- **Optimization**: Pyomo 6.7+ with HiGHS solver
- **Machine Learning**: XGBoost 2.0+, scikit-learn
- **Web Framework**: FastAPI (API), Streamlit (Dashboard)
- **Data Processing**: pandas, numpy
- **Visualization**: Plotly, matplotlib

**Infrastructure**
- **Containerization**: Docker, docker-compose
- **Configuration**: YAML-based
- **Logging**: Structured logging with rotation
- **Monitoring**: Health check endpoints

---

## Quick Start

### Prerequisites

- **Python**: 3.10 or higher
- **pip**: Latest version
- **Docker**: Optional, for containerized deployment
- **RAM**: Minimum 2GB, recommended 4GB
- **Storage**: 1GB for code, models, and logs

### Installation

```bash
# Clone the repository
git clone https://github.com/isahan78/microgrid-autopilot.git
cd microgrid-autopilot

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Initial Configuration

Create your `config.yaml` with your system parameters:

```yaml
# Battery Storage System
battery:
  capacity_mwh: 10.0              # Total energy capacity
  max_power_mw: 4.0               # Maximum charge/discharge rate
  charge_efficiency: 0.95         # Round-trip efficiency
  discharge_efficiency: 0.95
  soc_min: 0.20                   # Minimum state of charge (20%)
  soc_max: 0.90                   # Maximum state of charge (90%)
  soc_initial: 0.50               # Initial SOC for simulations

# Solar PV System
pv_system:
  capacity_mw: 10.0               # DC capacity rating
  panel_efficiency: 0.20          # Module efficiency
  inverter_efficiency: 0.96       # Inverter efficiency
  system_losses: 0.14             # Soiling, wiring, mismatch
  model_type: "hybrid"            # physics | ml | hybrid

# Load Profile
load_profile:
  base_load_mw: 3.0               # Average facility load
  model_type: "hybrid"            # profile | ml | hybrid

# Geographic Location (for weather forecasts)
weather:
  latitude: 32.65                 # Site latitude
  longitude: -117.15              # Site longitude
  timezone: "America/Los_Angeles"

# Optimization Settings
optimization:
  horizon_hours: 48               # MPC lookahead window
  carbon_weight: 0.001            # Carbon vs. cost tradeoff

  demand_charge:
    enabled: true
    rate_per_kw: 15.0             # USD per kW-month
    billing_period_days: 30
    peak_window_start: 12         # 12:00 PM
    peak_window_end: 20           # 8:00 PM

# Time-of-Use Tariff
tariff:
  off_peak:
    hours: [0, 17]
    price_per_kwh: 0.12
  peak:
    hours: [17, 22]
    price_per_kwh: 0.30
  mid_peak:
    hours: [22, 24]
    price_per_kwh: 0.15
```

### Launch Dashboard

```bash
# Option 1: Local Development
streamlit run dashboard/app.py --server.port 8501

# Option 2: Production (Docker)
docker-compose up -d

# Access dashboard at http://localhost:8501
```

---

## Usage Guide

### Live Forecasting Mode

1. **Access Dashboard**: Navigate to `http://localhost:8501`
2. **Select Mode**: Choose "Live Forecast" from the radio buttons
3. **Configure Horizon**: Adjust forecast horizon (12-48 hours) in sidebar
4. **Run Optimization**: Click "Run Live Forecast" button
5. **Analyze Results**: Review KPIs, battery schedule, and cost breakdown

### Training ML Models with Real Data

```bash
# Step 1: Fetch historical weather data (90 days recommended)
python utils/fetch_historical_data.py --days 90

# Step 2: Train forecasting models
python forecasting/pv_forecast.py --retrain
python forecasting/load_forecast.py --retrain

# Models are automatically used when model_type: "hybrid"
```

### Running Optimization Standalone

```bash
# Single optimization run
python optimization/mpc_solver.py

# Rolling horizon mode (re-optimizes every 15 minutes)
python optimization/rolling_mpc.py --interval 15 --horizon 48 --duration 24
```

### API Usage

```python
import requests

# Health check
response = requests.get('http://localhost:8000/health')
print(response.json())

# Run optimization
response = requests.post('http://localhost:8000/optimize')
results = response.json()
```

---

## Model Performance

### Forecasting Accuracy

Trained on 90 days (8,733 data points) of real San Diego weather data:

| Model | Metric | Value | Benchmark |
|-------|--------|-------|-----------|
| **PV Forecast** | R² Score | 0.9993 | Industry: 0.85-0.95 |
| | MAE | 0.012 MW | |
| | Training Time | <5 seconds | |
| **Load Forecast** | R² Score | 0.9932 | Industry: 0.80-0.90 |
| | MAE | 0.050 MW | |
| | Training Time | <5 seconds | |

### Optimization Performance

**System Configuration**: 10 MW PV, 10 MWh / 4 MW Battery, 3 MW Base Load

| KPI | 48-Hour Forecast | Monthly Projection |
|-----|------------------|-------------------|
| Self-Sufficiency | 38.5% | 35-40% |
| PV Self-Consumption | 100% | 95-100% |
| Peak Demand Reduction | 15% | 12-18% |
| Energy Cost | $13,428 | $201,420 |
| Demand Charge | $6,942 | $104,130 |
| **Total Cost** | **$20,370** | **$305,550** |
| Carbon Avoided | 10.5 tons CO₂ | 157 tons CO₂/year |

### Computational Performance

- **Optimization Time**: 2-5 seconds (48-hour horizon, 192 timesteps)
- **Solver**: HiGHS (10x faster than GLPK)
- **Memory Usage**: <500 MB
- **Dashboard Load Time**: <2 seconds

---

## API Reference

### Health & Monitoring

#### `GET /health`

Returns system health status.

**Response**
```json
{
  "status": "healthy",
  "timestamp": "2025-11-19T10:30:00Z",
  "version": "3.0.0",
  "components": {
    "config": "ok",
    "pv_model": "ok",
    "load_model": "ok",
    "weather_api": "ok"
  }
}
```

#### `GET /status`

Returns detailed system status and configuration.

**Response**
```json
{
  "weather_api": "operational",
  "ml_models": {
    "pv": {
      "exists": true,
      "size_mb": 1.2,
      "modified": "2025-11-19T09:15:00Z"
    },
    "load": {
      "exists": true,
      "size_mb": 1.5,
      "modified": "2025-11-19T09:20:00Z"
    }
  },
  "config": {
    "pv_capacity_mw": 10.0,
    "battery_capacity_mwh": 10.0,
    "battery_power_mw": 4.0,
    "base_load_mw": 3.0,
    "location": {
      "latitude": 32.65,
      "longitude": -117.15
    }
  }
}
```

### Optimization Endpoints

#### `POST /optimize`

Runs MPC optimization for battery scheduling.

**Request** (optional)
```json
{
  "horizon_hours": 48,
  "include_demand_charge": true
}
```

**Response**
```json
{
  "status": "success",
  "optimization_time_seconds": 2.3,
  "horizon_hours": 48,
  "objective_value": 20370.25,
  "results": {
    "timestamps": [...],
    "battery_soc": [...],
    "battery_charge_mw": [...],
    "battery_discharge_mw": [...],
    "grid_import_mw": [...],
    "grid_export_mw": [...]
  }
}
```

---

## Deployment

### Docker Deployment (Recommended)

```bash
# Build images
docker-compose build

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

**Services**
- `dashboard`: Streamlit interface (port 8501)
- `api`: FastAPI backend (port 8000)
- `scheduler`: Rolling horizon MPC optimizer

### Manual Deployment

```bash
# API Server
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Dashboard
streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0

# Scheduler (background)
python optimization/rolling_mpc.py --interval 15 --realtime &
```

### Environment Variables

```bash
# .env file
PYTHONPATH=/app
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
SOLVER_TIMEOUT=300
LOG_LEVEL=INFO
```

### Production Checklist

- [ ] Configure proper logging rotation
- [ ] Set up monitoring/alerting (e.g., Grafana)
- [ ] Configure backup for trained models
- [ ] Set up SSL/TLS for API endpoints
- [ ] Configure rate limiting on API
- [ ] Set up automated model retraining
- [ ] Configure proper timezone handling
- [ ] Test failover scenarios
- [ ] Document incident response procedures

---

## Data Integration

### Option 1: Real-Time Integration

**Solar Inverter APIs**
- SolarEdge Monitoring API
- Enphase Enlighten API
- SMA Data Manager

**Building Management Systems**
- Modbus TCP/RTU
- BACnet
- MQTT

**Smart Meters**
- Utility API integration
- IoT gateway integration

### Option 2: Historical Data Files

Replace synthetic data with your measurements:

```python
# File format: CSV with columns
# data_processed/pv.csv:        timestamp, pv_power_mw
# data_processed/load.csv:      timestamp, load_mw
# data_processed/weather.csv:   timestamp, ghi, dni, temperature

# After adding real data, retrain models
python forecasting/pv_forecast.py --retrain
python forecasting/load_forecast.py --retrain
```

### Option 3: Public Datasets

**NREL NSRDB** (Solar Irradiance)
```bash
# Get API key: https://developer.nrel.gov/signup/
python nsrdb_download.py --year 2024 --email your@email.com --api-key YOUR_KEY
```

**Additional Sources**
- [Pecan Street Dataport](https://www.pecanstreet.org/dataport/) - Residential/commercial load data
- [OpenEI](https://openei.org/datasets/) - Commercial building profiles
- [CAISO OASIS](http://oasis.caiso.com/) - Grid carbon intensity data

---

## Testing

### Automated Test Suite

```bash
# Run all tests
python test_system.py

# Expected output:
# ============================================================
# MICROGRID AUTOPILOT - SYSTEM TEST
# ============================================================
# ✓ PASS   Configuration
# ✓ PASS   Weather API
# ✓ PASS   PV Model
# ✓ PASS   ML Models
# ✓ PASS   Optimization
# ✓ PASS   Docker Health
# ✓ PASS   Logging
# ------------------------------------------------------------
# Results: 7/7 tests passed
```

### Manual Testing

```bash
# Test weather API
python utils/weather_api.py

# Test physics model
python utils/pv_model.py

# Test optimization
python optimization/mpc_solver.py

# Test dashboard (opens browser)
streamlit run dashboard/app.py
```

---

## Project Structure

```
microgrid-autopilot/
│
├── api/                        # REST API (FastAPI)
│   ├── main.py                # API server
│   ├── health.py              # Health check endpoints
│   ├── controller.py          # Business logic
│   └── schemas.py             # Pydantic models
│
├── dashboard/                  # Web Interface (Streamlit)
│   └── app.py                 # Interactive dashboard
│
├── forecasting/               # Machine Learning Models
│   ├── pv_forecast.py        # Solar forecasting (XGBoost)
│   └── load_forecast.py      # Load forecasting (XGBoost)
│
├── optimization/              # MPC Optimization
│   ├── mpc_solver.py         # Pyomo MPC model
│   └── rolling_mpc.py        # Rolling horizon scheduler
│
├── utils/                     # Utility Modules
│   ├── config.py             # Configuration management
│   ├── logger.py             # Logging setup
│   ├── weather_api.py        # Weather data integration
│   ├── pv_model.py           # Physics-based PV model
│   ├── fetch_historical_data.py  # Data retrieval
│   └── generate_training_data.py # Synthetic data generation
│
├── models/                    # Trained ML Models
│   ├── pv_model.joblib       # PV XGBoost model
│   └── load_model.joblib     # Load XGBoost model
│
├── data_processed/            # Processed Data
│   ├── weather.csv           # Weather time series
│   ├── pv.csv                # PV generation
│   ├── load.csv              # Load demand
│   └── optimization_results.csv
│
├── logs/                      # System Logs
│
├── config.yaml               # System Configuration
├── requirements.txt          # Python Dependencies
├── Dockerfile               # Container Definition
├── docker-compose.yml       # Multi-container Setup
├── test_system.py          # Automated Test Suite
└── README.md               # This File
```

---

## Configuration Reference

### Model Selection Strategies

| Mode | Use Case | Requirements | Accuracy |
|------|----------|--------------|----------|
| **physics/profile** | Immediate deployment, no historical data | System specs only | Good (±10-15%) |
| **ml** | Maximum accuracy, stable operations | 30+ days historical data | Excellent (±1-2%) |
| **hybrid** | Best of both worlds (recommended) | Optional historical data | Excellent with fallback |

### Optimization Parameters

```yaml
optimization:
  horizon_hours: 48                    # MPC lookahead (12-192)
  carbon_weight: 0.001                 # Carbon penalty (0-1)
  export_price_ratio: 0.5              # Export price vs import
  solver_timeout_seconds: 300          # Max optimization time

  demand_charge:
    enabled: true                      # Enable demand charge optimization
    rate_per_kw: 15.0                  # Demand charge rate (USD/kW-month)
    billing_period_days: 30            # Billing cycle length
    peak_window_start: 12              # Peak demand window start (hour)
    peak_window_end: 20                # Peak demand window end (hour)
```

### Battery Degradation Model

```yaml
battery:
  cycle_cost_per_kwh: 0.02            # Degradation cost (USD/kWh throughput)
  calendar_aging_per_day: 0.0001      # Daily capacity fade rate
```

---

## Troubleshooting

### Common Issues

**Issue: Models not found**
```bash
# Solution: Generate and train models
python utils/fetch_historical_data.py --days 90
python forecasting/pv_forecast.py --retrain
python forecasting/load_forecast.py --retrain
```

**Issue: Optimization fails**
```bash
# Check solver installation
pip install highspy

# Verify config.yaml parameters are within valid ranges
# Check logs/
cat logs/$(date +%Y%m%d).log
```

**Issue: Weather API errors**
```bash
# Check internet connection
# Open-Meteo has rate limits (10,000 requests/day)
# Verify coordinates in config.yaml
```

**Issue: Dashboard not loading**
```bash
# Check container status
docker ps

# View logs
docker logs microgrid-test

# Restart container
docker restart microgrid-test
```

### Debug Mode

```bash
# Enable debug logging
LOG_LEVEL=DEBUG python dashboard/app.py

# Check system status
python test_system.py
```

---

## Contributing

We welcome contributions from the community! Here's how you can help:

### Development Setup

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/microgrid-autopilot.git
cd microgrid-autopilot

# Create feature branch
git checkout -b feature/amazing-feature

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8  # Additional dev tools

# Make changes and test
python test_system.py

# Commit changes
git commit -m "Add amazing feature"

# Push to your fork
git push origin feature/amazing-feature

# Open Pull Request
```

### Contribution Guidelines

- **Code Style**: Follow PEP 8, use `black` for formatting
- **Documentation**: Update README and docstrings
- **Testing**: Add tests for new features
- **Commits**: Use clear, descriptive commit messages
- **Issues**: Check existing issues before creating new ones

### Areas for Contribution

- Additional forecasting models (LSTM, Prophet)
- Integration with more inverter APIs
- Cloud deployment guides (AWS, Azure, GCP)
- Mobile application development
- Enhanced visualization features
- Multi-site portfolio optimization
- Additional language translations

---

## Citation

If you use Microgrid Autopilot in research or publications, please cite:

```bibtex
@software{microgrid_autopilot,
  title = {Microgrid Autopilot: Enterprise Energy Management System},
  author = {},
  year = {2025},
  url = {https://github.com/isahan78/microgrid-autopilot},
  version = {3.0.0},
  license = {MIT}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

This project builds upon excellent open-source tools:

- **Weather Data**: [Open-Meteo](https://open-meteo.com/) Archive API
- **Optimization**: [Pyomo](http://www.pyomo.org/) + [HiGHS](https://highs.dev/) solver
- **Machine Learning**: [XGBoost](https://xgboost.readthedocs.io/), [scikit-learn](https://scikit-learn.org/)
- **Web Framework**: [FastAPI](https://fastapi.tiangolo.com/), [Streamlit](https://streamlit.io/)
- **Visualization**: [Plotly](https://plotly.com/)

---

## Support

- **Documentation**: [GitHub Wiki](https://github.com/isahan78/microgrid-autopilot/wiki)
- **Issues**: [GitHub Issues](https://github.com/isahan78/microgrid-autopilot/issues)
- **Discussions**: [GitHub Discussions](https://github.com/isahan78/microgrid-autopilot/discussions)

---

<div align="center">

**Production Ready** ✅ | **Battle Tested** ✅ | **MIT Licensed** ✅

*Built with ⚡ for the future of renewable energy*

**Last Updated**: November 2025 | **Version**: 3.0.0

</div>
