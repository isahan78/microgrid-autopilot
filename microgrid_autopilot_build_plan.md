# ⚡ **MICROGRID AUTOPILOT — STATE-OF-THE-ART MVP BUILD PLAN**

## Full Claude-Ready Markdown Document

---

# 🧠 **1. Project Overview**

**Microgrid Autopilot** is an intelligent control system for **PV + Battery + Load** that:

- Forecasts solar generation  
- Forecasts load demand  
- Optimizes battery charging/discharging  
- Minimizes cost & carbon  
- Simulates operations  
- Runs behind an API  
- Provides a visual dashboard  

This MVP uses **real PV**, **real load**, **real tariff**, and **real weather** data:

```
Actual_32.65_-117.15_2006_DPV_11MW_5_Min.csv     → pv_raw.csv  
time_series_60min_singleindex.csv               → load_raw.csv  
usurdb.csv                                       → tariff_raw.csv  
weather_goes_psm4.csv                            → weather_raw.csv  
carbon_raw.csv (single monthly value)
```

The output is a complete working system structured for R&D, investors, and productization.

---

# 📁 **2. Required Project Structure**

```
microgrid_autopilot/
├── data_raw/
│   ├── pv_raw.csv
│   ├── load_raw.csv
│   ├── tariff_raw.csv
│   ├── weather_raw.csv
│   └── carbon_raw.csv
├── data_processed/
│   ├── pv.csv
│   ├── load.csv
│   ├── tariff.csv
│   ├── weather.csv
│   └── carbon.csv
├── data_prep/
│   └── process_data.py
├── forecasting/
│   ├── pv_forecast.py
│   └── load_forecast.py
├── optimization/
│   ├── mpc_solver.py
│   └── fallback_rules.py
├── simulation/
│   ├── battery_sim.py
│   └── power_flow.py
├── api/
│   ├── main.py
│   ├── controller.py
│   └── schemas.py
├── dashboard/
│   └── app.py
├── tests/
│   ├── test_data_pipeline.py
│   ├── test_forecasting.py
│   └── test_mpc.py
├── requirements.txt
└── README.md
```

---

# 📊 **3. Data Processing Requirements**

Claude must create `data_prep/process_data.py` implementing the following logic.

## **3.1 PV Data (pv_raw.csv)**

- Read raw file  
- Parse timestamps  
- Extract:

```
2006-07-15 → 2006-07-16
```

- Resample to **15 minutes**  
- Save as:

```
timestamp, pv_power_mw
```

## **3.2 Load Data (load_raw.csv)**

- Extract any 48-hour window  
- Resample hourly → 15 minutes  
- Align timestamps to PV window  
- Save as:

```
timestamp, load_mw
```

## **3.3 Tariff Data (tariff_raw.csv)**

Synthetic TOU schedule:

| Hour | Price ($/kWh) |
|------|---------------|
| 00–16 | 0.12 |
| 17–21 | 0.30 |
| 22–24 | 0.15 |

Save as:

```
timestamp, price_per_kwh
```

## **3.4 Carbon Data (carbon_raw.csv)**

- Read monthly carbon value  
- Repeat for entire window  
- Save as:

```
timestamp, carbon_intensity
```

## **3.5 Weather Data (weather_raw.csv)**

- Extract:
```
ghi, dni, air_temperature
```
- Resample to 15 minutes  
- Reassign timestamps to PV window  
- Save as:

```
timestamp, ghi, dni, temperature
```

---

# 🔮 **4. Forecasting Requirements**

## **4.1 PV Forecast**

Model inputs:
- Past PV lag features  
- GHI, DNI  
- Temperature  
- Hour-of-day, day-of-year  

Model: **XGBoost Regressor**

Output:
```
forecast_pv.csv
```

## **4.2 Load Forecast**

Model inputs:
- Past load lags  
- Hour-of-day, day-of-week  
- Temperature  

Model: **XGBoost Regressor**

Output:
```
forecast_load.csv
```

---

# 🧩 **5. MPC Optimization Requirements**

## **5.1 Objective**

```
min Σ (price[t] * grid_import[t] + carbon_weight * carbon_intensity[t] * grid_import[t])
```

## **5.2 Constraints**

Battery model:
- 0.2 ≤ SOC ≤ 0.9  
- charge/discharge power limits  
- efficiency model  

Power balance:
```
grid = load - pv - battery
```

## **5.3 Solver**

- Pyomo + HiGHS or OR-Tools CP-SAT
- Horizon: **96 steps** (48 hours @ 30m or 15m)

## **5.4 Fallback Logic**

- Charge during cheap hours  
- Discharge during expensive hours  

---

# 🪫 **6. Simulation Layer**

## **6.1 Battery Simulation**

Computes:
- SOC trajectory  
- charge/discharge time series  

## **6.2 Power Flow**

```
net_power = pv + battery - load
```

Outputs:
- grid import/export  
- cost  
- carbon  
- peak demand  

---

# 🌐 **7. API Layer (FastAPI)**

Endpoints:
```
POST /forecast
POST /optimize
POST /simulate
POST /run
```

---

# 📊 **8. Dashboard (Streamlit)**

Visualization panels:
- PV forecast vs actual  
- Load forecast vs actual  
- Battery SOC  
- Grid import/export  
- Tariff overlays  
- Carbon/cost KPIs  

---

# 📦 **9. requirements.txt**

```
pandas
numpy
xgboost
scikit-learn
pyomo
ortools
fastapi
uvicorn
streamlit
plotly
pydantic
python-dotenv
```

---

# 🧠 **10. End-to-End Pipeline**

```
python data_prep/process_data.py
python forecasting/pv_forecast.py
python forecasting/load_forecast.py
python optimization/mpc_solver.py
python simulation/power_flow.py
streamlit run dashboard/app.py
```

---

# 🏁 **11. MVP Success Criteria**

The Claude-generated system must:
- Run end-to-end  
- Produce PV + load forecasts  
- Generate optimized battery schedule  
- Simulate grid behavior  
- Compute cost/carbon savings  
- Provide a dashboard  
- Offer full API control  
- Be investor-ready  

---

# 🚀 **READY FOR CLAUDE**

Use this command:

> **“Generate the full project scaffold exactly as described.”**

