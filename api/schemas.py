"""
Pydantic schemas for Microgrid Autopilot API.

Defines request/response models for API endpoints.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


# Request schemas
class ForecastRequest(BaseModel):
    """Request for generating forecasts."""
    horizon_hours: Optional[int] = Field(default=48, description="Forecast horizon in hours")


class OptimizeRequest(BaseModel):
    """Request for running optimization."""
    carbon_weight: Optional[float] = Field(default=0.001, description="Weight for carbon in objective")
    initial_soc: Optional[float] = Field(default=0.5, description="Initial battery SOC (0-1)")


class SimulateRequest(BaseModel):
    """Request for running simulation."""
    initial_soc: Optional[float] = Field(default=0.5, description="Initial battery SOC (0-1)")


class RunPipelineRequest(BaseModel):
    """Request for running the full pipeline."""
    process_data: Optional[bool] = Field(default=True, description="Run data processing")
    run_forecast: Optional[bool] = Field(default=True, description="Run forecasting")
    run_optimization: Optional[bool] = Field(default=True, description="Run optimization")
    run_simulation: Optional[bool] = Field(default=True, description="Run simulation")


# Response schemas
class TimeSeriesPoint(BaseModel):
    """Single point in a time series."""
    timestamp: datetime
    value: float


class ForecastResponse(BaseModel):
    """Response containing forecast results."""
    pv_forecast: List[TimeSeriesPoint]
    load_forecast: List[TimeSeriesPoint]
    pv_mae: float
    load_mae: float


class OptimizationResult(BaseModel):
    """Single optimization result point."""
    timestamp: datetime
    pv_forecast_mw: float
    load_forecast_mw: float
    battery_charge_mw: float
    battery_discharge_mw: float
    grid_import_mw: float
    grid_export_mw: float
    soc_percent: float
    price_per_kwh: float


class KPIs(BaseModel):
    """Key performance indicators."""
    total_pv_mwh: float
    total_load_mwh: float
    total_import_mwh: float
    total_export_mwh: float
    net_cost_usd: float
    total_carbon_kg: float
    self_consumption_rate: float
    peak_demand_mw: float


class OptimizeResponse(BaseModel):
    """Response containing optimization results."""
    results: List[OptimizationResult]
    kpis: KPIs


class SimulationResult(BaseModel):
    """Simulation result point."""
    timestamp: datetime
    pv_mw: float
    load_mw: float
    battery_charge_mw: float
    battery_discharge_mw: float
    grid_import_mw: float
    grid_export_mw: float
    soc_percent: float
    net_cost_usd: float
    carbon_emissions_kg: float


class SimulationMetrics(BaseModel):
    """Simulation summary metrics."""
    total_pv_mwh: float
    total_load_mwh: float
    total_import_mwh: float
    total_export_mwh: float
    net_cost_usd: float
    total_carbon_kg: float
    self_consumption_rate_pct: float
    self_sufficiency_pct: float
    peak_import_mw: float


class BaselineComparison(BaseModel):
    """Comparison with baseline scenario."""
    baseline_cost_usd: float
    optimized_cost_usd: float
    cost_savings_usd: float
    cost_savings_pct: float
    carbon_reduction_kg: float
    carbon_reduction_pct: float


class SimulateResponse(BaseModel):
    """Response containing simulation results."""
    results: List[SimulationResult]
    metrics: SimulationMetrics
    comparison: BaselineComparison


class PipelineStatus(BaseModel):
    """Status of pipeline execution."""
    step: str
    status: str
    message: str


class RunPipelineResponse(BaseModel):
    """Response from running the full pipeline."""
    status: str
    steps: List[PipelineStatus]
    kpis: Optional[KPIs] = None
    comparison: Optional[BaselineComparison] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    timestamp: datetime


class BatteryState(BaseModel):
    """Current battery state."""
    soc_percent: float
    soc_mwh: float
    available_energy_mwh: float
    available_capacity_mwh: float
    max_charge_power_mw: float
    max_discharge_power_mw: float


class SystemStatus(BaseModel):
    """Overall system status."""
    health: str
    battery: BatteryState
    last_optimization: Optional[datetime] = None
    data_freshness: Optional[datetime] = None
