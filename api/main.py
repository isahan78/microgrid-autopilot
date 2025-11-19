"""
FastAPI main application for Microgrid Autopilot.

Provides REST API endpoints for the control system.
"""

import sys
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.schemas import (
    ForecastRequest, OptimizeRequest, SimulateRequest, RunPipelineRequest,
    ForecastResponse, OptimizeResponse, SimulateResponse, RunPipelineResponse,
    HealthResponse
)
from api.controller import controller


# Create FastAPI app
app = FastAPI(
    title="Microgrid Autopilot API",
    description="Intelligent control system for PV + Battery + Load optimization",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now()
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now()
    )


@app.post("/forecast", response_model=ForecastResponse)
async def run_forecast(request: ForecastRequest = None):
    """
    Generate PV and load forecasts.

    Uses XGBoost models to forecast solar generation and load demand.
    """
    try:
        return controller.run_forecast()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimize", response_model=OptimizeResponse)
async def run_optimization(request: OptimizeRequest = None):
    """
    Run MPC optimization for battery scheduling.

    Optimizes battery charge/discharge to minimize cost and carbon emissions.
    """
    try:
        return controller.run_optimization()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simulate", response_model=SimulateResponse)
async def run_simulation(request: SimulateRequest = None):
    """
    Run power flow simulation.

    Simulates grid import/export, costs, and carbon emissions based on
    optimization results.
    """
    try:
        return controller.run_simulation()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run", response_model=RunPipelineResponse)
async def run_pipeline(request: RunPipelineRequest = None):
    """
    Run the complete pipeline.

    Executes: data processing -> forecasting -> optimization -> simulation
    """
    try:
        if request is None:
            request = RunPipelineRequest()

        return controller.run_full_pipeline(
            process_data=request.process_data,
            run_forecast=request.run_forecast,
            run_optimization=request.run_optimization,
            run_simulation=request.run_simulation
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
