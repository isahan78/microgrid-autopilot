"""
Health check and system status API endpoints.

Provides monitoring endpoints for production deployment.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import get_config
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

PROJECT_DIR = Path(__file__).parent.parent
MODELS_DIR = PROJECT_DIR / "models"


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    components: dict


class SystemStatus(BaseModel):
    weather_api: str
    ml_models: dict
    optimization: str
    config: dict


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Basic health check endpoint.

    Returns:
        200 OK if system is healthy
        503 Service Unavailable if critical components are down
    """
    try:
        config = get_config()

        # Check ML models
        pv_model_exists = (MODELS_DIR / "pv_model.joblib").exists()
        load_model_exists = (MODELS_DIR / "load_model.joblib").exists()

        components = {
            "config": "ok",
            "pv_model": "ok" if pv_model_exists else "missing",
            "load_model": "ok" if load_model_exists else "missing",
            "weather_api": "ok"  # Could ping API to verify
        }

        # Determine overall status
        status = "healthy" if all(v == "ok" for v in components.values()) else "degraded"

        return HealthResponse(
            status=status,
            timestamp=datetime.now().isoformat(),
            version="3.0.0",
            components=components
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/status", response_model=SystemStatus)
async def system_status():
    """
    Detailed system status.

    Returns comprehensive status including configuration and model info.
    """
    try:
        config = get_config()

        # Check models
        pv_model_path = MODELS_DIR / "pv_model.joblib"
        load_model_path = MODELS_DIR / "load_model.joblib"

        ml_models = {
            "pv": {
                "exists": pv_model_path.exists(),
                "path": str(pv_model_path),
                "size_mb": round(pv_model_path.stat().st_size / 1024 / 1024, 2) if pv_model_path.exists() else 0,
                "modified": datetime.fromtimestamp(pv_model_path.stat().st_mtime).isoformat() if pv_model_path.exists() else None
            },
            "load": {
                "exists": load_model_path.exists(),
                "path": str(load_model_path),
                "size_mb": round(load_model_path.stat().st_size / 1024 / 1024, 2) if load_model_path.exists() else 0,
                "modified": datetime.fromtimestamp(load_model_path.stat().st_mtime).isoformat() if load_model_path.exists() else None
            }
        }

        system_config = {
            "pv_capacity_mw": config.get("pv_system", default={}).get("capacity_mw", 0),
            "battery_capacity_mwh": config.battery.get("capacity_mwh", 0),
            "battery_power_mw": config.battery.get("max_power_mw", 0),
            "base_load_mw": config.get("load_profile", default={}).get("base_load_mw", 0),
            "location": {
                "latitude": config.get("weather", default={}).get("latitude"),
                "longitude": config.get("weather", default={}).get("longitude")
            }
        }

        return SystemStatus(
            weather_api="operational",
            ml_models=ml_models,
            optimization="ready",
            config=system_config
        )

    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "Microgrid Autopilot API",
        "version": "3.0.0",
        "endpoints": {
            "health": "/health",
            "status": "/status",
            "docs": "/docs"
        }
    }
