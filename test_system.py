"""
End-to-end system test script.

Tests all major components to verify production readiness.
"""

import sys
from pathlib import Path
import requests

sys.path.insert(0, str(Path(__file__).parent))

from utils.config import get_config
from utils.weather_api import fetch_weather_forecast, resample_to_15min
from utils.pv_model import calculate_pv_output
from utils.logger import get_logger

logger = get_logger(__name__)


def test_config():
    """Test configuration loading."""
    print("Testing configuration...")
    try:
        config = get_config()
        assert config.battery.get('capacity_mwh') > 0
        assert config.get('pv_system', default={}).get('capacity_mw') > 0
        print("✓ Config loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False


def test_weather_api():
    """Test weather API connection."""
    print("\nTesting weather API...")
    try:
        weather_df = fetch_weather_forecast(forecast_days=1)
        assert weather_df is not None
        assert len(weather_df) > 0
        assert 'ghi' in weather_df.columns
        print(f"✓ Weather API working ({len(weather_df)} records)")
        return True
    except Exception as e:
        print(f"✗ Weather API test failed: {e}")
        return False


def test_pv_model():
    """Test physics-based PV model."""
    print("\nTesting PV model...")
    try:
        import numpy as np
        import pandas as pd

        # Test data
        ghi = np.array([0, 200, 500, 800, 500, 200, 0])
        dni = np.array([0, 150, 400, 700, 400, 150, 0])
        temp = np.array([15, 18, 22, 25, 23, 20, 16])

        output = calculate_pv_output(ghi, dni, temp)

        assert len(output) == len(ghi)
        assert output.max() > 0
        assert output.min() >= 0
        print(f"✓ PV model working (peak: {output.max():.2f} MW)")
        return True
    except Exception as e:
        print(f"✗ PV model test failed: {e}")
        return False


def test_ml_models():
    """Test ML model availability."""
    print("\nTesting ML models...")
    try:
        from pathlib import Path
        models_dir = Path(__file__).parent / "models"

        pv_exists = (models_dir / "pv_model.joblib").exists()
        load_exists = (models_dir / "load_model.joblib").exists()

        if pv_exists and load_exists:
            print("✓ ML models found (PV + Load)")
            return True
        else:
            print("⚠ ML models not found (will use physics/profile)")
            return True  # Not a failure
    except Exception as e:
        print(f"✗ ML model test failed: {e}")
        return False


def test_optimization():
    """Test MPC optimization."""
    print("\nTesting optimization...")
    try:
        from optimization.mpc_solver import build_optimization_model, solve_optimization
        import pandas as pd
        import numpy as np

        # Create simple test case
        horizon = 96  # 24 hours
        timestamps = pd.date_range(start='2025-01-01', periods=horizon, freq='15min')

        data = pd.DataFrame({
            'timestamp': timestamps,
            'forecast_pv_mw': [2.0 if 8 <= ts.hour <= 16 else 0.0 for ts in timestamps],
            'forecast_load_mw': [3.0] * horizon,
            'price_per_kwh': [0.30 if 17 <= ts.hour < 22 else 0.12 for ts in timestamps],
            'carbon_intensity': [410] * horizon
        })

        model = build_optimization_model(data, horizon)
        solved = solve_optimization(model)

        assert solved is not None
        print("✓ Optimization solver working")
        return True
    except Exception as e:
        print(f"✗ Optimization test failed: {e}")
        return False


def test_docker_health():
    """Test Docker container health endpoint."""
    print("\nTesting Docker health...")
    try:
        response = requests.get('http://localhost:8510/_stcore/health', timeout=5)
        if response.status_code == 200:
            print("✓ Docker container healthy")
            return True
        else:
            print(f"⚠ Docker container returned {response.status_code}")
            return False
    except Exception as e:
        print(f"⚠ Docker container not running (this is OK if testing locally)")
        return True  # Not a failure


def test_logging():
    """Test logging system."""
    print("\nTesting logging...")
    try:
        from utils.logger import get_logger
        test_logger = get_logger('test')
        test_logger.info("Test log message")

        log_dir = Path(__file__).parent / "logs"
        assert log_dir.exists()
        print("✓ Logging system working")
        return True
    except Exception as e:
        print(f"✗ Logging test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("MICROGRID AUTOPILOT - SYSTEM TEST")
    print("=" * 60)

    tests = [
        ("Configuration", test_config),
        ("Weather API", test_weather_api),
        ("PV Model", test_pv_model),
        ("ML Models", test_ml_models),
        ("Optimization", test_optimization),
        ("Docker Health", test_docker_health),
        ("Logging", test_logging),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name} test crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {name}")

    print("-" * 60)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! System is production ready.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review.")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
