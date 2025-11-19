"""
Tests for the forecasting modules.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from forecasting.pv_forecast import create_pv_features, train_pv_model, generate_forecast
from forecasting.load_forecast import create_load_features, train_load_model


PROCESSED_DIR = Path(__file__).parent.parent / "data_processed"


class TestPVForecasting:
    """Tests for PV forecasting."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        n = 100
        timestamps = pd.date_range('2006-07-15', periods=n, freq='15T')

        pv_df = pd.DataFrame({
            'timestamp': timestamps,
            'pv_power_mw': np.random.uniform(0, 10, n)
        })

        weather_df = pd.DataFrame({
            'timestamp': timestamps,
            'ghi': np.random.uniform(0, 1000, n),
            'dni': np.random.uniform(0, 800, n),
            'temperature': np.random.uniform(15, 35, n)
        })

        return pv_df, weather_df

    def test_create_pv_features(self, sample_data):
        """Test PV feature creation."""
        pv_df, weather_df = sample_data
        features = create_pv_features(pv_df, weather_df)

        # Check features exist
        assert 'hour' in features.columns
        assert 'hour_sin' in features.columns
        assert 'hour_cos' in features.columns
        assert 'day_of_year' in features.columns
        assert 'pv_lag_1' in features.columns
        assert 'ghi' in features.columns
        assert 'dni' in features.columns
        assert 'temperature' in features.columns

    def test_train_pv_model(self, sample_data):
        """Test PV model training."""
        pv_df, weather_df = sample_data
        features = create_pv_features(pv_df, weather_df)

        model, feature_cols = train_pv_model(features)

        # Model may be None if not enough data, that's ok
        if model is not None:
            assert hasattr(model, 'predict')
            assert len(feature_cols) > 0

    def test_generate_forecast(self, sample_data):
        """Test forecast generation."""
        pv_df, weather_df = sample_data
        features = create_pv_features(pv_df, weather_df)
        model, feature_cols = train_pv_model(features)

        result = generate_forecast(model, features, feature_cols)

        assert 'forecast_pv_mw' in result.columns
        # Forecasts should be non-negative
        assert result['forecast_pv_mw'].dropna().min() >= 0


class TestLoadForecasting:
    """Tests for load forecasting."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        n = 100
        timestamps = pd.date_range('2006-07-15', periods=n, freq='15T')

        load_df = pd.DataFrame({
            'timestamp': timestamps,
            'load_mw': np.random.uniform(2, 6, n)
        })

        weather_df = pd.DataFrame({
            'timestamp': timestamps,
            'ghi': np.random.uniform(0, 1000, n),
            'dni': np.random.uniform(0, 800, n),
            'temperature': np.random.uniform(15, 35, n)
        })

        return load_df, weather_df

    def test_create_load_features(self, sample_data):
        """Test load feature creation."""
        load_df, weather_df = sample_data
        features = create_load_features(load_df, weather_df)

        # Check features exist
        assert 'hour' in features.columns
        assert 'day_of_week' in features.columns
        assert 'is_weekend' in features.columns
        assert 'load_lag_1' in features.columns
        assert 'temperature' in features.columns

    def test_train_load_model(self, sample_data):
        """Test load model training."""
        load_df, weather_df = sample_data
        features = create_load_features(load_df, weather_df)

        model, feature_cols = train_load_model(features)

        # Model may be None if not enough data
        if model is not None:
            assert hasattr(model, 'predict')
            assert len(feature_cols) > 0


class TestForecastIntegration:
    """Integration tests for forecasting."""

    def test_end_to_end_pv_forecast(self):
        """Test complete PV forecasting pipeline."""
        # Check if processed data exists
        if not (PROCESSED_DIR / "pv.csv").exists():
            pytest.skip("Processed data not available")

        pv_df = pd.read_csv(PROCESSED_DIR / "pv.csv", parse_dates=['timestamp'])
        weather_df = pd.read_csv(PROCESSED_DIR / "weather.csv", parse_dates=['timestamp'])

        # Create features
        features = create_pv_features(pv_df, weather_df)
        assert len(features) > 0

        # Train model
        model, feature_cols = train_pv_model(features)

        # Generate forecast
        result = generate_forecast(model, features, feature_cols)
        assert 'forecast_pv_mw' in result.columns

    def test_end_to_end_load_forecast(self):
        """Test complete load forecasting pipeline."""
        # Check if processed data exists
        if not (PROCESSED_DIR / "load.csv").exists():
            pytest.skip("Processed data not available")

        load_df = pd.read_csv(PROCESSED_DIR / "load.csv", parse_dates=['timestamp'])
        weather_df = pd.read_csv(PROCESSED_DIR / "weather.csv", parse_dates=['timestamp'])

        # Create features
        features = create_load_features(load_df, weather_df)
        assert len(features) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
