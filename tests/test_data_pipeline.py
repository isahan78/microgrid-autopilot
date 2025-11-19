"""
Tests for the data processing pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_prep.process_data import (
    process_pv_data, process_load_data, process_tariff_data,
    process_carbon_data, process_weather_data
)


PROCESSED_DIR = Path(__file__).parent.parent / "data_processed"


class TestDataPipeline:
    """Tests for data processing functions."""

    def test_pv_data_processing(self):
        """Test PV data processing."""
        # Run processing
        df = process_pv_data()

        # Check output
        assert df is not None
        assert len(df) > 0
        assert 'timestamp' in df.columns
        assert 'pv_power_mw' in df.columns

        # Check data quality
        assert df['pv_power_mw'].min() >= 0, "PV power should be non-negative"
        assert not df['timestamp'].isnull().any(), "No null timestamps"

        # Check 15-minute intervals
        if len(df) > 1:
            time_diff = (df['timestamp'].iloc[1] - df['timestamp'].iloc[0]).total_seconds()
            assert time_diff == 900, "Should be 15-minute intervals"

    def test_load_data_processing(self):
        """Test load data processing."""
        # Need PV timestamps first
        pv_df = process_pv_data()
        pv_timestamps = pv_df['timestamp']

        # Run processing
        df = process_load_data(pv_timestamps)

        # Check output
        assert df is not None
        assert len(df) > 0
        assert 'timestamp' in df.columns
        assert 'load_mw' in df.columns

        # Check data quality
        assert df['load_mw'].min() >= 0, "Load should be non-negative"

    def test_tariff_data_processing(self):
        """Test tariff data processing."""
        # Need PV timestamps first
        pv_df = process_pv_data()
        pv_timestamps = pv_df['timestamp']

        # Run processing
        df = process_tariff_data(pv_timestamps)

        # Check output
        assert df is not None
        assert len(df) == len(pv_timestamps)
        assert 'timestamp' in df.columns
        assert 'price_per_kwh' in df.columns

        # Check TOU pricing
        prices = df['price_per_kwh'].unique()
        assert 0.12 in prices, "Should have off-peak price"
        assert 0.30 in prices, "Should have peak price"

    def test_carbon_data_processing(self):
        """Test carbon data processing."""
        # Need PV timestamps first
        pv_df = process_pv_data()
        pv_timestamps = pv_df['timestamp']

        # Run processing
        df = process_carbon_data(pv_timestamps)

        # Check output
        assert df is not None
        assert len(df) == len(pv_timestamps)
        assert 'timestamp' in df.columns
        assert 'carbon_intensity' in df.columns

        # Check values are reasonable
        assert df['carbon_intensity'].min() > 0, "Carbon intensity should be positive"
        assert df['carbon_intensity'].max() < 1000, "Carbon intensity should be reasonable"

    def test_weather_data_processing(self):
        """Test weather data processing."""
        # Need PV timestamps first
        pv_df = process_pv_data()
        pv_timestamps = pv_df['timestamp']

        # Run processing
        df = process_weather_data(pv_timestamps)

        # Check output
        assert df is not None
        assert len(df) > 0
        assert 'timestamp' in df.columns
        assert 'ghi' in df.columns
        assert 'dni' in df.columns
        assert 'temperature' in df.columns

        # Check data quality
        assert df['ghi'].min() >= 0, "GHI should be non-negative"
        assert df['dni'].min() >= 0, "DNI should be non-negative"

    def test_data_alignment(self):
        """Test that all processed data is properly aligned."""
        # Process all data
        pv_df = process_pv_data()
        pv_timestamps = pv_df['timestamp']

        load_df = process_load_data(pv_timestamps)
        tariff_df = process_tariff_data(pv_timestamps)
        carbon_df = process_carbon_data(pv_timestamps)
        weather_df = process_weather_data(pv_timestamps)

        # Check alignment
        n = len(pv_df)
        assert len(load_df) == n, "Load should align with PV"
        assert len(tariff_df) == n, "Tariff should align with PV"
        assert len(carbon_df) == n, "Carbon should align with PV"
        assert len(weather_df) == n, "Weather should align with PV"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
