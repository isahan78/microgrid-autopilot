"""
Load forecast module for Microgrid Autopilot.

Uses XGBoost to forecast load demand.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


# Paths
PROCESSED_DIR = Path(__file__).parent.parent / "data_processed"
OUTPUT_DIR = Path(__file__).parent.parent / "data_processed"


def create_load_features(df, weather_df):
    """
    Create features for load forecasting:
    - Past load lags
    - Hour-of-day, day-of-week
    - Temperature
    """
    # Merge load with weather data
    data = df.merge(weather_df, on='timestamp', how='left')

    # Time-based features
    data['hour'] = data['timestamp'].dt.hour
    data['day_of_week'] = data['timestamp'].dt.dayofweek
    data['minute'] = data['timestamp'].dt.minute
    data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)

    # Cyclical encoding for hour
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)

    # Cyclical encoding for day of week
    data['dow_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
    data['dow_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)

    # Lag features (previous time steps)
    for lag in [1, 2, 4, 8, 96]:  # 15min, 30min, 1h, 2h, 24h lags
        data[f'load_lag_{lag}'] = data['load_mw'].shift(lag)

    # Rolling statistics
    data['load_rolling_mean_4'] = data['load_mw'].rolling(window=4).mean()
    data['load_rolling_std_4'] = data['load_mw'].rolling(window=4).std()
    data['load_rolling_mean_8'] = data['load_mw'].rolling(window=8).mean()

    # Temperature features
    data['temp_squared'] = data['temperature'] ** 2

    return data


def train_load_model(data):
    """Train XGBoost model for load forecasting."""
    # Define features
    feature_cols = [
        'temperature', 'temp_squared',
        'hour', 'hour_sin', 'hour_cos',
        'day_of_week', 'dow_sin', 'dow_cos', 'is_weekend',
        'load_lag_1', 'load_lag_2', 'load_lag_4', 'load_lag_8',
        'load_rolling_mean_4', 'load_rolling_std_4', 'load_rolling_mean_8'
    ]

    # Remove rows with NaN (due to lag features)
    # Note: load_lag_96 might have many NaNs for short datasets
    available_cols = [col for col in feature_cols if col in data.columns]
    data_clean = data.dropna(subset=available_cols)

    if len(data_clean) < 10:
        print("  Warning: Not enough data for training, using simple model")
        return None, available_cols

    X = data_clean[available_cols]
    y = data_clean['load_mw']

    # Split data (use last 20% for testing)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Train XGBoost model
    model = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        verbosity=0
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"  Model Performance:")
    print(f"    MAE:  {mae:.4f} MW")
    print(f"    RMSE: {rmse:.4f} MW")
    print(f"    R2:   {r2:.4f}")

    return model, available_cols


def generate_forecast(model, data, feature_cols):
    """Generate load forecast for the entire dataset."""
    if model is None:
        # Simple fallback: use rolling average
        data['forecast_load_mw'] = data['load_mw'].rolling(window=4, min_periods=1).mean()
    else:
        # Use XGBoost model
        data_clean = data.dropna(subset=feature_cols)
        X = data_clean[feature_cols]
        predictions = model.predict(X)

        # Ensure non-negative predictions
        predictions = np.clip(predictions, 0, None)

        # Assign predictions
        data.loc[data_clean.index, 'forecast_load_mw'] = predictions

        # Fill NaN forecasts with actual values (for lag warmup period)
        data['forecast_load_mw'] = data['forecast_load_mw'].fillna(data['load_mw'])

    return data


def main():
    """Run load forecasting pipeline."""
    print("=" * 50)
    print("MICROGRID AUTOPILOT - Load Forecasting")
    print("=" * 50)

    # Load processed data
    print("\nLoading data...")
    load_df = pd.read_csv(PROCESSED_DIR / "load.csv", parse_dates=['timestamp'])
    weather_df = pd.read_csv(PROCESSED_DIR / "weather.csv", parse_dates=['timestamp'])

    print(f"  Load records: {len(load_df)}")
    print(f"  Weather records: {len(weather_df)}")

    # Create features
    print("\nCreating features...")
    data = create_load_features(load_df, weather_df)

    # Train model
    print("\nTraining XGBoost model...")
    model, feature_cols = train_load_model(data)

    # Generate forecast
    print("\nGenerating forecast...")
    data = generate_forecast(model, data, feature_cols)

    # Save forecast
    forecast_df = data[['timestamp', 'load_mw', 'forecast_load_mw']].copy()
    forecast_df = forecast_df.rename(columns={'load_mw': 'actual_load_mw'})
    forecast_df.to_csv(OUTPUT_DIR / "forecast_load.csv", index=False)

    # Calculate overall metrics
    valid_data = forecast_df.dropna()
    mae = mean_absolute_error(valid_data['actual_load_mw'], valid_data['forecast_load_mw'])

    print(f"\nForecast saved to forecast_load.csv")
    print(f"Overall MAE: {mae:.4f} MW")
    print("=" * 50)

    return forecast_df


if __name__ == "__main__":
    main()
