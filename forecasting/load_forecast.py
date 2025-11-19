"""
Load forecast module for Microgrid Autopilot.

Uses XGBoost to forecast load demand with model persistence.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import get_config


# Load configuration
config = get_config()

# Paths
PROJECT_DIR = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_DIR / config.paths.get('data_processed', 'data_processed')
OUTPUT_DIR = PROCESSED_DIR
MODELS_DIR = PROJECT_DIR / config.paths.get('models', 'models')

# Ensure models directory exists
MODELS_DIR.mkdir(exist_ok=True)

# Model file path
MODEL_PATH = MODELS_DIR / "load_model.joblib"
FEATURE_COLS_PATH = MODELS_DIR / "load_feature_cols.joblib"


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


def train_load_model(data, force_retrain=False):
    """Train XGBoost model for load forecasting with persistence."""
    # Define features
    feature_cols = [
        'temperature', 'temp_squared',
        'hour', 'hour_sin', 'hour_cos',
        'day_of_week', 'dow_sin', 'dow_cos', 'is_weekend',
        'load_lag_1', 'load_lag_2', 'load_lag_4', 'load_lag_8',
        'load_rolling_mean_4', 'load_rolling_std_4', 'load_rolling_mean_8'
    ]

    # Check if model already exists
    if MODEL_PATH.exists() and FEATURE_COLS_PATH.exists() and not force_retrain:
        print("  Loading existing model from disk...")
        model = joblib.load(MODEL_PATH)
        saved_feature_cols = joblib.load(FEATURE_COLS_PATH)
        print(f"  Model loaded from {MODEL_PATH}")
        return model, saved_feature_cols

    # Remove rows with NaN (due to lag features)
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

    # Save model
    joblib.dump(model, MODEL_PATH)
    joblib.dump(available_cols, FEATURE_COLS_PATH)
    print(f"  Model saved to {MODEL_PATH}")

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


def main(force_retrain=False):
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

    # Train model (or load existing)
    print("\nPreparing model...")
    model, feature_cols = train_load_model(data, force_retrain=force_retrain)

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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrain', action='store_true', help='Force model retraining')
    args = parser.parse_args()
    main(force_retrain=args.retrain)
