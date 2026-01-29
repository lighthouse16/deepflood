"""Generate improved predictions for Long Dai test set with enhanced feature engineering.

This script uses the same improved feature engineering as the training script
to generate better predictions.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model


def engineer_features(df):
    """Enhanced feature engineering to match training."""
    df = df.copy().sort_values(['gauge_id', 'date']).reset_index(drop=True)
    
    # Cyclical time features
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # Precipitation rolling windows
    for window in [3, 7, 15, 30, 60]:
        df[f'prcp_rolling_{window}d'] = df.groupby('gauge_id')['prcp'].transform(
            lambda x: x.rolling(window, min_periods=1).sum())
    
    # Precipitation statistics
    for window in [7, 15, 30]:
        df[f'prcp_mean_{window}d'] = df.groupby('gauge_id')['prcp'].transform(
            lambda x: x.rolling(window, min_periods=1).mean())
        df[f'prcp_max_{window}d'] = df.groupby('gauge_id')['prcp'].transform(
            lambda x: x.rolling(window, min_periods=1).max())
    
    # Temperature features
    df['temp_range'] = df['tmax'] - df['tmin']
    for window in [7, 15]:
        df[f'tmax_rolling_{window}d'] = df.groupby('gauge_id')['tmax'].transform(
            lambda x: x.rolling(window, min_periods=1).mean())
    
    # Enhanced streamflow lag features
    for lag in [1, 2, 3, 5, 7, 14]:
        df[f'streamflow_lag_{lag}'] = df.groupby('gauge_id')['streamflow'].shift(lag)
    
    # Streamflow rolling statistics
    for window in [3, 7, 14]:
        df[f'streamflow_mean_{window}d'] = df.groupby('gauge_id')['streamflow'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        df[f'streamflow_std_{window}d'] = df.groupby('gauge_id')['streamflow'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).std().fillna(0))
    
    # Rate of change features
    df['streamflow_change'] = df.groupby('gauge_id')['streamflow'].diff()
    df['prcp_change'] = df.groupby('gauge_id')['prcp'].diff()
    
    # Interaction features
    df['prcp_x_streamflow_lag1'] = df['prcp'] * df['streamflow_lag_1']
    
    keep_cols = ['gauge_id', 'date', 'dayl', 'prcp', 'srad', 'tmax', 'tmin', 'vp', 'streamflow',
                 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'temp_range'] + \
                [c for c in df.columns if c.startswith(('prcp_rolling_', 'prcp_mean_', 'prcp_max_',
                                                          'tmax_rolling_', 'streamflow_lag_', 
                                                          'streamflow_mean_', 'streamflow_std_',
                                                          'streamflow_change', 'prcp_change',
                                                          'prcp_x_streamflow'))]
    df = df[keep_cols]
    df = df.dropna()
    return df


def create_sequences_with_dates(df, feature_cols, target_col, sequence_length=7, prediction_horizon=1):
    """Create sequences with the same sequence length as training."""
    X_list, y_list, dates = [], [], []
    for gauge_id, group in df.groupby('gauge_id'):
        data = group.sort_values('date').reset_index(drop=True)
        X_vals = data[feature_cols].values
        y_vals = data[target_col].values
        for i in range(len(data) - sequence_length - prediction_horizon + 1):
            X_list.append(X_vals[i:i+sequence_length])
            y_list.append(y_vals[i+sequence_length:i+sequence_length+prediction_horizon])
            dates.append(data.loc[i+sequence_length, 'date'])
    return np.array(X_list), np.array(y_list), np.array(dates)


def main():
    root = Path(__file__).resolve().parents[1]
    train_path = root / 'data' / 'vietnam_training_dataset_1980_2014.csv'
    test_path = root / 'data' / 'test_dataset_longdai_2024_2025_with_streamflow.csv'
    model_path = root / 'model' / 'best_flood_model.h5'
    out_path = root / 'frontend' / 'data' / 'predicted_streamflow_long_dai.csv'

    print(f"Loading training data: {train_path}")
    df_train = pd.read_csv(train_path, parse_dates=['date'])
    df_train = engineer_features(df_train)

    feature_cols = [c for c in df_train.columns if c not in ['gauge_id', 'date', 'streamflow']]
    target_col = 'streamflow'
    
    print(f"Number of features: {len(feature_cols)}")

    # Create sequences from training data to fit scalers
    X_train, y_train, _ = create_sequences_with_dates(df_train, feature_cols, target_col, sequence_length=7)
    n_samples, n_timesteps, n_features = X_train.shape
    
    print(f"Training data shape: {X_train.shape}")

    # Fit scalers on training data using MinMaxScaler
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    X_flat = X_train.reshape(-1, n_features)
    scaler_X.fit(X_flat)
    scaler_y.fit(y_train)
    
    print(f"Y scaler range: min={scaler_y.data_min_[0]:.2f}, max={scaler_y.data_max_[0]:.2f}")

    # Load model
    print(f"Loading model: {model_path}")
    model = load_model(str(model_path), compile=False)

    # Load test set and prepare sequences
    print(f"Loading test data: {test_path}")
    df_test = pd.read_csv(test_path)
    if 'date' not in df_test.columns:
        if {'year', 'month', 'day'}.issubset(df_test.columns):
            df_test['date'] = pd.to_datetime(df_test[['year', 'month', 'day']])
    if 'gauge_id' not in df_test.columns:
        df_test['gauge_id'] = 'longdai'
    df_test = engineer_features(df_test)
    
    X_test, y_test, dates = create_sequences_with_dates(df_test, feature_cols, target_col, sequence_length=7)
    
    if len(X_test) == 0:
        print("No sequences created from test data. Check feature availability and sequence length.")
        return
    
    print(f"Test data shape: {X_test.shape}")

    # Scale test X
    X_test_flat = X_test.reshape(-1, n_features)
    X_test_scaled = scaler_X.transform(X_test_flat).reshape(X_test.shape)

    # Predict
    print("Generating predictions...")
    preds_scaled = model.predict(X_test_scaled, verbose=0)
    preds = scaler_y.inverse_transform(preds_scaled)
    
    print(f"Predictions stats: min={preds.min():.2f}, max={preds.max():.2f}, mean={preds.mean():.2f}")

    # Prepare output DataFrame
    out_df = pd.DataFrame({
        'date': pd.to_datetime(dates),
        'predicted_streamflow_cms': preds.flatten()
    })
    out_df = out_df.sort_values('date')
    out_df.to_csv(out_path, index=False)
    print(f"Wrote predictions to: {out_path} (rows={len(out_df)})")


if __name__ == '__main__':
    main()
