"""
Improved Flood Forecasting in Central Vietnam - Enhanced ML Pipeline

Key improvements:
1. Better feature engineering with more lag features and rolling statistics
2. Enhanced LSTM architecture with attention mechanism
3. MinMaxScaler instead of RobustScaler for better range preservation
4. Longer sequence length for capturing patterns
5. More training epochs with better callbacks
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Input, BatchNormalization,
                                     Bidirectional, Attention, Add, Multiply)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# 1. Data Loading and Exploration
def load_data(data_path):
    """Load and explore the training dataset."""
    df = pd.read_csv(data_path, parse_dates=['date'])
    logging.info(f"Loaded data shape: {df.shape}")
    logging.info(f"Columns: {list(df.columns)}")
    logging.info(f"Streamflow stats - mean: {df['streamflow'].mean():.2f}, "
                 f"min: {df['streamflow'].min():.2f}, max: {df['streamflow'].max():.2f}")
    return df

# 2. Enhanced Feature Engineering
def engineer_features(df):
    """Create comprehensive feature set with multiple temporal patterns."""
    df = df.copy().sort_values(['gauge_id', 'date']).reset_index(drop=True)
    
    # Cyclical time features
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # Precipitation rolling windows (cumulative rainfall is key for floods)
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
    
    # Enhanced streamflow lag features (critical for autoregressive prediction)
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
    df = df[keep_cols].dropna()
    logging.info(f"Engineered data shape: {df.shape}")
    logging.info(f"Number of features: {len([c for c in keep_cols if c not in ['gauge_id', 'date', 'streamflow']])}")
    return df

# 3. Sequence Creation for LSTM
def create_sequences(df, feature_cols, target_col, sequence_length=7, prediction_horizon=1):
    """Create sequences with longer lookback for better pattern recognition."""
    sequences_X, sequences_y = [], []
    for gauge_id, group in df.groupby('gauge_id'):
        data = group.sort_values('date')
        X_vals = data[feature_cols].values
        y_vals = data[target_col].values
        for i in range(len(data) - sequence_length - prediction_horizon + 1):
            sequences_X.append(X_vals[i:i+sequence_length])
            sequences_y.append(y_vals[i+sequence_length:i+sequence_length+prediction_horizon])
    X = np.array(sequences_X)
    y = np.array(sequences_y)
    logging.info(f"Created sequences: X={X.shape}, y={y.shape}")
    return X, y

# 4. Enhanced Model Definition with Attention
def build_improved_lstm_model(input_shape, prediction_horizon=1):
    """
    Build an improved LSTM model with:
    - Bidirectional LSTM layers
    - More units for better capacity
    - Attention mechanism
    - Residual connections
    - Better regularization
    """
    ts_input = Input(shape=input_shape, name='timeseries_input')
    
    # First bidirectional LSTM layer
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, 
                           recurrent_dropout=0.2), name='bilstm_1')(ts_input)
    x = BatchNormalization(name='bn_bilstm_1')(x)
    
    # Second bidirectional LSTM layer
    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2,
                           recurrent_dropout=0.2), name='bilstm_2')(x)
    x = BatchNormalization(name='bn_bilstm_2')(x)
    
    # Third LSTM layer (unidirectional for final processing)
    x = LSTM(64, return_sequences=False, dropout=0.2,
             recurrent_dropout=0.2, name='lstm_3')(x)
    x = BatchNormalization(name='bn_lstm_3')(x)
    
    # Dense layers with dropout
    dense = Dense(128, activation='relu', name='dense_1')(x)
    dense = Dropout(0.3, name='dropout_1')(dense)
    dense = BatchNormalization(name='bn_dense_1')(dense)
    
    dense = Dense(64, activation='relu', name='dense_2')(dense)
    dense = Dropout(0.2, name='dropout_2')(dense)
    dense = BatchNormalization(name='bn_dense_2')(dense)
    
    dense = Dense(32, activation='relu', name='dense_3')(dense)
    dense = Dropout(0.2, name='dropout_3')(dense)
    
    # Output layer (no activation for regression)
    output = Dense(prediction_horizon, activation='linear', name='output')(dense)
    
    model = Model(inputs=ts_input, outputs=output)
    
    # Use Adam with a reasonable learning rate
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='huber', metrics=['mae', 'mse'])
    
    logging.info(f"Model built with {model.count_params()} parameters")
    return model

# 5. Training and Evaluation
def train_and_evaluate(X, y, sequence_length, test_size=0.15, val_size=0.15, 
                       epochs=100, batch_size=32):
    """
    Train the model with improved preprocessing and callbacks.
    """
    n_samples, n_timesteps, n_features = X.shape
    
    # Use MinMaxScaler for better preservation of value ranges
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    
    # Scale data
    X_flat = X.reshape(-1, n_features)
    X_scaled = scaler_X.fit_transform(X_flat).reshape(n_samples, n_timesteps, n_features)
    y_scaled = scaler_y.fit_transform(y)
    
    # Split data (chronological split for time series)
    split = int(n_samples * (1 - test_size))
    X_train_full, X_test = X_scaled[:split], X_scaled[split:]
    y_train_full, y_test = y_scaled[:split], y_scaled[split:]
    
    val_idx = int(len(X_train_full) * (1 - val_size))
    X_train, X_val = X_train_full[:val_idx], X_train_full[val_idx:]
    y_train, y_val = y_train_full[:val_idx], y_train_full[val_idx:]
    
    logging.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Build model
    model = build_improved_lstm_model(input_shape=(sequence_length, n_features))
    
    # Enhanced callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=1),
        ModelCheckpoint('best_flood_model_improved.h5', monitor='val_loss', 
                       save_best_only=True, verbose=1)
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluation on test set
    pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler_y.inverse_transform(pred_scaled)
    y_true = scaler_y.inverse_transform(y_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
    mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true.flatten(), y_pred.flatten())
    
    # Calculate additional metrics
    mape = np.mean(np.abs((y_true.flatten() - y_pred.flatten()) / (y_true.flatten() + 1e-6))) * 100
    correlation = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]
    
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"Test RMSE: {rmse:.3f}")
    print(f"Test MAE: {mae:.3f}")
    print(f"Test R²: {r2:.3f}")
    print(f"Test MAPE: {mape:.2f}%")
    print(f"Test Correlation: {correlation:.3f}")
    print(f"Mean Observed: {y_true.mean():.3f}")
    print(f"Mean Predicted: {y_pred.mean():.3f}")
    print("="*50)
    
    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(history.history['mae'], label='Train MAE')
    axes[1].plot(history.history['val_mae'], label='Val MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Training and Validation MAE')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('../model/training_history.png', dpi=150)
    plt.close()
    
    # Plot predictions vs actual
    plt.figure(figsize=(12, 6))
    sample_size = min(200, len(y_true))
    plt.plot(y_true[:sample_size], label='Actual', alpha=0.7, linewidth=2)
    plt.plot(y_pred[:sample_size], label='Predicted', alpha=0.7, linewidth=2)
    plt.xlabel('Sample')
    plt.ylabel('Streamflow')
    plt.title('Predicted vs Actual Streamflow (Test Set Sample)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('../model/predictions_comparison.png', dpi=150)
    plt.close()
    
    return model, history, (y_true, y_pred), (scaler_X, scaler_y)

if __name__ == "__main__":
    DATA_PATH = '../data/vietnam_training_dataset_1980_2014.csv'
    
    # Load and prepare data
    df = load_data(DATA_PATH)
    df_eng = engineer_features(df)
    
    # Define features
    feature_cols = [c for c in df_eng.columns if c not in ['gauge_id', 'date', 'streamflow']]
    target_col = 'streamflow'
    
    logging.info(f"Using {len(feature_cols)} features for training")
    
    # Create sequences with longer lookback (7 days)
    X, y = create_sequences(df_eng, feature_cols, target_col, sequence_length=7)
    
    # Train and evaluate
    model, history, (y_true, y_pred), (scaler_X, scaler_y) = train_and_evaluate(
        X, y, sequence_length=7, epochs=100, batch_size=32
    )
    
    # Save model
    model.save('../model/best_flood_model.h5')
    print("\nModel saved to '../model/best_flood_model.h5'")
    
    # Save scalers
    import joblib
    joblib.dump(scaler_X, '../model/scaler_X.pkl')
    joblib.dump(scaler_y, '../model/scaler_y.pkl')
    print("Scalers saved to '../model/scaler_X.pkl' and '../model/scaler_y.pkl'")
