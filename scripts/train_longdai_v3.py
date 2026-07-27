import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Conv1D, Dropout, Multiply
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from generate_predictions_improved import engineer_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DATA_PATH = os.path.join(BASE_DIR, 'data', 'test_dataset_longdai_2024_2025_with_streamflow.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'best_flood_model_v3.h5')
SCALER_X_PATH = os.path.join(BASE_DIR, 'model', 'scaler_X_v3.pkl')
SCALER_Y_PATH = os.path.join(BASE_DIR, 'model', 'scaler_y_v3.pkl')

def create_sequences_zero_lag(df, feature_cols, target_col, sequence_length=7):
    X, y, dates, raw_targets = [], [], [], []
    grouped = df.groupby('gauge_id')
    for _, group in grouped:
        group = group.sort_values('date').reset_index(drop=True)
        features = group[feature_cols].values
        targets = group[target_col].values
        dts = group['date'].values
        for k in range(sequence_length - 1, len(group)):
            start_idx = k - sequence_length + 1
            end_idx = k + 1
            X.append(features[start_idx : end_idx])
            y.append(targets[k])
            dates.append(dts[k])
            raw_targets.append(targets[k]) # for sample weights later
    return np.array(X), np.array(y), np.array(dates), np.array(raw_targets)

def build_model(seq_len, n_features):
    inputs = Input(shape=(seq_len, n_features))
    
    # Feature Extractor
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    
    # Temporal Attention
    attention = Dense(1, activation='tanh')(x)
    attention = tf.keras.layers.Softmax(axis=1)(attention)
    context = Multiply()([x, attention])
    context = tf.keras.layers.Lambda(lambda tensor: tf.reduce_sum(tensor, axis=1))(context)
    
    # Regressor
    x = Dense(64, activation='relu')(context)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='linear')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse', metrics=['mae'])
    return model

def main():
    print("--- TRAINING MODEL V3: TRAIN FROM SCRATCH WITH LOCAL SCALER AND WEIGHTS ---")
    df = pd.read_csv(TRAIN_DATA_PATH)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    else:
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    
    if 'gauge_id' not in df.columns:
        df['gauge_id'] = 'longdai'
        
    df = engineer_features(df)
    feature_cols = [c for c in df.columns if c not in ['date', 'gauge_id', 'streamflow', 'year', 'month', 'day', 'day_of_year']]
    target_col = 'streamflow'
    
    print("Creating sequences...")
    X, y, dates, raw_targets = create_sequences_zero_lag(df, feature_cols, target_col, sequence_length=7)
    
    print(f"Data shape: {X.shape}")
    
    # 1. LOCAL SCALER
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_flat = X.reshape(-1, len(feature_cols))
    scaler_X.fit(X_flat)
    X_scaled = scaler_X.transform(X_flat).reshape(X.shape)
    
    scaler_y.fit(y.reshape(-1, 1))
    y_scaled = scaler_y.transform(y.reshape(-1, 1)).flatten()
    
    joblib.dump(scaler_X, SCALER_X_PATH)
    joblib.dump(scaler_y, SCALER_Y_PATH)
    print("Local Scalers saved.")
    
    # 2. SAMPLE WEIGHTS
    # Compute weight based on the proportion of the streamflow.
    # Base weight 1.0, maximum weight ~ 20.0
    median_flow = np.median(raw_targets)
    # y = mx + b. We want median -> 1.0, and peak(7990) -> 20.0
    # Actually, let's just use 1.0 + (raw_targets / median_flow)
    # e.g. 200 -> 2.0, 8000 -> 41.0
    sample_weights = 1.0 + 30.0 * (raw_targets / np.max(raw_targets))
    
    # 3. TRAIN/VAL TIME-BASED SPLIT (70/30)
    split_idx = int(len(X_scaled) * 0.7)
    
    X_train, y_train, w_train = X_scaled[:split_idx], y_scaled[:split_idx], sample_weights[:split_idx]
    X_val, y_val, w_val = X_scaled[split_idx:], y_scaled[split_idx:], sample_weights[split_idx:]
    
    model = build_model(X_train.shape[1], X_train.shape[2])
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=60, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_loss'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=1e-6)
    ]
    
    print("Starting Training V3...")
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        validation_data=(X_val, y_val, w_val),
        epochs=300,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )
    print(f"Model V3 saved to {MODEL_PATH}")

if __name__ == '__main__':
    main()
