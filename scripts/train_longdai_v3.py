import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Conv1D, Dropout, Multiply
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from pipeline import engineer_features, get_feature_columns, create_sequences

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
    feature_cols = get_feature_columns(df)
    target_col = 'streamflow'
    
    print("Creating sequences...")
    X, y, dates, raw_targets = create_sequences(df, feature_cols, target_col)
    
    print(f"Data shape: {X.shape}")
    
    # 1. TRAIN/VAL TIME-BASED SPLIT (70/30) FIRST
    split_idx = int(len(X) * 0.7)
    
    X_train_raw, y_train_raw = X[:split_idx], y[:split_idx]
    X_val_raw, y_val_raw = X[split_idx:], y[split_idx:]
    
    # 2. LOCAL SCALER (Fit on Train only)
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_train_flat = X_train_raw.reshape(-1, len(feature_cols))
    scaler_X.fit(X_train_flat)
    X_train = scaler_X.transform(X_train_flat).reshape(X_train_raw.shape)
    
    X_val_flat = X_val_raw.reshape(-1, len(feature_cols))
    X_val = scaler_X.transform(X_val_flat).reshape(X_val_raw.shape)
    
    scaler_y.fit(y_train_raw.reshape(-1, 1))
    y_train = scaler_y.transform(y_train_raw.reshape(-1, 1)).flatten()
    y_val = scaler_y.transform(y_val_raw.reshape(-1, 1)).flatten()
    
    joblib.dump(scaler_X, SCALER_X_PATH)
    joblib.dump(scaler_y, SCALER_Y_PATH)
    print("Local Scalers (Fitted on Train only) saved.")
    
    # 3. SAMPLE WEIGHTS
    # Compute weight based on the proportion of the streamflow.
    # We use max of train targets to avoid leaking the val max
    max_train_flow = np.max(raw_targets[:split_idx])
    sample_weights = 1.0 + 30.0 * (raw_targets / max_train_flow)
    
    w_train = sample_weights[:split_idx]
    w_val = sample_weights[split_idx:]
    
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

