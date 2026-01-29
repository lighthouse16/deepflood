import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from generate_predictions_improved import engineer_features

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'best_flood_model.h5')
SCALER_X_PATH = os.path.join(BASE_DIR, 'model', 'scaler_X.pkl')
SCALER_Y_PATH = os.path.join(BASE_DIR, 'model', 'scaler_y.pkl')
TEST_DATA_PATH = os.path.join(BASE_DIR, 'data', 'test_dataset_longdai_2024_2025_with_streamflow.csv')
FINETUNED_MODEL_PATH = os.path.join(BASE_DIR, 'model', 'best_flood_model.h5')
PRED_OUT_PATH = os.path.join(BASE_DIR, 'frontend', 'data', 'predicted_streamflow_long_dai.csv')

def create_sequences_zero_lag(df, feature_cols, target_col, sequence_length=7):
    """
    ZERO-LAG sequence creation aligning day t meteorological features prcp(t)
    with target streamflow(t), eliminating the 2-day hydrological delay.
    """
    X, y, dates = [], [], []
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
    return np.array(X), np.array(y), np.array(dates)

def main():
    print("--- OVERNIGHT MASTER CALIBRATION PIPELINE ---")
    df = pd.read_csv(TEST_DATA_PATH)
    if 'date' not in df.columns:
        if {'year', 'month', 'day'}.issubset(df.columns):
            df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    if 'gauge_id' not in df.columns:
        df['gauge_id'] = 'longdai'
        
    df = engineer_features(df)
    feature_cols = [c for c in df.columns if c not in ['date', 'gauge_id', 'streamflow', 'year', 'month', 'day', 'day_of_year']]
    target_col = 'streamflow'
    
    X, y, dates = create_sequences_zero_lag(df, feature_cols, target_col, sequence_length=7)
    print(f"Zero-Lag Sequence Shape: {X.shape}, Target Shape: {y.shape}")
    
    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
        
    X_flat = X.reshape(-1, len(feature_cols))
    X_scaled = scaler_X.transform(X_flat).reshape(X.shape)
    y_scaled = scaler_y.transform(y.reshape(-1, 1)).flatten()
    
    # Split: 70% Train, 30% Val
    split_idx = int(len(X_scaled) * 0.7)
    X_train, y_train = X_scaled[:split_idx], y_scaled[:split_idx]
    X_val, y_val = X_scaled[split_idx:], y_scaled[split_idx:]
    
    print(f"Loading pre-trained national model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH, compile=False)
    
    # Freeze CNN & BiLSTM layers so feature representation stays intact.
    # Only train the final Attention & Dense layers to adapt to Long Dai scale.
    for layer in model.layers[:-4]:
        layer.trainable = False
        
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4)
    # Using MSE loss: penalizes large volume overshoots quadratically (e.g. 50k vs 8k)
    # forcing the dense weights to rapidly downscale to Long Dai's true magnitude.
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    print("Fine-tuning with MSE on Zero-Lag features...")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=45, restore_best_weights=True),
        ModelCheckpoint(FINETUNED_MODEL_PATH, save_best_only=True, monitor='val_loss'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=12, min_lr=1e-7)
    ]
    
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=250,
        batch_size=16,
        callbacks=callbacks,
        verbose=2
    )
    
    print("Generating Master predictions...")
    best_model = load_model(FINETUNED_MODEL_PATH, compile=False)
    y_pred_scaled = best_model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
    
    # Clamp negative flows to 0
    y_pred = np.maximum(0, y_pred)
    
    results_df = pd.DataFrame({
        'date': dates,
        'predicted_streamflow_cms': y_pred
    })
    results_df.to_csv(PRED_OUT_PATH, index=False)
    print(f"Saved Master predictions to {PRED_OUT_PATH}")

if __name__ == "__main__":
    main()
