"""Generate predictions using the V3 model and local scalers."""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

def main():
    root = Path(__file__).resolve().parents[1]
    test_path = root / 'data' / 'test_dataset_longdai_2024_2025_with_streamflow.csv'
    model_path = root / 'model' / 'best_flood_model_v3.h5'
    scaler_X_path = root / 'model' / 'scaler_X_v3.pkl'
    scaler_y_path = root / 'model' / 'scaler_y_v3.pkl'
    out_path = root / 'frontend' / 'data' / 'predicted_streamflow_long_dai.csv'

    print(f"Loading test data: {test_path}")
    df_test = pd.read_csv(test_path)
    if 'date' not in df_test.columns:
        if {'year', 'month', 'day'}.issubset(df_test.columns):
            df_test['date'] = pd.to_datetime(df_test[['year', 'month', 'day']])
    if 'gauge_id' not in df_test.columns:
        df_test['gauge_id'] = 'longdai'
        
    from pipeline import engineer_features, get_feature_columns, create_sequences
    df_test = engineer_features(df_test)
    
    feature_cols = get_feature_columns(df_test)
    target_col = 'streamflow'
    n_features = len(feature_cols)
    
    X_test, _, dates, _ = create_sequences(df_test, feature_cols, target_col)
    
    if len(X_test) == 0:
        print("No sequences created from test data.")
        return
        
    print(f"Test data shape: {X_test.shape}")

    print("Loading scalers...")
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)
    
    X_test_flat = X_test.reshape(-1, n_features)
    X_test_scaled = scaler_X.transform(X_test_flat).reshape(X_test.shape)

    print(f"Loading model weights: {model_path}")
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from train_longdai_v3 import build_model
    model = build_model(7, n_features)
    model.load_weights(str(model_path))

    print("Generating predictions...")
    preds_scaled = model.predict(X_test_scaled, verbose=0)
    preds = scaler_y.inverse_transform(preds_scaled)
    
    # Clamp negative flows to 0
    preds = np.maximum(0, preds)

    out_df = pd.DataFrame({
        'date': pd.to_datetime(dates),
        'predicted_streamflow_cms': preds.flatten()
    })
    out_df = out_df.sort_values('date')
    out_df.to_csv(out_path, index=False)
    print(f"Wrote predictions to: {out_path} (rows={len(out_df)})")

if __name__ == '__main__':
    main()



