import os
import pandas as pd
import numpy as np

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_PATH = os.path.join(BASE_DIR, 'frontend', 'data', 'predicted_streamflow_long_dai.csv')
OBS_PATH = os.path.join(BASE_DIR, 'data', 'test_dataset_longdai_2024_2025_with_streamflow.csv')
OUT_PATH = os.path.join(BASE_DIR, 'frontend', 'data', 'model_comparison_data.csv')

def main():
    print("--- PREPARING COMBINED FRONTEND DATA ---")
    
    # 1. Load predictions
    if not os.path.exists(PRED_PATH):
        raise FileNotFoundError(f"Missing predictions at {PRED_PATH}")
    df_pred = pd.read_csv(PRED_PATH)
    df_pred['date'] = pd.to_datetime(df_pred['date'])
    for col in ['predicted_streamflow_cms', 'streamflow_predicted', 'prediction']:
        if col in df_pred.columns:
            df_pred.rename(columns={col: 'predicted_streamflow'}, inplace=True)
    
    # 2. Load observed dataset (which also has precipitation prcp)
    if not os.path.exists(OBS_PATH):
        raise FileNotFoundError(f"Missing observed data at {OBS_PATH}")
    df_obs = pd.read_csv(OBS_PATH)
    
    if 'date' not in df_obs.columns:
        if {'year', 'month', 'day'}.issubset(df_obs.columns):
            df_obs['date'] = pd.to_datetime(df_obs[['year', 'month', 'day']])
            
    df_obs = df_obs[['date', 'streamflow', 'prcp']].rename(
        columns={'streamflow': 'observed_streamflow', 'prcp': 'prcp_mm'}
    )
    
    # 3. Merge on date
    merged = pd.merge(df_obs, df_pred, on='date', how='inner')
    merged = merged.sort_values('date').reset_index(drop=True)
    
    # 4. Clean & format
    merged['observed_streamflow'] = merged['observed_streamflow'].round(2)
    merged['predicted_streamflow'] = merged['predicted_streamflow'].round(2)
    merged['prcp_mm'] = merged['prcp_mm'].round(2)
    merged['error_abs'] = np.abs(merged['observed_streamflow'] - merged['predicted_streamflow']).round(2)
    
    # Format date as YYYY-MM-DD string cleanly
    merged['date'] = merged['date'].dt.strftime('%Y-%m-%d')
    
    # 5. Save output
    merged.to_csv(OUT_PATH, index=False)
    print(f"Successfully created {OUT_PATH} with {len(merged)} rows.")
    print("Columns:", merged.columns.tolist())
    print("Top rows preview:")
    print(merged.head(5))

if __name__ == "__main__":
    main()
