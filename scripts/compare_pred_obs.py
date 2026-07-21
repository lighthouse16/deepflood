import pandas as pd
import numpy as np
from pathlib import Path

pred_path = Path(__file__).resolve().parents[1] / 'frontend' / 'data' / 'predicted_streamflow_long_dai.csv'
# try multiple observed file candidates (prefer one that contains streamflow)
obs_candidates = [
    Path(__file__).resolve().parents[1] / 'data' / 'test_dataset_longdai_2024_2025_with_streamflow.csv',
    Path(__file__).resolve().parents[1] / 'data' / 'longdai_mixed_data.csv',
    Path(__file__).resolve().parents[1] / 'frontend' / 'data' / 'longdai_mixed_data.csv'
]
obs_path = None
for p in obs_candidates:
    if p.exists():
        obs_path = p
        break
if obs_path is None:
    raise SystemExit('No observed CSV found in candidates: ' + ','.join(str(p) for p in obs_candidates))
print('Using observed file:', obs_path)

pred = pd.read_csv(pred_path, parse_dates=['date'])
# ensure column exists
if 'predicted_streamflow_cms' not in pred.columns:
    print('predicted column not found. Columns:', pred.columns.tolist())

# observed file may have headers year,month,day,...,streamflow
obs = pd.read_csv(obs_path)
# build date column if not present
if 'date' not in obs.columns:
    if {'year','month','day'}.issubset(obs.columns):
        obs['date'] = pd.to_datetime(obs[['year','month','day']])
    else:
        # try first three columns
        obs['date'] = pd.to_datetime(obs.iloc[:,0:3].rename(columns={obs.columns[0]:'year', obs.columns[1]:'month', obs.columns[2]:'day'}))

# find streamflow column (may have empty header)
stream_col = None
for c in obs.columns[::-1]:
    if 'stream' in str(c).lower():
        stream_col = c
        break
if stream_col is None:
    print('Observed streamflow column not found. Columns:', obs.columns.tolist())
    raise SystemExit(1)

# sometimes there's trailing empty column, coerce numeric
obs[stream_col] = pd.to_numeric(obs[stream_col], errors='coerce')

# prepare dataframes - check which column name exists
if 'predicted_streamflow_cms' in pred.columns:
    pred2 = pred[['date','predicted_streamflow_cms']].copy()
    pred2.rename(columns={'predicted_streamflow_cms': 'predicted_streamflow'}, inplace=True)
elif 'predicted_streamflow' in pred.columns:
    pred2 = pred[['date','predicted_streamflow']].copy()
elif 'streamflow_predicted' in pred.columns:
    pred2 = pred[['date','streamflow_predicted']].copy()
    pred2.rename(columns={'streamflow_predicted': 'predicted_streamflow'}, inplace=True)
else:
    raise ValueError(f"No prediction column found in: {pred.columns.tolist()}")

obs2 = obs[['date', stream_col]].rename(columns={stream_col: 'observed_streamflow'}).copy()

# align on date
merged = pd.merge(pred2, obs2, on='date', how='inner')
if merged.empty:
    print('No overlapping dates between predicted and observed.\nPred date range:', pred2['date'].min(), 'to', pred2['date'].max())
    print('Obs date range:', obs2['date'].min(), 'to', obs2['date'].max())
    raise SystemExit(1)

# stats
pred_vals = merged['predicted_streamflow'].astype(float)
obs_vals = merged['observed_streamflow'].astype(float)

mae = np.mean(np.abs(pred_vals - obs_vals))
rmse = np.sqrt(np.mean((pred_vals - obs_vals)**2))
cc = np.corrcoef(pred_vals, obs_vals)[0,1]

# linear fit (slope) predicted -> observed
A = np.vstack([pred_vals, np.ones(len(pred_vals))]).T
slope, intercept = np.linalg.lstsq(A, obs_vals, rcond=None)[0]

print('Matched rows:', len(merged))
print('Predicted: mean {:.3f}, min {:.3f}, max {:.3f}'.format(pred_vals.mean(), pred_vals.min(), pred_vals.max()))
print('Observed:  mean {:.3f}, min {:.3f}, max {:.3f}'.format(obs_vals.mean(), obs_vals.min(), obs_vals.max()))
print('MAE: {:.3f}, RMSE: {:.3f}, Corr: {:.3f}'.format(mae, rmse, cc))
print('Linear fit: observed ~= {:.3f} * predicted + {:.3f}'.format(slope, intercept))

# show top disagreements
merged['error_abs'] = np.abs(pred_vals - obs_vals)
top = merged.sort_values('error_abs', ascending=False).head(10)
print('\nTop 10 absolute errors:')
for _, r in top.iterrows():
    print(r['date'].strftime('%Y-%m-%d'), 'pred', r['predicted_streamflow'], 'obs', r['observed_streamflow'], 'err', r['error_abs'])
