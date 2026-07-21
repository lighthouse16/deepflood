"""Plot predicted vs observed streamflow diagnostics.

Creates two PNGs in the `model/` directory:
- `pred_vs_obs_timeseries.png` — time series of observed and predicted
- `pred_vs_obs_scatter.png` — scatter plot with 1:1 line

Run: python scripts/plot_pred_obs.py
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
pred_path = ROOT / 'frontend' / 'data' / 'predicted_streamflow_long_dai.csv'
obs_path = ROOT / 'data' / 'test_dataset_longdai_2024_2025_with_streamflow.csv'
out_dir = ROOT / 'model'
out_dir.mkdir(parents=True, exist_ok=True)


def load_data():
    pred = pd.read_csv(pred_path, parse_dates=['date'])
    if 'streamflow_predicted' in pred.columns:
        pred = pred.rename(columns={'streamflow_predicted': 'predicted_streamflow'})
    elif 'predicted_streamflow' not in pred.columns:
        raise SystemExit('No predicted column found in ' + str(pred_path))

    obs = pd.read_csv(obs_path)
    if 'date' not in obs.columns:
        if {'year','month','day'}.issubset(obs.columns):
            obs['date'] = pd.to_datetime(obs[['year','month','day']])
        else:
            raise SystemExit('Observed CSV missing date information')

    # find streamflow column
    stream_col = None
    for c in obs.columns[::-1]:
        if 'stream' in str(c).lower():
            stream_col = c
            break
    if stream_col is None:
        raise SystemExit('Observed streamflow column not found')

    obs = obs[['date', stream_col]].rename(columns={stream_col: 'observed_streamflow'})
    merged = pd.merge(pred[['date','predicted_streamflow']], obs, on='date', how='inner')
    return merged


def plot_timeseries(df):
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(df['date'], df['observed_streamflow'], label='Observed', color='tab:green')
    ax.plot(df['date'], df['predicted_streamflow'], label='Predicted', color='tab:blue')
    ax.set_ylabel('Streamflow (m³/s)')
    ax.set_title('Predicted vs Observed Streamflow — Time Series')
    ax.legend()
    fig.autofmt_xdate()
    out = out_dir / 'pred_vs_obs_timeseries.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved', out)


def plot_scatter(df):
    x = df['observed_streamflow'].values
    y = df['predicted_streamflow'].values
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(x, y, alpha=0.6, s=12)
    mx = max(np.nanmax(x), np.nanmax(y))
    ax.plot([0, mx], [0, mx], 'r--', label='1:1')
    ax.set_xlabel('Observed')
    ax.set_ylabel('Predicted')
    ax.set_title('Predicted vs Observed — Scatter')
    ax.legend()
    out = out_dir / 'pred_vs_obs_scatter.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved', out)


def main():
    df = load_data()
    if df.empty:
        print('No overlapping dates to plot')
        return
    plot_timeseries(df)
    plot_scatter(df)


if __name__ == '__main__':
    main()
