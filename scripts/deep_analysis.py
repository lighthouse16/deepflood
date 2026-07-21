"""Deep analysis of predictions vs observed.

Generates plots and a text summary in `model/`:
- residual_hist.png
- error_vs_observed.png
- error_by_bin.png
- feature_error_correlations.png
- deep_analysis_summary.txt

Run with the project's Python (conda), for example:
    conda run -p /path/to/conda-env --no-capture-output python scripts/deep_analysis.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
pred_path = ROOT / 'frontend' / 'data' / 'predicted_streamflow_long_dai.csv'
obs_path = ROOT / 'data' / 'test_dataset_longdai_2024_2025_with_streamflow.csv'
train_script = ROOT / 'notebooks' / 'train_best_model.py'
out_dir = ROOT / 'model'
out_dir.mkdir(parents=True, exist_ok=True)


def load_pred_obs():
    pred = pd.read_csv(pred_path)
    if 'streamflow_predicted' in pred.columns:
        pred = pred.rename(columns={'streamflow_predicted': 'predicted_streamflow'})
    pred['date'] = pd.to_datetime(pred['date'])

    obs = pd.read_csv(obs_path)
    if 'date' not in obs.columns:
        if {'year','month','day'}.issubset(obs.columns):
            obs['date'] = pd.to_datetime(obs[['year','month','day']])
        else:
            raise SystemExit('Observed CSV missing date info')
    # find streamflow column
    stream_col = None
    for c in obs.columns[::-1]:
        if 'stream' in str(c).lower():
            stream_col = c
            break
    obs = obs[['date', stream_col]].rename(columns={stream_col: 'observed_streamflow'})
    obs['date'] = pd.to_datetime(obs['date'])

    merged = pd.merge(pred[['date','predicted_streamflow']], obs, on='date', how='inner').sort_values('date')
    merged['residual'] = merged['predicted_streamflow'] - merged['observed_streamflow']
    merged['abs_residual'] = merged['residual'].abs()
    return merged


def residual_plots(df):
    # histogram
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(df['residual'].dropna(), bins=60, color='C1', alpha=0.9)
    ax.set_xlabel('Residual (pred - obs)')
    ax.set_ylabel('Count')
    ax.set_title('Residual Histogram')
    fig.savefig(out_dir / 'residual_hist.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # error vs observed scatter
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(df['observed_streamflow'], df['predicted_streamflow'], alpha=0.6, s=12)
    mx = max(df['observed_streamflow'].max(), df['predicted_streamflow'].max())
    ax.plot([0, mx], [0, mx], 'r--')
    ax.set_xlabel('Observed')
    ax.set_ylabel('Predicted')
    ax.set_title('Predicted vs Observed')
    fig.savefig(out_dir / 'error_vs_observed.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # error by observed bin
    bins = [0,50,100,200,500,1000,2000,5000,10000]
    df['obs_bin'] = pd.cut(df['observed_streamflow'], bins=bins)
    stats = df.groupby('obs_bin')['abs_residual'].agg(['mean','median','count']).reset_index()
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(range(len(stats)), stats['mean'], color='C2', alpha=0.9)
    ax.set_xticks(range(len(stats)))
    ax.set_xticklabels([str(x) for x in stats['obs_bin'].astype(str)], rotation=45, ha='right')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('MAE by Observed Flow Bin')
    fig.savefig(out_dir / 'error_by_bin.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def feature_error_correlation(df):
    # Extract feature sequences from test CSV using training engineer_features + create_sequences
    import ast
    src = train_script.read_text()
    mod = ast.parse(src)
    func_srcs = []
    for node in mod.body:
        if isinstance(node, ast.FunctionDef) and node.name in ('engineer_features','create_sequences'):
            func_srcs.append(ast.get_source_segment(src, node))
    ns = {'np': np, 'pd': pd, 'logger': type('L',(),{'info':lambda *a,**k:None})()}
    exec('\n\n'.join(func_srcs), ns)
    tb_engineer = ns['engineer_features']
    tb_create = ns['create_sequences']

    test_raw = pd.read_csv(obs_path)
    if 'date' not in test_raw.columns:
        test_raw['date'] = pd.to_datetime(test_raw[['year','month','day']])
    if 'gauge_id' not in test_raw.columns:
        test_raw['gauge_id'] = 'longdai'
    df_test = tb_engineer(test_raw)
    feature_cols = [c for c in df_test.columns if c not in ['gauge_id','date']]
    # tb_create returns sequences; it does not return dates, so reconstruct them
    X, y, _ = tb_create(df_test, feature_cols, 'streamflow', sequence_length=14)
    if X.size == 0:
        print('No sequences for feature correlation')
        return []
    last_feats = X[:, -1, :]
    # reconstruct sequence target dates (the date at index i+sequence_length)
    seq_dates = []
    SEQ = 14
    for gid, group in df_test.groupby('gauge_id'):
        data = group.sort_values('date').reset_index(drop=True)
        for i in range(len(data) - SEQ):
            seq_dates.append(data.loc[i+SEQ, 'date'])
    seq_dates = pd.to_datetime(seq_dates)

    # align residuals to sequences by date
    merged_seq = pd.merge(pd.DataFrame({'date': seq_dates}), df[['date','abs_residual']], on='date', how='left')
    resid = merged_seq['abs_residual'].values
    # filter nan
    valid = ~np.isnan(resid)
    resid = resid[valid]
    feats = last_feats[valid]

    corrs = []
    for i, name in enumerate(feature_cols):
        f = feats[:, i]
        if np.nanstd(f) == 0:
            corr = 0
        else:
            corr = np.corrcoef(f, resid)[0,1]
        corrs.append((name, corr))
    corrs_sorted = sorted(corrs, key=lambda x: abs(x[1]), reverse=True)[:30]

    # plot top correlations
    names = [c[0] for c in corrs_sorted]
    vals = [c[1] for c in corrs_sorted]
    fig, ax = plt.subplots(figsize=(8,6))
    ax.barh(range(len(vals)), vals, color='C3')
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names)
    ax.set_xlabel('Pearson corr with abs(error)')
    ax.set_title('Top feature correlations with absolute error')
    fig.savefig(out_dir / 'feature_error_correlations.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    return corrs_sorted


def main():
    df = load_pred_obs()
    if df.empty:
        print('No matched rows')
        return
    # save summary
    with open(out_dir / 'deep_analysis_summary.txt', 'w') as f:
        f.write(f'Matched rows: {len(df)}\n')
        f.write(f'Predicted mean {df.predicted_streamflow.mean():.3f}, obs mean {df.observed_streamflow.mean():.3f}\n')
        f.write(f'MAE {df.abs_residual.mean():.3f}, RMSE {np.sqrt((df.residual**2).mean()):.3f}\n')
    residual_plots(df)
    corrs = feature_error_correlation(df)
    if corrs:
        with open(out_dir / 'deep_analysis_summary.txt', 'a') as f:
            f.write('\nTop feature correlations with abs(error):\n')
            for name, c in corrs[:20]:
                f.write(f'{name}: {c:.3f}\n')
    print('Deep analysis complete. Outputs in', out_dir)


if __name__ == '__main__':
    main()
