"""Evaluate V3.1 on its chronological validation period and a persistence baseline."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def metrics(observed, predicted):
    observed, predicted = np.asarray(observed), np.asarray(predicted)
    errors = predicted - observed
    denominator = np.sum((observed - observed.mean()) ** 2)
    return {
        "mae_m3s": float(np.mean(np.abs(errors))),
        "rmse_m3s": float(np.sqrt(np.mean(errors ** 2))),
        "nse": float(1 - np.sum(errors ** 2) / denominator) if denominator else None,
    }


def main():
    root = Path(__file__).resolve().parents[1]
    data = pd.read_csv(root / "frontend/data/model_comparison_data.csv")
    data["date"] = pd.to_datetime(data["date"])
    data["persistence_prediction"] = data["observed_streamflow"].shift(1)
    split = int(len(data) * 0.7)
    holdout = data.iloc[split:].copy()
    comparable = holdout.copy()

    model = metrics(holdout["observed_streamflow"], holdout["predicted_streamflow"])
    persistence = metrics(comparable["observed_streamflow"], comparable["persistence_prediction"])
    peak_row = holdout.loc[holdout["observed_streamflow"].idxmax()]
    model_peak_date = holdout.loc[holdout["predicted_streamflow"].idxmax(), "date"]
    report = {
        "scope": "chronological validation period used for model selection",
        "prediction_mode": "same-day estimation using day-T weather and streamflow through T-1",
        "rows": len(holdout), "start": holdout.iloc[0]["date"].date().isoformat(),
        "end": holdout.iloc[-1]["date"].date().isoformat(), "model": model,
        "persistence_baseline": persistence,
        "observed_peak_m3s": float(peak_row["observed_streamflow"]),
        "prediction_at_observed_peak_m3s": float(peak_row["predicted_streamflow"]),
        "peak_magnitude_error_percent": float((peak_row["predicted_streamflow"] / peak_row["observed_streamflow"] - 1) * 100),
        "observed_peak_date": peak_row["date"].date().isoformat(),
        "model_peak_date": model_peak_date.date().isoformat(),
        "peak_timing_error_days": int((model_peak_date - peak_row["date"]).days),
    }
    output = root / "frontend/data/evaluation_report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    holdout.to_csv(root / "frontend/data/evaluation_predictions.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
