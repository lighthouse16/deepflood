import unittest
import numpy as np
import pandas as pd
from pipeline import engineer_features
from evaluate_model import metrics


class PipelineTests(unittest.TestCase):
    def test_streamflow_features_use_past_values(self):
        dates = pd.date_range("2025-01-01", periods=20)
        frame = pd.DataFrame({"date": dates, "gauge_id": "g", "dayl": 1, "prcp": 1.0,
            "srad": 1, "tmax": 2, "tmin": 1, "vp": 1, "streamflow": np.arange(20.0)})
        result = engineer_features(frame)
        row = result.iloc[0]
        source = frame.loc[frame["date"] == row["date"]].index[0]
        self.assertEqual(row["streamflow_lag_1"], frame.loc[source - 1, "streamflow"])
        self.assertEqual(row["streamflow_change"], frame.loc[source - 1, "streamflow"] - frame.loc[source - 2, "streamflow"])

    def test_metrics(self):
        result = metrics([1, 2, 3], [1, 3, 2])
        self.assertAlmostEqual(result["mae_m3s"], 2 / 3)
        self.assertAlmostEqual(result["rmse_m3s"], (2 / 3) ** 0.5)


if __name__ == "__main__":
    unittest.main()
