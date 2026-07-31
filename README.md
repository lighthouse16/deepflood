# DeepFlood — Hyper-Local Flash-Flood Forecasting

> **A hyper-local same-day flood estimation prototype with a reproducible evaluation dashboard.**

DeepFlood was originally developed with my SEAS Summer School team in August 2025 and later extended independently. I rebuilt the forecasting pipeline around a leakage-aware time-series workflow, combining 1D-CNN feature extraction, BiLSTM temporal modeling, and temporal attention. The system uses basin-specific scaling, lagged hydrological features, rolling rainfall windows, and peak-aware sample weighting to address a core challenge in localized flood prediction: general-purpose models often underestimate extreme events.

I also developed a reproducible hindcast evaluation dashboard for comparing observed and predicted streamflow, inspecting model error, and exploring rainfall-based scenarios.

**Live Dashboard:** [deepflood.haidangtrih.me](https://deepflood.haidangtrih.me)

---

## 👨‍💻 My Independent Continuation

After the SEAS Summer School, I independently revisited the original system and extended its modeling and deployment pipeline. My work focused on leakage-free validation, basin-specific calibration, peak-event handling, feature engineering, prediction analysis, and dashboard delivery.

**Key Contributions:**
- Reworked time-based validation (70/30 chronological split) to prevent Scaler and Feature Data Leakage.
- Added lagged streamflow features and multi-scale rainfall windows — all target-derived features use strictly past observations.
- Developed the CNN–BiLSTM–Temporal Attention architecture from scratch.
- Added peak-aware sample weighting (up to 30x multiplier) for extreme flood events.
- Built the observed-vs-predicted hindcast evaluation dashboard.
- Containerized local delivery with Nginx and Docker Compose.
- Prepared frontend data and inference outputs for public deployment.

## 🎯 The Core Insight (Global vs. Local)
Generalized hydrologic models trained on national datasets often suffer from "Continental Deafness" — scaling issues cause them to treat local extreme flood events (e.g., 7,990 m³/s) as minor noise compared to global mega-floods. 

This project uses a **Hyper-Local Architecture**:
- **Strict Local Scaler**: `MinMaxScaler` is fitted *exclusively* on the 70% training split — the chronological split is applied before scaler fitting.
- **Dynamic Sample Weighting**: Prioritizes extreme flood days over normal dry days without distorting the MSE loss surface.

## 📊 Evaluation

The current model performs **same-day streamflow estimation** using current meteorological conditions and river observations available through the previous day.

### Chronological Holdout (30% — 2025-04-24 to 2025-08-08)

The holdout period was used for early stopping and checkpoint selection, so these results represent **validation performance**, not an independent test benchmark.

| Metric | Model | Persistence |
|---|---:|---:|
| MAE | 130.77 m³/s | **128.12 m³/s** |
| RMSE | **148.14 m³/s** | 314.27 m³/s |
| NSE | **0.726** | -0.223 |
| Holdout peak error | **+8.7%** | - |
| Peak timing error | **0 days** | - |

On a 107-row chronological validation holdout, the model slightly trailed persistence on MAE (130.8 vs. 128.1 m³/s), but substantially improved RMSE (148.1 vs. 314.3 m³/s) and NSE (0.726 vs. −0.223). It captured the 2,576.39 m³/s holdout peak on the correct day with an 8.7% magnitude error (predicting 2,800.68 m³/s).

### In-Sample Fit Diagnostic

Across the full hindcast, including the training period, the model reproduced 7,175.8 m³/s of the recorded 7,990.3 m³/s peak. This is an **in-sample fit diagnostic** showing the model's capacity to represent extreme magnitude, not unseen-event performance.

### Limitations
- A persistence baseline is reported by `scripts/evaluate_model.py`; broader linear/tree/ablation comparisons remain future work.
- The holdout is not a fully independent test set (used for model selection via early stopping).
- Same-day estimation requires weather data for the prediction day; operational multi-day lead time has not been demonstrated.

## 📂 Key Source Code
1. `scripts/train_longdai_v3.py`: Training pipeline — BiLSTM+Attention architecture, time-based split, train-only scaling.
2. `scripts/pipeline.py`: Shared feature and sequence contract with strictly lagged target-derived features.
3. `scripts/generate_predictions_v3.py`: Reproducible V3.1 inference loading the saved model and scalers.
4. `scripts/evaluate_model.py`: Chronological holdout evaluation and persistence baseline.

## 📜 Experiment History
See [`IMPROVEMENTS.md`](IMPROVEMENTS.md) for historical experiment logs from earlier development phases (transfer learning, ensemble Random Forest, etc.). Those metrics belong to different model versions and are **not** the current deployed pipeline.

## 🚀 Tech Stack
`Python` · `TensorFlow` · `Keras` · `BiLSTM` · `Temporal Attention` · `Conv1D` · `Time-Series Forecasting` · `Pandas` · `Scikit-learn` · `Chart.js` · `Docker` · `Nginx`

