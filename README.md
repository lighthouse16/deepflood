# DeepFlood — Hyper-Local Flash-Flood Forecasting

> **DeepFlood is a hyper-local flash-flood forecasting system for Vietnam's Long Dai River basin, originally developed with my SEAS Summer School team in August 2025 and later extended independently.**

I rebuilt the forecasting pipeline around a leakage-free time-series workflow, combining 1D-CNN feature extraction, BiLSTM temporal modeling, and temporal attention. The system uses basin-specific scaling, lagged hydrological features, rolling rainfall windows, and peak-aware sample weighting to address a core challenge in localized flood prediction: general-purpose models often underestimate or misrepresent extreme events.

I also developed a production-style hindcast evaluation dashboard for comparing observed and predicted streamflow, visualizing rainfall, inspecting model error, and exploring rainfall-based what-if scenarios. The current version includes Docker/Nginx delivery and reproducible frontend deployment.

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

The holdout period was used for early stopping and checkpoint selection, so these results should be treated as **validation performance** rather than an independent test benchmark.

| Metric | Value |
|---|---:|
| MAE | **130.77 m³/s** |
| Observed holdout peak | 2,576.39 m³/s |
| Prediction at holdout peak | 2,800.68 m³/s |
| Peak magnitude error | +8.7% |

### Full Hindcast (Training + Validation Period)

Across the complete fitted hindcast, including the training period:

| Metric | Value |
|---|---:|
| Overall MAE | 123.28 m³/s |
| Recorded historic peak | 7,990.27 m³/s |
| Model reproduction at that peak | 7,175.78 m³/s |
| Peak magnitude ratio | 89.8% |

> ⚠️ The historic 7,990 m³/s peak falls within the training period. The 89.8% figure is a **fit diagnostic** showing the model learned to reconstruct extreme events — it is not unseen-event performance.

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

