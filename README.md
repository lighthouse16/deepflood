# DeepFlood: Same Day Streamflow Nowcasting

Basin specific time series modeling with leakage aware validation and peak sensitive training.

DeepFlood was originally developed with my SEAS Summer School team in August 2025 and later extended independently. I rebuilt the modeling and evaluation pipeline around a leakage aware time series workflow, combining Conv1D feature extraction, BiLSTM temporal modeling, and Temporal Attention. The system uses basin-specific scaling, lagged hydrological features, rolling rainfall windows, and peak-aware sample weighting to address a core challenge in modeling local streamflow extremes.

I also developed a reproducible hindcast evaluation dashboard for comparing observed and predicted streamflow and inspecting model error. The dashboard visualizes precomputed hindcast evaluation outputs.

**Live Dashboard:** [deepflood.haidangtrih.me](https://deepflood.haidangtrih.me)

---

## Independent Continuation

After the SEAS Summer School, I independently revisited the original system and extended its modeling and deployment pipeline. My work focused on leakage-controlled validation, basin-specific calibration, peak-event handling, feature engineering, prediction analysis, and dashboard delivery.

**Key Contributions:**
- Reworked preprocessing and validation around a 70/30 chronological split, fitting scalers only on training data and using strictly lagged target-derived features.
- Added lagged streamflow features and multi-scale rainfall windows for temporal modeling.
- Independently implemented the current Conv1D + BiLSTM + Temporal Attention architecture.
- Added magnitude based sample weighting for high flow events.
- Built the observed-vs-predicted hindcast evaluation dashboard.
- Containerized local delivery with Nginx and Docker Compose.
- Prepared frontend data and inference outputs for public deployment.

## Design Rationale

Earlier project experiments suggested that general-purpose models underrepresented large local streamflow extremes in the Long Đại data. Separately, the original evaluation pipeline also required stricter temporal boundaries to avoid future information entering preprocessing.

The current system therefore focuses on two distinct concerns:

- **Leakage control:** `MinMaxScaler` is fitted exclusively on the training period, with the chronological split applied before preprocessing to prevent future-data leakage.
- **Extreme event sensitivity:** magnitude-based sample weighting gives high-flow observations greater influence during training, increasing the optimization emphasis placed on rare large events.

## Evaluation

The current model estimates day $t$ streamflow using meteorological observations from day $t$ and streamflow observations available through day $t-1$. The bidirectional layers operate only within the already observed input window. They do not access observations after prediction time $t$.

### Chronological Validation Period (30% — 2025-04-24 to 2025-08-08)

The validation period was used for early stopping and checkpoint selection, so these results are not an independent test benchmark.

| Metric | Model | Persistence |
|---|---:|---:|
| MAE | 130.77 m³/s | **126.93 m³/s** |
| RMSE | **148.14 m³/s** | 312.79 m³/s |
| NSE | **0.726** | -0.222 |
| Validation peak error | **+8.7%** | - |
| Peak timing error | **0 days** | - |

Across 107 chronological validation observations, the model slightly trailed persistence on MAE (130.8 vs. 126.9 m³/s; approximately 3.0% higher), but substantially improved RMSE (148.1 vs. 312.8 m³/s; approximately 52.6% lower) and NSE (0.726 vs. -0.222). It estimated the 2,576.39 m³/s validation peak on the correct day with an 8.7% magnitude error (predicting 2,800.68 m³/s). This diagnostic covers a single event and is not evidence of reliable peak performance across extreme events.

### In-Sample Fit Diagnostic

Across the full hindcast, including the training period, the model reproduced 7,175.8 m³/s of the 7,990.3 m³/s peak recorded in the project dataset. This is an in-sample fit diagnostic showing the model's capacity to represent extreme magnitude, not unseen-event performance.

### Limitations
- A persistence baseline is reported by `scripts/evaluate_model.py`; broader comparisons against simpler ML models or alternative architectures remain future work.
- The validation period is not an independent test set because it was used for model selection through early stopping.
- Same-day estimation requires weather data for the prediction day; operational multi-day lead time has not been demonstrated.
- Dataset provenance is not yet documented in this repository.

## Key Source Code
1. `scripts/train_longdai_v3.py`: Training pipeline with the Conv1D + BiLSTM + Temporal Attention architecture, chronological splitting, and train-only scaling.
2. `scripts/pipeline.py`: Shared feature and sequence contract with strictly lagged target-derived features.
3. `scripts/generate_predictions_v3.py`: Reproducible V3.1 inference loading the saved model and scalers.
4. `scripts/evaluate_model.py`: Chronological validation period evaluation and persistence baseline.

## Experiment History
See [`IMPROVEMENTS.md`](IMPROVEMENTS.md) for historical experiment logs from earlier development phases (transfer learning, ensemble Random Forest, etc.). Those metrics belong to different model versions and are not the current deployed pipeline.

## Model

`Conv1D · BiLSTM · Temporal Attention · Magnitude Based Sample Weighting`

## Stack

`Python · TensorFlow/Keras · Pandas · Scikit-learn · Chart.js · Docker · Nginx`
