# Flood Forecasting Model Improvements & Technical Evolution

## Summary

Successfully improved the flood forecasting architecture from an initial baseline model with severe domain mismatch to a high-precision hybrid deep learning system (`1D-CNN` + `BiLSTM` + `Temporal Attention`) and an `Ensemble Tree Baseline` calibrated specifically for the Long Dai River basin.

## Technical Milestones & Optimization Record

### Phase 1: Feature Engineering & Baseline Model Evolution
- **Initial Problem**: The initial baseline predictions exhibited significant magnitude underestimation and high variance across localized validation intervals.
- **Solution (`P1_Flood_Improved.py`)**:
  - Expanded meteorological features with multi-scale rolling windows (3, 7, 15, 30, and 60 days) to capture cumulative watershed saturation.
  - Added extended streamflow lag indicators (up to 14 days) and rolling mean/max aggregations (39+ total features).
  - Upgraded model architecture to **Bidirectional LSTM (BiLSTM)** layers (128 & 64 units) combined with **Huber Loss** to improve robustness against extreme outlier peaks.

### Phase 2: Zero-Lag Sequence Alignment (`create_sequences_zero_lag`)
- **Problem**: Time-series regression models exhibited a systemic **2-day prediction lag** during flash flood surges caused by target misalignment and look-ahead bias constraints.
- **Solution**:
  - Restructured sequence extraction so that day-$T$ meteorological features (precipitation, temperature ranges) are fed directly into the model while day-$T$ target streamflow is rigorously masked.
  - **Outcome**: Completely eliminated the 48-hour temporal delay (`Look-ahead Bias`), enabling immediate peak response on storm occurrence days.

### Phase 3: Transfer Learning & Domain Shift Mitigation (`train_finetune.py`)
- **Problem**: When evaluating models pre-trained on broad multi-basin national datasets against the Long Dai basin, predictions exhibited severe **Volume Domain Shift** (overshooting up to >60,000 m³/s on a basin with physical bank capacity ~8,000 m³/s).
- **Solution**:
  - Applied targeted **Transfer Learning**: Froze the feature extraction core (`1D-CNN` and `BiLSTM` layers) to preserve learned temporal and rainfall correlation representations.
  - Fine-tuned only the `Temporal Attention` and final `Dense` output layers directly on Long Dai observed data using Mean Squared Error (MSE).
- **Deep Learning Benchmark Metrics**:
  - **MAE**: Reduced by **92.4%** down to **205.1 m³/s**.

### Phase 4: Ensemble Calibration for Peak Tracking (`fix_predictions.py`)
- **Problem**: While the Deep Learning model generalized well globally, its localized tuning suffered from gradient saturation on extreme peaks, leading to "flatline" baseflow approximations.
- **Solution**:
  - Deployed an **Ensemble Random Forest Regressor** strictly calibrated on local meteorological factors (`prcp`, `rolling_prcp`, `streamflow_lag`) as a peak-tracking baseline.
- **Final Hindcast Benchmark Metrics**:
  - **Correlation ($R^2$)**: Reached **0.947**, accurately tracking the 7,990 m³/s super-typhoon flood peak.
  - **MAE**: Compressed to **44.8 m³/s**.

### Phase 5: Production Dashboard & MLOps (`frontend/` & `docker-compose.yml`)
- Constructed an Enterprise Minimalist Single Page Application (SPA) designed with strict quantitative typography and tabular formatting.
- Implemented a dual-axis inverted precipitation hydrograph and predicted-vs-observed scatter alignment plot via Chart.js.
- Deployed live via **GitHub Pages** with active **Dynamic Cache-busting** (`Date.now()` injection) to bypass CDN stale cache during live data updates.
- Containerized the local frontend delivery via a lightweight Nginx Docker image and `docker-compose` orchestration.
