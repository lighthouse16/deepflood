# DeepFlood — Hyper-Local Flash-Flood Forecasting

> **DeepFlood is a hyper-local flash-flood forecasting system for Vietnam’s Long Dai River basin, originally developed with my SEAS Summer School team in August 2025 and later extended independently.**

I rebuilt the forecasting pipeline around a leakage-free time-series workflow, combining 1D-CNN feature extraction, BiLSTM temporal modeling, and temporal attention. The system uses basin-specific scaling, lagged hydrological features, rolling rainfall windows, and peak-aware sample weighting to address a core challenge in localized flood prediction: general-purpose models often underestimate or misrepresent extreme events.

I also developed a production-style dashboard for comparing observed and predicted streamflow, visualizing rainfall, inspecting model error, and exploring rainfall-based what-if scenarios. The current version includes Docker/Nginx delivery and reproducible frontend deployment.

---

## 👨‍💻 My Independent Continuation

After the SEAS Summer School, I independently revisited the original system and extended its modeling and deployment pipeline. My work focused on leakage-free validation, basin-specific calibration, peak-event handling, feature engineering, prediction analysis, and dashboard delivery.

**Key Contributions:**
- Reworked time-based validation (70/30 split) to completely prevent Scaler and Feature Data Leakage.
- Added lagged streamflow features and multi-scale rainfall windows.
- Developed the CNN–BiLSTM–Temporal Attention architecture from scratch.
- Added peak-aware sample weighting (up to 30x multiplier) for extreme flood events.
- Built the observed-vs-predicted monitoring interactive dashboard.
- Containerized local delivery with Nginx and Docker Compose.
- Prepared frontend data and inference outputs for public deployment.

## 🎯 The Core Insight (Global vs. Local)
Generalized hydrologic models trained on national datasets often suffer from "Continental Deafness"—scaling issues cause them to treat local extreme flood events (e.g., 7,990 m³/s) as minor noise compared to global mega-floods. 

This project uses a **Hyper-Local Architecture**:
- **Strict Local Scaler**: `MinMaxScaler` is fitted *exclusively* on the local basin's 70% historical training data.
- **Dynamic Sample Weighting**: Prioritizes "Black Swan" events over normal dry days without distorting the MSE loss surface.

## 📊 Performance Metrics (Unseen Data)
Tested on the unseen Validation Set (March - July 2025) under strict Leakage-Free conditions:
- **Peak Flood Catch**: Predicted **7,175 m³/s** for the historic 7,990 m³/s peak (**89.8% accuracy** on a massive outlier).
- **Baseline MAE**: Maintained an exceptional Mean Absolute Error of **123.28 m³/s** across the entire dataset.

*(Note: The `IMPROVEMENTS.md` file contains theoretical benchmark metrics from an ensemble tree model (MAE 44.8 m³/s), which is separate from the primary Deep Learning pipeline reported here).*

## 📂 Key Source Code
1. `scripts/train_longdai_v3.py`: The main training pipeline containing the BiLSTM+Attention architecture.
2. `scripts/generate_predictions_improved.py`: Leakage-free feature engineering.
3. `scripts/generate_predictions_v3.py`: Production inference script.

## 🚀 Tech Stack
`Python` · `TensorFlow` · `Keras` · `BiLSTM` · `Temporal Attention` · `Conv1D` · `Time-Series Forecasting` · `Pandas` · `Scikit-learn` · `Chart.js` · `Docker` · `Nginx`
