# Hyper-Local Flash Flood Prediction AI (V3.1 - True Leakage-Free)

An advanced, hyper-localized Deep Learning system (BiLSTM + Temporal Attention) designed to predict extreme flash flood events. Developed for the Long Dai river basin (Vietnam), this project demonstrates how domain-specific localized architectures outperform generalized global models in disaster warning systems.

## 🎯 The Core Insight (Global vs. Local)
Generalized hydrologic models trained on national datasets often suffer from "Continental Deafness"—scaling issues cause them to treat local extreme flood events (e.g., 7,990 m³/s) as minor noise compared to global mega-floods (e.g., 74,000 m³/s). 

This project uses a **Hyper-Local Architecture**:
- **Strict Local Scaler**: `MinMaxScaler` is fitted *exclusively* on the local basin's 70% historical training data, eliminating Scaler Leakage.
- **Dynamic Sample Weighting**: A weighting algorithm that multiplies the loss penalty (up to 30x) for historical peak flood days, forcing the network to prioritize "Black Swan" events over normal dry days without distorting the MSE loss surface.
- **True Leakage-Free Pipeline**: Time-based split (70% Train / 30% Val) is enforced *before* any scaling or feature engineering that could peer into the future, ensuring 100% rigorous validation.

## 🧠 Model Architecture
- **Feature Engineering**: 39 features including meteorological data (Precipitation, Temperature) and strictly lagging autoregressive hydrologic data to prevent Feature Leakage.
- **Network**: `Conv1D` (Feature extraction) -> `BiLSTM` (Temporal dependencies) -> `Temporal Attention Mechanism` -> `Dense` layers.

## 📊 Performance Metrics (Unseen Data)
Tested on the unseen Validation Set (March - July 2025) under strict Leakage-Free conditions:
- **Peak Flood Catch**: Predicted **7,175 m³/s** for the historic 7,990 m³/s peak (**89.8% accuracy** on a massive outlier).
- **Baseline MAE**: Maintained an exceptional Mean Absolute Error of **123.28 m³/s** across the entire dataset.

*(Note: Fixing the data leakages actually improved the model's robustness by forcing it to learn true auto-regressive trends rather than memorizing current-day features).*

## 📂 Key Source Code
1. `scripts/train_longdai_v3.py`: The main training pipeline containing the BiLSTM+Attention architecture, Sample Weighting logic, and strict Time-based splitting.
2. `scripts/generate_predictions_improved.py`: Leakage-free feature engineering (cyclical time, rolling windows, lag features).
3. `scripts/generate_predictions_v3.py`: Production inference script using the trained `.h5` weights and local `.pkl` scalers.

## 🚀 Tech Stack
- **AI/ML**: TensorFlow, Keras, Scikit-Learn, Pandas, NumPy
- **Frontend (Dashboard)**: Vanilla JS, Chart.js, HTML/CSS (Modern Slate/Teal UI)
