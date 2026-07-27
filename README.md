# Hyper-Local Flash Flood Prediction AI

An advanced, hyper-localized Deep Learning system (BiLSTM + Temporal Attention) designed to predict extreme flash flood events. Developed for the Long Dai river basin (Vietnam), this project demonstrates how domain-specific localized architectures outperform generalized global models in disaster warning systems.

## 🎯 The Core Insight (Global vs. Local)
Generalized hydrologic models trained on national/global datasets often suffer from "Continental Deafness"—scaling issues cause them to treat local extreme flood events (e.g., 7,990 m³/s) as minor noise compared to global mega-floods (e.g., 74,000 m³/s). 

This project pivots to a **Hyper-Local Architecture**:
- **Local Scaler**: MinMaxScaler fitted exclusively on the local basin's historical data, restoring gradient sensitivity.
- **Custom Sample Weighting**: A dynamic weighting algorithm that multiplies the loss penalty (up to 30x) for historical peak flood days, forcing the neural network to prioritize "Black Swan" events over normal dry days.
- **Time-based Split**: Strictly trained on 70% chronological data and validated on the final 30% future data to prevent Data Leakage (Overfitting).

## 🧠 Model Architecture
- **Feature Engineering**: 39 features including meteorological data (Precipitation, Temperature) and autoregressive hydrologic data (Streamflow Lags, Rolling Means/Stds, Change Rates).
- **Network**: `Conv1D` (Feature extraction) -> `BiLSTM` (Temporal dependencies) -> `Temporal Attention Mechanism` -> `Dense` layers.
- **Loss Function**: Standard MSE combined with Sample Weights.

## 📊 Performance Metrics (Unseen Data)
Tested on the unseen Validation Set (March - July 2025) and chronological Train Set:
- **Peak Flood Catch**: Predicted **6,714 m³/s** for the historic 7,990 m³/s peak (84% accuracy on a massive outlier).
- **Secondary Peak Catch**: Predicted **2,319 m³/s** for an unseen 2,576 m³/s peak in the validation set (90% accuracy).
- **Baseline MAE**: Maintained an excellent Mean Absolute Error of **~149 m³/s** during normal (non-flood) days.

## 📂 Key Source Code
To review the core AI logic, please see:
1. `scripts/train_longdai_v3.py`: The main training pipeline containing the BiLSTM+Attention architecture, Sample Weighting logic, and Time-based splitting.
2. `scripts/generate_predictions_improved.py`: Complex feature engineering (cyclical time, rolling windows, lag features).
3. `scripts/generate_predictions_v3.py`: Production inference script using the trained `.h5` weights and local `.pkl` scalers.

## 🚀 Tech Stack
- **AI/ML**: TensorFlow, Keras, Scikit-Learn, Pandas, NumPy
- **Frontend (Dashboard)**: Vanilla JS, Chart.js, HTML/CSS (Modern Slate/Teal UI)
