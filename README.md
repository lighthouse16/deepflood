# DeepFlood: AI-Powered Streamflow & Flood Forecasting System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg)
![Architecture](https://img.shields.io/badge/Architecture-1D--CNN%20+%20BiLSTM%20+%20Attention-green.svg)

An enterprise-grade quantitative forecasting system engineered to model high-variance streamflow and predict flash flood peaks in the Long Dai River basin (Central Vietnam). The system integrates a deep neural network architecture with robust temporal feature engineering, zero-lag sequence alignment, transfer learning calibration, and a production-ready dashboard.

## 🌟 Key Technical Highlights

### 1. Hybrid Architecture (`CNN-BiLSTM-Attention`)
- **1D-CNN Feature Extractor**: Convolves across 7-day lookback windows to capture rapid meteorological shifts and precipitation spikes.
- **Bidirectional LSTM**: Maintains bidirectional temporal memory across historical rolling windows (up to 60-day cumulative rainfall and 14-day streamflow lags) to model watershed soil saturation.
- **Temporal Attention Layer**: Dynamically assigns importance weights across sequence timesteps, allowing the model to focus sharply on peak storm events rather than baseline dry periods.

### 2. Zero-Lag Sequence Slicing & Target Leakage Prevention
Traditional time-series models often suffer from a **2-day prediction lag** during sudden flood surges due to look-ahead bias or improper target alignment. 
- DeepFlood implements **Zero-Lag Sequence Slicing**, injecting day- $T$ precipitation features directly into the inference vector while rigorously masking day- $T$ streamflow.
- This guarantees **zero target leakage** (`Look-ahead Bias` elimination) and enables real-time responsiveness during flash flood alerts.

### 3. Transfer Learning for Domain Shift Mitigation
When pre-trained on broad multi-basin national datasets, models frequently exhibit severe **Volume Domain Shift** (predicting >60,000 m³/s on a river whose physical bank capacity is ~8,000 m³/s).
- We resolved this via **Transfer Learning**: freezing the feature extraction sub-network (`CNN` + `BiLSTM`) and fine-tuning the `Attention` and `Dense` layers directly on localized basin data using Mean Squared Error (MSE).
- **Benchmark Results on Long Dai Test Set**:
  - **MAE**: Reduced by **92.4%** down to **205.1 m³/s**.
  - **Linear Correlation ($R$)**: Reached **0.668**, accurately tracking the timing and magnitude of the October 2024 flood peak.

## 🏗️ System Architecture & Repository Structure

```
flood-forecasting-ml/
├── data/                               # Dataset directory (Observed streamflow & meteorological features)
├── model/                              # Model artifacts, scalers, and evaluation distributions
│   ├── best_flood_model.h5             # Base model
│   └── best_flood_model_master.h5      # Calibrated fine-tuned master model
├── scripts/
│   ├── train_finetune.py               # Transfer learning calibration & zero-lag training pipeline
│   ├── generate_predictions_improved.py# Rolling statistical feature engineering & inference engine
│   └── prepare_frontend_data.py        # Data consolidation script for dashboard rendering
├── frontend/                           # Enterprise Minimalist Single Page Application (SPA)
│   ├── index.html                      # Semantic SPA markup (Live Dashboard, Architecture & Metrics, Simulator)
│   ├── css/styles.css                  # Data-driven typographic design system (Light Theme)
│   └── js/script.js                    # Dual-axis inverted hydrograph & scatter plot controller
├── Dockerfile                          # Production Nginx container configuration
└── docker-compose.yml                  # MLOps orchestration setup
```

## 📊 Interactive Dashboard & Visualization

The frontend is built using pure, zero-dependency HTML/CSS/JS designed around strict quantitative aesthetics (tabular typography, high data-to-ink ratio):
- **Dual-Axis Hydrograph**: Plots observed vs. predicted streamflow alongside an **inverted precipitation axis** hanging from the top (standard domain representation in quantitative hydrology).
- **Error Scatter Plot**: Displays predicted vs. observed flow against an ideal $y=x$ alignment line for immediate visual verification of model reliability and variance distribution.
- **Elasticity Simulator (Scenario Analysis)**: Allows interactive what-if stress testing by simulating rainfall increases ($+mm$) to estimate potential flood surge amplification using empirical attention weighting.

## 🚀 Quickstart & Deployment (Docker)

To deploy the entire inference dashboard locally using Docker without managing local Python virtual environments:

```bash
# Clone repository
git clone <repository_url>
cd flood-forecasting-ml

# Launch containerized environment via Docker Compose
docker-compose up -d
```

Open your browser and navigate to **http://localhost:8080** to view the live dashboard and interactive simulation environment.

## 📈 Local Python Development Setup

If you wish to retrain the models or execute data evaluation scripts directly:

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt

# Run the master transfer learning pipeline
python scripts/train_finetune.py

# Prepare dashboard data
python scripts/prepare_frontend_data.py
```

## 📄 License
This project is released under the MIT License.
