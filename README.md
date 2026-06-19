🚀 Network Intrusion Detection System (Edge-Deployable)

📌 Overview

This project implements a Machine Learning-based Intrusion Detection System (IDS) for identifying different types of network attacks using flow-based features drawn from the CICIDS-2017 dataset.

The system is designed to be:
	•	⚡ Lightweight (optimized for edge devices)
	•	📡 Real-time capable (API-based inference)
	•	🧠 Interpretable (SHAP-based feature selection)
	•	🛡️ Extensible to IPS (attack blocking support)

---

## Project structure

- `deployment/`
  - `app.py` - FastAPI application endpoint for uploaded flow CSV files.
  - `main.py` - end-to-end pipeline that runs network capture, converts captured traffic to flows, and calls the deployed API.
  - `flow_generator.py` - converts `traffic.pcap` into `flows.csv` using `cicflowmeter`.
  - `traffic_capture.py` - captures live traffic to `traffic.pcap` using `tcpdump`.
  - `predictor.py` - sends `flows.csv` to the FastAPI `/predict` endpoint to receive attack predictions.
  - `X_test-top15.csv` - example test flow dataset using the selected 15 features.
  - `test/` - deployment tests for application behavior.

- `training/`
  - `data_loading.py` - loads and preprocesses the dataset, handles label encoding, splits train/validation/test, and balances the training data.
  - `final.py` - trains and evaluates the final XGBoost model using the reduced top 15 features, saves model artifacts and results.
  - `comparison.py` - loads saved results and visualizes model accuracy, F1 scores, training time, and model size.
  - `lgb.py`, `logistic_regression.py`, `random_forest.py`, `xgb.py` - training scripts and experiments for individual models.

- `models/` - saved model artifacts used by deployment and inference.
- `results/` - output metrics and performance comparison CSV files.
- `inference/` - inference utilities and example data files.
- `dockerfile` - container build definition.
- `requirements.txt` - Python dependencies required to run the application.
- `requirements_train.txt` - training-specific dependencies if separate installation is desired.

---

## Clone and run the project

### 1. Clone the repository

```bash
git clone https://github.com/vijais0604-cloud/traffic_classification.git
cd traffic_classification
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> If you plan to re-run model training or use training-only dependencies, also install `requirements_train.txt`.

### 4. Start the deployment API

From the repository root:

```bash
uvicorn deployment.app:app --reload --host 0.0.0.0 --port 8000
```

The API exposes:
- `GET /` - returns required feature names
- `POST /predict` - accepts a CSV file and returns detected attacks

### 5. Run the deployment pipeline

From the repository root, with the API running in another terminal:

```bash
uv run python deployment/main.py
```

This will:
- capture live traffic to `deployment/traffic.pcap`
- convert the capture to `deployment/flows.csv`
- post the flow CSV to the FastAPI `/predict` endpoint
- print detected attacks

---

## Training workflow

### Pre-requisites

- A dataset file at `traffic_ml_filtered.parquet` in the repository root, or
- A PostgreSQL database accessible with the configured connection in `training/data_loading.py`.

### Train the final model

```bash
python training/final.py
```

This script:
- loads and preprocesses the data via `training/data_loading.py`
- performs train/validation/test splitting and balancing
- trains a tuned XGBoost model on the selected top 15 features
- saves `models/xgb_model_f15.pkl`, `models/features_f15.pkl`, and `results/xgb_result_f15.csv`
- exports `deployment/X_test-top15.csv` for API testing

### Compare models

```bash
python training/comparison.py
```

This script reads saved result CSV files and plots comparisons for accuracy, macro F1, F1 score, training time, and model size.

---

## Notes

- `deployment/app.py` expects model artifacts in `models/`.
- `deployment/main.py` uses system commands: `tcpdump` and `cicflowmeter`.
- Ensure `sudo` privileges, network interface access, and `cicflowmeter` installation before using the live capture pipeline.
- For API usage, send a CSV file with the required 15 feature columns plus `src_ip` and `dst_ip`.

---

## Dataset and labels

The dataset contains labeled network traffic flows with features such as:
  • Flow duration
  • Packet lengths
  • Inter-arrival times (IAT)
  • Packet rates
  • TCP flags

Classes include:
  • BENIGN
  • DDoS
  • DoS (Hulk, GoldenEye, Slowloris, Slowhttptest)
  • PortScan
  • Bot
  • FTP-Patator
  • SSH-Patator
  • Rare attacks (merged)


---

## Improvements and future work

- Add an IP blocking module in the deployment pipeline so detected malicious source addresses can be blocked automatically or flagged for firewall rules.
- Implement a real IPS mode that triggers network-level blocking or quarantine actions for confirmed attacks.
- Add logging and alerting for blocked IPs, attack types, and repeated offenders.
- Support batch and streaming ingestion so the API can process both captured PCAP flows and live flow feeds.
- Add user-configurable thresholds for blocking, false-positive handling, and attack severity.
