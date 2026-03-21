🚀 Network Intrusion Detection System (Edge-Deployable)

📌 Overview

This project implements a Machine Learning-based Intrusion Detection System (IDS) for identifying different types of network attacks using flow-based features.

The system is designed to be:
	•	⚡ Lightweight (optimized for edge devices)
	•	📡 Real-time capable (API-based inference)
	•	🧠 Interpretable (SHAP-based feature selection)
	•	🛡️ Extensible to IPS (attack blocking support)

⸻

🎯 Key Features
	•	Multi-class classification of network attacks
	•	Feature engineering and data cleaning pipeline
	•	Handling of severe class imbalance
	•	Model comparison:
	•	Logistic Regression
	•	Random Forest
	•	SVM (RBF + Nystroem)
	•	XGBoost (final selected model)
	•	SHAP-based feature importance & reduction
	•	REST API deployment using FastAPI
	•	Real-time traffic simulation
	•	Edge deployment ready

⸻

🧠 Final Model
	•	Model Used: XGBoost
	•	Feature Count: Reduced from 42 → 15 (using SHAP)
	•	Performance:
	•	High Accuracy (~98–99%)
	•	Strong Macro F1 (balanced performance across classes)
	•	Model Size: ~KB range (optimized for edge deployment)

⸻

📊 Dataset

The dataset contains labeled network traffic flows with features such as:
	•	Flow duration
	•	Packet lengths
	•	Inter-arrival times (IAT)
	•	Packet rates
	•	TCP flags

Classes include:
	•	BENIGN
	•	DDoS
	•	DoS (Hulk, GoldenEye, Slowloris, Slowhttptest)
	•	PortScan
	•	Bot
	•	FTP-Patator
	•	SSH-Patator
	•	Rare attacks (merged)
