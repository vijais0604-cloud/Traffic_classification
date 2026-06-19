import os
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import joblib
import numpy as np

app = FastAPI()

# Resolve artifact directory from environment or default relative path
MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).resolve().parent.parent / "models"))

# Load artifacts
model = joblib.load(MODEL_DIR / "xgb_model_f15.pkl")
feature_columns = joblib.load(MODEL_DIR / "features_f15.pkl")
label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")

@app.get("/")
def main():
    
    return {
        "Features required" : [
        "init_win_bytes_backward",
        "init_win_bytes_forward",
        "min_seg_size_forward",
        "flow_iat_min",
        "fwd_iat_min",
        "bwd_packet_length_mean",
        "bwd_packet_length_max",
        "flow_iat_max",
        "fwd_packet_length_max",
        "fwd_iat_mean",
        "flow_duration",
        "fwd_iat_std",
        "flow_iat_mean",
        "flow_packets_per_s",
        "flow_bytes_per_s"
        ]}
@app.post("/predict")
def predict(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    df = df.rename(columns={
    "init_fwd_win_byts": "init_win_bytes_forward",
    "init_bwd_win_byts": "init_win_bytes_backward",
    "fwd_seg_size_min": "min_seg_size_forward",
    "bwd_pkt_len_mean": "bwd_packet_length_mean",
    "bwd_pkt_len_max": "bwd_packet_length_max",
    "fwd_pkt_len_max": "fwd_packet_length_max",
    "flow_pkts_s": "flow_packets_per_s",
    "flow_byts_s": "flow_bytes_per_s"})
    
    metadata = df[
    [
        "src_ip",
        "dst_ip"
    ]
     ].copy() 
    
    df = df.drop(columns=["src_ip", "dst_ip"])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    # Validate features
    missing = [col for col in feature_columns if col not in df.columns]
    
    if missing:
        raise HTTPException(

        status_code=400,

        detail=f"Missing features: {missing}"

    )

    ordered_features = df[feature_columns]

   
    prediction = model.predict(ordered_features)

    attack_label = label_encoder.inverse_transform(prediction)

    metadata["prediction"] = attack_label

    attacks = metadata[metadata["prediction"] != "BENIGN"]

    return {
    "attacks": attacks.to_dict(
        orient="records"
    )}