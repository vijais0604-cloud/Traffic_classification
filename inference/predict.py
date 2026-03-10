import joblib
import numpy as np


model = joblib.load("models/xgb_model_tuned.pkl")
features =  joblib.load("models/features.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")


def predict_attack(flow_features: dict):
    
    # Arrange features in correct order
    ordered_features = [flow_features[col] for col in features]

    # Convert to numpy array
    input_array = np.array(ordered_features).reshape(1, -1)

    # Predict
    prediction = model.predict(input_array)

    # Decode label
    attack_label = label_encoder.inverse_transform(prediction)

    return attack_label[0]

sample_flow = {
    "flow_duration": 10000,
    "total_fwd_packets": 10,
    "total_length_fwd_packets": 5000,
    "fwd_packet_length_max": 800,
    "fwd_packet_length_min": 100,
    "fwd_packet_length_mean": 400,
    "bwd_packet_length_max": 900,
    "bwd_packet_length_min": 120,
    "bwd_packet_length_mean": 420,
    "flow_bytes_per_s": 3000,
    "flow_packets_per_s": 45,
    "flow_iat_mean": 200,
    "flow_iat_std": 50,
    "flow_iat_max": 400,
    "flow_iat_min": 10,
    "fwd_iat_total": 3000,
    "fwd_iat_mean": 200,
    "fwd_iat_std": 50,
    "fwd_iat_min": 10,
    "bwd_iat_total": 2500,
    "bwd_iat_mean": 180,
    "bwd_iat_std": 40,
    "bwd_iat_max": 350,
    "bwd_iat_min": 15,
    "min_packet_length": 60,
    "max_packet_length": 1500,
    "packet_length_mean": 600,
    "packet_length_std": 200,
    "packet_length_variance": 40000,
    "psh_flag_count": 1,
    "ack_flag_count": 3,
    "down_up_ratio": 1,
    "init_win_bytes_forward": 1024,
    "init_win_bytes_backward": 1024,
    "act_data_pkt_fwd": 8,
    "min_seg_size_forward": 32,
    "active_mean": 100,
    "active_std": 20,
    "active_max": 150,
    "active_min": 50,
    "idle_mean": 300,
    "idle_std": 100
}

result = predict_attack(sample_flow)

print("Predicted Attack:", result)


