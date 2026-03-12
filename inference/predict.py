import joblib
import numpy as np


model = joblib.load("models/xgb_model_f15.pkl")
features =  joblib.load("models/features_f15.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")


def predict_attack(flow_features: list[dict]):
    res=[]
    for f in flow_features:
        ordered_features = [f[col] for col in features]
        # Convert to numpy array
        input_array = np.array(ordered_features).reshape(1, -1)

        # Predict
        prediction = model.predict(input_array)

        # Decode label
        attack_label = label_encoder.inverse_transform(prediction)
        res.append(attack_label[0])
    return res

sample=[
 {
 "init_win_bytes_forward": 512,
 "init_win_bytes_backward": 0,
 "min_seg_size_forward": 32,
 "flow_iat_min": 50,
 "fwd_iat_min": 50,
 "bwd_packet_length_mean": 0,
 "bwd_packet_length_max": 0,
 "flow_iat_max": 200,
 "fwd_packet_length_max": 60,
 "fwd_iat_mean": 100,
 "flow_duration": 300,
 "fwd_iat_std": 20,
 "flow_iat_mean": 120,
 "flow_packets_per_s": 8,
 "flow_bytes_per_s": 400
}
]
result = predict_attack(sample)

print("Predicted Attack:", result)


