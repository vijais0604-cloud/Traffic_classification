import pytest
from fastapi.testclient import TestClient
from deployment.app import app

client = TestClient(app)

def test_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "Features required": [
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
        ]
    }

def test_predict_valid_features():
    response = client.post("/predict", json={
        "init_win_bytes_backward": 10,
        "init_win_bytes_forward": 20,
        "min_seg_size_forward": 30,
        "flow_iat_min": 40,
        "fwd_iat_min": 50,
        "bwd_packet_length_mean": 60,
        "bwd_packet_length_max": 70,
        "flow_iat_max": 80,
        "fwd_packet_length_max": 90,
        "fwd_iat_mean": 100,
        "flow_duration": 110,
        "fwd_iat_std": 120,
        "flow_packets_per_s": 130,
        "flow_bytes_per_s": 140,
        "flow_iat_mean": 45
    })
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_predict_missing_features():
    response = client.post("/predict", json={
        "init_win_bytes_backward": 10,
        "init_win_bytes_forward": 20,
        "min_seg_size_forward": 30,
        "flow_iat_min": 40
    })
    assert response.status_code == 400
    
