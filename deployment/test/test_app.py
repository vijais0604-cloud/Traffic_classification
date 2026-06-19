import io
import shutil
import sys
import joblib
import pandas as pd
from fastapi.testclient import TestClient


class DummyLE:
    def inverse_transform(self, arr):
        return ["BENIGN" for _ in arr]


class DummyModel:
    def predict(self, X):
        import numpy as np

        return np.zeros(len(X), dtype=int)


def _write_dummy_artifacts(tmp_models_dir):
    tmp_models_dir.mkdir(parents=True, exist_ok=True)
    features = [
        "init_win_bytes_forward",
        "init_win_bytes_backward",
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
        "flow_bytes_per_s",
    ]
    joblib.dump(features, str(tmp_models_dir / "features_f15.pkl"))
    joblib.dump(DummyLE(), str(tmp_models_dir / "label_encoder.pkl"))
    joblib.dump(DummyModel(), str(tmp_models_dir / "xgb_model_f15.pkl"))


def test_main(tmp_path, monkeypatch):
    tmp_models = tmp_path / "models"
    _write_dummy_artifacts(tmp_models)
    monkeypatch.setenv("MODEL_DIR", str(tmp_models))
    sys.modules.pop("deployment.app", None)

    from deployment import app as app_module

    client = TestClient(app_module.app)
    response = client.get("/")
    assert response.status_code == 200
    assert "Features required" in response.json()

    df = pd.DataFrame({col: [0] for col in app_module.feature_columns})
    df["src_ip"] = ["1.1.1.1"]
    df["dst_ip"] = ["2.2.2.2"]
    csv_buf = io.BytesIO()
    df.to_csv(csv_buf, index=False)
    csv_buf.seek(0)

    response = client.post("/predict", files={"file": ("flows.csv", csv_buf, "text/csv")})
    assert response.status_code == 200
    assert "attacks" in response.json()


def test_predict_missing_features(tmp_path, monkeypatch):
    tmp_models = tmp_path / "models"
    _write_dummy_artifacts(tmp_models)
    monkeypatch.setenv("MODEL_DIR", str(tmp_models))
    sys.modules.pop("deployment.app", None)

    from deployment import app as app_module

    client = TestClient(app_module.app)
    response = client.post("/predict", files={"file": ("flows.csv", "src_ip,dst_ip\n1.1.1.1,2.2.2.2\n", "text/csv")})
    assert response.status_code == 400
    
