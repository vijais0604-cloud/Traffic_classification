import io
import shutil
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
    # save feature list (without src_ip/dst_ip)
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


def test_get_root_and_post_predict(tmp_path, monkeypatch):
    # prepare models directory in tmp and copy into repo models/
    tmp_models = tmp_path / "models"
    _write_dummy_artifacts(tmp_models)

    # copy to repository models directory
    shutil.rmtree("models", ignore_errors=True)
    shutil.copytree(str(tmp_models), "models")

    # import app after artifacts exist
    from deployment import app as app_module

    client = TestClient(app_module.app)

    # GET /
    r = client.get("/")
    assert r.status_code == 200
    assert "Features required" in r.json()

    # Build a CSV using the feature columns and src/dst
    df = pd.DataFrame({col: [0] for col in app_module.feature_columns})
    df["src_ip"] = ["1.1.1.1"]
    df["dst_ip"] = ["2.2.2.2"]

    csv_buf = io.BytesIO()
    df.to_csv(csv_buf, index=False)
    csv_buf.seek(0)

    r2 = client.post("/predict", files={"file": ("flows.csv", csv_buf, "text/csv")})
    assert r2.status_code == 200
    data = r2.json()
    assert "attacks" in data
