from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# Load artifacts
model = joblib.load("models/xgb_model_tuned.pkl")
feature_columns = joblib.load("models/features.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")


@app.post("/predict")
def predict(flow_features: dict):

    # Validate features
    missing = [col for col in feature_columns if col not in flow_features]

    if missing:
        return {"error": f"Missing features: {missing}"}

    ordered_features = [flow_features[col] for col in feature_columns]

    input_array = np.array(ordered_features).reshape(1, -1)

    prediction = model.predict(input_array)

    attack_label = label_encoder.inverse_transform(prediction)

    return {"prediction": attack_label[0]}