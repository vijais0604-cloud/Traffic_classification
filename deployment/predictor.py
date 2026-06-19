# predictor.py

import requests

def predict():

    with open("flows.csv", "rb") as f:

        response = requests.post(
            "http://localhost:8000/predict",
            files={"file": f}
        )

    return response.json()