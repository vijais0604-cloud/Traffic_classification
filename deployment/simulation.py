import requests
import pandas as pd
import time
import random

# Load validation dataset
df = pd.read_csv("deployment/X_val.csv")   # or use any CSV with the 42 features

API_URL = "http://127.0.0.1:8000/predict"

while True:

    # pick random flow
    sample = df.sample(1).iloc[0].to_dict()

    try:
        response = requests.post(API_URL, json=sample)

        result = response.json()

        print("Flow Sent → Prediction:", result)

    except Exception as e:
        print("Error:", e)

    # simulate network delay
    time.sleep(random.uniform(0.3, 1.5))


